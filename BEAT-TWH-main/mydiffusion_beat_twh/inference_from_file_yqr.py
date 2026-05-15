from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
import yaml
from easydict import EasyDict

# Local imports (keep consistent with existing scripts)
[sys.path.append(i) for i in ['.', '..', '../process', '../model']]

from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip

from process.process_TWH_bvh_yqr import (
    load_bvh,
    load_audio,
    load_tsv,
    load_wordvectors,
    pose2bvh as pose2bvh_twh,
    wavlm_init,
)


_THIS_DIR = Path(__file__).resolve().parent


@dataclass
class Timings:
    init_wavlm_s: float = 0.0
    init_word2vec_s: float = 0.0
    init_model_s: float = 0.0
    process_bvh_s: float = 0.0
    process_audio_s: float = 0.0
    process_tsv_s: float = 0.0
    inference_s: float = 0.0
    export_bvh_s: float = 0.0

    @property
    def processing_s(self) -> float:
        return self.process_bvh_s + self.process_audio_s + self.process_tsv_s

    @property
    def per_file_total_s(self) -> float:
        return self.processing_s + self.inference_s + self.export_bvh_s

    @property
    def full_total_s(self) -> float:
        return self.init_wavlm_s + self.init_word2vec_s + self.init_model_s + self.per_file_total_s


def _torch_sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _timed_call(device: torch.device, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _torch_sync(device)
    return out, time.perf_counter() - t0


def _load_config(config_path: str, cli_overrides: dict) -> EasyDict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for k, v in cli_overrides.items():
        if v is not None:
            config[k] = v

    config = EasyDict(config)

    # Force TWH-only behavior (mirrors sample_yqr.py)
    config.dataset = 'TWH'

    assert config.name in ['DiffuseStyleGesture', 'DiffuseStyleGesture+', 'DiffuseStyleGesture++']
    if config.name == 'DiffuseStyleGesture+':
        config.cond_mode = 'cross_local_attention4_style1_sample'
    elif config.name == 'DiffuseStyleGesture':
        config.cond_mode = 'cross_local_attention3_style1_sample'
    elif config.name == 'DiffuseStyleGesture++':
        config.cond_mode = 'cross_local_attention5_style1_sample'

    return config


def _device_from_args(gpu: str, no_cuda: bool) -> torch.device:
    if no_cuda or (not torch.cuda.is_available()):
        return torch.device('cpu')
    return torch.device(f'cuda:{gpu}')


def _create_model_and_diffusion(args: EasyDict, device: torch.device):
    model = MDM(
        modeltype='',
        njoints=args.njoints,
        nfeats=1,
        cond_mode=args.cond_mode,
        audio_feat=args.audio_feat,
        arch='trans_enc',
        latent_dim=args.latent_dim,
        n_seed=args.n_seed,
        cond_mask_prob=args.cond_mask_prob,
        device=str(device),
        style_dim=args.style_dim,
        source_audio_dim=args.audio_feature_dim,
        audio_feat_dim_latent=args.audio_feat_dim_latent,
    )
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def _load_twh_stats() -> tuple[np.ndarray, np.ndarray]:
    data_mean_ = np.load("../process/gesture_TWH_mean_v0_yqr.npy")
    data_std_ = np.load("../process/gesture_TWH_std_v0_yqr.npy")
    return np.array(data_mean_), np.array(data_std_)


def _make_seed_features(seed_gesture: np.ndarray, data_mean: np.ndarray, data_std: np.ndarray) -> np.ndarray:
    seed_gesture = (seed_gesture - data_mean) / data_std
    seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
    seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
    seed_gesture_feat = np.concatenate((seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1)
    return seed_gesture_feat


def _infer_one(
    args: EasyDict,
    device: torch.device,
    model,
    diffusion,
    wavlm_model,
    wavlm_cfg,
    word2vector: dict,
    bvh_path: str,
    wav_path: str,
    tsv_path: str,
    save_dir: str,
    output_name: str,
    speaker_id: int,
    seed: int,
    skip_timesteps: int,
    max_len: int,
    pipeline_path: str,
    ref_bvh_path: str | None,
) -> Timings:
    timings = Timings()

    # --- preprocessing ---
    bvh, timings.process_bvh_s = _timed_call(device, load_bvh, bvh_path, False, 'rotmat')

    audio, timings.process_audio_s = _timed_call(device, load_audio, wav_path, wavlm_model, wavlm_cfg, device)
    clip_len = audio.shape[0]

    tsv, timings.process_tsv_s = _timed_call(device, load_tsv, tsv_path, word2vector, clip_len)

    textaudio = np.concatenate((audio, tsv), axis=-1)
    if textaudio.shape[-1] != args.audio_feature_dim:
        raise ValueError(
            f"audio+text feature dim mismatch: got {textaudio.shape[-1]}, expected {args.audio_feature_dim}. "
            f"(audio={audio.shape[-1]}, text={tsv.shape[-1]})"
        )

    textaudio_t = torch.from_numpy(textaudio).float().to(device)

    # --- build style ---
    if not (0 <= speaker_id < int(args.style_dim)):
        raise ValueError(f"speaker_id must be in [0, {int(args.style_dim)-1}], got {speaker_id}")
    style = np.zeros([args.style_dim], dtype=np.float32)
    style[int(speaker_id)] = 1.0

    # --- seed gesture from provided BVH ---
    if bvh.shape[0] < (args.n_seed + 2):
        raise ValueError(
            f"BVH is too short for seeding: frames={bvh.shape[0]}, need at least n_seed+2={args.n_seed+2}"
        )

    data_mean, data_std = _load_twh_stats()
    seed_gesture = bvh[: args.n_seed + 2]
    seed_gesture_feat = _make_seed_features(seed_gesture, data_mean, data_std)
    seed_gesture_feat_t = (
        torch.from_numpy(seed_gesture_feat)
        .float()
        .transpose(0, 1)
        .unsqueeze(0)
        .to(device)
    )

    # Optional extra conditioning for DiffuseStyleGesture++
    seed_last_feat_t = None
    if args.name == 'DiffuseStyleGesture++':
        if ref_bvh_path is None:
            # Default: reuse the same clip as reference
            ref_bvh_path = bvh_path
        ref_bvh = load_bvh(ref_bvh_path, False, 'rotmat')
        if ref_bvh.shape[0] < (args.n_seed + 2):
            raise ValueError(
                f"Reference BVH too short for seed_last: frames={ref_bvh.shape[0]}, need {args.n_seed+2}"
            )
        gesture_flag = ref_bvh[: args.n_seed + 2]
        gesture_flag_feat = _make_seed_features(gesture_flag, data_mean, data_std)
        seed_last_feat_t = (
            torch.from_numpy(gesture_flag_feat)
            .float()
            .transpose(0, 1)
            .unsqueeze(0)
            .to(device)
            .unsqueeze(2)
        )

    # --- inference (mirrors sample_yqr.py) ---
    torch.manual_seed(seed)

    n_frames = textaudio_t.shape[0]
    if max_len and max_len > 0:
        n_frames = min(n_frames, int(max_len))
        textaudio_t = textaudio_t[:n_frames]

    real_n_frames = copy.deepcopy(n_frames)
    stride_poses = args.n_poses - args.n_seed

    if n_frames < stride_poses:
        num_subdivision = 1
        n_frames = stride_poses
    else:
        num_subdivision = math.ceil(n_frames / stride_poses)
        n_frames = num_subdivision * stride_poses

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, args.n_poses], device=device) < 1)
    model_kwargs_['y']['style'] = torch.as_tensor([style], device=device).float()
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses, device=device).bool()

    if seed_last_feat_t is not None:
        model_kwargs_['y']['seed_last'] = seed_last_feat_t

    # Pad features to fit subdivision
    textaudio_pad = torch.zeros([n_frames - real_n_frames, args.audio_feature_dim], device=device)
    textaudio_t = torch.cat((textaudio_t, textaudio_pad), 0)
    audio_reshape = textaudio_t.reshape(num_subdivision, stride_poses, args.audio_feature_dim).transpose(0, 1)

    shape_ = (1, model.njoints, model.nfeats, args.n_poses)
    out_list = []

    t0 = time.perf_counter()

    for i in range(0, num_subdivision):
        model_kwargs_['y']['audio'] = audio_reshape[:, i : i + 1]

        if i == 0:
            if args.name == 'DiffuseStyleGesture':
                pad_zeros = torch.zeros([args.n_seed, 1, args.audio_feature_dim], device=device)
                model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0).transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture+':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture++':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][: -args.n_seed, ...].transpose(0, 1)

            model_kwargs_['y']['seed'] = seed_gesture_feat_t.unsqueeze(2)
        else:
            if args.name == 'DiffuseStyleGesture':
                pad_audio = audio_reshape[-args.n_seed :, i - 1 : i]
                model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0).transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture+':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture++':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][: -args.n_seed, ...].transpose(0, 1)

            model_kwargs_['y']['seed'] = out_list[-1][..., -args.n_seed :].to(device)

        sample = diffusion.p_sample_loop(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Smooth motion transition across overlapped seed frames
        if len(out_list) > 0 and args.n_seed != 0:
            last_poses = out_list[-1][..., -args.n_seed :]
            out_list[-1] = out_list[-1][..., : -args.n_seed]

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[..., j]
                nxt = sample[..., j]
                sample[..., j] = prev * (n - j) / (n + 1) + nxt * (j + 1) / (n + 1)

        out_list.append(sample)

    _torch_sync(device)
    timings.inference_s = time.perf_counter() - t0

    if "v0" in str(args.version):
        motion_feature_division = 3
    elif "v2" in str(args.version):
        motion_feature_division = 1
    else:
        raise ValueError("wrong version name")

    out_list = [i.detach().data.cpu().numpy()[:, : args.njoints // motion_feature_division] for i in out_list]

    if len(out_list) > 1:
        out_dir_vec_1 = np.vstack(out_list[:-1])
        sampled_seq_1 = (
            out_dir_vec_1.squeeze(2)
            .transpose(0, 2, 1)
            .reshape(1, -1, model.njoints // motion_feature_division)
        )
        out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)

    sampled_seq = sampled_seq[:, args.n_seed :]

    out_poses = np.multiply(sampled_seq[0], data_std) + data_mean
    out_poses = out_poses[:real_n_frames]

    # --- export BVH ---
    os.makedirs(save_dir, exist_ok=True)
    _, timings.export_bvh_s = _timed_call(
        device,
        pose2bvh_twh,
        out_poses,
        save_dir,
        output_name,
        pipeline_path=pipeline_path,
    )

    return timings


def _find_pairs_in_dir(input_dir: str, recursive: bool) -> list[tuple[str, str, str, str]]:
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(input_dir)

    pattern = '**/*' if recursive else '*'

    wavs = {p.stem: p for p in base.glob(f'{pattern}.wav') if p.is_file()}
    tsvs = {p.stem: p for p in base.glob(f'{pattern}.tsv') if p.is_file()}
    bvhs = {p.stem: p for p in base.glob(f'{pattern}.bvh') if p.is_file()}

    common = sorted(set(wavs.keys()) & set(tsvs.keys()) & set(bvhs.keys()))
    pairs = []
    for stem in common:
        pairs.append((stem, str(bvhs[stem]), str(wavs[stem]), str(tsvs[stem])))
    return pairs


def _print_timings(tag: str, t: Timings) -> None:
    print(f"[{tag}] preprocess(bvh/audio/tsv)={t.processing_s:.3f}s "
          f"(bvh={t.process_bvh_s:.3f}, audio={t.process_audio_s:.3f}, tsv={t.process_tsv_s:.3f}) | "
          f"inference={t.inference_s:.3f}s | export={t.export_bvh_s:.3f}s | per_file_total={t.per_file_total_s:.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Measure preprocessing + diffusion inference time from paired BVH/WAV/TSV (TWH-only).'
    )
    parser.add_argument('--mode', choices=['file', 'folder'], required=True)

    # Common model/config args
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture_yqr.yml')
    parser.add_argument('--model_path', type=str, default='./model001200000.pt')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_dir', type=str, default='sample_dir_from_file')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--speaker_id', type=int, default=0)
    parser.add_argument(
        '--wavlm_path',
        type=str,
        default=str((_THIS_DIR / '../process/WavLM/WavLM-Large.pt').resolve()),
    )
    parser.add_argument(
        '--word2vector_path',
        type=str,
        default=str((_THIS_DIR / '../process/crawl-300d-2M.vec').resolve()),
    )
    parser.add_argument('--pipeline_path', type=str, default='../process/pipeline_rotmat_64_yqr.sav')
    parser.add_argument('--ref_bvh_path', type=str, default=None, help='Optional reference BVH for DiffuseStyleGesture++')

    # File mode args
    parser.add_argument('--bvh_path', type=str, default=None)
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--tsv_path', type=str, default=None)
    parser.add_argument('--output_name', type=str, default=None)

    # Folder mode args
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--max_files', type=int, default=0, help='0 means no limit')

    args_cli = parser.parse_args()

    device = _device_from_args(args_cli.gpu, args_cli.no_cuda)
    if device.type == 'cuda':
        torch.cuda.set_device(int(args_cli.gpu))

    config = _load_config(
        args_cli.config,
        {
            'gpu': args_cli.gpu,
            'model_path': args_cli.model_path,
            'save_dir': args_cli.save_dir,
            'max_len': args_cli.max_len,
            'skip_timesteps': args_cli.skip_timesteps,
        },
    )

    # --- init heavy components once ---
    init = Timings()

    wavlm_path = Path(args_cli.wavlm_path)
    if not wavlm_path.is_file():
        raise FileNotFoundError(
            "WavLM checkpoint not found: {}\n"
            "Pass --wavlm_path /full/path/to/WavLM-Large.pt\n"
            "Repo default is: {}".format(
                wavlm_path,
                (_THIS_DIR / '../process/WavLM/WavLM-Large.pt').resolve(),
            )
        )

    (wavlm_model, wavlm_cfg), init.init_wavlm_s = _timed_call(device, wavlm_init, str(wavlm_path), device)

    word2vector_path = Path(args_cli.word2vector_path)
    if not word2vector_path.is_file():
        raise FileNotFoundError(
            "word2vector file not found: {}\n"
            "Pass --word2vector_path /full/path/to/crawl-300d-2M.vec\n"
            "Repo default is: {}".format(
                word2vector_path,
                (_THIS_DIR / '../process/crawl-300d-2M.vec').resolve(),
            )
        )

    word2vector, init.init_word2vec_s = _timed_call(device, load_wordvectors, str(word2vector_path))

    # Model + diffusion
    t0 = time.perf_counter()
    model, diffusion = _create_model_and_diffusion(config, device)
    state_dict = torch.load(config.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()
    _torch_sync(device)
    init.init_model_s = time.perf_counter() - t0

    print(f"Init times: wavlm={init.init_wavlm_s:.3f}s, word2vec={init.init_word2vec_s:.3f}s, model={init.init_model_s:.3f}s")

    if args_cli.mode == 'file':
        if not (args_cli.bvh_path and args_cli.wav_path and args_cli.tsv_path):
            raise ValueError("file mode requires --bvh_path --wav_path --tsv_path")

        output_name = args_cli.output_name
        if not output_name:
            output_name = Path(args_cli.bvh_path).stem + "_pred"

        t = _infer_one(
            config,
            device,
            model,
            diffusion,
            wavlm_model,
            wavlm_cfg,
            word2vector,
            args_cli.bvh_path,
            args_cli.wav_path,
            args_cli.tsv_path,
            args_cli.save_dir,
            output_name,
            args_cli.speaker_id,
            args_cli.seed,
            args_cli.skip_timesteps,
            args_cli.max_len,
            args_cli.pipeline_path,
            args_cli.ref_bvh_path,
        )

        _print_timings(output_name, t)
        print(
            f"Total including init: {init.init_wavlm_s + init.init_word2vec_s + init.init_model_s + t.per_file_total_s:.3f}s"
        )

    else:  # folder
        if not args_cli.input_dir:
            raise ValueError("folder mode requires --input_dir")

        pairs = _find_pairs_in_dir(args_cli.input_dir, args_cli.recursive)
        if not pairs:
            raise FileNotFoundError(
                f"No paired files found in '{args_cli.input_dir}'. Need matching stems for .bvh/.wav/.tsv"
            )

        if args_cli.max_files and args_cli.max_files > 0:
            pairs = pairs[: int(args_cli.max_files)]

        per_file_totals = []
        preprocess_totals = []
        inference_totals = []

        for idx, (stem, bvh_path, wav_path, tsv_path) in enumerate(pairs, start=1):
            out_name = stem + "_pred"
            print(f"[{idx}/{len(pairs)}] {stem}")

            t = _infer_one(
                config,
                device,
                model,
                diffusion,
                wavlm_model,
                wavlm_cfg,
                word2vector,
                bvh_path,
                wav_path,
                tsv_path,
                args_cli.save_dir,
                out_name,
                args_cli.speaker_id,
                args_cli.seed,
                args_cli.skip_timesteps,
                args_cli.max_len,
                args_cli.pipeline_path,
                args_cli.ref_bvh_path,
            )

            _print_timings(stem, t)
            per_file_totals.append(t.per_file_total_s)
            preprocess_totals.append(t.processing_s)
            inference_totals.append(t.inference_s)

        print("\nAverages (seconds):")
        print(f"  preprocess_mean={mean(preprocess_totals):.3f} (std={pstdev(preprocess_totals):.3f})")
        print(f"  inference_mean={mean(inference_totals):.3f} (std={pstdev(inference_totals):.3f})")
        print(f"  per_file_total_mean={mean(per_file_totals):.3f} (std={pstdev(per_file_totals):.3f})")
        print(
            f"  init_once_total={init.init_wavlm_s + init.init_word2vec_s + init.init_model_s:.3f} (not included above)"
        )


if __name__ == '__main__':
    main()


"""
cd /home/yqr/DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh

file mode (single file):

/home/yqr/miniconda3/envs/DiffuseStyleGesture/bin/python 
python inference_from_file_yqr.py \
  --mode file \
  --bvh_path /mnt/e/QR/DATASET/Genea2023/val/main-agent/meta_bvh/val_2023_v0_004_main-agent.bvh \
  --wav_path /mnt/e/QR/DATASET/Genea2023/val/main-agent/wav/val_2023_v0_004_main-agent.wav \
  --tsv_path /mnt/e/QR/DATASET/Genea2023/val/main-agent/tsv/val_2023_v0_004_main-agent.tsv \
  --model_path /home/yqr/DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/TWH_mymodel4_512_v0_yqr_windows_150_seed_30/model000400000.pt \
  --save_dir /home/yqr/DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/TWH_mymodel4_512_v0_yqr_windows_150_seed_30/sample_dir_from_file


folder mode (process multiple files in a directory, requires matching stems for .bvh/.wav/.tsv):

/home/yqr/miniconda3/envs/DiffuseStyleGesture/bin/python inference_from_file_yqr.py \
  --mode folder \
  --input_dir /path/to/folder \
  --save_dir sample_dir_timing \
  --max_files 10
"""
