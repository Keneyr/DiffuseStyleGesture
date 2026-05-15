import sys
import os
import copy
import math
import argparse
from datetime import datetime

import numpy as np
import yaml
import torch
from easydict import EasyDict

# Local imports (keep consistent with the original script)
[sys.path.append(i) for i in ['.', '..', '../process', '../model']]

from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip

from process.process_TWH_bvh_yqr import (
    pose2bvh as pose2bvh_twh,
    wavlm_init,
    load_metadata,
    load_wordvectors,
    load_audio,
    load_tsv,
)


def create_model_and_diffusion(args):
    model = MDM(
        modeltype='',
        njoints=args.njoints,
        nfeats=1,
        cond_mode=config.cond_mode,
        audio_feat=args.audio_feat,
        arch='trans_enc',
        latent_dim=args.latent_dim,
        n_seed=args.n_seed,
        cond_mask_prob=args.cond_mask_prob,
        device=device_name,
        style_dim=args.style_dim,
        source_audio_dim=args.audio_feature_dim,
        audio_feat_dim_latent=args.audio_feat_dim_latent,
    )
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def _load_twh_stats():
    data_mean_ = np.load("../process/gesture_TWH_mean_v0_yqr.npy")
    data_std_ = np.load("../process/gesture_TWH_std_v0_yqr.npy")
    return np.array(data_mean_), np.array(data_std_)


def _make_seed_features(seed_gesture, data_mean, data_std):
    """Convert raw gesture frames into the model's seed feature format.

    The original code builds (pose, velocity, acceleration) and concatenates them.
    Assumes the seed_gesture is loaded with length (n_seed + 2).
    """
    seed_gesture = (seed_gesture - data_mean) / data_std
    seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
    seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
    # (n_seed, njoints)
    seed_gesture_feat = np.concatenate(
        (seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1
    )
    return seed_gesture_feat


def inference(
    args,
    save_dir,
    prefix,
    textaudio,
    sample_fn,
    model,
    n_frames=0,
    smoothing=False,
    skip_timesteps=0,
    style=None,
    seed=123456,
):
    """Run TWH-only sampling and export BVH."""

    torch.manual_seed(seed)

    # TWH: style is a 17-dim vector, speaker is argmax
    speaker = int(np.where(style == np.max(style))[0][0])

    if n_frames == 0:
        n_frames = textaudio.shape[0]
    else:
        textaudio = textaudio[:n_frames]

    real_n_frames = copy.deepcopy(n_frames)
    stride_poses = args.n_poses - args.n_seed

    if n_frames < stride_poses:
        num_subdivision = 1
        n_frames = stride_poses
    else:
        num_subdivision = math.ceil(n_frames / stride_poses)
        n_frames = num_subdivision * stride_poses
        print(
            'real_n_frames: {}, num_subdivision: {}, stride_poses: {}, n_frames: {}, speaker_id: {}'.format(
                real_n_frames,
                num_subdivision,
                stride_poses,
                n_frames,
                speaker,
            )
        )

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    # Pad features to fit subdivision
    textaudio_pad = torch.zeros([n_frames - real_n_frames, args.audio_feature_dim]).to(mydevice)
    textaudio = torch.cat((textaudio, textaudio_pad), 0)
    audio_reshape = textaudio.reshape(num_subdivision, stride_poses, args.audio_feature_dim).transpose(0, 1)

    data_mean, data_std = _load_twh_stats()

    # Optional extra conditioning for DiffuseStyleGesture++
    if args.name == 'DiffuseStyleGesture++':
        # Use a TWH reference clip rather than a BEAT one
        ref_path = "../../TWH_dataset/processed/gesture_metahuman_TWH/val_2023_v0_014_main-agent.npy"
        gesture_flag1 = np.load(ref_path)[: args.n_seed + 2]
        gesture_flag1_feat = _make_seed_features(gesture_flag1, data_mean, data_std)
        gesture_flag1_feat = (
            torch.from_numpy(gesture_flag1_feat)
            .float()
            .transpose(0, 1)
            .unsqueeze(0)
            .to(mydevice)
            .unsqueeze(2)
        )
        model_kwargs_['y']['seed_last'] = gesture_flag1_feat

    shape_ = (1, model.njoints, model.nfeats, args.n_poses)
    out_list = []

    for i in range(0, num_subdivision):
        print(i, num_subdivision)

        # audio_reshape is (stride_poses, num_subdivision, audio_feature_dim)
        model_kwargs_['y']['audio'] = audio_reshape[:, i : i + 1]

        if i == 0:
            # Match the original attention variants
            if args.name == 'DiffuseStyleGesture':
                pad_zeros = torch.zeros([args.n_seed, 1, args.audio_feature_dim]).to(mydevice)
                model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0).transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture+':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture++':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][: -args.n_seed, ...].transpose(0, 1)

            seed_path = "../../TWH_dataset/processed/gesture_metahuman_TWH/val_2023_v0_014_main-agent.npy"
            seed_gesture = np.load(seed_path)[: args.n_seed + 2]

            seed_gesture_feat = _make_seed_features(seed_gesture, data_mean, data_std)
            seed_gesture_feat = (
                torch.from_numpy(seed_gesture_feat)
                .float()
                .transpose(0, 1)
                .unsqueeze(0)
                .to(mydevice)
            )
            model_kwargs_['y']['seed'] = seed_gesture_feat.unsqueeze(2)

        else:
            if args.name == 'DiffuseStyleGesture':
                pad_audio = audio_reshape[-args.n_seed :, i - 1 : i]
                model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0).transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture+':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)
            elif args.name == 'DiffuseStyleGesture++':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][: -args.n_seed, ...].transpose(0, 1)

            # seed = previous chunk tail
            model_kwargs_['y']['seed'] = out_list[-1][..., -args.n_seed :].to(mydevice)

        sample = sample_fn(
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

            # NOTE: This matches the original script (it blends frames). The loop uses
            # len(last_poses) which is 1 for batch=1, but we keep it unchanged.
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[..., j]
                nxt = sample[..., j]
                sample[..., j] = prev * (n - j) / (n + 1) + nxt * (j + 1) / (n + 1)

        out_list.append(sample)

    if "v0" in args.version:
        motion_feature_division = 3
    elif "v2" in args.version:
        motion_feature_division = 1
    else:
        raise ValueError("wrong version name")

    out_list = [i.detach().data.cpu().numpy()[:, : args.njoints // motion_feature_division] for i in out_list]

    if len(out_list) > 1:
        out_dir_vec_1 = np.vstack(out_list[:-1])
        sampled_seq_1 = (
            out_dir_vec_1.squeeze(2)
            .transpose(0, 2, 1)
            .reshape(batch_size, -1, model.njoints // motion_feature_division)
        )
        out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)

    sampled_seq = sampled_seq[:, args.n_seed :]

    out_poses = np.multiply(sampled_seq[0], data_std) + data_mean
    print(out_poses.shape, real_n_frames)
    out_poses = out_poses[:real_n_frames]

    pose2bvh_twh(
        out_poses,
        save_dir,
        prefix,
        pipeline_path="../process/pipeline_rotmat_64_yqr.sav",
    )


def main(
    args,
    save_dir,
    model_path,
    tst_path=None,
    max_len=0,
    skip_timesteps=0,
    tst_prefix=None,
    wav_path=None,
    txt_path=None,
    wavlm_path=None,
    word2vector_path=None,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop

    # the TWH val dataset for testing trained model
    # main_agent_audio.npy, main_agent_gesture.npy, main_agent_text.npy, metadata.csv
    if tst_path is not None:
        metadata_path = os.path.join(tst_path, "metadata.csv")
        participant = "main-agent"
        _, metadict_byfname, _, filenames = load_metadata(metadata_path, participant, return_filenames=True)

        # If user doesn't provide --tst_prefix, run all rows in metadata.csv
        if not tst_prefix:
            tst_prefix = filenames

        tst_audio_dir = os.path.join(tst_path, 'audio_TWH')
        tst_text_dir = os.path.join(tst_path, 'text_TWH')

        for filename in tst_prefix:
            # Accept raw prefix or the fully-qualified key.
            if filename in metadict_byfname:
                filename_key = filename
            else:
                filename_key = filename if filename.endswith(f"_{participant}") else f"{filename}_{participant}"
                if filename_key not in metadict_byfname:
                    raise KeyError(
                        f"Filename '{filename}' not found in metadata. Tried '{filename_key}'."
                    )

            print(f"Processing: {filename_key}")

            _, speaker_id = metadict_byfname[filename_key]
            speaker = np.zeros([args.style_dim])
            speaker[int(speaker_id)] = 1

            audio_path = os.path.join(tst_audio_dir, filename_key + '.npy')
            audio = np.load(audio_path)

            text_path = os.path.join(tst_text_dir, filename_key + '.npy')
            text = np.load(text_path)

            textaudio = np.concatenate((audio, text), axis=-1)
            textaudio = torch.FloatTensor(textaudio).to(mydevice)

            inference(
                args,
                save_dir,
                filename_key,
                textaudio,
                sample_fn,
                model,
                n_frames=max_len,
                smoothing=True,
                skip_timesteps=skip_timesteps,
                style=speaker,
                seed=123456,
            )

    # your own audio and aligned text tsv file for testing
    else:
        # Single-file path: wav + aligned TSV -> features -> sample
        wavlm_model, cfg = wavlm_init(wavlm_path, mydevice)
        word2vector = load_wordvectors(fname=word2vector_path)

        t0 = datetime.now()

        wav = load_audio(wav_path, wavlm_model, cfg)
        clip_len = wav.shape[0]
        tsv = load_tsv(txt_path, word2vector, clip_len)

        textaudio = np.concatenate((wav, tsv), axis=-1)
        textaudio = torch.FloatTensor(textaudio).to(mydevice)

        speaker = np.zeros([args.style_dim])
        speaker[0] = 1

        filename = 'tts'
        inference(
            args,
            save_dir,
            filename,
            textaudio,
            sample_fn,
            model,
            n_frames=max_len,
            smoothing=True,
            skip_timesteps=skip_timesteps,
            style=speaker,
            seed=123456,
        )

        t1 = datetime.now()
        print(f"Total time: {(t1 - t0).total_seconds():.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffuseStyleGesture (TWH-only)')
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture_yqr.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', nargs='+')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--model_path', type=str, default='./model001200000.pt')
    parser.add_argument('--tst_path', type=str, default=None)
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--wavlm_path', type=str, default='./WavLM/WavLM-Large.pt')
    parser.add_argument('--word2vector_path', type=str, default='./crawl-300d-2M.vec')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)

    # Force TWH-only behavior
    config.dataset = 'TWH'

    assert config.name in ['DiffuseStyleGesture', 'DiffuseStyleGesture+', 'DiffuseStyleGesture++']
    if config.name == 'DiffuseStyleGesture+':
        config.cond_mode = 'cross_local_attention4_style1_sample'
    elif config.name == 'DiffuseStyleGesture':
        config.cond_mode = 'cross_local_attention3_style1_sample'
    elif config.name == 'DiffuseStyleGesture++':
        config.cond_mode = 'cross_local_attention5_style1_sample'

    device_name = 'cuda:' + str(config.gpu)
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(config.gpu))

    args.no_cuda = args.gpu

    batch_size = 1

    # Keep save_dir behavior similar to the original
    model_root = config.model_path.split('/')[1] if '/' in config.model_path else '.'
    model_specific = os.path.splitext(os.path.basename(config.model_path))[0]
    config.save_dir = "./" + model_root + '/' + 'sample_dir_' + model_specific + '/'

    if config.tst_prefix is not None and config.tst_path is None:
        config.tst_path = "../../TWH_dataset/processed/"

    print('tst_path', config.tst_path, 'save_dir', config.save_dir)

    main(
        config,
        config.save_dir,
        config.model_path,
        tst_path=config.tst_path,
        max_len=config.max_len,
        skip_timesteps=config.skip_timesteps,
        tst_prefix=config.tst_prefix,
        wav_path=config.wav_path,
        txt_path=config.txt_path,
        wavlm_path=config.wavlm_path,
        word2vector_path=config.word2vector_path,
    )
