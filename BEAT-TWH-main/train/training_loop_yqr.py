import functools
import os
import re

import blobfile as bf
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from diffusion.resample import create_named_schedule_sampler

import sys

[sys.path.append(i) for i in ['../process']]


class TrainLoop:
    def __init__(self, args, model, diffusion, device, data=None):
        self.args = args
        self.data = data
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        # `step` counts optimizer updates in the current run.
        # `resume_step` is an offset (e.g., when resuming from a checkpoint).
        self.step = 0
        self.resume_step = int(getattr(args, "resume_step", 0) or 0)
        self.global_batch = self.batch_size

        # Optional resume.
        # - resume_checkpoint: path to model#########.pt
        # - resume_step: offset inferred from checkpoint name if possible
        self.resume_checkpoint = getattr(args, "resume_checkpoint", "") or ""

        # In this yqr loop: max_num_steps means *optimizer update steps*.
        self.max_steps = int(args.max_num_steps)

        self.save_iters = int(args.save_iters)
        self.n_seed = args.n_seed

        # Keep only the latest N checkpoints to avoid filling disk.
        self.max_checkpoints = int(getattr(args, "max_checkpoints", 3))

        self.sync_cuda = torch.cuda.is_available()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Put the OpenAI-baselines-style logger outputs into save_dir.
        # This creates progress.csv you can plot and log.txt for inspection.
        logger.configure(dir=self.save_dir, format_strs=["stdout", "log", "csv"])

        # Optional TensorBoard writer for loss curves.
        self.tb_writer = None
        tb_dir = os.path.join(self.save_dir, "tb")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=tb_dir)
        except Exception:
            self.tb_writer = None

        self.device = device
        if args.audio_feat == "mfcc" or args.audio_feat == "wavlm":
            self.opt = AdamW(
                [
                    {
                        "params": self.mp_trainer.master_params,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                ]
            )
        else:
            self.opt = AdamW(
                [
                    {
                        "params": self.mp_trainer.master_params,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                ]
            )

        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )

        self.use_ddp = False
        self.ddp_model = self.model
        # Masks depend on the *actual* batch size. Do not pre-allocate using
        # args.batch_size because the final batch can be smaller when drop_last=False.
        self.n_poses = int(args.n_poses)

        self._load_resume_state_if_requested()

    def _global_step(self) -> int:
        return int(self.step + self.resume_step)

    def _parse_resume_step_from_filename(self, filename: str) -> int:
        base = os.path.basename(filename)
        m = re.match(r"^model(\d+)\.pt$", base)
        if not m:
            return 0
        try:
            return int(m.group(1))
        except ValueError:
            return 0

    def _resolve_resume_checkpoint(self) -> str:
        if self.resume_checkpoint:
            return self.resume_checkpoint

        # Fallback: if user provided resume_step, try to load from save_dir.
        if self.resume_step and self.resume_step > 0:
            candidate = os.path.join(self.save_dir, f"model{int(self.resume_step):09d}.pt")
            if os.path.exists(candidate):
                return candidate

        return ""

    def _load_resume_state_if_requested(self) -> None:
        ckpt_path = self._resolve_resume_checkpoint()
        if not ckpt_path:
            return
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"resume_checkpoint not found: {ckpt_path}")

        inferred_step = self._parse_resume_step_from_filename(ckpt_path)
        if inferred_step > 0:
            self.resume_step = inferred_step

        logger.log(f"Resuming from checkpoint: {ckpt_path} (resume_step={self.resume_step})")

        state_dict = torch.load(ckpt_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.log(f"Missing keys when loading checkpoint (ok if expected): {len(missing)}")
        if unexpected:
            logger.log(f"Unexpected keys when loading checkpoint (ok if expected): {len(unexpected)}")

        # Load optimizer state if present.
        opt_path = os.path.join(self.save_dir, f"opt{int(self.resume_step):09d}.pt")
        if os.path.exists(opt_path):
            logger.log(f"Loading optimizer state: {opt_path}")
            opt_state = torch.load(opt_path, map_location="cpu")
            self.opt.load_state_dict(opt_state)
        else:
            logger.log(f"Optimizer state not found, continuing without it: {opt_path}")

    def _stop_step(self) -> int:
        if self.lr_anneal_steps and int(self.lr_anneal_steps) > 0:
            return min(self.max_steps, int(self.lr_anneal_steps))
        return self.max_steps

    def run_loop(self):
        stop_step = self._stop_step()
        epoch = 0

        while self._global_step() < stop_step:
            print(f"epoch {epoch}:")
            for batch in tqdm(self.data, desc=f"Epoch {epoch}"):
                global_step = self._global_step()
                if global_step >= stop_step:
                    break

                cond_ = {"y": {}}

                wavlm, pose_seq, style = batch
                bs = pose_seq.shape[0]

                # pose_seq: [B, T, D] -> motion: [B, D, 1, T]
                motion = pose_seq.permute(0, 2, 1).unsqueeze(2).to(
                    self.device, non_blocking=True
                )

                cond_["y"]["seed"] = motion[..., 0 : self.n_seed]
                cond_["y"]["style"] = style.to(self.device, non_blocking=True)
                cond_["y"]["mask_local"] = torch.ones(bs, self.n_poses, device=self.device, dtype=torch.bool)
                cond_["y"]["audio"] = (
                    wavlm.to(torch.float32)[:, self.n_seed :]
                    .to(self.device, non_blocking=True)
                )
                cond_["y"]["mask"] = torch.ones(bs, 1, 1, self.n_poses, device=self.device, dtype=torch.bool)

                self.run_step(motion, cond_)

                # Dump logger values to progress.csv periodically.
                global_step = self._global_step()
                if self.log_interval > 0 and global_step % self.log_interval == 0:
                    out = logger.dumpkvs()
                    if "loss" in out:
                        print(
                            "step[{}]: loss[{:0.5f}]".format(
                                global_step, out["loss"]
                            )
                        )

                if (
                    self.save_iters > 0
                    and global_step > 0
                    and global_step % self.save_iters == 0
                ):
                    self.save()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", ""):
                        return

                self.step += 1

            epoch += 1

        # Final save at end (optional but useful).
        self.save()

        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

    def run_step(self, batch, cond):
        losses = self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

        if losses is not None and self.tb_writer is not None:
            try:
                self.tb_writer.add_scalar(
                    "train/loss",
                    float(losses["loss"].detach().mean().item()),
                    self._global_step(),
                )
            except Exception:
                pass

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        losses = None

        for i in range(0, batch.shape[0], self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                dataset=getattr(self.args, "dataset", None),
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

        return losses

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        if int(self.lr_anneal_steps) <= 0:
            return
        frac_done = self._global_step() / int(self.lr_anneal_steps)
        frac_done = min(max(frac_done, 0.0), 1.0)
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        global_step = self._global_step()
        logger.logkv("step", global_step)
        logger.logkv("samples", (global_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{self._global_step():09d}.pt"

    def _rotate_checkpoints(self):
        if self.max_checkpoints <= 0:
            return

        model_re = re.compile(r"^model(\d{9})\.pt$")
        opt_re = re.compile(r"^opt(\d{9})\.pt$")

        entries = []
        for name in os.listdir(self.save_dir):
            m = model_re.match(name)
            if m:
                entries.append((int(m.group(1)), "model", name))
                continue
            m = opt_re.match(name)
            if m:
                entries.append((int(m.group(1)), "opt", name))

        # Group by step, keep newest steps.
        steps = sorted({s for (s, _, _) in entries})
        if len(steps) <= self.max_checkpoints:
            return
        to_delete_steps = set(steps[: -self.max_checkpoints])

        for s, _, name in entries:
            if s in to_delete_steps:
                try:
                    os.remove(os.path.join(self.save_dir, name))
                except OSError:
                    pass

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self._global_step():09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)

        self._rotate_checkpoints()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
