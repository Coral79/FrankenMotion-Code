"""In-training eval callback.

Fires every ``every_n_epochs`` training epochs. Saves a checkpoint, then spawns
``eval_part.py`` + ``src.eval.run`` + ``src.eval.paper_table`` as subprocesses
to produce a paper-Table-1-style CSV. Parses the AVERAGE row, appends one
summary row to ``{run_dir}/eval_in_training/summary.csv``, and forwards the
headline metrics to the Lightning logger.

Subprocess isolation is intentional: the eval pipeline reuses the release
entry points unchanged, gets its own torch state (no shared autograd / cuDNN
caches), and a failure inside eval can't kill training.

Default config matches the user's request: every 200 epochs, 1 turn, val split.
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


_CSV_HEADER = [
    "epoch", "global_step", "timestamp",
    "avg_part_R1", "per_action_R1", "per_seq_R1",
    "per_action_FID", "per_seq_FID",
    "exp_dir",
]

_PAPER_TABLE_COLUMNS = {
    "avg_part_R1":     "Avg-part R@1",
    "per_action_R1":   "Per-action R@1",
    "per_seq_R1":      "Per-seq R@1",
    "per_action_FID":  "Per-action FID",
    "per_seq_FID":     "Per-seq FID",
}


class EvalInTraining(Callback):
    """Run an N-turn eval on val/test every K training epochs.

    Args:
        every_n_epochs: fires at the end of epochs ``K-1, 2K-1, 3K-1, ...``.
        num_turns: how many sampling passes (paper Table 1 uses 20). 1 is fast
            but noisy (~1 pt R@1 swing between adjacent ckpts).
        split: ``val`` or ``test``. Default val so test stays held-out for the
            final paper number.
        device: ``cuda`` or ``cpu``. cuda is faster but introduces ~1pt R@1
            generation drift vs cpu — fine for ranking epochs, not for absolute
            numbers. cpu matches the release reference but is slower.
        protocols: which retrieval protocols to compute. Both are needed to
            populate the full paper Table 1 (Avg-part = guo+threshold; Per-*
            = guo). Set ``("guo",)`` for ~2× speedup if only Per-* matters.
        dataset_name / eval_model_dir / annotations_root: paths forwarded to
            ``src.eval.run``.
        max_runtime_seconds: per-eval timeout. Default 2h.
        enabled: master switch. Off by default so existing reproductions are
            unaffected. Flip to true in trainer.yaml or override on CLI to
            opt in.
    """

    def __init__(
        self,
        every_n_epochs: int = 200,
        num_turns: int = 1,
        split: str = "val",
        device: str = "cuda",
        protocols: Iterable[str] = ("guo", "guo+threshold"),
        dataset_name: str = "frankenstein-dataset",
        eval_model_dir: str = "pretrained/eval_model",
        annotations_root: str = "datasets/annotations",
        max_runtime_seconds: int = 7200,
        enabled: bool = False,
    ):
        super().__init__()
        self.every_n_epochs = int(every_n_epochs)
        self.num_turns = int(num_turns)
        self.split = str(split)
        self.device = str(device)
        self.protocols = list(protocols)
        self.dataset_name = str(dataset_name)
        self.eval_model_dir = str(eval_model_dir)
        self.annotations_root = str(annotations_root)
        self.max_runtime_seconds = int(max_runtime_seconds)
        self.enabled = bool(enabled)

        self._eval_root: Path | None = None
        self._summary_csv: Path | None = None
        self._run_dir: Path | None = None

    def _resolve_run_dir(self, trainer: Trainer) -> Path:
        # train_part.py wires CSVLogger(save_dir=${run_dir}, name="logs") →
        # ckpts land at {run_dir}/logs/checkpoints/. Trainer.default_root_dir
        # also points at {run_dir} when CWD is the run dir; fall back to it.
        try:
            log_dir = trainer.loggers[0].save_dir
            if log_dir:
                return Path(log_dir)
        except (IndexError, AttributeError):
            pass
        return Path(trainer.default_root_dir or os.getcwd())

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None):
        if not self.enabled:
            return
        self._run_dir = self._resolve_run_dir(trainer)
        self._eval_root = self._run_dir / "eval_in_training"
        self._eval_root.mkdir(parents=True, exist_ok=True)
        self._summary_csv = self._eval_root / "summary.csv"
        if not self._summary_csv.exists():
            with open(self._summary_csv, "w", newline="") as f:
                csv.writer(f).writerow(_CSV_HEADER)
        logger.info(
            "[EvalInTraining] enabled — every %d epochs, %d turn(s), split=%s, "
            "device=%s, protocols=%s, run_dir=%s",
            self.every_n_epochs, self.num_turns, self.split, self.device,
            self.protocols, self._run_dir,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.enabled or self._run_dir is None:
            return
        # current_epoch is 0-indexed and increments AFTER on_train_epoch_end of
        # PL >=1.6, but on_train_epoch_end is called when the epoch is "done"
        # so we treat it as already 0-indexed completed. Fire when (e+1) % K == 0.
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        if trainer.global_rank != 0:
            return  # only rank 0 runs eval

        ckpt_name = f"eval_at_epoch_{epoch:04d}"
        ckpt_path = self._run_dir / "logs" / "checkpoints" / f"{ckpt_name}.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(ckpt_path))
        logger.info("[EvalInTraining] saved %s", ckpt_path)

        try:
            exp_dir = self._run_eval_pipeline(ckpt_name)
        except subprocess.TimeoutExpired:
            logger.warning(
                "[EvalInTraining] timed out after %ds (epoch %d) — skipping",
                self.max_runtime_seconds, epoch,
            )
            return
        except subprocess.CalledProcessError as e:
            logger.warning(
                "[EvalInTraining] subprocess rc=%d (epoch %d): %s",
                e.returncode, epoch, e,
            )
            return
        except Exception as e:
            logger.warning("[EvalInTraining] unexpected error: %s", e)
            return

        if exp_dir is None:
            return

        metrics = self._parse_paper_table(exp_dir / "paper_table_in_training.csv")
        self._append_summary_row(epoch, int(trainer.global_step), metrics, exp_dir)
        self._log_to_lightning(trainer, metrics)

    def _run_eval_pipeline(self, ckpt_name: str) -> Path | None:
        run_dir = str(self._run_dir)

        # Stage 1: generation
        gen_cmd = [
            "python", "eval_part.py",
            f"run_dir={run_dir}",
            f"ckpt={ckpt_name}",
            f"num_runs={self.num_turns}",
            f"input_type={self.split}",
            f"device={self.device}",
            "video_samples_per_run=0",
        ]
        logger.info("[EvalInTraining] $ %s", " ".join(gen_cmd))
        subprocess.run(gen_cmd, check=True, timeout=self.max_runtime_seconds)

        gen_root = Path(run_dir) / "generations" / "t2m"
        candidates = sorted(
            gen_root.glob(f"{self.split}_{ckpt_name}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning("[EvalInTraining] no generation dir under %s", gen_root)
            return None
        exp_dir = candidates[0]

        # Stage 2: metrics for each protocol (cpu — release reference)
        for protocol in self.protocols:
            m_cmd = [
                "python", "-m", "src.eval.run", str(exp_dir),
                "--calculate_gt_metrics",
                f"--dataset_name={self.dataset_name}",
                f"--max_runs={self.num_turns}",
                f"--protocol={protocol}",
                f"--split={self.split}",
                f"--eval_model_dir={self.eval_model_dir}",
                f"--base_splits_root={self.annotations_root}",
                f"--base_annotations_root={self.annotations_root}",
                "--device=cpu",
            ]
            logger.info("[EvalInTraining] $ %s", " ".join(m_cmd))
            subprocess.run(m_cmd, check=True, timeout=self.max_runtime_seconds)

        # Stage 3: paper_table summary
        out_csv = exp_dir / "paper_table_in_training.csv"
        pt_cmd = [
            "python", "-m", "src.eval.paper_table", str(exp_dir),
            f"--max_runs={self.num_turns}",
            f"--dataset_name={self.dataset_name}",
            f"--save_csv={out_csv}",
        ]
        logger.info("[EvalInTraining] $ %s", " ".join(pt_cmd))
        subprocess.run(pt_cmd, check=True, timeout=600)
        return exp_dir

    def _parse_paper_table(self, csv_path: Path) -> dict:
        out = {k: float("nan") for k in _PAPER_TABLE_COLUMNS}
        if not csv_path.exists():
            logger.warning("[EvalInTraining] paper_table csv missing: %s", csv_path)
            return out
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("row") == "AVERAGE":
                    for key, col in _PAPER_TABLE_COLUMNS.items():
                        try:
                            out[key] = float(row[col])
                        except (KeyError, ValueError):
                            pass
                    break
        return out

    def _append_summary_row(self, epoch: int, global_step: int, metrics: dict, exp_dir: Path):
        if self._summary_csv is None:
            return
        with open(self._summary_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, global_step, datetime.now().isoformat(timespec="seconds"),
                metrics.get("avg_part_R1"),
                metrics.get("per_action_R1"),
                metrics.get("per_seq_R1"),
                metrics.get("per_action_FID"),
                metrics.get("per_seq_FID"),
                str(exp_dir),
            ])
        logger.info(
            "[EvalInTraining] epoch %d → Avg-part R@1=%.3f / Per-action R@1=%.3f / "
            "Per-seq R@1=%.3f / Per-action FID=%.3f / Per-seq FID=%.3f",
            epoch,
            metrics.get("avg_part_R1", float("nan")),
            metrics.get("per_action_R1", float("nan")),
            metrics.get("per_seq_R1", float("nan")),
            metrics.get("per_action_FID", float("nan")),
            metrics.get("per_seq_FID", float("nan")),
        )

    def _log_to_lightning(self, trainer: Trainer, metrics: dict):
        if not trainer.loggers:
            return
        clean = {f"in_training_eval/{k}": v for k, v in metrics.items() if v == v}  # drop NaN
        if not clean:
            return
        for lg in trainer.loggers:
            try:
                lg.log_metrics(clean, step=int(trainer.global_step))
            except Exception as e:
                logger.debug("[EvalInTraining] logger %s rejected metrics: %s", type(lg).__name__, e)
