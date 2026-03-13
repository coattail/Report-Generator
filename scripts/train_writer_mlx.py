from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import MODEL_DIR, TRAINING_DIR
from app.db import get_connection, init_db
from app.services.corpus import latest_style_profile
from app.services.training import DEFAULT_WRITER_MODEL, FALLBACK_WRITER_MODEL, get_production_model, prepare_sft_dataset
from app.utils import json_dumps, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local MLX LoRA training for the writer model.")
    parser.add_argument("--model", default=DEFAULT_WRITER_MODEL, help="Base MLX model repo or local path.")
    parser.add_argument("--iters", type=int, default=220, help="Training iterations.")
    parser.add_argument("--segments", type=int, default=1, help="Number of warm-start training segments to run.")
    parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size.")
    parser.add_argument("--grad-accumulation-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of transformer layers to tune.")
    parser.add_argument("--val-batches", type=int, default=25, help="Validation batches per eval run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint save interval.")
    parser.add_argument("--steps-per-report", type=int, default=10, help="Loss reporting interval.")
    parser.add_argument("--steps-per-eval", type=int, default=50, help="Validation interval.")
    parser.add_argument("--artifact-id", default=None, help="Explicit artifact id, mainly for retries.")
    parser.add_argument("--resume-artifact-id", default=None, help="Resume training from an existing artifact id.")
    parser.add_argument("--resume-adapter-file", default=None, help="Explicit adapter weights file to warm-start from.")
    parser.add_argument("--cooldown-seconds", type=int, default=0, help="Sleep time between segments.")
    parser.add_argument("--conservative-7b", action="store_true", help="Apply a low-load preset for 7B segmented training.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a 1-iteration smoke test.")
    parser.add_argument("--skip-test", action="store_true", help="Skip test-set evaluation after training.")
    return parser.parse_args()


def _insert_artifact(artifact_id: str, model_name: str, adapter_path: Path, dataset_dir: Path, log_path: Path, args: argparse.Namespace) -> None:
    production = get_production_model("writer")
    metrics = {
        "runtime": "mlx-lm-lora",
        "status_detail": "starting",
        "dataset_dir": str(dataset_dir),
        "log_path": str(log_path),
        "style_profile_version": latest_style_profile()["version"],
        "training_args": {
            "iters": 1 if args.smoke_test else args.iters,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "num_layers": args.num_layers,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
        },
    }
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO model_artifacts (
              id, role, base_model, adapter_path, status, is_production, parent_artifact_id, metrics_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                "writer",
                model_name,
                str(adapter_path),
                "running",
                0,
                production["id"] if production.get("id") else None,
                json_dumps(metrics),
                now_iso(),
            ),
        )


def _update_artifact(artifact_id: str, status: str, extra_metrics: dict) -> None:
    with get_connection() as connection:
        row = connection.execute("SELECT metrics_json FROM model_artifacts WHERE id = ?", (artifact_id,)).fetchone()
        current_metrics = json.loads(row["metrics_json"]) if row else {}
        current_metrics.update(extra_metrics)
        connection.execute(
            "UPDATE model_artifacts SET status = ?, metrics_json = ? WHERE id = ?",
            (status, json_dumps(current_metrics), artifact_id),
        )


def _load_artifact(artifact_id: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "role": row["role"],
        "base_model": row["base_model"],
        "adapter_path": row["adapter_path"],
        "status": row["status"],
        "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
        "created_at": row["created_at"],
    }


def _apply_conservative_7b_preset(args: argparse.Namespace) -> None:
    if not args.conservative_7b:
        return
    args.model = DEFAULT_WRITER_MODEL
    if args.iters == 220:
        args.iters = 10
    if args.segments == 1:
        args.segments = 6
    if args.grad_accumulation_steps == 4:
        args.grad_accumulation_steps = 1
    if args.learning_rate == 1e-5:
        args.learning_rate = 5e-6
    if args.max_seq_length == 1024:
        args.max_seq_length = 384
    if args.num_layers == 8:
        args.num_layers = 2
    if args.val_batches == 25:
        args.val_batches = 5
    if args.save_every == 50:
        args.save_every = 5
    if args.steps_per_report == 10:
        args.steps_per_report = 5
    if args.steps_per_eval == 50:
        args.steps_per_eval = 999999
    if args.cooldown_seconds == 0:
        args.cooldown_seconds = 90
    args.skip_test = True


def _build_command(
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    adapter_path: Path,
    resume_adapter_file: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(dataset_dir),
        "--batch-size",
        str(args.batch_size),
        "--iters",
        str(1 if args.smoke_test else args.iters),
        "--val-batches",
        str(args.val_batches),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--grad-accumulation-steps",
        str(args.grad_accumulation_steps),
        "--adapter-path",
        str(adapter_path),
        "--save-every",
        str(args.save_every),
        "--max-seq-length",
        str(args.max_seq_length),
        "--num-layers",
        str(args.num_layers),
        "--fine-tune-type",
        "lora",
        "--optimizer",
        "adamw",
        "--grad-checkpoint",
        "--seed",
        str(args.seed),
    ]
    if resume_adapter_file:
        command.extend(["--resume-adapter-file", resume_adapter_file])
    if not args.skip_test:
        command.append("--test")
    return command


def main() -> int:
    args = parse_args()
    init_db()
    _apply_conservative_7b_preset(args)

    resuming = bool(args.resume_artifact_id)
    existing = _load_artifact(args.resume_artifact_id) if resuming else None
    if resuming and not existing:
        print(f"Resume artifact not found: {args.resume_artifact_id}", file=sys.stderr)
        return 2

    artifact_id = args.resume_artifact_id or args.artifact_id or uuid.uuid4().hex
    dataset_dir = Path(existing["metrics"]["dataset_dir"]) if existing else TRAINING_DIR / "mlx_sft" / artifact_id
    adapter_path = Path(existing["adapter_path"]) if existing else MODEL_DIR / "writer" / artifact_id
    log_path = TRAINING_DIR / "mlx_sft" / f"{artifact_id}.log"
    adapter_path.mkdir(parents=True, exist_ok=True)
    dataset_meta = existing["metrics"].get("dataset_meta") if existing else None
    if not dataset_meta:
        dataset_meta = prepare_sft_dataset(dataset_dir, seed=args.seed)
    if not resuming:
        _insert_artifact(artifact_id, args.model, adapter_path, dataset_dir, log_path, args)
    else:
        _update_artifact(
            artifact_id,
            "running",
            {
                "runtime": "mlx-lm-lora",
                "status_detail": "resuming",
                "dataset_dir": str(dataset_dir),
                "dataset_meta": dataset_meta,
                "log_path": str(log_path),
                "last_resumed_at": now_iso(),
            },
        )

    log_mode = "a" if resuming else "w"
    result_code = 0
    last_resume_file = args.resume_adapter_file
    if not last_resume_file:
        adapter_file = adapter_path / "adapters.safetensors"
        if adapter_file.exists():
            last_resume_file = str(adapter_file)

    with log_path.open(log_mode, encoding="utf-8") as log_handle:
        if log_mode == "a":
            log_handle.write("\n")
        log_handle.write(
            "RUN_META: "
            + json.dumps(
                {
                    "artifact_id": artifact_id,
                    "resuming": resuming,
                    "segments": args.segments,
                    "conservative_7b": args.conservative_7b,
                    "started_at": now_iso(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_handle.write("DATASET: " + json.dumps(dataset_meta, ensure_ascii=False) + "\n\n")
        log_handle.flush()

        for segment_index in range(args.segments):
            command = _build_command(
                args=args,
                dataset_dir=dataset_dir,
                adapter_path=adapter_path,
                resume_adapter_file=last_resume_file,
            )
            log_handle.write(
                f"SEGMENT {segment_index + 1}/{args.segments} COMMAND: " + " ".join(command) + "\n"
            )
            log_handle.flush()
            result = subprocess.run(command, cwd=ROOT_DIR, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            result_code = result.returncode
            if result_code != 0:
                break
            latest_adapter = adapter_path / "adapters.safetensors"
            if latest_adapter.exists():
                last_resume_file = str(latest_adapter)
            _update_artifact(
                artifact_id,
                "running",
                {
                    "dataset_dir": str(dataset_dir),
                    "dataset_meta": dataset_meta,
                    "last_segment_completed": segment_index + 1,
                    "segments_requested": args.segments,
                    "segment_iters": 1 if args.smoke_test else args.iters,
                    "cooldown_seconds": args.cooldown_seconds,
                    "last_resume_adapter_file": last_resume_file,
                    "last_segment_completed_at": now_iso(),
                },
            )
            if segment_index + 1 < args.segments and args.cooldown_seconds > 0:
                log_handle.write(f"Cooling down for {args.cooldown_seconds}s before next segment.\n")
                log_handle.flush()
                time.sleep(args.cooldown_seconds)

    metrics = {
        "dataset_dir": str(dataset_dir),
        "dataset_meta": dataset_meta,
        "dataset_counts": dataset_meta["counts"],
        "example_count": dataset_meta["example_count"],
        "log_path": str(log_path),
        "completed_at": now_iso(),
        "return_code": result_code,
        "runtime": "mlx-lm-lora",
        "smoke_test": args.smoke_test,
        "segments_requested": args.segments,
        "segment_iters": 1 if args.smoke_test else args.iters,
        "cooldown_seconds": args.cooldown_seconds,
        "last_resume_adapter_file": last_resume_file,
    }
    if result_code == 0:
        _update_artifact(artifact_id, "trained", metrics)
        print(f"Training completed: {artifact_id}")
        print(f"Adapter path: {adapter_path}")
        print(f"Log path: {log_path}")
        return 0

    _update_artifact(artifact_id, "failed", metrics)
    print(f"Training failed: {artifact_id}", file=sys.stderr)
    print(f"See log: {log_path}", file=sys.stderr)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
