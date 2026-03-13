from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import TRAINING_DIR


VAL_LOSS_RE = re.compile(r"Iter\s+\d+:\s+Val loss\s+([0-9.]+)")
PEAK_MEM_RE = re.compile(r"Peak mem\s+([0-9.]+)\s+GB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive overnight segmented training for the writer model.")
    parser.add_argument("--artifact-id", required=True, help="Existing writer artifact to continue training.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--resume-adapter-file", default=None, help="Optional explicit adapter file to warm-start the first segment from.")
    parser.add_argument("--max-segments", type=int, default=24, help="Maximum number of additional segments to run.")
    parser.add_argument("--segment-iters", type=int, default=6, help="Iterations per segment.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=3)
    parser.add_argument("--steps-per-report", type=int, default=3)
    parser.add_argument("--steps-per-eval", type=int, default=999999)
    parser.add_argument("--min-segments", type=int, default=8, help="Minimum new segments before early stopping.")
    parser.add_argument("--patience", type=int, default=6, help="Stop after this many non-improving segments.")
    parser.add_argument("--min-delta", type=float, default=0.03, help="Minimum val-loss improvement to reset patience.")
    parser.add_argument("--target-val-loss", type=float, default=2.15, help="Optional target val loss for graceful stop.")
    parser.add_argument("--min-cooldown", type=int, default=90, help="Minimum rest seconds between segments.")
    parser.add_argument("--max-cooldown", type=int, default=300, help="Maximum rest seconds between segments.")
    parser.add_argument("--pid-file", default=None, help="Optional pid file path.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT_DIR, text=True)


def get_load_average() -> float:
    try:
        output = subprocess.check_output(["uptime"], text=True)
    except Exception:
        return 0.0
    match = re.search(r"load averages?:\s*([0-9.]+)", output)
    return float(match.group(1)) if match else 0.0


def get_memory_free_percentage() -> float:
    try:
        output = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    except Exception:
        return 100.0
    match = re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", output)
    return float(match.group(1)) if match else 100.0


def parse_latest_metrics(training_log: Path) -> tuple[float | None, float | None]:
    try:
        text = training_log.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None, None
    val_matches = VAL_LOSS_RE.findall(text)
    mem_matches = PEAK_MEM_RE.findall(text)
    latest_val = float(val_matches[-1]) if val_matches else None
    latest_mem = float(mem_matches[-1]) if mem_matches else None
    return latest_val, latest_mem


def choose_cooldown(load1: float, free_pct: float, peak_mem: float | None, minimum: int, maximum: int) -> int:
    cooldown = minimum
    if load1 >= 5.0:
        cooldown += 60
    elif load1 >= 3.5:
        cooldown += 30
    if free_pct <= 55:
        cooldown += 90
    elif free_pct <= 70:
        cooldown += 45
    if peak_mem is not None:
        if peak_mem >= 6.5:
            cooldown += 60
        elif peak_mem >= 5.9:
            cooldown += 30
    return max(minimum, min(maximum, cooldown))


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def retain_best_checkpoint(artifact_id: str, latest_val: float | None, segment_index: int) -> dict[str, str] | None:
    if latest_val is None:
        return None
    adapter_dir = ROOT_DIR / "data" / "models" / "writer" / artifact_id
    source = adapter_dir / "adapters.safetensors"
    if not source.exists():
        return None
    best_fixed = adapter_dir / "best_adapters.safetensors"
    best_named = adapter_dir / f"best_seg{segment_index:03d}_val{latest_val:.3f}.safetensors"
    shutil.copy2(source, best_fixed)
    shutil.copy2(source, best_named)
    return {"best_fixed": str(best_fixed), "best_named": str(best_named)}


def write_pid(pid_file: Path | None) -> None:
    if not pid_file:
        return
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")


def remove_pid(pid_file: Path | None) -> None:
    if pid_file and pid_file.exists():
        pid_file.unlink()


def main() -> int:
    args = parse_args()
    pid_file = Path(args.pid_file) if args.pid_file else None
    write_pid(pid_file)

    train_log = TRAINING_DIR / "mlx_sft" / f"{args.artifact_id}.log"
    session_log = TRAINING_DIR / "mlx_sft" / f"{args.artifact_id}.overnight.jsonl"
    state_path = TRAINING_DIR / "mlx_sft" / f"{args.artifact_id}.overnight.state.json"

    best_val: float | None = None
    no_improve_segments = 0
    completed_segments = 0

    try:
        current_resume_file = args.resume_adapter_file
        for segment_index in range(1, args.max_segments + 1):
            started_at = now_iso()
            command = [
                sys.executable,
                "scripts/train_writer_mlx.py",
                "--resume-artifact-id",
                args.artifact_id,
                "--model",
                args.model,
                "--iters",
                str(args.segment_iters),
                "--segments",
                "1",
                "--batch-size",
                str(args.batch_size),
                "--grad-accumulation-steps",
                str(args.grad_accumulation_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--max-seq-length",
                str(args.max_seq_length),
                "--num-layers",
                str(args.num_layers),
                "--val-batches",
                str(args.val_batches),
                "--save-every",
                str(args.save_every),
                "--steps-per-report",
                str(args.steps_per_report),
                "--steps-per-eval",
                str(args.steps_per_eval),
                "--skip-test",
            ]
            if current_resume_file:
                command.extend(["--resume-adapter-file", current_resume_file])

            result = run_command(command)
            latest_val, latest_mem = parse_latest_metrics(train_log)
            completed_segments += 1 if result.returncode == 0 else 0

            improved = False
            retained_paths = None
            if latest_val is not None:
                if best_val is None or latest_val < (best_val - args.min_delta):
                    best_val = latest_val
                    no_improve_segments = 0
                    improved = True
                    retained_paths = retain_best_checkpoint(args.artifact_id, latest_val, segment_index)
                else:
                    no_improve_segments += 1

            load1 = get_load_average()
            free_pct = get_memory_free_percentage()
            cooldown = choose_cooldown(load1, free_pct, latest_mem, args.min_cooldown, args.max_cooldown)

            state = {
                "artifact_id": args.artifact_id,
                "segment_index": segment_index,
                "segment_completed": result.returncode == 0,
                "started_at": started_at,
                "finished_at": now_iso(),
                "latest_val_loss": latest_val,
                "best_val_loss": best_val,
                "latest_peak_mem_gb": latest_mem,
                "load1": load1,
                "memory_free_pct": free_pct,
                "improved": improved,
                "no_improve_segments": no_improve_segments,
                "cooldown_seconds": cooldown,
                "return_code": result.returncode,
                "resume_adapter_file": current_resume_file,
                "retained_paths": retained_paths,
            }
            append_jsonl(session_log, state)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            if result.returncode != 0:
                return result.returncode

            current_resume_file = None

            if (
                completed_segments >= args.min_segments
                and latest_val is not None
                and latest_val <= args.target_val_loss
                and no_improve_segments >= 2
            ):
                return 0

            if completed_segments >= args.min_segments and no_improve_segments >= args.patience:
                return 0

            if segment_index < args.max_segments:
                time.sleep(cooldown)
        return 0
    finally:
        remove_pid(pid_file)


if __name__ == "__main__":
    raise SystemExit(main())
