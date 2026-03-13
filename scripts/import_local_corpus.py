from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import init_db
from app.services.corpus import import_local_paths
from app.services.training import run_sft_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local briefing corpus files into Local Briefing Studio.")
    parser.add_argument("source", type=Path, help="Folder containing PDF/DOCX/TXT files.")
    parser.add_argument("--password", default=None, help="Default password for encrypted PDFs.")
    parser.add_argument("--train-sft", action="store_true", help="Run initial SFT dataset build after import.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}
    )
    if not paths:
        print(f"No supported files found in: {source_dir}", file=sys.stderr)
        return 1

    init_db()
    result = import_local_paths(paths, pdf_password=args.password)
    print(
        f"Imported {result['imported_documents']} documents, skipped {result['skipped_documents']}, "
        f"profile {result['style_profile_version']}"
    )

    if args.train_sft:
        training_result = run_sft_training()
        print(
            f"SFT artifact {training_result['artifact_id']} ready at {training_result['dataset_path']} "
            f"with {training_result['metrics']['example_count']} examples"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
