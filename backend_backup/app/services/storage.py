import json
import shutil
import time
from pathlib import Path


def cleanup_old_runs(runs_dir: Path, ttl_seconds: int):
    if ttl_seconds <= 0:
        return

    now = time.time()

    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue

        meta = d / "meta.json"

        try:
            created = d.stat().st_mtime

            if meta.exists():
                with open(meta, "r", encoding="utf-8") as f:
                    mj = json.load(f)
                created = float(mj.get("created_at", created))

            if (now - created) > ttl_seconds:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass