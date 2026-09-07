"""Fresh-process depth experiment stages with complete logs and exit status."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root/"outputs/pre_frontend_comparison/depth_replacement_full_60s"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=parent/"run_001")
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == parent or not out.is_relative_to(parent) or out.is_relative_to(parent/"process_logs"):
        raise ValueError("Use an isolated numbered depth run directory")
    stages = ("prepare", "pilot", "render-pilot") if args.phase == "pilot" else ("validate", "full", "render", "report")
    if args.phase == "pilot" and out.exists():
        raise ValueError("Pilot requires a fresh run directory")
    logs = parent/"process_logs"/out.name
    logs.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        command = [sys.executable, str(root/"scripts/run_depth_replacement.py"), "--output", str(out), "--stage", stage]
        log = logs/f"{time.time_ns()}_{stage}.log"
        print(f"START {stage}: {log}", flush=True)
        start = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        record = {"command": command, "exit_code": result.returncode, "seconds": time.perf_counter()-start,
                  "complete_process_log": str(log), "stage": stage}
        log.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
        if out.exists():
            (out/"logs").mkdir(exist_ok=True)
            for path in (log, log.with_suffix(".json")):
                shutil.copyfile(path, out/"logs"/path.name)
        print(f"END {stage}: exit={result.returncode}, seconds={record['seconds']:.3f}", flush=True)
        if result.returncode:
            print(log.read_text()[-15000:], flush=True)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
