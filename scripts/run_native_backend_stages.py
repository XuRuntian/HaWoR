"""Run GPU stages sequentially in fresh processes, retaining each complete log."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "full"], required=True)
    parser.add_argument("--output", type=Path, default=root/"outputs/pre_frontend_comparison/calibrated_native_backend_full_60s")
    parser.add_argument("--buffer", type=int, default=512)
    args = parser.parse_args()
    logs = args.output / "logs"
    assert (args.output / "manifest.json").exists(), "Run --stage prepare first"
    logs.mkdir(exist_ok=True)
    tasks = [("masks", "full", branch) for branch in ("native", "optimized")] if args.phase == "pilot" else []
    tasks += [(stage,args.phase,branch) for branch in ("native", "optimized") for stage in ("slam", "metric", "world", "fill")]
    for stage,phase,branch in tasks:
        command = [sys.executable,str(root/"scripts/run_native_backend_comparison.py"),"--output",str(args.output),
                   "--stage",stage,"--phase",phase,"--branch",branch,"--buffer",str(args.buffer)]
        stamp = time.time_ns()
        log = logs / f"{phase}_{branch}_{stage}_{stamp}.log"
        print(f"START {phase} {branch} {stage}: {log}",flush=True)
        start = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        record = {"command":command,"exit_code":result.returncode,"seconds":time.perf_counter()-start,"log":str(log)}
        (log.with_suffix(".json")).write_text(json.dumps(record,indent=2)+"\n")
        print(f"END {phase} {branch} {stage}: exit={result.returncode}, seconds={record['seconds']:.1f}",flush=True)
        if result.returncode:
            print(log.read_text()[-10000:],flush=True)
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
