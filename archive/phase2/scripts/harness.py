import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Dict, List, Optional


def run_cmd(cmd: List[str], stdout_path: Optional[str] = None, stderr_path: Optional[str] = None) -> int:
    
    # Output and error paths optional #
    stdout_f = open(stdout_path, "w") if stdout_path else None
    stderr_f = open(stderr_path, "w") if stderr_path else None
    try:
        p = subprocess.run(
            cmd,
            stdout=stdout_f if stdout_f else None,
            stderr=stderr_f if stderr_f else None,
            check=False,
        )
        return p.returncode
    finally:
        if stdout_f:
            stdout_f.close()
        if stderr_f:
            stderr_f.close()


def get_git_info(repo_dir: str) -> Dict:
    def git(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(["git"] + args, cwd=repo_dir, stderr=subprocess.DEVNULL)
            return out.decode().strip()
        except Exception:
            return None

    commit = git(["rev-parse", "HEAD"])
    dirty = None
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir).decode().strip())
    except Exception:
        pass

    return {"commit": commit, "dirty": dirty, "repo_dir": repo_dir}


def get_host_info() -> Dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    # Kernel / uname details
    try:
        info["uname"] = " ".join(platform.uname())
    except Exception:
        pass

    # Best-effort GPU info (NVIDIA). If not present, it just stays None.
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["nvidia_smi"] = smi
    except Exception:
        info["nvidia_smi"] = None

    return info


def parse_kv_list(kvs: List[str]) -> Dict:
    """Parse ['k=v', 'x=1'] into dict, with simple int/float/bool coercion."""
    out: Dict = {}
    for item in kvs:
        if "=" not in item:
            # Keep as a flag-like key
            out[item] = True
            continue
        k, v = item.split("=", 1)
        v2: object = v
        if v.lower() in ("true", "false"):
            v2 = (v.lower() == "true")
        else:
            # int/float coercion
            try:
                v2 = int(v)
            except ValueError:
                try:
                    v2 = float(v)
                except ValueError:
                    v2 = v
        out[k] = v2
    return out


def select_env(keys: List[str]) -> Dict:
    env: Dict = {}
    for k in keys:
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Run a workload under strace/perf/perf sched and store a reproducible run folder."
    )
    parser.add_argument("--output-dir", required=True, help="Root directory to write runs into.")
    parser.add_argument("--workload", required=True, help="Short name (e.g. hf_gpt2, vanilla_alloc).")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Structured params as k=v. Repeatable. Example: --param prompt_tokens=256 --param threads=8",
    )
    parser.add_argument(
        "--env-keys",
        default="CUDA_VISIBLE_DEVICES,OMP_NUM_THREADS,TORCH_LOGS,TORCHDYNAMO_VERBOSE,CUDA_LAUNCH_BLOCKING",
        help="Comma-separated env vars to persist into config.json.",
    )
    parser.add_argument(
        "--git-repo",
        default=".",
        help="Path to git repo root for commit/dirty capture (default: current dir).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional free-form tag to distinguish runs (e.g. 'cold', 'warm').",
    )
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="Disable perf tools (useful if not permitted on the machine).",
    )
    parser.add_argument(
        "--no-strace",
        action="store_true",
        help="Disable strace.",
    )
    parser.add_argument(
        "--no-sched",
        action="store_true",
        help="Disable perf sched.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Workload command after --. Example: -- python hf.py --prompt_tokens 256",
    )

    args = parser.parse_args()
    if not args.command or args.command[0] != "--":
        print("ERROR: You must separate the workload command with `--`.\n"
              "Example: run_harness.py ... -- python hf.py --prompt_tokens 256")
        sys.exit(2)

    cmd = args.command[1:]  # strip the leading --
    if not cmd:
        print("ERROR: Empty workload command.")
        sys.exit(2)

    # Run folder
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_folder = f"run_{ts}_{args.workload}{tag}"
    run_path = os.path.join(args.output_dir, run_folder)

    raw_dir = os.path.join(run_path, "raw")
    sums_dir = os.path.join(run_path, "sums")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(sums_dir, exist_ok=True)

    # Build config.json
    params = parse_kv_list(args.param)
    env_keys = [k.strip() for k in args.env_keys.split(",") if k.strip()]
    config = {
        "workload": args.workload,
        "tag": args.tag,
        "cmd": cmd,
        "params": params,
        "env": select_env(env_keys),
        "host": get_host_info(),
        "git": get_git_info(args.git_repo),
        "timestamp_iso": dt.datetime.now().isoformat(),
    }

    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # --- strace ---
    if not args.no_strace:
        # Summary table
        strace_summary = os.path.join(sums_dir, "strace_c.txt")
        rc = run_cmd(["strace", "-c", "-o", strace_summary, "--"] + cmd)
        with open(os.path.join(sums_dir, "strace_exitcode.txt"), "w") as f:
            f.write(str(rc) + "\n")

        # Optional full raw trace (useful for later slicing)
        # NOTE: this can be huge; keep it, but in raw/
        strace_prefix = os.path.join(raw_dir, "strace")
        run_cmd(["strace", "-ff", "-tt", "-T", "-o", strace_prefix, "--"] + cmd)

    # --- perf record/report ---
    if not args.no_perf:
        perf_data = os.path.join(raw_dir, "perf.data")
        perf_report = os.path.join(sums_dir, "perf_report.txt")

        # record with callgraphs
        run_cmd(["perf", "record", "-g", "-o", perf_data, "--"] + cmd)

        # text report
        run_cmd(["perf", "report", "--stdio", "-i", perf_data], stdout_path=perf_report)

    # --- perf sched ---
    if not args.no_sched and not args.no_perf:
        sched_data = os.path.join(raw_dir, "perf_sched.data")
        sched_timehist = os.path.join(sums_dir, "sched_timehist.txt")

        run_cmd(["perf", "sched", "record", "-o", sched_data, "--"] + cmd)
        run_cmd(["perf", "sched", "timehist", "-i", sched_data], stdout_path=sched_timehist)

    # Minimal run manifest
    with open(os.path.join(run_path, "MANIFEST.txt"), "w") as f:
        f.write("raw/: large binary/raw logs (perf.data, perf_sched.data, strace.*)\n")
        f.write("sums/: human-readable summaries (strace -c, perf report, sched timehist)\n")
        f.write("config.json: reproduction + parameters + env + host + git\n")

    print(f"Run written to: {run_path}")


if __name__ == "__main__":
    main()

