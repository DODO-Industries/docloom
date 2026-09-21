"""
Autonomous overnight pipeline — fully offline after this launch (TinyStories
model + SQuAD data already cached locally; LM Studio runs locally on
127.0.0.1, no internet needed). Runs each stage to completion, logs
everything, never crashes the whole run on one stage's failure, and writes a
final summary report at the end.

Stages:
  1. Generate + ingest a big TinyStories batch (more Loom-native fluent data)
  2. Retrain Stage A on the enlarged corpus
  3. Grow real Loom/Qwen Stage B data a bit more (if LM Studio is reachable)
  4. Retrain Stage B (SQuAD pass + real-Loom pass) on the new Stage A checkpoint
  5. Test generation with the final checkpoint, write summary report

Usage:
    python -m module_AI.scripts.run_overnight
"""
import json
import os
import subprocess
import sys
import time
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCH_DIR = os.path.join(PROJECT_ROOT, "module_AI", "benchmarks")
REPORT_PATH = os.path.join(BENCH_DIR, "overnight_report.json")
os.makedirs(BENCH_DIR, exist_ok=True)

PYTHON = sys.executable
report = {"stages": [], "started": time.time()}


def run_stage(name, args, timeout_s=None):
    print(f"\n{'='*60}\n[overnight] STARTING: {name}\n{'='*60}", flush=True)
    t0 = time.perf_counter()
    entry = {"name": name, "args": args}
    try:
        result = subprocess.run(
            [PYTHON, "-u"] + args, cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=timeout_s,
        )
        entry["returncode"] = result.returncode
        entry["stdout_tail"] = result.stdout[-4000:]
        entry["stderr_tail"] = result.stderr[-2000:]
        # Also print live to this script's own stdout so a tail -f shows progress
        print(result.stdout[-6000:])
        if result.returncode != 0:
            print(f"[overnight] STAGE FAILED (rc={result.returncode}): {name}")
            print(result.stderr[-2000:])
    except subprocess.TimeoutExpired as e:
        entry["returncode"] = "timeout"
        entry["stdout_tail"] = (e.stdout or "")[-4000:] if e.stdout else ""
        print(f"[overnight] STAGE TIMED OUT: {name}")
    except Exception as e:
        entry["returncode"] = "exception"
        entry["error"] = str(e)
        traceback.print_exc()
    entry["elapsed_s"] = round(time.perf_counter() - t0, 1)
    print(f"[overnight] FINISHED: {name} in {entry['elapsed_s']:.0f}s (rc={entry['returncode']})", flush=True)
    report["stages"].append(entry)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return entry


def lm_studio_reachable():
    try:
        import requests
        return requests.get("http://127.0.0.1:1234/v1/models", timeout=5).status_code == 200
    except Exception:
        return False


def main():
    brain_dir = os.path.join(PROJECT_ROOT, "module_AI", "data", "stress_brain")

    # Stage 1: more Loom-native fluent data (offline, local model, no internet)
    run_stage(
        "Generate + ingest TinyStories batch",
        ["-m", "module_AI.scripts.generate_tinystories_data", "--count", "8000", "--batch-size", "64",
         "--brain-dir", brain_dir],
    )

    # Stage 2: retrain Stage A on the enlarged corpus
    run_stage(
        "Retrain Stage A (self-supervised, enlarged corpus)",
        ["-m", "module_AI.model.pretrain", "--brain-dir", brain_dir,
         "--limit", "90000", "--epochs", "3", "--batch-size", "16", "--lr", "3e-4"],
    )

    # Stage 3: grow real Loom/Qwen Stage B data, only if LM Studio is up
    if lm_studio_reachable():
        run_stage(
            "Grow real Loom/Qwen Stage B data",
            ["-m", "module_AI.scripts.grow_stage_b_data"],
            timeout_s=1800,
        )
    else:
        report["stages"].append({"name": "Grow real Loom/Qwen Stage B data", "skipped": "LM Studio not reachable"})

    # Stage 4: retrain Stage B (SQuAD pass + real-Loom pass) on the new Stage A checkpoint
    run_stage(
        "Retrain Stage B (SQuAD pass + real-Loom pass)",
        ["-m", "module_AI.model.train", "--squad-epochs", "1", "--loom-epochs", "8",
         "--lr", "3e-4", "--batch-size", "16"],
    )

    # Stage 5: final generation test + summary
    run_stage(
        "Final generation sanity test",
        ["-m", "module_AI.scripts.final_gen_check"],
    )

    report["finished"] = time.time()
    report["total_elapsed_h"] = round((report["finished"] - report["started"]) / 3600, 2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'='*60}\n[overnight] ALL STAGES DONE in {report['total_elapsed_h']:.2f}h\n"
          f"Report: {REPORT_PATH}\n{'='*60}")


if __name__ == "__main__":
    main()
