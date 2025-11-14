import argparse
import os
import subprocess
import sys

"""
Lightweight entrypoint to call your existing data pipeline for a specific shard/ticker.
This keeps your code unchanged while letting Vertex AI run many parallel containers.

Contract:
- Expects your data pipeline to be runnable via data-pipeline/src/run_pipeline.py
- We pass through ticker list (single or multiple), input/output URIs, and any flags.
- Outputs should be written to GCS paths provided by --output_gcs.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, nargs="+", required=True, help="Tickers to process")
    parser.add_argument("--output_gcs", type=str, required=True, help="gs:// bucket/prefix for outputs")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional args forwarded to pipeline")
    args = parser.parse_args()

    # Build command to call your existing script; adjust if your CLI differs
    script = os.path.join("/app", "src", "run_pipeline.py")
    if not os.path.exists(script):
        print(f"Expected pipeline script not found at {script}", file=sys.stderr)
        sys.exit(2)

    cmd = [sys.executable, script, "--output_gcs", args.output_gcs]
    for t in args.tickers:
        cmd += ["--ticker", t]
    if args.extra_args:
        cmd += args.extra_args

    print("Launching:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
