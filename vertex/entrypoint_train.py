import argparse
import os
import subprocess
import sys

"""
Entrypoint for training on Vertex AI.
Handles both single-node and multi-node (DDP) launches via torchrun.
Assumes your training scripts are train_ddp.py and train_single.py located at /app.

Environment (provided by Vertex AI or args):
- MASTER_ADDR / MASTER_PORT for multi-node
- RANK (node rank) and WORLD_SIZE (num nodes)
- NUM_WORKERS_PER_NODE -> nproc_per_node (GPUs per node)
"""


def env_or(default: str, *keys: str) -> str:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", help="Use multi-node DDP")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Processes per node")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="Args to forward to training script")
    args = parser.parse_args()

    if args.distributed:
        master_addr = env_or("127.0.0.1", "MASTER_ADDR")
        master_port = env_or("23456", "MASTER_PORT")
        node_rank = int(env_or("0", "RANK", "NODE_RANK", "AIP_NODE_RANK"))
        world_size = int(env_or("1", "WORLD_SIZE", "AIP_WORLD_SIZE", "AIP_NUM_WORKERS"))

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.gpus_per_node}",
            f"--nnodes={world_size}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            "/app/train_ddp.py",
        ]
    else:
        cmd = [sys.executable, "/app/train_single.py"]

    if args.args:
        cmd += args.args

    print("Launching:", " ".join(map(str, cmd)))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
