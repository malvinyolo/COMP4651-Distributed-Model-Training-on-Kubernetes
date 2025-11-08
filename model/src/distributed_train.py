import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datamod import load_datasets   # your existing dataset loader
from models import build_model      # your existing model factory
from train import fit               # your teammate's unmodified training loop
from utils import load_config       # however config is loaded normally

from torch.utils.data import DataLoader, DistributedSampler

def main():

    # --- Setup distributed ---
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    distributed = WORLD_SIZE > 1
    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    # --- Load config, model, datasets (no modification to existing code) ---
    cfg = load_config("config_example.yaml")
    model = build_model(cfg).to(device)

    train_ds, val_ds, test_ds = load_datasets(cfg)

    # --- Use DistributedSamplers ONLY here ---
    train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True) if distributed else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=WORLD_SIZE, rank=RANK, shuffle=False) if distributed else None

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False, sampler=val_sampler)

    # --- Wrap model in DDP OUTSIDE teammate code ---
    if distributed:
        model = DDP(model, device_ids=[LOCAL_RANK] if torch.cuda.is_available() else None)

    # --- Train using your teammate's original fit() [unchanged] ---
    save_path = "./best_model.pt"
    if distributed and RANK != 0:
        # non-master workers write model to temp path (gets ignored)
        save_path = f"./best_model_rank_{RANK}.pt"

    fit(model, train_loader, val_loader, cfg, device, save_path)

    # --- Cleanup ---
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
