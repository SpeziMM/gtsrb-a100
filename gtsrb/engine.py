import contextlib
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms as T

from .data import digest, write_json
from .model import Net


class Signs(Dataset):
    def __init__(self, data, split, augmentation="none", limit=None):
        self.images = np.load(Path(data) / f"{split}_images.npy", allow_pickle=False)
        self.labels = np.load(Path(data) / f"{split}_labels.npy", allow_pickle=False)
        if limit is not None:
            # Spread smoke samples through the dataset rather than taking one class.
            idx = np.linspace(0, len(self.labels) - 1, min(limit, len(self.labels)), dtype=int)
            self.images, self.labels = self.images[idx], self.labels[idx]
        self.augmentation = augmentation
        self.flip = T.RandomHorizontalFlip()
        if augmentation == "assignment":
            self.transform = T.Compose([
                T.RandomAffine(5, translate=(.05, .05), scale=(.95, 1.05), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomCrop(28), T.ColorJitter(.8, contrast=.4),
                T.RandomApply([T.GaussianBlur(3, sigma=(.1, .6))], p=.15),
            ])
        elif augmentation in ("mild", "moderate"):
            degrees, shift, zoom, brightness, contrast = (
                (8, .05, .05, .2, .2) if augmentation == "mild" else (12, .08, .1, .3, .25))
            # Keep every sign's handedness and color identity intact.
            self.transform = T.Compose([
                T.RandomAffine(degrees, translate=(shift, shift), scale=(1-zoom, 1+zoom),
                               interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(28), T.ColorJitter(brightness=brightness, contrast=contrast),
            ])
        elif augmentation == "none":
            self.transform = T.CenterCrop(28)
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        image = self.transform(Image.fromarray(self.images[idx]))
        if self.augmentation == "assignment" and label in (11, 12, 13, 17, 18, 26, 30, 35):
            image = self.flip(image)
        return self.to_tensor(image), label


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, batch, workers, device, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                      worker_init_fn=seed_worker, generator=generator)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else contextlib.nullcontext()


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Run time budget reached")


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    true, pred, hit = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(hit, true, out=np.zeros(43, dtype=float), where=true > 0)
    precision = np.divide(hit, pred, out=np.zeros(43, dtype=float), where=pred > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(43), where=precision+recall > 0)
    return {"n": int(cm.sum()), "accuracy": float(hit.sum()/max(1, cm.sum())),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall[true > 0].mean()),
            "confusion_matrix": cm.tolist(),
            "per_class": [{"label": i, "support": int(true[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i in range(43)]}


@torch.inference_mode()
def evaluate(model, batches, device, precision="fp32", deadline=math.inf):
    model.eval()
    cm = torch.zeros(43*43, dtype=torch.int64, device=device)
    loss_sum = torch.zeros((), device=device)
    count = 0
    for x, y in batches:
        check_deadline(deadline)
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("Non-finite evaluation loss")
        cm += torch.bincount(43*y + logits.argmax(1), minlength=43*43)
        loss_sum += loss
        count += len(y)
    result = metrics(cm.cpu().numpy().reshape(43, 43))
    result["loss"] = loss_sum.item()/max(count, 1)
    return result


def save_checkpoint(path, state):
    tmp = Path(path).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(config, data, output, device, workers, seed, deadline, resume=False, limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    trainset = Signs(data, "train", config["augmentation"], limit)
    valset = Signs(data, "val", "none", limit)
    training = loader(trainset, config["batch_size"], workers, device, True, generator)
    validation = loader(valset, 512, workers, device)
    model = Net().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"], eta_min=config["lr"]*.01) if config["cosine"] else None
    precision = config["precision"] if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; use an A100 or set precision to fp32")
    manifest_sha = digest(Path(data)/"manifest.json")
    start_epoch, history, best, stale, elapsed = 0, [], -1., 0, 0.
    last = output/"last.pt"
    if last.exists() and not resume:
        raise FileExistsError(f"{last} exists; use --resume or a new output directory")
    if resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["seed"] != seed or saved["manifest_sha256"] != manifest_sha:
            raise ValueError("Cannot resume with changed config, seed, or dataset")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler:
            scheduler.load_state_dict(saved["scheduler"])
        start_epoch, history, best, stale = saved["epoch"]+1, saved["history"], saved["best"], saved["stale"]
        elapsed = saved["elapsed_seconds"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        generator.set_state(saved["loader_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    write_json(output/"config.json", {**config, "seed": seed, "actual_precision": precision,
                                      "manifest_sha256": manifest_sha})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    stop_reason = "epoch_limit"
    try:
        for epoch in range(start_epoch, config["epochs"]):
            check_deadline(deadline)
            epoch_start = time.monotonic()
            model.train()
            total_loss = torch.zeros((), device=device)
            seen = 0
            for x, y in training:
                check_deadline(deadline)
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, precision):
                    loss = nn.functional.cross_entropy(model(x), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                total_loss += loss.detach()*len(y)
                seen += len(y)
            val = evaluate(model, validation, device, "fp32", deadline)
            sync(device)
            row = {"epoch": epoch+1, "train_loss": total_loss.item()/seen,
                   "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                   "val_macro_f1": val["macro_f1"], "lr": optimizer.param_groups[0]["lr"],
                   "seconds": time.monotonic()-epoch_start}
            history.append(row)
            improved = val["accuracy"] > best
            best = max(best, val["accuracy"])
            stale = 0 if improved else stale+1
            if scheduler:
                scheduler.step()
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict() if scheduler else None,
                     "epoch": epoch, "config": config, "seed": seed, "history": history,
                     "best": best, "stale": stale, "elapsed_seconds": elapsed+time.monotonic()-began,
                     "manifest_sha256": manifest_sha, "torch_rng": torch.get_rng_state(),
                     "python_rng": random.getstate(), "loader_rng": generator.get_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            save_checkpoint(last, state)
            if improved:
                save_checkpoint(output/"best.pt", state)
            write_json(output/"history.json", history)
            print(f"{config['name']} epoch {epoch+1}: val {100*val['accuracy']:.2f}%, {row['seconds']:.1f}s", flush=True)
            if stale >= config["patience"]:
                stop_reason = "early_stopping"
                break
    except TimeoutError:
        # A partial epoch is deliberately discarded; last.pt is the last full epoch.
        stop_reason = "time_budget"
    result = {"name": config["name"], "best_val_accuracy": best if best >= 0 else None,
              "epochs_completed": len(history), "elapsed_seconds": elapsed+time.monotonic()-began,
              "stop_reason": stop_reason, "precision": precision,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    write_json(output/"summary.json", result)
    del training, validation
    return result


@torch.inference_mode()
def benchmark(model, device, deadline=math.inf):
    """Model-only timing: fixed FP32, input resident on device, synchronized."""
    model.eval()
    results = {}
    for batch in (1, 256):
        x = torch.zeros(batch, 3, 28, 28, device=device).contiguous(memory_format=torch.channels_last)
        for _ in range(10):
            check_deadline(deadline)
            model(x)
        sync(device)
        start = time.perf_counter()
        for _ in range(50):
            check_deadline(deadline)
            model(x)
        sync(device)
        seconds = time.perf_counter()-start
        results[str(batch)] = {"batch_latency_ms": seconds/50*1000,
                              "images_per_second": 50*batch/seconds}
    return {"precision": "fp32", "warmup": 10, "iterations": 50,
            "scope": "model only; input on device; excludes preprocessing and transfer", "batches": results}


def environment(device):
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "git_revision": command(["git", "rev-parse", "HEAD"]),
            "git_dirty": command(["git", "status", "--porcelain"]),
            "dependencies": command([__import__("sys").executable, "-m", "pip", "freeze"])}
