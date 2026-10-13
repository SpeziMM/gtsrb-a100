"""Export selected weights and allowlisted benchmark fields for publication."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtsrb.data import digest, write_json
from gtsrb.model import Net


RECIPES = ("assignment", "mild", "moderate")
NUMERIC_CONFIG = ("batch_size", "lr", "eps", "weight_decay", "epochs", "patience", "seed")
NUMERIC_SUMMARY = ("best_val_accuracy", "epochs_completed", "elapsed_seconds", "peak_gpu_memory_bytes")
NUMERIC_HISTORY = ("epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr", "seconds")
NUMERIC_METRICS = ("n", "accuracy", "macro_f1", "balanced_accuracy", "loss", "parameters")


def numbers(source, keys):
    result = {}
    for key in keys:
        value = source.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Invalid numeric field: " + key)
        if key in source:
            result[key] = value
    return result


def enum(source, key, allowed):
    value = source[key]
    if value not in allowed:
        raise ValueError("Unexpected " + key)
    return value


def export(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError("Export directory already exists; choose another output name")
    selection = json.loads((run/"selection.json").read_text())
    status = json.loads((run/"status.json").read_text())
    if status.get("status") != "complete" or status.get("smoke"):
        raise ValueError("Finish the full benchmark before exporting")
    chosen = enum(selection, "chosen", RECIPES)
    manifest = selection["manifest_sha256"]
    if not re.fullmatch("[0-9a-f]{64}", manifest):
        raise ValueError("Invalid manifest checksum")
    names = list(dict.fromkeys((["assignment"] if "assignment" in selection["checkpoints"] else []) + [chosen]))
    prepared = []
    expected = Net().state_dict()
    for name in names:
        path = run/name
        checksum = selection["checkpoints"][name]
        if digest(path/"best.pt") != checksum:
            raise ValueError("Checkpoint changed since selection: " + name)
        saved = torch.load(path/"best.pt", map_location="cpu", weights_only=True)
        state = saved["model"]
        if set(state) != set(expected) or any(not isinstance(state[k], torch.Tensor) or state[k].shape != expected[k].shape or not torch.isfinite(state[k]).all() for k in expected):
            raise ValueError("Unexpected model weights")
        config = json.loads((path/"config.json").read_text())
        safe_config = numbers(config, NUMERIC_CONFIG)
        safe_config.update(augmentation=enum(config, "augmentation", ("assignment", "mild", "moderate")),
                           precision=enum(config, "precision", ("fp32", "bf16")),
                           actual_precision=enum(config, "actual_precision", ("fp32", "bf16")))
        for key in ("cosine",):
            if not isinstance(config[key], bool):
                raise ValueError("Invalid boolean config")
            safe_config[key] = config[key]
        summary = json.loads((path/"summary.json").read_text())
        safe_summary = numbers(summary, NUMERIC_SUMMARY)
        safe_summary["stop_reason"] = enum(summary, "stop_reason", ("epoch_limit", "early_stopping", "time_budget"))
        metrics = json.loads((path/"test_metrics.json").read_text())
        if metrics["checkpoint_sha256"] != checksum:
            raise ValueError("Metrics use another checkpoint")
        safe_metrics = numbers(metrics, NUMERIC_METRICS)
        cm = metrics["confusion_matrix"]
        if len(cm) != 43 or any(len(row) != 43 or any(type(v) is not int or v < 0 for v in row) for row in cm):
            raise ValueError("Invalid confusion matrix")
        safe_metrics["confusion_matrix"] = cm
        safe_metrics["per_class"] = [numbers(row, ("label", "support", "precision", "recall", "f1")) for row in metrics["per_class"]]
        bench = metrics["benchmark"]
        safe_metrics["benchmark"] = {"precision": enum(bench, "precision", ("fp32",)),
            **numbers(bench, ("warmup", "iterations")),
            "scope": "model only; input on device; excludes preprocessing and transfer",
            "batches": {batch: numbers(bench["batches"][batch], ("batch_latency_ms", "images_per_second")) for batch in ("1", "256")}}
        history = [numbers(row, NUMERIC_HISTORY) for row in json.loads((path/"history.json").read_text())]
        prepared.append((name, state, safe_config, safe_summary, safe_metrics, history))
    # Do not export environment.json, pip output, raw checkpoints, absolute paths,
    # arbitrary added fields, credentials, or session state from the run directory.
    output.mkdir(parents=True)
    export_meta = {"selected_recipe": chosen, "manifest_sha256": manifest,
                   "architecture": "Assignment 5 Net", "parameters": 151403,
                   "evaluation_precision": "fp32", "single_seed": True, "recipes": names}
    environment_path = run/"environment.json"
    if environment_path.exists():
        env = json.loads(environment_path.read_text())
        versions = {}
        for key in ("python", "torch", "torchvision", "numpy", "cuda_runtime"):
            value = env.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9][0-9.]{0,15}(?:\+[a-z0-9.]{1,12})?", value):
                versions[key] = value
        export_meta["versions"] = versions
        revision = env.get("git_revision", "")
        if re.fullmatch("[0-9a-f]{40}", revision):
            export_meta["source_revision"] = revision
        dirty = env.get("git_dirty")
        export_meta["source_was_dirty"] = None if dirty is None or dirty == "unavailable" else bool(dirty)
        export_meta["hardware"] = numbers(env, ("gpu_memory_bytes",))
        if isinstance(env.get("gpu"), str) and "A100" in env["gpu"]:
            export_meta["hardware"]["gpu"] = "NVIDIA A100"
    lines = ["# A100 benchmark", "", "Single-seed results. Selected using validation accuracy.", "",
             "| Recipe | Validation accuracy | Test accuracy | Macro-F1 | Training + validation seconds |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, state, config, summary, metrics, history in prepared:
        dest = output/name
        dest.mkdir()
        torch.save({k: v.detach().cpu() for k, v in state.items()}, dest/"weights.pt")
        for filename, value in (("config.json", config), ("summary.json", summary),
                                ("metrics.json", metrics), ("history.json", history)):
            write_json(dest/filename, value)
        lines.append("| {} | {:.2%} | {:.2%} | {:.4f} | {:.1f} |".format(
            name, summary["best_val_accuracy"], metrics["accuracy"], metrics["macro_f1"], summary["elapsed_seconds"]))
    export_meta["sha256"] = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"export.json", export_meta)
    (output/"README.md").write_text("\n".join(lines) + "\n\nWeights contain only the model state dictionary. Load with `torch.load(path, weights_only=True)` and `Net().load_state_dict(...)`.\n")
    print("Exported public artifacts to " + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/a100"))
    parser.add_argument("--output", type=Path, default=Path("results/a100"))
    args = parser.parse_args()
    export(args.run, args.output)
