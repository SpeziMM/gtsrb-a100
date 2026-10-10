import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import digest, verify, write_json
from .engine import (Signs, benchmark, check_deadline, environment, evaluate,
                     loader, seed_all, train)
from .model import Net


def choose_device(requested, smoke):
    if requested == "cpu" and not smoke:
        raise ValueError("Full runs require the requested A100. CPU is supported for --smoke only")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable: activate CUDA-enabled PyTorch inside your GPU allocation")
        if not smoke and ("A100" not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory < 35*1024**3):
            raise ValueError("Full run requires an allocated A100 with at least 40 GB nominal memory")
    return torch.device(requested)


def curves(output, summaries):
    colors = ["#2463eb", "#16804a", "#d3690e"]
    paths, labels = [], []
    histories = [json.loads((output/s["name"]/"history.json").read_text()) for s in summaries if (output/s["name"]/"history.json").exists()]
    max_epoch = max((len(h) for h in histories), default=1)
    for idx, summary in enumerate(summaries):
        path = output/summary["name"]/"history.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        color = colors[idx % len(colors)]
        points = " ".join(f"{60+(r['epoch']-1)/max(1,max_epoch-1)*650:.1f},{330-r['val_accuracy']*280:.1f}" for r in history)
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        labels.append(f'<text x="{65+220*idx}" y="385" fill="{color}">{summary["name"]}</text>')
    (output/"validation.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="780" height="410" viewBox="0 0 780 410"><rect width="780" height="410" fill="white"/><g font-family="sans-serif" font-size="14"><text x="60" y="25">Validation accuracy by completed epoch</text><path d="M60 45V330H725" fill="none" stroke="#666"/><text x="20" y="60">100%</text><text x="30" y="330">0%</text><text x="60" y="350">1</text><text x="680" y="350">'+str(max_epoch)+'</text>'+''.join(paths+labels)+'</g></svg>')


def report(output, summaries, results, chosen, smoke):
    lines = ["# GTSRB run report", "", "SMOKE TEST ONLY — these scores are not a benchmark." if smoke else "Single-seed, time-bounded comparison; no statistical superiority claim.", "",
             "| Recipe | Validation accuracy | Test accuracy | Test macro-F1 | Epochs | Training + validation seconds | Stop reason |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for s in summaries:
        r = results.get(s["name"], {})
        fmt = lambda x: "—" if x is None else f"{100*x:.2f}%"
        lines.append(f"| {s['name']} | {fmt(s['best_val_accuracy'])} | {fmt(r.get('accuracy'))} | {fmt(r.get('macro_f1'))} | {s['epochs_completed']} | {s['elapsed_seconds']:.1f} | {s['stop_reason']} |")
    lines += ["", f"Selected using validation only: **{chosen or 'none'}**.", "",
              "The baseline preserves the assignment architecture, optimizer settings and augmentation recipe. Its duration/checkpoint selection follows this project's validation protocol. Mild versus moderate isolates the augmentation recipe; baseline versus candidates also changes optimizer settings, batch size and precision.", "",
              "![Validation curves](validation.svg)", "",
              "Published external context: TrafficSignNet reports 96.82% on the official GTSRB test set with 99,019 parameters. This is an author-reported result, not reproduced here. Different preprocessing and training protocols prevent a controlled ranking; CPU timing is not comparable to A100 timing. See [reference details](../../docs/research.md).", "",
              "Historical test exposure before this project is unknown. This run selects only on validation. The saved manifest records the official train/test sources and grouped validation split.", "",
              "Raw evidence: `environment.json`, `selection.json`, per-recipe `config.json`, `history.json`, `summary.json`, and `test_metrics.json` when final evaluation completed. Model-only FP32 timings use a common batch size and exclude decoding/transfers. All actual training precisions and peak GPU memory measurements are recorded separately.", ""]
    (output/"report.md").write_text("\n".join(lines))
    curves(output, summaries)


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description="Fixed CNN, bounded tuning, then locked final evaluation")
    p.add_argument("--config", type=Path, default=Path("configs/a100.json"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("runs/a100"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--workers", type=int)
    p.add_argument("--smoke", action="store_true", help="tiny train/validation check; never evaluates test")
    p.add_argument("--train-only", action="store_true", help="keep final test sealed")
    p.add_argument("--recipe", choices=("assignment", "mild", "moderate"))
    p.add_argument("--resume", action="store_true", help="resume last completed epoch before final test access")
    p.add_argument("--evaluate-only", action="store_true", help="finish an already frozen final evaluation; never retrain")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if not 0 < cfg["max_minutes"] <= 115:
        p.error("max_minutes must be in (0, 115]; launcher reserves a margin before two hours")
    if not 0 < cfg["trial_seconds"] <= 6600:
        p.error("trial_seconds must be in (0, 6600]")
    hard_remaining = float(os.environ.get("GTSRB_HARD_DEADLINE", time.time()+7140))-time.time()
    deadline = started + min(cfg["max_minutes"]*60, hard_remaining-30)
    try:
        device = choose_device(args.device, args.smoke)
    except ValueError as exc:
        p.error(str(exc))
    workers = args.workers if args.workers is not None else cfg["workers"]
    if workers < 0:
        p.error("workers cannot be negative")
    if args.smoke:
        workers = 0
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    # Use exact FP32 for the reference training/timing path, not implicit TF32.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    output = args.output
    marker = args.data/"test_access.json"
    if args.evaluate_only and (args.smoke or args.train_only or args.recipe):
        p.error("--evaluate-only cannot be combined with --smoke, --train-only, or --recipe")
    if marker.exists() and not args.smoke and not args.evaluate_only:
        p.error("Final test has already been accessed for this prepared dataset; inspect test_access.json. Do not tune further against that result")
    if output.exists() and any(output.iterdir()) and not (args.resume or args.evaluate_only):
        p.error("Output directory is nonempty; choose a new --output or --resume")
    if (output/"selection.json").exists() and not (args.smoke or args.evaluate_only):
        p.error("This run has frozen its selection; choose a new output for validation-only work before test access")
    output.mkdir(parents=True, exist_ok=True)
    print("Verifying data integrity (no test predictions)", flush=True)
    verify(args.data)
    manifest_sha = digest(args.data/"manifest.json")
    write_json(output/("evaluation_environment.json" if args.evaluate_only else "environment.json"), environment(device))
    if not args.evaluate_only:
        write_json(output/"run_config.json", {"config": cfg, "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                           "manifest_sha256": manifest_sha, "historical_test_exposure": "unknown"})
    experiments = [dict(c) for c in cfg["experiments"] if not args.recipe or c["name"] == args.recipe]
    names = [c["name"] for c in experiments]
    if not names or len(names) != len(set(names)) or any(n not in ("assignment", "mild", "moderate") for n in names):
        p.error("Use unique assignment/mild/moderate experiment names")
    summaries, results = [], {}
    chosen = None
    if args.evaluate_only:
        selection = json.loads((output/"selection.json").read_text())
        if selection["manifest_sha256"] != manifest_sha:
            p.error("Frozen selection uses a different data manifest")
        summaries, chosen = selection["summaries"], selection["chosen"]
        experiments = []
    for config in experiments:
        check_deadline(deadline)
        if args.smoke:
            config.update(epochs=1, batch_size=16)
        remaining_trials = len(experiments)-len(summaries)
        allowance = min(cfg["trial_seconds"], (deadline-time.monotonic()-300)/remaining_trials)
        if allowance <= 0:
            break
        print(f"Starting {config['name']}: at most {allowance/60:.1f} minutes", flush=True)
        summary = train(config, args.data, output/config["name"], device, workers, cfg["seed"],
                        time.monotonic()+allowance, args.resume, 96 if args.smoke else None)
        summaries.append(summary)
        report(output, summaries, results, chosen, args.smoke)
    valid = [s for s in summaries if s["best_val_accuracy"] is not None]
    if not valid:
        raise RuntimeError("No completed training+validation epoch; see logs and increase trial budget or lower workers")
    if not args.evaluate_only:
        chosen = max(valid, key=lambda s: s["best_val_accuracy"])["name"]
        selection = {"chosen": chosen, "criterion": "highest validation accuracy; ties retain earlier recipe",
                 "manifest_sha256": manifest_sha, "summaries": summaries,
                 "checkpoints": {s["name"]: digest(output/s["name"]/"best.pt") for s in valid}}
    # No test loader is constructed until after this selection has been written.
    if not args.smoke and not args.train_only:
        if not args.evaluate_only:
            write_json(output/"selection.json", selection)
        if marker.exists():
            if json.loads(marker.read_text())["selection"] != selection:
                p.error("Existing test access belongs to another frozen comparison")
        else:
            with marker.open("x") as f:
                json.dump({"output": str(output.resolve()), "selection": selection,
                           "note": "Test access began after freezing selection; do not retune from these scores."}, f, indent=2)
        test_loader = loader(Signs(args.data, "test"), 512, workers, device)
        evaluation_names = list(dict.fromkeys((["assignment"] if "assignment" in selection["checkpoints"] else []) + [chosen]))
        for name in evaluation_names:
            check_deadline(deadline)
            path = output/name/"best.pt"
            if digest(path) != selection["checkpoints"][name]:
                raise ValueError("Frozen checkpoint changed")
            saved_metrics = output/name/"test_metrics.json"
            if saved_metrics.exists():
                result = json.loads(saved_metrics.read_text())
                if result["checkpoint_sha256"] != selection["checkpoints"][name]:
                    raise ValueError("Existing metrics belong to another checkpoint")
                results[name] = result
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = Net().to(device, memory_format=torch.channels_last)
            model.load_state_dict(checkpoint["model"])
            # All final accuracies and inference timings use the same FP32 path.
            result = evaluate(model, test_loader, device, "fp32", deadline)
            result["benchmark"] = benchmark(model, device, deadline)
            result["checkpoint_sha256"] = selection["checkpoints"][name]
            result["parameters"] = sum(p.numel() for p in model.parameters())
            write_json(output/name/"test_metrics.json", result)
            with (output/name/"per_class.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result["per_class"][0].keys())
                writer.writeheader()
                writer.writerows(result["per_class"])
            np.savetxt(output/name/"confusion_matrix.csv", np.array(result["confusion_matrix"]), fmt="%d", delimiter=",")
            results[name] = result
            report(output, summaries, results, chosen, args.smoke)
    else:
        write_json(output/"validation_selection.json", selection)
    report(output, summaries, results, chosen, args.smoke)
    write_json(output/"status.json", {"status": "complete", "smoke": args.smoke,
                                     "test_evaluated": list(results), "total_seconds": time.monotonic()-started})
    print(f"Finished. Report: {output/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
