import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from gtsrb.data import digest, write_json
from gtsrb.model import Net


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PublicationTests(unittest.TestCase):
    def test_ignored_backups_credentials_and_raw_outputs(self):
        paths = [".venv-old-1791895555/bin/pip", ".venv-python-old/bin/python", ".env",
                 ".env.production", ".ssh/id_ed25519", "client.key", "session.json",
                 "runs/a100/best.pt", "runs/a100/environment.json", "debug.log"]
        for path in paths:
            result = subprocess.run(["git", "check-ignore", "--no-index", path], cwd=ROOT,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, path)
        result = subprocess.run(["git", "check-ignore", "--no-index", "results/a100/mild/weights.pt"], cwd=ROOT)
        self.assertEqual(result.returncode, 1)

    def test_secret_patterns_report_presence_without_values(self):
        audit = module("audit_repository")
        token = ("gh" + "p_" + "x"*36).encode()
        self.assertTrue(audit.PATTERNS["GitHub token"].search(token))
        self.assertTrue(audit.PATTERNS["assigned secret"].search(json.dumps({"session_key": token.decode()}).encode()))
        jwt = ("ey" + "J" + "x"*12 + "." + "y"*12 + "." + "z"*12).encode()
        self.assertTrue(audit.PATTERNS["JWT session token"].search(jwt))
        self.assertTrue(audit.private_path(".venv-old-123/bin/activate"))
        self.assertTrue(audit.private_path(".env.production"))
        self.assertFalse(audit.private_path("configs/a100.json"))

    def test_export_drops_credentials_optimizer_and_raw_environment(self):
        exporter = module("export_results")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, output = root/"run", root/"results"
            recipe = run/"mild"
            recipe.mkdir(parents=True)
            secret = "gh" + "p_" + "x"*36
            torch.save({"model": Net().state_dict(), "optimizer": {"private": secret},
                        "session_key": secret}, recipe/"best.pt")
            checksum = digest(recipe/"best.pt")
            write_json(run/"selection.json", {"chosen": "mild", "manifest_sha256": "a"*64,
                                               "checkpoints": {"mild": checksum}})
            write_json(run/"status.json", {"status": "complete", "smoke": False})
            write_json(run/"environment.json", {"python": "3.12.8", "torch": "2.8.0+cu126",
                "git_revision": "b"*40, "gpu": "NVIDIA A100-SXM4-40GB", "gpu_memory_bytes": 40000000000,
                "dependencies": "https://account:" + secret + "@private.example", "session_key": secret})
            config = dict(name="mild", augmentation="mild", batch_size=256, lr=.001, eps=1e-8,
                          weight_decay=.0001, epochs=80, patience=15, seed=42, cosine=True,
                          precision="bf16", actual_precision="bf16", api_key=secret)
            write_json(recipe/"config.json", config)
            write_json(recipe/"summary.json", {"best_val_accuracy": .97, "epochs_completed": 30,
                "elapsed_seconds": 400, "peak_gpu_memory_bytes": 1000000, "stop_reason": "early_stopping"})
            write_json(recipe/"history.json", [{"epoch": 1, "val_accuracy": .6, "private": secret}])
            write_json(recipe/"test_metrics.json", {"checkpoint_sha256": checksum, "accuracy": .96,
                "macro_f1": .95, "n": 12630, "confusion_matrix": [[0]*43 for _ in range(43)],
                "per_class": [], "private": secret, "benchmark": {"precision": "fp32",
                "warmup": 10, "iterations": 50, "batches": {b: {"batch_latency_ms": .5,
                "images_per_second": 1000} for b in ("1", "256")}}})
            exporter.export(run, output)
            state = torch.load(output/"mild/weights.pt", weights_only=True)
            self.assertEqual(set(state), set(Net().state_dict()))
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in state.values()))
            for path in output.rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes())
            self.assertFalse((output/"environment.json").exists())
            meta = json.loads((output/"export.json").read_text())
            self.assertEqual(meta["versions"]["python"], "3.12.8")
            self.assertEqual(meta["source_revision"], "b"*40)
            with self.assertRaises(FileExistsError):
                exporter.export(run, output)


if __name__ == "__main__":
    unittest.main()
