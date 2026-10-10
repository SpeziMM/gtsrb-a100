import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gtsrb.data import audit, digest, split_tracks, verify, write_json
from gtsrb.engine import Signs, evaluate, loader, metrics, train
from gtsrb.model import Net
from run import supervise


def fixture(root):
    rng = np.random.default_rng(5)
    rows, cache = [], {}
    for split in ("train", "val", "test"):
        x = rng.integers(0, 256, (43, 32, 32, 3), dtype=np.uint8)
        y = np.arange(43, dtype=np.int64)
        for name, value in [(f"{split}_images.npy", x), (f"{split}_labels.npy", y)]:
            np.save(root/name, value, allow_pickle=False)
            cache[name] = digest(root/name)
        rows.extend({"source": split, "path": str(i), "label": i, "split": split,
                     "track": f"{split}:{i}" if split != "test" else None,
                     "pixel_sha256": f"{split}:{i}", "dhash": f"{split}:{i}"} for i in range(43))
    write_json(root/"manifest.json", {"seed": 42, "samples": rows, "cache_sha256": cache})
    return rows


class SplitTests(unittest.TestCase):
    def test_grouping_and_exact_duplicates(self):
        rows = [{"track": f"{label}:{track}", "label": label,
                 "pixel_sha256": f"{label}:{track}:{frame}"}
                for label in range(43) for track in range(6) for frame in range(3)]
        rows[3]["pixel_sha256"] = rows[0]["pixel_sha256"]
        a = split_tracks(rows)
        self.assertEqual(a, split_tracks(rows))
        self.assertNotEqual(a, split_tracks(rows, seed=7))
        self.assertEqual(a[0], a[3])
        for label in range(43):
            self.assertEqual({s for r, s in zip(rows, a) if r["label"] == label}, {"train", "val"})
        for track in {r["track"] for r in rows}:
            self.assertEqual(len({s for r, s in zip(rows, a) if r["track"] == track}), 1)

    def test_audit_rejects_track_and_pixel_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = fixture(Path(tmp))
            audit(rows)
            bad = copy.deepcopy(rows)
            bad[43]["track"] = bad[0]["track"]
            with self.assertRaisesRegex(ValueError, "track"):
                audit(bad)
            bad = copy.deepcopy(rows)
            bad[-1]["pixel_sha256"] = bad[0]["pixel_sha256"]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                audit(bad)

    def test_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            verify(root)
            with (root/"train_images.npy").open("ab") as f:
                f.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(root)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_output_and_small_batch_overfit(self):
        torch.manual_seed(7)
        model = Net()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 151403)
        x = torch.rand(4, 3, 28, 28)
        y = torch.tensor([0, 1, 2, 3])
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        initial = torch.nn.functional.cross_entropy(model(x), y).item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertLess(torch.nn.functional.cross_entropy(model(x), y).item(), initial*.2)

    def test_metrics_known_confusion(self):
        cm = np.eye(43, dtype=int)*2
        cm[0, 0] -= 1
        cm[0, 1] += 1
        result = metrics(cm)
        self.assertAlmostEqual(result["accuracy"], 85/86)
        self.assertAlmostEqual(result["per_class"][0]["recall"], .5)
        self.assertAlmostEqual(result["per_class"][1]["precision"], 2/3)

    def test_evaluation_is_deterministic_and_never_augmented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            data = Signs(root, "val")
            self.assertTrue(torch.equal(data[0][0], data[0][0]))
            model = Net()
            batches = loader(data, 16, 0, torch.device("cpu"))
            a = evaluate(model, batches, torch.device("cpu"))
            b = evaluate(model, batches, torch.device("cpu"))
            self.assertEqual(a, b)

    def test_resume_restores_completed_epoch_and_rejects_changes(self):
        config = dict(name="mild", augmentation="mild", batch_size=16, lr=.001,
                      eps=1e-8, weight_decay=.0001, epochs=2, patience=5, cosine=True, precision="fp32")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            from gtsrb import engine
            actual_save = engine.save_checkpoint

            def interrupt_after_best(path, state):
                actual_save(path, state)
                if Path(path).name == "best.pt":
                    raise TimeoutError("simulate interruption after first epoch")

            with patch.object(engine, "save_checkpoint", side_effect=interrupt_after_best):
                train(config, root, root/"run", torch.device("cpu"), 0, 42, time.monotonic()+60)
            first = torch.load(root/"run/last.pt", weights_only=True)
            self.assertEqual(first["epoch"], 0)
            result = train(config, root, root/"run", torch.device("cpu"), 0, 42, time.monotonic()+60, resume=True)
            self.assertEqual(result["epochs_completed"], 2)
            final = torch.load(root/"run/last.pt", weights_only=True)
            self.assertEqual(final["optimizer"]["state"][0]["step"].item(), 6)
            with self.assertRaisesRegex(ValueError, "changed"):
                train({**config, "lr": .1}, root, root/"run", torch.device("cpu"), 0, 42, time.monotonic()+60, resume=True)

    def test_expired_budget_produces_no_unvalidated_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            config = dict(name="mild", augmentation="mild", batch_size=16, lr=.001,
                          eps=1e-8, weight_decay=0, epochs=2, patience=5, cosine=False, precision="fp32")
            result = train(config, root, root/"run", torch.device("cpu"), 0, 42, time.monotonic()-1)
            self.assertIsNone(result["best_val_accuracy"])
            self.assertFalse((root/"run/best.pt").exists())


class WatchdogTests(unittest.TestCase):
    def test_stops_a_hung_process(self):
        start = time.monotonic()
        code = supervise([sys.executable, "-c", "import time; time.sleep(30)"], seconds=.2)
        self.assertEqual(code, 124)
        self.assertLess(time.monotonic()-start, 5)


class EndToEndTests(unittest.TestCase):
    def test_selection_precedes_test_and_frozen_evaluation_can_resume(self):
        from gtsrb import run
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            config = json.loads(Path("configs/a100.json").read_text())
            for recipe in config["experiments"]:
                recipe.update(epochs=1, batch_size=16)
            write_json(root/"config.json", config)
            output = root/"results"
            argv = ["run", "--config", str(root/"config.json"), "--data", str(root),
                    "--output", str(output), "--workers", "0"]
            actual_signs = run.Signs

            def checked_signs(data, split, *args, **kwargs):
                if split == "test":
                    self.assertTrue((output/"selection.json").exists())
                    self.assertTrue((root/"test_access.json").exists())
                return actual_signs(data, split, *args, **kwargs)

            # Only the device-allocation check is substituted. No claim of CUDA testing.
            with patch.object(run, "choose_device", return_value=torch.device("cpu")), patch.object(run, "Signs", side_effect=checked_signs):
                with patch.object(sys, "argv", argv):
                    run.main()
                selection = json.loads((output/"selection.json").read_text())
                self.assertIn(selection["chosen"], ("assignment", "mild", "moderate"))
                before = digest(output/"assignment/test_metrics.json")
                with patch.object(sys, "argv", argv+["--resume"]):
                    with self.assertRaises(SystemExit) as err:
                        run.main()
                    self.assertEqual(err.exception.code, 2)
                with patch.object(sys, "argv", argv+["--evaluate-only"]), patch.object(run, "train", side_effect=AssertionError("must not retrain")), patch.object(run, "evaluate", side_effect=AssertionError("must reuse completed metrics")):
                    run.main()
                self.assertEqual(before, digest(output/"assignment/test_metrics.json"))
                self.assertEqual(selection, json.loads((output/"selection.json").read_text()))


if __name__ == "__main__":
    unittest.main()
