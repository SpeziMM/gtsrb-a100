# Verification record

Local verification completed on 2026-10-11 using Python 3.12.10, PyTorch 2.8.0, torchvision 0.23.0, NumPy 2.2.6 and Pillow 11.3.0 on macOS ARM64/CPU. This does not establish CUDA, A100 runtime or final model accuracy.

## Code checks

Ten automated tests passed:

- Whole tracks stay in one partition; identical training images join their tracks; class coverage is retained; the split is seeded.
- Deliberate track leakage and exact image leakage are rejected.
- Cache corruption is rejected.
- The unchanged 151,403-parameter model learns a tiny synthetic batch, with finite gradients.
- Accuracy and per-class metrics match a known confusion matrix.
- Evaluation transforms and scores are deterministic.
- Resume restores model/optimizer/scheduler progress and rejects changed configuration.
- An expired training budget cannot create a checkpoint with no validation.
- The process watchdog terminates a deliberately stuck subprocess.
- A synthetic end-to-end comparison saves selection before constructing the test loader, blocks subsequent tuning, and reuses completed metrics when finishing frozen evaluation. Only the device-allocation check is substituted for this CPU test; GPU execution is not simulated or claimed as tested.

A separate integration smoke run used 96 real training images and 96 validation images for each recipe, completed all three paths, and produced its report and checkpoints. The real test set was not evaluated. Smoke scores are not benchmark results.

## Real data checks

The archive SHA256 values are pinned in `gtsrb/data.py`; they were calculated from the authors' ERDA downloads. Test archive MD5 values additionally match torchvision's implementation. These are integrity records, not cryptographic signatures from the dataset authors.

| Partition | Images | Training-source tracks |
| --- | ---: | ---: |
| Train | 31,349 | 1,045 |
| Validation | 7,830 | 261 |
| Official test | 12,630 | Not exposed in shuffled filenames |

All 43 classes occur in every partition. One training track (`class 14, track 00023`, 30 images) was quarantined because it contains an original RGB image exactly matching a test image. No exact pixel duplicates cross the retained partitions. No known training-source track crosses training/validation. The original official training archive has 39,209 images; the retained development pool has 39,179. These removals slightly change comparability with published results, and are disclosed rather than silently changing the test set.

The perceptual difference-hash audit flagged 75 cross-partition candidate groups. A pair from different partitions in each group was visually inspected at cached resolution; many collisions are similar circular signs, sometimes from different classes. Every cross-partition pair within these groups was additionally compared by RGB RMSE; none was within 5/255 at 32×32. This threshold check is descriptive, not a validated proof of independence. The complete candidate list is in `data/audit.json` in the private prepared bundle. A perceptual hash is not exhaustive; claims are limited to documented source-track separation and the duplicate checks performed. Official train/test grouping relies on the authors' dataset provenance because test track identifiers are not provided in the chosen test archive.

No test accuracy was used to choose preprocessing or augmentation. Historical test exposure from coursework before this project remains unknown.

## Checks remaining on the cluster

Run the GPU smoke command inside the allocated A100 session to check the installed CUDA runtime, BF16, loader workers and GPU memory. Then run the default comparison. The generated environment file and measured results are the evidence for A100 compatibility and performance. Package installation and offline Linux wheel preparation depend on the cluster's available Python, operating system and driver; those have not been inspected remotely.
