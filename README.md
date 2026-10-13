# GTSRB on an A100

Run the Assignment 5 CNN on one **A100 40 GB**, with accuracy as the priority and a **two-hour maximum per invocation**. The network stays unchanged: two 64-channel convolutions, ELU, max pooling, a 43-class linear head, and 28×28 inputs (151,403 parameters).

The default run compares the assignment recipe with two modest augmentation/training recipes, chooses using validation accuracy, and tests only the original baseline and the selected recipe. There is no architecture search or pretraining.

## Quick start

Prepare the environment and data **before reserving GPU time**, on a node where installation/downloads are allowed. Use Python 3.12 on Linux x86_64 (supported Python versions: 3.10–3.13; the pinned PyTorch wheels require glibc 2.28+). The setup script checks the version before creating an environment and searches for a supported installed interpreter.

```bash
git clone https://github.com/SpeziMM/gtsrb-a100.git
cd gtsrb-a100
bash scripts/setup.sh
source .venv/bin/activate
.venv/bin/python -m gtsrb.data --download
```

Inside your course's allocated A100 session:

```bash
source .venv/bin/activate
.venv/bin/python run.py
```

The output is **`runs/a100/report.md`**, with validation curves, checkpoints, raw metrics, confusion matrices, per-class scores and environment details. The launcher uses the GPU exposed as CUDA device 0; follow the course's allocation rules or set `CUDA_VISIBLE_DEVICES` to the device assigned to you. It does not reserve a GPU or assume a scheduler/partition name.

**No A100 accuracy or speed result is claimed yet.** Local CPU verification exercises the pipeline; your cluster run produces the actual benchmark.

### If the cluster uses Python 3.6

An error pointing at `while chunk := ...` means the command ran with Python older than 3.8. The project's pinned dependencies need Python 3.10–3.13, so changing that line alone will not make a Python 3.6 environment usable. Updating pip also cannot change the Python interpreter inside an existing virtual environment.

Check available versions on the course machine:

```bash
python3 --version
module avail python
```

Load a listed Python 3.10, 3.11, 3.12 or 3.13 module using its exact displayed name. Module names vary; the course guide does not establish a specific modern Python module. If `python3.12` is already installed, select it directly:

```bash
PYTHON_BIN=python3.12 bash scripts/setup.sh --check
PYTHON_BIN=python3.12 bash scripts/setup.sh
.venv/bin/python -m gtsrb.data --download
```

If the previous `.venv/bin/python --version` reports Python 3.6, rename that environment first with `mv .venv .venv-python-old` (choose another backup name if it exists), then rerun setup. This leaves the prepared `data/` directory in place.

Where Conda is available, another option is:

```bash
conda create -n gtsrb-python python=3.12 pip -y
conda activate gtsrb-python
bash scripts/setup.sh
.venv/bin/python -m gtsrb.data --download
```

If there is no modern Python module, installed interpreter or Conda, collect `module avail python`, `command -v conda`, and `getconf GNU_LIBC_VERSION` to choose a compatible installation method. No admin access is needed for the module/Conda options above. Dependencies and Python must be provisioned before an offline run.

## If the cluster cannot download data

Data preparation works on a laptop. The prepared bundle contains 32×32 RGB caches and source/split manifests; training then requires no network.

```bash
# On a connected machine, after setup:
python -m gtsrb.data --download
python scripts/bundle_data.py
```

Copy `dist/gtsrb-prepared.tar.gz` and its `.sha256` file to the cluster with your normal file-transfer method. From the repository directory there:

```bash
sha256sum -c gtsrb-prepared.tar.gz.sha256
tar -xzf gtsrb-prepared.tar.gz
python -m gtsrb.data --verify
python run.py
```

The public Git repository contains code and metadata, **not dataset images or checkpoints**. The authors' archive establishes the download source but does not clearly grant redistribution on the page reviewed, so this project does not rehost the dataset publicly. A Git pull alone cannot supply an offline Python environment or dataset. A prepared data bundle is also available locally to the project owner from development.

If package downloads are blocked too, prepare wheels on an internet-connected **Linux x86_64 machine with the same Python version**:

```bash
bash scripts/download_wheels.sh
```

Transfer `wheelhouse/` to the cluster, verify it, and install offline:

```bash
(cd wheelhouse && sha256sum -c SHA256SUMS)
WHEELHOUSE="$PWD/wheelhouse" bash scripts/setup.sh
```

CUDA wheels can occupy several GB. Mac wheels cannot be reused on Linux. An existing course PyTorch environment can also be used if it provides the compatible packages; record its versions with the generated run evidence. The NVIDIA driver must support the selected CUDA runtime; installing a CUDA toolkit alone does not replace the driver.

## What is compared

| Recipe | Training augmentation | Optimizer / schedule | Batch / precision |
| --- | --- | --- | --- |
| `assignment` | Original affine ±5°, random 28 crop, brightness ±80%, contrast ±40%, occasional blur, existing class-specific flips | Adam, lr 0.001, epsilon 0.1, no weight decay/schedule | 128 / FP32 |
| `mild` | Affine ±8°, shift ±5%, scale 0.95–1.05, center crop, brightness/contrast ±20%, no flips | Adam, lr 0.001, epsilon 1e-8, weight decay 1e-4, cosine schedule | 256 / BF16 |
| `moderate` | Affine ±12°, shift ±8%, scale 0.9–1.1, center crop, brightness ±30%, contrast ±25%, no flips | Same as `mild` | 256 / BF16 |

The candidate ranges are research-informed choices, not previously proven optima for this model. Candidate recipes omit blur and mirroring to preserve small digits and handedness. [Research notes](docs/research.md) explain the choices and external reference.

`mild` versus `moderate` isolates the augmentation recipe. Comparing either with `assignment` measures a combined recipe change; it cannot attribute improvement to augmentation alone. Every recipe uses the same images and fixed validation split. The original baseline is capped at 200 epochs; candidates at 80. Validation-based early stopping, best-checkpoint selection and the time cap apply to all runs. This is a practical budgeted comparison, not an exact replay of the notebook's 200-epoch output.

All final test metrics and model-only inference benchmarks use **FP32**, with identical input sizes and timing batches. Training throughput is affected by the CPU loader and should be interpreted with recorded worker counts, precision, completed epochs and elapsed time. Since the architecture is fixed, large inference-speed differences are not expected.

## Time limit

- Each trial gets at most 2,000 seconds (33⅓ minutes), and less if necessary to leave time for final evaluation.
- The normal pipeline deadline is 115 minutes, including verification, training, validation, final test and reporting.
- An independent parent process stops the whole process group at 119 minutes; forced termination follows within five seconds if needed. A stuck loader therefore cannot silently run for hours.
- A partial epoch is discarded. Checkpoints retain the last completed training/validation epoch.
- Dataset downloads, environment installation and preparation are separate setup tasks. A two-hour GPU reservation should be made after setup.

The default uses one fixed seed per recipe to keep the scope small. It reports measured results without a claim of statistical superiority. No minimum accuracy or guaranteed improvement is promised.

## Verification and other runs

```bash
# No dataset needed: split isolation, metrics, tiny-batch learning,
# checkpoint recovery, time budget and watchdog checks.
python -m unittest discover -s tests -v

# CPU integration check on a small train/validation sample; never tests.
python run.py --smoke --device cpu --output runs/smoke

# GPU smoke check inside the allocation, also without test access.
python run.py --smoke --output runs/smoke-gpu

# One recipe or a custom config, validation only:
python run.py --recipe mild --train-only --output runs/mild
python run.py --config configs/a100.json --train-only --output runs/exploration

# Resume a training run interrupted before final selection/test access:
python run.py --resume

# Finish interrupted evaluation of exactly the already-frozen checkpoints:
python run.py --evaluate-only
```

Edit the small JSON config to change training settings or use `--workers 2` if the course allocates fewer CPU cores. Every run writes the resolved settings. Resuming requires the same settings, seed and data manifest. Worker augmentation streams are reseeded after restart, so recovery is valid but not guaranteed bit-for-bit identical to an uninterrupted multiworker run. Resuming starts a new time budget; the two-hour cap is per invocation.

After final test access, the data directory records `test_access.json`. The launcher refuses further training against that supposedly untouched test set. `--evaluate-only` may complete a frozen comparison without changing its checkpoints. Do exploratory runs with `--train-only` first. Earlier exposure of the assignment test results is unknown and is recorded as a limitation; splitting again cannot undo previous test-driven decisions.

## Data protocol

Use the authors' **full final training archive (39,209 images)** and official test set (12,630). Torchvision's older `GTSRB-Training_fixed.zip` has 26,640 training images; it is deliberately not used here.

Preparation verifies archive checksums, groups training frames by `(class, track)`, joins tracks sharing an exact duplicate image, and makes a seeded approximately 80/20 split within each class. If a training image is pixel-identical to a test image, its entire training track is quarantined. The test set stays unchanged.

The verified dataset has **31,349 training, 7,830 validation and 12,630 test images**, with **30 training images from one track excluded** due to exact train/test overlap. All 43 classes are present in each split; retained partitions have no exact pixel overlap and no train/validation track overlap. `data/audit.json` also records perceptual-hash candidates. These are review hints, not proof that every possible near duplicate has been found. [Verification notes](docs/verification.md) state the checks and limits.

Resizing is fixed and does not learn from data. Pixel conversion is `/255`; no normalization statistics are fitted on validation/test. Validation/test use deterministic center crops. No test predictions are computed during preparation, smoke tests, tuning or checkpoint selection. Only the frozen baseline and validation winner are evaluated, and selection is saved before the test loader is created.

## Files

`configs/a100.json` controls the run. `gtsrb/model.py` defines the fixed model; `data.py` prepares/verifies data; `engine.py` trains and evaluates; `run.py` coordinates the experiment. `docs/research.md` records sources and the reviewed data-science skill. Raw data, environments and outputs are ignored by Git.

## Recover an accidental local environment commit

If your last **unpushed** commit includes `.venv-old-*`, undo that commit while preserving its files, then bring in the corrected ignore rules:

```bash
git reset --mixed HEAD~1
git pull --rebase --autostash origin codex/a100
git config user.name "SpeziMM"
git config user.email "42320336+SpeziMM@users.noreply.github.com"
.venv/bin/python scripts/audit_repository.py --history
git status --short
```

`reset --mixed` preserves the working files, including datasets and completed runs. Pulling with `--autostash` temporarily saves and reapplies tracked edits. If Git reports a conflict, resolve it before proceeding. These commands assume the unwanted commit is the last local commit and has not been pushed; they do not rewrite published history. Check with `git log --oneline origin/codex/a100..HEAD` before undoing a commit if unsure.

`.gitignore` now covers all `.venv*` directories, common credential/key/session files, raw run folders, logs and full checkpoints. Ignore rules do not remove files already present in commits or in the index; undoing the accidental commit is the required cleanup step.

## Publish model weights and benchmark results

After a complete benchmark, generate a new export directory. The exporter includes only the baseline/selected model state dictionaries, numeric metrics, histories, training settings, version metadata, source revision and checksums. It excludes optimizer state, arbitrary extra fields, raw environment logs, dependency URLs, session state and absolute run paths.

```bash
.venv/bin/python scripts/export_results.py --run runs/a100 --output results/a100
git add -- results/a100
.venv/bin/python scripts/audit_repository.py --history
git diff --cached --stat
git commit --only -m "Add A100 model and benchmark results" -- results/a100
git pull --rebase --autostash origin codex/a100
git push origin HEAD:codex/a100
```

Use a different export folder for another completed run; the exporter refuses to overwrite an existing directory. The raw `runs/a100/` folder remains available locally for checkpoint recovery. Public `weights.pt` files contain only tensors matching the fixed `Net` architecture. To load one, use `Net().load_state_dict(torch.load(path, map_location="cpu", weights_only=True))`.

The audit inspects staged files and, with `--history`, current-branch ancestry for environment/private paths and common secret patterns. It reports only filenames and rule names, never matching values. It is a targeted check, not proof that every possible secret format is absent. Avoid `git add .` for publishing results; stage the generated export directory explicitly.
