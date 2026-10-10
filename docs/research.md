# Research and comparison protocol

## Augmentation decision

Geometric perturbations are established in traffic-sign CNN work: see Sermanet and LeCun, [Traffic Sign Recognition with Multi-Scale Convolutional Networks](https://yann.lecun.org/exdb/publis/pdf/sermanet-ijcnn-11.pdf), IJCNN 2011. That system differs from this assignment model and competition-phase results are not a directly comparable final-test benchmark.

The original small CNN has limited input resolution. We therefore propose modest affine variation and brightness/contrast changes, preserving hue and sign handedness. A horizontal mirror can change directional meaning or reverse digits; the new recipes do not mirror. Existing class-dependent flips remain only in the assignment baseline. Strong blur, large rotations, large crops, MixUp, CutMix, and learned augmentation searches are omitted to keep the two-hour experiment focused. This is a design choice, not evidence that these methods can never help.

The two exact ranges in the config are hypotheses to compare on validation. No source proves they are optimal for this architecture. The winner may be the original recipe. Brightness/contrast magnitudes describe multiplicative ranges; e.g. 0.2 samples factors from 0.8 to 1.2. Affine transforms are followed by the documented 28×28 crop.

PyTorch references: [RandomAffine](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html), [ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html), and [published version pairs](https://pytorch.org/get-started/previous-versions/). BF16 is used only for candidate training on supported CUDA hardware. Validation, test evaluation and model-only timing consistently use FP32.

## Open-source reference

Reference: [tanmayai23/gtsrb-traffic-sign-recognition](https://github.com/tanmayai23/gtsrb-traffic-sign-recognition/tree/965591fae02372a0b2d856685da4f813cfadb08a), revision `965591fae02372a0b2d856685da4f813cfadb08a`, MIT-licensed. Its [saved metrics](https://github.com/tanmayai23/gtsrb-traffic-sign-recognition/blob/965591fae02372a0b2d856685da4f813cfadb08a/results/metrics.json) report accuracy **0.968171**, macro-F1 **0.956721**, and 12,630 test images. The [model](https://github.com/tanmayai23/gtsrb-traffic-sign-recognition/blob/965591fae02372a0b2d856685da4f813cfadb08a/src/model.py) has three small convolution blocks and 99,019 parameters, versus our two-convolution 151,403-parameter model. It uses 32×32 inputs, ROI/CLAHE preprocessing and class balancing; our protocol differs. Its CPU timing is not comparable to A100 timing. These are author-reported numbers, not independently reproduced here. Its test-set ablations also warrant caution about selection history. Use it as compact-CNN context, not a controlled ranking or state-of-the-art claim.

Reproducing another architecture is outside this small fixed-model run. The controlled benchmark is assignment recipe versus the validation-selected recipe on identical retained data, with common FP32 test evaluation and inference timing. It uses a single seed because the user's priority is a simple runnable experiment under two hours.

## Evaluation methodology and reviewed skill

Applied the [K-Dense scikit-learn skill](https://github.com/K-Dense-AI/claude-scientific-skills/blob/92ace75ac21efe19a620434e0ca4e356081fe807/skills/scikit-learn/SKILL.md) and its model-evaluation guidance at revision `92ace75ac21efe19a620434e0ca4e356081fe807`. Relevant instructions were checked against primary documentation: [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) and [avoiding leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Independent sign tracks define groups, preprocessing does not fit on held-out images, and validation drives model selection. Accuracy is primary, while macro-F1 and per-class recall expose class imbalance. No sklearn model or external skill script is executed; the skill informs the evaluation protocol. It was reviewed for this use, not certified as universally correct.

Skill citation: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). The current arXiv record was checked when writing this note.

Data provenance: Stallkamp, Schlipsing, Salmen and Igel, [official GTSRB archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html); *Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition*, Neural Networks 32 (2012), 323–332. The archive distinguishes full final training from earlier competition data. Actual source URLs, checksums and sample paths are saved in the prepared manifest.
