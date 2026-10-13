# A100 benchmark

Single-seed results. Selected using validation accuracy.

| Recipe | Validation accuracy | Test accuracy | Macro-F1 | Training + validation seconds |
| --- | ---: | ---: | ---: | ---: |
| assignment | 95.25% | 93.06% | 0.8936 | 1350.7 |
| moderate | 97.46% | 95.05% | 0.9252 | 399.2 |

Weights contain only the model state dictionary. Load with `torch.load(path, weights_only=True)` and `Net().load_state_dict(...)`.
