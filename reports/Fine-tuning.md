# Extended ResNet18 fine tuning

The initial ResNet18 had already undergone layer4 fine tuning. This experiment
starts from that selected checkpoint and extends training to layer3 + layer4.
No test data is loaded by fine_tune.py. The initial checkpoint is preserved.

Protocol fixed before observing new test results:
- Same cleaned split and five training exclusions: 1,401 train, 302 validation.
- Baseline validation was recomputed: 83.7748% (134 fake correct, 17 fake accepted,
  32 real rejected, 119 real correct).
- Frozen stem/layer1/layer2; all BN affine parameters and running statistics frozen.
- Fresh AdamW: layer3 LR 2e-6, layer4 LR 5e-6, head LR 1e-5; weight decay 0.001.
- Cosine learning rate decay to 5e-7, maximum 8 epochs, batch size 16.
- Training-only horizontal flips, cross entropy with label smoothing 0.05,
  gradient norm clipping at 1.0. No augmentation at validation/inference.
- Stop after four epochs without validation improvement.
- Select by validation accuracy, fewer validation counterfeit-to-real errors
  breaking exact ties. Include initial model as epoch-zero candidate.
- No test-derived threshold tuning or changes to validation/test membership.

The same test set has been examined in previous development rounds. Any reported
change is a within-dataset development comparison, not a blind external test or
proof of physical authenticity. This experiment changes several training settings,
so its results cannot isolate the benefit of opening another block alone.

## Results

Selected additional epoch: 6. Validation accuracy: 86.09%.

| Metric | Previous ResNet18 | Extended fine tuning |
| --- | ---: | ---: |
| Test accuracy | 84.16% | 84.82% |
| Real precision | 87.14% | 85.81% |
| Real recall | 80.26% | 83.55% |
| Real F1 | 83.56% | 84.67% |
| Counterfeit predicted real | 18/151 | 21/151 |
| Real predicted counterfeit | 30/152 | 25/152 |
| Total test errors | 48/303 | 46/303 |

Accuracy increased by only 0.66 percentage points (2 more correct images). False acceptance of counterfeit images increased. This is not an improvement on every operational metric, and statistical significance has not been established. No additional tuning followed test inspection.

Active checkpoint: artifacts/gold_finetuned.pt. The previous checkpoint remains at artifacts/gold_transfer.pt and can be explicitly selected using GOLD_MODEL_PATH.

New checkpoint SHA-256: `76ce3315e8847ca6605164787b54b7d6849adf401f76a8e89fae20f3fd861043`. Original SHA-256 remains `5c7bad61e95df242a9496b22303032618a451f3defa43cb179c21f5268b19872`.

All 13 software tests passed. Training loss uses label smoothing and is not directly comparable to the earlier plain cross-entropy loss. Complete additional epoch logs: fine_tuning_history.json.
