# CNN refinement — 2026-11-02

Purpose: improve the available baseline without blocking on ImageNet access.
The MobileNetV2 ImageNet download was retried and still returned HTTP 403 from
the environment proxy for storage.googleapis.com. No random pretrained
substitute was used. Its pending network requirement remains unchanged.

Experiment fixed before inspecting new test predictions:
- Same deduplicated GoldNet dataset and fixed train/validation/test split.
- Initialize from the original best checkpoint, artifacts/gold.pt (epoch 8).
- Same GoldCNN architecture and preprocessing.
- New Adam optimizer, learning rate 0.0001, batch size 32, seed 42.
- 20 additional epochs; choose checkpoint using validation accuracy only.
- Preserve original weights and save candidate as artifacts/gold_refined.pt.
- Evaluate the selected checkpoint on the existing 303-image test set once.

The original test set has previously been examined. This comparison is a
within-dataset development result, not a fresh blind validation or evidence of
performance on arbitrary gold photographs. Independent external evaluation
is still required. Neither model establishes physical purity or authenticity.

## Measured results

Selected additional epoch: 19. Validation accuracy: 74.50% (baseline 72.52%).

| Metric | Baseline | Refined |
| --- | ---: | ---: |
| Test accuracy | 72.94% | 73.93% |
| Real recall | 62.50% | 65.79% |
| Real F1 | 0.6985 | 0.7168 |
| Counterfeit predicted real | 25/151 | 27/151 |
| Real predicted counterfeit | 57/152 | 52/152 |

Accuracy increased by 0.99 percentage points (3 additional correct predictions out of 303). This small observed gain has not been established as statistically significant. Counterfeit-to-real errors increased, so the gain is not an improvement in every operational metric.

Refined checkpoint SHA-256: `304122b18f026efce9d713a271213d2586d269fadf69d5cf552f16f75fb190d7`.
