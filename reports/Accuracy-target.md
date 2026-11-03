# Accuracy improvement experiment — existing data only

Goal: attempt 90% accuracy, without adding labelled gold images. Reaching this
number is not guaranteed and is not a criterion for altering labels or test data.

Protocol defined before new test evaluation:
- Same GoldNet source and five reviewed training exclusions: 1,401 train,
  302 validation, 303 test. No new gold photographs, relabeling, or moved images.
- New initialization is official EfficientNet-B0 ImageNet weights from PyTorch,
  SHA prefix 7f5810bc; existing ResNet18 weights are retained.
- EfficientNet at 320x320, preserving the whole photo with aspect padding and
  ImageNet normalization. Train frozen-feature head for 35 epochs.
- Fine tune features[3:] with all BN parameters/statistics frozen, differential
  AdamW rates 1e-5/3e-5/1e-4, cosine decay, weight decay 0.001, label smoothing
  0.05, training-only horizontal flips and gradient clipping at 1.0.
- Maximum 12 fine-tuning epochs; stop after 5 epochs without validation improvement.
- Select checkpoint by validation accuracy, fewer fake-to-real errors break ties.
- Validation-only comparison of standalone ResNet18, standalone EfficientNet,
  and probability fusion with EfficientNet weights 0.25, 0.5, 0.75.
- Predeclared decision thresholds: 0.4, 0.45, 0.5, 0.55, 0.6 (25 combinations).
  Accuracy primary, fewer counterfeit accepted as real secondary, then closeness
  to 0.5. No threshold selection on test data.
- Fusion uses its actual production forward pass: 320px normalized input,
  resized to 224px for ResNet. This differs slightly from direct PIL preparation
  at 224px, and is evaluated that way on validation instead of assuming equivalence.
- Test evaluation occurs only after candidate selection; no further tuning based
  on test results. No abstention or discarded test cases to inflate accuracy.

The validation set and test set have both been used in previous development.
This remains a development comparison, not a fresh blind external evaluation.
The 25-combination validation search introduces additional model-selection risk.
Softmax/fused scores are uncalibrated, and this task does not prove physical purity.
A changed decision threshold affects the label, not the raw displayed scores.

Before new test evaluation, a training-only BN adaptation comparison was added:
recompute BN running statistics from cleaned training images (no gradients,
model dropout/stochastic depth disabled), then validate five fixed blends of
original/new means and variances (0, .25, .5, .75, 1). Keep original statistics
as a candidate. This changes buffers after frozen-BN fine tuning, not trainable
weights; it is explicitly separate from the frozen-BN training protocol.
The additional validation search further increases selection bias risk.

A pre-test inference-stability comparison was also defined: average probabilities
for the original and horizontal-flip views (TTA), versus a single view. The
validation search therefore contains 50 model/threshold/TTA combinations, not
25. TTA adds inference latency but neither new labelled images nor discarded
cases. Exact ties prefer single-view inference. This additional search further
increases the risk of adapting to the repeatedly used validation set.

## Results

**The 90% overall accuracy target was NOT reached.**

EfficientNet best epoch: fine tuning 11 (88.08% validation). BN blend 0.25 tied its accuracy with one fewer fake-to-real error.

Selected deployment: fusion with EfficientNet weight 0.75 and ResNet18 weight 0.25, flip TTA, threshold 0.55. Selected validation accuracy: 89.40%. Exported artifact was re-evaluated on validation and reproduced the selected confusion matrix.

| Metric | Prior fine-tuned ResNet18 | Selected fusion |
| --- | ---: | ---: |
| Test accuracy | 84.82% | 88.45% |
| Real precision | 85.81% | 92.70% |
| Real recall | 83.55% | 83.55% |
| Real F1 | 84.67% | 87.89% |
| Counterfeit predicted real | 21/151 | 10/151 |
| Real predicted counterfeit | 25/152 | 25/152 |
| Total test errors | 46/303 | 35/303 |

268 of 303 images were classified correctly (88.45%). Reaching at least 90% on this set requires 273 correct predictions. Observed gain: 3.63 percentage points, 11 additional correct predictions. No extra training or threshold search followed this test evaluation.

All prepared image SHA-256 values matched the original split manifest; no new gold images or relabeling were introduced. Existing validation/test membership and all 303 test cases were retained.

Active checkpoint: artifacts/gold_selected.pt. SHA-256: `d2532eba3c97efa5bfdd8fada2605b814d5e5839a600d3d6fb7f524c09a07782`. Scores are raw uncalibrated fusion scores; the decision threshold is 55%, not 50%. Inference evaluates two models and two views, so it is slower than the previous single CNN.

The prior ResNet18 checkpoints remain available for rollback. Validation reused across experiments and the previously inspected test set do not establish independent external performance or physical authenticity.
