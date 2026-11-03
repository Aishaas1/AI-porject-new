# GoldNet data engineering — 2026-11-02

## Audit and leakage correction

Input: existing GoldNet split, source commit
f206c941a39d3126b6ce867e01abfad87e10f525. The prior exact decoded-pixel
deduplication already removed 116 redundant images from 2,127 source images.

The new non-destructive audit inspected all 2,011 remaining images:
- 0 undecodable/corrupt images.
- 1 dark validation image flagged for review, retained unchanged.
- 5 near-duplicate candidate pairs at 256-bit dHash Hamming distance <=8.
- All 5 pairs reviewed visually and showed the same items/scenes with small
  image changes. Four were within training; one crossed training and test.

The training image r_img_986.jpg overlaps the test scene r_img_950.jpg.
It was excluded from new training without changing the test set. Four redundant
training images were also excluded. Files and labels are preserved; exclusions
are applied by the loader and recorded in engineering_exclusions.json.

New training membership: 1,401 images (702 fake, 699 real).
Validation: unchanged 302 (151 per class). Test: unchanged 303 (151 fake, 152 real).
All new experiments initialize weights from scratch; previous weights had already
seen the overlapping scene and are unsuitable as leakage-free initializations.
The audit is a screening method, not proof that no other related images exist.

## Predeclared experiment design

Same GoldCNN architecture, seed 42, batch size 32, Adam 0.001, weight decay 0.0001,
maximum 30 epochs per trial. A validation-accuracy plateau halves the learning
rate after 3 stagnant epochs; training stops after 7 non-improving epochs.
Class weights computed only from cleaned training labels (nearly balanced).
No artificial oversampling or label editing. Best checkpoint by validation accuracy.

Three trials:
1. clean_control: original deterministic 128×128 preprocessing, no augmentation.
2. reflect_aug: EXIF correction and RGB conversion; Lanczos resize preserving the
   entire image/aspect ratio, reflection padding rather than a black frame;
   training-only augmentation.
3. luminance_aug: same as reflect_aug plus 1% luminance autocontrast in YCbCr.
   RGB channels/hues are not independently equalized.

Training augmentation: horizontal flips, rotation within ±12 degrees, small
scale/translation changes, brightness/contrast within ±10%. No generated images
are added to validation/test; validation/test preprocessing is deterministic.
The deterministic preprocessing configuration is embedded in each checkpoint
and reused by evaluation and the upload interface. Original photos are unchanged.

Selection uses only validation scores. Only the selected new model is evaluated
on the 303-image test set after selection. This test set has been examined in
previous development rounds; it is not a new blind external evaluation. Old
results also involved a now-discovered overlapping scene, so they are historical
context, not a perfectly controlled leakage-free benchmark. No statistical
significance or physical gold-authentication guarantee is implied.

## Measured results

| Trial | Best validation accuracy | Selection |
| --- | ---: | --- |
| clean_control | 75.50% | selected |
| reflect_aug | 75.50% | not evaluated on test |
| luminance_aug | 75.17% | not evaluated on test |

The simpler clean_control won the tie with reflect_aug. Augmentation and luminance adjustments were implemented and tested, but were not enabled in the active model because they did not beat the clean baseline. The gain cannot be attributed to augmentation or contrast enhancement.

| Test metric | Previous refined CNN | Selected cleaned CNN |
| --- | ---: | ---: |
| Accuracy | 73.93% | 76.90% |
| Real precision | 78.74% | 78.08% |
| Real recall | 65.79% | 75.00% |
| Real F1 | 71.68% | 76.51% |
| Counterfeit predicted real | 27/151 | 32/151 |
| Real predicted counterfeit | 52/152 | 38/152 |
| Total errors | 79/303 | 70/303 |

Observed accuracy gain: 2.97 percentage points, 9 more correct predictions on the same 303 images. This small development comparison is not evidence of statistical significance or external-domain generalization. False acceptance of counterfeit images increased, so the active model is not improved on every metric. No thresholds were tuned on test results.

Active artifact: artifacts/gold_engineered.pt. Selected epoch: 27. SHA-256: `b4bed11b7c4f027db6ef847e08c88af5165794472bc55fee6ba8d4d94fe00c24`.

All 2,011 original images were checked against the saved manifest SHA-256 values and remain byte-for-byte unchanged. Source, exclusions and trial membership are recorded in data_audit.json, engineering_exclusions.json and engineering_membership.json.
