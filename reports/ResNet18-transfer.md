# ImageNet transfer learning — ResNet18

This experiment uses a different official pretrained CNN from the Keras
MobileNetV2 implementation. Google-hosted Keras weights remain blocked; the
ResNet18 ImageNet weights were successfully obtained from the policy-allowed
PyTorch distribution at https://download.pytorch.org/models/resnet18-f37072fd.pth.
Torchvision checks the artifact hash during download. No proxy bypass or TLS
verification override was used.

## Protocol fixed before inspecting new test results

- Same cleaned membership: 1,401 train, 302 validation, 303 test.
- All five reviewed near-duplicate training exclusions retained.
- No initialization from previously trained GoldCNN weights.
- Official torchvision ResNet18_Weights.IMAGENET1K_V1 initialization.
- Preserve complete photograph using aspect-preserving padding to 224×224;
  RGB conversion/EXIF correction, scaling to [0,1], ImageNet mean/std normalization.
- Stage 1: freeze backbone, cache features from train/validation only, train a
  dropout + two-class head for 25 epochs (AdamW 0.001, weight decay 0.001).
- Stage 2: start from best validation checkpoint; fine-tune layer4 and head for
  6 epochs with a fresh AdamW at 1e-5 (weight decay 0.0001).
- BatchNormalization affine parameters and running statistics remain frozen.
- No augmentation, test-time augmentation or test-derived threshold tuning.
- Select across stages by validation accuracy; fewer counterfeit-to-real errors
  on validation break exact accuracy ties. Test data is not loaded by training.

This previously inspected test set is a development comparison, not new blind
external validation. Scores remain uncalibrated and do not establish physical
authenticity/purity. Software tests using random temporary weights are separate
from empirical performance measurements of the ImageNet-initialized model.

## Measured results

Selected stage: fine_tune, epoch 4. Validation accuracy: 83.77%.

| Metric | Prior cleaned GoldCNN | ImageNet ResNet18 |
| --- | ---: | ---: |
| Test accuracy | 76.90% | 84.16% |
| Real precision | 78.08% | 87.14% |
| Real recall | 75.00% | 80.26% |
| Real F1 | 76.51% | 83.56% |
| Counterfeit predicted real | 32/151 | 18/151 |
| Real predicted counterfeit | 38/152 | 30/152 |
| Total test errors | 70/303 | 48/303 |

Observed accuracy gain: 7.26 percentage points, 22 additional correct predictions. Both types of error decreased on this split. The repeated-use and external-validation limitations above still apply.

Selected model SHA-256: `5c7bad61e95df242a9496b22303032618a451f3defa43cb179c21f5268b19872`.

Official initialization SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

The 11 software tests passed. Offline tests verify last-block trainability, frozen BN parameters/statistics, weight updates, local save/load and HTTP preprocessing consistency. Training and all 31 epoch records are in transfer_history.json.
