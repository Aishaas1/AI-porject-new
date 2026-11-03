# GoldNet CNN training — 2026-11-02

Source: https://github.com/zobeirraisi/GoldNet
Commit: f206c941a39d3126b6ce867e01abfad87e10f525
Author: Zobeir Raisi. Dataset license: CC BY 4.0 (see GoldNet-LICENSE-DATA).
Labels are used as supplied by the source; they were not independently physically verified.

The main real/fake directories contain 2,127 images. Decoded RGB pixel hashing
found 116 redundant images within their respective labels. No identical images
had conflicting labels. After deduplication, 2,011 images remained.
The pairs directory was excluded. No image was edited on disk.

Seed 42 stratified split:

| Split | Fake | Real | Total |
| --- | ---: | ---: | ---: |
| Train | 702 | 704 | 1,406 |
| Validation | 151 | 151 | 302 |
| Test | 151 | 152 | 303 |

Architecture: the application's three-convolution GoldCNN, input 128×128 RGB,
Adam learning rate 0.001, batch size 32, cross entropy, 10 epochs on CPU,
PyTorch 2.5.1+cpu. No pretrained external weights or augmentation.
Checkpoint selected only by validation accuracy: epoch 8, 72.52% validation accuracy.
Independent test set evaluated after model selection: 72.94% accuracy (221/303).
Real precision 79.17%, real recall 62.50%, real F1 69.85%.

Test confusion matrix, rows actual / columns predicted:

| | Predicted fake | Predicted real |
| --- | ---: | ---: |
| Actual fake | 126 | 25 |
| Actual real | 57 | 95 |

Weights: artifacts/gold.pt (local, ignored by Git).
Metrics: goldnet_validation_metrics.json, goldnet_test_metrics.json.
Per-image provenance and split hashes: goldnet_split.json.

The application was restarted with these weights. HTTP health reported
model_ready=true and an actual multipart image request returned a classification.
Three software tests passed, including the synthetic training/inference workflow
and deduplication/split preparation. Synthetic tests are not accuracy evidence.

This is an initial baseline, not a reproduction of GoldFormer or its published
cross-validation. Pixel deduplication does not rule out near-duplicate scenes,
shared acquisition conditions or other dataset bias. There is no external-domain
validation or score calibration. The observed 25 counterfeit-to-real errors make
this unsuitable for purchase decisions or proof of authenticity/purity.

Checkpoint SHA-256: `fcbce957dd99fa49d932c797c9fe89be8defaf459aa519de2c6debbef99cfd68`.
