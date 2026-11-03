# MobileNetV2 implementation status — 2026-11-02

Implemented Keras 3.11.3 MobileNetV2 with the installed PyTorch CPU backend.
Requested ImageNet backbone uses include_top=False, initial full freezing,
last-30-layer fine tuning with all BatchNormalization layers frozen,
training=False backbone calls, and recompilation using a new Adam at 1e-5.
The frozen head phase uses Adam 1e-3. Default schedule is 3 + 7 epochs.
Preprocessing is embedded in the saved graph; real/fake order matches baseline.

Checkpoint selection uses validation accuracy across both phases; test data
is not loaded by training. The existing GoldNet split and old CNN weights
are preserved. The application supports both checkpoint types.

Actual ImageNet training has NOT run. The official weight URL at
storage.googleapis.com returned HTTP tunnel 403 Forbidden under the current
restricted network policy. The domain was saved to the environment draft;
saving does not apply or publish it. User must review/save settings and publish
before retrying the ImageNet operation.

Offline tests deliberately use weights=None ONLY in temporary fixtures:
- All backbone weights initially frozen.
- Only last 30 layers eligible; all BN layers remain frozen.
- Optimizer replaced by recompilation at 1e-5.
- Frozen weights and BN statistics unchanged by a training step.
- At least one eligible backbone weight changes.
- Save/load preserves predictions; multipart inference works.
- Both fit phases and validation checkpoint selection execute end to end.

These checks establish implementation behavior, not ImageNet transfer accuracy.
No MobileNetV2 accuracy or improvement is claimed. Baseline test accuracy remains
72.94%. Evaluation on this previously inspected test set is a development
comparison, not a new blind external validation.
