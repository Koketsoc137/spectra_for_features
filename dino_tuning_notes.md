# DINO on Galaxy10 / radio: diagnosis and literature settings

## 1. Why the `dino_galaxy10` runs score so low

Val kNN (k=5) stays at about 12–17% for all 100 epochs in both v1 (`dino_galaxy10`, Adam) and v2 (`dino_galaxy10_v2`, AdamW, warmup and cosine decay, clipping). That is chance level: guessing in proportion to class frequency gives about 11%, and always picking the majority class gives 14.9%. BYOL with the same loader and `evaluate()` reaches 46.6% (`byol_galaxy10`, epoch 99).

Evidence (from `data_/dino_galaxy10_v2_*` and `logs/dino_galaxy10_14417317.log`):

| | epoch | mean ‖z‖ | exact zeros | kNN (half/half on val) |
|---|---|---|---|---|
| DINO v2 | 0 | 15 819 | 14.6% | 12.2% |
| DINO v2 | 99 | 25.4 | 0% | 15.4% |
| BYOL | 0 | 12.3 | 0% | 18.0% |
| BYOL | 99 | 7.0 | 0% | 43.3% |

- The loss starts at 8.32 (= ln 4096, a uniform output) and stays there for about 8 epochs, then falls to about 1.5. The student learns to match the teacher, but kNN accuracy never improves.
- **Cause 1: the student sees only small crops.** In `vit_pytorch.dino.Dino.forward`, the student is fed only `local_crop`, which keeps 5% (hard-coded) to `local_upper_crop_scale` = 40% of the area. The teacher sees only global crops. On Galaxy10 cutouts, local crops are mostly empty sky, so the objective carries no information about morphology.
- **Cause 2: evaluation uses the student.** `evaluate()` uses `learner(x, return_embedding=True)`, which returns the student encoder's features. Its BatchNorm statistics are fitted to the zoomed-in local crops, so on full images it is out of distribution. This explains the 15 819 norm at epoch 0. The DINO paper (Caron et al. 2021) evaluates the teacher.

Proposed fixes (not yet tested):
1. Evaluate with the teacher: `learner.teacher_encoder(images, return_projection=False)`.
2. Enlarge the local crops by overriding `learner.local_crop`, for example `T.RandomResizedCrop((224, 224), scale=(0.3, 0.6))`, and raise `global_lower_crop_scale`.
3. Bring the remaining settings closer to the literature (Section 2).

## 2. How DINO has been tuned on astronomy images

| | Radio benchmark (Cecconello et al. 2024) | AstroCLIP image encoder (Parker et al. 2024) | This repo (`dino.yaml`) |
|---|---|---|---|
| Data | MeerKAT and other radio cutouts, 224×224, per-image min-max | DESI Legacy Survey DR9 g,r,z, 76M galaxies, 144×144 | Galaxy10 DECaLS, 224×224 |
| Backbone | ResNet-18 / ResNet-50 | ViT-L/12 (DINOv2 with iBOT and KoLeo) | EfficientNet-B0 (ImageNet init) |
| Crops | 2 crops, scale 0.65–1.0; no local crops set | 2 global (144 px, 94.7% of area), 8 local (60 px, 39.4% of area) | student: local 0.05–0.4; teacher: global 0.5–1.0 |
| Augmentations | flips, blur p=0.25, contrast 0.8 p=0.5; no grayscale or solarisation | flips, rotation, PSF-like blur and noise; no colour jitter or solarisation | rotation-crop, flips, resized crop, blur |
| Head | 4096 prototypes, hidden 2048, out 256, normalised last layer | 3 layers, hidden 2048 | K=4096, hidden 256, 4 layers |
| Temperatures | student 0.1; teacher 0.04 → 0.07 over 50 ep | not stated | student 0.1, teacher 0.04 fixed |
| Teacher momentum | 0.9995 → 1.0 (cosine) | 0.994 → 1.0 (cosine) | 0.996 fixed |
| Optimiser | LARS, lr 0.3, wd 1e-6, batch 128, 2 GPUs, sync BN, fp16 | Adam, lr 0 → 0.004 over 80 ep then cosine, wd 0.001, batch 1536 | AdamW, lr 1e-4, wd 0.04, batch 128 |
| Epochs | 400 (config); paper reports 100 or 600 | 500 | 100 |
| Evaluation | linear probe and fine-tune | kNN and MLP on embeddings | kNN on student embedding |

Results and lessons from the literature:
- Neither astronomy setup uses DINO's very small local crops for the student. The radio benchmark trains on two large crops (scale ≥ 0.65). AstroCLIP's local crops still cover about 40% of a tightly centred 144 px cutout.
- Both use few augmentations and drop the natural-image colour ones (jitter, grayscale, solarisation).
- Both run DINO much longer (400–600 epochs) with a teacher momentum schedule that increases to 1.
- In the radio benchmark, a DINO linear probe scored below BYOL in 11 of 12 cases (for example, MiraBest on curated data with ResNet-18: 64.2 vs 89.6). After fine-tuning they were close, and DINO was best on MiraBest (97.1 vs 96.5). Expect DINO kNN to lag BYOL here even after the fixes.
- AstroCLIP keeps weight decay fixed at 0.001 and found that increasing it during training led to underfitting.

Caveats: the AstroCLIP crop sizes, augmentations and backbone were read from a summary of the arXiv HTML, not checked against the MNRAS text. The radio DINO values (temperatures, momentum, head) come from the authors' code config (`scripts/pretrain/banner/dino.yaml` plus solo-learn defaults), not the paper, which lists only base lr 0.016 and projection size 256 for DINO.

## References

- Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Cecconello, T., Riggi, S., Becciani, U., Vitello, F., Hopkins, A. M., Vizzari, G., Spampinato, C., & Palazzo, S. (2024). *Self-supervised learning for radio-astronomy source classification: a benchmark*. [arXiv:2411.14078](https://arxiv.org/abs/2411.14078). Code and configs: [dr4thmos/solo-learn-radio](https://github.com/dr4thmos/solo-learn-radio/blob/master/scripts/pretrain/banner/dino.yaml)
- Parker, L., Lanusse, F., Golkar, S., et al. (2024). *AstroCLIP: a cross-modal foundation model for galaxies*. MNRAS. [doi:10.1093/mnras/stae1450](https://doi.org/10.1093/mnras/stae1450), [arXiv:2310.03024](https://arxiv.org/abs/2310.03024)
- Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Slijepcevic, I. V., et al. (2024). *Radio Galaxy Zoo: towards building the first multi-purpose foundation model for radio astronomy with self-supervised learning*. RASTI. [arXiv:2305.16127](https://arxiv.org/abs/2305.16127). Uses BYOL on RGZ DR1 and MiraBest, a BYOL baseline to compare against.
- Mohale, K., & Lochner, M. (2024). *Enabling unsupervised discovery in astronomical images through self-supervised representations*. MNRAS. [arXiv:2311.14157](https://arxiv.org/abs/2311.14157). Uses BYOL on Galaxy Zoo and MiraBest.
- `vit_pytorch.dino` (lucidrains): the implementation used here. The student sees only local crops: `.venv/deepclustering3/.../vit_pytorch/dino.py` L283–290.
