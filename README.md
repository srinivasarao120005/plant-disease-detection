# AgroVision AI | Plant Disease Recognition with Confidence Gating

**Target roles:** AI/ML Engineer / Applied Scientist / Data Scientist. Original project: https://github.com/srinivasarao120005/plant-disease-detection.

## Decision problem and technically defensible improvement
Simple image CNN prediction can be overconfident on unknown/blurred/dark leaves. Compare **scratch CNN vs pretrained MobileNetV3-Small**, evaluate the same independent split, calibrate confidence using a **validation-only temperature**, and introduce a **review-required decision gate** with measurable accepted coverage and accuracy. Controlled blur/low-light stress tests measure robustness degradation. Temperature scaling and MobileNet are established, not invented algorithms: the demonstrated contribution must be improved safety/coverage under transparent test conditions.

## Dataset and cleaning
[PlantVillage (color images)](https://huggingface.co/datasets/mohanty/PlantVillage) contains labeled healthy/diseased leaf images (standard dataset ~54K images across 38 classes). PlantDoc/field-photo external validation is recommended before field-use claims.

**Recommended internet-connected setup**
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install datasets
python src/download_data.py --out data/raw/color
python src/prepare_data.py --root data/raw/color --out results
python src/train.py --manifest results/manifest.csv --model cnn --out results/cnn --epochs 8
python src/train.py --manifest results/manifest.csv --model mobilenet --out results/mobilenet --epochs 8
python src/predict.py --model results/mobilenet/model.pt --image path/to/leaf.jpg
pytest -q
```
Alternative: manually download [original PlantVillage GitHub data](https://github.com/spMohanty/PlantVillage-Dataset) and place only its `raw/color/<class_name>/<image>` folders under `data/raw/color/`. **Do not put both `color` and `segmented` images into the same experiment**, because they show the same underlying leaves.

Data audit (`prepare_data.py`) checks image decode, valid dimensions, exact RGB pixel hash duplicates, conflicting classes (fails loudly), split leakage and class distributions. **The automatic download preserves the dataset's original leaf-disjoint train/test partition**, plus the `leaf_id` field in `data/raw/dataset_index.csv`. Training and validation come only from the official training partition, grouped by `leaf_id`; the original test is untouched. `results/manifest.csv` is the canonical split. If you manually supply images **without** `dataset_index.csv`, the fallback split groups exact RGB duplicates but does **not** guarantee physical-leaf independence. For industry-quality claims use the metadata-preserving download and ideally external field images.

## Evaluation and plots
Training loss/validation accuracy, independent-test confusion matrix, accuracy, macro-F1, balanced accuracy, expected calibration error, confidence gate coverage and accepted accuracy; robustness CSV/plot for blur and low light. All artifacts written under `results/<model>/`. Retain model settings and raw evaluation CSV with your published plot.

## Experimental protocol
Training data only for network updates; validation labels only for early checkpoint selection, temperature and review threshold; **test only once at the end** for each fixed model. `--max-per-split` is a development-only subset selector, not a portfolio benchmark. The gate may abstain on most examples if 90% validation selective accuracy is unachievable; do not hide that failure. Compare CNN and MobileNet under identical splits, class definitions, image sizes, hardware and report training/inference costs.

## Limitations
PlantVillage studio-like backgrounds are not representative of farm conditions. Synthetic corruption is a stress test, not external OOD validation. This classifier does not diagnose crop diseases clinically or prescribe treatments. Pretrained model weights download is network-dependent; test here validates pipeline math and code, not a trained field model.

## Development-only previews
These screenshots are from **synthetic image smoke testing, NOT measured PlantVillage results**.

![Synthetic training curves](examples/synthetic_training_curves.png)
![Synthetic corruption stress](examples/synthetic_robustness.png)

See [results status](docs/RESULTS_STATUS.md).
