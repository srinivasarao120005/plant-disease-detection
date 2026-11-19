# Plant Disease Detection

CNN-based image classification with TensorFlow/Keras and a Streamlit image prediction interface.

## Dataset
Use the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage). Download it from the source and extract the class folders into `data/PlantVillage/`. Respect the dataset repository's license and attribution requirements.

Expected structure:
```
data/PlantVillage/
  Apple___Apple_scab/
  Apple___Black_rot/
  ...
```

## Run
```bash
python -m venv .venv
# activate environment, then:
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training creates `models/plant_disease.keras` and `models/class_names.json`. Reports and training curves are written to `reports/`.

## Notes
The model predicts among the classes present in the downloaded dataset. Image-domain shift, lighting, camera quality and field conditions can reduce performance. This is an educational decision-support prototype, not an agronomic diagnosis.
