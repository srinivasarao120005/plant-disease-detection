from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt

DATA = Path("data/PlantVillage")
MODEL_DIR = Path("models"); REPORTS = Path("reports")
MODEL_DIR.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
IMG_SIZE=(224,224); BATCH=32; SEED=42

def main():
    if not DATA.exists():
        raise SystemExit("Dataset missing. Download PlantVillage and extract class folders into data/PlantVillage/. See README.")
    train = tf.keras.utils.image_dataset_from_directory(
        DATA, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH)
    val = tf.keras.utils.image_dataset_from_directory(
        DATA, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH)
    names = train.class_names
    (MODEL_DIR/"class_names.json").write_text(json.dumps(names, indent=2))
    autotune=tf.data.AUTOTUNE
    train=train.cache().shuffle(1000).prefetch(autotune)
    val=val.cache().prefetch(autotune)
    aug=tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1)])
    inputs=tf.keras.Input(shape=(*IMG_SIZE,3))
    x=tf.keras.layers.Rescaling(1./255)(inputs)
    x=aug(x)
    x=tf.keras.layers.Conv2D(32,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(128,3,padding="same",activation="relu")(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(0.35)(x)
    outputs=tf.keras.layers.Dense(len(names),activation="softmax")(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True),
               tf.keras.callbacks.ModelCheckpoint(MODEL_DIR/"plant_disease.keras",save_best_only=True)]
    hist=model.fit(train,validation_data=val,epochs=25,callbacks=callbacks)
    model.save(MODEL_DIR/"plant_disease.keras")
    plt.figure(); plt.plot(hist.history["accuracy"],label="train"); plt.plot(hist.history["val_accuracy"],label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS/"accuracy_curve.png",dpi=160); plt.close()
    loss,acc=model.evaluate(val,verbose=0)
    print({"validation_loss":float(loss),"validation_accuracy":float(acc),"classes":len(names)})
if __name__=="__main__": main()
