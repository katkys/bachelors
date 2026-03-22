import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.utils import image_dataset_from_directory, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import evaluation as eval

train_dir = "" #path to directory containing images from training dataset
val_dir = "" #path to directory containing images from validation dataset

IMG_SIZE = (299, 299)

BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
FT_LEARNING_RATE = 1e-5
LABEL_SMOOTHING = 0.00
EARLY_STOP_PATIENCE = 4

SAVE_MODEL = True
MODEL_ID = "M0X_fine-tuned" #an identificator for the model used for naming the output directory where history files and train/val plot will be saved
SAVE_DIR_PATH = f"./models_and_histories/{MODEL_ID}"
os.makedirs(SAVE_DIR_PATH, exist_ok=True)

RANDOM_SEED = 27

BEST_MODEL_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_best_model_before_ft.keras"
BEST_MODEL_FT_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_best_model_after_ft.keras"
HISTORY_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_train_history_before_ft.pkl"
HISTORY_FT_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_train_history_after_ft.pkl"

set_random_seed(RANDOM_SEED)

train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)

val_dataset = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)  

classes_count = len(train_dataset.class_names)

data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.08),
    RandomZoom(0.08),
    RandomTranslation(0.05, 0.05)],
    name="data_augmentation"
)

#load the InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False)

#freeze the base model
base_model.trainable = False

#create a new model on top
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(classes_count, activation="softmax")(x) 
model = keras.Model(inputs, outputs)

metrics = [
    "accuracy",
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.F1Score(average="macro", name="f1_score")
]

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                metrics=metrics)

callbacks = [EarlyStopping(monitor="val_loss",patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
if SAVE_MODEL:
    callbacks.append(ModelCheckpoint(BEST_MODEL_SAVE_PATH, monitor="val_f1_score", mode="max", save_best_only=True))

history = model.fit(train_dataset,
                    epochs=INITIAL_EPOCHS,
                    validation_data=val_dataset,
                    verbose=1,
                    callbacks=callbacks)
    
with open(HISTORY_SAVE_PATH, 'wb') as file:
        pickle.dump(history.history, file)

#unfreeze the base model
base_model.trainable = True

#freeze batch-normalization layers
for layer in base_model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

#fine-tune 
model.compile(optimizer=keras.optimizers.Adam(learning_rate=FT_LEARNING_RATE), 
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
              metrics=metrics)

callbacks = [EarlyStopping(monitor="val_loss",patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
if SAVE_MODEL:
    callbacks.append(ModelCheckpoint(BEST_MODEL_FT_SAVE_PATH, monitor="val_f1_score", mode="max", save_best_only=True))
    
history_ft = model.fit(train_dataset,
            epochs=INITIAL_EPOCHS+FINE_TUNE_EPOCHS,
            initial_epoch=history.epoch[-1]+1,
            validation_data=val_dataset,
            verbose=1,
            callbacks=callbacks)

with open(HISTORY_FT_SAVE_PATH, 'wb') as file:
        pickle.dump(history_ft.history, file)

# Training curves
combined_history = {}
for k in history.history.keys():
    combined_history[k] = history.history[k] + history_ft.history[k]
eval.plot_training_history(
    combined_history,
    save_path=f"{SAVE_DIR_PATH}/training_loss_acc.png"
)

best_epoch_criterium = "val_f1_score"
metrics = eval.get_best_epoch_metrics(combined_history, criterium=best_epoch_criterium)
eval.print_best_epoch_metrics(metrics, criterium=best_epoch_criterium)

