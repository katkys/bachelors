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
test_dir = "" #path to directory containing images from test dataset

BATCH_SIZE = 20
IMG_SIZE = (299, 299)
INITIAL_EPOCHS = 15

SAVE_MODEL = False
EVALUATE_FINAL_MODEL = False

RANDOM_SEED = 27

BEST_MODEL_SAVE_PATH = "inception_base_model.keras"
HISTORY_SAVE_PATH = "inception_base_train_history.pkl"

set_random_seed(RANDOM_SEED)

train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)

val_dataset = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)  

test_dataset = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    shuffle=False,
    image_size=IMG_SIZE
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
    input_shape=(299, 299, 3),
    include_top=False)

#freeze the base model
base_model.trainable = False

#create a new model on top
inputs = keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(classes_count, activation="softmax")(x) 
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

callbacks = [EarlyStopping(monitor="val_loss",patience=5, restore_best_weights=True)]
if SAVE_MODEL:
    callbacks.append(ModelCheckpoint(BEST_MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss"))  
    
history = model.fit(train_dataset,
                    epochs=INITIAL_EPOCHS,
                    validation_data=val_dataset,
                    verbose=1,
                    callbacks=callbacks)
    
with open(HISTORY_SAVE_PATH, 'wb') as file:
        pickle.dump(history.history, file)

#Training curves
eval.plot_training_history(
    history.history,
    save_path="training_loss_acc.png"
)

metrics = eval.get_best_epoch_metrics(history.history)

print("\n===== MODEL METRICS (model selected based on min val loss) =====")
print(f"Epoch: {metrics['epoch']}")
print(f"Validation loss: {metrics['val_loss']:.4f}")
print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
print(f"Training loss: {metrics['train_loss']:.4f}")
print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
print("=================================================================")

if EVALUATE_FINAL_MODEL:
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"\nTEST RESULTS:\nTest accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}")

    class_names = train_dataset.class_names

    # Confusion matrix
    eval.plot_confusion_matrix(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        normalize=False,
        save_path="confusion_matrix.png"
    )

    # Confusion matrix (normalized)

    eval.plot_confusion_matrix(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        normalize=True,
        save_path="confusion_matrix_normalized.png"
    )

    # ROC curves (one-vs-rest)
    eval.plot_roc_curve(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        save_path="roc_curves.png"
    )

