import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import pickle
import numpy as np

train_dir = "" #path to directory containing images from training dataset
val_dir = "" #path to directory containing images from validation dataset
test_dir = "" #path to directory containing images from test dataset

BATCH_SIZE = 8
IMG_SIZE = (299, 299)
INITIAL_EPOCHS = 10
SAVE_MODEL = False
EVALUATE_FINAL_MODEL = False

BEST_MODEL_SAVE_PATH = "inception_best_model_before_ft.keras"
HISTORY_SAVE_PATH = "inception_train_history_before_ft.pkl"

train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

val_dataset = image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)  

test_dataset = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    shuffle=False,
    image_size=IMG_SIZE
)

classes_count = len(train_dataset.class_names)

y_train = np.concatenate([y.numpy() for x,y in train_dataset], axis=0)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(classes_count),
    y=y_train
)

class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights)

data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomTranslation(0.1, 0.1)],
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
# x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(classes_count, activation="softmax")(x) 
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
if SAVE_MODEL:
    history = model.fit(train_dataset,
                        epochs=INITIAL_EPOCHS,
                        validation_data=val_dataset,
                        verbose=1,
                        class_weight=class_weights,
                        callbacks=[EarlyStopping(monitor="val_loss",patience=5, restore_best_weights=True),
                                ModelCheckpoint(BEST_MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss")])
    with open(HISTORY_SAVE_PATH, 'wb') as file:
        pickle.dump(history.history, file)
else:
    history = model.fit(train_dataset,
                        epochs=INITIAL_EPOCHS,
                        validation_data=val_dataset,
                        verbose=1,
                        class_weight=class_weights,
                        callbacks=[EarlyStopping(monitor="val_loss",patience=5, restore_best_weights=True)])


# Plot training curves
eval.plot_training_history(
    history.history,
    save_path="training_loss_acc.png"
)

if EVALUATE_FINAL_MODEL:
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"\nTEST RESULTS:\nTest accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}")

    # Get class names (label -> class mapping)
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