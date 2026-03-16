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
import numpy as np

RANDOM_SEEDS = [27, 336, 10101, 5821, 970] # number of random seeds = number of trained models (whose test metrics will be averaged)

MODEL_ID = "MXX_final" #an identificator for the model used for naming the output directory where history files and train/val plot will be saved

train_dir = "" #path to directory containing images from training dataset
val_dir = "" #path to directory containing images from validation dataset
test_dir = "" #path to directory containing images from test dataset

IMG_SIZE = (299, 299)

BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
FT_LEARNING_RATE = 1e-5
LABEL_SMOOTHING = 0.0
EARLY_STOP_PATIENCE = 4

SAVE_MODEL = True

print("\nLoading test dataset...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False  
)

class_names = test_dataset.class_names

all_test_results = {}

for seed in RANDOM_SEEDS:
    SAVE_DIR_PATH = f"./final_models/{MODEL_ID}/{seed}"
    os.makedirs(SAVE_DIR_PATH, exist_ok=True)

    BEST_MODEL_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_best_model_before_ft.keras"
    BEST_MODEL_FT_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_best_model_after_ft.keras"
    HISTORY_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_train_history_before_ft.pkl"
    HISTORY_FT_SAVE_PATH = f"{SAVE_DIR_PATH}/inception_train_history_after_ft.pkl"

    print(f"\nSetting random seed to: {seed}")
    set_random_seed(seed)

    print("Loading train dataset...")
    train_dataset = image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=seed
    )

    print("Loading validation dataset...")
    val_dataset = image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=seed
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

    print("\nStarting the training process...")
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

    # #unfreeze only top 160 layers of base model
    # for layer in base_model.layers[-160:]:
    #     if not isinstance(layer, keras.layers.BatchNormalization):
    #         layer.trainable = True

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

    print("Training complete.")
    # Training curves
    combined_history = {}
    for k in history.history.keys():
        combined_history[k] = history.history[k] + history_ft.history[k]
    eval.plot_training_history(
        combined_history,
        save_path=f"{SAVE_DIR_PATH}/training_loss_acc.png"
    )

    print("\nStarting evaluation of best model:")
    model = tf.keras.models.load_model(BEST_MODEL_FT_SAVE_PATH)
    results = model.evaluate(test_dataset, return_dict=True)
    all_test_results[seed] = results

    print("\n===== TEST RESULTS =====")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    print("========================\n")

    print("Plotting confusion matrices and ROC curves...")
    # Confusion matrix 
    eval.plot_confusion_matrix(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        normalize=False,
        save_path=os.path.join(SAVE_DIR_PATH, "confusion_matrix.png")
    )

    # Confusion matrix (normalized)
    eval.plot_confusion_matrix(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        normalize=True,
        save_path=os.path.join(SAVE_DIR_PATH, "confusion_matrix_normalized.png")
    )

    # ROC curves (one-vs-rest)
    eval.plot_roc_curve(
        model=model,
        dataset=test_dataset,
        class_names=class_names,
        save_path=os.path.join(SAVE_DIR_PATH, "roc_curves.png")
    )

    print(f"Model, training history and plots were saved to: {SAVE_DIR_PATH}")

print("\n===== AVERAGE RESULTS ACROSS ALL SEEDS =====")
avg_results = {}
for metric in all_test_results[RANDOM_SEEDS[0]].keys(): 
    avg_results[metric] = np.mean([res[metric] for res in all_test_results.values()])
    
    print(f"{metric}: {avg_results[metric]:.4f}")

print("============================================\n")

    

