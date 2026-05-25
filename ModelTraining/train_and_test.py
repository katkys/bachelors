import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import F1Score
import pickle
import json

import evaluation as eval
import base_models as bm

def train_chosen_model(model_name, dataset, data_type, id, config):
    img_size = bm.get_input_size(model_name)
    
    model_id = f"{id}_{dataset}_{data_type}_{model_name}" 
    dataset_dir = f"./Datasets/{dataset}/{img_size[0]}x{img_size[1]}/{data_type}"
    save_dir_path = f"./Final_{dataset}/{model_id}"
    if os.path.exists(save_dir_path):
        raise ValueError(f"Final model with given id '{id}' already exists: {save_dir_path}")
    os.makedirs(save_dir_path)

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"{k:20}: {v}")
    with open(f"{save_dir_path}/config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)

    train_dir = dataset_dir + "/train" 
    val_dir = dataset_dir + "/val"
    test_dir = dataset_dir + "/test"

    history_save_path = f"{save_dir_path}/train_history_before_ft.pkl"
    history_ft_save_path = f"{save_dir_path}/train_history_after_ft.pkl"
    best_model_save_path = f"{save_dir_path}/best_model.keras"


    print("\nLoading train dataset...")
    train_dataset = image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=config['batch'],
        image_size=img_size,
        shuffle=True)

    print("Loading validation dataset...")
    val_dataset = image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=config['batch'],
        image_size=img_size)
    
    print("Loading test dataset...")
    test_dataset = image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=config['batch'],
        image_size=img_size,
        shuffle=False)

    class_names = test_dataset.class_names
    classes_count = len(class_names)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


    data_augmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.08),
        RandomZoom(0.08),
        RandomTranslation(0.05, 0.05)],
        name="data_augmentation")


    base_model, preprocess_function = bm.get_model_and_preprocess_function(model_name)
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = data_augmentation(inputs)
    x = preprocess_function(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(config['dense_size'], activation='relu')(x)
    x = Dropout(config['dropout'])(x)
    outputs = Dense(classes_count, activation="softmax")(x) 
    model = keras.Model(inputs, outputs)

    outputs = Dense(classes_count, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    metrics = ["accuracy",
                F1Score(average="macro", name="f1_score")]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['lr']),
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=config['label_smoothing']),
                    metrics=metrics)

    callbacks = [EarlyStopping(monitor="val_loss",
                                patience=config['early_stop_patience'],
                                restore_best_weights=True),
                ModelCheckpoint(best_model_save_path,
                                    monitor="val_loss",
                                    mode="min",
                                    save_best_only=True)]

    print("\nStarting the training process...")
    history = model.fit(train_dataset,
                        epochs=config['epochs'],
                        validation_data=val_dataset,
                        verbose=1,
                        callbacks=callbacks)
        
    with open(history_save_path, 'wb') as file:
        pickle.dump(history.history, file)


    combined_history = {}
    if config['ft_epochs'] > 0:
        for layer in base_model.layers:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True
            
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['ft_lr']), 
                    loss=keras.losses.CategoricalCrossentropy(label_smoothing=config['label_smoothing']),
                    metrics=metrics)
        
        # best saved model from initial training is only updated during fine-tuning
        # if the val loss decreases compared to lowest val loss achieved during initial training
        min_val_loss_before_ft = min(history.history["val_loss"])
        callbacks_ft = [EarlyStopping(monitor="val_loss",
                                        patience=config['early_stop_patience'],
                                        restore_best_weights=True),
                        ModelCheckpoint(best_model_save_path,
                                    monitor="val_loss",
                                    mode="min",
                                    save_best_only=True,
                                    initial_value_threshold=min_val_loss_before_ft)]
                
        history_ft = model.fit(train_dataset,
                    epochs=config['epochs']+config['ft_epochs'],
                    initial_epoch=len(history.history["loss"]),
                    validation_data=val_dataset,
                    verbose=1,
                    callbacks=callbacks_ft)

        with open(history_ft_save_path, 'wb') as file:
                pickle.dump(history_ft.history, file)

        for key in history.history.keys():
            combined_history[key] = history.history[key] + history_ft.history[key]

    else:
        combined_history = history.history

    print("Training complete.")


    eval.plot_training_history(
        combined_history,
        save_path=f"{save_dir_path}/training_loss_acc.png")

    train_val_metrics = eval.get_best_epoch_metrics(combined_history)
    eval.print_best_epoch_metrics(train_val_metrics)

    print("\nEvaluating the model on test dataset...")
    model = keras.models.load_model(best_model_save_path) 
    model.evaluate(test_dataset, verbose=1, return_dict=True)

    y_pred, y_true, y_score = eval.get_preds_labels_scores(model, test_dataset)

    print("\nClassification report:")
    report_path = save_dir_path + f"/classification_report.txt"
    eval.print_classification_report(y_true, y_pred, class_names, save_path=report_path)

    mean_auc = eval.get_mean_auc(y_true, y_score, class_names)
    print(f"Mean AUC: {mean_auc:.2f}")

    cm_path = save_dir_path + f"/confusion_matrix.png"
    cm_norm_path = save_dir_path + f"/confusion_matrix_normalized.png"
    roc_path = save_dir_path + f"/roc_curves.png"

    eval.plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path=cm_path)
    eval.plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_path=cm_norm_path)
    eval.plot_roc_curve(y_true, y_score, class_names, save_path=roc_path)

    print(f"\nAll output files were saved to: {save_dir_path}")

    
def main():
    parser = ArgumentParser()
    
    #DEFAULT SETTINGS => values we used for training final models for Dataset A
    parser.add_argument("--dataset", type=str, required=True, choices=['A', 'B'])
    parser.add_argument("--data_type", type=str, required=True, choices=['original', 'faces', 'faces_gray', 'masked_faces'])
    parser.add_argument("--model", type=str, required=True, choices=bm.get_supported_models())
    parser.add_argument("--id", type=str, required=True, help="Experiment ID used for naming the output folder.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ft_lr", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dense_size", type=int, default=512)

    args = parser.parse_args()


    config = {"epochs" : args.epochs,
                "ft_epochs" : args.ft_epochs,
                "batch" : args.batch,
                "lr" : args.lr,
                "ft_lr" : args.ft_lr,
                "label_smoothing" : args.label_smoothing,
                "early_stop_patience" : args.early_stop_patience,
                "dropout" : args.dropout,
                "dense_size" : args.dense_size }
    
    train_chosen_model(args.model, args.dataset, args.data_type, args.id, config)


if __name__ == "__main__":
    main()