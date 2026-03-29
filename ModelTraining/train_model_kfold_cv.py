import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.utils import image_dataset_from_directory, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import F1Score
import pickle
import json

import evaluation as eval
import base_models as bm

K = 5
RANDOM_SEED = 27
SAVE_MODEL = True

def train_chosen_model(model_name, dataset, data_type, id, config):
    img_size = bm.get_input_size(model_name)
    
    model_id = f"{id}_{dataset}_{data_type}_{model_name}" 
    dataset_dir = f"./Dataset_{dataset}/{img_size[0]}x{img_size[1]}/{data_type}"
    save_dir_path = f"./Models_{dataset}/{model_id}"
    if os.path.exists(save_dir_path):
        raise ValueError(f"Experiment with given id '{id}' already exists: {save_dir_path}")
    os.makedirs(save_dir_path)

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"{k:20}: {v}")
    with open(f"{save_dir_path}/config.json", "w") as config_file:
        json.dump(config, config_file)


    print(f"\nSetting random seed to: {RANDOM_SEED}")
    set_random_seed(RANDOM_SEED)


    all_val_results = {}

    for fold in range(1, K+1):
        print(f"\nFOLD: {fold}")

        fold_dir = dataset_dir + f"/fold_{fold}"
        train_dir = fold_dir + "/training" 
        val_dir = fold_dir + "/validation" 

        history_save_path = f"{save_dir_path}/train_history_before_ft_fold{fold}.pkl"
        history_ft_save_path = f"{save_dir_path}/train_history_after_ft_fold{fold}.pkl"
        best_model_save_path = f"{save_dir_path}/best_model_fold{fold}.keras"


        print("\nLoading train dataset...")
        train_dataset = image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=config['batch'],
            image_size=img_size,
            shuffle=True,
            seed=RANDOM_SEED)

        print("Loading validation dataset...")
        val_dataset = image_dataset_from_directory(
            val_dir,
            labels='inferred',
            label_mode='categorical',
            batch_size=config['batch'],
            image_size=img_size,
            seed=RANDOM_SEED)  

        classes_count = len(train_dataset.class_names)
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

        metrics = ["accuracy",
                    F1Score(average="macro", name="f1_score")]

        model.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.CategoricalCrossentropy(label_smoothing=config['label_smoothing']),
                        metrics=metrics)

        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=config['early_stop_patience'],
                                   restore_best_weights=True)]
        if SAVE_MODEL:
            callbacks.append(ModelCheckpoint(best_model_save_path,
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True))

        print("\nStarting the training process...")
        history = model.fit(train_dataset,
                            epochs=config['epochs'],
                            validation_data=val_dataset,
                            verbose=1,
                            callbacks=callbacks)
            
        with open(history_save_path, 'wb') as file:
            pickle.dump(history.history, file)


        combined_history = {}
        fine_tuning_depth = config['ft_depth']
        if fine_tuning_depth != "none":

            n = len(base_model.layers)
            if fine_tuning_depth == "half":
                n //= 2

            for layer in base_model.layers[-n:]:
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
                
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['lr']), 
                        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config['label_smoothing']),
                        metrics=metrics)
            
            
            callbacks_ft = [EarlyStopping(monitor="val_loss",
                                          patience=config['early_stop_patience'],
                                            restore_best_weights=True)]
            if SAVE_MODEL:
                min_val_loss_before_ft = min(history.history["val_loss"])
                callbacks_ft.append(ModelCheckpoint(best_model_save_path,
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True,
                                     initial_value_threshold=min_val_loss_before_ft)) 
                    
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
            save_path=f"{save_dir_path}/training_loss_acc_fold{fold}.png")

        train_val_metrics = eval.get_best_epoch_metrics(combined_history)
        all_val_results[fold] = eval.extract_val_metrics(train_val_metrics)
        eval.print_best_epoch_metrics(train_val_metrics)

    eval.print_avg_metrics_summary(all_val_results)
    print(f"\nConfiguration info, training histories and train&val plots were saved to: {save_dir_path}")

    
def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['A', 'B'], 
                        help="Which dataset you want to use.")
    parser.add_argument("--data_type", type=str, required=True, choices=['original', 'faces', 'faces_gray', 'masked_faces'], 
                        help="Which version of the images you want to use.")
    parser.add_argument("--model", type=str, required=True, choices=bm.get_supported_models(), 
                        help="Base model which will be used with weights pretrained on ImageNet.")
    parser.add_argument("--id", type=str, required=True, help="Experiment ID used for naming the output folder.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, choices=range(2, 6), default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dense_size", type=int, default=512)
    parser.add_argument("--ft_depth", type=str, choices=['all', 'half', 'none'], default='all', help="Number of layers of base model to unfreeze for fine-tuning.")

    args = parser.parse_args()


    config = {"epochs" : args.epochs,
                "ft_epochs" : args.ft_epochs,
                "batch" : args.batch,
                "lr" : args.lr,
                "label_smoothing" : args.label_smoothing,
                "early_stop_patience" : args.early_stop_patience,
                "ft_depth" : args.ft_depth,
                "dropout" : args.dropout,
                "dense_size" : args.dense_size }
    
    train_chosen_model(args.model, args.dataset, args.data_type, args.id, config)


if __name__ == "__main__":
    main()