import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.metrics import F1Score

import base_models as bm
import evaluation as eval


def load_test_dataset(dataset, data_type, model_name):
    img_size = bm.get_input_size(model_name)

    dataset_dir = f"./new_split_datasets/{dataset}/{img_size[0]}x{img_size[1]}/{data_type}"
    test_dir = dataset_dir + "/test"

    test_dataset = image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=16,
        image_size=img_size,
        shuffle=False
    )

    return test_dataset, test_dir


def main():
    parser = ArgumentParser()

    parser.add_argument("--dataset", required=True, choices=['A', 'B'])
    parser.add_argument("--data_type", required=True, choices=['original', 'faces', 'faces_gray', 'masked_faces'])
    parser.add_argument("--model", required=True, choices=bm.get_supported_models())
    parser.add_argument("--id", required=True)

    args = parser.parse_args()

    model_id = f"{args.id}_{args.dataset}_{args.data_type}_{args.model}"
    model_path = f"./new_split_Final_{args.dataset}/{model_id}/best_model.keras"

    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")

    print("\nLoading test dataset...")
    test_dataset, test_dir = load_test_dataset(args.dataset, args.data_type, args.model)

    print("Loading model...")
    model = keras.models.load_model(model_path,custom_objects={"F1Score": F1Score})

    print("Evaluating model on test dataset...")
    y_pred, y_true, y_score = eval.get_preds_labels_scores(model, test_dataset)
    eval.print_prediction_stats(y_true, y_pred, y_score)
    eval.print_predictions(test_dataset, test_dir, y_true, y_pred, y_score)


if __name__ == "__main__":
    main()