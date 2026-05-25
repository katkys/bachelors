import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from argparse import ArgumentParser
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import F1Score

import base_models as bm
import evaluation as eval

VALID_EXTS = ('.jpg', '.jpeg', '.png')


def collect_img_groups(src_path): # img_group = original test sample + its augmented versions
    img_groups = defaultdict(list)

    for artist_dir in src_path.iterdir():
        if not artist_dir.is_dir():
            continue

        for file in artist_dir.iterdir():
            if file.suffix.lower() not in VALID_EXTS:
                continue

            stem = file.stem
            base_name = stem.split("_aug")[0]
            img_groups[(artist_dir.name, base_name)].append(file)

    return img_groups


def load_and_preprocess_image(img_path, img_size, preprocess_function):
    image = keras.utils.load_img(img_path, target_size=img_size)
    x = keras.utils.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_function(x)

    return x


def predict_with_majority_voting(model, img_groups, class_names, img_size,preprocess_function):
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    y_true = []
    y_pred = []
    y_score = []

    total_groups = len(img_groups)
    for i, ((true_class_name, base_name), files) in enumerate(img_groups.items(), start=1):

        print(f"Processing image group {i}/{total_groups} ({true_class_name}/{base_name}):")

        class_votes = []
        confidences = []
        pred_class_for_original = None

        files = sorted(files)
        for file_idx, img_path in enumerate(files):
            image = load_and_preprocess_image(img_path, img_size, preprocess_function)

            pred = model.predict(image, verbose=0)[0]
            pred_class = np.argmax(pred)
            confidences.append(pred)
            class_votes.append(pred_class)

            if "aug" not in img_path.stem:
                pred_class_for_original = pred_class

        confidences = np.array(confidences)

        vote_counts = Counter(class_votes)
        max_votes = max(vote_counts.values())
        tied_classes = [cls for cls, vote_count in vote_counts.items() if vote_count == max_votes]

        
        if len(tied_classes) == 1: # one clear winner
            final_class = tied_classes[0]
        else: # more classes with same max_votes -> choosing the one with highest mean confidence
            tied_confidences = {}

            for tied_class in tied_classes:
                tied_confidences[tied_class] = np.mean(confidences[:, tied_class])

            best_conf = max(tied_confidences.values())
            best_classes = [cls for cls, conf in tied_confidences.items() if conf == best_conf]
            final_class = best_classes[0]

            # if there's still a tie and prediction for original is among best -> choose it
            # (else final_class is the first one from best_classes)
            if len(best_classes) > 1 and pred_class_for_original in best_classes:
                final_class = pred_class_for_original


        # predictions of only majority-vote images from image group 
        majority_confidences = []
        for conf, voted_class in zip(confidences, class_votes):
            if voted_class == final_class:
                majority_confidences.append(conf)
        majority_confidences = np.array(majority_confidences)

        # average only majority-vote predictions
        mean_confidences = np.mean(majority_confidences, axis=0)

        true_class = class_to_idx[true_class_name]
        final_mean_conf = mean_confidences[final_class]
        predicted_class_name = class_names[final_class]

        print(f" True: {true_class_name:20} Pred: {predicted_class_name:20} Votes: {vote_counts[final_class]} Conf: {final_mean_conf:.4f}")

        y_true.append(true_class)
        y_pred.append(final_class)
        y_score.append(mean_confidences)

    return (np.array(y_true), np.array(y_pred), np.array(y_score))


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['A', 'B'])
    parser.add_argument("--data_type", type=str, required=True, choices=['original', 'faces', 'faces_gray', 'masked_faces'])
    parser.add_argument("--model", type=str, required=True, choices=bm.get_supported_models())
    parser.add_argument("--id", type=str, required=True)
    args = parser.parse_args()

    img_size = bm.get_input_size(args.model)
    model_id = f"{args.id}_{args.dataset}_{args.data_type}_{args.model}"
    
    experiment_results_path = f"./Final_{args.dataset}_augmented/{model_id}"
    save_dir_path = experiment_results_path + "/majority_vote_results"
    os.makedirs(save_dir_path)
    model_path = experiment_results_path + "/best_model.keras"

    dataset_dir = f"./AugmentedDatasets/{args.dataset}/{img_size[0]}x{img_size[1]}/{args.data_type}"
    test_dir = dataset_dir + "/test_augmented"

    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}")


    print("\nLoading model...")
    model = keras.models.load_model(model_path, custom_objects={"F1Score": F1Score})

    _, preprocess_function = bm.get_model_and_preprocess_function(args.model)

    print("Collecting image groups...")
    img_groups = collect_img_groups(Path(test_dir))

    test_dataset = keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=16,
        image_size=img_size,
        shuffle=False)
    class_names = test_dataset.class_names

    print("\nEvaluating model using a majority voting approach...")
    y_true, y_pred, y_score = predict_with_majority_voting(
        model=model,
        img_groups=img_groups,
        class_names=class_names,
        img_size=img_size,
        preprocess_function=preprocess_function)


    print("\nFINAL RESULTS:")

    accuracy = np.mean(y_true == y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("Classification report:\n")
    report_path = save_dir_path + f"/classification_report.txt"
    eval.print_classification_report(
        y_true,
        y_pred,
        class_names,
        save_path=report_path)

    mean_auc = eval.get_mean_auc(y_true, y_score, class_names)
    print(f"\nMean AUC: {mean_auc:.4f}")

    print("\nGenerating plots...")

    cm_path = save_dir_path + f"/confusion_matrix.png"
    cm_norm_path = save_dir_path + f"/confusion_matrix_normalized.png"
    roc_path = save_dir_path + f"/roc_curves.png"

    eval.plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path=cm_path)
    eval.plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, save_path=cm_norm_path)
    eval.plot_roc_curve(y_true, y_score, class_names, save_path=roc_path)

    print(f"All output files were saved to: {save_dir_path}")


if __name__ == "__main__":
    main()