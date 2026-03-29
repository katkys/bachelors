import os
import argparse
import tensorflow as tf

import evaluation as eval
import base_models as bm

K = 5  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["A", "B"])
    parser.add_argument("--data_type", required=True,
                        choices=["original", "faces", "faces_gray", "masked_faces"])
    parser.add_argument("--model", required=True, choices=list(bm.get_supported_models()))
    parser.add_argument("--id", required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    img_size = bm.get_input_size(args.model)    
    model_id = f"{args.id}_{args.dataset}_{args.data_type}_{args.model}"
    base_dir = f"./Models_{args.dataset}/{model_id}"

    if not os.path.exists(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")

    dataset_root = f"./Dataset_{args.dataset}/{img_size[0]}x{img_size[1]}/{args.data_type}"
    test_path = f"{dataset_root}/test"
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=img_size,
        batch_size=args.batch,
        label_mode='categorical',
        shuffle=False)
    class_names = test_dataset.class_names

    all_test_results = {}
    print(f"\nEvaluating experiment: {model_id}")

    for fold in range(1, K+1):
        print(f"\nFOLD {fold}:")
        model_path = os.path.join(base_dir, f"best_model_fold{fold}.keras")

        if not os.path.exists(model_path):
            print(f"Model not found for fold {fold}, skipping...")
            continue

        print(f"\nLoading model: {model_path}")
        model = tf.keras.models.load_model(model_path)

        print("\nEvaluating on test dataset...")
        results = model.evaluate(test_dataset, verbose=1, return_dict=True)
        all_test_results[fold] = results

        y_pred, y_true, y_score = eval.get_preds_labels_scores(model, test_dataset)

        print("\nClassification report:")
        report_path = base_dir + f"/classification_report_fold{fold}.txt"
        eval.print_classification_report(y_true, y_pred, class_names, save_path=report_path)


        show_plots = input("\nDo you want to see the confusion matrix and ROC curves? (y/n): ").strip().lower()
        if show_plots == 'y':
            save_plots = input("Do you want to save them? (y/n): ").strip().lower()

            cm_path, cm_norm_path, roc_path = None, None, None
            if save_plots == 'y':
                cm_path = base_dir + f"/confusion_matrix_fold{fold}.png"
                cm_norm_path = base_dir + f"/confusion_matrix_normalized_fold{fold}.png"
                roc_path = base_dir + f"/roc_curves_fold{fold}.png"

            eval.plot_confusion_matrix(y_pred, y_true, class_names, normalize=False, save_path=cm_path)
            eval.plot_confusion_matrix(y_pred, y_true, class_names, normalize=True, save_path=cm_norm_path)
            eval.plot_roc_curve(y_true, y_score, class_names, save_path=roc_path)

        show_detailed_predictions = input("\nDo you want to see detailed predictions for each test sample? (y/n): ").strip().lower()
        if show_detailed_predictions == 'y':
            file_paths = test_dataset.file_paths

            correct = 0
            incorrect = 0

            for i, (pred, true) in enumerate(zip(y_pred, y_true)):
                rel_path = os.path.relpath(file_paths[i], test_path)
                
                pred_label = class_names[pred]
                correct_pred = "✓" if (pred == true) else "✗"

                print(f"'{rel_path}' was predicted to belong to class {pred_label}: {correct_pred}")

                if pred == true:
                    correct += 1
                else:
                    incorrect += 1

            print(f"\nCorrect predictions: {correct}")
            print(f"Incorrect predictions: {incorrect}")

    print("\n===== SUMMARY OF TEST RESULTS ACROSS FOLDS =====")
    eval.print_avg_metrics(all_test_results)

if __name__ == "__main__":
    main()
