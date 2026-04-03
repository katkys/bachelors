import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report


def plot_training_history(history, save_path=None):
    loss = history.get("loss")
    val_loss = history.get("val_loss")
    acc = history.get("accuracy") or history.get("acc")
    val_acc = history.get("val_accuracy") or history.get("val_acc")

    epochs = range(1, len(loss) + 1)
    epoch_xticks = range(1, len(loss) + 1, 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, loss, label="Trénovacia strata")
    axes[0].plot(epochs, val_loss, label="Validačná strata")
    axes[0].set_title(f"Trénovacia a validačná strata")
    axes[0].set_xlabel("Počet epoch")
    axes[0].set_ylabel("Strata")
    axes[0].set_xticks(epoch_xticks)
    axes[0].legend()

    axes[1].plot(epochs, acc, label="Trénovacia celková presnosť")
    axes[1].plot(epochs, val_acc, label="Validačná celková presnosť")
    axes[1].set_title(f"Trénovacia a validačná celková presnosť")
    axes[1].set_xlabel("Počet epoch")
    axes[1].set_ylabel("Celková presnosť")
    axes[1].set_xticks(epoch_xticks)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close(fig)

def get_preds_labels_scores(model, dataset):
    y_true = []
    y_pred = []
    y_score = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        true_classes = np.argmax(labels.numpy(), axis=1)
        pred_classes = np.argmax(preds, axis=1)

        y_true.extend(true_classes)
        y_pred.extend(pred_classes)
        y_score.extend(preds)

    return np.array(y_pred), np.array(y_true), np.array(y_score)


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(12, 12))  
    cm_display.plot(
        ax=ax,
        xticks_rotation=90,  
        colorbar=normalize
    )

    ax.set_title("Matica zmätku" + (" (normalizovaná)" if normalize else ""))
    plt.subplots_adjust(bottom=0.35) 
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_roc_curve(y_true, y_score, class_names, save_path=None):
    num_of_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(num_of_classes):
        y_true_bin = (y_true == i).astype(int) # 1 -> sample belongs to class i, 0 -> sample doesn't belong to class i
        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC krivky (One-vs-Rest)")
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close(fig)

def print_classification_report(y_true, y_pred, class_names, save_path=None):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(report)

def get_avg_auc(y_true, y_score, class_names):
    num_of_classes = len(class_names)
    sum_auc = 0
    for i in range(num_of_classes):
        y_true_bin = (y_true == i).astype(int) 
        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        sum_auc += roc_auc

    return sum_auc/num_of_classes

def get_best_epoch_metrics(history_dict, criterium="val_loss"):
    if criterium not in history_dict:
        raise ValueError(f"Metric for choosing best epoch ('{criterium}') not found in history keys: {list(history_dict.keys())}")
    
    values = np.array(history_dict[criterium])
    if "loss" in criterium.lower():
        best_epoch_idx = np.argmin(values)
    else:
        best_epoch_idx = np.argmax(values)

    best_metrics = {"epoch": int(best_epoch_idx + 1)}
    for key, metric_values in history_dict.items():
        metric_values_np_array = np.array(metric_values)
        best_metrics[key] = float(metric_values_np_array[best_epoch_idx])

    return best_metrics


def extract_val_metrics(metrics):
    return {k: v for k, v in metrics.items() if k.startswith("val_")}


def print_best_epoch_metrics(metrics_dict, criterium="val_loss"):
    print(f"\nMODEL METRICS (model selected based on {criterium}):")
    for key, value in metrics_dict.items():
        if key == "epoch":
            print(f"Epoch: {value}")
        else:
            print(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")
    print()


def print_avg_metrics(results):
    first_key = next(iter(results))
    for metric in results[first_key].keys():
        avg_result = np.mean([res[metric] for res in results.values()])
        print(f"{metric.replace('_', ' ').capitalize()}: {avg_result:.4f}")


def print_avg_metrics_summary(all_val_results, all_test_results=None):
    print("\n===== AVERAGED RESULTS FROM MODELS TRAINED ON DIFFERENT FOLDS =====")

    print("VALIDATION:")
    print_avg_metrics(all_val_results)

    if all_test_results:
        print("TEST:")
        print_avg_metrics(all_test_results)

    print("===================================================================\n")

