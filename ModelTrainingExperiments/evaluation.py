import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def plot_training_history(history, save_path=None):
    loss = history.get("loss")
    val_loss = history.get("val_loss")
    acc = history.get("accuracy") or history.get("acc")
    val_acc = history.get("val_accuracy") or history.get("val_acc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(loss, label="Trénovacia strata")
    axes[0].plot(val_loss, label="Validačná strata")
    axes[0].set_title(f"Trénovacia a validačná strata")
    axes[0].set_xlabel("Počet epoch")
    axes[0].set_ylabel("Strata")
    axes[0].legend()

    # Accuracy
    axes[1].plot(acc, label="Trénovacia celková presnosť")
    axes[1].plot(val_acc, label="Validačná celková presnosť")
    axes[1].set_title(f"Trénovacia a validačná celková presnosť")
    axes[1].set_xlabel("Počet epoch")
    axes[1].set_ylabel("Celková presnosť")
    axes[1].legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_confusion_matrix(model, dataset, class_names, normalize=False, save_path=None):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(12, 12))  
    disp.plot(
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


def plot_roc_curve(model, dataset, class_names, save_path=None):
    y_true = []
    y_score = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_score.extend(preds)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
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

def compute_per_class_accuracy(model, dataset, class_names):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        results[name] = np.mean(y_pred[mask] == idx)

    return results
