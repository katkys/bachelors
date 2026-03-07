import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import evaluation as eval

MODEL_PATH = "" # Path to saved model (.keras file)
TEST_DATASET_DIR = "" # Path to directory containing test dataset (images organized in artist folders)
OUTPUT_DIR = "" # Path to directory where evaluation results (metrics, confusion matrix, ROC curves) will be saved

IMG_SIZE = (299, 299)  
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading saved model...")
model = tf.keras.models.load_model(MODEL_PATH)


print("Loading test dataset...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  
)

class_names = test_dataset.class_names

test_loss, test_acc = model.evaluate(test_dataset)

print("\n===== TEST RESULTS =====")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss:     {test_loss:.4f}")
print("=========================\n")

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")

# Confusion matrix 
eval.plot_confusion_matrix(
    model=model,
    dataset=test_dataset,
    class_names=class_names,
    normalize=False,
    save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
)

# Confusion matrix (normalized)
eval.plot_confusion_matrix(
    model=model,
    dataset=test_dataset,
    class_names=class_names,
    normalize=True,
    save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png")
)

# ROC curves (one-vs-rest)
eval.plot_roc_curve(
    model=model,
    dataset=test_dataset,
    class_names=class_names,
    save_path=os.path.join(OUTPUT_DIR, "roc_curves.png")
)

print(f"Evaluation results saved to: {OUTPUT_DIR}")