import pickle
import evaluation as eval 

HISTORY_PATH = "" # Path to saved model history (.pkl file)

with open(HISTORY_PATH, "rb") as file:
    history = pickle.load(file)

metrics = eval.get_best_epoch_metrics(history)

print("\n===== MODEL METRICS (model selected based on min val loss) =====")
print(f"Epoch: {metrics['epoch']}")
print(f"Validation loss: {metrics['val_loss']:.4f}")
print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
print(f"Training loss: {metrics['train_loss']:.4f}")
print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
print("================================================================")