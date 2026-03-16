import pickle
import evaluation as eval 

HISTORY_PATH = "" #path to training history before fine-tuning
HISTORY_FT_PATH = "" #path to fine-tuning training history

with open(HISTORY_PATH, "rb") as file:
    history = pickle.load(file)

with open(HISTORY_FT_PATH, "rb") as file:
    history_ft = pickle.load(file)

combined_history = {}
for k in history.keys():
    combined_history[k] = history[k] + history_ft[k]

best_epoch_criterium = "val_f1_score" #or val_loss / val_accuracy ...
metrics = eval.get_best_epoch_metrics(combined_history, criterium=best_epoch_criterium)

print(f"\nMODEL METRICS (model selected based on {best_epoch_criterium}):")
print(f"Epoch: {metrics['epoch']}")
print(f"Validation loss: {metrics['val_loss']:.4f}")
print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
print(f"Validation precision: {metrics['val_precision']:.4f}")
print(f"Validation recall: {metrics['val_recall']:.4f}")
print(f"Validation F1-score: {metrics['val_f1']:.4f}")
print(f"Training loss: {metrics['train_loss']:.4f}")
print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
print(f"Training precision: {metrics['train_precision']:.4f}")
print(f"Training recall: {metrics['train_recall']:.4f}")
print(f"Training F1-score: {metrics['train_f1']:.4f}")

eval.plot_training_history(combined_history)
