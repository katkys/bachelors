import pickle
import evaluation as eval 

HISTORY_PATH = "" #path to training history before fine-tuning
HISTORY_FT_PATH = "" #path to fine-tuning training history (can be None/empty if model was not fine-tuned)

with open(HISTORY_PATH, "rb") as file:
    history = pickle.load(file)

if HISTORY_FT_PATH is not None and HISTORY_FT_PATH.strip() != "":
    with open(HISTORY_FT_PATH, "rb") as file:
        history_ft = pickle.load(file)

    combined_history = {}
    for k in history.keys():
        combined_history[k] = history[k] + history_ft[k]
else:
    print("\nWarning: No fine-tuning history provided. Best epoch metrics will be based on pre-fine-tuning history only.")
    combined_history = history

best_epoch_criterium = "val_f1_score"
metrics = eval.get_best_epoch_metrics(combined_history, criterium=best_epoch_criterium)
eval.print_best_epoch_metrics(metrics, criterium=best_epoch_criterium)

eval.plot_training_history(combined_history)