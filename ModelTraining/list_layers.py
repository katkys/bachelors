import tensorflow as tf
from tensorflow import keras
import base_models as bm

model_name = input("\nEnter the name of the model whose layers you want to list: ").strip()

supported_models = bm.get_supported_models()
if model_name not in supported_models:
    raise ValueError(f"Model '{model_name}' is not supported. Available models: {supported_models}")
    

base_model = bm.get_model_and_preprocess_function(model_name)[0]

print(f"\nTotal number of layers: {len(base_model.layers)}")
print(f"List of layers:")
for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)