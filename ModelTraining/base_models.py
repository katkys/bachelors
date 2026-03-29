from tensorflow import keras


INPUT_SIZES = {'InceptionV3': (299, 299),
                'Resnet50V2': (224, 224),
                'EfficientNetV2B0' : (224, 224),
                'VGG16' : (224, 224),
                'MobileNetV2' : (224, 224)}

def get_input_size(model_name):
    if model_name not in INPUT_SIZES:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return INPUT_SIZES[model_name]

def get_supported_models():
    return list(INPUT_SIZES.keys())

def get_model_and_preprocess_function(model_name):
    img_size = get_input_size(model_name)
    input_shape=(img_size[0], img_size[1], 3)
    
    if model_name == "InceptionV3":
        from keras.applications.inception_v3 import InceptionV3, preprocess_input
        return InceptionV3(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False), preprocess_input
    elif model_name == "EfficientNetV2B0":
        from keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
        return EfficientNetV2B0(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False), preprocess_input
    elif model_name == "ResNet50V2":
        from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        return ResNet50V2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False), preprocess_input
    elif model_name == "VGG16":
        from keras.applications.vgg16 import VGG16, preprocess_input
        return VGG16(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False), preprocess_input    
    elif model_name == 'MobileNetV2':
        from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        return MobileNetV2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False), preprocess_input
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    