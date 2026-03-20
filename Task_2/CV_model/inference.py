import tensorflow as tf
import numpy as np
import argparse
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_and_predict(image_path, model_path):
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'preprocess_input': preprocess_input}
    )

    with open('../class_names.json', 'r') as f:
        class_names = json.load(f)

    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    return class_names[class_idx], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str, default='./animal_classifier.h5')
    args = parser.parse_args()

    label, conf = load_and_predict(args.image, args.model)
    print(f"Predicted: {label} (Confidence: {conf:.2f})")