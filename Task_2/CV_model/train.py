import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
import json


def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(256, 256, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(256, 256, 3)),    
        layers.Rescaling(1./127.5, offset=-1.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main(args):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        seed=42,
        image_size=(256, 256),
        batch_size=args.batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        seed=42,
        image_size=(256, 256),
        batch_size=args.batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    with open('../class_names.json', 'w') as f:
        json.dump(class_names, f)

    model = build_model(num_classes)

    print("\n--- Starting Training ---")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    model.save(args.model_path)
    print(f"\nModel saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to Training Data', default='./data/Training Data/Training Data')
    parser.add_argument('--val_dir', type=str, help='Path to Validation Data', default='./data/Validation Data/Validation Data')
    parser.add_argument('--model_path', type=str, default='./animal_classifier.h5')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    main(args)