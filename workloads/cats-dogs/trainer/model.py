from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation_model() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_model(
    *,
    image_size: int,
    learning_rate: float,
    dropout_rate: float,
    dense_units: int,
    trainable_backbone: bool = False,
) -> keras.Model:
    inputs = keras.Input(
        shape=(image_size, image_size, 3),
        name="image",
    )

    x = build_augmentation_model()(inputs)

    x = keras.applications.mobilenet_v2.preprocess_input(x)

    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )

    backbone.trainable = trainable_backbone

    # Giống notebook: kể cả khi fine-tune, BatchNorm vẫn chạy
    # inference mode để ổn định hơn với dataset nhỏ.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="dog_probability",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cats_dogs_mobilenetv2",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model
