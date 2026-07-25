from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
NUM_CHANNELS = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_TO_ID = {"cats": 0, "dogs": 1}
ID_TO_CLASS = {0: "cat", 1: "dog"}


def download_dataset(
    dataset_uri: str,
    target_path: Path,
) -> Path:
    parsed = urlparse(dataset_uri)

    if parsed.scheme != "s3":
        raise ValueError(
            "DATASET_URI phải có dạng s3://bucket/key"
        )

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ[
            "AWS_SECRET_ACCESS_KEY"
        ],
        region_name=os.getenv("AWS_DEFAULT_REGION", "auto"),
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)

    client.download_file(
        bucket,
        key,
        str(target_path),
    )

    return target_path


def prepare_dataset(
    dataset_uri: str,
    work_dir: str = "/tmp/cats-dogs",
) -> Path:
    root = Path(work_dir)
    zip_path = root / "data.zip"
    extracted_dir = root / "data"

    root.mkdir(parents=True, exist_ok=True)

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    download_dataset(dataset_uri, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extracted_dir)

    return extracted_dir


def collect_image_paths(root_dir: Path) -> pd.DataFrame:
    rows = []

    for class_folder, class_id in CLASS_TO_ID.items():
        class_dir = root_dir / class_folder

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {class_dir}"
            )

        for image_path in sorted(class_dir.rglob("*")):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                rows.append(
                    {
                        "filepath": str(image_path),
                        "label": class_id,
                        "label_name": ID_TO_CLASS[class_id],
                        "filename": image_path.name,
                    }
                )

    return (
        pd.DataFrame(rows)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )


def stratified_split(
    df: pd.DataFrame,
    validation_size: float,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=seed,
    )

    train_idx, val_idx = next(
        splitter.split(df["filepath"], df["label"])
    )

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def get_class_weights(
    df: pd.DataFrame,
) -> dict[int, float]:
    classes = np.array(sorted(df["label"].unique()))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df["label"].values,
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }


def decode_and_resize(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=NUM_CHANNELS,
        expand_animations=False,
    )

    image.set_shape([None, None, NUM_CHANNELS])
    image = tf.image.resize(
        image,
        (image_size, image_size),
    )
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.float32)


def make_dataset(
    df: pd.DataFrame,
    *,
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    paths = df["filepath"].astype(str).values
    labels = df["label"].astype("float32").values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: decode_and_resize(
            path,
            label,
            image_size,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return (
        dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
