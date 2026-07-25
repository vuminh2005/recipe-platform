FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV KERAS_HOME=/opt/keras

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

# Cache sẵn đúng MobileNetV2 weights mà notebook sử dụng.
RUN python -c "\
from tensorflow.keras.applications import MobileNetV2; \
MobileNetV2(\
input_shape=(224,224,3), \
include_top=False, \
weights='imagenet'\
)"

COPY trainer ./trainer

ENTRYPOINT ["python", "-m", "trainer.train"]
