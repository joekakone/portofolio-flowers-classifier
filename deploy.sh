# /usr/bin/bash

docker stop tf_serving_flowers

docker rm tf_serving_flowers

MSYS_NO_PATHCONV=1 docker run -d --name tf_serving_flowers \
  -p 8501:8501 \
  -v "$(pwd)/models/flower_classifier:/models/flower_classifier" \
  -e MODEL_NAME=flower_classifier \
  tensorflow/serving
