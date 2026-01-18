#!/usr/bin/env bash

if [ ! -f model/mnist_cnn.h5 ]; then
  echo "Training model..."
  python train.py
fi

uvicorn main:app --host 0.0.0.0 --port $PORT
