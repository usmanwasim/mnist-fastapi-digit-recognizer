#!/usr/bin/env bash

# Auto-train if missing
if [ ! -f model/mnist_lr.joblib ]; then
    echo "Training model..."
    python train.py
fi

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port $PORT
