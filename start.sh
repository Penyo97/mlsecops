#!/bin/bash
mlflow server \
  --host 0.0.0.0 \
  --port 5102 \
  --backend-store-uri file:/app/mlruns \
  --default-artifact-root /app/mlartifacts &

sleep 5

python app.py