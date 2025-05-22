FROM python:3.10-slim

# Set working directory within the container
WORKDIR /app

# Copy environment.yml, app.py, and additional files
COPY app.py /app
COPY constants.py /app
COPY MLModel.py /app
COPY mlruns /app/mlruns
COPY requirements.txt /app
COPY start.sh /app/start.sh


# Set permissions for the mlruns folder
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN chmod +x /app/start.sh

ENV MLFLOW_TRACKING_URI="file:/app/mlruns"


CMD ["/app/start.sh"]

