# Base Docker image: Anaconda
FROM continuumio/anaconda3

# Set working directory within the container
WORKDIR /app

# Copy environment.yml, app.py, and additional files
COPY environment.yml /app
COPY app.py /app
COPY constants.py /app
COPY MLModel.py /app
COPY mlruns /app/mlruns

# Install environment based on environment.yml
RUN conda env create -f environment.yml

# Activate the environment and install additional pip dependencies (flask-restx specifically)
RUN /bin/bash -c "source activate cubix_mlops_pipelines && pip install flask-restx"

# Set environment variables
ENV MLFLOW_TRACKING_URI="file:/app/mlruns"
ENV PATH="/opt/conda/envs/cubix_mlops_pipelines/bin:$PATH"

# Set permissions for the mlruns folder
RUN chmod -R 777 /app/mlruns

# Start MLflow server and app.py in the cubix_mlops_pipelines environment
CMD ["/bin/bash", "-c", "source activate cubix_mlops_pipelines && mlflow server --host 0.0.0.0 --port 5102 --backend-store-uri file:/app/mlruns --default-artifact-root /app/mlruns & python app.py"]

#docker build -t pipelines_with_mlflow_docker .    

#conda env export --name környezet_neve > environment.yml

#docker build -t mlflow_docker_image .

#docker run --name self_container_name -p 5102:5102 -p 8080:8080 mlflow_docker_image

#docker exec -it my_cont2 /bin/bash
