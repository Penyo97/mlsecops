from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import mlflow
from mlflow import MlflowClient
from datetime import datetime
from mlflow.exceptions import MlflowException
import os
import pandas as pd
from MLModel import MLModel

app = Flask(__name__)
api = Api(app, version='1.0', title='Petrik Ádám GVERV7')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5102")

experiment_name = "default_experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

client = MlflowClient()

try:
    obj_mlmodel = MLModel(client=client)
    if obj_mlmodel.model is None:
        print("""⚠️  Warning: No 'Staging' model found. 
              Training is still possible.""")
except Exception as e:
    print(f"""⚠️  Warning: Could not load 'Staging' model. 
          Training is still possible. Error: {e}""")
    obj_mlmodel = MLModel(client=client)


predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True,
                                 description='A row of data for inference')
})



file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV file for training')

ns = api.namespace('model', description='Model operations')


@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Invalid file type'}, 400

        data_path = 'data/car_price_prediction.csv'
        uploaded_file.save(data_path)

        try:

            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = "Car_Price_Model"

            with mlflow.start_run(run_name=run_name) as run:
                # Load and preprocess data
                df = pd.read_csv(data_path)
                input_example = df.drop(columns="Price").iloc[:1]  # Use a single row as an example
                signature = mlflow.models.infer_signature(df.drop(columns="Price"), df["Price"])
                df = obj_mlmodel.preprocessing_pipeline(df)


                mlflow.log_artifact(data_path, artifact_path="datasets")


                mae, r2, xgb = obj_mlmodel.train_and_save_model(df)

                # Log metrics to MLflow
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log the trained model with input example and signature
                mlflow.sklearn.log_model(
                    sk_model=xgb,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature
                )

                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

                # Transition model version to "Staging"
                mlflow_client = mlflow.tracking.MlflowClient()
                mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model_version.version,
                    stage="Staging"  # or Production
                )

                os.remove(data_path)

                return {'message': 'Model Trained and Transitioned to Staging Successfully',
                        'mae': mae,
                        'r2': r2}, 200

        except MlflowException as mfe:
            return {'message': 'MLflow Error', 'error': str(mfe)}, 500
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500

@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        try:
            data = request.get_json()
            if 'inference_row' not in data:
                return {'error': 'No inference_row found'}, 400

            infer_array = data['inference_row']
            if obj_mlmodel.model is None:
                return {'error': """No staging model is loaded. 
                         Train a model first."""}, 400

            # Set a unique name for the inference run based on timestamp
            run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                y_pred = obj_mlmodel.predict(infer_array)

                # Log input and output of inference to MLflow
                mlflow.log_param("inference_input", infer_array)
                mlflow.log_param("inference_output", y_pred)

            return {'message': 'Inference Successful', 'prediction': y_pred}, 200
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)