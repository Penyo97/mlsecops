import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy import stats
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from constans import NOMINAL_COLUMNS, DISCRETE_COLUMNS, CONTINOUS_COLUMNS, DROP_COLUMNS, TARGET
import mlflow
from mlflow.artifacts import download_artifacts


class MLModel:
    def __init__(self, client):
        self.client = client
        self.model = None
        self.fill_values_nominal = None
        self.fill_values_discrete = None
        self.fill_values_continuous = None
        self.min_max_scaler_dict = None
        self.onehot_encoders = None
        self.load_staging_model()


    def load_staging_model(self):

        try:
            latest_staging_model = None
            for model in self.client.search_registered_models():
                for latest_version in model.latest_versions:
                    if latest_version.current_stage == "Staging":
                        latest_staging_model = latest_version
                        break
                if latest_staging_model:
                    break

            if latest_staging_model:
                model_uri = latest_staging_model.source
                self.model = mlflow.sklearn.load_model(model_uri)
                print("Staging model loaded successfully.")

                # Load associated artifacts
                artifact_uri = latest_staging_model.source.rpartition('/')[0]
                self.load_artifacts(artifact_uri)
            else:
                print("No staging model found.")

        except Exception as e:
            print(f"Error loading model or artifacts: {e}")


    def load_artifacts(self, artifact_uri):

        try:
            # Load nominal fill values
            nominal_path = download_artifacts(artifact_uri=f"""
                         {artifact_uri}/fill_values_nominal.json""")
            with open(nominal_path, 'r') as f:
                self.fill_values_nominal = json.load(f)

            # Load discrete fill values
            discrete_path = download_artifacts(artifact_uri=f"""
                         {artifact_uri}/fill_values_discrete.json""")
            with open(discrete_path, 'r') as f:
                self.fill_values_discrete = json.load(f)

            # Load continuous fill values
            continuous_path = download_artifacts(artifact_uri=f"""
                         {artifact_uri}/fill_values_continuous.json""")
            with open(continuous_path, 'r') as f:
                self.fill_values_continuous = json.load(f)

            # Load MinMaxScaler dictionary
            scaler_path = download_artifacts(artifact_uri=f"""
                         {artifact_uri}/min_max_scaler_dict.pkl""")
            with open(scaler_path, 'rb') as f:
                self.min_max_scaler_dict = pickle.load(f)

            # Load OneHotEncoders
            encoders_path = download_artifacts(artifact_uri=f"""
                         {artifact_uri}/onehot_encoders.pkl""")
            with open(encoders_path, 'rb') as f:
                self.onehot_encoders = pickle.load(f)

            print("Artifacts loaded successfully.")

        except Exception as e:
            print(f"Error loading artifacts: {e}")


    def preprocessing_pipeline(self, df):
        folder = 'artifacts/encoders'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/preprocessed_data'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/models'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/nan_outlier_handler'
        MLModel.create_new_folder(folder)

        # Oszlopok törlése
        df.drop(columns=DROP_COLUMNS, inplace=True)

        # Levy, Engine volume és Mileage átalakítás
        df["Levy"] = df["Levy"].replace("-", np.nan)
        df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")

        df["Engine volume"] = df["Engine volume"].str.replace("Turbo", "", regex=False)
        df["Engine volume"] = pd.to_numeric(df["Engine volume"], errors="coerce")

        df["Mileage"] = df["Mileage"].astype(str).str.replace(" km", "").str.replace(",", "")
        df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")


        df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0})
        df['Doors'] = df['Doors'].str.extract(r'(\d+)').astype(float)

        fill_values_nominal = {col: df[col].mode()[0] for col in NOMINAL_COLUMNS}
        fill_values_discrete = {col: df[col].median() for col in DISCRETE_COLUMNS}
        fill_values_continuous = {col: df[col].mean() for col in CONTINOUS_COLUMNS}

        for col in NOMINAL_COLUMNS:
            df[col] = df[col].fillna(fill_values_nominal[col])

        for col in DISCRETE_COLUMNS:
            df[col] = df[col].fillna(fill_values_discrete[col])

        for col in CONTINOUS_COLUMNS:
            df[col] = df[col].fillna(fill_values_continuous[col])

        # Outlier kezelés Z-score alapján
        for col in CONTINOUS_COLUMNS:
            df[col] = df[col].astype(float)
            z_scores = stats.zscore(df[col])
            outliers = np.abs(z_scores) > 3
            df.loc[outliers, col] = df[col].mean()

        # OneHot Encoding
        encoder_dict = {}
        for col in NOMINAL_COLUMNS:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            transformed = encoder.fit_transform(df[[col]])
            df = pd.concat([df, pd.DataFrame(transformed, columns=encoder.get_feature_names_out([col]))], axis=1)
            encoder_dict[col] = encoder

        df.drop(columns=NOMINAL_COLUMNS, inplace=True)

        # Skálázás
        scaler_dict = {}
        for col in df.columns:
            if col == TARGET:
                continue
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scaler_dict[col] = scaler

        MLModel.save_model(self.fill_values_nominal,
                           'artifacts/nan_outlier_handler/fill_values_nominal.pkl')
        MLModel.save_model(self.fill_values_discrete,
                           'artifacts/nan_outlier_handler/fill_values_discrete.pkl')
        MLModel.save_model(self.fill_values_continuous,
                           'artifacts/nan_outlier_handler/fill_values_continuous.pkl')
        MLModel.save_model(self.min_max_scaler_dict,
                           'artifacts/encoders/min_max_scaler_dict.pkl')
        MLModel.save_model(self.onehot_encoders,
                           'artifacts/encoders/onehot_encoders_dict.pkl')

        return df

    def train_and_save_model(self,df):
        X = df.drop(columns=TARGET)
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb = XGBRegressor()
        xgb.fit(X_train, y_train)

        self.model = xgb

        mae, r2 = self.get_accuracy(X_test,y_test)

        return mae, r2, xgb

    def get_accuracy(self, X_test, y_test):
        print("\nXGB Regression:")
        y_train_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_train_pred, y_test)
        r2 = r2_score(y_train_pred, y_test)
        print("MAE:", mae)
        print("R²:", r2)

        return mae, r2

    def preprocessing_pipeline_inference(self, sample_data):
        sample_data = [np.nan if item == '?' else item for item in sample_data]
        sample_data = pd.DataFrame([sample_data])

        sample_data.drop(columns=DROP_COLUMNS, inplace=True)

        for col in NOMINAL_COLUMNS:
            sample_data[col].fillna(self.fill_values_nominal[col],
                                    inplace=True)
        for col in DISCRETE_COLUMNS:
            sample_data[col].fillna(self.fill_values_discrete[col],
                                    inplace=True)
        for col in CONTINOUS_COLUMNS:
            sample_data[col].fillna(self.fill_values_continuous[col],
                                    inplace=True)


        for col, encoder in self.onehot_encoders.items():
            new_data = encoder.transform(sample_data[col].to_numpy().reshape(-1, 1))
            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names([col]))
            sample_data = pd.concat([sample_data, new_df], axis=1).drop(columns=[col])

            # Skálázás
        for col, scaler in self.min_max_scaler_dict.items():
            if col in sample_data.columns:
                sample_data[col] = scaler.transform(sample_data[[col]])

        if TARGET in sample_data.columns:
            sample_data = sample_data.drop(columns=TARGET)
        return sample_data

    def predict(self, inference_row):
        if self.model is None:
            return {'error': 'No staging model is loaded'}, 400

        processed_data = self.preprocessing_pipeline_inference(inference_row)
        prediction = self.model.predict(processed_data)
        return int(prediction)

    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)



    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model

    @staticmethod
    def create_new_folder(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)