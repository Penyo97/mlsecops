import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy import stats
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from constans import NOMINAL_COLUMNS, DISCRETE_COLUMNS, CONTINOUS_COLUMNS, DROP_COLUMNS, TARGET


class MLModel:
    def __init__(self):
        self.fill_values_nominal = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_nominal.pkl')
                                    if os.path.exists('artifacts/nan_outlier_handler/fill_values_nominal.pkl')
                                    else print('fill_values_nominal.pkl does not exist'))
        self.fill_values_discrete = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_discrete.pkl')
                                     if os.path.exists('artifacts/nan_outlier_handler/fill_values_discrete.pkl')
                                     else print('fill_values_discrete.pkl does not exist'))
        self.fill_values_continuous = (MLModel.load_model(
            'artifacts/nan_outlier_handler/fill_values_continuous.pkl')
                                       if os.path.exists('artifacts/nan_outlier_handler/fill_values_continuous.pkl')
                                       else print('fill_values_continuous.pkl does not exist'))
        self.min_max_scaler_dict = (MLModel.load_model(
            'artifacts/encoders/min_max_scaler_dict.pkl')
                                    if os.path.exists('artifacts/encoders/min_max_scaler_dict.pkl')
                                    else print('min_max_scaler_dict.pkl does not exist'))
        self.onehot_encoders = (MLModel.load_model(
            'artifacts/encoders/onehot_encoders_dict.pkl')
                                if os.path.exists('artifacts/encoders/onehot_encoders_dict.pkl')
                                else print('onehot_encoders_dict.pkl does not exist'))
        self.model = (MLModel.load_model(
            'artifacts/models/xgb_model.pkl')
                      if os.path.exists('artifacts/models/xgb_model.pkl')
                      else print('xgb_model.pkl does not exist'))


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
            df[col].fillna(fill_values_nominal[col], inplace=True)

        for col in DISCRETE_COLUMNS:
            df[col].fillna(fill_values_discrete[col], inplace=True)

        for col in CONTINOUS_COLUMNS:
            df[col].fillna(fill_values_continuous[col], inplace=True)

        # Outlier kezelés Z-score alapján
        for col in CONTINOUS_COLUMNS:
            z_scores = stats.zscore(df[col])
            outliers = np.abs(z_scores) > 3
            df.loc[outliers, col] = df[col].mean()

        # OneHot Encoding
        encoder_dict = {}
        for col in NOMINAL_COLUMNS:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            transformed = encoder.fit_transform(df[[col]])
            df = pd.concat([df, pd.DataFrame(transformed, columns=encoder.get_feature_names([col]))], axis=1)
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



        if TARGET in sample_data.columns:
            sample_data = sample_data.drop(columns=TARGET)
        return sample_data

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