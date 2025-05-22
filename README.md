# mlsecops

{
  "inference_row": [
    {
      "ID": 45654403,
      "Levy": 1399,
      "Manufacturer": "LEXUS",
      "Model": "RX 450",
      "Prod. year": 2010,
      "Category": "Jeep",
      "Leather interior": "Yes",
      "Fuel type": "Hybrid",
      "Engine volume": 3.5,
      "Mileage": "186005 km",
      "Cylinders": 6.0,
      "Gear box type": "Automatic",
      "Drive wheels": "4x4",
      "Doors": "04-May",
      "Wheel": "Left wheel",
      "Color": "Silver",
      "Airbags": 12
    }
  ]
}

## Docker image build
docker build -t flask-mlflow-app . 
docker run -p 8080:8080 -p 5102:5102 flask-mlflow-app 

## mlflow start
mlflow ui --port 5102

## Conda set enviroment
conda activate mlflow_env


