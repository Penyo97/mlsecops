# mlsecops
## Petrik Ádám GVERV7


## Példa infenerce sor
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

## Egy új conda környezett lett létrehozva az mlflow telepitése miatt, mert a base enviroment túl sok package-et tartalmazott
## Conda set enviroment
conda activate mlflow_env

## Docker image build
docker build -t flask_mlflow_app .

## Amint megvolt az anaconda enviroment change utána tudjuk futattni magát az alkalmazást 
docker compose up

## Stremlit start
python -m streamlit run monitor_with_streamlit_train_data.py

