NOMINAL_COLUMNS = [
    "Manufacturer",
    "Category",
    "Fuel type",
    "Gear box type",
    "Drive wheels",
    "Doors",
    "Wheel",
    "Color"
]

DISCRETE_COLUMNS = [
    "Prod. year",
    "Cylinders",
    "Airbags",
    "Leather interior"
]

CONTINOUS_COLUMNS = [
    "Levy",          # (előbb tisztítani kell, mivel string: pl. '-')
    "Engine volume", # (stringből float kell legyen)
    "Mileage"        # (szintén stringből szám: "123456 km")
]

DROP_COLUMNS = [
    "ID",    # azonosító, nincs prediktív ereje
    "Model"  # túl sok egyedi érték (1590 db), túl nagy dimenzió lenne
]

TARGET = 'Price'