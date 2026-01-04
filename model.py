# %%
import os
from datetime import date
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
import cdsapi
import xarray as xr
import matplotlib.pyplot as plt
from ecmwf.opendata import Client

load_dotenv()

ecmwf_api_key = os.getenv("ECMWF_API_KEY")

# config for local computer
config = f"""url: https://cds.climate.copernicus.eu/api
key: {ecmwf_api_key}
"""

home = os.path.expanduser("~")
config_path = os.path.join(home, ".cdsapirc")

with open(config_path, "w") as f:
    f.write(config)

print(f"Config written to {config_path}")

# %%
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
    ],
    "year": ["2022", "2023", "2024"],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "day": [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
    ],
    "time": [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [66, -101, 59, -82],
}

client = cdsapi.Client()
output_path = "train.grib"  # Change to your preferred file name
client.retrieve(dataset, request).download(output_path)

# %%
