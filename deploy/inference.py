import torch
import numpy as np
from torch import nn
import xarray as xr
from azure.storage.blob import BlobServiceClient
import pyodbc
import os

class WeatherLanding2DNet(nn.Module):
    def __init__(self, in_ch, base=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base*2, 1)  # logit
        )
    def forward(self, x):  # x: (B, C, H, W)
        z = self.features(x)
        return self.head(z).squeeze(1)  # (B,)



# --- Azure ADLS download ---
ADLS_CONNECTION_STRING = os.getenv("ADLS_CONNECTION_STRING", "<your_adls_connection_string>")
ADLS_CONTAINER = os.getenv("ADLS_CONTAINER", "<your_container>")
ADLS_BLOB_NAME = os.getenv("ADLS_BLOB_NAME", "realtime_forecast.grib2")
LOCAL_GRIB2_PATH = "realtime_forecast.grib2"

blob_service_client = BlobServiceClient.from_connection_string(ADLS_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(ADLS_CONTAINER)
with open(LOCAL_GRIB2_PATH, "wb") as f:
    blob_data = container_client.download_blob(ADLS_BLOB_NAME)
    f.write(blob_data.readall())

# --- Prepare ds_forecast for inference ---
ds_forecast_t2m = xr.open_dataset(
    LOCAL_GRIB2_PATH,
    engine="cfgrib",
    filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2}
)
ds_forecast_other = xr.open_dataset(
    LOCAL_GRIB2_PATH,
    engine="cfgrib"
)
ds_forecast_t2m = ds_forecast_t2m.sel(
    latitude=slice(66, 59),
    longitude=slice(-101, -82)
)
ds_forecast_other = ds_forecast_other.sel(
    latitude=slice(66, 59),
    longitude=slice(-101, -82)
)
t2m = ds_forecast_t2m['t2m'].values
u10 = ds_forecast_other['u10'].values
v10 = ds_forecast_other['v10'].values
msl = ds_forecast_other['msl'].values
X_predict = np.stack([t2m, u10, v10, msl], axis=1)
X_predict = np.expand_dims(X_predict, axis=0)
X_predict = np.transpose(X_predict, (0, 2, 1, 3))

# Load the bundle
bundle = torch.load("deploy/weather_landing_bundle.pth", weights_only=False)
mu = bundle["mu"]
sd = bundle["sd"]
variables = bundle["variables"]

# Now normalization will broadcast correctly
Xn_predict = (X_predict - mu) / sd

# Convert to PyTorch tensor
Xn_predict_tensor = torch.tensor(Xn_predict, dtype=torch.float32)

# Load model
in_ch = bundle["in_ch"]
base = bundle["base"]
model = WeatherLanding2DNet(in_ch=in_ch, base=base)
model.load_state_dict(bundle["state_dict"])
model.eval()

# Run inference
with torch.no_grad():
    logits = model(Xn_predict_tensor)
    probs = torch.sigmoid(logits).numpy()


# --- Write output to Azure SQL ---
AZURE_SQL_CONN_STR = os.getenv("AZURE_SQL_CONN_STR", "DRIVER={ODBC Driver 17 for SQL Server};SERVER=<your_server>;DATABASE=<your_db>;UID=<your_user>;PWD=<your_password>")

try:
    conn = pyodbc.connect(AZURE_SQL_CONN_STR)
    cursor = conn.cursor()
    # Example: create table if not exists
    cursor.execute("""
        IF OBJECT_ID('dbo.InferenceResults', 'U') IS NULL
        CREATE TABLE dbo.InferenceResults (
            id INT IDENTITY(1,1) PRIMARY KEY,
            probability FLOAT,
            timestamp DATETIME DEFAULT GETDATE()
        )
    """)
    conn.commit()
    # Insert probabilities
    for prob in probs:
        cursor.execute("INSERT INTO dbo.InferenceResults (probability) VALUES (?)", float(prob))
    conn.commit()
    print("Probabilities written to Azure SQL.")
except Exception as e:
    print(f"Azure SQL write failed: {e}")
finally:
    try:
        conn.close()
    except:
        pass