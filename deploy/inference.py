import torch
import numpy as np
from torch import nn
import xarray as xr
from azure.storage.blob import BlobServiceClient
import pyodbc
import os
from azure.identity import DefaultAzureCredential
from azure.storage.queue import QueueClient
from azure.storage.filedatalake import DataLakeServiceClient
import json
import base64
import tempfile
import pandas as pd
from datetime import datetime
import re

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
    
def extract_timestamp(file_url):
    # Extract the filename from the URL
    filename = os.path.basename(file_url)
    # Find the last 12 digits before '.grib2'
    match = re.search(r'(\d{12})\.grib2$', filename)
    if match:
        return match.group(1)
    else:
        return None
    
def download_adls_file_from_message(message: dict, local_path: str):
    """
    Downloads a file from ADLS using managed identity, given an event message dict.
    The file is saved to local_path.
    """
    # Extract the ADLS file URL from the message
    adls_url = message["data"]["url"]
    # Parse storage account and file system from the URL
    # Example: https://cyrtdata.dfs.core.windows.net/weather/ecmwf/ifs/0p25/ifs_20250817_t06z_s012_0p25.grib2
    parts = adls_url.replace("https://", "").split(".dfs.core.windows.net/")
    account_name = parts[0]
    file_path = parts[1]
    file_system = file_path.split("/")[0]
    file_path_in_fs = "/".join(file_path.split("/")[1:])

    # Authenticate using managed identity
    credential = DefaultAzureCredential()
    service_client = DataLakeServiceClient(
        account_url=f"https://{account_name}.dfs.core.windows.net",
        credential=credential
    )
    file_system_client = service_client.get_file_system_client(file_system)
    file_client = file_system_client.get_file_client(file_path_in_fs)

    # Download the file
    with open(local_path, "wb") as f:
        download = file_client.download_file()
        f.write(download.readall())
    print(f"Downloaded {adls_url} to {local_path}")

# Example usage:
# message = { ... }  # your event message dict
# download_adls_file_from_message(message, "./downloaded_file.grib2")

def run_inference(local_path):
    # --- Prepare ds_forecast for inference ---
    ds_forecast_t2m = xr.open_dataset(
        local_path,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
        backend_kwargs={'indexpath': ''}
    )
    ds_forecast_other = xr.open_dataset(
        local_path,
        engine="cfgrib",
        backend_kwargs={'indexpath': ''},
        drop_variables=['t2m']  # Drop t2m since we handle it separately
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

    return probs[0]  # or whatever you want to append

    # Example DataFrame to collect results
results_df = pd.DataFrame(columns=["file", "probability", 'timestamp'])

print(results_df)

# From your setup
ACCOUNT = "cyrtdata"                       # <--- storage account name
QUEUE   = "forecast-data"                  # <--- your queue

cred = DefaultAzureCredential()
queue_url = f"https://{ACCOUNT}.queue.core.windows.net"

qc = QueueClient(account_url=queue_url, queue_name=QUEUE, credential=cred)

# Example: receive one message
msgs = qc.receive_messages(messages_per_page=32, visibility_timeout=600)
for m in msgs:
    # Azure Storage Queue messages are base64-encoded by default
    try:
        decoded_content = base64.b64decode(m.content).decode('utf-8')
        message = json.loads(decoded_content)
    except Exception as e:
        print(f"Failed to decode message: {e}")
        print(f"Raw message content: {repr(m.content)}")
        continue
    
    api_type = message.get("data", {}).get("api")
    if api_type == "FlushWithClose":
        file_url = message["data"]["url"]
        print(f"File location: {file_url}")
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.basename(file_url)
            local_path = os.path.join(temp_dir, filename)
            download_adls_file_from_message(message, local_path)
            prob = run_inference(local_path)
            new_row = {"file": filename, "probability": prob, "timestamp": extract_timestamp(file_url)}
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        print(f"Skipping message with unsupported API type: {api_type}")

print("Results DataFrame:")
print(results_df)


# connect to sqldb
connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:cyrt-server.database.windows.net,1433;Database=cyrt-db;Uid=PCSUSER;Pwd=welcome123!;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

conn = pyodbc.connect(connection_string)
print("Connected to SQL database")
cursor = conn.cursor()

# Insert results into SQL database
for index, row in results_df.iterrows():
    cursor.execute(
        "INSERT INTO dbo.forecast_results ([file], probability, [timestamp]) VALUES (?, ?, ?)",
        row['file'], row['probability'], row['timestamp']
    )
conn.commit()
print("Results inserted into SQL database")

for m in msgs:
    qc.delete_message(m)
    print(f"Deleted message: {m.id}")
