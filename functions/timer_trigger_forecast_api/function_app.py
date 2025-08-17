import os
import logging
import datetime as dt
import tempfile
from pathlib import Path
from math import floor
from datetime import datetime, timezone

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError

from ecmwf.opendata import Client

app = func.FunctionApp()

def _get_fs_and_dir():
    account_url = os.environ["ADLS_ACCOUNT_URL"]
    filesystem  = os.environ["ADLS_FILESYSTEM"]
    base_dir    = os.environ.get("ADLS_DIR", "")
    cred = DefaultAzureCredential()
    svc = DataLakeServiceClient(account_url=account_url, credential=cred)
    fs  = svc.get_file_system_client(filesystem)
    if base_dir:
        d = fs.get_directory_client(base_dir)
        try:
            d.create_directory()
        except ResourceExistsError:
            pass
        return fs, d
    else:
        return fs, fs.get_directory_client("")

def _upload_to_adls(dir_client, local_path: Path, remote_name: str):
    file_client = dir_client.get_file_client(remote_name)
    with open(local_path, "rb") as f:
        file_client.upload_data(f, overwrite=True)
    logging.info("Uploaded %s (%s bytes) to %s/%s",
                 remote_name, local_path.stat().st_size,
                 dir_client.file_system_name, dir_client.path_name or "/")

def snap_to_valid_step(run_hour, lead_h):
    # HRES rules (docs)
    if run_hour in (0, 12):
        valid = list(range(0, 145, 3)) + list(range(150, 241, 6))
    else:  # 06/18 UTC
        valid = list(range(0, 91, 3))
    return max([s for s in valid if s <= lead_h] or [0])

@app.timer_trigger(schedule="0 0 * * * *", arg_name="myTimer", run_on_startup=False, use_monitor=False)
def timer_trigger_forecast_api(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info("The timer is past due!")

    client = Client(source="ecmwf", model="ifs", resol="0p25")

    try:
        run_dt = client.latest(type="fc")  # e.g., 2025-08-17 12:00:00 UTC
        run_dt = client.latest(type="fc")
        if run_dt.tzinfo is None:
            run_dt = run_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        lead = max(0, round((now - run_dt).total_seconds() / 3600))
        step   = snap_to_valid_step(run_dt.hour, lead)
        logging.info("Latest run %s, lead=%d, chosen step=%d", run_dt, lead, step)
    except Exception as e:
        logging.exception("Failed to determine latest run/step: %s", e)
        return

    with tempfile.TemporaryDirectory() as tdir:
        tmp = Path(tdir) / "latest_now.grib2"
        try:
            client.retrieve(
                date=run_dt.date(),
                time=run_dt.hour,
                type="fc",
                step=step,
                param=["2t","msl","10u","10v"],
                target=str(tmp),
            )
            logging.info("Downloaded forecast %s step %d -> %s", run_dt, step, tmp)
        except Exception as e:
            logging.exception("ECMWF download failed: %s", e)
            return
        
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        remote_name = f"ifs_{run_dt.strftime('%Y%m%d')}_t{run_dt.hour:02d}z_s{step:03d}_0p25_{timestamp}.grib2"

        try:
            _, dir_client = _get_fs_and_dir()
            _upload_to_adls(dir_client, tmp, remote_name)
        except Exception as e:
            logging.exception("ADLS upload failed: %s", e)
            return

    logging.info("Done.")
