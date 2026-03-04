#!/usr/bin/env python3
"""
Preprocess ECCO4 ocean current data for SeaCurrentRNN training.

Downloads monthly ECCO4 surface velocity fields (EVEL, NVEL) from NASA's
WebDAV server, selects an Arctic region, upsamples spatially, computes
time derivatives, and saves the result as ``data/ECCO4.pkl``.

Requirements:
    pip install xarray requests netcdf4 torch

Credentials:
    Register at https://ecco.jpl.nasa.gov/ to get WebDAV credentials.
    Set ECCO_WEBDAV_USERNAME and ECCO_WEBDAV_PASSWORD environment variables.

Ported from AUTO-IceNav (scripts/dynamics/seanet_preprocess.py).

Usage:
    python scripts/preprocess_ecco4.py
"""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ECCO4"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WEBDAV_USERNAME = os.environ.get("ECCO_WEBDAV_USERNAME", "")
WEBDAV_PASSWORD = os.environ.get("ECCO_WEBDAV_PASSWORD", "")
WEBDAV_BASE_URL = os.environ.get(
    "ECCO_WEBDAV_BASE_URL",
    "https://ecco.jpl.nasa.gov/drive/files/Version4/Release4/interp_monthly",
)
DEFAULT_YEARS = tuple(range(1992, 2017))
MAX_CONCURRENT_DOWNLOADS = 12

OFFSET = 100
START_LAT = 85.0
START_LON = 0.05
TARGET_HEIGHT = 600
TARGET_WIDTH = 600


def download_monthly_velocity(
    component: str,
    *,
    years: Iterable[int] = DEFAULT_YEARS,
    months: Iterable[int] = range(1, 13),
    overwrite: bool = False,
) -> None:
    import requests

    if not WEBDAV_USERNAME or not WEBDAV_PASSWORD:
        raise RuntimeError(
            "WebDAV credentials not set. Export ECCO_WEBDAV_USERNAME / ECCO_WEBDAV_PASSWORD."
        )
    component = component.upper()
    base_url = f"{WEBDAV_BASE_URL.rstrip('/')}/{component}"
    jobs = []
    for year in years:
        for month in months:
            fn = f"{component}_{year}_{month:02d}.nc"
            url = f"{base_url}/{year}/{fn}"
            local = DATA_DIR / fn
            if local.exists() and not overwrite:
                continue
            jobs.append((url, local))
    if not jobs:
        print(f"All {component} files present; skipping download.")
        return

    def _dl(job):
        url, local = job
        print(f"  {url} -> {local.name}")
        tmp = local.with_suffix(local.suffix + ".tmp")
        with requests.Session() as s:
            s.auth = (WEBDAV_USERNAME, WEBDAV_PASSWORD)
            with s.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    print(f"  MISSING: {url}")
                    return
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
        tmp.replace(local)

    n_w = min(MAX_CONCURRENT_DOWNLOADS, len(jobs))
    with ThreadPoolExecutor(max_workers=n_w) as pool:
        futs = {pool.submit(_dl, j): j for j in jobs}
        for fut in as_completed(futs):
            fut.result()


def upsample_spatial(field_np, target_h, target_w):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(field_np).float().unsqueeze(0).unsqueeze(0)
    T = t.shape[2]
    t = F.interpolate(t, size=(T, target_h, target_w),
                       mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def main():
    import xarray as xr

    print("=== ECCO4 Preprocessing ===")

    if WEBDAV_USERNAME and WEBDAV_PASSWORD:
        print("Downloading EVEL...")
        download_monthly_velocity("EVEL")
        print("Downloading NVEL...")
        download_monthly_velocity("NVEL")
    else:
        print("WebDAV credentials not set; assuming files already in", DATA_DIR)

    def _ls(pat):
        m = sorted(DATA_DIR.glob(pat))
        if not m:
            raise FileNotFoundError(f"No files matching '{pat}' in {DATA_DIR}")
        return [str(p) for p in m]

    ds_u = xr.open_mfdataset(_ls("EVEL_*.nc"), combine="by_coords")
    ds_v = xr.open_mfdataset(_ls("NVEL_*.nc"), combine="by_coords")
    u_name = list(ds_u.data_vars)[0]
    v_name = list(ds_v.data_vars)[0]
    U = ds_u[u_name].isel(k=0)
    V = ds_v[v_name].isel(k=0)
    print(f"Loaded U{U.shape}, V{V.shape}")

    lat_range = (START_LAT, START_LAT + OFFSET)
    lon_range = (START_LON, START_LON + OFFSET)

    def _sel(field, lr, latr):
        ln = "lon" if "lon" in field.coords else "longitude"
        lt = "lat" if "lat" in field.coords else "latitude"
        if ln not in field.dims:
            field = field.swap_dims({field.coords[ln].dims[0]: ln})
        if lt not in field.dims:
            field = field.swap_dims({field.coords[lt].dims[0]: lt})
        return field.sel({ln: slice(*lr), lt: slice(*latr)})

    Ur = _sel(U, lon_range, lat_range)
    Vr = _sel(V, lon_range, lat_range)
    print(f"Regional: {Ur.shape}")

    print(f"Upsampling to ({TARGET_HEIGHT}, {TARGET_WIDTH})...")
    Uu = upsample_spatial(np.asarray(Ur.values), TARGET_HEIGHT, TARGET_WIDTH)
    Vu = upsample_spatial(np.asarray(Vr.values), TARGET_HEIGHT, TARGET_WIDTH)

    import torch
    Ut = torch.from_numpy(Uu)
    Vt = torch.from_numpy(Vu)
    du = Ut[1:] - Ut[:-1]
    dv = Vt[1:] - Vt[:-1]

    obs = np.stack([Ut[1:].numpy(), Vt[1:].numpy(),
                    du.numpy(), dv.numpy()], axis=1)
    print(f"Observation tensor: {obs.shape}  (T, [U,V,dU,dV], H, W)")

    out = Path(__file__).resolve().parent.parent / "data" / "ECCO4.pkl"
    with open(out, "wb") as f:
        pickle.dump(obs, f)
    print(f"Saved to {out}")

    x0 = torch.stack([Ut[0], Vt[0]], dim=0)   # (2, H, W)
    x0_hwc = x0.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)
    x0_path = Path(__file__).resolve().parent.parent / "data" / "ecco4_x0.pth"
    torch.save(x0_hwc, x0_path)
    print(f"Saved initial field to {x0_path}")


if __name__ == "__main__":
    main()
