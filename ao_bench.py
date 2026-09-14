"""xarray benchmark: a year of gridded weather, processed week by week.

    python ao_bench.py                # 52 weeks over a 1460 x 180 x 360 float32 cube
    python ao_bench.py --weeks 12 --lat 90 --lon 180

Builds a synthetic 6-hourly temperature cube with a diurnal/seasonal signal,
computes the climatology once, and then for each week: selects the slice,
removes the climatology, smooths along latitude with a rolling mean, coarsens
longitude, aggregates by latitude band, and writes the result as NetCDF. The
loop over weeks is the unit of work. Writes out/summary.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))


def build_cube(steps, nlat, nlon, seed=0):
    rng = np.random.default_rng(seed)
    time_ = pd.date_range("2025-01-01", periods=steps, freq="6h")
    lat = np.linspace(-89.5, 89.5, nlat, dtype=np.float32)
    lon = np.linspace(0.5, 359.5, nlon, dtype=np.float32)
    hours = np.arange(steps, dtype=np.float32) * 6.0
    seasonal = 15.0 * np.sin(2 * np.pi * hours / (365.0 * 24))[:, None, None]
    diurnal = 4.0 * np.sin(2 * np.pi * hours / 24.0)[:, None, None]
    latitude = (30.0 - 0.6 * np.abs(lat))[None, :, None]
    noise = rng.standard_normal((steps, nlat, nlon), dtype=np.float32) * 2.0
    t2m = (seasonal + diurnal + latitude + noise).astype(np.float32)
    return xr.Dataset({"t2m": (("time", "lat", "lon"), t2m)},
                      coords={"time": time_, "lat": lat, "lon": lon})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1460, help="6-hourly steps (1460 = one year)")
    ap.add_argument("--lat", type=int, default=180)
    ap.add_argument("--lon", type=int, default=360)
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    ds = build_cube(args.steps, args.lat, args.lon)
    clim = ds.t2m.mean("time")
    print(f"cube: {dict(ds.sizes)} float32 {ds.t2m.nbytes / 1e6:.0f} MB, "
          f"climatology ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    steps_per_week = 28  # 7 days x 4 steps
    rows, walls = [], []
    for week in range(args.weeks):
        t0 = time.perf_counter()
        sl = ds.isel(time=slice(week * steps_per_week, (week + 1) * steps_per_week))
        anom = sl.t2m - clim
        smooth = anom.rolling(lat=5, center=True, min_periods=1).mean()
        coarse = smooth.coarsen(lon=4, boundary="trim").mean()
        bands = coarse.groupby_bins("lat", 18).mean(dim=["lat", "time"])
        daily_max = anom.resample(time="1D").max()
        hot = (daily_max > 3.0).sum(dim=["lat", "lon"])
        result = xr.Dataset({"anomaly_by_band": bands, "hot_cells_per_day": hot,
                             "week_mean": anom.mean(dim=["time", "lat", "lon"])})
        result.to_netcdf(os.path.join(args.out, f"week_{week:02d}.nc"))
        dt = time.perf_counter() - t0
        walls.append(dt)
        rows.append({"week": week, "s": round(dt, 4),
                     "week_mean": float(result.week_mean),
                     "hot_total": int(hot.sum())})
        print(f"  week {week:2d} {dt:.3f}s mean anomaly {float(result.week_mean):+.3f} "
              f"hot cells {int(hot.sum())}", flush=True)

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"weeks": rows, "total_s": sum(walls),
                   "median_s": float(np.median(walls))}, fh, indent=1)
    print(f"done: {len(walls)} weeks, median {np.median(walls):.3f}s, total {sum(walls):.2f}s")


if __name__ == "__main__":
    main()
