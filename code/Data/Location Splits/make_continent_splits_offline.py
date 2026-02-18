import argparse
import pandas as pd
import numpy as np

def continent_from_latlon(lat, lon):
    lat = float(lat); lon = float(lon)

    if lat < -60:
        return "Antarctica"

    if (-170 <= lon <= -30) and (-60 <= lat <= 80):
        return "North America" if lat >= 15 else "South America"

    if (-30 < lon < 60) and (-40 <= lat <= 75):
        return "Europe" if lat >= 35 else "Africa"

    if (60 <= lon <= 180) and (-10 <= lat <= 80):
        return "Asia"

    if (110 <= lon <= 180) and (-50 <= lat < -10):
        return "Oceania"

    if (-30 <= lon <= 60) and (-60 <= lat < 35):
        return "Africa"

    return "Unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col", default="location_id")
    ap.add_argument("--lat_col", default="lat")
    ap.add_argument("--lon_col", default="lon")
    args = ap.parse_args()

    loc = pd.read_csv(args.locations_csv)
    loc["continent"] = [
        continent_from_latlon(lat, lon)
        for lat, lon in zip(loc[args.lat_col], loc[args.lon_col])
    ]
    loc.to_csv(args.out_csv, index=False)

    vc = loc["continent"].value_counts(dropna=False)
    print(vc.to_string())

if __name__ == "__main__":
    main()