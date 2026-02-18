import argparse
import pandas as pd
import geopandas as gpd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locations_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col", default="location_id")
    ap.add_argument("--lat_col", default="lat")
    ap.add_argument("--lon_col", default="lon")
    args = ap.parse_args()

    loc = pd.read_csv(args.locations_csv)
    gdf = gpd.GeoDataFrame(
        loc,
        geometry=gpd.points_from_xy(loc[args.lon_col], loc[args.lat_col]),
        crs="EPSG:4326"
    )

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[["continent", "geometry"]]
    joined = gpd.sjoin(gdf, world, how="left", predicate="within").drop(columns=["index_right"])

    out = joined.drop(columns=["geometry"]).copy()
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()