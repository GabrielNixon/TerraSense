import argparse
import pandas as pd
import folium


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--locations_csv", required=True)
    p.add_argument("--out_html", default="locations_map.html")
    args = p.parse_args()

    df = pd.read_csv(args.locations_csv)

    if not {"id", "name", "lat", "lon"}.issubset(df.columns):
        raise RuntimeError("CSV must contain columns: id, name, lat, lon")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # Center map at global mean
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="cartodbpositron")

    for _, row in df.iterrows():
        popup_text = f"""
        <b>ID:</b> {row['id']}<br>
        <b>Name:</b> {row['name']}<br>
        <b>Lat:</b> {row['lat']}<br>
        <b>Lon:</b> {row['lon']}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="blue",
            fill=True,
            fill_color="red",
            fill_opacity=0.7,
            popup=popup_text,
        ).add_to(m)

    m.save(args.out_html)

    print("Saved map to:", args.out_html)
    print("Total locations plotted:", len(df))


if __name__ == "__main__":
    main()
