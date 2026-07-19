# -*- coding: utf-8 -*-
"""
Org Site Selection Dashboard — Traffic Store client.

Unlike the live transit twin / operator sim console, this tool doesn't poll
anything continuously. It queries the Traffic Store REST API (the product
described on the org marketing page) on demand, filtered by region /
time-of-day / day-type, and lets a business client (retail, site selection,
logistics) rank and compare H3 cells by footfall metrics.

ASSUMED API CONTRACT — align with your actual Traffic Store endpoints:

    GET {API_BASE_URL}/api/v1/h3/discharge
        query params: bbox=minLon,minLat,maxLon,maxLat, res, hour, day_type
        auth: Authorization: Bearer {API_KEY}
        returns: [{ "h3_cell": str, "discharge_volume": int,
                     "walking_to_waiting_ratio": float,
                     "node_saturation": float }, ...]

    The per-cell / per-node endpoints on the marketing page
    (GET /api/v1/nodes/{id}/saturation, GET /api/v1/h3/{cell_id}/metrics)
    don't scale to "rank every cell in a region" — that needs a bulk/bbox
    query. If no such bulk endpoint exists yet, this is the one to build
    server-side; the alternative (fan out N single-cell requests from here)
    defeats the point of a "queryable inventory."
"""
import os

import h3
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(layout="wide", page_title="Site Selection — Traffic Store", page_icon=":round_pushpin:")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
API_BASE_URL = os.environ.get("TRAFFIC_STORE_API_URL", "http://localhost:8080")
API_KEY = os.environ.get("TRAFFIC_STORE_API_KEY", "")
CACHE_TTL_SECONDS = 300  # Traffic Store data isn't second-by-second; avoid hammering the API on every widget tweak

# Rough Nairobi-area bounding boxes — placeholders, refine to your actual node/corridor boundaries
REGIONS = {
    "CBD":        (36.815, -1.295, 36.835, -1.275),
    "Westlands":  (36.790, -1.275, 36.815, -1.255),
    "Eastleigh":  (36.840, -1.285, 36.865, -1.265),
    "Rongai":     (36.720, -1.410, 36.760, -1.370),
    "Kikuyu":     (36.640, -1.260, 36.680, -1.230),
    "Thika Road": (36.850, -1.230, 36.900, -1.170),
    "Ngong Road": (36.740, -1.320, 36.800, -1.290),
}

METRIC_LABELS = {
    "discharge_volume": "Discharge Volume (passengers)",
    "walking_to_waiting_ratio": "Walking-to-Waiting Ratio",
    "node_saturation": "Node Saturation",
}


# -----------------------------------------------------------------------------
# API CLIENT
# -----------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_discharge_data(bbox: tuple, res: int, hour: int, day_type: str) -> tuple[pd.DataFrame, str | None]:
    """
    Returns (df, error). df columns: h3_cell, discharge_volume,
    walking_to_waiting_ratio, node_saturation, lat, lon.
    """
    url = f"{API_BASE_URL}/api/v1/h3/discharge"
    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "res": res,
        "hour": hour,
        "day_type": day_type,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Traffic Store API error: {e}"

    if not rows:
        return pd.DataFrame(), None

    df = pd.DataFrame(rows)
    for col in ("discharge_volume", "walking_to_waiting_ratio", "node_saturation"):
        if col not in df.columns:
            df[col] = None

    lats, lons = [], []
    for cell in df["h3_cell"]:
        try:
            lat, lon = h3.cell_to_latlng(cell)
        except (h3.H3CellError, ValueError):
            lat, lon = None, None
        lats.append(lat)
        lons.append(lon)
    df["lat"] = lats
    df["lon"] = lons
    df = df.dropna(subset=["lat", "lon"])
    return df, None


# -----------------------------------------------------------------------------
# MAP
# -----------------------------------------------------------------------------
def draw_footfall_map(df: pd.DataFrame, metric: str, bbox: tuple):
    if df.empty:
        center_lat, center_lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    else:
        center_lat, center_lon = df["lat"].mean(), df["lon"].mean()

    max_val = df[metric].max() if not df.empty and df[metric].notna().any() else 1

    return pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state={"latitude": center_lat, "longitude": center_lon, "zoom": 13, "pitch": 45},
        layers=[
            pdk.Layer(
                "H3HexagonLayer",
                data=df,
                get_hexagon="h3_cell",
                get_fill_color=f"[255, 140 - 140 * ({metric} / {max_val if max_val else 1}), 60, 180]",
                get_elevation=metric,
                elevation_scale=5 if metric == "discharge_volume" else 200,
                extruded=True,
                pickable=True,
            ),
        ],
        tooltip={"text": "{h3_cell}\n" + metric + ": {" + metric + "}"},
    )


# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------
st.title("Site Selection — Traffic Store")
st.caption("Query H3-grid discharge volume and footfall metrics to evaluate candidate sites.")

with st.sidebar:
    st.subheader("Query")
    region_name = st.selectbox("Region", options=list(REGIONS.keys()))
    h3_res = st.selectbox("H3 Resolution", options=[7, 8, 9], index=1)
    hour = st.slider("Hour of day", min_value=0, max_value=23, value=8)
    day_type = st.selectbox("Day type", options=["weekday", "weekend"])
    metric = st.selectbox("Metric", options=list(METRIC_LABELS.keys()), format_func=lambda m: METRIC_LABELS[m])
    compare_offpeak = st.checkbox("Compare against an off-peak hour")
    offpeak_hour = st.slider("Off-peak hour", min_value=0, max_value=23, value=22) if compare_offpeak else None
    run = st.button("Run Query", type="primary", use_container_width=True)

if not API_KEY:
    st.warning("No TRAFFIC_STORE_API_KEY set — requests will be sent unauthenticated and will likely be rejected.")

if run or "last_query" in st.session_state:
    bbox = REGIONS[region_name]
    df, err = fetch_discharge_data(bbox, h3_res, hour, day_type)
    st.session_state["last_query"] = (region_name, hour, day_type, metric)

    if err:
        st.error(err)
    elif df.empty:
        st.info("No data returned for this region/time combination.")
    else:
        col_map, col_table = st.columns((2, 1))

        with col_map:
            st.subheader(f"{region_name} — {METRIC_LABELS[metric]} at {hour:02d}:00 ({day_type})")
            st.pydeck_chart(draw_footfall_map(df, metric, bbox))

        with col_table:
            st.subheader("Top Cells")
            top = df.sort_values(metric, ascending=False).head(15)
            st.dataframe(
                top[["h3_cell", metric]].rename(columns={metric: METRIC_LABELS[metric]}),
                use_container_width=True, hide_index=True,
            )

        if compare_offpeak:
            off_df, off_err = fetch_discharge_data(bbox, h3_res, offpeak_hour, day_type)
            if off_err:
                st.error(off_err)
            elif not off_df.empty:
                merged = df[["h3_cell", metric]].merge(
                    off_df[["h3_cell", metric]], on="h3_cell", suffixes=("_peak", "_offpeak")
                )
                merged["delta"] = merged[f"{metric}_peak"] - merged[f"{metric}_offpeak"]
                st.subheader(f"Peak ({hour:02d}:00) vs Off-Peak ({offpeak_hour:02d}:00) Delta")
                st.dataframe(
                    merged.sort_values("delta", ascending=False).head(15),
                    use_container_width=True, hide_index=True,
                )
else:
    st.info("Set your filters and click **Run Query**.")
