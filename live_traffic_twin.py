# -*- coding: utf-8 -*-
"""
Live Transit Twin Engine — Redis/H3-backed.

Reads pre-aggregated H3 cell counts from Redis (written by your Go
batch-writer / StreamRegistry pipeline) instead of raw telemetry rows.
No client-side aggregation of raw points — Streamlit only renders.

ASSUMED REDIS SCHEMA (adjust to match your actual key names):

    h3agg:{org_id}:{res}   HASH   h3_cell -> count
        Incremented by whatever service ingests vehicle pings
        (HINCRBY h3agg:{org}:{res} {h3_cell} 1). Overwritten/decayed
        periodically or bucketed by a rolling window key if you want
        true "live" density rather than cumulative counts.

    events:{org_id}        ZSET   event_id -> unix_timestamp
        Used only to drive the volume-over-time histogram. If you
        already have a time-bucketed counter elsewhere, swap
        fetch_recent_event_timestamps() for a direct read of that.

If your StreamRegistry names things differently, only the two fetch_*
functions below need to change — everything downstream operates on the
normalized DataFrame / array shapes they return.
"""
import os
import time

import altair as alt
import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import redis
import streamlit as st

st.set_page_config(layout="wide", page_title="Live Transit Twin Engine", page_icon=":taxi:")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
DEFAULT_ORG_ID = os.environ.get("MATATU_ORG_ID", "default")
DEFAULT_H3_RES = int(os.environ.get("H3_RES", 8))
HISTOGRAM_WINDOW_SECONDS = 3600  # past hour

# -----------------------------------------------------------------------------
# 1. REDIS CLIENT (cached across reruns, not recreated per fragment tick)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=1.5,
        socket_timeout=1.5,
    )


# -----------------------------------------------------------------------------
# 2. SERVER-SIDE-AGGREGATED DATA FETCH
# -----------------------------------------------------------------------------
def fetch_h3_aggregate(org_id: str, h3_res: int) -> tuple[pd.DataFrame, bool]:
    """
    Reads the precomputed h3_cell -> count hash. O(cells), not O(vehicles) —
    this is the whole point of aggregating upstream instead of SCANning
    raw vehicle keys from the dashboard.

    Returns (df, ok). df has columns: h3_cell, count, lat, lon.
    ok=False signals a Redis failure so the caller can fall back gracefully.
    """
    r = get_redis_client()
    key = f"h3agg:{org_id}:{h3_res}"
    try:
        raw = r.hgetall(key)
    except redis.exceptions.RedisError:
        return pd.DataFrame(columns=["h3_cell", "count", "lat", "lon"]), False

    if not raw:
        return pd.DataFrame(columns=["h3_cell", "count", "lat", "lon"]), True

    cells, counts, lats, lons = [], [], [], []
    for cell, count in raw.items():
        try:
            lat, lon = h3.cell_to_latlng(cell)
        except (h3.H3CellError, ValueError):
            continue  # skip malformed/stale keys rather than crash the fragment
        cells.append(cell)
        counts.append(int(count))
        lats.append(lat)
        lons.append(lon)

    df = pd.DataFrame({"h3_cell": cells, "count": counts, "lat": lats, "lon": lons})
    return df, True


def fetch_recent_event_timestamps(org_id: str, window_seconds: int) -> np.ndarray:
    """Pulls event timestamps from the last `window_seconds` via ZRANGEBYSCORE."""
    r = get_redis_client()
    now = time.time()
    key = f"events:{org_id}"
    try:
        scored = r.zrangebyscore(key, now - window_seconds, now, withscores=True)
    except redis.exceptions.RedisError:
        return np.array([])
    return np.array([score for _, score in scored])


# -----------------------------------------------------------------------------
# 3. GEOMETRY / AGGREGATION HELPERS (pure functions, no I/O)
# -----------------------------------------------------------------------------
def weighted_midpoint(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return (-1.2921, 36.8219)  # fallback: Nairobi center
    w = df["count"].to_numpy()
    return (np.average(df["lat"], weights=w), np.average(df["lon"], weights=w))


def filter_to_hub(df: pd.DataFrame, hub_lat: float, hub_lon: float, res: int, k_rings: int) -> pd.DataFrame:
    """
    Restricts the aggregate to cells within k_rings of the hub's H3 cell.
    Filtering happens on the already-aggregated (small) DataFrame, not on
    raw points, so this stays cheap even at 100K-vehicle scale.
    """
    if df.empty:
        return df
    center = h3.latlng_to_cell(hub_lat, hub_lon, res)
    ring = h3.grid_disk(center, k_rings)
    return df[df["h3_cell"].isin(ring)]


def calculate_histogram(timestamps: np.ndarray, window_seconds: int) -> pd.DataFrame:
    """
    Buckets by elapsed minutes from *now*, not raw dt.minute — the earlier
    version aliased e.g. 10:05 and 11:05 into the same bin. This buckets
    strictly by how long ago each event happened.
    """
    n_bins = window_seconds // 60
    bins = pd.DataFrame({"minutes_ago": range(n_bins), "pickups": 0})
    if timestamps.size == 0:
        return bins
    now = time.time()
    elapsed_minutes = ((now - timestamps) // 60).astype(int)
    elapsed_minutes = elapsed_minutes[(elapsed_minutes >= 0) & (elapsed_minutes < n_bins)]
    counts = np.bincount(elapsed_minutes, minlength=n_bins)
    bins["pickups"] = counts
    return bins


# -----------------------------------------------------------------------------
# 4. MAP RENDERING — native H3HexagonLayer instead of synthetic HexagonLayer
# -----------------------------------------------------------------------------
def draw_h3_hex_map(df: pd.DataFrame, lat: float, lon: float, zoom: int):
    return pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v9",
        initial_view_state={"latitude": lat, "longitude": lon, "zoom": zoom, "pitch": 50},
        layers=[
            pdk.Layer(
                "H3HexagonLayer",
                data=df,
                get_hexagon="h3_cell",
                get_fill_color="[0, 180 - count, 200, 160]",
                get_elevation="count",
                elevation_scale=20,
                extruded=True,
                pickable=True,
            ),
        ],
        tooltip={"text": "{h3_cell}\ncount: {count}"},
    )


# -----------------------------------------------------------------------------
# 5. INTERFACE LAYOUT
# -----------------------------------------------------------------------------
row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    st.title("Live Transit Twin Control")
    refresh_rate = st.slider("Data Poll Frequency (seconds)", min_value=1, max_value=30, value=3)
    org_id = st.text_input("Org ID", value=DEFAULT_ORG_ID)
    h3_res = st.selectbox("H3 Resolution", options=[7, 8, 9], index=1)

with row1_2:
    st.write(
        """
    ## 
    Reads H3-aggregated vehicle density directly from Redis (`h3agg:{org}:{res}`),
    written server-side by the ingestion pipeline. No raw-point SCAN or
    client-side binning happens in this dashboard.
    """
    )

# Nairobi hub coordinates — swap for your actual matatu terminus/stage coords
hub_a = {"name": "CBD / Railways", "lat": -1.2864, "lon": 36.8272}
hub_b = {"name": "Westlands", "lat": -1.2647, "lon": 36.8027}
hub_c = {"name": "Eastleigh", "lat": -1.2749, "lon": 36.8500}
zoom_level = 13
HUB_K_RINGS = 3  # ~3 hex rings around each hub center at res 8

# -----------------------------------------------------------------------------
# 6. THE REALTIME STREAM FRAGMENT
# -----------------------------------------------------------------------------
@st.fragment
def realtime_stream_container(poll_interval: int, org_id: str, h3_res: int):
    agg_df, redis_ok = fetch_h3_aggregate(org_id, h3_res)
    timestamps = fetch_recent_event_timestamps(org_id, HISTOGRAM_WINDOW_SECONDS)

    if not redis_ok:
        st.warning(
            f"Could not reach Redis at {REDIS_HOST}:{REDIS_PORT} — showing empty state. "
            "Check REDIS_HOST/REDIS_PORT env vars and network reachability."
        )

    midpoint = weighted_midpoint(agg_df)
    st.subheader(
        f"🔄 Shared Environment Telemetry — org={org_id} — "
        f"Last Updated: {pd.Timestamp.now().strftime('%H:%M:%S')} — "
        f"{len(agg_df)} active cells"
    )

    row2_1, row2_2, row2_3, row2_4 = st.columns((2, 1, 1, 1))
    with row2_1:
        st.write("**All Active Cells**")
        st.pydeck_chart(draw_h3_hex_map(agg_df, midpoint[0], midpoint[1], 11))
    with row2_2:
        st.write(f"**{hub_a['name']}**")
        hub_df = filter_to_hub(agg_df, hub_a["lat"], hub_a["lon"], h3_res, HUB_K_RINGS)
        st.pydeck_chart(draw_h3_hex_map(hub_df, hub_a["lat"], hub_a["lon"], zoom_level))
    with row2_3:
        st.write(f"**{hub_b['name']}**")
        hub_df = filter_to_hub(agg_df, hub_b["lat"], hub_b["lon"], h3_res, HUB_K_RINGS)
        st.pydeck_chart(draw_h3_hex_map(hub_df, hub_b["lat"], hub_b["lon"], zoom_level))
    with row2_4:
        st.write(f"**{hub_c['name']}**")
        hub_df = filter_to_hub(agg_df, hub_c["lat"], hub_c["lon"], h3_res, HUB_K_RINGS)
        st.pydeck_chart(draw_h3_hex_map(hub_df, hub_c["lat"], hub_c["lon"], zoom_level))

    st.write("**Event Volume (minutes elapsed, past hour)**")
    chart_data = calculate_histogram(timestamps, HISTOGRAM_WINDOW_SECONDS)
    chart = (
        alt.Chart(chart_data)
        .mark_area(interpolate="step-after")
        .encode(
            x=alt.X("minutes_ago:Q", scale=alt.Scale(nice=False, reverse=True), title="minutes ago"),
            y=alt.Y("pickups:Q"),
            tooltip=["minutes_ago", "pickups"],
        )
        .configure_mark(opacity=0.4, color="cyan")
    )
    st.altair_chart(chart, use_container_width=True)

    time.sleep(poll_interval)
    st.rerun(scope="fragment")


realtime_stream_container(refresh_rate, org_id, h3_res)
