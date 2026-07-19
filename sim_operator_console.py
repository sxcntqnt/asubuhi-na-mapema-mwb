# -*- coding: utf-8 -*-
"""
Operator Efficiency Simulation Console.

Submits "what if I add N units to corridor X during hour Y" scenarios to a
Redis-backed job queue, consumed by sim_worker.py which invokes the real
A/B Street + PPO calibration engine (AI²). Because that engine is slow
(not sub-second), this UI never blocks on a result — it submits, then
polls job status via a fragment.

REDIS SCHEMA:
    sim:queue:{org_id}   LIST    job_id   (producer: RPUSH, worker: BLPOP)
    sim:job:{job_id}     HASH    status | params(json) | result(json) |
                                 error | created_at | started_at | finished_at
    sim:jobs:{org_id}    ZSET    job_id -> created_at   (history listing)

RESULT CONTRACT (assumed — align with whatever your PPO reward /
A/B Street run actually emits; adjust parse_result() if it differs):
    {
      "yield_delta_kes": float,       # projected revenue delta vs baseline
      "cost_per_km_delta": float,     # projected operating cost delta
      "congestion_score": float,      # 0-1, higher = worse
      "plausibility_score": float,    # PPO reward / calibration confidence
      "avg_dwell_time_delta_s": float
    }
"""
import json
import os
import time
import uuid

import pandas as pd
import redis
import streamlit as st

st.set_page_config(layout="wide", page_title="Operator Efficiency Simulator", page_icon=":bar_chart:")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
DEFAULT_ORG_ID = os.environ.get("MATATU_ORG_ID", "default")
JOB_STALE_SECONDS = 600  # flag as possibly-stuck if "running" longer than this

CORRIDORS = [
    "CBD - Westlands",
    "CBD - Eastleigh",
    "CBD - Rongai",
    "CBD - Kikuyu",
    "Thika Road (CBD - Thika)",
    "Ngong Road (CBD - Ngong)",
]


# -----------------------------------------------------------------------------
# REDIS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        decode_responses=True, socket_connect_timeout=1.5, socket_timeout=1.5,
    )


def submit_job(org_id: str, params: dict) -> str | None:
    r = get_redis_client()
    job_id = str(uuid.uuid4())
    now = time.time()
    payload = {
        "status": "queued",
        "org_id": org_id,
        "params": json.dumps(params),
        "created_at": now,
    }
    try:
        pipe = r.pipeline()
        pipe.hset(f"sim:job:{job_id}", mapping=payload)
        pipe.rpush(f"sim:queue:{org_id}", job_id)
        pipe.zadd(f"sim:jobs:{org_id}", {job_id: now})
        pipe.execute()
        return job_id
    except redis.exceptions.RedisError as e:
        st.error(f"Could not submit job — Redis error: {e}")
        return None


def fetch_job(job_id: str) -> dict | None:
    r = get_redis_client()
    try:
        data = r.hgetall(f"sim:job:{job_id}")
    except redis.exceptions.RedisError:
        return None
    return data or None


def fetch_recent_job_ids(org_id: str, limit: int = 20) -> list[str]:
    r = get_redis_client()
    try:
        # highest score (most recent) first
        return r.zrevrange(f"sim:jobs:{org_id}", 0, limit - 1)
    except redis.exceptions.RedisError:
        return []


def queue_depth(org_id: str) -> int | None:
    r = get_redis_client()
    try:
        return r.llen(f"sim:queue:{org_id}")
    except redis.exceptions.RedisError:
        return None


def parse_result(job: dict) -> dict | None:
    raw = job.get("result")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# -----------------------------------------------------------------------------
# SUBMISSION FORM
# -----------------------------------------------------------------------------
st.title("Operator Efficiency Simulator")
st.caption("Submits scenarios to the AI² calibration engine (A/B Street + PPO). Results arrive asynchronously.")

org_id = st.sidebar.text_input("Org ID", value=DEFAULT_ORG_ID)
depth = queue_depth(org_id)
if depth is not None:
    st.sidebar.metric("Jobs ahead in queue", depth)

with st.form("sim_submit_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        corridor = st.selectbox("Corridor", options=CORRIDORS)
        baseline_units = st.number_input("Current vehicle count", min_value=1, max_value=200, value=12)
    with c2:
        proposed_units = st.number_input("Proposed vehicle count", min_value=1, max_value=200, value=15)
        hour_of_day = st.slider("Hour of day", min_value=0, max_value=23, value=7)
    with c3:
        day_type = st.selectbox("Day type", options=["weekday", "weekend"])
        notes = st.text_input("Notes (optional)")

    submitted = st.form_submit_button("Run Simulation", type="primary")

    if submitted:
        params = {
            "corridor": corridor,
            "baseline_units": baseline_units,
            "proposed_units": proposed_units,
            "hour_of_day": hour_of_day,
            "day_type": day_type,
            "notes": notes,
        }
        job_id = submit_job(org_id, params)
        if job_id:
            st.session_state.setdefault("tracked_jobs", [])
            st.session_state["tracked_jobs"].insert(0, job_id)
            st.success(f"Simulation queued — job {job_id[:8]}")


# -----------------------------------------------------------------------------
# STATUS BADGE
# -----------------------------------------------------------------------------
def status_badge(job: dict) -> str:
    status = job.get("status", "unknown")
    if status == "queued":
        return "🟡 queued"
    if status == "running":
        started = float(job.get("started_at", 0) or 0)
        if started and (time.time() - started) > JOB_STALE_SECONDS:
            return "🟠 running (slow — check worker)"
        return "🔵 running"
    if status == "done":
        return "🟢 done"
    if status == "error":
        return "🔴 error"
    return f"⚪ {status}"


# -----------------------------------------------------------------------------
# POLLING FRAGMENT — job history + live status
# -----------------------------------------------------------------------------
@st.fragment(run_every=4)
def job_status_panel(org_id: str):
    job_ids = fetch_recent_job_ids(org_id, limit=20)
    if not job_ids:
        st.info("No simulations submitted yet for this org.")
        return

    st.subheader("Simulation Runs")
    rows = []
    jobs_by_id = {}
    for jid in job_ids:
        job = fetch_job(jid)
        if not job:
            continue
        jobs_by_id[jid] = job
        p = json.loads(job.get("params", "{}"))
        rows.append({
            "job_id": jid[:8],
            "corridor": p.get("corridor", "?"),
            "baseline -> proposed": f"{p.get('baseline_units', '?')} -> {p.get('proposed_units', '?')}",
            "hour": p.get("hour_of_day", "?"),
            "status": status_badge(job),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    done_jobs = {jid: j for jid, j in jobs_by_id.items() if j.get("status") == "done"}
    if done_jobs:
        st.subheader("Results")
        for jid, job in done_jobs.items():
            result = parse_result(job)
            p = json.loads(job.get("params", "{}"))
            with st.expander(f"{p.get('corridor', '?')} — job {jid[:8]}"):
                if not result:
                    st.warning("Job marked done but no parseable result payload.")
                    continue
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Yield delta (KES)", f"{result.get('yield_delta_kes', 0):+.0f}")
                rc2.metric("Cost/km delta", f"{result.get('cost_per_km_delta', 0):+.2f}")
                rc3.metric("Congestion score", f"{result.get('congestion_score', 0):.2f}")
                rc4.metric("Plausibility (PPO)", f"{result.get('plausibility_score', 0):.2f}")

    error_jobs = {jid: j for jid, j in jobs_by_id.items() if j.get("status") == "error"}
    if error_jobs:
        st.subheader("Errors")
        for jid, job in error_jobs.items():
            st.error(f"job {jid[:8]}: {job.get('error', 'unknown error')}")


job_status_panel(org_id)
