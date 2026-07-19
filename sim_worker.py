# -*- coding: utf-8 -*-
"""
Simulation worker — consumes sim:queue:{org_id} jobs and runs them through
the AI² calibration engine (A/B Street + tinygrad PPO agent).

This is deliberately a skeleton: run_calibration_job() is the injection
point. Swap its body for however you currently invoke the engine — direct
Python import of your PPO runner, or a subprocess call into the A/B Street
binary/scenario runner, whichever your existing AI² pipeline uses.

Run one or more of these per org (or one process polling multiple org
queues in round-robin — see main loop) depending on how much concurrent
simulation load you expect from operators.
"""
import json
import logging
import os
import time
import traceback

import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sim_worker")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))

# Orgs this worker process services. For a single-tenant deploy this is
# just [DEFAULT_ORG_ID]; for multi-tenant, list them or read from a
# registry key instead of hardcoding.
ORG_IDS = os.environ.get("SIM_WORKER_ORG_IDS", os.environ.get("MATATU_ORG_ID", "default")).split(",")

BLOCK_TIMEOUT_SECONDS = 5  # BLPOP timeout per org before moving to the next


def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        decode_responses=True, socket_connect_timeout=5, socket_timeout=None,
    )


def run_calibration_job(params: dict) -> dict:
    """
    INJECTION POINT.

    `params` matches what sim_operator_console.py submits:
        corridor, baseline_units, proposed_units, hour_of_day, day_type, notes

    Must return a dict matching the result contract the console renders:
        yield_delta_kes, cost_per_km_delta, congestion_score,
        plausibility_score, avg_dwell_time_delta_s

    Replace the body below with your actual A/B Street scenario setup +
    PPO agent rollout, e.g.:

        from ai2.calibration import run_scenario
        sim_result = run_scenario(
            corridor=params["corridor"],
            fleet_delta=params["proposed_units"] - params["baseline_units"],
            hour=params["hour_of_day"],
            day_type=params["day_type"],
        )
        return {
            "yield_delta_kes": sim_result.yield_delta,
            "cost_per_km_delta": sim_result.cost_per_km_delta,
            "congestion_score": sim_result.congestion_score,
            "plausibility_score": sim_result.ppo_reward,
            "avg_dwell_time_delta_s": sim_result.dwell_delta_s,
        }
    """
    raise NotImplementedError(
        "Wire this up to your AI² A/B Street + PPO calibration engine."
    )


def process_job(r: redis.Redis, org_id: str, job_id: str) -> None:
    job_key = f"sim:job:{job_id}"
    job = r.hgetall(job_key)
    if not job:
        log.warning("job %s vanished before processing", job_id)
        return

    params = json.loads(job.get("params", "{}"))
    r.hset(job_key, mapping={"status": "running", "started_at": time.time()})
    log.info("running job %s org=%s params=%s", job_id, org_id, params)

    try:
        result = run_calibration_job(params)
        r.hset(job_key, mapping={
            "status": "done",
            "result": json.dumps(result),
            "finished_at": time.time(),
        })
        log.info("job %s done", job_id)
    except Exception as e:
        r.hset(job_key, mapping={
            "status": "error",
            "error": f"{e}\n{traceback.format_exc(limit=3)}",
            "finished_at": time.time(),
        })
        log.error("job %s failed: %s", job_id, e)


def main():
    r = get_redis_client()
    queue_keys = [f"sim:queue:{org_id}" for org_id in ORG_IDS]
    log.info("sim_worker started, watching queues: %s", queue_keys)

    while True:
        try:
            # BLPOP across all org queues at once — returns as soon as any
            # org has a job, so idle orgs don't starve busy ones.
            popped = r.blpop(queue_keys, timeout=BLOCK_TIMEOUT_SECONDS)
        except redis.exceptions.RedisError as e:
            log.error("redis error, retrying in 5s: %s", e)
            time.sleep(5)
            continue

        if popped is None:
            continue  # timeout, no job — loop and block again

        queue_key, job_id = popped
        org_id = queue_key.split(":", 2)[-1]
        process_job(r, org_id, job_id)


if __name__ == "__main__":
    main()
