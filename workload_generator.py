"""
Workload Generator for the Edge Scheduler cluster.

Automatically submits jobs at configurable rates so the scheduler
behaviour can be observed without manual curl commands.

Usage examples
--------------
# Steady: 1 job/sec to node a
python workload_generator.py --target http://localhost:8001 --rate 1

# Burst: send 30 jobs as fast as possible to flood node a
python workload_generator.py --target http://localhost:8001 --mode burst --burst-size 30

# Ramp: start at 1 job/sec, double every 15 s up to 8 jobs/sec
python workload_generator.py --target http://localhost:8001 --mode ramp

# Round-robin across all three nodes, steady 2 jobs/sec
python workload_generator.py \\
    --target http://localhost:8001 http://10.213.43.101:8002 http://10.213.43.252:8003 \\
    --rate 2
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random
import time
from datetime import datetime

import httpx

# ---------------------------------------------------------------------------
# Job templates
# ---------------------------------------------------------------------------

JOB_TEMPLATES = [
    {"function": "echo",            "args": {"message": "ping"}},
    {"function": "echo",            "args": {"message": "hello from generator"}},
    {"function": "matrix_multiply", "args": {"size": 50}},
    {"function": "matrix_multiply", "args": {"size": 100}},
    {"function": "hash_chain",      "args": {"seed": "abc", "iterations": 500}},
    {"function": "hash_chain",      "args": {"seed": "xyz", "iterations": 1000}},
]

# ANSI colours
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
GREY   = "\033[90m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def pick_job() -> dict:
    return random.choice(JOB_TEMPLATES).copy()


# ---------------------------------------------------------------------------
# Core submission
# ---------------------------------------------------------------------------

async def submit_job(client: httpx.AsyncClient, target: str, job: dict) -> dict | None:
    try:
        resp = await client.post(
            f"{target}/invoke",
            json=job,
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def print_result(job: dict, result: dict | None, target: str, elapsed: float) -> None:
    ts = now_str()
    fn = job.get("function", "?")
    if result is None or "error" in result:
        err = result.get("error", "unknown") if result else "no response"
        print(f"{GREY}[{ts}]{RESET} {RED}✗{RESET} {fn:20s} → {target}  {RED}ERROR: {err}{RESET}")
        return

    job_id   = result.get("job_id", "?")[:8]
    node     = result.get("scheduled_on", "?")
    fwd      = result.get("forwarded", False)
    fwd_icon = f"{CYAN}↪ forwarded{RESET}" if fwd else f"{GREEN}● local{RESET}"
    print(
        f"{GREY}[{ts}]{RESET} {GREEN}✓{RESET} {fn:20s} → "
        f"{BOLD}node {node}{RESET}  {fwd_icon}  "
        f"{GREY}id={job_id}  {elapsed*1000:.0f}ms{RESET}"
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def run_steady(targets: list[str], rate: float, duration: int | None) -> None:
    """Submit one job every 1/rate seconds, cycling across targets."""
    interval = 1.0 / rate
    target_cycle = itertools.cycle(targets)
    count = 0
    print(f"{CYAN}Mode: steady  rate={rate}/s  targets={targets}{RESET}\n")

    async with httpx.AsyncClient() as client:
        start = time.time()
        while True:
            if duration and (time.time() - start) >= duration:
                break
            target = next(target_cycle)
            job    = pick_job()
            t0     = time.time()
            result = await submit_job(client, target, job)
            print_result(job, result, target, time.time() - t0)
            count += 1
            await asyncio.sleep(max(0, interval - (time.time() - t0)))


async def run_burst(targets: list[str], burst_size: int) -> None:
    """Fire burst_size jobs concurrently to the first target, then stop."""
    target = targets[0]
    print(f"{YELLOW}Mode: burst  size={burst_size}  target={target}{RESET}\n")

    async with httpx.AsyncClient() as client:
        jobs = [pick_job() for _ in range(burst_size)]

        async def _one(j):
            t0 = time.time()
            r  = await submit_job(client, target, j)
            print_result(j, r, target, time.time() - t0)

        await asyncio.gather(*[_one(j) for j in jobs])

    print(f"\n{GREEN}Burst complete — {burst_size} jobs submitted{RESET}")


async def run_ramp(targets: list[str], start_rate: float, max_rate: float, step_secs: int) -> None:
    """Double the rate every step_secs seconds until max_rate, then hold."""
    target_cycle = itertools.cycle(targets)
    rate  = start_rate
    print(f"{YELLOW}Mode: ramp  start={start_rate}/s  max={max_rate}/s  step={step_secs}s{RESET}\n")

    async with httpx.AsyncClient() as client:
        step_start = time.time()
        while True:
            interval = 1.0 / rate
            target   = next(target_cycle)
            job      = pick_job()
            t0       = time.time()
            result   = await submit_job(client, target, job)
            print_result(job, result, target, time.time() - t0)

            if time.time() - step_start >= step_secs and rate < max_rate:
                rate = min(rate * 2, max_rate)
                step_start = time.time()
                print(f"\n{YELLOW}⬆ Rate increased to {rate:.1f} jobs/s{RESET}\n")

            await asyncio.sleep(max(0, interval - (time.time() - t0)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workload generator for the edge scheduler cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target", nargs="+", default=["http://localhost:8001"],
        metavar="URL",
        help="One or more node URLs (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--mode", choices=["steady", "burst", "ramp"], default="steady",
        help="Submission mode (default: steady)",
    )
    parser.add_argument(
        "--rate", type=float, default=1.0,
        help="Jobs per second for steady/ramp mode (default: 1.0)",
    )
    parser.add_argument(
        "--burst-size", type=int, default=20,
        help="Number of concurrent jobs for burst mode (default: 20)",
    )
    parser.add_argument(
        "--max-rate", type=float, default=8.0,
        help="Max rate for ramp mode (default: 8.0)",
    )
    parser.add_argument(
        "--ramp-step", type=int, default=15,
        help="Seconds before doubling rate in ramp mode (default: 15)",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Stop after N seconds (default: run forever)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}Edge Scheduler — Workload Generator{RESET}")
    print(f"{GREY}Press Ctrl+C to stop\n{RESET}")

    try:
        if args.mode == "steady":
            asyncio.run(run_steady(args.target, args.rate, args.duration))
        elif args.mode == "burst":
            asyncio.run(run_burst(args.target, args.burst_size))
        elif args.mode == "ramp":
            asyncio.run(run_ramp(args.target, args.rate, args.max_rate, args.ramp_step))
    except KeyboardInterrupt:
        print(f"\n{GREY}Generator stopped.{RESET}")


if __name__ == "__main__":
    main()
