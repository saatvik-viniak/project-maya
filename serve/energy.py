"""serve/energy.py - the Usage tab's electricity: what this machine's own cost tracker measured.

The tracker is a separate script (MAYA_ENERGY_SCRIPT, default ~/Maya-data/tools/energy-cost.py: the live ComEd price x
the machine's power, kept in SQLite by its own service).  Its `json` command prints the power now, today's and each
day's and month's kWh and dollars, and the recent GPU-active runs; this module runs it in a background thread, keeps
the answer TTL seconds, and hands the page a trimmed copy.  A request never waits for the script except the very
first one (up to its timeout).  No script (or MAYA_ENERGY_SCRIPT=off): {"available": false}."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

TTL = 15.0          # seconds an answer is kept (the tracker samples every 5 s)
TIMEOUT = 30.0      # the script's own limit
DAYS = 400          # the daily history asked for: all of it (the page shows the last 14 days; the first day dates
                    # the all-time total)

_lock = threading.Lock()
_cache: dict = {"t": 0.0, "data": None, "err": None}
_busy = False


def script_path(env=None) -> Path | None:
    v = ((os.environ if env is None else env).get("MAYA_ENERGY_SCRIPT") or "").strip()
    if v.lower() in ("off", "none", "0"):
        return None
    p = Path(v).expanduser() if v else Path.home() / "Maya-data" / "tools" / "energy-cost.py"
    return p if p.is_file() else None


def _refresh(p: Path) -> None:
    global _busy
    data, err = None, None
    try:
        out = subprocess.run([sys.executable or "python3", str(p), "json", "--days", str(DAYS)],
                             capture_output=True, text=True, timeout=TIMEOUT)
        if out.returncode == 0:
            data = json.loads(out.stdout)
        else:
            err = ((out.stderr or "").strip().splitlines() or ["the tracker failed"])[-1][:300]
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        err = str(e)[:300]
    with _lock:
        if data is not None:
            _cache["data"] = data
            _cache["err"] = None
        else:
            _cache["err"] = err
        _cache["t"] = time.time()
        _busy = False


def _r(v, n=4):
    return None if v is None else round(float(v), n)


def _period(x: dict | None) -> dict | None:
    if not x:
        return None
    return {"kwh": _r(x.get("kwh")), "usd": _r(x.get("cost_usd"), 6), "idle_kwh": _r(x.get("idle_kwh")),
            "idle_usd": _r(x.get("idle_cost_usd"), 6), "active_kwh": _r(x.get("active_kwh")),
            "active_usd": _r(x.get("active_cost_usd"), 6), "avg_w": _r(x.get("avg_w"), 1), "peak_w": _r(x.get("peak_w"), 1)}


def summarize(d: dict) -> dict:
    now = d.get("now") or {}
    month_key = datetime.now().strftime("%Y-%m")
    months = d.get("months") or []
    month = next((m for m in months if m.get("group") == month_key), None)
    days = d.get("days") or []
    # all time: every month the tracker holds (its month rows are never windowed), split into the hours the model
    # worked (a GPU busy: "active") and the rest
    tot = lambda k: sum(float(m.get(k) or 0) for m in months)
    tracked_kwh, tracked_usd = tot("kwh"), tot("cost_usd")
    first_day = min((x.get("group") for x in days if x.get("group")), default=None)
    return {
        "now": {"time": now.get("time"), "ac_w": _r(now.get("ac_w"), 1), "gpu_w": _r(now.get("gpu_w"), 1),
                "gpu_each": [_r(w, 1) for w in (now.get("gpu_each") or [])], "cpu_w": _r(now.get("cpu_w"), 1),
                "base_w": _r(now.get("base_w"), 1), "price_cents": _r(now.get("price_cents"), 2)},
        "today": _period(d.get("today")),
        "month": {"name": month_key, **(_period(month) or {"kwh": 0.0, "usd": 0.0})},
        "days": [{"day": x.get("group"), "kwh": _r(x.get("kwh")), "usd": _r(x.get("cost_usd"), 6)} for x in days],
        "tracked": {"kwh": round(tracked_kwh, 4), "usd": round(tracked_usd, 6), "since": first_day,
                    "active_kwh": round(tot("active_kwh"), 4), "active_usd": round(tot("active_cost_usd"), 6),
                    "idle_kwh": round(tot("idle_kwh"), 4), "idle_usd": round(tot("idle_cost_usd"), 6)},
        "runs": [{"start": x.get("start_ts"), "end": x.get("end_ts"), "kwh": _r(x.get("kwh")),
                  "usd": _r(x.get("cost_usd"), 6), "avg_w": _r(x.get("avg_w"), 1)} for x in (d.get("runs") or [])[:12]],
    }


def report(wait_first: bool = True) -> dict:
    """GET /api/energy: the tracker's numbers, at most TTL seconds old (refreshed in the background)."""
    global _busy
    p = script_path()
    if p is None:
        return {"available": False, "why": "no electricity tracker (MAYA_ENERGY_SCRIPT names its script)"}
    th = None
    with _lock:
        if time.time() - _cache["t"] > TTL and not _busy:
            _busy = True
            th = threading.Thread(target=_refresh, args=(p,), daemon=True, name="maya-energy")
    if th is not None:
        th.start()
        if wait_first and _cache["data"] is None:
            th.join(TIMEOUT + 5)
    with _lock:
        d, e, t = _cache["data"], _cache["err"], _cache["t"]
    if d is None:
        return {"available": False, "why": e or "reading the electricity tracker"}
    return {"available": True, "age_s": round(time.time() - t, 1), "error": e, "script": str(p), **summarize(d)}
