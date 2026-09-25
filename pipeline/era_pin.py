#!/usr/bin/env python3
"""Generate a pip constraints file pinning each dependency to the latest
release published BEFORE a given cutoff date (era-consistent install).

Usage: era_pin.py <cutoff_iso> pkg1 pkg2 ...  -> prints constraints lines
"""
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CACHE = Path.home() / "kaggle_comp" / "data" / "pypi_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def releases(pkg):
    f = CACHE / f"{pkg}.json"
    if f.exists():
        data = json.loads(f.read_text())
    else:
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{pkg}/json", timeout=20
            ) as r:
                data = json.loads(r.read())
        except Exception:
            return {}
        f.write_text(json.dumps(data))
    out = {}
    for ver, files in (data.get("releases") or {}).items():
        for fh in files:
            t = fh.get("upload_time_iso_8601") or fh.get("upload_time")
            if t:
                out.setdefault(ver, t)
                break
    return out


def latest_before(pkg, cutoff):
    best, best_t = None, None
    for ver, t in releases(pkg).items():
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt <= cutoff and (best_t is None or dt > best_t):
            best, best_t = ver, dt
    return best


if __name__ == "__main__":
    cutoff = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    for pkg in sys.argv[2:]:
        v = latest_before(pkg, cutoff)
        if v:
            print(f"{pkg}=={v}")
