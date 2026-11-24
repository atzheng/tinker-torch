# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.15.0", "numpy==2.4.6", "scikit-learn==1.9.1"]
# ///
"""Run the generated torch scripts (tools/torch-smoke/smoke.mjs writes them) and check each one.

Usage: uv run tools/torch-smoke/run.py <manifest.json>

Per case: exit 0, every printed train and val loss finite, final train loss not above the first.
A case with no script is "skipped"; a case in the expected-divergence list is "expected NaN" when
it prints a non-finite loss (and a failure when it does not, so the list stays honest).
Exit status 1 when any case fails.
"""
import json
import math
import re
import subprocess
import sys

LINE = re.compile(r"step (\d+): train loss ([^,\s]+)(?:, val loss (\S+))?")


def check(script):
    p = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["no stderr"])[-1]
        return "FAIL", f"exit {p.returncode}: {tail}", False
    rows = LINE.findall(p.stdout)
    if not rows:
        return "FAIL", "no loss lines printed", False
    vals = [float(v) for r in rows for v in r[1:] if v != ""]
    if not all(math.isfinite(v) for v in vals):
        return "FAIL", "non-finite loss printed", True
    first, last = float(rows[0][1]), float(rows[-1][1])
    if last > first:
        return "FAIL", f"final train loss {last:.4g} above first {first:.4g}", False
    return "ok", f"train loss {first:.4g} -> {last:.4g}", False


def main():
    with open(sys.argv[1]) as f:
        cases = json.load(f)
    counts = {"ok": 0, "FAIL": 0, "skipped": 0, "expected NaN": 0}
    for c in cases:
        name = c["name"]
        if c["script"] is None:
            status, msg = "skipped", c.get("skip", "")
        else:
            status, msg, nonfinite = check(c["script"])
            if c.get("expectedNaN"):
                if nonfinite:
                    status, msg = "expected NaN", c["expectedNaN"]
                else:
                    status, msg = "FAIL", "listed as expected NaN but the losses are finite"
            if status == "FAIL":
                msg += f"  [{c['script']}]"
        counts[status] += 1
        print(f"{status:12} {name:50} {msg}", flush=True)
    print(
        f"\n{counts['ok']} ok, {counts['FAIL']} failed, {counts['skipped']} skipped, "
        f"{counts['expected NaN']} expected NaN"
    )
    sys.exit(1 if counts["FAIL"] else 0)


main()
