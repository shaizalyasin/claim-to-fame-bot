"""Spawned detached by the bot after each round. Waits until that round has closed and settled, then
refreshes the read-only public data and rebuilds data/recall_index.json. Never submits anything,
never touches the bot or its lock. Two passes (+RECALL_REFRESH_WAIT s and +240 s later) in case the
round settles late; each pass ends long before the next round opens (rounds are 800 s apart).
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
gid = sys.argv[1] if len(sys.argv) > 1 else "?"
LOG = os.path.join(HERE, "logs", "recall_refresh.log")


def note(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} after game {gid}: {msg}\n")
    except OSError:
        pass


def one_pass(tag):
    for args in (["collect_leaderboard.py"], ["fetch_released_cases.py"], ["recall.py", "--build"]):
        try:
            r = subprocess.run([sys.executable, *args], cwd=HERE, capture_output=True, text=True,
                               timeout=200, encoding="utf-8", errors="replace")
            tail = (r.stdout or r.stderr or "").strip().splitlines()[-1:] or [""]
            note(f"{tag} {args[0]} rc={r.returncode} {tail[0][:160]}")
        except Exception as e:  # noqa: BLE001
            note(f"{tag} {args[0]} failed: {e!r}"[:200])


time.sleep(float(os.environ.get("RECALL_REFRESH_WAIT", "100")))
one_pass("pass1")
time.sleep(240)
one_pass("pass2")
