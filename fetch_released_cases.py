"""Offline: decrypt the cases of games that have ALREADY STARTED and save their texts to data/released/.

Uses only the documented endpoint GET /api/games/{id}/key, which the API handbook says is
"only available after the game's start_time"; a game that has not started is skipped, never probed.
Read-only, does not submit anything, never touches the live bot or its lock.
    python fetch_released_cases.py            # all started games not yet saved
"""
import io, json, os, sys, time, datetime as dt
import requests, pyzipper
from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(HERE, ".env"), encoding="utf-8"):
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
BASE = os.environ.get("C2F_URL", "https://c2f.public.quantco.cloud").rstrip("/")
H = {"X-API-Key": os.environ["TEAM_API_KEY"]}
OUT = os.path.join(HERE, "data", "released")
os.makedirs(OUT, exist_ok=True)

games = requests.get(f"{BASE}/api/games/list", headers=H, timeout=10).json()
games = games.get("games", games) if isinstance(games, dict) else games
now = dt.datetime.now(dt.timezone.utc)
done = 0
for g in sorted(games, key=lambda x: int(x["id"] if "id" in x else x["game_id"])):
    gid = int(g.get("id", g.get("game_id")))
    st = dt.datetime.fromisoformat(str(g["start_time"]).replace("Z", "+00:00"))
    if st.tzinfo is None:
        st = st.replace(tzinfo=dt.timezone.utc)
    dst = os.path.join(OUT, f"case_{gid:03d}.json")
    if gid == 0 or st > now - dt.timedelta(seconds=90) or os.path.exists(dst):
        continue                     # not started (or still open), or already saved
    r = requests.get(f"{BASE}/api/games/{gid}/key", headers=H, timeout=8)
    if r.status_code != 200:
        print(f"game {gid}: key {r.status_code}, skipped"); time.sleep(0.3); continue
    key = r.json()["decryption_key"]
    zp = os.path.join(HERE, "Cases", f"case_{gid:03d}.zip")
    with pyzipper.AESZipFile(zp) as z:
        z.setpassword(key.encode())
        files = {n: z.read(n) for n in z.namelist()}
    get = lambda suf: next((v for k, v in files.items() if k.endswith(suf)), b"")  # noqa: E731
    inv_b = get("invoices.pdf")
    inv = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(inv_b)).pages) if inv_b else ""
    json.dump({"game": gid, "start": g["start_time"], "files": sorted(files),
               "policy": get("policy.txt").decode("utf-8", "ignore"),
               "desc": get("description.txt").decode("utf-8", "ignore"), "invoice": inv},
              open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    done += 1
    time.sleep(0.3)
print(f"saved {done} released case(s) to {OUT}")
