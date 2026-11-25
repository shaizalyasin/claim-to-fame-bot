"""Read-only collector for the PUBLIC leaderboard at c2f.public.quantco.cloud.

Fair play: this calls ONLY the five endpoints the leaderboard page itself fetches
(read off its own Alpine.js source), with no authentication, no probing and no
guessing of undocumented routes:

    /leaderboard/api/games?page=N            /leaderboard/api/games?completed_only=true&page_size=1000
    /leaderboard/api/matrix?page=N&game_limit=K
    /leaderboard/api/matchup?team=T&page=N&game_limit=K
    /leaderboard/api/transactions?game_id=G&team=T&page=N[&line_item=L]
    /leaderboard/api/performance?team=T

It writes snapshots under data/ and NEVER touches the bot or its submissions.
The transaction snapshots it writes are what recall.py (and recall_refresh.py, after every round) build
their index from. Analysis reports: --calibrate.

Usage:
    python collect_leaderboard.py                  # snapshot everything completed so far
    python collect_leaderboard.py --calibrate      # snapshot, then print the t-bracket report
    python collect_leaderboard.py --team Lumenauts

How t is bracketed, from the payoff table in GAME_DESCRIPTION.pdf:
    accepted=False, amount == 0   -> rightful reject  -> that issuer's a  >  t
    accepted=False, amount  > 0   -> wrongful reject  -> a = amount/1.5 and a <= t
    accepted=True                 -> amount = min(a, c)                  -> recovers a
So per line item:  lower = max(a over fair charges),  upper = min(a over fraud charges),
and  lower <= t < upper.
"""
import os, sys, json, time, math, urllib.request, urllib.parse, datetime as dt

BASE = "https://c2f.public.quantco.cloud/leaderboard/api"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
UA = "c2f-team-client/1.0"
OUR_TEAM = os.environ.get("TEAM_NAME", "Lumenauts")


def get(path, tries=3):
    url = f"{BASE}/{path}"
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5 * (i + 1))
    print(f"  ! failed {path}: {last}")
    return None


def save(relpath, obj):
    p = os.path.join(DATA, relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def paged(path_fmt, **kw):
    """Follow the page/total_pages pagination the API returns."""
    out, page = [], 1
    while True:
        d = get(path_fmt.format(page=page, **kw))
        if not d:
            break
        out.extend(d.get("items", []))
        if page >= d.get("total_pages", 1) or not d.get("items"):
            break
        page += 1
    return out


# ---------- collection ----------
def snapshot(team=OUR_TEAM):
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    print(f"snapshot at {stamp}")

    games = get("games?completed_only=true&page_size=1000")
    completed = [g for g in (games or {}).get("items", [])]
    save("games_completed.json", {"fetched_at": stamp, "items": completed})
    done_ids = sorted({int(g["id"]) for g in completed})
    print(f"  completed games: {len(done_ids)} -> {done_ids[:12]}{'...' if len(done_ids) > 12 else ''}")

    matrix = get("matrix?page=1&game_limit=1000")
    if matrix:
        save("matrix.json", {"fetched_at": stamp, **matrix})
        print(f"  matrix: {matrix.get('total')} teams over games {matrix.get('game_ids')}")

    perf = get(f"performance?team={urllib.parse.quote(team)}")
    if perf:
        save(f"performance_{team}.json", {"fetched_at": stamp, **perf})
        print(f"  performance for {team}: {len(perf.get('items', perf.get('games', [])) or [])} rows")

    matchup = paged("matchup?team={t}&page={page}&game_limit=1000", t=urllib.parse.quote(team))
    if matchup:
        save(f"matchup_{team}.json", {"fetched_at": stamp, "items": matchup})
        print(f"  matchup rows: {len(matchup)}")

    # The transactions endpoint filters to one team, so a single query only shows rows
    # that involve us -- which leaves every bracket on t one-sided (we never learn the
    # charge of an issuer we rightfully rejected). The leaderboard page's own Team
    # dropdown queries any team, so we do the same and merge, giving both sides.
    teams = [r["team_name"] for r in (matrix or {}).get("items", [])] or [team]
    new = 0
    for gid in done_ids:
        p = os.path.join(DATA, "transactions", f"game_{gid:03d}.json")
        if os.path.exists(p):
            try:
                if json.load(open(p, encoding="utf-8")).get("teams_queried") == len(teams):
                    continue              # already have the full merge for this game
            except Exception:  # noqa: BLE001
                pass
        merged = {}
        for t in teams:
            for r in paged("transactions?game_id={g}&team={t}&page={page}", g=gid,
                           t=urllib.parse.quote(t)):
                merged[(r["issuer"], r["reviewer"], r["line_item_index"])] = r
        rows = list(merged.values())
        if rows:
            save(f"transactions/game_{gid:03d}.json",
                 {"fetched_at": stamp, "game_id": gid, "teams_queried": len(teams),
                  "items": rows})
            new += 1
            print(f"  transactions game {gid}: {len(rows)} unique rows "
                  f"(merged over {len(teams)} teams)")
    print(f"  {new} new transaction file(s); data/ is at {DATA}")
    try:
        build_memory(team)
    except Exception as e:  # noqa: BLE001
        print("  ! memory build failed:", repr(e)[:120])
    return done_ids


def build_memory(team=OUR_TEAM):
    """Distil every settled line item into bounds on t, for the assessors to calibrate on.

    This is OUR OWN ground truth, recovered from the public transaction log:
      reject with amount == 0 -> that issuer's a > t      (an upper bound on t)
      reject with amount  > 0 -> a = amount/1.5 and a <= t (a lower bound on t)
      accept                  -> amount = min(a, c), recovering a
    """
    tx_dir = os.path.join(DATA, "transactions")
    logs = os.path.join(HERE, "logs")
    if not os.path.isdir(tx_dir):
        return []
    rows = []
    for fn in sorted(os.listdir(tx_dir)):
        if not fn.startswith("game_"):
            continue
        gid = int(fn[5:8])
        try:
            tx = json.load(open(os.path.join(tx_dir, fn), encoding="utf-8"))["items"]
        except Exception:  # noqa: BLE001
            continue
        texts = {}
        lp = os.path.join(logs, f"game_{gid:03d}.json")
        if os.path.exists(lp):
            try:
                d = json.load(open(lp, encoding="utf-8"))
                for it in d.get("items") or []:
                    texts[int(it["index"])] = str(it.get("text") or "")
            except Exception:  # noqa: BLE001
                pass
        per = charges_and_brackets(tx)
        for li, b in sorted(per.items()):
            txt = texts.get(li, "").strip()
            if not txt or (b["lower"] is None and b["upper"] is None):
                continue
            rows.append({"game": gid, "line": li, "text": txt[:160],
                         "t_lo": b["lower"], "t_hi": b["upper"]})
    save("price_memory.json", {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                               "rows": rows})
    print(f"  price memory: {len(rows)} settled line items -> data/price_memory.json")
    return rows


# ---------- calibration ----------
def charges_and_brackets(rows):
    """From one game's transaction rows, recover each issuer's a and bracket t per line item.

    Returns {line_item: {"a": {team: a}, "fair": set, "fraud": set, "lower": x, "upper": y}}
    """
    per = {}
    for r in rows:
        li = r["line_item_index"]
        d = per.setdefault(li, {"a": {}, "fair": set(), "fraud": set()})
        iss, amt, acc = r["issuer"], float(r["amount"]), bool(r["accepted"])
        if acc:
            # amount = min(a, c); c >= 4t so for sane charges this IS a
            d["a"].setdefault(iss, amt)
        elif amt > 0:
            d["a"][iss] = amt / 1.5         # wrongful reject pays exactly 1.5a
            d["fair"].add(iss)
        else:
            d["fraud"].add(iss)             # rightful reject: a > t, amount tells us nothing
    for li, d in per.items():
        fair_a = [d["a"][t] for t in d["fair"] if t in d["a"]]
        fraud_a = [d["a"][t] for t in d["fraud"] if t in d["a"]]
        d["lower"] = max(fair_a) if fair_a else None
        d["upper"] = min(fraud_a) if fraud_a else None
    return per


def calibrate(team=OUR_TEAM):
    """Compare our own median/p10/p90 against the bracketed t. Reports only; changes nothing."""
    tx_dir = os.path.join(DATA, "transactions")
    logs = os.path.join(HERE, "logs")
    if not os.path.isdir(tx_dir):
        print("no transaction data yet"); return
    rows_out = []
    for fn in sorted(os.listdir(tx_dir)):
        gid = int(fn[5:8])
        tx = json.load(open(os.path.join(tx_dir, fn), encoding="utf-8"))["items"]
        per = charges_and_brackets(tx)
        lp = os.path.join(logs, f"game_{gid:03d}.json")
        ours = {}
        if os.path.exists(lp):
            d = json.load(open(lp, encoding="utf-8"))
            for it in d.get("items") or []:
                try:
                    ours[int(it["index"])] = it
                except Exception:  # noqa: BLE001
                    pass
        for li, b in sorted(per.items()):
            o = ours.get(li)
            lo, hi = b["lower"], b["upper"]
            # Only a two-sided bracket estimates t. A lone lower bound is CENSORED:
            # using it as t_mid would drag every ratio down and invent a false bias.
            t_mid = math.sqrt(lo * hi) if (lo is not None and hi is not None) else None
            rows_out.append({
                "game": gid, "line": li,
                "t_lower": lo, "t_upper": hi, "t_mid": t_mid,
                "our_median": (o or {}).get("median"), "our_p10": (o or {}).get("p10"),
                "our_p90": (o or {}).get("p90"), "our_a": (o or {}).get("a"),
                "our_b": (o or {}).get("b"), "our_q": (o or {}).get("p_valid"),
                "n_charges": len(b["a"]), "n_fair": len(b["fair"]), "n_fraud": len(b["fraud"]),
                "censored": (lo is not None) and (hi is None),
            })
    save("calibration.json", {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                              "rows": rows_out})

    print()
    print("=" * 100)
    print("CALIBRATION REPORT  (report only - nothing here is wired into pricing)")
    print("=" * 100)
    print(f"{'game':>5}{'line':>5}{'t_lower':>11}{'t_upper':>11}{'t_mid':>11}"
          f"{'our_med':>11}{'ratio':>8}{'in[p10,p90]':>13}{'our_a<=t?':>11}{'our_b>=t?':>11}")
    logr, cov, n_a_fair, n_b_ok, n = [], 0, 0, 0, 0
    for r in rows_out:
        tm, med = r["t_mid"], r["our_median"]
        ratio = (tm / med) if (tm and med) else None
        inband = ""
        if tm and r["our_p10"] and r["our_p90"]:
            ok = r["our_p10"] <= tm <= r["our_p90"]
            inband = "yes" if ok else "NO"
            cov += ok; n += 1
        a_ok = b_ok = ""
        if r["our_a"] is not None and r["t_upper"] is not None:
            a_ok = "yes" if r["our_a"] < r["t_upper"] else "NO"
            n_a_fair += (r["our_a"] < r["t_upper"])
        if r["our_b"] is not None and r["t_lower"] is not None:
            b_ok = "yes" if r["our_b"] >= r["t_lower"] else "NO"
            n_b_ok += (r["our_b"] >= r["t_lower"])
        if ratio:
            logr.append(math.log(ratio))
        f = lambda v, w=11, p=2: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")  # noqa: E731
        print(f"{r['game']:>5}{r['line']:>5}{f(r['t_lower'])}{f(r['t_upper'])}{f(r['t_mid'])}"
              f"{f(med)}{f(ratio,8)}{inband:>13}{a_ok:>11}{b_ok:>11}")

    print("-" * 100)
    if logr:
        logr.sort()
        med_log = logr[len(logr) // 2]
        bias = math.exp(med_log)
        spread = math.exp(logr[-1]) / math.exp(logr[0]) if len(logr) > 1 else 1
        print(f"  items with a usable bracket : {len(logr)}")
        print(f"  BIAS MULTIPLIER  median(t_mid / our_median) = {bias:.3f}")
        print(f"     -> our median is {'TOO LOW' if bias > 1.05 else 'TOO HIGH' if bias < 0.95 else 'well centred'}"
              f"; BIAS_MULT={bias:.3f} would re-centre it")
        print(f"  log-ratio spread (max/min)  : {spread:.2f}x")
        if n:
            print(f"  COVERAGE  t_mid inside our [p10,p90] : {cov}/{n} = {cov / n:.0%}  (target ~80%)")
            if cov / n < 0.6:
                print("     -> our p10/p90 are too NARROW; raise SIGMA_MULT")
            elif cov / n > 0.95:
                print("     -> our p10/p90 are too WIDE; lower SIGMA_MULT")
    else:
        print("  no line item yet has both a fair and a fraud charge; need more rounds")
    cens = sum(1 for r in rows_out if r.get("censored"))
    if cens:
        print(f"  censored items (lower bound only, EXCLUDED from the bias): {cens}")
    print(f"  our charge was in the fair zone : {n_a_fair} items")
    print(f"  our limit b was at or above t   : {n_b_ok} items")
    print()
    print("  NOTE: do not change SIGMA_MULT or BIAS_MULT on this report until the")
    print("        organizers confirm the public leaderboard data may be used.")


if __name__ == "__main__":
    team = OUR_TEAM
    if "--team" in sys.argv:
        team = sys.argv[sys.argv.index("--team") + 1]
    snapshot(team)
    if "--calibrate" in sys.argv:
        calibrate(team)
