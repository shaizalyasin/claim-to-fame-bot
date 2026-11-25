"""Recall: earlier settled verdicts on the SAME invoice line (idea from the August winner Codacabana).

Many invoice lines recur across cases ("Service technician hours 8 hrs", "Connection hose 1 pcs").
The public transaction log of a finished game brackets its fair value t per line:
    a fair charge wrongly rejected (paid 1.5a)  -> t >= a
    a charge rejected and paid 0                -> t <  a
So for a line seen before, LO = the largest charge already proven fair. In the next game with the same
line we (1) charge at least RECALL_A_FRAC x LO and (2) accept at least up to RECALL_B_FRAC x LO.
Never lowers anything. Exact match on (normalised text, quantity, unit) only.

Backtest, games 31-62 (only earlier games used for each game, exact public results):
  charge floor 0.9 x LO: +45k income, 11 rounds better / 3 worse (conservative scoring)
  accept floor 1.0 x LO: +8.3k (fewer 1.5x penalties net of extra inflated charges), 11/4.
Live, games 65-100 (logged values before/after recall, scored on the public results): +824k in total,
29 rounds better / 2 worse / 5 unchanged. Most of it came from rounds where the model call timed out: there the
fallback (a = 400, b = 0) was replaced by the proven-fair level on every recurring line.

    python recall.py --build      # rebuild data/recall_index.json from data/released + data/transactions
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
INDEX = os.path.join(DATA, "recall_index.json")
RECALL_A_FRAC = float(os.environ.get("RECALL_A_FRAC", "0.9"))
RECALL_B_FRAC = float(os.environ.get("RECALL_B_FRAC", "1.0"))

# ---------------------------------------------------------------- invoice parser
# Deterministic; indices come from the printed positions (tests/test_recall.py, and during the event it matched
# the model's own indices on every logged invoice).
UNIT = (r"(?:pcs|pc|piece|pieces|hrs|hr|hours|hour|h|m2|m²|sqm|m3|m³|lm|m|kg|l|litres?|days?|units?|"
        r"sets?|flat rate|lump sum|flat|item|items)")
QTY_LINE = re.compile(rf"^\s*(\d+(?:[.,]\d+)?)\s+({UNIT})\s*$", re.I)
QTY_TAIL = re.compile(rf"^(.*?\S)\s+(\d+(?:[.,]\d+)?)\s+({UNIT})\s*$", re.I)
ITEM_START = re.compile(r"^\s*(\d{1,3})\s+(\S.*)$")
STOP = re.compile(r"^\s*(INVOICE\b|Created on|Page\s*\d)", re.I)


def _num(s):
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_invoice(inv):
    items, cur, last = [], None, 0
    in_items = False
    for raw in (inv or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^POS\.?\s", line, re.I) or line.upper().startswith("POS. DESCRIPTION"):
            in_items = True
            continue
        if not in_items:
            continue
        if STOP.match(line):
            in_items, cur = False, None
            continue
        q = QTY_LINE.match(line)
        if q and cur is not None and cur["quantity"] is None:
            cur["quantity"], cur["unit"] = _num(q.group(1)), q.group(2).lower()
            continue
        m = ITEM_START.match(line)
        if m and int(m.group(1)) > last and not QTY_LINE.match(line):
            idx, rest = int(m.group(1)), m.group(2)
            t = QTY_TAIL.match(rest)
            cur = {"index": idx, "description": (t.group(1) if t else rest).strip(),
                   "quantity": _num(t.group(2)) if t else None, "unit": t.group(3).lower() if t else None}
            items.append(cur)
            last = idx
            continue
        if cur is not None:
            t = QTY_TAIL.match(line)
            if t and cur["quantity"] is None:
                cur["description"] += " " + t.group(1).strip()
                cur["quantity"], cur["unit"] = _num(t.group(2)), t.group(3).lower()
            else:
                cur["description"] += " " + line
    for it in items:
        it["description"] = re.sub(r"\s+", " ", it["description"]).strip()
    return items


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def line_key(item):
    return f"{norm(item['description'])}|{item.get('quantity')}|{item.get('unit')}"


# ---------------------------------------------------------------- brackets from the public log
def brackets(rows):
    """{line_index: (lo, hi)}: lo = largest charge proven fair, hi = smallest charge proven inflated."""
    a, st = {}, {}
    for r in rows:
        k = (r["issuer"], r["line_item_index"])
        if r["accepted"]:
            a.setdefault(k, r["amount"])
        elif r["amount"] > 0:
            a[k] = r["amount"] / 1.5
            st[k] = "fair"
        else:
            st.setdefault(k, "fraud")
    out = {}
    for li in {k[1] for k in a} | {k[1] for k in st}:
        fair = [a[k] for k in a if k[1] == li and st.get(k) == "fair"]
        fraud = [a[k] for k in a if k[1] == li and st.get(k) == "fraud"]
        out[li] = (max(fair) if fair else None, min(fraud) if fraud else None)
    return out


def build_index(released_dir=None, tx_dir=None, with_game_keys=False):
    released_dir = released_dir or os.path.join(DATA, "released")
    tx_dir = tx_dir or os.path.join(DATA, "transactions")
    idx, gkeys = {}, {}
    for f in sorted(os.listdir(released_dir)):
        if not f.endswith(".json"):
            continue
        c = json.load(open(os.path.join(released_dir, f), encoding="utf-8"))
        g = int(c["game"])
        tp = os.path.join(tx_dir, f"game_{g:03d}.json")
        if not os.path.exists(tp):
            continue
        br = brackets(json.load(open(tp, encoding="utf-8"))["items"])
        items = parse_invoice(c.get("invoice", ""))
        gkeys[str(g)] = sorted({line_key(it) for it in items})
        for it in items:
            if it["index"] in br:
                lo, hi = br[it["index"]]
                idx.setdefault(line_key(it), []).append([g, lo, hi])
    return (idx, gkeys) if with_game_keys else idx


# Ceiling, only inside a REPEATED case: an earlier game whose invoice shares >= RECALL_SIM of the smaller
# invoice's lines (and >= RECALL_MIN_SHARED lines). There a line proven "fair up to LO, inflated from HI" keeps
# its t, so we never charge at or above HI (charge RECALL_A_FRAC x LO instead) and never accept at or above HI.
# Outside repeated cases an upper bound often just means "excluded under that policy", so it is not used.
# Team analysis games 31-66: charge ceiling +20.0k (5 rounds better / 1 worse), limit ceiling +7.2k (6/1).
RECALL_SIM = float(os.environ.get("RECALL_SIM", "0.5"))
RECALL_MIN_SHARED = int(os.environ.get("RECALL_MIN_SHARED", "3"))


def similar_games(cur_keys, game_keys, before_game):
    """{earlier game: share of the smaller invoice} for earlier games that pass the repeated-case gate."""
    out = {}
    cur = set(cur_keys)
    for g, ks in (game_keys or {}).items():
        g = int(g)
        if g >= before_game:
            continue
        ks = set(ks)
        shared = len(cur & ks)
        if shared >= RECALL_MIN_SHARED:
            share = shared / max(1, min(len(cur), len(ks)))
            if share >= RECALL_SIM:
                out[g] = share
    return out


def ceiling_for(entries, before_game, sim):
    """HI: smallest charge proven inflated in an earlier REPEATED case that also proved a fair charge."""
    his = [e[2] for e in entries if e[0] < before_game and e[1] and e[2] and e[0] in sim]
    return min(his) if his else None


def floor_for(entries, before_game):
    """LO from earlier games, exactly as measured: max proven-fair charge, capped just below the smallest
    proven-inflated charge of any earlier game that also had a fair one. None if nothing proven fair."""
    los = [e[1] for e in entries if e[0] < before_game and e[1]]
    his = [e[2] for e in entries if e[0] < before_game and e[1] and e[2]]
    if not los:
        return None
    return max(los) if not his else min(max(los), min(his) * 0.999)


def load_index(path=INDEX):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["lines"]


def load_full(path=INDEX):
    """(lines index, game keys or None for an index built before the ceiling existed)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["lines"], d.get("game_keys")


def apply(subs, inv, gid, index, game_keys=None):
    """Recall floors on recurring lines (always) and ceilings inside a repeated case (when game_keys is
    given). Returns (new_subs, changes). Never raises into the caller's round: on any problem the original
    submission comes back unchanged."""
    try:
        keys = {it["index"]: line_key(it) for it in parse_invoice(inv)}
        sim = similar_games(keys.values(), game_keys, gid) if game_keys else {}
        out, changes = [], []
        for s in subs:
            i = int(s["index"])
            ent = index.get(keys.get(i, ""), []) if i in keys else []
            lo = floor_for(ent, gid) if ent else None
            if not lo or lo <= 0:
                out.append(s)
                continue
            hi = ceiling_for(ent, gid, sim) if sim else None
            a0, b0 = float(s["charge_price"]), float(s["acceptance_limit"])
            a1 = max(a0, RECALL_A_FRAC * lo)
            b1 = max(b0, RECALL_B_FRAC * lo)
            if hi:
                if a1 >= hi:
                    a1 = RECALL_A_FRAC * lo
                b1 = max(min(b1, 0.999 * hi), min(RECALL_B_FRAC * lo, 0.999 * hi))
            a1, b1 = round(a1, 2), round(b1, 2)
            if a1 != a0 or b1 != b0:
                c = {"index": i, "lo": round(lo, 2), "a": [a0, a1], "b": [b0, b1]}
                if hi:
                    c["hi"] = round(hi, 2)
                changes.append(c)
            out.append(dict(s, charge_price=a1, acceptance_limit=b1))
        return out, changes
    except Exception as e:  # noqa: BLE001
        return subs, [{"error": repr(e)[:160]}]


if __name__ == "__main__":
    if "--build" in sys.argv:
        ix, gk = build_index(with_game_keys=True)
        tmp = INDEX + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"lines": ix, "games": sorted({e[0] for v in ix.values() for e in v}), "game_keys": gk}, fh)
        os.replace(tmp, INDEX)
        print(f"recall index: {len(ix)} distinct lines, {sum(len(v) for v in ix.values())} settled rows -> {INDEX}")
