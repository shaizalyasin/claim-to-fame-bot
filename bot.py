"""Claim to Fame bot (QuantCo challenge, START Hack Munich 2026): one process that submits a charge a and an
acceptance limit b for every invoice line in every round.

Round, in order (every step has a deadline and a fallback; a round is never left at the a=0/b=0 default):
  1. blind submission the instant the game opens (a=PROVISIONAL_A, b=0 on positions 1..25)
  2. key -> decrypt the case -> invoice text -> provisional submission on the real line indices
  3. one LLM call (policy excerpt + damage description + invoice) -> per line: P(covered and related) q and a
     price belief (p10 / median / p90, gross EUR); runs in daemon threads, a second identical request is sent
     if the first is slow (hedge), and the round deadline always wins
  4. validation, then the decision rule below -> a, b
  5. recall (recall.py): lines already settled in earlier games get a floor (and, inside a repeated case,
     a ceiling) from the public transaction log
  6. final submission, acknowledged; a detached job refreshes the recall data after the round

Decision rule:
  belief: with prob q the item is covered+related and t ~ LogNormal(median m, sigma s); else t = 0
  insurer b: accept a charge x iff P(t >= x) >= 2/3   (accept costs x, wrong reject costs 1.5x)
             -> b = 1/3-quantile of t; b = 0 when q < 2/3
  handyman a: the issuer never pays a penalty, so maximise a * P(t >= a)  (below the median on purpose)

Usage:
  self-test:  SELFTEST=1 python bot.py
  test:       ONLY_GAME=0 DRY_RUN=1 python bot.py     (game 0, nothing sent)
  test:       ONLY_GAME=0 python bot.py               (real PUTs, game 0 stores nothing)
  live:       ALLOW_LIVE=1 python bot.py              (or watchdog.ps1, which restarts it if it dies)
"""
import os, sys, io, json, math, time, re, threading, datetime as dt, traceback


def _load_env():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
import requests, pyzipper
from pypdf import PdfReader

BASE = os.environ.get("C2F_URL", "https://c2f.public.quantco.cloud").rstrip("/")
TEAM_KEY = os.environ.get("TEAM_API_KEY", "")         # required to play (checked in main), not to self-test
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.environ.get("CASES_DIR", os.path.join(HERE, "Cases"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1")
LLM_BASE = os.environ.get("LLM_BASE_URL")            # None = OpenAI; or AMD Token Factory URL
LLM_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")

SIGMA_MULT = float(os.environ.get("SIGMA_MULT", "1.0"))   # widen LLM ranges until calibrated
SIGMA_MIN, SIGMA_MAX = 0.20, 1.20
# Calibrated from the public transaction log, games 5-11: of 17 line items where the field
# bounds t and we gave a positive median, 13 were PROVEN overestimates and 1 an underestimate
# (binomial p ~ 0.001). Measured median of t/our_median is <= 0.745 on single-call output and
# ~0.6 on ensemble output. Set to 0.75: a deliberate UNDER-correction toward 0.6, because the
# errors are heterogeneous (0.31 to 1.24) and over-correcting turns bought-fraud into 1.5a
# wrongful-reject penalties. Re-measure every few rounds; this is not a fixed constant.
# A_MULT scales our CHARGE a, B_MULT scales our ACCEPTANCE LIMIT b. They were one knob
# (BIAS_MULT) until games 11-14 showed it cut our earnings: issuer share of the pot fell
# from 11.9% to 9.1% while spending did not move. Backtest on 34 settled items:
#   a x1.00 b x1.00 net 11,121 | a x0.75 b x0.75 net 8,172 | a x1.00 b x0.75 net 15,694
# BUT that backtest scored OLD single-call medians. The out-of-sample replay of the real
# live pipeline (gpt-4.1, 3 voices, prompt, memory from earlier games only), games 5-14:
#   a0.75 b0.75 net 18,691 | a1.00 b0.75 17,518 | a1.00 b1.00 18,570 | original 4,496
# so the shipped default is a0.75 b0.75. Both scale the WHOLE distribution (output
# scaling), which fixes the old bug where only the median moved and p10/p90 did not.
# --- 17:21 Sat: transaction-derived components OFF by default -------------------------------
# Price memory (MEM_MAX), the 0.75 multipliers (A_MULT/B_MULT) and post-round transaction
# harvesting (COLLECT_AFTER_ROUND) all rely on bounds on t reconstructed from the public
# transaction log -- a fair-play grey area ("do not ... extract the secret thresholds").
# Out-of-sample replay, games 5-15, identical gpt-4.1 3-voice pipeline:
#   memory + bias (previous live) 19,237 | neither 19,378 | memory only 14,087 | bias only 14,137
# They added no measurable value, so they are off. The code stays for transparency.
A_MULT = float(os.environ.get("A_MULT", "1.0"))
B_MULT = float(os.environ.get("B_MULT", os.environ.get("BIAS_MULT", "1.0")))
# Guardrail, DEFAULT OFF (1e9). a is non-monotonic in sigma: when the LLM returns a very wide
# p10/p90 (sigma > ~0.8) the risk-neutral argmax of a*P(t>=a) sits ABOVE the median, i.e. the bot
# deliberately charges into the fraud zone. Set A_CAP_MULT=1.0 to cap a at the median instead.
A_CAP_MULT = float(os.environ.get("A_CAP_MULT", "1e9"))
COLLECT_AFTER_ROUND = os.environ.get("COLLECT_AFTER_ROUND", "0") == "1"
# Acceptance floor: on lines we are confident are covered and related (q >= B_FLOOR_Q), accept any
# charge up to B_FLOOR x our own estimate, instead of only up to the 1/3-quantile below it.
# Paired replay on the SAME model outputs, 17 rounds x 3 independent runs (team analysis only):
#   mean net 32,263 -> 33,443 (+1,179), better in 3/3 runs; wrongful-reject penalties 47,949 -> 29,657;
#   inflated charges paid 11,189 -> 16,107; no single round moves by more than 1,500.
# Small but consistent. B_FLOOR=0 switches it off.
B_FLOOR = float(os.environ.get("B_FLOOR", "1.0"))
B_FLOOR_Q = float(os.environ.get("B_FLOOR_Q", "0.9"))
# Coverage judge: one extra call to a gpt-6 model, in parallel with the gpt-4.1 voices. Its
# covered/excluded call replaces the ensemble's; gpt-4.1 keeps setting the price. Team analysis on
# 17 settled rounds: gpt-6 models reject 87-93% of truly uncovered lines vs 75% for gpt-4.1, while
# gpt-4.1 prices best; the combination beat gpt-4.1 alone for all 5 gpt-6 judges tried (+775..+1,194,
# worst round better each time). Small gain. If the judge fails or is late, nothing changes.
# COVERAGE_MODEL= (empty) switches it off.
COVERAGE_MODEL = os.environ.get("COVERAGE_MODEL", "gpt-6.2-sol")
# Price blend: on lines both models call covered, use the geometric mean of the ensemble (gpt-4.1)
# median and the judge (gpt-6.2-sol) median; the band is shifted by the same factor. Two model
# families make partly independent pricing errors, so the average is more accurate. Team analysis,
# 17 settled rounds, 6 independent pairings: +5,824 vs judge-only, better in 6/6, worst round
# -4,925 -> -1,755. PRICE_BLEND=0 switches it off.
PRICE_BLEND = os.environ.get("PRICE_BLEND", "1") == "1"
# Weight on the judge (gpt-6.2-sol) in the price blend, in log space. 0.5 = plain geometric mean.
# Team analysis: 0.65 was the only weight better than 0.5 on BOTH the tuning rounds 5-21
# (+2,973, 6 pairings) and the held-out rounds 22-27 (+3,214; worst held-out round -5,549 -> -2,969).
# gpt-6.2 reads labour durations ("two technicians from eight to three") far better than gpt-4.1.
PRICE_BLEND_W = float(os.environ.get("PRICE_BLEND_W", "0.65"))
# Blind-assessor ensemble. One call gives a band the model invents; several independent calls
# give a band the DISAGREEMENT measures. Game 9 priced a bare "TV set" at 1500 +/-20% when the
# true t was ~566 -- the error was overconfidence, not a bad centre. Set 1 for a single call.
# Assessors run in parallel, so wall-clock cost is about one call.
# Recall (recall.py): on an invoice line that appeared in an earlier,
# settled game, charge at least RECALL_A_FRAC (0.9) x and accept at least up to RECALL_B_FRAC (1.0) x the
# largest charge the public log already proved fair. Never lowers a value. Team analysis games 31-62, each
# game using only earlier games: charge floor +45k (11 rounds better / 3 worse), accept floor +8.3k (11/4).
# The index is refreshed after every round by recall_refresh.py (spawned detached; read-only public data).
RECALL = os.environ.get("RECALL", "1") == "1"
# Slow-gateway guard. When no answer is back after HEDGE_AFTER s, a second, identical request is sent and the first
# valid answer wins. LLM_CAP bounds the call and the round deadline (ROUND_BUDGET) always wins. HEDGE_AFTER=0 = off.
# Shared gateways slow down at round start, when every team calls at once. With these defaults the model may run
# until about +47 s and the final submission goes out by about +48 s, well inside the 60 s round.
LLM_CAP = float(os.environ.get("LLM_CAP", "45"))
HEDGE_AFTER = float(os.environ.get("HEDGE_AFTER", "10"))
RECALL_REFRESH = os.environ.get("RECALL_REFRESH", "1") == "1"
# Recall ceiling (recall.py): inside a repeated case (>= 50% of the smaller invoice and >= 3 lines shared with an
# earlier game) never charge or accept at/above a level that case already proved inflated. Team analysis games
# 31-66, exactly this code: charge ceiling +21.1k (5 rounds better / 1 worse), limit ceiling +7.7k (6/1).
RECALL_CEIL = os.environ.get("RECALL_CEIL", "1") == "1"
MEM_MAX = int(os.environ.get("MEM_MAX", "0"))   # settled items injected into the prompt
N_ASSESSORS = int(os.environ.get("N_ASSESSORS", "3"))
ASSESSOR_TEMPS = [float(x) for x in os.environ.get("ASSESSOR_TEMPS", "0.0,0.7,1.0").split(",")]

DRY = os.environ.get("DRY_RUN") == "1"
ONLY = os.environ.get("ONLY_GAME")
ALLOW_LIVE = os.environ.get("ALLOW_LIVE") == "1"
LOCK_NAME = os.environ.get("LOCK_FILE", "bot.lock")

ROUND_BUDGET = float(os.environ.get("ROUND_BUDGET", "50"))   # seconds after key release (see LLM_CAP)
WINDOW = float(os.environ.get("WINDOW", "55"))               # seconds after start_time to stop trying
# Measured over games 5-9 on 121 charges we actually received: accepting everything
# cost 20,454 while rejecting everything cost 14,521 (robust under both bounds on the
# unknown-status charges). This field charges aggressively -- 49 of the 69 charges we
# could classify were above t -- so the accept-all fallback CLAUDE.md assumed
# ("beats default b=0 by a mile") is backwards here. Default flipped to 0.
PROVISIONAL_B = float(os.environ.get("PROVISIONAL_B", "0"))
# The issuer NEVER pays: if a > t the reviewer just rejects and we get 0, there is no penalty
# on the charging side. So a blind non-zero charge is free upside -- a=0 earns nothing by
# construction, while a=250 collects on every line where 250 <= t and costs nothing where it
# does not. Only ever in force if the LLM step fails or omits a line.
PROVISIONAL_A = float(os.environ.get("PROVISIONAL_A", "400"))
BLIND_MAX_IDX = int(os.environ.get("BLIND_MAX_IDX", "25"))
POLICY_CAP = int(os.environ.get("POLICY_CAP", "12000"))
INVOICE_CAP = int(os.environ.get("INVOICE_CAP", "15000"))

H = {"X-API-Key": TEAM_KEY}
LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True)


# ---------- logging ----------
class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        try:
            self.fh = open(path, "a", encoding="utf-8", errors="replace")
        except OSError:
            self.fh = None

    def write(self, s):
        self.stream.write(s)
        if self.fh:
            try:
                self.fh.write(s); self.fh.flush()
            except OSError:
                pass

    def flush(self):
        self.stream.flush()
        if self.fh:
            try:
                self.fh.flush()
            except OSError:
                pass


sys.stdout = _Tee(sys.stdout, os.path.join(LOGS, "console.log"))


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ---------- maths (unchanged decision rule) ----------
def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def Phi_inv(p):
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def decide(median, p10, p90, q, a_mult=None, b_mult=None, b_floor=None):
    """Return (charge a, acceptance limit b) as gross EUR line totals.

    Computed on the unbiased belief, then a *= A_MULT and b *= B_MULT. A multiplicative
    bias shifts the whole lognormal, so scaling the outputs is exact."""
    a_mult = A_MULT if a_mult is None else a_mult
    b_floor = B_FLOOR if b_floor is None else b_floor
    b_mult = B_MULT if b_mult is None else b_mult
    try:
        median = float(median or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not math.isfinite(median) or median <= 0:
        return 0.0, 0.0
    try:
        p10 = float(p10)
        if not math.isfinite(p10) or p10 <= 0:
            raise ValueError
    except (TypeError, ValueError):
        p10 = median * 0.7
    try:
        p90 = float(p90)
        if not math.isfinite(p90):
            raise ValueError
    except (TypeError, ValueError):
        p90 = median * 1.4
    p10 = min(max(p10, 1e-6), median)
    p90 = max(p90, median)
    s = (math.log(p90) - math.log(p10)) / 2.563 * SIGMA_MULT
    s = min(max(s, SIGMA_MIN), SIGMA_MAX)
    mu = math.log(median)
    # handyman: argmax a*(1-Phi((ln a - mu)/s)) ; independent of q
    best_a, best_v = median, -1.0
    for k in range(-300, 101):
        z = k / 100
        a = math.exp(mu + s * z)
        v = a * (1 - Phi(z))
        if v > best_v:
            best_a, best_v = a, v
    # insurer: b = x with q*(1-Phi(z)) = 2/3
    try:
        q = float(q)
    except (TypeError, ValueError):
        q = 0.9
    if not math.isfinite(q):
        q = 0.9
    q = min(max(q, 0.0), 1.0)
    if q < 2 / 3:
        b = 0.0
    else:
        z = Phi_inv(1 - 2 / (3 * q))
        b = math.exp(mu + s * z)
        if b_floor > 0 and q >= B_FLOOR_Q:
            b = max(b, median * b_floor)
    best_a = min(best_a, median * A_CAP_MULT)
    a, b = round(best_a * a_mult, 2), round(b * b_mult, 2)
    if not (math.isfinite(a) and a >= 0):
        a = 0.0
    if not (math.isfinite(b) and b >= 0):
        b = 0.0
    return a, b


# ---------- HTTP with retries ----------
def http(method, url, tries=3, timeout=5, **kw):
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, url, headers=H, timeout=timeout, **kw)
            if r.status_code < 500:
                return r
            last = f"{r.status_code}"
        except Exception as e:  # noqa: BLE001 - network layer, keep going
            last = repr(e)
        time.sleep(0.3 * (i + 1))
    log("http give up", method, url, last)
    return None


# ---------- index extraction ----------
STOP_HEAD = re.compile(
    r"\s*(INVOICE\b|Created on|Page\s*\d|TOTAL\b|SUBTOTAL|Sub-?total|VAT\b|Summe|Gesamt|"
    r"Zwischensumme|Net\b|Brutto|Netto|Thank you|Payment|Bank)", re.I)


def _run_from_one(nums):
    """Longest run 1,2,3,... consumable in order from a noisy number list."""
    want = 1
    for n in nums:
        if n == want:
            want += 1
    return list(range(1, want))


def idx_items_block(inv):
    """Numbers at line start inside the ITEMS block only."""
    lines = inv.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if re.search(r"\bPOS\b", l, re.I) and re.search(r"DESCRIPTION", l, re.I):
            start = i + 1
            break
    end = len(lines)
    for i in range(start, len(lines)):
        if STOP_HEAD.match(lines[i]):
            end = i
            break
    nums = []
    for l in lines[start:end]:
        m = re.match(r"\s*(\d{1,3})[\s.)]", l)
        if m:
            n = int(m.group(1))
            if 0 < n < 200:
                nums.append(n)
    return _run_from_one(nums)


def idx_legacy(inv):
    """bot.py's original rule: contiguous set of line-leading numbers anywhere."""
    nums = sorted({int(m) for m in re.findall(r"(?m)^\s*(\d{1,3})[\s.)]", inv) if 0 < int(m) < 200})
    n = 0
    while n + 1 in nums:
        n += 1
    return list(range(1, n + 1))


def idx_pos_anywhere(inv):
    """Numbers followed by a letter-ish description on the same line, anywhere."""
    nums = []
    for l in inv.splitlines():
        m = re.match(r"\s*(\d{1,3})[\s.)]+\s*[A-Za-zÄÖÜäöüß]", l)
        if m:
            n = int(m.group(1))
            if 0 < n < 200:
                nums.append(n)
    return _run_from_one(nums)


def guess_indices(inv):
    """Union of three extractors -- over-covering is cheap, missing a line is not."""
    cands = [idx_items_block(inv), idx_legacy(inv), idx_pos_anywhere(inv)]
    n = max((len(c) for c in cands), default=0)
    return list(range(1, n + 1)), {"items_block": len(cands[0]), "legacy": len(cands[1]),
                                   "pos_anywhere": len(cands[2])}


def invoice_line_texts(inv):
    """Rough per-line descriptions, used only to score policy sections."""
    out = []
    for l in inv.splitlines():
        m = re.match(r"\s*(\d{1,3})[\s.)]+\s*(.{4,})", l)
        if m:
            out.append(m.group(2))
    return out


# ---------- policy excerpting ----------
SEC_SPLIT = re.compile(r"(?m)^(?=(?:PART\s+\d+|\d+\.\d+(?:\.\d+)?\s))")
WORD = re.compile(r"[a-zäöüß]{4,}")
STOPWORDS = set("""this that with from have been were will shall which their there under
where when they them then than such been also into more most other some only same both
each been does done must were above below after before during other than upon paid pays
policy insured insurer insurance cover covered conditions contract clause part section
schedule stated shown form parts terms""".split())
BOOST = re.compile(
    r"exclusion|excluded|not insured|no indemnit|limit|sub-limit|deductible|excess|"
    r"indemnit|valuation|replacement|depreciat|new for old|repair cost|portable|"
    r"accidental damage|wear and tear|maintenance|servicing", re.I)


def policy_excerpt(policy, desc, inv, cap=None):
    cap = cap or POLICY_CAP
    if len(policy) <= cap:
        return policy, {"mode": "full", "chars": len(policy)}
    parts = SEC_SPLIT.split(policy)
    if len(parts) < 4:
        return policy[:cap], {"mode": "truncate", "chars": cap}
    query = (desc + "\n" + "\n".join(invoice_line_texts(inv))).lower()
    qtok = {w for w in WORD.findall(query) if w not in STOPWORDS}
    scored = []
    for i, p in enumerate(parts):
        toks = {w for w in WORD.findall(p.lower()) if w not in STOPWORDS}
        hits = len(qtok & toks)
        score = hits / math.sqrt(max(len(p), 200))
        if BOOST.search(p[:200]):
            score += 0.05
        scored.append((score, i, p))
    keep = {0}                                   # always keep the preamble/definitions
    budget = cap - min(len(parts[0]), 2500)
    for score, i, p in sorted(scored, key=lambda x: -x[0]):
        if i in keep:
            continue
        if len(p) > budget:
            continue
        keep.add(i)
        budget -= len(p)
        if budget <= 0:
            break
    out, prev = [], -1
    for i in sorted(keep):
        if prev >= 0 and i != prev + 1:
            out.append("\n[... policy sections omitted ...]\n")
        out.append(parts[i][:2500] if i == 0 else parts[i])
        prev = i
    txt = "".join(out)[:cap]
    return txt, {"mode": "excerpt", "chars": len(txt), "sections_kept": len(keep),
                 "sections_total": len(parts), "orig_chars": len(policy)}


# ---------- case handling ----------
def case_path(gid):
    for name in (f"case_{gid:03d}.zip", f"case_{gid}.zip"):
        p = os.path.join(CASES, name)
        if os.path.exists(p):
            return p
    return None


def unpack(path, key):
    out = {}
    with pyzipper.AESZipFile(path) as z:
        z.setpassword(key.encode())
        for n in z.namelist():
            out[n] = z.read(n)
    return out


def pdf_text(b):
    if not b:
        return ""
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(b)).pages)


# ---------- LLM ----------
PROMPT = """You are a senior German insurance claims adjuster (property / water damage).
For EVERY line item on the invoice, return JSON only:
{"items":[{"index":int,"text":str,"quantity":float,"unit":str,
  "p_valid":float,   // probability the item is covered by the policy AND related to the described damage
  "p10":float,"median":float,"p90":float}]}  // fair GROSS line total in EUR (quantity x unit price, incl. 19% VAT), as a claims expert would accept, assuming it is valid
Rules:
- index = the position number printed on the invoice. Return one object per printed position, no more, no fewer.
- Prices are for the WHOLE line incl. VAT, never per unit, never net.
- Use realistic 2026 German trade prices (labour, materials, drying, disposal).
- If the damage description or the invoice names an actual price, purchase price or sum insured, that is strong
  evidence: anchor the median on it rather than on a generic estimate.
- p10 and p90 are your genuine 10th/90th percentiles. Do not collapse them; if you are unsure, widen them.
- CRITICAL: if the line gives no make, model, size, spec or quantity detail (bare entries like
  "TV set", "Speaker system", "Laptop"), you genuinely cannot know its value. Anchor the median on a
  TYPICAL MID-MARKET item, never a premium one, and set p10 <= median/2 and p90 >= median*2.
  A narrow band on an unspecified item is a serious error.
- Where the policy sets an indemnity basis (current value / depreciation for age and wear) rather than
  replacement cost, price the indemnity, not the new-for-old price.
- Ancillary charges raised alongside a repair (carriage, freight, delivery, packaging, transport,
  call-out, travel, fitting, mounting, commissioning, configuration, administration, handling) are
  frequently NOT covered: judge them against the policy and set p_valid accordingly.
- p_valid is ONLY the probability that the item is covered at all AND related to the described damage.
  Set it low only for a clear policy exclusion or an item unrelated to the damage. If the item is covered
  but limited (a cap, a deductible, depreciation, or a like-for-like replacement of an upgraded item), it IS
  covered: keep p_valid high and price only the covered amount (e.g. the like-for-like value, not the
  upgrade). Doubt, missing paperwork or a reviewer's note is not an exclusion.
- The policy below may be excerpted; sections marked omitted were judged irrelevant.

{memory}
POLICY:
{policy}

DAMAGE DESCRIPTION:
{desc}

INVOICE (prices removed):
{inv}
"""


MEM_WORD = re.compile(r"[a-zA-ZäöüÄÖÜß]{3,}")
MEM_STOP = set("the and for with from into per pcs hrs flat rate incl new all one two".split())


def _mem_tokens(t):
    return {w.lower() for w in MEM_WORD.findall(t or "")} - MEM_STOP


def load_memory():
    """Settled t bounds recovered from earlier rounds. Missing or broken file -> no memory."""
    p = os.path.join(HERE, "data", "price_memory.json")
    try:
        rows = json.load(open(p, encoding="utf-8"))["rows"]
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in rows:
        lo, hi = r.get("t_lo"), r.get("t_hi")
        if lo is None and hi is None:
            continue
        out.append({"text": str(r.get("text") or "")[:160], "lo": lo, "hi": hi,
                    "toks": _mem_tokens(r.get("text"))})
    return out


def memory_block(line_texts, mem, k=MEM_MAX):
    """Rank settled items by word overlap with this invoice, most relevant first."""
    if not mem:
        return ""
    want = set()
    for t in line_texts:
        want |= _mem_tokens(t)
    scored = []
    for r in mem:
        inter = len(want & r["toks"])
        score = inter / math.sqrt(len(r["toks"]) + 1) if inter else 0.0
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    lines = []
    for score, r in scored[:k]:
        lo, hi = r["lo"], r["hi"]
        if lo is not None and hi is not None:
            bound = f"between EUR {lo:.0f} and EUR {hi:.0f}"
        elif hi is not None:
            bound = f"no more than EUR {hi:.0f}"
        else:
            bound = f"at least EUR {lo:.0f}"
        lines.append(f'- "{r["text"][:90]}": fair value {bound}')
    if not lines:
        return ""
    return ("\nSETTLED FAIR VALUES FROM EARLIER ROUNDS OF THIS SAME GAME (ground truth, "
            "recovered from the published settlements):\n" + "\n".join(lines) +
            "\nThese are what the adjudicator actually treated as fair. In this game fair values run "
            "WELL BELOW normal German market prices. Calibrate your price level to these figures, "
            "not to what the work would cost in the real world.\n")


def _extract_json(txt):
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON in LLM reply")


def llm_items(policy, desc, inv, timeout, temperature=0.0, memory="", model=None):
    from openai import OpenAI
    cli = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE) if LLM_BASE else OpenAI(api_key=LLM_KEY)
    msg = (PROMPT.replace("{policy}", policy).replace("{desc}", desc)
           .replace("{inv}", inv[:INVOICE_CAP]).replace("{memory}", memory))
    mdl = model or LLM_MODEL
    kw = dict(model=mdl, timeout=timeout, messages=[{"role": "user", "content": msg}])
    if mdl.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        kw["reasoning_effort"] = "low"          # reasoning models reject a temperature setting
    else:
        kw["temperature"] = temperature
    try:
        r = cli.chat.completions.create(response_format={"type": "json_object"}, **kw)
    except Exception as e:  # noqa: BLE001 - some gateways reject response_format
        if "Timeout" in type(e).__name__:
            raise          # a timed-out call is not a format problem; a plain retry only burns the round's time
        log("response_format rejected, retrying plain:", repr(e)[:120])
        r = cli.chat.completions.create(**kw)
    usage = getattr(r, "usage", None)
    tok = {"prompt": getattr(usage, "prompt_tokens", None),
           "completion": getattr(usage, "completion_tokens", None)} if usage else {}
    return _extract_json(r.choices[0].message.content)["items"], tok


def _med(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def merge_coverage(items, judge):
    """Coverage from the judge, price from the ensemble.

    judge says excluded/unrelated (p_valid < 2/3): take the judge's line as is.
    judge says covered: keep the ensemble's price, use the judge's p_valid.
    A line only one side has is kept from that side. Never raises.
    """
    try:
        P = {int(x["index"]): x for x in (items or []) if isinstance(x, dict)}
        C = {}
        for x in judge or []:
            try:
                q = float(x.get("p_valid"))
                med = float(x.get("median"))
                if math.isfinite(q) and math.isfinite(med) and med >= 0:
                    C[int(x["index"])] = dict(x, p_valid=min(max(q, 0.0), 1.0))
            except (TypeError, ValueError, KeyError):
                continue
        out, changed = [], 0
        for i in sorted(set(P) | set(C)):
            p, c = P.get(i), C.get(i)
            if c is None:
                out.append(p); continue
            if p is None:
                out.append(c); continue
            if c["p_valid"] < 2 / 3:
                changed += p.get("p_valid", 0.9) >= 2 / 3
                out.append(dict(c, judged_by="coverage_model"))
            else:
                changed += p.get("p_valid", 0.9) < 2 / 3
                x = dict(p, p_valid=c["p_valid"], judged_by="coverage_model")
                mp, mc = float(p.get("median") or 0), float(c.get("median") or 0)
                if PRICE_BLEND and mp > 0 and mc > 0:
                    w = min(max(PRICE_BLEND_W, 0.0), 1.0)
                    k = math.exp((1 - w) * math.log(mp) + w * math.log(mc)) / mp
                    x["median"] = round(mp * k, 2)
                    x["p10"] = round(float(p.get("p10") or mp * 0.7) * k, 2)
                    x["p90"] = round(float(p.get("p90") or mp * 1.4) * k, 2)
                    x["blend_factor"] = round(k, 3)
                out.append(x)
        return out, changed
    except Exception:  # noqa: BLE001
        return items, 0


def aggregate(runs):
    """Combine several assessors into one belief per line item.

    centre = median of the assessors' medians (robust to one wild voice)
    sigma  = sqrt(within^2 + between^2): each assessor's own band, widened by how much the
             assessors DISAGREE. Unanimity keeps it tight; a split opens it up.
    """
    by_idx = {}
    for items in runs:
        if not isinstance(items, list):
            continue
        seen = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                i = int(it["index"])
                m = float(it["median"])
            except (TypeError, ValueError, KeyError):
                continue
            if i in seen or not math.isfinite(m) or m < 0:
                continue
            seen.add(i)
            by_idx.setdefault(i, []).append(it)
    out = []
    for i, lst in sorted(by_idx.items()):
        meds = [float(x["median"]) for x in lst]
        qs = []
        for x in lst:
            try:
                q = float(x.get("p_valid", 0.9))
                if math.isfinite(q):
                    qs.append(min(max(q, 0.0), 1.0))
            except (TypeError, ValueError):
                pass
        centre = _med(meds)
        q = _med(qs) if qs else 0.9
        base = {"index": i, "text": str(lst[0].get("text", ""))[:200], "p_valid": q,
                "n_votes": len(lst), "votes": [round(v, 2) for v in meds]}
        if centre is None or centre <= 0:
            base.update(median=0.0, p10=0.0, p90=0.0)
            out.append(base)
            continue
        within = []
        for x in lst:
            try:
                lo, hi = float(x["p10"]), float(x["p90"])
                if math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi:
                    within.append((math.log(hi) - math.log(lo)) / 2.563)
            except (TypeError, ValueError, KeyError):
                pass
        s_within = _med(within) or 0.25
        pos = [m for m in meds if m > 0]
        if len(pos) > 1:
            lm = [math.log(m) for m in pos]
            mu = sum(lm) / len(lm)
            s_between = math.sqrt(sum((x - mu) ** 2 for x in lm) / (len(lm) - 1))
        else:
            s_between = 0.0
        # assessors that priced a line at 0 while others priced it positive are a coverage
        # disagreement: that is extra uncertainty, not a zero price
        if len(pos) < len(meds):
            s_between = max(s_between, 0.35)
        s = math.sqrt(s_within ** 2 + s_between ** 2)
        base.update(median=round(centre, 2),
                    p10=round(centre * math.exp(-1.2816 * s), 2),
                    p90=round(centre * math.exp(1.2816 * s), 2),
                    sigma_within=round(s_within, 3), sigma_between=round(s_between, 3))
        out.append(base)
    return out


def llm_ensemble(policy, desc, inv, timeout, n, memory=""):
    """Run n assessors in parallel; use whichever return before the deadline."""
    runs, toks, errs = [], [], []
    lock = threading.Lock()

    def one(temp):
        try:
            items, tok = llm_items(policy, desc, inv, timeout, temperature=temp, memory=memory)
            with lock:
                runs.append(items)
                toks.append(tok)
        except Exception:  # noqa: BLE001
            with lock:
                errs.append(traceback.format_exc().splitlines()[-1][:160])

    temps = (ASSESSOR_TEMPS * n)[:n]
    th = [threading.Thread(target=one, args=(t,), daemon=True) for t in temps]
    for t in th:
        t.start()
    end = time.monotonic() + timeout
    for t in th:
        t.join(max(0.1, end - time.monotonic()))
    tot = {"prompt": sum((t or {}).get("prompt") or 0 for t in toks),
           "completion": sum((t or {}).get("completion") or 0 for t in toks),
           "calls": len(runs)}
    return runs, tot, errs


def llm_async(policy, desc, inv, timeout, memory="", hedge_after=None):
    """Run the LLM call in daemon threads; a hang can never block the final PUT. If no answer is back after
    hedge_after s (and enough time is left), one identical second request is sent; first valid answer wins."""
    hedge_after = HEDGE_AFTER if hedge_after is None else hedge_after
    t_end = time.monotonic() + timeout
    boxes = []

    def start():
        b = {"t0": time.monotonic()}

        def run():
            try:
                b["items"], b["tokens"] = llm_items(policy, desc, inv, max(1.0, t_end - time.monotonic()),
                                                    memory=memory)
            except Exception:  # noqa: BLE001
                b["error"] = traceback.format_exc()
            b["done"] = True
        th = threading.Thread(target=run, daemon=True)
        th.start()
        boxes.append(b)

    start()
    hedged = False
    while time.monotonic() < t_end:
        win = next((b for b in boxes if b.get("done") and b.get("items")), None)
        if win is not None:
            out = {"items": win["items"], "tokens": win.get("tokens"),
                   "secs": round(time.monotonic() - boxes[0]["t0"], 1), "hedged": hedged,
                   "winner": boxes.index(win)}
            return out
        if all(b.get("done") for b in boxes) and (hedged or not hedge_after):
            break                                   # every request failed: no point waiting
        if (hedge_after and not hedged and time.monotonic() - boxes[0]["t0"] >= hedge_after
                and t_end - time.monotonic() > 8.0):
            hedged = True
            log(f"LLM slow (> {hedge_after:.0f}s): sending one hedged second request")
            start()
        elif (hedge_after and not hedged and boxes[0].get("done") and t_end - time.monotonic() > 8.0):
            hedged = True                           # first request failed fast: retry once at once
            log("LLM request failed, sending one retry:", (boxes[0].get("error") or "").strip().splitlines()[-1:])
            start()
        time.sleep(0.1)
    err = next((b.get("error") for b in boxes if b.get("error")), None)
    return {"error": err or f"LLM exceeded {timeout:.1f}s deadline, abandoned", "hedged": hedged}


# ---------- validation ----------
def validate(items, expected_idx):
    """Return (submissions, report). Every expected index gets a row, no exceptions."""
    rep = {"raw": len(items) if isinstance(items, list) else 0, "dropped": [], "filled": [],
           "clamped": []}
    by_idx = {}
    if isinstance(items, list):
        for it in items:
            if not isinstance(it, dict):
                rep["dropped"].append("non-dict"); continue
            try:
                i = int(it["index"])
            except Exception:  # noqa: BLE001
                rep["dropped"].append("bad index"); continue
            if not (0 < i < 200):
                rep["dropped"].append(f"index {i} out of range"); continue
            if i in by_idx:
                rep["dropped"].append(f"duplicate index {i}"); continue
            med, p10, p90 = it.get("median"), it.get("p10"), it.get("p90")
            try:
                med = float(med)
            except (TypeError, ValueError):
                med = float("nan")
            if not math.isfinite(med) or med < 0:
                rep["dropped"].append(f"index {i} bad median"); continue
            # median must lie inside [p10, p90]; clamp rather than discard
            try:
                fp10, fp90 = float(p10), float(p90)
                if not (math.isfinite(fp10) and math.isfinite(fp90)) or not (fp10 <= med <= fp90):
                    rep["clamped"].append(i)
                    fp10, fp90 = min(fp10 if math.isfinite(fp10) else med * 0.7, med), \
                                 max(fp90 if math.isfinite(fp90) else med * 1.4, med)
            except (TypeError, ValueError):
                rep["clamped"].append(i)
                fp10, fp90 = med * 0.7, med * 1.4
            a, b = decide(med, fp10, fp90, it.get("p_valid", 0.9))
            it["a"], it["b"] = a, b
            by_idx[i] = {"index": i, "charge_price": a, "acceptance_limit": b}
    for i in expected_idx:
        if i not in by_idx:
            rep["filled"].append(i)
            by_idx[i] = {"index": i, "charge_price": PROVISIONAL_A, "acceptance_limit": PROVISIONAL_B}
    subs = [by_idx[i] for i in sorted(by_idx)]
    rep["submitted"] = len(subs)
    rep["indices"] = [s["index"] for s in subs]
    return subs, rep


# ---------- submission ----------
def verify_ack(gid, body, r):
    """Return None if the API acknowledgement matches what we sent, else a short reason.

    Never raises and never fails a round on its own: a stored submission with an odd
    acknowledgement is still better than no submission.
    """
    try:
        ack = r.json()
    except Exception:  # noqa: BLE001
        return "unparseable body"
    if not isinstance(ack, list):
        return f"not a list ({type(ack).__name__})"
    sent = {int(x["index"]): x for x in body}
    seen = {}
    for a in ack:
        if not isinstance(a, dict):
            return "non-dict row"
        if a.get("game_id") not in (None, gid):
            return f"game_id {a.get('game_id')} != {gid}"
        try:
            seen[int(a["line_item_index"])] = a
        except Exception:  # noqa: BLE001
            return "row without line_item_index"
    missing = sorted(set(sent) - set(seen))
    if missing:
        return f"{len(missing)} index(es) not stored: {missing[:6]}"
    for i, x in sent.items():
        a = seen[i]
        for fld in ("charge_price", "acceptance_limit"):
            try:
                if abs(float(a[fld]) - float(x[fld])) > 0.011:
                    return f"index {i} {fld} {a[fld]} != {x[fld]}"
            except Exception:  # noqa: BLE001
                return f"index {i} {fld} unreadable"
    return None


def submit(gid, subs, kind, rec, deadline=None, retry_403_until=None):
    """PUT with a fallback ladder; records status in rec['submissions'].

    retry_403_until: monotonic time to keep retrying a 403 "Game has not started yet".
    The blind safety net arms a few seconds BEFORE start_time, so without this it
    always 403s and never actually protects the round."""
    entry = {"kind": kind, "n": len(subs), "at": dt.datetime.now().isoformat(timespec="seconds")}
    if not subs:
        entry["status"] = "skipped-empty"
        rec.setdefault("submissions", []).append(entry)
        return False
    if DRY:
        log("DRY submit", kind, gid, len(subs), "items", subs[:2])
        entry["status"] = "dry"
        rec.setdefault("submissions", []).append(entry)
        return True
    ladders = [subs]
    if len(subs) > 1:
        half = [s for s in subs if s["index"] <= max(1, len(subs) // 2)]
        ladders.append(half)
        ladders.append(subs[:1])
    for attempt, body in enumerate(ladders):
        if deadline and time.monotonic() > deadline:
            entry["status"] = "deadline"
            break
        r = http("PUT", f"{BASE}/api/games/{gid}/submissions", tries=3, timeout=6, json=body)
        # wait out the pre-open 403 rather than burning the attempt
        n403 = 0
        while (r is not None and r.status_code == 403 and retry_403_until
               and time.monotonic() < retry_403_until
               and "not started" in r.text.lower()):
            n403 += 1
            time.sleep(0.4)
            r = http("PUT", f"{BASE}/api/games/{gid}/submissions", tries=1, timeout=4, json=body)
        if n403:
            entry["waited_403"] = n403
        if r is None:
            entry["status"] = "no-response"
            continue
        log("submit", kind, gid, r.status_code,
            f"{len(body)} items" if r.status_code == 200 else r.text[:160])
        entry["status"] = r.status_code
        entry["n"] = len(body)
        if r.status_code == 200:
            # A 200 is not proof the server stored what we sent. Check the acknowledgement
            # echoes every index we submitted, for THIS game, with the same amounts. A silent
            # truncation would otherwise look like a clean round in the log.
            bad = verify_ack(gid, body, r)
            entry["ack"] = "ok" if not bad else bad
            if bad:
                log("  ACK MISMATCH:", bad)
                entry.setdefault("errors", []).append(f"ack:{bad}")
            rec.setdefault("submissions", []).append(entry)
            return True
        entry.setdefault("errors", []).append(f"{r.status_code}:{r.text[:120]}")
        if r.status_code in (401, 403, 404):
            break        # narrowing the body will not help
    rec.setdefault("submissions", []).append(entry)
    return False


def write_log(rec):
    gid = rec["game"]
    path = os.path.join(LOGS, f"game_{gid:03d}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def append_round(rec):
    """One compact append-only line per round -- the uptime record for the write-up."""
    subs = rec.get("submissions", [])
    row = {
        "game": rec["game"], "start": rec.get("start"),
        "ok_provisional": any(s["kind"] in ("blind", "provisional") and s.get("status") in (200, "dry")
                              for s in subs),
        "ok_final": any(s["kind"] == "final" and s.get("status") in (200, "dry") for s in subs),
        "statuses": [[s["kind"], s.get("status"), s.get("n")] for s in subs],
        "ack": [s.get("ack") for s in subs if s.get("ack") not in (None, "ok")] or "ok",
        "n_lines": rec.get("n_lines"),
        "elapsed_s": rec.get("elapsed_s"),
        "tokens": rec.get("tokens"),
        "error": (rec.get("error") or "").splitlines()[-1][:160] if rec.get("error") else None,
    }
    with open(os.path.join(LOGS, "rounds.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


# ---------- one round ----------
def recall_step(gid, inv, subs, rec):
    """Apply the recall floors to the final submission. Never fails the round: on any problem the
    submission goes out unchanged."""
    try:
        import recall as _rc
        index, gkeys = _rc.load_full()
        new, changes = _rc.apply(subs, inv, gid, index, gkeys if RECALL_CEIL else None)
        rec["recall"] = {"changes": changes, "index_lines": len(index)}
        if changes and "error" in changes[0]:
            log("recall skipped:", changes[0]["error"])
            return subs
        if changes:
            log(f"recall: {len(changes)} line(s) raised: " + ", ".join(
                f"#{c['index']} a {c['a'][0]:.0f}->{c['a'][1]:.0f} b {c['b'][0]:.0f}->{c['b'][1]:.0f}"
                + (f" (ceiling {c['hi']:.0f})" if c.get("hi") else "")
                for c in changes[:10]))
        else:
            log(f"recall: no recurring line with a proven fair value ({len(index)} lines indexed)")
        return new
    except Exception as e:  # noqa: BLE001 - recall must never cost the round
        rec["recall"] = {"error": repr(e)[:160]}
        log("recall not used:", repr(e)[:120])
        return subs


def play(gid, start_iso, start_dt):
    t0 = time.monotonic()
    rec = {"game": gid, "start": start_iso, "submissions": []}
    # Absolute end of our attempt window. Floored at ROUND_BUDGET+10s rather than the
    # wall-clock remainder: a PUT after the real close just 403s (harmless), whereas a fast
    # local clock would otherwise silently cut the LLM out of the round. Also makes
    # ONLY_GAME=0 replays of long-past games usable.
    remaining = (start_dt + dt.timedelta(seconds=WINDOW)
                 - dt.datetime.now(dt.timezone.utc)).total_seconds()
    window_end = t0 + max(ROUND_BUDGET + 10.0, remaining)

    # 0. blind pre-submission: never leave the round at the a=0/b=0 default
    blind = [{"index": i, "charge_price": PROVISIONAL_A, "acceptance_limit": PROVISIONAL_B}
             for i in range(1, BLIND_MAX_IDX + 1)]
    open_by = t0 + max(8.0, (start_dt - dt.datetime.now(dt.timezone.utc)).total_seconds() + 6.0)
    blind_ok = submit(gid, blind, "blind", rec, deadline=window_end, retry_403_until=open_by)

    # 1. key (poll until released, never past the window)
    key = None
    while time.monotonic() < window_end:
        r = http("GET", f"{BASE}/api/games/{gid}/key", tries=1, timeout=3)
        if r is not None and r.status_code == 200:
            try:
                key = r.json()["decryption_key"]
                break
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.4)
    if not key:
        rec["error"] = "no key before window close"
        log("no key for", gid)
        rec["elapsed_s"] = round(time.monotonic() - t0, 1)
        write_log(rec); log("round", gid, append_round(rec)); return
    key_t = time.monotonic()
    deadline = min(key_t + ROUND_BUDGET, window_end)
    rec["key_after_s"] = round(key_t - t0, 1)

    # 2. decrypt + extract
    try:
        path = case_path(gid)
        if not path:
            raise FileNotFoundError(f"no case file for game {gid} in {CASES}")
        files = unpack(path, key)
        get = lambda suf: next((v for k, v in files.items() if k.endswith(suf)), b"")  # noqa: E731
        policy = get("policy.txt").decode("utf-8", "ignore")
        desc = get("description.txt").decode("utf-8", "ignore")
        inv = pdf_text(get("invoices.pdf"))
    except Exception as e:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        log("case unpack failed, blind submission stands:", e)
        rec["elapsed_s"] = round(time.monotonic() - t0, 1)
        write_log(rec); log("round", gid, append_round(rec)); return
    rec.update(policy=policy, desc=desc, invoice=inv,
               files=sorted(files), has_images=any(k.lower().endswith((".png", ".jpg")) for k in files))

    # 3. provisional over the real indices
    idx, idx_detail = guess_indices(inv)
    rec["indices_guessed"] = idx
    rec["index_extractors"] = idx_detail
    if idx and (not blind_ok or max(idx) > BLIND_MAX_IDX):
        submit(gid, [{"index": i, "charge_price": PROVISIONAL_A, "acceptance_limit": PROVISIONAL_B} for i in idx],
               "provisional", rec, deadline=deadline)
    elif idx:
        log(f"provisional skipped: blind 1..{BLIND_MAX_IDX} already covers {idx}")
    else:
        log("WARNING: no indices extracted for game", gid, "- blind set stands")

    # 4. one LLM call, excerpted policy, hard deadline
    pol, pol_meta = policy_excerpt(policy, desc, inv)
    rec["policy_excerpt"] = pol_meta
    try:
        mem = load_memory()
        memblock = memory_block(invoice_line_texts(inv), mem)
    except Exception:  # noqa: BLE001
        mem, memblock = [], ""
    rec["memory_items"] = len(mem)
    rec["memory_chars"] = len(memblock)
    budget = deadline - time.monotonic() - 6.0        # leave 6s for validation + final PUT
    if budget < 3:
        rec["error"] = "no time for LLM call"
        log("no LLM budget for", gid)
    else:
        judge_box = {}
        judge_th = None
        if COVERAGE_MODEL:
            def _judge():
                t0j = time.monotonic()
                try:
                    judge_box["items"], _ = llm_items(pol, desc, inv, min(budget, LLM_CAP), 0.0, memblock,
                                                      model=COVERAGE_MODEL)
                except Exception as e:  # noqa: BLE001
                    judge_box["error"] = repr(e)[:160]
                judge_box["secs"] = round(time.monotonic() - t0j, 1)
            judge_th = threading.Thread(target=_judge, daemon=True)
            judge_th.start()
        if N_ASSESSORS > 1:
            runs, tok, errs = llm_ensemble(pol, desc, inv, min(budget, LLM_CAP), N_ASSESSORS, memblock)
            rec["tokens"] = tok
            rec["assessor_errors"] = errs
            if not runs:
                rec["error"] = "; ".join(errs) or "all assessors failed"
                log("all assessors failed, provisional stands")
                items = []
            else:
                items = aggregate(runs)
                log(f"ensemble: {len(runs)}/{N_ASSESSORS} voices, {len(items)} lines, "
                    f"{tok['prompt'] + tok['completion']} tokens")
                if judge_th is not None:
                    judge_th.join(max(0.0, deadline - time.monotonic() - 6.0))
                    if judge_box.get("items"):
                        items, changed = merge_coverage(items, judge_box["items"])
                        log(f"coverage judge {COVERAGE_MODEL}: {judge_box.get('secs')}s, "
                            f"{changed} coverage call(s) changed")
                    else:
                        log(f"coverage judge {COVERAGE_MODEL} not used: "
                            f"{judge_box.get('error') or 'not back in time'}")
                    rec["coverage_judge"] = {"model": COVERAGE_MODEL, "secs": judge_box.get("secs"),
                                             "used": bool(judge_box.get("items")),
                                             "error": judge_box.get("error")}
                if errs:
                    log(f"  {len(errs)} assessor(s) failed: {errs[0][:100]}")
        else:
            box = llm_async(pol, desc, inv, min(budget, LLM_CAP), memblock)
            rec["tokens"] = box.get("tokens")
            rec["llm"] = {k: box.get(k) for k in ("secs", "hedged", "winner")}
            if box.get("hedged") and box.get("items"):
                log(f"LLM answer after {box.get('secs')}s from request #{(box.get('winner') or 0) + 1} (hedged)")
            if "error" in box:
                rec["error"] = box["error"]
                log("LLM failed, provisional stands:", box["error"].splitlines()[-1][:160])
            items = box.get("items", [])
        subs, vrep = validate(items, idx)
        rec["items"] = items if isinstance(items, list) else []
        rec["validation"] = vrep
        if vrep["dropped"] or vrep["filled"] or vrep["clamped"]:
            log("validation:", json.dumps(vrep)[:300])
        if RECALL and subs:
            subs = recall_step(gid, inv, subs, rec)
        if subs:
            submit(gid, subs, "final", rec, deadline=window_end)

    rec["n_lines"] = len(idx)
    rec["elapsed_s"] = round(time.monotonic() - t0, 1)
    write_log(rec)
    row = append_round(rec)
    # Read-only leaderboard snapshot, detached, strictly AFTER both PUTs. Wrapped so
    # it can never affect a submission. Keeps data/ current for calibration overnight.
    # Recall refresh, detached, strictly AFTER the final PUT: waits until this round has settled, then
    # pulls the public log + released case and rebuilds data/recall_index.json. Never submits.
    if RECALL and RECALL_REFRESH and not DRY:
        try:
            import subprocess
            script = os.path.join(HERE, "recall_refresh.py")
            if os.path.exists(script):
                subprocess.Popen([sys.executable, script, str(gid)], cwd=HERE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            log("recall refresh spawn failed (harmless):", repr(e)[:120])
    if COLLECT_AFTER_ROUND and not DRY:
        try:
            import subprocess
            script = os.path.join(HERE, "collect_leaderboard.py")
            if os.path.exists(script):
                subprocess.Popen([sys.executable, "-I", script], cwd=HERE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:  # noqa: BLE001
            log("collector spawn failed (harmless):", repr(e)[:120])
    log(f"round {gid} done in {rec['elapsed_s']}s  provisional={row['ok_provisional']} "
        f"final={row['ok_final']} lines={row['n_lines']} tokens={row['tokens']}")


# ---------- loop ----------
def acquire_lock(path):
    """Take an OS-level advisory lock on `path`, or return None if another process holds it.

    An O_EXCL lockfile leaves a stale file behind when the process is killed hard or the
    machine loses power, and the next start then refuses to run until someone deletes it --
    the worst possible failure at 3am. An OS lock is released by the kernel when the holding
    process dies, so a restart always succeeds and a live double-submitter remains impossible.
    The handle must stay open for the life of the process; the caller keeps it.
    """
    # Transition guard: an older bot version held an O_EXCL lockfile containing its PID and
    # takes NO OS lock, so the OS lock below cannot see it. If the file names a live PID
    # other than ours, refuse -- this is the one case where two submitters could overlap.
    try:
        prev = int((open(path).read().strip() or "0"))
    except (OSError, ValueError):
        prev = 0
    if prev and prev != os.getpid() and pid_alive(prev):
        log(f"lockfile names live PID {prev} (older bot still running); refusing")
        return None
    try:
        fh = open(path, "a+b")
    except OSError as e:  # noqa: BLE001
        log("cannot open lockfile:", repr(e)[:120])
        return None
    try:
        fh.seek(0)
        if not fh.read(1):
            fh.write(b"0")
            fh.flush()
        fh.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None            # a live process holds it
    except Exception as e:  # noqa: BLE001 - no locking primitive available
        log("OS lock unavailable, falling back to PID check:", repr(e)[:100])
        try:
            prev = int(open(path + ".pid").read().strip())
        except (OSError, ValueError):
            prev = None
        if prev and pid_alive(prev):
            fh.close()
            return None
    try:
        with open(path + ".pid", "w") as f:
            f.write(str(os.getpid()))
    except OSError:
        pass
    return fh


def release_lock(fh, path):
    if fh is None:
        return
    try:
        fh.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001
        pass
    try:
        fh.close()
    except Exception:  # noqa: BLE001
        pass
    for q in (path, path + ".pid"):
        try:
            os.remove(q)
        except OSError:
            pass


def pid_alive(pid):
    """True if a process with this PID exists. Errs towards True: refusing to start is
    safer than running a second submitter."""
    try:
        import subprocess
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=10).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:  # noqa: BLE001
        return True


def parse_t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main():
    if not TEAM_KEY:
        sys.exit("TEAM_API_KEY is not set: put it in .env (see .env.example).")
    if ONLY is None and not ALLOW_LIVE:
        sys.exit("REFUSING to run live. bot.py needs ALLOW_LIVE=1 for real games.\n"
                 "  test : set ONLY_GAME=0 & set DRY_RUN=1 & python bot.py\n"
                 "  test : set ONLY_GAME=0 & python bot.py\n"
                 "  live : set ALLOW_LIVE=1 & python bot.py   (exactly one submitter per team)")
    if ONLY is not None and ONLY.strip() != "0" and not ALLOW_LIVE:
        sys.exit(f"REFUSING: ONLY_GAME={ONLY} is not the test game. Set ALLOW_LIVE=1 if you mean it.")

    lock = os.path.join(HERE, LOCK_NAME)
    handle = acquire_lock(lock)
    if handle is None:
        sys.exit(f"Another instance already owns submissions ({lock}).")
    # Never delete bot.lock. Instead decide by PID liveness: if the old bot is still
    # running we refuse; if its PID is gone the lock is stale and we leave it in place.
    old_lock = os.path.join(HERE, "bot.lock")
    if ALLOW_LIVE and os.path.exists(old_lock):
        try:
            old_pid = int(open(old_lock).read().strip())
        except (OSError, ValueError):
            old_pid = None
        if old_pid and pid_alive(old_pid):
            release_lock(handle, lock)
            sys.exit(f"bot.lock holds live PID {old_pid} -- old bot still running. Stop it first.")
        log(f"bot.lock is stale (PID {old_pid} gone); leaving the file untouched")

    done = set()
    for f in os.listdir(LOGS):
        m = re.match(r"game_(\d+)\.json$", f)
        if m:
            done.add(int(m.group(1)))
    log(f"start pid={os.getpid()} only={ONLY} dry={DRY} live={ALLOW_LIVE} known_logs={len(done)}")
    try:
        while True:
            wait = 10
            try:
                r = http("GET", f"{BASE}/api/games/list", tries=3, timeout=8)
                games = r.json() if r is not None else []
                now = dt.datetime.now(dt.timezone.utc)
                for g in games:
                    gid, st = int(g["id"]), parse_t(g["start_time"])
                    if ONLY is not None and gid != int(ONLY):
                        continue
                    if gid in done and ONLY is None:
                        continue
                    live = st - dt.timedelta(seconds=3) <= now <= st + dt.timedelta(seconds=WINDOW - 5)
                    if live or ONLY is not None:
                        log(f"game {gid} live (start {st.astimezone():%H:%M:%S})")
                        try:
                            play(gid, g["start_time"], st)
                        except Exception:  # noqa: BLE001 - never let one round kill the loop
                            log("play crashed:", traceback.format_exc().splitlines()[-1])
                        done.add(gid)
                        if ONLY is not None:
                            return
                nxt = [parse_t(g["start_time"]) for g in games if parse_t(g["start_time"]) > now]
                if nxt:
                    left = (min(nxt) - now).total_seconds()
                    log("next game in", int(left), "s")
                    wait = 1 if left < 15 else min(10, left - 10)
            except Exception as e:  # noqa: BLE001
                log("loop error:", repr(e)[:200])
                wait = 3
            time.sleep(max(wait, 1))
    finally:
        release_lock(handle, lock)


# ---------- self-test ----------
def selftest():
    global B_FLOOR
    saved_floor, B_FLOOR = B_FLOOR, 0.0
    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    a, b = decide(200, 160, 240, 0.95, 1.0, 1.0)
    chk(0 < a < 200, f"a={a} should sit below the median")
    chk(0 < b < 200, f"b={b} should sit below the median (1/3-quantile)")
    chk(a < b, f"a={a} should be below b={b} for a confident fair item")
    chk(decide(100, 80, 120, 0.5)[1] == 0.0, "b must be 0 when q < 2/3")
    # A_MULT / B_MULT scale the two outputs independently and exactly
    a1, b1 = decide(200, 160, 240, 0.95, 1.0, 1.0)
    a2, b2 = decide(200, 160, 240, 0.95, 1.0, 0.75)
    chk(abs(a2 - a1) < 0.02, f"B_MULT must not move the charge: {a1} -> {a2}")
    chk(abs(b2 - b1 * 0.75) < 0.02, f"B_MULT must scale b exactly: {b1} -> {b2}")
    a3, b3 = decide(200, 160, 240, 0.95, 0.8, 1.0)
    chk(abs(b3 - b1) < 0.02, f"A_MULT must not move the limit: {b1} -> {b3}")
    chk(abs(a3 - a1 * 0.8) < 0.02, f"A_MULT must scale a exactly: {a1} -> {a3}")
    chk(decide(200, 160, 240, 0.95, 1.0, 0.75)[0] == decide(200, 160, 240, 0.95, 1.0, 1.0)[0],
        "shipping defaults: charge is the honest estimate")
    chk(decide(0, 0, 0, 0.9) == (0.0, 0.0), "median 0 -> (0,0)")
    chk(decide(None, None, None, 0.9) == (0.0, 0.0), "None median -> (0,0)")
    chk(decide(float("nan"), 1, 2, 0.9) == (0.0, 0.0), "NaN median -> (0,0)")
    chk(decide(float("inf"), 1, 2, 0.9) == (0.0, 0.0), "inf median -> (0,0)")
    chk(decide(-5, 1, 2, 0.9) == (0.0, 0.0), "negative median -> (0,0)")
    a2, b2 = decide(100, 95, 105, 0.99, 1.0, 1.0)           # degenerate range -> sigma floor
    chk(0 < a2 < 100 and 0 < b2 < 100, f"sigma floor case gave a={a2} b={b2}")
    wide = decide(100, 20, 500, 0.99, 1.0, 1.0)
    narrow = decide(100, 95, 105, 0.99, 1.0, 1.0)
    # b is the 1/3-quantile, so it falls monotonically as sigma widens.
    chk(wide[1] < narrow[1], "wider uncertainty must lower the acceptance limit")
    # a is NOT monotonic in sigma: argmax of a*(1-Phi(z)) sits at z* solving s(1-Phi(z))=phi(z),
    # which crosses zero near s=0.8. Below that we charge under the median, above it the fat
    # right tail makes charging OVER the median the risk-neutral optimum. Documented, not a bug.
    chk(narrow[0] < 100, f"tight range must charge below the median, got {narrow[0]}")
    chk(decide(100, 60, 165, 0.99, 1.0, 1.0)[0] < 100, "moderate range still charges below the median")
    if A_CAP_MULT >= 1e8:
        chk(wide[0] > 100, f"uncapped, a very wide range charges above the median, got {wide[0]}")
    else:
        chk(wide[0] <= 100 * A_CAP_MULT + 0.01, f"A_CAP_MULT must bind, got {wide[0]}")
    for args in [(100, 200, 50, 0.9), (100, "x", None, "y"), (1e9, 1, 1e12, 1.5)]:
        aa, bb = decide(*args)
        chk(math.isfinite(aa) and aa >= 0 and math.isfinite(bb) and bb >= 0,
            f"decide{args} -> ({aa},{bb}) must be finite and >= 0")

    inv0 = ("ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Leak detection next to the washbasin\n"
            "2 hrs\n2 Replacement of split copper pipe\n1 pcs\n3 Re-plastering approx 0.2 m2\n"
            "0.2 m2\n4 Annual service of the gas boiler\n1 pcs\nINVOICE\nPage 1 / 1\n")
    idx, det = guess_indices(inv0)
    chk(idx == [1, 2, 3, 4], f"guess_indices -> {idx}, expected [1,2,3,4] ({det})")
    idx1, _ = guess_indices("POS. DESCRIPTION\n1 Laptop damage payment\n1 pcs\nINVOICE\n")
    chk(idx1 == [1], f"single-line invoice -> {idx1}")
    chk(guess_indices("")[0] == [], "empty invoice -> []")

    subs, rep = validate([{"index": 1, "median": 100, "p10": 80, "p90": 120, "p_valid": 0.9}], [1, 2, 3])
    chk(len(subs) == 3, f"validate must cover every expected index, got {len(subs)}")
    chk(rep["filled"] == [2, 3], f"missing lines must be filled, got {rep['filled']}")
    chk(subs[1]["charge_price"] == PROVISIONAL_A and subs[1]["acceptance_limit"] == PROVISIONAL_B,
        "filled lines must carry the provisional charge and limit")
    subs, rep = validate([{"index": 1, "median": 100, "p10": 150, "p90": 50, "p_valid": 0.9}], [1])
    chk(rep["clamped"] == [1], "inverted p10/p90 must be clamped, not dropped")
    chk(subs[0]["charge_price"] > 0, "clamped item must still price")
    subs, rep = validate([{"index": 1, "median": 100, "p10": 80, "p90": 120},
                          {"index": 1, "median": 999, "p10": 1, "p90": 2}], [1])
    chk(len(subs) == 1 and any("duplicate" in d for d in rep["dropped"]), "duplicate index dropped")
    subs, rep = validate("not a list", [1, 2])
    chk(len(subs) == 2 and rep["filled"] == [1, 2], "garbage LLM output still covers every line")
    subs, rep = validate([{"index": 1, "median": float("nan"), "p10": 1, "p90": 2}], [1])
    chk(subs[0]["charge_price"] == PROVISIONAL_A, "NaN median falls back to the provisional row")
    for s in subs:
        chk(math.isfinite(s["charge_price"]) and s["charge_price"] >= 0, "charge finite >= 0")
        chk(math.isfinite(s["acceptance_limit"]) and s["acceptance_limit"] >= 0, "limit finite >= 0")

    # --- verify_ack(): the API acknowledgement is not taken on trust ---
    class _R:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            if isinstance(self._p, Exception):
                raise self._p
            return self._p

    body = [{"index": 1, "charge_price": 10.0, "acceptance_limit": 20.0},
            {"index": 2, "charge_price": 5.0, "acceptance_limit": 0.0}]
    good = [{"game_id": 3, "line_item_index": 1, "charge_price": 10.0, "acceptance_limit": 20.0},
            {"game_id": 3, "line_item_index": 2, "charge_price": 5.0, "acceptance_limit": 0.0}]
    chk(verify_ack(3, body, _R(good)) is None, "a matching acknowledgement must pass")
    chk(verify_ack(3, body, _R(good[:1])) is not None, "a dropped index must be caught")
    bent = [dict(good[0], charge_price=99.0), good[1]]
    chk(verify_ack(3, body, _R(bent)) is not None, "a changed amount must be caught")
    wrong = [dict(good[0], game_id=9), good[1]]
    chk(verify_ack(3, body, _R(wrong)) is not None, "a wrong game_id must be caught")
    chk(verify_ack(3, body, _R({"detail": "x"})) is not None, "a non-list body must be caught")
    chk(verify_ack(3, body, _R(ValueError("no json"))) is not None, "unparseable body must be caught")
    rounded = [dict(good[0], charge_price=10.005), good[1]]
    chk(verify_ack(3, body, _R(rounded)) is None, "cent-level rounding must not trip the check")

    # --- aggregate(): the ensemble belief ---
    def mk(i, med, lo, hi, q=0.9):
        return {"index": i, "median": med, "p10": lo, "p90": hi, "p_valid": q, "text": "x"}

    ag = aggregate([[mk(1, 100, 90, 110)], [mk(1, 100, 90, 110)], [mk(1, 100, 90, 110)]])
    chk(len(ag) == 1 and ag[0]["median"] == 100, f"unanimous centre wrong: {ag}")
    chk(ag[0]["sigma_between"] == 0.0, f"unanimity must give zero between-sigma: {ag[0]}")
    chk(ag[0]["n_votes"] == 3, "n_votes must count the voices")
    tight = ag[0]["p90"] / ag[0]["p10"]

    ag = aggregate([[mk(1, 50, 45, 55)], [mk(1, 100, 90, 110)], [mk(1, 200, 180, 220)]])
    chk(ag[0]["median"] == 100, f"centre must be the median vote, got {ag[0]['median']}")
    chk(ag[0]["sigma_between"] > 0.4, f"disagreement must raise sigma: {ag[0]['sigma_between']}")
    chk(ag[0]["p90"] / ag[0]["p10"] > tight * 2,
        "a split field must widen the band well beyond unanimity")

    # one voice says the line is worthless, two price it: a coverage split, not a zero price
    ag = aggregate([[mk(1, 0, 0, 0, 0.1)], [mk(1, 100, 90, 110)], [mk(1, 120, 110, 130)]])
    chk(ag[0]["median"] == 100, f"zero vote must not drag the centre to 0: {ag[0]['median']}")
    chk(ag[0]["sigma_between"] >= 0.35, "a coverage split must open the band")

    ag = aggregate([[mk(1, 0, 0, 0, 0.0)], [mk(1, 0, 0, 0, 0.0)], [mk(1, 0, 0, 0, 0.0)]])
    chk(ag[0]["median"] == 0.0 and ag[0]["p10"] == 0.0, "unanimous zero stays zero")
    chk(decide(ag[0]["median"], ag[0]["p10"], ag[0]["p90"], ag[0]["p_valid"]) == (0.0, 0.0),
        "unanimous zero must price (0,0)")

    # robustness: garbage, duplicates, partial coverage, a dead assessor
    ag = aggregate([[mk(1, 100, 90, 110), mk(1, 999, 1, 2)],
                    ["junk", {"no": "index"}, mk(2, 20, 10, 30)],
                    None, [], [mk(1, 110, 100, 120)]])
    by = {x["index"]: x for x in ag}
    chk(set(by) == {1, 2}, f"aggregate indices wrong: {sorted(by)}")
    chk(by[1]["n_votes"] == 2, f"duplicate index within one voice must count once: {by[1]}")
    chk(by[2]["n_votes"] == 1, "a line only one voice saw still survives")
    for x in ag:
        a_, b_ = decide(x["median"], x["p10"], x["p90"], x["p_valid"])
        chk(math.isfinite(a_) and a_ >= 0 and math.isfinite(b_) and b_ >= 0,
            f"aggregate -> decide must stay finite and >= 0, got ({a_},{b_})")
    chk(aggregate([]) == [], "no voices -> no items")
    chk(aggregate([[]]) == [], "empty voice -> no items")

    # wider ensemble disagreement must not raise the acceptance limit
    narrow = aggregate([[mk(1, 100, 95, 105)]] * 3)[0]
    wide = aggregate([[mk(1, 60, 55, 65)], [mk(1, 100, 95, 105)], [mk(1, 170, 160, 180)]])[0]
    chk(decide(wide["median"], wide["p10"], wide["p90"], 0.95)[1]
        < decide(narrow["median"], narrow["p10"], narrow["p90"], 0.95)[1],
        "disagreement must lower b, not raise it")

    pol = ("HEAD preamble text\n" + "\n".join(
        f"{i}.1 Section about {'water damage pipe leak' if i == 7 else 'unrelated matter'} "
        + "filler " * 60 for i in range(1, 30)))
    ex, meta = policy_excerpt(pol, "a pipe leak soaked the bathroom", "1 pipe repair", cap=3000)
    chk(len(ex) <= 3000, f"excerpt must respect the cap, got {len(ex)}")
    chk("water damage pipe leak" in ex, "excerpt must keep the relevant section")
    chk(meta["mode"] == "excerpt", f"excerpt mode was {meta['mode']}")
    short, meta2 = policy_excerpt("tiny policy", "d", "i", cap=3000)
    chk(short == "tiny policy" and meta2["mode"] == "full", "short policy passes through whole")

    # --- hedged LLM call ---
    _real_items = llm_items
    try:
        _calls = []

        def _fake(policy, desc, inv, timeout, memory=""):
            _calls.append(time.monotonic())
            if len(_calls) == 1 and desc == "slow":
                time.sleep(3.0)
                return [{"index": 1, "median": 1.0}], {"n": 1}
            if desc == "fail":
                raise RuntimeError("boom")
            return [{"index": 1, "median": 2.0}], {"n": len(_calls)}
        globals()["llm_items"] = _fake
        _b = llm_async("p", "slow", "i", 12.0, hedge_after=0.5)
        chk(_b.get("items") == [{"index": 1, "median": 2.0}] and _b.get("hedged") and _b.get("winner") == 1,
            f"slow first request: hedge must win: {_b}")
        _calls.clear()
        _b = llm_async("p", "fast", "i", 6.0, hedge_after=2.0)
        chk(_b.get("items") and not _b.get("hedged") and len(_calls) == 1, f"fast answer: no hedge: {_b}")
        _calls.clear()
        _b = llm_async("p", "fail", "i", 9.0, hedge_after=2.0)
        chk("error" in _b and len(_calls) == 2, f"failing model: one retry then error: {_b}, calls {len(_calls)}")
        _t = time.monotonic()
        _calls.clear()
        globals()["llm_items"] = lambda *a, **k: (time.sleep(30), None)[1]
        _b = llm_async("p", "hang", "i", 1.5, hedge_after=0.5)
        chk("error" in _b and time.monotonic() - _t < 2.5, f"hung model: deadline holds: {_b}")
    finally:
        globals()["llm_items"] = _real_items
    # --- recall ---
    try:
        import recall as _rc
        _inv = "ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Service technician hours 8 hrs\n2 Vehicle costs 1 pcs\nINVOICE\n"
        _ix = {"service technician hours|8.0|hrs": [[21, 622.0, 680.0], [70, 9999.0, None]]}
        _s = [{"index": 1, "charge_price": 300.0, "acceptance_limit": 300.0},
              {"index": 2, "charge_price": 50.0, "acceptance_limit": 0.0}]
        _n, _c = _rc.apply(_s, _inv, 62, _ix)
        chk(_n[0]["charge_price"] == round(0.9 * 622.0, 2) and _n[0]["acceptance_limit"] == 622.0,
            f"recall raises a to 0.9 LO and b to LO: {_n[0]}")
        chk(_n[1] == _s[1], "recall leaves a line without history alone")
        chk(len(_c) == 1, "one recall change")
        _hi = [{"index": 1, "charge_price": 900.0, "acceptance_limit": 900.0}]
        chk(_rc.apply(_hi, _inv, 62, _ix)[0] == _hi, "recall never lowers a value")
        chk(_rc.apply(_s, None, 62, _ix)[0] == _s, "recall on no invoice: unchanged")
        chk(_rc.apply(_s, _inv, 62, None)[0] == _s, "recall on a broken index: unchanged")
        chk(_rc.floor_for([[70, 9999.0, None]], 62) is None, "recall ignores later games")
        _inv3 = ("ITEMS\nPOS. DESCRIPTION AMOUNTUNIT TOTAL\n1 Room drying 1 pcs\n2 Vehicle costs 1 pcs\n"
                 "3 Connection hose 1 pcs\nINVOICE\n")
        _gk = {"62": ["room drying|1.0|pcs", "vehicle costs|1.0|pcs", "connection hose|1.0|pcs"]}
        _ix3 = {"room drying|1.0|pcs": [[62, 91.0, 117.0]]}
        _c3 = _rc.apply([{"index": 1, "charge_price": 591.0, "acceptance_limit": 750.0}], _inv3, 65, _ix3, _gk)[0][0]
        chk(_c3["charge_price"] == round(0.9 * 91, 2) and _c3["acceptance_limit"] == round(0.999 * 117, 2),
            f"recall ceiling in a repeated case: {_c3}")
        chk(callable(getattr(_rc, "load_full", None)), "recall.load_full exists")
    except ImportError as e:
        chk(False, f"recall.py not importable: {e!r}")
    # --- coverage judge merge ---
    ens = [{"index": 1, "median": 100.0, "p10": 80.0, "p90": 130.0, "p_valid": 0.95},
           {"index": 2, "median": 50.0, "p10": 40.0, "p90": 70.0, "p_valid": 0.95},
           {"index": 3, "median": 20.0, "p10": 10.0, "p90": 30.0, "p_valid": 0.1}]
    jd = [{"index": 1, "median": 999.0, "p10": 1.0, "p90": 2000.0, "p_valid": 0.9},
          {"index": 2, "median": 0.0, "p10": 0.0, "p90": 0.0, "p_valid": 0.05},
          {"index": 3, "median": 25.0, "p10": 15.0, "p90": 35.0, "p_valid": 0.95},
          {"index": 4, "median": "x", "p_valid": 0.9}]
    mg, ch = merge_coverage(ens, jd)
    by = {x["index"]: x for x in mg}
    if PRICE_BLEND:
        _w = min(max(PRICE_BLEND_W, 0.0), 1.0)
        chk(abs(by[1]["median"] - 100.0 ** (1 - _w) * 999.0 ** _w) < 0.05 and by[1]["p_valid"] == 0.9,
            f"covered: geometric-mean price and judge q, got {by[1]['median']}")
        chk(abs(by[1]["p10"] / by[1]["median"] - 0.8) < 0.01, "blend must shift the band, not reshape it")
    else:
        chk(by[1]["median"] == 100.0 and by[1]["p_valid"] == 0.9, "covered: keep ensemble price, judge q")
    chk(by[2]["p_valid"] == 0.05 and by[2]["median"] == 0.0, "judge exclusion must win")
    chk(by[3]["p_valid"] == 0.95 and by[3]["median"] > 0, "judge can re-cover an ensemble exclusion")
    chk(4 not in by, "a malformed judge line must be ignored")
    chk(ch == 2, f"two coverage calls changed, got {ch}")
    chk(merge_coverage(ens, None)[0] == ens, "no judge output: ensemble unchanged")
    chk(merge_coverage(ens, "garbage")[0] == ens, "garbage judge output: ensemble unchanged")
    # --- acceptance floor ---
    B_FLOOR = saved_floor
    fa, fb = decide(200, 160, 240, 0.95, 1.0, 1.0, 1.0)
    chk(abs(fb - 200) < 0.02, f"floor: confident line must accept up to the estimate, got b={fb}")
    _, nb = decide(200, 160, 240, 0.95, 1.0, 1.0, 0.0)
    chk(nb < 200, f"floor off: b must be the 1/3-quantile below the estimate, got {nb}")
    _, ub = decide(200, 160, 240, 0.75, 1.0, 1.0, 1.0)
    chk(ub < 200, f"floor must not apply below B_FLOOR_Q, got b={ub}")
    chk(decide(200, 160, 240, 0.5, 1.0, 1.0, 1.0)[1] == 0.0, "floor must never override q < 2/3")
    chk(decide(0, 0, 0, 0.95, 1.0, 1.0, 1.0) == (0.0, 0.0), "floor on a zero estimate stays zero")
    print(f"selftest: {len(fails)} failure(s)")
    for f in fails:
        print("  FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    if os.environ.get("SELFTEST") == "1":
        sys.exit(selftest())
    main()
