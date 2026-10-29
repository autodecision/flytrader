"""Read-only active-coin discovery and causal quote-activity checks."""
from collections import deque
import json
import math
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .market import valid_mint, WSOL

EXPLORE_URL = "https://pump.fun/explore"
COINS_API = "https://frontend-api-v3.pump.fun/coins"
NATIVE_SOL = "11111111111111111111111111111111"
MIN_AGE_SECONDS = 120
MAX_TRADE_AGE_SECONDS = 90
MIN_REAL_SOL = 2.0
QUIET_SECONDS = 90
WINDOW_SECONDS = 120
MIN_WINDOW_CHANGES = 3
PRICE_TOLERANCE = 1e-6


def price_changed(first, second):
    return not math.isclose(first, second, rel_tol=PRICE_TOLERANCE, abs_tol=0)


def activity_limits(interval_seconds):
    # Leave enough sampling opportunities when the user chooses a slower feed.
    return {"quiet_seconds": max(QUIET_SECONDS, 3 * interval_seconds),
            "window_seconds": max(WINDOW_SECONDS, (MIN_WINDOW_CHANGES + 1) * interval_seconds)}


def eligible_candidates(rows, now, exclude=(), *, min_cap=0, min_reserves=MIN_REAL_SOL, include_migrated=False):
    """Listing metadata is a shortlist only; the runner also verifies RPC prices."""
    candidates = []
    for row in rows:
        if not isinstance(row, dict) or not valid_mint(row.get("mint")) or row["mint"] in exclude:
            continue
        migrated = row.get("complete") is True
        if row.get("complete") not in (False, True) or (migrated and not include_migrated) or row.get("initialized") is not True or row.get("is_banned"):
            continue
        if row.get("program") != "pump" or row.get("quote_mint") not in (None, NATIVE_SOL, WSOL):
            continue
        if row.get("raydium_pool") or (row.get("pump_swap_pool") and not include_migrated):
            continue
        try:
            created = float(row["created_timestamp"]) / 1000
            traded = float(row["last_trade_timestamp"]) / 1000
            reserve = float(row["real_sol_reserves"]) / 1e9
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(v) for v in (created, traded, reserve)):
            continue
        try:
            cap = float(row.get("market_cap_quote", row.get("market_cap")) or 0)
        except (TypeError, ValueError, OverflowError):
            cap = 0
        if not math.isfinite(cap) or cap < min_cap:
            continue
        if now - created < MIN_AGE_SECONDS or not -5 <= now - traded <= MAX_TRADE_AGE_SECONDS or (not migrated and reserve < min_reserves):
            continue
        candidates.append({"mint": row["mint"], "listed_last_trade": traded,
                           "listed_real_sol": reserve, "listed_market_cap_sol": cap,
                           "name": str(row.get("name", ""))[:80], "symbol": str(row.get("symbol", ""))[:30],
                           "source": "Pump.fun Explore · last trade", "url": EXPLORE_URL})
    return sorted(candidates, key=lambda coin: (coin["listed_market_cap_sol"] if min_cap else coin["listed_last_trade"],
                                               coin["listed_real_sol"], coin["listed_last_trade"]), reverse=True)


def fetch_candidates(offset=0):
    url = COINS_API + "?" + urlencode({"offset": offset, "limit": 50, "sort": "last_trade_timestamp",
                                     "order": "DESC", "includeNsfw": "false"})
    request = Request(url, headers={"User-Agent": "Flytrader paper research/0.1", "Accept": "application/json"})
    with urlopen(request, timeout=12) as response:
        payload = response.read(4_000_001)
    if len(payload) > 4_000_000:
        raise ValueError("Active-coin response exceeded its size limit")
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise ValueError("Active-coin service returned an unsupported response")
    return rows


class ExploreDiscovery:
    def __init__(self, fetch=fetch_candidates, clock=time.time, *, min_cap=0, min_reserves=MIN_REAL_SOL, include_migrated=False):
        self.fetch, self.clock = fetch, clock
        self.filters = {"min_cap": min_cap, "min_reserves": min_reserves, "include_migrated": include_migrated}
        self.rows, self.fetched_at, self.last_candidate = [], 0, None

    def __call__(self, exclude=()):
        now = self.clock()
        if now - self.fetched_at < 30:
            candidates = eligible_candidates(self.rows, now, exclude, **self.filters)
            if candidates:
                self.last_candidate = candidates[0]
                return candidates[0]["mint"]
        self.rows = []
        for offset in (0, 50):
            self.rows.extend(self.fetch(offset))
            self.fetched_at = self.clock()
            candidates = eligible_candidates(self.rows, self.fetched_at, exclude, **self.filters)
            if candidates:
                self.last_candidate = candidates[0]
                return candidates[0]["mint"]
        raise RuntimeError("No recently traded native-SOL curve passed the activity filters; waiting for another candidate")


class QuoteActivity:
    """Quiet describes observed quotes, not proof that every on-chain trade stopped."""
    def __init__(self, interval_seconds=10):
        self.first = self.last_move = self.last_time = self.reference = None
        self.changes = deque()
        self.limits = activity_limits(interval_seconds)

    def observe(self, timestamp, price):
        if not math.isfinite(timestamp) or not math.isfinite(price) or price <= 0:
            raise ValueError("Activity checks require a finite time and positive price")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Activity checks require advancing observation timestamps")
        if self.first is None:
            self.first = self.last_move = timestamp
            self.reference = price
        elif price_changed(self.reference, price):
            self.last_move = timestamp
            self.reference = price
            self.changes.append(timestamp)
        self.last_time = timestamp
        window = self.limits["window_seconds"]
        quiet = self.limits["quiet_seconds"]
        while self.changes and self.changes[0] <= timestamp - window:
            self.changes.popleft()
        observed = timestamp - self.first
        idle = timestamp - self.last_move
        reason = None
        if idle >= quiet:
            reason = f"No meaningful price movement for {int(idle)} seconds"
        elif observed >= window and len(self.changes) < MIN_WINDOW_CHANGES:
            reason = f"Only {len(self.changes)} price changes in the last {window:g} seconds"
        return {"status": "quiet" if reason else "watching" if observed < window else "active",
                "changes_in_window": len(self.changes), "window_seconds": min(observed, window),
                "seconds_since_change": idle, "reason": reason,
                "quiet_after_seconds": quiet, "min_changes_per_window": MIN_WINDOW_CHANGES}
