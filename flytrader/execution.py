"""Small-order constant-product paper fills against confirmed reserve snapshots.

Shadow reserve offsets carry our hypothetical swaps across subsequent snapshots.
External order flow is assumed unchanged by our presence. This is an approximation,
not an executable quote; configured fees also approximate contract/network costs.
"""
import math


class FillUnavailable(ValueError):
    pass


class ReserveExecution:
    def __init__(self, cost, fixed_fee_sol=.000005, max_impact=.02):
        self.cost, self.fixed_fee, self.max_impact = cost, fixed_fee_sol, max_impact
        self.offsets = {}

    def reserves(self, mint, market):
        if market.get("regime") not in ("bonding_curve", "pumpswap", "raydium_clmm"):
            raise FillUnavailable("Verified execution reserves are unavailable")
        key = (mint, market["regime"])
        quote, tokens = market.get("quote_reserves_sol"), market.get("token_reserves")
        real_sol, real_tokens = market.get("real_sol"), market.get("real_tokens")
        if any(v is None or not math.isfinite(v) or v < 0 for v in (quote, tokens, real_sol, real_tokens)):
            raise FillUnavailable("Fresh real and pricing reserves are required for paper fills")
        dq, dt = self.offsets.get(key, (0., 0.))
        quote, tokens, real_sol, real_tokens = quote + dq, tokens + dt, real_sol + dq, real_tokens + dt
        if min(quote, tokens, real_sol) <= 0 or real_tokens < 0:
            raise FillUnavailable("Insufficient reserves for a paper fill")
        transfer_fee_bps = market.get("transfer_fee_bps", 0)
        if not isinstance(transfer_fee_bps, (int, float)) or not math.isfinite(transfer_fee_bps) or not 0 <= transfer_fee_bps < 10_000:
            raise FillUnavailable("Token transfer fee is invalid")
        return key, quote, tokens, real_sol, real_tokens, transfer_fee_bps / 10_000

    def buy(self, mint, market, budget, *, commit=True):
        key, quote, tokens, _, real_tokens, transfer_fee = self.reserves(mint, market)
        available = budget - self.fixed_fee
        if available <= 0:
            raise FillUnavailable("Allocation is too small to cover assumed transaction cost")
        net = available / (1 + self.cost)
        gross_units = tokens * net / (quote + net)
        units = gross_units * (1 - transfer_fee)
        impact = net / quote
        if gross_units > real_tokens or impact > self.max_impact or units <= 0:
            raise FillUnavailable("Paper buy exceeds available tokens or the 2% price-impact limit")
        if commit:
            dq, dt = self.offsets.get(key, (0., 0.))
            self.offsets[key] = (dq + net, dt - gross_units)
        return {"units": units, "amount_sol": budget, "fill_price": budget / units,
                "cost_sol": budget - net * (1 - transfer_fee), "impact_pct": impact * 100, "regime": key[1],
                "execution_model": "active-liquidity shadow reserves" if key[1] == "raydium_clmm" else "constant-product shadow reserves",
                "fixed_fee_sol": self.fixed_fee, "transfer_fee_bps": transfer_fee * 10_000}

    def sell(self, mint, market, units, *, commit=True):
        key, quote, tokens, real_sol, _, transfer_fee = self.reserves(mint, market)
        net_units = units * (1 - transfer_fee)
        gross = quote * net_units / (tokens + net_units)
        gross_without_transfer = quote * units / (tokens + units)
        proceeds = gross * (1 - self.cost) - self.fixed_fee
        if gross > real_sol or proceeds <= 0:
            raise FillUnavailable("Fresh reserves cannot fund a positive net paper exit")
        if commit:
            dq, dt = self.offsets.get(key, (0., 0.))
            self.offsets[key] = (dq - gross, dt + net_units)
        return {"units": units, "amount_sol": proceeds, "fill_price": proceeds / units,
                "cost_sol": gross_without_transfer - proceeds,
                "impact_pct": (1 - tokens / (tokens + net_units)) * 100,
                "regime": key[1],
                "execution_model": "active-liquidity shadow reserves" if key[1] == "raydium_clmm" else "constant-product shadow reserves",
                "fixed_fee_sol": self.fixed_fee, "transfer_fee_bps": transfer_fee * 10_000}
