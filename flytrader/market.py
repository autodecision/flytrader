"""Read-only public RPC observations. No trading or paid subscriptions."""
import asyncio
import base64
import csv
import json
import math
from pathlib import Path
import struct
import time
from urllib.parse import urlencode, urlparse, unquote
from urllib.request import Request, urlopen

PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
DISCRIMINATOR = bytes([23, 183, 248, 55, 96, 216, 172, 96])
SWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
SWAP_DISCRIMINATOR = bytes([241, 154, 109, 4, 17, 177, 109, 188])
WSOL = "So11111111111111111111111111111111111111112"
TOKEN_PROGRAMS = ("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA", "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
RAYDIUM_API = "https://api-v3.raydium.io/pools/info/list-v2"
RAYDIUM_CLMM_PROGRAM = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
RAYDIUM_POOL_DISCRIMINATOR = bytes.fromhex("f7ede3f5d7c3de46")
MAX_RAYDIUM_RESPONSE = 1_000_000


class UnsupportedCurve(ValueError):
    """A permanent stop: never substitute a price for an unsupported curve."""


def valid_mint(mint):
    return isinstance(mint, str) and 32 <= len(mint) <= 44 and all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in mint)


def parse_coin(value):
    """Accept a mint or a Pump.fun coin page, never an arbitrary fetch URL."""
    value = str(value or "").strip()
    if not value or valid_mint(value):
        return value
    if value.startswith("0x") and len(value) == 42:
        raise ValueError("This coin uses an EVM address (0x…). Flytrader only supports Solana Pump.fun coins.")
    url = urlparse("https://" + value if value.startswith(("pump.fun/", "www.pump.fun/")) else value)
    if url.scheme != "https" or url.hostname not in ("pump.fun", "www.pump.fun") or url.username or url.password or url.port:
        raise ValueError("Paste a Solana mint address or an https://pump.fun/coin/… link")
    parts = unquote(url.path).strip("/").split("/")
    mint = parts[1] if len(parts) == 2 and parts[0] == "coin" else ""
    if mint.startswith("0x") and len(mint) == 42:
        raise ValueError("This coin uses an EVM address (0x…). Flytrader only supports Solana Pump.fun coins.")
    if not valid_mint(mint):
        raise ValueError("This Pump.fun link does not contain a valid Solana coin address")
    return mint


def pump_creation(event):
    # PumpPortal's creation stream also carries other launch platforms.
    return isinstance(event, dict) and event.get("pool") == "pump" and event.get("txType") == "create" and valid_mint(event.get("mint"))


def rpc(endpoint, method, params):
    request = Request(endpoint, data=json.dumps({"jsonrpc": "2.0", "id": 1,
                      "method": method, "params": params}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        result = json.load(response)
    if "error" in result:
        raise ValueError(f"RPC rejected {method}: code {result['error'].get('code')}")
    return result["result"]


def raydium_sol_pool(mint):
    """Find a SOL-paired Raydium CLMM, then let RPC validate every account."""
    query = urlencode({"size": 100, "mint1": mint, "sortField": "liquidity", "sortType": "desc"})
    request = Request(f"{RAYDIUM_API}?{query}", headers={"Accept": "application/json", "User-Agent": "Flytrader/1"})
    with urlopen(request, timeout=15) as response:
        if int(response.headers.get("Content-Length") or 0) > MAX_RAYDIUM_RESPONSE:
            raise ValueError("Raydium pool response was unexpectedly large")
        raw = response.read(MAX_RAYDIUM_RESPONSE + 1)
    if len(raw) > MAX_RAYDIUM_RESPONSE:
        raise ValueError("Raydium pool response was unexpectedly large")
    payload = json.loads(raw)
    rows = payload.get("data", {}).get("data", []) if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else []
    if not isinstance(payload, dict) or payload.get("success") is not True or not isinstance(rows, list) or len(rows) > 100:
        raise ValueError("Raydium pool discovery returned an invalid response")
    candidates = []
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "Concentrated" or row.get("programId") != RAYDIUM_CLMM_PROGRAM:
            continue
        pair = [row.get("mintA"), row.get("mintB")]
        addresses = [item.get("address") for item in pair if isinstance(item, dict)]
        if len(addresses) != 2 or set(addresses) != {mint, WSOL} or not valid_mint(row.get("id")):
            continue
        try:
            liquidity = float(row.get("tvl", 0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(liquidity) and liquidity > 0:
            candidates.append((liquidity, row["id"]))
    if not candidates:
        raise UnsupportedCurve("No verified native-SOL PumpSwap or Raydium pool is available for this token")
    return max(candidates)[1]


def decode_raydium_clmm(account, mint, mint_decimals):
    """Decode the small, security-critical subset of Raydium's PoolState."""
    from solders.pubkey import Pubkey
    if not account or account["owner"] != RAYDIUM_CLMM_PROGRAM:
        raise UnsupportedCurve("Raydium pool owner did not match the official CLMM program")
    raw = base64.b64decode(account["data"][0], validate=True)
    if len(raw) < 273 or raw[:8] != RAYDIUM_POOL_DISCRIMINATOR:
        raise UnsupportedCurve("Unsupported Raydium pool layout")
    mint_a, mint_b = str(Pubkey.from_bytes(raw[73:105])), str(Pubkey.from_bytes(raw[105:137]))
    if {mint_a, mint_b} != {mint, WSOL}:
        raise UnsupportedCurve("Raydium pool does not match the selected token/SOL pair")
    decimals_a, decimals_b = raw[233], raw[234]
    expected = (9, mint_decimals) if mint_a == WSOL else (mint_decimals, 9)
    if (decimals_a, decimals_b) != expected:
        raise UnsupportedCurve("Raydium pool mint decimals do not match the verified mints")
    liquidity = int.from_bytes(raw[237:253], "little")
    sqrt_price = int.from_bytes(raw[253:269], "little")
    if not sqrt_price:
        raise UnsupportedCurve("Raydium pool has no on-chain price")
    q64 = 1 << 64
    price_b_per_a = (sqrt_price / q64) ** 2 * 10 ** (decimals_a - decimals_b)
    if not math.isfinite(price_b_per_a) or price_b_per_a <= 0:
        raise UnsupportedCurve("Raydium pool price is invalid")
    vault_a, vault_b = str(Pubkey.from_bytes(raw[137:169])), str(Pubkey.from_bytes(raw[169:201]))
    virtual_a = liquidity * q64 / sqrt_price / 10 ** decimals_a
    virtual_b = liquidity * sqrt_price / q64 / 10 ** decimals_b
    if mint_a == WSOL:
        price_sol, quote_vault, token_vault = 1 / price_b_per_a, vault_a, vault_b
        quote_reserves, token_reserves = virtual_a, virtual_b
    else:
        price_sol, quote_vault, token_vault = price_b_per_a, vault_b, vault_a
        quote_reserves, token_reserves = virtual_b, virtual_a
    return {"price_sol": price_sol, "quote_vault": quote_vault, "token_vault": token_vault,
            "quote_reserves_sol": quote_reserves, "token_reserves": token_reserves}


def transfer_fee_bps(parsed):
    for extension in parsed.get("extensions", []):
        if extension.get("extension") == "transferFeeConfig":
            fee = extension.get("state", {}).get("newerTransferFee", {}).get("transferFeeBasisPoints")
            if isinstance(fee, int) and 0 <= fee < 10_000:
                return fee
    return 0


def transfer_fee_authority(parsed):
    for extension in parsed.get("extensions", []):
        if extension.get("extension") == "transferFeeConfig":
            authority = extension.get("state", {}).get("transferFeeConfigAuthority")
            return authority if valid_mint(authority) else None
    return None


def decode_curve(account, decimals):
    if not account or account["owner"] != PROGRAM:
        raise UnsupportedCurve("Missing account or unexpected bonding-curve owner")
    data = base64.b64decode(account["data"][0], validate=True)
    if len(data) < 115 or data[:8] != DISCRIMINATOR:
        raise UnsupportedCurve("Unsupported bonding-curve layout")
    if any(data[83:115]):
        raise UnsupportedCurve("Only native SOL quote curves are supported")
    if data[48] != 0:
        raise UnsupportedCurve("Curve has completed or migrated; no supported quote is available")
    virtual_tokens, virtual_sol = struct.unpack_from("<QQ", data, 8)
    if not virtual_tokens or not virtual_sol:
        raise UnsupportedCurve("Empty virtual reserves")
    return (virtual_sol / 1e9) / (virtual_tokens / 10 ** decimals)


def curve_snapshot(account, decimals, supply):
    price = decode_curve(account, decimals)
    raw = base64.b64decode(account["data"][0], validate=True)
    real_sol = struct.unpack_from("<Q", raw, 32)[0] / 1e9
    return {"market_cap_sol": price * supply, "real_sol": real_sol,
            "regime": "bonding_curve", "quote_reserves_sol": struct.unpack_from("<Q", raw, 16)[0] / 1e9,
            "token_reserves": struct.unpack_from("<Q", raw, 8)[0] / 10 ** decimals,
            "real_tokens": struct.unpack_from("<Q", raw, 24)[0] / 10 ** decimals,
            "price_sol": price, "source": "Confirmed Solana bonding-curve reserves",
            "cap_kind": "Price × total minted supply; not available liquidity"}


def decode_pool(account, mint, creator):
    from solders.pubkey import Pubkey
    if not account or account["owner"] != SWAP_PROGRAM:
        raise UnsupportedCurve("No canonical native-SOL PumpSwap pool is available")
    data = base64.b64decode(account["data"][0], validate=True)
    if (len(data) < 243 or data[:8] != SWAP_DISCRIMINATOR or data[9:11] != bytes(2) or
            data[11:43] != bytes(Pubkey.from_string(creator)) or data[43:75] != bytes(Pubkey.from_string(mint)) or
            data[75:107] != bytes(Pubkey.from_string(WSOL))):
        raise UnsupportedCurve("PumpSwap pool does not match the canonical token/SOL pair")
    return {"base_vault": str(Pubkey.from_bytes(data[139:171])), "quote_vault": str(Pubkey.from_bytes(data[171:203])),
            "virtual_quote_raw": int.from_bytes(data[245:261], "little", signed=True) if len(data) >= 261 else 0}


def vault_amount(account, mint, pool):
    from solders.pubkey import Pubkey
    if not account or account["owner"] not in TOKEN_PROGRAMS:
        raise UnsupportedCurve("Unsupported pool token vault owner")
    data = base64.b64decode(account["data"][0], validate=True)
    if len(data) < 165 or data[:32] != bytes(Pubkey.from_string(mint)) or data[32:64] != bytes(Pubkey.from_string(pool)) or data[108] != 1:
        raise UnsupportedCurve("Pool vault mint, authority, or initialized state did not match")
    return struct.unpack_from("<Q", data, 64)[0]


async def discover(exclude=()):
    import websockets
    async with websockets.connect("wss://pumpportal.fun/api/data", open_timeout=15) as ws:
        # Only the documented free token-creation stream.
        await ws.send(json.dumps({"method": "subscribeNewToken"}))
        async with asyncio.timeout(60):
            async for message in ws:
                event = json.loads(message)
                if pump_creation(event) and event["mint"] not in exclude:
                    return event["mint"]
    raise TimeoutError("The token creation stream ended before a new coin arrived")


class CurveFeed:
    """One read-only native-SOL curve, accepting only advancing confirmed slots."""

    def __init__(self, mint, endpoint):
        from solders.pubkey import Pubkey
        if not valid_mint(mint):
            raise UnsupportedCurve("Invalid Solana mint address")
        self.mint, self.endpoint = mint, endpoint
        curve, _ = Pubkey.find_program_address([b"bonding-curve", bytes(Pubkey.from_string(mint))],
                                              Pubkey.from_string(PROGRAM))
        self.curve = str(curve)
        info = rpc(endpoint, "getAccountInfo", [mint, {"encoding": "jsonParsed", "commitment": "confirmed"}])["value"]
        if not info or info["owner"] not in TOKEN_PROGRAMS or info["data"].get("parsed", {}).get("type") != "mint":
            raise UnsupportedCurve("Token mint could not be verified")
        parsed = info["data"]["parsed"]["info"]
        self.decimals = int(parsed["decimals"])
        self.supply = int(parsed["supply"]) / 10 ** self.decimals
        if self.supply <= 0:
            raise UnsupportedCurve("Token supply is empty")
        self.security = {"mint_authority": parsed.get("mintAuthority"), "freeze_authority": parsed.get("freezeAuthority"),
                         "authorities_verified": "mintAuthority" in parsed and "freezeAuthority" in parsed,
                         "largest_account_pct": None, "holder_check": "Not yet checked",
                         "extensions": [e.get("extension") for e in parsed.get("extensions", [])],
                         "transfer_fee_bps": transfer_fee_bps(parsed),
                         "transfer_fee_authority": transfer_fee_authority(parsed)}
        self.token_program = info["owner"]
        authority, _ = Pubkey.find_program_address([b"pool-authority", bytes(Pubkey.from_string(mint))], Pubkey.from_string(PROGRAM))
        pool, _ = Pubkey.find_program_address([b"pool", bytes(2), bytes(authority), bytes(Pubkey.from_string(mint)),
                                              bytes(Pubkey.from_string(WSOL))], Pubkey.from_string(SWAP_PROGRAM))
        self.pool, self.pool_creator = str(pool), str(authority)
        self.raydium_pool = None
        self.pool_vaults = None
        self.mode = "curve"
        self.snapshot = {}
        self.holder_attempt_at = 0
        self.slot = -1

    def check_holders(self):
        """Largest token accounts are not necessarily independent owners."""
        from solders.pubkey import Pubkey
        self.holder_attempt_at = time.time()
        associated_program = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
        excluded = set()
        for owner in (self.curve, self.pool):
            vault, _ = Pubkey.find_program_address([bytes(Pubkey.from_string(owner)),
                       bytes(Pubkey.from_string(self.token_program)), bytes(Pubkey.from_string(self.mint))], associated_program)
            excluded.add(str(vault))
        if self.pool_vaults and self.pool_vaults.get("token_vault"):
            excluded.add(self.pool_vaults["token_vault"])
        try:
            result = rpc(self.endpoint, "getTokenLargestAccounts", [self.mint, {"commitment": "confirmed"}])
            amounts = [int(row["amount"]) / 10 ** self.decimals for row in result["value"] if row["address"] not in excluded]
            self.security.update(largest_account_pct=100 * max(amounts, default=0) / self.supply,
                                 holder_check="Top 20 token accounts; recognized market vaults excluded; owners may control multiple accounts",
                                 holder_checked_at=time.time())
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.security.update(largest_account_pct=None, holder_check=f"Unavailable ({type(exc).__name__}); ownership concentration is unknown")
        return dict(self.security)

    def read(self):
        if time.time() - self.holder_attempt_at >= 300:
            self.check_holders()
        if self.mode == "pool":
            return self.read_pool()
        if self.mode == "raydium_clmm":
            return self.read_raydium_clmm()
        result = rpc(self.endpoint, "getAccountInfo", [self.curve, {"encoding": "base64", "commitment": "confirmed"}])
        slot = result["context"]["slot"]
        if slot <= self.slot:
            raise ValueError("RPC has not advanced; waiting for a fresh confirmed slot")
        if result["value"] is None:
            self.raydium_pool = raydium_sol_pool(self.mint)
            self.pool, self.pool_vaults, self.mode = self.raydium_pool, None, "raydium_clmm"
            return self.read_raydium_clmm()
        try:
            details = curve_snapshot(result["value"], self.decimals, self.supply)
        except UnsupportedCurve:
            account = result["value"]
            raw = base64.b64decode(account["data"][0], validate=True)
            if account["owner"] != PROGRAM or len(raw) < 115 or raw[:8] != DISCRIMINATOR or raw[48] != 1 or any(raw[83:115]):
                raise
            self.mode = "pool"
            return self.read_pool()
        self.snapshot = {**details,
                         **self.security, "slot": slot, "observed_at": time.time()}
        price = self.snapshot["price_sol"]
        self.slot = slot
        return time.time(), price, 0.0

    def read_raydium_clmm(self):
        if self.pool_vaults is None:
            account = rpc(self.endpoint, "getAccountInfo", [self.pool, {"encoding": "base64", "commitment": "confirmed"}])["value"]
            self.pool_vaults = decode_raydium_clmm(account, self.mint, self.decimals)
            self.holder_attempt_at = 0
            self.check_holders()
        addresses = [self.pool, self.pool_vaults["token_vault"], self.pool_vaults["quote_vault"]]
        result = rpc(self.endpoint, "getMultipleAccounts", [addresses, {"encoding": "base64", "commitment": "confirmed"}])
        slot = result["context"]["slot"]
        if slot <= self.slot:
            raise ValueError("RPC has not advanced; waiting for a fresh confirmed slot")
        pool = decode_raydium_clmm(result["value"][0], self.mint, self.decimals)
        if [pool["token_vault"], pool["quote_vault"]] != addresses[1:]:
            raise UnsupportedCurve("Raydium pool vault addresses changed")
        real_tokens = vault_amount(result["value"][1], self.mint, self.pool) / 10 ** self.decimals
        real_sol = vault_amount(result["value"][2], WSOL, self.pool) / 1e9
        if min(real_tokens, real_sol) <= 0:
            raise UnsupportedCurve("Raydium pool vaults do not provide a supported price")
        price = pool["price_sol"]
        active = pool["quote_reserves_sol"] > 0 and pool["token_reserves"] > 0
        self.slot = slot
        self.snapshot = {"price_sol": price, "market_cap_sol": price * self.supply, "real_sol": real_sol,
                         "regime": "raydium_clmm", "quote_reserves_sol": pool["quote_reserves_sol"],
                         "token_reserves": pool["token_reserves"], "real_tokens": real_tokens,
                         "source": ("Confirmed Raydium CLMM price and active-liquidity virtual reserves" if active else
                                    "Confirmed Raydium CLMM price; no active liquidity, so paper fills are blocked"),
                         "liquidity_status": "active" if active else "inactive",
                         "pool": self.pool, "cap_kind": "Price × total minted supply; not available liquidity",
                         **self.security, "slot": slot, "observed_at": time.time()}
        return time.time(), price, 0.0

    def read_pool(self):
        if self.pool_vaults is None:
            account = rpc(self.endpoint, "getAccountInfo", [self.pool, {"encoding": "base64", "commitment": "confirmed"}])["value"]
            try:
                self.pool_vaults = decode_pool(account, self.mint, self.pool_creator)
            except UnsupportedCurve:
                self.raydium_pool = raydium_sol_pool(self.mint)
                self.pool, self.pool_vaults, self.mode = self.raydium_pool, None, "raydium_clmm"
                return self.read_raydium_clmm()
        addresses = [self.pool, self.pool_vaults["base_vault"], self.pool_vaults["quote_vault"]]
        result = rpc(self.endpoint, "getMultipleAccounts", [addresses, {"encoding": "base64", "commitment": "confirmed"}])
        slot = result["context"]["slot"]
        if slot <= self.slot:
            raise ValueError("RPC has not advanced; waiting for a fresh confirmed slot")
        pool = decode_pool(result["value"][0], self.mint, self.pool_creator)
        if [pool["base_vault"], pool["quote_vault"]] != addresses[1:]:
            raise UnsupportedCurve("PumpSwap vault addresses changed")
        base = vault_amount(result["value"][1], self.mint, self.pool) / 10 ** self.decimals
        quote_raw = vault_amount(result["value"][2], WSOL, self.pool)
        effective = (quote_raw + pool["virtual_quote_raw"]) / 1e9
        if base <= 0 or effective <= 0:
            raise UnsupportedCurve("PumpSwap reserves do not provide a supported price")
        price = effective / base
        self.slot = slot
        self.snapshot = {"price_sol": price, "market_cap_sol": price * self.supply, "real_sol": quote_raw / 1e9,
                         "regime": "pumpswap", "quote_reserves_sol": effective, "token_reserves": base, "real_tokens": base,
                         "source": "Confirmed canonical PumpSwap vaults; effective quote reserves", "pool": self.pool,
                         "cap_kind": "Price × total minted supply; not available liquidity", **self.security,
                         "slot": slot, "observed_at": time.time()}
        return time.time(), price, 0.0


def capture(args):
    from solders.pubkey import Pubkey
    if args.count < 1 or args.interval < 10:
        raise ValueError("Use count >= 1 and interval >= 10 seconds for public RPC")
    mint = args.mint or asyncio.run(discover())
    curve, _ = Pubkey.find_program_address([b"bonding-curve", bytes(Pubkey.from_string(mint))],
                                          Pubkey.from_string(PROGRAM))
    decimals = rpc(args.rpc, "getTokenSupply", [mint, {"commitment": "confirmed"}])["value"]["decimals"]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"mint": mint, "bonding_curve": str(curve), "kind": "reserve-price-snapshots",
                "volume": "Unavailable; zero placeholder disables volume input",
                "selection": "User supplied mint" if args.mint else "First observed creation; not a recommendation",
                "interval_seconds": args.interval, "observations": 0}
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "mint", "close", "volume_sol"])
        previous_slot = -1
        try:
            for i in range(args.count):
                result = rpc(args.rpc, "getAccountInfo", [str(curve), {"encoding": "base64", "commitment": "confirmed"}])
                slot = result["context"]["slot"]
                if slot <= previous_slot:
                    raise ValueError("RPC slot is stale; capture stopped")
                price = decode_curve(result["value"], decimals)
                writer.writerow([time.time(), mint, price, 0])
                stream.flush()
                previous_slot = slot
                metadata["observations"] += 1
                print(f"{i + 1}/{args.count}: {mint} {price:.12g} SOL/token", flush=True)
                if i + 1 < args.count:
                    time.sleep(args.interval)
        finally:
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
