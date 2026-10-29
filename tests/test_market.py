import base64
import json
import struct
import unittest
from unittest.mock import patch

from flytrader.market import (decode_curve, curve_snapshot, decode_pool, decode_raydium_clmm,
                              transfer_fee_bps, transfer_fee_authority, vault_amount, CurveFeed, PROGRAM, DISCRIMINATOR,
                              SWAP_PROGRAM, SWAP_DISCRIMINATOR, WSOL, TOKEN_PROGRAMS,
                              RAYDIUM_CLMM_PROGRAM, RAYDIUM_POOL_DISCRIMINATOR, raydium_sol_pool)


class CurveTests(unittest.TestCase):
    def account(self, complete=False, quote=bytes(32)):
        data = DISCRIMINATOR + struct.pack("<QQQQQ", 10**12, 30 * 10**9, 0, 0, 0)
        data += bytes([complete]) + bytes(32) + bytes(2) + quote
        return {"owner": PROGRAM, "data": [base64.b64encode(data).decode(), "base64"]}

    def test_decimal_conversion(self):
        self.assertAlmostEqual(decode_curve(self.account(), 6), 0.00003)

    def test_completed_curve_rejected(self):
        with self.assertRaises(ValueError):
            decode_curve(self.account(complete=True), 6)

    def test_non_sol_curve_rejected(self):
        with self.assertRaises(ValueError):
            decode_curve(self.account(quote=bytes([1]) * 32), 6)

    def test_foreign_owner_rejected(self):
        account = self.account()
        account["owner"] = "11111111111111111111111111111111"
        with self.assertRaises(ValueError):
            decode_curve(account, 6)

    def test_curve_market_cap_and_real_reserves_are_distinct(self):
        account = self.account()
        raw = bytearray(base64.b64decode(account['data'][0]))
        struct.pack_into('<Q', raw, 32, 5 * 10**9)
        account['data'][0] = base64.b64encode(raw).decode()
        result = curve_snapshot(account, 6, 1_000_000_000)
        self.assertAlmostEqual(result['market_cap_sol'], 30000)
        self.assertEqual(result['real_sol'], 5)


class PumpSwapTests(unittest.TestCase):
    def setUp(self):
        from solders.pubkey import Pubkey
        self.key = lambda n: str(Pubkey.from_bytes(bytes([n]) * 32))
        self.mint, self.creator, self.pool = self.key(7), self.key(8), self.key(9)
        self.base_vault, self.quote_vault = self.key(10), self.key(11)

    def pool_account(self, virtual=0):
        from solders.pubkey import Pubkey
        raw = (SWAP_DISCRIMINATOR + bytes(3) + bytes(Pubkey.from_string(self.creator)) +
               bytes(Pubkey.from_string(self.mint)) + bytes(Pubkey.from_string(WSOL)) + bytes(32) +
               bytes(Pubkey.from_string(self.base_vault)) + bytes(Pubkey.from_string(self.quote_vault)) +
               bytes(8 + 32 + 2) + int(virtual).to_bytes(16, 'little', signed=True))
        return {'owner': SWAP_PROGRAM, 'data': [base64.b64encode(raw).decode(), 'base64']}

    def vault(self, mint, amount):
        from solders.pubkey import Pubkey
        raw = bytearray(165)
        raw[:32] = bytes(Pubkey.from_string(mint)); raw[32:64] = bytes(Pubkey.from_string(self.pool))
        struct.pack_into('<Q', raw, 64, amount); raw[108] = 1
        return {'owner': TOKEN_PROGRAMS[0], 'data': [base64.b64encode(raw).decode(), 'base64']}

    def test_pool_requires_the_canonical_creator_mint_and_sol_quote(self):
        pool = self.pool_account(7)
        self.assertEqual(decode_pool(pool, self.mint, self.creator)['virtual_quote_raw'], 7)
        for mint, creator in [(self.key(12), self.creator), (self.mint, self.key(12))]:
            with self.assertRaises(ValueError): decode_pool(pool, mint, creator)
        pool['owner'] = PROGRAM
        with self.assertRaises(ValueError): decode_pool(pool, self.mint, self.creator)

    def test_pool_vaults_require_matching_mint_and_authority(self):
        vault = self.vault(self.mint, 123)
        self.assertEqual(vault_amount(vault, self.mint, self.pool), 123)
        with self.assertRaises(ValueError): vault_amount(vault, WSOL, self.pool)
        with self.assertRaises(ValueError): vault_amount(vault, self.mint, self.key(12))

    def test_pool_quotes_use_effective_reserves_but_report_spendable_reserves_separately(self):
        feed = object.__new__(CurveFeed)
        feed.pool, feed.pool_creator, feed.mint = self.pool, self.creator, self.mint
        feed.endpoint, feed.mode, feed.slot = 'unused', 'pool', 5
        feed.holder_attempt_at = float('inf')
        feed.decimals, feed.supply, feed.security, feed.pool_vaults = 6, 1000, {}, None
        pool = self.pool_account(2 * 10**9)
        result = {'context': {'slot': 6}, 'value': [pool, self.vault(self.mint, 10 * 10**6), self.vault(WSOL, 3 * 10**9)]}
        def rpc(_, method, params):
            return {'value': pool} if method == 'getAccountInfo' else result
        with patch('flytrader.market.rpc', rpc):
            bar = feed.read()
            self.assertAlmostEqual(bar[1], .5)
            self.assertEqual(feed.snapshot['real_sol'], 3)
            self.assertEqual(feed.snapshot['market_cap_sol'], 500)
            with self.assertRaisesRegex(ValueError, 'not advanced'): feed.read()


class RaydiumClmmTests(unittest.TestCase):
    def setUp(self):
        from solders.pubkey import Pubkey
        self.key = lambda n: str(Pubkey.from_bytes(bytes([n]) * 32))
        self.mint, self.pool = self.key(7), self.key(9)
        self.token_vault, self.quote_vault = self.key(10), self.key(11)

    def pool_account(self, *, mint_a=WSOL, mint_b=None, owner=RAYDIUM_CLMM_PROGRAM):
        from solders.pubkey import Pubkey
        raw = bytearray(1544)
        raw[:8] = RAYDIUM_POOL_DISCRIMINATOR
        raw[73:105] = bytes(Pubkey.from_string(mint_a))
        raw[105:137] = bytes(Pubkey.from_string(mint_b or self.mint))
        raw[137:169] = bytes(Pubkey.from_string(self.quote_vault))
        raw[169:201] = bytes(Pubkey.from_string(self.token_vault))
        raw[233:235] = bytes((9, 6))
        raw[237:253] = (100 * 10**9).to_bytes(16, 'little')
        raw[253:269] = (1 << 64).to_bytes(16, 'little')
        return {'owner': owner, 'data': [base64.b64encode(raw).decode(), 'base64']}

    def vault(self, mint, amount):
        from solders.pubkey import Pubkey
        raw = bytearray(165)
        raw[:32] = bytes(Pubkey.from_string(mint)); raw[32:64] = bytes(Pubkey.from_string(self.pool))
        struct.pack_into('<Q', raw, 64, amount); raw[108] = 1
        return {'owner': TOKEN_PROGRAMS[0], 'data': [base64.b64encode(raw).decode(), 'base64']}

    def test_pool_requires_official_program_selected_mint_and_sol(self):
        pool = decode_raydium_clmm(self.pool_account(), self.mint, 6)
        self.assertAlmostEqual(pool['price_sol'], .001)
        self.assertEqual(pool['quote_vault'], self.quote_vault)
        self.assertEqual(pool['token_vault'], self.token_vault)
        self.assertAlmostEqual(pool['quote_reserves_sol'], 100)
        self.assertAlmostEqual(pool['token_reserves'], 100_000)
        with self.assertRaisesRegex(ValueError, 'owner'):
            decode_raydium_clmm(self.pool_account(owner=PROGRAM), self.mint, 6)
        with self.assertRaisesRegex(ValueError, 'selected token'):
            decode_raydium_clmm(self.pool_account(mint_b=self.key(12)), self.mint, 6)
        with self.assertRaisesRegex(ValueError, 'decimals'):
            decode_raydium_clmm(self.pool_account(), self.mint, 8)

    def test_discovery_selects_only_official_sol_paired_clmm(self):
        valid_pool = self.key(14)
        payload = {'success': True, 'data': {'data': [
            {'type': 'Standard', 'programId': RAYDIUM_CLMM_PROGRAM, 'id': self.key(15),
             'mintA': {'address': WSOL}, 'mintB': {'address': self.mint}, 'tvl': 999},
            {'type': 'Concentrated', 'programId': PROGRAM, 'id': self.key(16),
             'mintA': {'address': WSOL}, 'mintB': {'address': self.mint}, 'tvl': 999},
            {'type': 'Concentrated', 'programId': RAYDIUM_CLMM_PROGRAM, 'id': valid_pool,
             'mintA': {'address': WSOL}, 'mintB': {'address': self.mint}, 'tvl': 10},
        ]}}
        class Response:
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _): return json.dumps(payload).encode()
        with patch('flytrader.market.urlopen', return_value=Response()) as fetch:
            self.assertEqual(raydium_sol_pool(self.mint), valid_pool)
        self.assertTrue(fetch.call_args.args[0].full_url.startswith('https://api-v3.raydium.io/'))

    def test_feed_uses_confirmed_pool_price_and_separate_real_vaults(self):
        feed = object.__new__(CurveFeed)
        feed.pool, feed.raydium_pool, feed.mint = self.pool, self.pool, self.mint
        feed.endpoint, feed.mode, feed.slot = 'unused', 'raydium_clmm', 5
        feed.holder_attempt_at = float('inf')
        feed.decimals, feed.supply = 6, 1_000_000
        feed.security = {'transfer_fee_bps': 300}
        feed.pool_vaults = decode_raydium_clmm(self.pool_account(), self.mint, 6)
        result = {'context': {'slot': 6}, 'value': [self.pool_account(),
                  self.vault(self.mint, 75_000 * 10**6), self.vault(WSOL, 40 * 10**9)]}
        with patch('flytrader.market.rpc', return_value=result):
            bar = feed.read()
        self.assertAlmostEqual(bar[1], .001)
        self.assertEqual(feed.snapshot['regime'], 'raydium_clmm')
        self.assertEqual(feed.snapshot['real_sol'], 40)
        self.assertEqual(feed.snapshot['real_tokens'], 75_000)
        self.assertEqual(feed.snapshot['transfer_fee_bps'], 300)

    def test_inactive_pool_keeps_confirmed_price_but_exposes_no_fill_reserves(self):
        account = self.pool_account()
        raw = bytearray(base64.b64decode(account['data'][0]))
        raw[237:253] = bytes(16)
        account['data'][0] = base64.b64encode(raw).decode()
        pool = decode_raydium_clmm(account, self.mint, 6)
        self.assertAlmostEqual(pool['price_sol'], .001)
        self.assertEqual(pool['quote_reserves_sol'], 0)
        self.assertEqual(pool['token_reserves'], 0)

    def test_transfer_fee_is_read_from_verified_mint_extensions(self):
        authority = self.key(13)
        parsed = {'extensions': [{'extension': 'transferFeeConfig', 'state': {
            'newerTransferFee': {'transferFeeBasisPoints': 300}, 'transferFeeConfigAuthority': authority}}]}
        self.assertEqual(transfer_fee_bps(parsed), 300)
        self.assertEqual(transfer_fee_authority(parsed), authority)
        self.assertEqual(transfer_fee_bps({'extensions': []}), 0)
