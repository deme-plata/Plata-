from blockcypher import generate_new_address
from eth_account import Account
from web3 import Web3
import time 
from user_management import fake_users_db
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
import os
from SecureHybridZKStark import SecureHybridZKStark  # Ensure you import the SecureHybridZKStark class
import random
# Custom Exception for Rate Limit Errors
class RateLimitError(Exception):
    pass

class WalletRegistration:
    def __init__(self, blockcypher_api_key, alchemy_key):
        self.blockcypher_api_key = blockcypher_api_key
        self.alchemy_key = alchemy_key
        self.w3 = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{self.alchemy_key}"))
        self.chain_ids = {
            'ethereum': 1,
            'polygon': 137
        }
        self.requests_per_hour = 100
        self.requests_made = 0
        self.next_request_time = time.time()
        self.solana_client = AsyncClient("https://api.mainnet-beta.solana.com")
        self.zk_system = SecureHybridZKStark(security_level=20)

    def _check_rate_limit(self):
        """ Check if the rate limit is reached and calculate the next available request time. """
        current_time = time.time()
        if self.requests_made >= self.requests_per_hour:
            wait_time = max(self.next_request_time - current_time, 0)
            self._show_error_popup(wait_time)
            time.sleep(wait_time)  # Wait for the next available request time
            self.requests_made = 0  # Reset the counter after the wait
            self.next_request_time = time.time() + 3600  # Reset the request window
    def register_solana_wallet(self):
        # Generate a new Solana keypair using the correct method
        keypair = Keypair()  # This should generate a new keypair

        solana_wallet = {
            'address': str(keypair.pubkey()),
            'private_key': keypair.secret().hex(),
        }
        return solana_wallet










    def _show_error_popup(self, wait_time):
        """ Show an error message with a countdown timer. """
        print(f"Rate limit reached. Please wait {int(wait_time)} seconds before trying again.")
        # Additional logic can be added here to show a UI popup with the countdown timer

    def register_eth_wallet(self):
        eth_account = Account.create()
        eth_wallet = {
            'address': eth_account.address,
            'private_key': eth_account.key.hex(),
            'public_key': eth_account.key.hex(),
        }
        return eth_wallet

    def register_btc_wallet(self):
        self._check_rate_limit()  # Check and handle rate limit before making a request
        try:
            btc_wallet = generate_new_address(coin_symbol='btc', api_key=self.blockcypher_api_key)
            self.requests_made += 1
            return btc_wallet
        except Exception as e:
            # If the error message indicates a rate limit issue, raise the custom RateLimitError
            if "rate limit" in str(e).lower():
                raise RateLimitError("Rate limit reached for BlockCypher API.")
            else:
                raise e

    def register_ltc_wallet(self):
        self._check_rate_limit()  # Check and handle rate limit before making a request
        try:
            ltc_wallet = generate_new_address(coin_symbol='ltc', api_key=self.blockcypher_api_key)
            self.requests_made += 1
            return ltc_wallet
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise RateLimitError("Rate limit reached for BlockCypher API.")
            else:
                raise e

    def register_doge_wallet(self):
        self._check_rate_limit()  # Check and handle rate limit before making a request
        try:
            doge_wallet = generate_new_address(coin_symbol='doge', api_key=self.blockcypher_api_key)
            self.requests_made += 1
            return doge_wallet
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise RateLimitError("Rate limit reached for BlockCypher API.")
            else:
                raise e

    def register_all_wallets(self, user_id):
        eth_wallet = self.register_eth_wallet()
        btc_wallet = None
        ltc_wallet = None
        doge_wallet = None
        solana_wallet = self.register_solana_wallet()
        zk_wallet = self.register_zk_wallet()  # Add this line

        try:
            btc_wallet = self.register_btc_wallet()
        except RateLimitError as e:
            print(f"BTC wallet registration failed: {e}")

        try:
            ltc_wallet = self.register_ltc_wallet()
        except RateLimitError as e:
            print(f"LTC wallet registration failed: {e}")

        try:
            doge_wallet = self.register_doge_wallet()
        except RateLimitError as e:
            print(f"DOGE wallet registration failed: {e}")

        user_wallets = {
            'ethereum': eth_wallet,
            'bitcoin': btc_wallet if btc_wallet else "BTC wallet registration failed due to rate limits.",
            'litecoin': ltc_wallet if ltc_wallet else "LTC wallet registration failed due to rate limits.",
            'dogecoin': doge_wallet if doge_wallet else "DOGE wallet registration failed due to rate limits.",
            'solana': solana_wallet,
            'zk_wallet': zk_wallet,  # Add this line


        }

        # Create user entry if it doesn't exist
        if user_id not in fake_users_db:
            fake_users_db[user_id] = {'wallets': {}}

        # Update user's wallets
        fake_users_db[user_id]['wallets'] = user_wallets

        return user_wallets
        
    def register_zk_wallet(self):
        zk_keypair = self.zk_system.stark.field.element(random.randint(1, self.zk_system.stark.field.modulus - 1))
        zk_wallet = {
            'private_key': zk_keypair.value,
            'public_key': self.zk_system.stark.hash(zk_keypair.value).value,
        }
        return zk_wallet
