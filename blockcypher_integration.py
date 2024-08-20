import requests
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import os

class BlockCypherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.blockcypher.com/v1/btc/main"

    def get_blockchain_info(self):
        url = f"{self.base_url}"
        response = requests.get(url)
        return response.json()

    def get_block(self, block_hash_or_height):
        url = f"{self.base_url}/blocks/{block_hash_or_height}"
        response = requests.get(url)
        return response.json()

    def get_address_info(self, address):
        url = f"{self.base_url}/addrs/{address}"
        response = requests.get(url)
        return response.json()

    def generate_address(self):
        url = f"{self.base_url}/addrs"
        response = requests.post(url)
        return response.json()

    def generate_multisig_address(self, pubkeys, script_type="multisig-2-of-3"):
        url = f"{self.base_url}/addrs"
        data = {
            "pubkeys": pubkeys,
            "script_type": script_type
        }
        response = requests.post(url, json=data)
        return response.json()

    def create_transaction(self, inputs, outputs):
        url = f"{self.base_url}/txs/new"
        data = {
            "inputs": inputs,
            "outputs": outputs
        }
        response = requests.post(url, json=data)
        return response.json()

    def send_transaction(self, tx_hex):
        url = f"{self.base_url}/txs/send"
        data = {"tx": tx_hex}
        response = requests.post(url, json=data)
        return response.json()

    def get_transaction(self, tx_hash):
        url = f"{self.base_url}/txs/{tx_hash}"
        response = requests.get(url)
        return response.json()

    def setup_webhook(self, event_type, url, address=None, confirmations=6):
        webhook_url = f"{self.base_url}/hooks"
        data = {
            "event": event_type,
            "url": url,
            "address": address,
            "confirmations": confirmations
        }
        response = requests.post(webhook_url, json=data, params={"token": self.api_key})
        return response.json()

    def get_confidence_factor(self, tx_hash):
        url = f"{self.base_url}/txs/{tx_hash}/confidence"
        response = requests.get(url)
        return response.json()

    def get_wallet_info(self, wallet_name):
        url = f"{self.base_url}/wallets/{wallet_name}"
        response = requests.get(url, params={"token": self.api_key})
        return response.json()

    def create_hd_wallet(self, wallet_name):
        url = f"{self.base_url}/wallets/hd"
        data = {"name": wallet_name}
        response = requests.post(url, json=data, params={"token": self.api_key})
        return response.json()

    def create_forwarding_address(self, destination_address, callback_url=None):
        url = f"{self.base_url}/payments"
        data = {
            "destination": destination_address,
            "callback_url": callback_url,
            "token": self.api_key
        }
        response = requests.post(url, json=data)
        return response.json()

    def delete_webhook(self, webhook_id):
        url = f"{self.base_url}/hooks/{webhook_id}"
        response = requests.delete(url, params={"token": self.api_key})
        return response.json()
