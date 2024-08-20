import requests
import json
from decimal import Decimal

class ZeroXSwapAPI:
    def __init__(self, api_key, chain_id=1):
        self.base_url = "https://api.0x.org/swap/v1"
        self.api_key = api_key
        self.chain_id = chain_id
        self.headers = {'0x-api-key': self.api_key}

    def get_price(self, sell_token, buy_token, sell_amount, taker_address):
        params = {
            'sellToken': sell_token,
            'buyToken': buy_token,
            'sellAmount': sell_amount,
            'taker': taker_address,
            'chainId': self.chain_id
        }
        response = requests.get(f"{self.base_url}/price", params=params, headers=self.headers)
        return response.json()

    def get_quote(self, sell_token, buy_token, sell_amount, taker_address):
        params = {
            'sellToken': sell_token,
            'buyToken': buy_token,
            'sellAmount': sell_amount,
            'taker': taker_address,
            'chainId': self.chain_id
        }
        response = requests.get(f"{self.base_url}/quote", params=params, headers=self.headers)
        return response.json()

    def set_token_allowance(self, token_address, spender_address, amount, wallet_client):
        try:
            # Use web3 or similar library to set token allowance on the blockchain
            contract = wallet_client.eth.contract(address=token_address, abi=ERC20_ABI)
            tx = contract.functions.approve(spender_address, amount).buildTransaction({
                'from': wallet_client.address,
                'gas': 200000,
                'gasPrice': wallet_client.eth.gas_price,
                'nonce': wallet_client.eth.getTransactionCount(wallet_client.address)
            })
            signed_tx = wallet_client.eth.account.signTransaction(tx)
            tx_hash = wallet_client.eth.sendRawTransaction(signed_tx.rawTransaction)
            return tx_hash
        except Exception as e:
            print(f"Error setting token allowance: {e}")
            return None

    def sign_permit2_message(self, permit2_data, wallet_client):
        try:
            signature = wallet_client.eth.account.signTypedData(permit2_data)
            return signature
        except Exception as e:
            print(f"Error signing Permit2 message: {e}")
            return None

    def submit_transaction(self, quote_data, signature, wallet_client):
        try:
            tx_data = quote_data['transactions'][0]['data'] + signature
            tx = {
                'from': wallet_client.address,
                'to': quote_data['transactions'][0]['to'],
                'data': tx_data,
                'gas': int(quote_data['transactions'][0]['gas']),
                'gasPrice': int(quote_data['transactions'][0]['gasPrice']),
                'nonce': wallet_client.eth.getTransactionCount(wallet_client.address),
                'chainId': self.chain_id
            }
            signed_tx = wallet_client.eth.account.signTransaction(tx)
            tx_hash = wallet_client.eth.sendRawTransaction(signed_tx.rawTransaction)
            return tx_hash
        except Exception as e:
            print(f"Error submitting transaction: {e}")
            return None
