# helius_integration.py

import requests
import json
from typing import List, Dict, Any

class HeliusAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.helius.xyz/v0"

    def _make_request(self, endpoint: str, method: str = "GET", params: Dict[str, Any] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}?api-key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        if method == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()
        return response.json()

    def get_balance(self, address: str) -> Dict[str, Any]:
        return self._make_request(f"addresses/{address}/balances")

    def get_nfts(self, address: str) -> List[Dict[str, Any]]:
        return self._make_request(f"addresses/{address}/nfts")

    def get_transactions(self, address: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self._make_request(f"addresses/{address}/transactions", params={"limit": limit})

    def get_token_metadata(self, mint_addresses: List[str]) -> List[Dict[str, Any]]:
        return self._make_request("token-metadata", method="POST", data={"mintAccounts": mint_addresses})

    def get_transaction_details(self, signature: str) -> Dict[str, Any]:
        return self._make_request(f"transactions/{signature}")

    def parse_transaction(self, transaction: str) -> Dict[str, Any]:
        return self._make_request("parse-transaction", method="POST", data={"transaction": transaction})

    def get_token_holders(self, mint_address: str) -> List[Dict[str, Any]]:
        return self._make_request(f"tokens/{mint_address}/holders")

    def name_service_lookup(self, name: str) -> Dict[str, Any]:
        return self._make_request("name-service/lookup", params={"name": name})

    def name_service_reverse_lookup(self, address: str) -> Dict[str, Any]:
        return self._make_request("name-service/reverse-lookup", params={"address": address})