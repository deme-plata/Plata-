import os
import time
import random
import requests
import threading
import aiohttp
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from bip_utils import Bip39SeedGenerator, Bip39MnemonicValidator, Bip44, Bip44Coins, Bip44Changes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature, decode_dss_signature
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64
import hashlib
from typing import List

# Define the number of transactions to generate
NUM_TRANSACTIONS = 10

# Define the number of threads to use for parallel processing
NUM_THREADS = 100

# Define the URLs for both servers
SERVER_ONE = "http://161.35.219.10:50503"
SERVER_TWO = "http://159.89.106.101:50503"
TRANSACTION_URLS = [f"{SERVER_ONE}/send_transaction", f"{SERVER_TWO}/send_transaction"]
BATCH_TRANSACTION_URLS = [f"{SERVER_ONE}/send_batch_transactions", f"{SERVER_TWO}/send_batch_transactions"]
TOKEN_URLS = [f"{SERVER_ONE}/token", f"{SERVER_TWO}/token"]
REGISTER_URLS = [f"{SERVER_ONE}/register", f"{SERVER_TWO}/register"]
MINE_BLOCK_URLS = [f"{SERVER_ONE}/mine_block", f"{SERVER_TWO}/mine_block"]
BALANCE_URLS = [f"{SERVER_ONE}/get_balance", f"{SERVER_TWO}/get_balance"]

# Example mnemonic and pincode (replace these with actual values)
MNEMONIC = "leaf virus apology link math image suggest until blast mom usual glance"
USER_PINCODE = "1234"

# Function to derive a private key from the mnemonic
def derive_private_key(mnemonic):
    Bip39MnemonicValidator().Validate(mnemonic)
    seed = Bip39SeedGenerator(mnemonic).Generate()
    bip44_mst = Bip44.FromSeed(seed, Bip44Coins.BITCOIN)
    bip44_acc = bip44_mst.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    private_key_bytes = bip44_acc.PrivateKey().Raw().ToBytes()
    private_key = ec.derive_private_key(int.from_bytes(private_key_bytes, byteorder="big"), ec.SECP256R1(), default_backend())
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')
    return private_key_pem

# Function to derive a wallet address from the mnemonic
def derive_wallet_address(mnemonic):
    Bip39MnemonicValidator().Validate(mnemonic)
    seed = Bip39SeedGenerator(mnemonic).Generate()
    bip44_mst = Bip44.FromSeed(seed, Bip44Coins.BITCOIN)
    bip44_acc = bip44_mst.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    private_key_bytes = bip44_acc.PrivateKey().Raw().ToBytes()
    private_key = ec.derive_private_key(int.from_bytes(private_key_bytes, byteorder="big"), ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    address = hashlib.sha256(public_key_bytes).hexdigest()
    return address

# Function to register a user on both servers
def register_user(pincode):
    for register_url in REGISTER_URLS:
        for _ in range(5):
            try:
                response = requests.post(register_url, json={"pincode": pincode})
                if response.status_code == 200:
                    print("User registered successfully")
                    break
                elif response.status_code == 400 and "already registered" in response.text.lower():
                    print("User already registered")
                    break
                else:
                    print(f"Failed to register user on {register_url}: {response.text}")
            except Exception as e:
                print(f"Exception occurred while registering user on {register_url}: {e}")
            time.sleep(2)
        else:
            raise Exception(f"Failed to register user on {register_url} after multiple attempts")

# Function to get an access token from both servers
def get_access_tokens(pincode):
    tokens = []
    for token_url in TOKEN_URLS:
        for _ in range(5):
            try:
                response = requests.post(token_url, json={"pincode": pincode})
                if response.status_code == 200:
                    tokens.append(response.json()["access_token"])
                    break
                else:
                    print(f"Failed to get access token from {token_url}: {response.text}")
            except Exception as e:
                print(f"Exception occurred while getting access token from {token_url}: {e}")
            time.sleep(2)
        else:
            raise Exception(f"Failed to get access token from {token_url} after multiple attempts")
    return tokens

# Function to mine a block on both servers
def mine_blocks(node_id, tokens):
    headers = [{"Authorization": f"Bearer {token}"} for token in tokens]
    for i, mine_block_url in enumerate(MINE_BLOCK_URLS):
        for _ in range(5):
            try:
                response = requests.post(mine_block_url, json={"node_id": node_id}, headers=headers[i])
                if response.status_code == 200:
                    print(f"Block mined successfully on {mine_block_url}")
                    return response.json()
                else:
                    print(f"Failed to mine block on {mine_block_url}: {response.text}")
            except Exception as e:
                print(f"Exception occurred while mining block on {mine_block_url}: {e}")
            time.sleep(2)
        else:
            raise Exception(f"Failed to mine block on {mine_block_url} after multiple attempts")

# Function to get the balance of a wallet from both servers
def get_balances(address, tokens):
    headers = [{"Authorization": f"Bearer {token}"} for token in tokens]
    balances = []
    for i, balance_url in enumerate(BALANCE_URLS):
        for _ in range(5):
            try:
                response = requests.post(balance_url, json={"address": address}, headers=headers[i])
                if response.status_code == 200:
                    balances.append(response.json()["balance"])
                    break
                else:
                    print(f"Failed to get balance from {balance_url}: {response.text}")
            except Exception as e:
                print(f"Exception occurred while getting balance from {balance_url}: {e}")
            time.sleep(2)
        else:
            raise Exception(f"Failed to get balance from {balance_url} after multiple attempts")
    return balances

# Function to generate a random transaction
def generate_transaction(sender):
    receiver = f"receiver_{random.randint(1, 10000)}"
    amount = 0.0000000000000001  # Minimal amount for transaction
    private_key = derive_private_key(MNEMONIC)
    return {
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "private_key": private_key
    }

# Function to sign a transaction
def sign_transaction(private_key_pem, message):
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())
    signature = private_key.sign(
        message.encode(),
        ec.ECDSA(hashes.SHA256())
    )
    r, s = decode_dss_signature(signature)
    signature_bytes = encode_dss_signature(r, s)
    return base64.b64encode(signature_bytes).decode('utf-8')

# Function to send a transaction to both servers
async def send_transaction(session, transaction, tokens):
    transaction['signature'] = sign_transaction(transaction['private_key'], f"{transaction['sender']}{transaction['receiver']}{transaction['amount']}")
    results = []
    for transaction_url, token in zip(TRANSACTION_URLS, tokens):
        headers = {"Authorization": f"Bearer {token}"}
        async with session.post(transaction_url, json=transaction, headers=headers) as response:
            result = await response.json()
            results.append(result)
    return results

# Function to send batch transactions to both servers
async def send_batch_transactions(session, transactions, tokens):
    results = []
    for batch_transaction_url, token in zip(BATCH_TRANSACTION_URLS, tokens):
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with session.post(batch_transaction_url, json=transactions, headers=headers) as response:
            result = await response.json()
            results.append(result)
    return results

# Function to generate and send transactions in parallel to both servers
async def generate_and_send_transactions(num_transactions, sender):
    start_time = time.time()
    successful_transactions = 0
    failed_transactions = 0
    tokens = get_access_tokens(USER_PINCODE)  # Get the access tokens

    conn = aiohttp.TCPConnector(limit=NUM_THREADS, keepalive_timeout=300)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        batch_size = 100  # Number of transactions per batch
        transactions = [generate_transaction(sender) for _ in range(num_transactions)]
        for i in range(0, num_transactions, batch_size):
            batch = transactions[i:i + batch_size]
            try:
                results = await send_batch_transactions(session, batch, tokens)
                for result_batch in results:
                    for result in result_batch:
                        if result.get("success"):
                            successful_transactions += 1
                        else:
                            failed_transactions += 1
            except Exception as e:
                print(f"Error sending batch: {e}")
                failed_transactions += len(batch)

    end_time = time.time()
    duration = end_time - start_time
    tps = successful_transactions / duration
    print(f"Total transactions: {num_transactions}")
    print(f"Successful transactions: {successful_transactions}")
    print(f"Failed transactions: {failed_transactions}")
    print(f"Transactions per second: {tps:.2f}")
    print(f"Total time: {duration:.2f} seconds")

    # Get and print the final balance from both servers
    try:
        balances = get_balances(sender, tokens)
        for i, balance in enumerate(balances):
            print(f"Final wallet balance from server {i + 1}: {balance}")
    except Exception as e:
        print(f"Error getting final balances: {e}")

# Ensure the user is registered before attempting to get an access token
register_user(USER_PINCODE)

# Derive wallet address from the mnemonic
wallet_address = derive_wallet_address(MNEMONIC)
print(f"Derived wallet address: {wallet_address}")

# Get the current balance from both servers
try:
    tokens = get_access_tokens(USER_PINCODE)
    balances = get_balances(wallet_address, tokens)
    for i, balance in enumerate(balances):
        print(f"Current wallet balance from server {i + 1}: {balance}")
except Exception as e:
    print(f"Error checking balances: {e}")

# Mine a block on both servers and receive the reward if balance is insufficient
if any(balance < NUM_TRANSACTIONS * 0.0000000000000001 for balance in balances):
    try:
        mine_blocks(wallet_address, tokens)
    except Exception as e:
        print(f"Error mining block: {e}")

# Generate and send transactions to both servers
try:
    asyncio.run(generate_and_send_transactions(NUM_TRANSACTIONS, wallet_address))
except Exception as e:
    print(f"Error generating and sending transactions: {e}")

# Function to calculate blocks per second on both servers
def calculate_bps():
    start_time = time.time()
    block_count = 0
    tokens = get_access_tokens(USER_PINCODE)  # Get the access tokens
    node_id = wallet_address

    while time.time() - start_time < 60:  # Calculate BPS over 1 minute
        try:
            results = mine_blocks(node_id, tokens)
            if all(result.get("success") for result in results):
                block_count += 1
        except Exception as e:
            print(f"Error mining block: {e}")

    end_time = time.time()
    duration = end_time - start_time
    bps = block_count / duration
    print(f"Total blocks mined: {block_count}")
    print(f"Blocks per second: {bps:.2f}")

# Calculate blocks per second on both servers
calculate_bps()
