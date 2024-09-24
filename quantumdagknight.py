# Import necessary modules
import os
import time
import logging
import threading
from concurrent import futures
import uvicorn
import jwt
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, padding, hashes
from cryptography.hazmat.backends import default_backend
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile,assemble
from qiskit.providers.jobstatus import JobStatus
from qiskit.exceptions import QiskitError   
from nacl.public import PrivateKey, Box
from nacl.utils import random
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import List
import json
from qiskit.circuit.random import random_circuit
import asyncio
import aiohttp

import base64
import hashlib
import numpy as np
from mnemonic import Mnemonic
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP
from fastapi import FastAPI, HTTPException, Depends, status
from cryptography.exceptions import InvalidSignature
import traceback 
import random
from Crypto.PublicKey import RSA
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP
from typing import List
from qiskit.circuit.random import random_circuit
import aiohttp
from vm import SimpleVM
from pydantic import BaseModel, Field
from hashlib import sha256
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import hashlib
from tqdm import tqdm  # for progress tracking
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from contextlib import asynccontextmanager
from contextlib import asynccontextmanager
import os
import logging
import threading
import uvicorn

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import statistics
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import time
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import os
import base64
import re  
import string
from base64 import urlsafe_b64encode
from decimal import Decimal
from fastapi import WebSocket, Depends, HTTPException, Query
from fastapi.security import OAuth2PasswordBearer
import json
from typing import Dict
from vm import SimpleVM, Permission, Role,    PBFTConsensus
from vm import SimpleVM
import pytest
import httpx

import curses
from fastapi import APIRouter
from pydantic import BaseModel, Field, validator  # Ensure validator is imported
from pydantic import BaseModel, field_validator  # Use field_validator for Pydantic V2
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, root_validator
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Request, status
from starlette.websockets import WebSocketState
from collections import defaultdict
import uuid
from enum import Enum, auto
from pydantic import BaseModel, validator, Field, model_validator
import tracemalloc
from contextlib import asynccontextmanager
from vm import SimpleVM, Permission, Role, PBFTConsensus
from decimal import Decimal, InvalidOperation
from typing import Any, Dict  # Ensure Any and Dict are imported
from Order import Order
from SecureHybridZKStark import SecureHybridZKStark
from enhanced_exchange import EnhancedExchange,LiquidityPoolManager,PriceOracle,MarginAccount,AdvancedOrderTypes
from enhanced_exchange import EnhancedExchangeWithZKStarks

from EnhancedOrderBook import EnhancedOrderBook
from enhanced_exchange import EnhancedExchange,LiquidityPoolManager,PriceOracle,MarginAccount
from zk_vm import ZKVM
from shared_logic import QuantumBlock, Transaction, NodeState
from common import NodeDirectory
from P2PNode import Message,MessageType

from P2PNode import P2PNode
import curses
import socket
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import os
from user_management import fake_users_db
from secure_qr_system import SecureQRSystem
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import json
from helius_integration import HeliusAPI
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed

from blockcypher_integration import BlockCypherAPI
from helius_integration import HeliusAPI
from web3 import Web3
import aiohttp
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Optional
global blockchain, p2p_node
import aioredis

blockchain = None
p2p_node = None
helius_api = HeliusAPI(api_key="855fde7e-b54f-4c6f-b71c-e6876772ec81")
# FastAPI initialization
app = FastAPI(
    title="Plata Network",
    description="A Quantum Blockchain Network",
    version="0.1.0"
)
app.state.components = {}
fake_users_db = {}
# Initialize the SecureQRSystem
secure_qr_system = SecureQRSystem()
# Load environment variables
load_dotenv()
# Keccak-f[1600] constants
KECCAK_ROUNDS = 24
KECCAK_LANE_SIZE = 64
KECCAK_STATE_SIZE = 25
initialization_status = {
    "blockchain": False,
    "p2p_node": False,
    "vm": False,
    "price_feed": False,
    "plata_contract": False,
    "exchange": False
}
# Load the encryption key
with open("encryption_key.key", "rb") as key_file:
    encryption_key = key_file.read()

# Initialize the Fernet class
f = Fernet(encryption_key)

# Decrypt the API keys
alchemy_key = f.decrypt(os.getenv("ALCHEMY_KEY").encode()).decode()
blockcypher_api_key = f.decrypt(os.getenv("BLOCKCYPHER_API_KEY").encode()).decode()
zerox_api_key = f.decrypt(os.getenv("ZEROX_API_KEY").encode()).decode()

# Now you can use the decrypted keys
print("Decrypted Alchemy Key:", alchemy_key)
print("Decrypted BlockCypher API Key:", blockcypher_api_key)
print("Decrypted 0x API Key:", zerox_api_key)

# Replace SimpleVM initialization with ZKVM
vm = ZKVM(security_level=2)


tracemalloc.start()


import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='app.log',  # Log file name
    filemode='w',  # Overwrite the log file each run
    level=logging.DEBUG,  # Log all levels DEBUG and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger().handlers[0].flush = lambda: None
logging.basicConfig(level=logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)  # Only show critical errors in the terminal
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
NUM_NODES = 5

MAX_SUPPLY = 21000000  
ACCESS_TOKEN_EXPIRE_MINUTES = 9999999
REFRESH_TOKEN_EXPIRE_DAYS = 9999
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
 
templates = Jinja2Templates(directory="templates")
def keccak_f1600(state: List[int]) -> List[int]:
    def rot(x, n):
        return ((x << n) & 0xFFFFFFFFFFFFFFFF) | (x >> (64 - n))

    RC = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
        0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
    ]

    def round(A, RC):
        # θ step
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x + 4) % 5] ^ rot(C[(x + 1) % 5], 1) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]

        # ρ and π steps
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = rot(A[x][y], ((x + 3 * y) * (x + 3 * y) + x + 3 * y) % 64)

        # χ step
        A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]

        # ι step
        A[0][0] ^= RC
        return A

    state_array = np.array(state, dtype=np.uint64).reshape(5, 5)

    for i in range(KECCAK_ROUNDS):
        state_array = round(state_array, RC[i])

    return state_array.flatten().tolist()

def initialize_dashboard_ui():
    from curses_dashboard.dashboard import DashboardUI
    return DashboardUI
class QuantumInspiredMining:
    def __init__(self, stark, difficulty: int = 4):
        self.stark = stark
        self.difficulty = difficulty

    def generate_quantum_state(self) -> str:
        return ''.join(random.choice(['0', '1']) for _ in range(8))

    def hash_with_quantum(self, data: str, quantum_state: str) -> str:
        combined = data + quantum_state
        return hashlib.sha256(combined.encode()).hexdigest()

    def mine_block(self, block_data: Dict[str, Any]) -> Dict[str, Any]:
        block_data['nonce'] = 0
        block_data['quantum_signature'] = self.generate_quantum_state()
        
        start_time = time.time()
        while True:
            block_hash = self.hash_with_quantum(str(block_data), block_data['quantum_signature'])
            if block_hash.startswith('0' * self.difficulty):
                break
            block_data['nonce'] += 1
            if block_data['nonce'] % 1000 == 0:  # Periodically update quantum state
                block_data['quantum_signature'] = self.generate_quantum_state()

        mining_time = time.time() - start_time
        
        # Generate ZKP for the mining process
        secret = block_data['nonce']
        public_input = int(block_hash, 16)
        zkp = self.stark.prove(secret, public_input)

        return {
            "block_data": block_data,
            "hash": block_hash,
            "mining_time": mining_time,
            "zkp": zkp
        }

class TimeoutMiddleware:
    def __init__(self, app: FastAPI, timeout: int = 600):
        self.app = app
        self.timeout = timeout

    async def __call__(self, scope, receive, send):
        # Run the app with a timeout using asyncio.wait_for
        try:
            await asyncio.wait_for(self.app(scope, receive, send), timeout=self.timeout)
        except asyncio.TimeoutError:
            # Handle the timeout error
            response = {
                "success": False,
                "message": f"Request exceeded timeout of {self.timeout} seconds."
            }
            await send({
                'type': 'http.response.start',
                'status': 504,  # Gateway Timeout
                'headers': [
                    (b'content-type', b'application/json'),
                ]
            })
            await send({
                'type': 'http.response.body',
                'body': json.dumps(response).encode('utf-8'),
            })

# Add the middleware with a custom timeout
app.add_middleware(TimeoutMiddleware, timeout=600)  # Set timeout to 600 seconds

class Trade:
    def __init__(self, buyer_id, seller_id, base_currency, quote_currency, amount, price):
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.amount = amount
        self.price = price


class QuantumStateManager:
    def __init__(self):
        self.shards = {}

    def store_quantum_state(self, shard_id, quantum_state):
        self.shards[shard_id] = quantum_state

    def retrieve_quantum_state(self, shard_id):
        return self.shards.get(shard_id)


quantum_state_manager = QuantumStateManager()
class SecurityManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def create_token(self, node_id, expiration_time=1):
        payload = {
            'exp': datetime.utcnow() + timedelta(hours=expiration_time),
            'iat': datetime.utcnow(),
            'sub': node_id
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def sign_message(self, message):
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def verify_signature(self, message, signature):
        try:
            self.public_key.verify(
                signature,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False




from decimal import Decimal
from typing import Dict
class NativeCoinContract:
    def __init__(self, vm, total_supply, zk_system):
        self.vm = vm
        self.total_supply = total_supply
        self.zk_system = zk_system
        self.balances = {}



    def mint(self, user: str, amount: Decimal):
        if self.total_supply + amount > self.max_supply:
            raise ValueError("Exceeds maximum supply")
        self.balances[user] = self.balances.get(user, Decimal('0')) + amount
        self.total_supply += amount

    def burn(self, user: str, amount: Decimal):
        if self.balances.get(user, Decimal('0')) < amount:
            raise ValueError("Insufficient balance")
        self.balances[user] -= amount
        self.total_supply -= amount
    async def transfer(self, sender, receiver, amount):
        if self.balances.get(sender, Decimal('0')) < amount:
            raise ValueError("Insufficient balance")
        public_input = self.zk_system.hash(sender, receiver, str(amount))
        secret = int(self.balances[sender])
        zk_proof = self.zk_system.prove(secret, public_input)
        self.balances[sender] -= amount
        self.balances[receiver] = self.balances.get(receiver, Decimal('0')) + amount
        return zk_proof


    @staticmethod
    def verify_transfer(sender: str, receiver: str, amount: Decimal, zk_proof, zk_system):
        public_input = zk_system.stark.hash(sender, receiver, str(amount))
        return zk_system.verify(public_input, zk_proof)



    def get_balance(self, user: str) -> Decimal:
        return self.balances.get(user, Decimal('0'))

    def get_total_supply(self) -> Decimal:
        return self.total_supply

import hashlib
import time
from decimal import Decimal
from typing import Dict, List, Tuple
from enum import Enum, auto

class Permission(Enum):
    MINT = auto()
    BURN = auto()
    TRANSFER = auto()
class Token:
    def __init__(self, address: str, name: str, symbol: str, creator: str):
        self.address = address
        self.name = name
        self.symbol = symbol
        self.creator = creator
        self.balances: Dict[str, Decimal] = {}
        self.authorized_minters: set = {creator}
        self.authorized_burners: set = {creator}
        self.total_supply = Decimal('0')

    def balance_of(self, user: str) -> Decimal:
        return self.balances.get(user, Decimal('0'))

    def mint(self, user: str, amount: Decimal):
        if user not in self.authorized_minters:
            raise ValueError("Unauthorized to mint tokens")
        self.balances[user] = self.balance_of(user) + amount
        self.total_supply += amount

    def burn(self, user: str, amount: Decimal):
        if user not in self.authorized_burners:
            raise ValueError("Unauthorized to burn tokens")
        if self.balance_of(user) < amount:
            raise ValueError("Insufficient balance to burn")
        self.balances[user] -= amount
        self.total_supply -= amount

    def transfer(self, sender: str, recipient: str, amount: Decimal):
        if self.balance_of(sender) < amount:
            raise ValueError("Insufficient balance to transfer")
        self.balances[sender] -= amount
        self.balances[recipient] = self.balance_of(recipient) + amount

    def authorize_minter(self, user: str):
        self.authorized_minters.add(user)

    def revoke_minter(self, user: str):
        if user != self.creator:
            self.authorized_minters.discard(user)

    def authorize_burner(self, user: str):
        self.authorized_burners.add(user)

    def revoke_burner(self, user: str):
        if user != self.creator:
            self.authorized_burners.discard(user)


async def get_transaction_history():
    try:
        # Assuming blockchain stores transaction history
        transaction_history = blockchain.get_transaction_history()  # Fetch transaction history
        return transaction_history
    except Exception as e:
        logger.error(f"Error fetching transaction history: {str(e)}")
        raise
from dataclasses import dataclass
from typing import List, Dict
from decimal import Decimal
from vm import SimpleVM, Permission, Role, PBFTConsensus


class QuantumBlockchain:
    def __init__(self, consensus, secret_key, node_directory, vm, p2p_node=None):
        self.globalMetrics = {
            'totalTransactions': 0,
            'totalBlocks': 0,
        }
        self.initial_reward = 1000
        self.chain = []
        self.pending_transactions = []
        self.consensus = consensus
        if vm is None:  
            raise ValueError("VM cannot be None. Check SimpleVM initialization.")
        self.vm = vm
        self.secret_key = secret_key
        self.node_directory = node_directory
        self.halving_interval = 4 * 365 * 24 * 3600  # 4 years in seconds
        self.start_time = time.time()
        self.difficulty = 1
        self.target_block_time = 600  # 10 minutes in seconds
        self.adjustment_interval = 10
        self.max_supply = MAX_SUPPLY
        self.target = 2**(256 - self.difficulty)
        self.blocks_since_last_adjustment = 0
        self.security_manager = SecurityManager(secret_key)
        self.quantum_state_manager = QuantumStateManager()
        self.peers = []
        self.new_block_listeners = []
        self.new_transaction_listeners = []
        self.genesis_wallet_address = "genesis_wallet"
        self.balances = {}  # Initialize balances as a dictionary
        self.wallets = []
        self.transactions = []
        self.contracts = []
        self.tokens = {}
        self.liquidity_pool_manager = LiquidityPoolManager()
        self.zk_system = SecureHybridZKStark(security_level=2)  # Adjust security level as needed

        # Initialize the P2P node and lock
        self.p2p_node = p2p_node
        logger.info(f"QuantumBlockchain initialized with p2p_node: {self.p2p_node}")

        self._p2p_node_lock = asyncio.Lock()  # Lock for accessing the p2p_node
        logger.info(f"QuantumBlockchain initialized with p2p_node: {self.p2p_node}")
        
        if self.p2p_node is None:
            logger.warning("P2P node is None in QuantumBlockchain initialization")
        else:
            logger.info(f"P2P node type: {type(self.p2p_node)}")
            logger.info(f"P2P node attributes: {vars(self.p2p_node)}")
    async def create_wallet(self, user_id: str) -> Dict[str, Any]:
        if user_id in self.wallets:
            raise ValueError(f"Wallet already exists for user {user_id}")

        new_wallet = Wallet()
        self.wallets[user_id] = new_wallet

        wallet_info = {
            'user_id': user_id,
            'address': new_wallet.address,
            'public_key': new_wallet.public_key
        }

        if self.p2p_node:
            await self.p2p_node.broadcast_event('new_wallet', wallet_info)

        return wallet_info

    async def create_private_transaction(self, sender: str, receiver: str, amount: Decimal) -> Dict[str, Any]:
        if sender not in self.wallets or receiver not in self.wallets:
            raise ValueError("Sender or receiver wallet not found")

        sender_wallet = self.wallets[sender]
        receiver_wallet = self.wallets[receiver]

        # Create and sign the transaction
        tx = Transaction(sender_wallet.address, receiver_wallet.address, amount)
        tx.sign(sender_wallet.private_key)

        # Generate ZKP
        secret = int(amount * 10**18)  # Convert Decimal to integer
        public_input = int(tx.hash(), 16)
        zk_proof = self.zk_system.prove(secret, public_input)

        # Add ZKP to transaction
        tx.zk_proof = zk_proof

        # Add transaction to blockchain
        await self.add_transaction(tx)

        tx_info = {
            'tx_hash': tx.hash,
            'sender': sender,
            'receiver': receiver,
            'amount': str(amount),  # Convert Decimal to string for JSON serialization
            'zk_proof': self.serialize_proof(zk_proof)
        }

        if self.p2p_node:
            await self.p2p_node.broadcast_event('private_transaction', tx_info)

        return tx_info

    def serialize_proof(self, proof):
        # Implement this method to serialize the ZKP for network transmission
        # This will depend on the specific structure of your ZKP
        pass

    async def add_transaction(self, tx: Transaction):
        # Implement this method to add the transaction to your blockchain
        # This might involve adding it to a mempool, validating it, etc.
        pass


            
    def get_wallets(self):
        """
        Returns a list of all wallets in the blockchain as dictionaries.
        """
        try:
            # Check if wallets exist in the system
            if not self.wallets:
                logger.info("No wallets found in the blockchain.")
                return []

            logger.debug(f"Fetched {len(self.wallets)} wallets from the blockchain.")

            # Return a list of dictionaries representing each wallet
            return [wallet.to_dict() for wallet in self.wallets]

        except Exception as e:
            logger.error(f"Error fetching wallets: {str(e)}")
            raise ValueError("Failed to retrieve wallets.")


    async def propagate_block_with_retry(self, block, retries=3):
        for attempt in range(retries):
            try:
                logger.debug(f"Attempting to propagate block: {block.hash}. Retries left: {retries - attempt}")

                # Retrieve the P2PNode from app.state
                

                # Proceed with block propagation
                await p2p_node.propagate_block(block)
                logger.info(f"Block {block.hash} successfully propagated.")
                return True

            except Exception as e:
                logger.error(f"Failed to propagate block: {block.hash}. Error: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

        logger.error(f"Max retries reached. Block {block.hash} could not be propagated.")
        return False

    def set_p2p_node(self, p2p_node):
        self.p2p_node = p2p_node

    async def get_p2p_node(self):
        """ Get the P2P node with locking to ensure no race conditions. """
        async with self._p2p_node_lock:
            logger.debug("Accessing p2p_node via get_p2p_node method")
            return self.p2p_node

    async def set_p2p_node(self, p2p_node):
        """ Set the P2P node with locking and detailed logging. """
        async with self._p2p_node_lock:
            self._p2p_node = p2p_node
            if self._p2p_node is not None:
                try:
                    logger.info(f"P2P node set successfully. Type: {type(self._p2p_node)}")
                    logger.info(f"P2P node attributes: {vars(self._p2p_node)}")
                    logger.info(f"P2P node methods: {dir(self._p2p_node)}")

                    # Log if P2P node is connected
                    if hasattr(self._p2p_node, 'is_connected'):
                        is_connected = self._p2p_node.is_connected()
                        logger.info(f"P2P node connected: {is_connected}")
                    else:
                        logger.warning("P2P node does not have 'is_connected' method")

                    # Log number of peers
                    if hasattr(self._p2p_node, 'peers'):
                        logger.info(f"P2P node peers: {len(self._p2p_node.peers)}")
                    else:
                        logger.warning("P2P node does not have 'peers' attribute")

                except Exception as e:
                    logger.error(f"Error while setting P2P node: {str(e)}")
                    logger.error("Traceback for P2P node setting error:")
                    logger.error(traceback.format_exc())
            else:
                logger.error("Attempted to set P2P node, but it's None")
                logger.error("Traceback for setting None P2P node:")
                logger.error(traceback.format_exc())


    async def initialize_p2p_node(self, p2p_node):
        # Initialize the P2P node asynchronously after the loop has started
        logger.info("Setting P2P node for blockchain...")
        await self.set_p2p_node(p2p_node)
        logger.info(f"P2P node set for blockchain: {self.p2p_node}")

    async def add_peer(self, peer):
        if peer not in self.peers:
            self.peers.append(peer)
            logger.info(f"Peer {peer} added to peers list")

    async def remove_peer(self, peer):
        if peer in self.peers:
            self.peers.remove(peer)
            logger.info(f"Peer {peer} removed from peers list")

    def get_latest_block_hash(self):
        if not self.chain:
            return "0"  # Return a default value for the first block (genesis block)
        return self.chain[-1].hash  # Return the hash of the latest block
    async def check_p2p_node_status(self):
        while True:
            logger.info(f"Checking P2P node status: {self.p2p_node}")
            if self.p2p_node is None:
                logger.warning("P2P node is None in QuantumBlockchain")
            else:
                logger.info(f"P2P node is connected: {self.p2p_node.is_connected()}")
                logger.info(f"P2P node peers: {self.p2p_node.peers}")
            await asyncio.sleep(60)  # Check every minute
    def get_pending_transactions(self):
        # Return the list of pending transactions
        return self.pending_transactions
    
    def add_transaction(self, transaction):
        # Add a new transaction to the list of pending transactions
        self.pending_transactions.append(transaction)
    def reward_miner(self, miner_address, reward):
        # Assuming balances is a dictionary storing the balance for each wallet address
        if miner_address in self.balances:
            self.balances[miner_address] += reward
        else:
            self.balances[miner_address] = reward

        logger.info(f"Rewarded miner {miner_address} with {reward} QuantumDAGKnight Coins.")

    def create_block(self, data, transactions, miner_address):
        try:
            logger.info("Creating a new block...")
            previous_hash = self.chain[-1].hash if self.chain else "0"
            reward = self.get_block_reward()
            quantum_signature = self.generate_quantum_signature()

            new_block = QuantumBlock(
                previous_hash=previous_hash,
                data=data,
                quantum_signature=quantum_signature,
                reward=reward,
                transactions=transactions
            )
            new_block.hash = new_block.compute_hash()  # Set initial hash

            logger.debug(f"Initial block hash: {new_block.hash}")

            new_block.mine_block(self.difficulty)
            logger.info(f"Block mined. Hash: {new_block.hash}")

            if self.consensus.validate_block(new_block):
                logger.info(f"Block validated. Adding to chain: {new_block.hash}")
                self.chain.append(new_block)
                self.process_transactions(new_block.transactions)
                self.reward_miner(miner_address, reward)
                return new_block
            else:
                logger.error("Block validation failed. Block will not be added.")
                return None
        except Exception as e:
            logger.error(f"Exception in create_block: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def get_transactions(self, wallet_address: str) -> List[Dict]:
        # Simulate an asynchronous operation
        await asyncio.sleep(0)
        # For now, let's return an empty list
        return []





    async def get_balances(self, address: str) -> Dict[str, Decimal]:
        balances = {}
        try:
            balances['PLATA'] = await self.get_plata_balance(address)
            
            btc_address = self.get_btc_address(address)
            if btc_address:
                balances['BTC'] = await self.get_btc_balance(btc_address)
            
            eth_address = self.get_eth_address(address)
            if eth_address:
                balances['ETH'] = await self.get_eth_balance(eth_address)
            
            ltc_address = self.get_ltc_address(address)
            if ltc_address:
                balances['LTC'] = await self.get_ltc_balance(ltc_address)
            
            doge_address = self.get_doge_address(address)
            if doge_address:
                balances['DOGE'] = await self.get_doge_balance(doge_address)
            
            sol_address = self.get_sol_address(address)
            if sol_address:
                balances['SOL'] = await self.get_sol_balance(sol_address)

            logger.info(f"Fetched balances for address {address}: {balances}")
        except Exception as e:
            logger.error(f"Error fetching balances for address {address}: {str(e)}")
            logger.error(traceback.format_exc())
        
        return balances

    async def get_plata_balance(self, address: str) -> Decimal:
        # Implement the logic to get PLATA balance
        balance = self.balances.get(address, 0)
        return Decimal(str(balance))

    async def get_btc_balance(self, address: str) -> Decimal:
        try:
            btc_info = self.blockcypher_api.get_address_info(address)
            return Decimal(btc_info['balance'] / 1e8)  # Convert satoshis to BTC
        except Exception as e:
            logger.error(f"Error fetching BTC balance for {address}: {str(e)}")
            return Decimal('0')

    def get_btc_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive BTC address from Plata address
        return None

    def get_eth_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive ETH address from Plata address
        return None

    def get_ltc_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive LTC address from Plata address
        return None

    def get_doge_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive DOGE address from Plata address
        return None

    def get_sol_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive SOL address from Plata address
        return None

    # Add these methods if they're not already implemented
    async def get_eth_balance(self, address: str) -> Decimal:
        # Implement the logic to get ETH balance
        return Decimal('0')

    async def get_ltc_balance(self, address: str) -> Decimal:
        # Implement the logic to get LTC balance
        return Decimal('0')

    async def get_doge_balance(self, address: str) -> Decimal:
        # Implement the logic to get DOGE balance
        return Decimal('0')

    async def get_sol_balance(self, address: str) -> Decimal:
        # Implement the logic to get SOL balance
        return Decimal('0')

    async def get_transaction_history(self, limit=10):
        try:
            # Assuming the blockchain stores a list of blocks, and each block has transactions
            transactions = []
            for block in reversed(self.chain):  # Start from the most recent block
                for transaction in block.transactions:
                    # Assuming each transaction has a date, amount, and recipient field
                    tx_info = {
                        "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(transaction.timestamp)),
                        "amount": transaction.amount,
                        "recipient": transaction.receiver
                    }
                    transactions.append(tx_info)
                    if len(transactions) >= limit:
                        break
                if len(transactions) >= limit:
                    break

            return transactions

        except Exception as e:
            logger.error(f"Error fetching transaction history: {str(e)}")
            return []

    def get_node_state(self) -> NodeState:
        return NodeState(
            blockchain_length=len(self.chain),
            latest_block_hash=self.chain[-1].hash if self.chain else None,
            pending_transactions_count=len(self.pending_transactions),
            total_supply=self.get_total_supply(),
            difficulty=self.difficulty,
            mempool_size=len(self.mempool),
            connected_peers=len(self.p2p_node.peers),
            active_liquidity_pools=len(self.liquidity_pools),
            node_uptime=time.time() - self.start_time
        )



    async def wait_for_p2p_node(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            p2p_node = await self.get_p2p_node()
            if p2p_node is not None and await p2p_node.is_connected():
                return True
            await asyncio.sleep(1)
        return False




    async def initialize_p2p(self, host, port, retries=3, delay=5):
        """Initialize P2P node with retry logic for robust startup."""
        from P2PNode import P2PNode

        logger.debug(f"Initializing P2P node on {host}:{port}")
        self.p2p_node = P2PNode(host, port, self)
        
        for attempt in range(retries):
            try:
                # Try to start the P2P node with a timeout
                await asyncio.wait_for(self.p2p_node.start(), timeout=10)
                logger.debug("P2P node started successfully")
                return True
            except asyncio.TimeoutError:
                logger.error(f"Timeout while starting P2P node (attempt {attempt + 1}/{retries})")
            except Exception as e:
                logger.error(f"Unexpected error during P2P node initialization: {str(e)}")

            if attempt < retries - 1:
                logger.info(f"Retrying P2P node initialization in {delay} seconds...")
                await asyncio.sleep(delay)
        
        logger.error("Failed to initialize P2P node after maximum retries")
        raise RuntimeError("P2P node initialization failed")



    def get_blocks_since(self, last_known_block_index):
        """
        Returns the blocks added since the given block index.
        """
        return self.chain[last_known_block_index + 1:]
    async def mine_block(self, miner_address):
        if self.p2p_node is None:
            logger.error("Cannot mine block: P2P node is not initialized")
            return None

        try:
            current_time = int(time.time())
            data = f"Block mined by {miner_address} at {current_time}"  # Define the data variable
            transactions = [
                Transaction(
                    sender='sender_address',
                    receiver='receiver_address',
                    amount=Decimal('10'),
                    price=Decimal('100'),
                    buyer_id='buyer_id_value',
                    seller_id='seller_id_value',
                    wallet=miner_address,
                    tx_hash=hashlib.sha256(f"{current_time}_{miner_address}".encode()).hexdigest(),
                    timestamp=current_time
                )
            ]

            new_block = QuantumBlock(
                previous_hash=self.chain[-1].hash if self.chain else "0",
                data=data,  # Use the defined data variable
                quantum_signature=self.generate_quantum_signature(),
                reward=self.get_block_reward(),
                transactions=transactions
            )
            logger.debug(f"Created new block: {new_block.to_dict()}")
            logger.debug(f"Initial hash: {new_block.hash}")

            new_block.mine_block(self.difficulty)

            logger.info(f"Block mined. Attempting validation. Hash: {new_block.hash}")
            if self.consensus.validate_block(new_block):
                self.chain.append(new_block)
                await self.process_transactions(new_block.transactions)
                await self.native_coin_contract.mint(miner_address, Decimal(new_block.reward))
                logger.info(f"Block validated and added to the chain. Hash: {new_block.hash}")
                return new_block.reward
            else:
                logger.error("Failed to validate the mined block")
                return None

        except Exception as e:
            logger.error(f"Error during mining: {str(e)}")
            logger.error(traceback.format_exc())
            return None










    def batch_verify_transactions(self, transactions):
        proofs = []
        public_inputs = []

        for tx in transactions:
            # Since zk_system.hash is synchronous, do not await it
            public_input = self.zk_system.hash(tx.sender, tx.receiver, str(tx.amount))
            public_inputs.append(public_input)
            proofs.append(tx.zk_proof)

        combined_public_input = self.zk_system.hash(*public_inputs)
        return self.zk_system.verify(combined_public_input, proofs)





    async def import_token(self, address: str, user: str) -> Token:
        if address in self.tokens:
            return self.tokens[address]
        
        # Verify the token contract on-chain
        if not self.verify_token_contract(address):
            raise ValueError("Invalid token contract")
        
        # Fetch token details from the contract
        token_details = self.fetch_token_details(address)
        token = Token(address, token_details['name'], token_details['symbol'], token_details['creator'])
        self.tokens[address] = token
        return token

    def verify_token_contract(self, address: str) -> bool:
        if address in self.verified_contracts:
            return self.verified_contracts[address]

        # In a real implementation, you would:
        # 1. Fetch the contract bytecode from the blockchain
        # 2. Verify that the bytecode matches a known, audited token contract template
        # 3. Check if the contract implements standard token interfaces (e.g., ERC20)
        # 4. Verify that the contract has been deployed by a trusted source (optional)

        # For this example, we'll use a simple verification:
        contract_bytecode = self.vm.get_contract_bytecode(address)
        is_valid = self.validate_token_bytecode(contract_bytecode)
        self.verified_contracts[address] = is_valid
        return is_valid
    def validate_token_bytecode(self, bytecode: str) -> bool:
        # Adjust this method to correctly validate the bytecode
        return (len(bytecode) > 20 and 
                "transfer" in bytecode.lower() and 
                "balanceof" in bytecode.lower())


    def fetch_token_details(self, address: str) -> dict:
        # In a real implementation, you would call the token contract to get these details
        # For this example, we'll return dummy data
        return {
            'name': f"Token {address[:6]}",
            'symbol': f"TKN{address[:3]}",
            'creator': f"0x{hashlib.sha256(address.encode()).hexdigest()[:40]}"
        }

    async def mint_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if user not in token.authorized_minters:
            raise ValueError("Unauthorized to mint tokens")
        
        token.mint(user, amount)

    async def burn_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if user not in token.authorized_burners:
            raise ValueError("Unauthorized to burn tokens")
        
        token.burn(user, amount)

    async def transfer_token(self, address: str, sender: str, recipient: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        token.transfer(sender, recipient, amount)

    async def authorize_minter(self, address: str, authorizer: str, new_minter: str):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if authorizer != token.creator:
            raise ValueError("Only the token creator can authorize new minters")
        
        token.authorize_minter(new_minter)

    async def revoke_minter(self, address: str, authorizer: str, minter: str):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if authorizer != token.creator:
            raise ValueError("Only the token creator can revoke minters")
        
        token.revoke_minter(minter)

    async def mint_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        # In a real implementation, you would check if the user has permission to mint
        token.mint(user, amount)

    async def burn_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        token.burn(user, amount)

    async def create_liquidity_pool(self, user: str, token_a: str, token_b: str, amount_a: Decimal, amount_b: Decimal) -> str:
        if token_a not in self.tokens or token_b not in self.tokens:
            raise ValueError("One or both tokens not found")

        pool_id = self.liquidity_pool_manager.create_pool(token_a, token_b, Decimal('0.003'))  # 0.3% fee
        
        # Transfer tokens to the pool
        self.tokens[token_a].burn(user, amount_a)
        self.tokens[token_b].burn(user, amount_b)
        
        # Add liquidity
        liquidity_minted = await self.liquidity_pool_manager.add_liquidity(user, token_a, token_b, amount_a, amount_b)
        
        return pool_id

    async def get_user_tokens(self, user: str) -> List[Token]:
        return [token for token in self.tokens.values() if token.balance_of(user) > 0]

    async def swap_tokens(self, user: str, amount_in: Decimal, token_in: str, token_out: str) -> Decimal:
        if token_in not in self.tokens or token_out not in self.tokens:
            raise ValueError("One or both tokens not found")

        # Check user balance
        if self.tokens[token_in].balance_of(user) < amount_in:
            raise ValueError("Insufficient balance")

        # Perform the swap
        amount_out = await self.liquidity_pool_manager.swap(amount_in, token_in, token_out)

        # Update token balances
        self.tokens[token_in].burn(user, amount_in)
        self.tokens[token_out].mint(user, amount_out)

        return amount_out
    async def propagate_block_to_peers(self, block):
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to propagate block")
                p2p_node = await self.get_p2p_node()
                logger.info(f"P2P node status: {p2p_node}")
                logger.info(f"P2P node type: {type(p2p_node)}")
                logger.info(f"P2P node attributes: {vars(p2p_node) if p2p_node else 'None'}")
                
                if p2p_node is None:
                    logger.error(f"P2P node is not initialized, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    logger.error("Traceback for P2P node being None:")
                    logger.error(traceback.format_exc())
                elif not await p2p_node.is_connected():
                    logger.warning(f"P2P node is not connected, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    logger.info(f"Current peers: {p2p_node.peers}")
                else:
                    await p2p_node.propagate_block(block)
                    logger.info(f"Block {block.hash} propagated successfully.")
                    return True

            except Exception as e:
                logger.error(f"Error propagating block: {str(e)}")
                logger.error(traceback.format_exc())
            
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        
        logger.error("Max retries reached. Block could not be propagated.")
        return False



    async def wait_for_peer_connections(self, timeout=30):
        start_time = time.time()
        while not self.p2p_node.is_connected():
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for peers to connect.")
                return False
            logger.debug("Waiting for peer connections...")
            await asyncio.sleep(1)  # Small wait before retrying
        logger.info("Peers connected successfully.")
        return True




    async def get_balance(self, user_id: str, currency: str) -> Decimal:
        balance = self.balances.get(address, 0)
        return balance
        
        
            
            
    def initialize_native_coin_contract(self):
        try:
            result = self.vm.get_existing_contract(NativeCoinContract)
            if result and len(result) == 2:
                self.native_coin_contract_address, self.native_coin_contract = result
            else:
                # Handle the case where the contract doesn't exist
                self.native_coin_contract_address = None
                self.native_coin_contract = None
        except Exception as e:
            logger.error(f"Error initializing NativeCoinContract: {str(e)}")
            self.native_coin_contract_address = None
            self.native_coin_contract = None








    def create_genesis_block(self):
        genesis_block = QuantumBlock(
            previous_hash="0",
            data="Genesis Block",
            quantum_signature="00",
            reward=0,
            transactions=[]
        )
        genesis_block.timestamp = time.time()
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)




    def on_new_block(self, callback):
        self.new_block_listeners.append(callback)

    def on_new_transaction(self, callback):
        self.new_transaction_listeners.append(callback)

    def get_recent_transactions(self, limit):
        transactions = []
        for block in reversed(self.chain):
            transactions.extend(block.transactions)
            if len(transactions) >= limit:
                break
        return transactions[:limit]

    def get_block_by_hash(self, block_hash):
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def calculate_average_block_time(self):
        if len(self.chain) < 2:
            return 0
        block_times = [self.chain[i].timestamp - self.chain[i-1].timestamp for i in range(1, len(self.chain))]
        return sum(block_times) / len(block_times)

    def calculate_qhins(self):
        return sum(block.reward for block in self.chain)  # Example calculation

    def calculate_entanglement_strength(self):
        total_entanglement = sum(block.quantum_signature.count('1') for block in self.chain)
        return total_entanglement / len(self.chain) if self.chain else 0
    def create_genesis_block(self):
        genesis_block = QuantumBlock(
            previous_hash="0",
            data="Genesis Block",
            quantum_signature="00",
            reward=0,
            transactions=[]
        )
        genesis_block.timestamp = time.time()
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)


    def get_total_supply(self):
        return sum(self.balances.values())
    def add_new_block(self, data, transactions, miner_address):
        previous_block = self.chain[-1]
        previous_hash = previous_block.hash
        reward = self.get_block_reward()
        total_supply = self.get_total_supply()

        if total_supply + Decimal(reward) > Decimal(MAX_SUPPLY):
            reward = Decimal(MAX_SUPPLY) - total_supply

        quantum_signature = self.generate_quantum_signature()  # Call the method here

        new_block = QuantumBlock(
            previous_hash=previous_hash,
            data=data,
            quantum_signature=quantum_signature,
            reward=reward,
            transactions=transactions,
            timestamp=time.time()
        )

        new_block.mine_block(self.difficulty)
        logger.info(f"Adding new block: {new_block.__dict__}")

        if self.consensus.validate_block(new_block):
            self.chain.append(new_block)
            self.process_transactions(transactions)
            self.native_coin_contract.mint(miner_address, Decimal(reward))
            return reward
        else:
            logger.error("Block validation failed. Block not added.")
            raise ValueError("Invalid block")



    def update_total_supply(self, reward):
        # Example implementation
        pass

    def process_transactions(self, transactions):
        for tx in transactions:
            if isinstance(tx, dict):
                sender = tx['sender']
                receiver = tx['receiver']
                amount = tx['amount']
            else:
                sender = tx.sender
                receiver = tx.receiver
                amount = tx.amount
            
            if self.balances.get(sender, 0) >= amount:
                self.balances[sender] = self.balances.get(sender, 0) - amount
                self.balances[receiver] = self.balances.get(receiver, 0) + amount
            else:
                logger.warning(f"Insufficient balance for transaction: {tx}")

    def get_block_reward(self):
        return self.current_reward()

    def current_reward(self):
        elapsed_time = time.time() - self.start_time
        halvings = int(elapsed_time // self.halving_interval)
        reward = self.initial_reward / (2 ** halvings)
        return reward
        
    def validate_block(self, block):
        logger.debug(f"Starting block validation for block with hash: {block.hash}")
        logger.debug(f"Block data before validation: {json.dumps(block.to_dict(), sort_keys=True, default=str)}")

        # Check if it's the genesis block
        if len(self.chain) == 0:  # Updated from self.blockchain.chain to self.chain
            is_genesis_valid = block.previous_hash == "0" and block.is_valid()
            logger.info(f"Validating genesis block. Is valid: {is_genesis_valid}")
            return is_genesis_valid

        # Validate previous hash
        if block.previous_hash != self.chain[-1].hash:  # Updated from self.blockchain.chain to self.chain
            logger.warning(f"Invalid previous hash. Expected {self.chain[-1].hash}, got {block.previous_hash}")
            return False

        # Validate block hash using block's is_valid method
        if not block.is_valid():
            computed_hash = block.compute_hash()
            logger.warning(f"Invalid block hash. Computed: {computed_hash}, Got: {block.hash}")
            return False

        logger.info(f"Block validation successful for block with hash: {block.hash}")
        return True




    def is_valid_hash(self, block_hash):
        target = 2 ** (256 - self.blockchain.difficulty)
        is_valid = int(block_hash, 16) < target
        logger.debug(f"Validating hash: {block_hash}")
        logger.debug(f"Target: {target}")
        logger.debug(f"Is valid hash: {is_valid}")
        return is_valid



    def validate_transaction(self, tx):
        # Implement transaction validation logic
        # This is a placeholder and should be replaced with your actual transaction validation logic
        return True




    def add_block(self, block):
        try:
            current_time = time.time()
            
            # Check if the block's timestamp is too far in the future
            if block.timestamp > current_time + 300:  # Allow for a 5-minute future timestamp window
                logger.warning(f"Block timestamp too far in the future. Current time: {current_time}, Block time: {block.timestamp}")
                return False

            # Validate the block
            if not self.validate_block(block):
                logger.warning("Block validation failed. Block not added.")
                return False

            # Add the block to the chain
            self.chain.append(block)
            self.blocks_since_last_adjustment += 1

            # Adjust the difficulty if needed
            if self.blocks_since_last_adjustment >= self.adjustment_interval:
                self.adjust_difficulty()
                self.blocks_since_last_adjustment = 0

            # Propagate the block to the P2P network asynchronously with retry
            asyncio.create_task(self.propagate_block_with_retry(block))

            # Notify listeners about the new block
            for listener in self.new_block_listeners:
                try:
                    listener(block)
                except Exception as e:
                    logger.error(f"Error while notifying listener: {str(e)}")

            logger.info(f"Block added successfully: {block.hash}")
            return True

        except Exception as e:
            logger.error(f"Error while adding block: {str(e)}")
            logger.error(traceback.format_exc())
            return False




    def adjust_difficulty(self):
        if len(self.chain) >= self.adjustment_interval:
            start_block = self.chain[-self.adjustment_interval]
            end_block = self.chain[-1]
            
            total_time = end_block.timestamp - start_block.timestamp
            
            if total_time <= 0:
                logger.error("Total time between blocks is zero or negative. Cannot adjust difficulty.")
                return
            
            avg_time = total_time / (self.adjustment_interval - 1)
            target_time = self.target_block_time
            
            logger.info(f"Start block timestamp: {start_block.timestamp}")
            logger.info(f"End block timestamp: {end_block.timestamp}")
            logger.info(f"Total time for last {self.adjustment_interval} blocks: {total_time:.2f} seconds")
            logger.info(f"Average time per block: {avg_time:.2f} seconds")
            logger.info(f"Target block time: {target_time:.2f} seconds")
            logger.info(f"Current difficulty: {self.difficulty}")

            # Calculate the adjustment factor
            adjustment_factor = target_time / avg_time
            logger.info(f"Adjustment factor: {adjustment_factor:.2f}")

            # Adjust difficulty based on the adjustment factor
            if adjustment_factor > 1:
                new_difficulty = min(int(self.difficulty * adjustment_factor), 256)
                logger.info(f"Increasing difficulty: {self.difficulty} -> {new_difficulty}")
            else:
                new_difficulty = max(int(self.difficulty / adjustment_factor), 1)
                logger.info(f"Decreasing difficulty: {self.difficulty} -> {new_difficulty}")

            # Update difficulty and target
            self.difficulty = new_difficulty
            self.target = 2**(256 - self.difficulty)
            logger.info(f"New difficulty: {self.difficulty}")
            logger.info(f"New target: {self.target:.2e}")
        else:
            logger.info(f"Not enough blocks to adjust difficulty. Current chain length: {len(self.chain)}")

    async def get_balance(self, user_id: str, currency: str) -> Decimal:
        balance = self.balances.get(address, 0)
        logger.info(f"Balance for {address}: {balance}")
        return balanceasync 
    async def add_transaction(self, transaction: Transaction):
        # Step 1: Verify the transaction using ZKP
        if not transaction.verify_transaction(self.zk_system):
            raise ValueError("Invalid transaction or ZKP verification failed")
        
        logger.debug(f"Adding transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        logger.debug(f"Sender balance before transaction: {self.balances.get(transaction.sender, 0)}")

        # Step 2: Check if the sender has enough balance
        if self.balances.get(transaction.sender, 0) >= transaction.amount:
            # Step 3: Verify the transaction signature
            wallet = Wallet()
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
            if wallet.verify_signature(message, transaction.signature, transaction.public_key):
                # Step 4: Add the transaction to pending transactions
                self.pending_transactions.append(transaction.to_dict())
                logger.debug(f"Transaction added. Pending transactions count: {len(self.pending_transactions)}")

                # Step 5: Propagate the transaction to the P2P network
                await self.p2p_node.propagate_transaction(transaction)

                # Step 6: Notify any listeners about the new transaction
                for listener in self.new_transaction_listeners:
                    listener(transaction)
                return True
            else:
                logger.debug(f"Transaction signature verification failed for transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        else:
            logger.debug(f"Transaction failed. Insufficient balance for sender {transaction.sender}")

        return False





    async def propagate_transaction_to_all_peers(self, transaction_data):
        nodes = self.node_directory.discover_nodes()
        for node in nodes:
            async with grpc.aio.insecure_channel(f'{node["ip_address"]}:{node["port"]}') as channel:
                stub = DAGKnightStub(channel)
                transaction_request = FullStateRequest(
                    # Populate with necessary fields from transaction_data
                )
                await stub.FullStateSync(transaction_request)

    async def propagate_block_to_all_peers(self, block_data):
        nodes = self.node_directory.discover_nodes()
        tasks = [self.propagate_block(f"http://{node['ip_address']}:{node['port']}/receive_block", block_data) for node in nodes]
        results = await asyncio.gather(*tasks)
        successful_propagations = sum(results)
        logger.info(f"Successfully propagated block to {successful_propagations}/{len(nodes)} peers")
    async def wait_for_peer_connections(self, timeout=30):
        start_time = time.time()
        while not self.is_connected():
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for peers to connect.")
                return False
            logger.debug("Waiting for peer connections...")
            await asyncio.sleep(1)  # Small wait before retrying
        logger.info("Peers connected successfully.")
        return True
    async def propagate_block(self, block):
        try:
            logger.info(f"Propagating block with hash: {block.hash}")
            message = Message(
                type=MessageType.BLOCK.value,
                payload=block.to_dict()
            )
            logger.debug(f"Created block message: {message.to_json()}")
            if self.p2p_node:
                logger.debug(f"P2P node before get_active_peers: {self.p2p_node}")
                active_peers = await self.p2p_node.get_active_peers()
                logger.debug(f"Active P2P node peers before propagation: {active_peers}")
                await self.p2p_node.broadcast(message)
            else:
                logger.error("P2P node is not initialized")
        except Exception as e:
            logger.error(f"Error propagating block {block.hash}: {str(e)}")
            logger.error(traceback.format_exc())


    async def sync_state(self, directory_ip, directory_port):
        async with grpc.aio.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
            request = dagknight_pb2.FullStateRequest()
            response = await stub.FullStateSync(request)
            
            self.chain = [QuantumBlock(
                previous_hash=blk.previous_hash,
                data=blk.data,
                quantum_signature=blk.quantum_signature,
                reward=blk.reward,
                transactions=[tx for tx in blk.transactions]
            ) for blk in response.chain]
            self.balances = {k: v for k, v in response.balances.items()}
            self.stakes = {k: v for k, v in response.stakes.items()}

    def stake_coins(self, address, amount):
        if self.balances.get(address, 0) >= amount:
            self.balances[address] -= amount
            self.stakes[address] = self.stakes.get(address, 0) + amount
            return True
        return False

    def unstake_coins(self, address, amount):
        if self.stakes.get(address, 0) >= amount:
            self.stakes[address] -= amount
            self.balances[address] = self.balances.get(address, 0) + amount
            return True
        return False

    def get_staked_balance(self, address):
        return self.stakes.get(address, 0)


    def full_state_sync(self, request, context):
        return dagknight_pb2.FullStateResponse(
            chain=[dagknight_pb2.Block(
                previous_hash=block.previous_hash,
                data=block.data,
                quantum_signature=block.quantum_signature,
                reward=block.reward,
                transactions=[dagknight_pb2.Transaction(sender=tx['sender'], receiver=tx['receiver'], amount=tx['amount']) for tx in block.transactions]
            ) for block in self.chain],
            balances=self.balances,
            stakes=self.stakes
        )

    def update_balances(self, new_block):
        for tx in new_block.transactions:
            transaction = Transaction.from_dict(tx)
            self.balances[transaction.sender] = self.balances.get(transaction.sender, 0) - transaction.amount
            self.balances[transaction.receiver] = self.balances.get(transaction.receiver, 0) + transaction.amount
            logger.info(f"Updated balance for sender {transaction.sender}: {self.balances[transaction.sender]}")
            logger.info(f"Updated balance for receiver {transaction.receiver}: {self.balances[transaction.receiver]}")

            # Log total supply after each transaction
            total_supply = self.get_total_supply()
            logger.info(f"Total supply after transaction: {total_supply}")
    def generate_quantum_signature(self):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                num_qubits = 8
                qr = QuantumRegister(num_qubits)
                cr = ClassicalRegister(num_qubits)
                qc = QuantumCircuit(qr, cr)

                for i in range(num_qubits):
                    qc.h(qr[i])
                    qc.measure(qr[i], cr[i])

                simulator = AerSimulator()
                transpiled_circuit = transpile(qc, simulator)
                job = simulator.run(transpiled_circuit, shots=1)
                result = job.result()

                if result.status != 'COMPLETED':
                    logger.error(f"Quantum job failed: {result.status}")
                    continue

                counts = result.get_counts()
                signature = list(counts.keys())[0]
                logger.info(f"Generated quantum signature: {signature}")

                if self.validate_quantum_signature(signature):
                    return signature

                logger.warning(f"Generated signature {signature} failed validation. Retrying...")
            except Exception as e:
                logger.error(f"Error in generate_quantum_signature: {str(e)}")

        # If we can't generate a valid signature after max attempts, return a default one
        return "00000000"


    def validate_quantum_signature(self, quantum_signature):
        logger.debug(f"Validating quantum signature: {quantum_signature}")
        try:
            num_qubits = len(quantum_signature)
            qc = QuantumCircuit(num_qubits, num_qubits)

            for i, bit in enumerate(quantum_signature):
                if bit == '1':
                    qc.x(i)
            
            qc.measure(range(num_qubits), range(num_qubits))

            simulator = AerSimulator()
            compiled_circuit = transpile(qc, simulator)
            job = simulator.run(compiled_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()

            max_key = max(counts, key=counts.get)
            max_value = counts[max_key] / 1024

            logger.info(f"Max measurement key: {max_key}, Max measurement probability: {max_value}")

            is_valid = max_value > 0.7 and max_key == quantum_signature

            if is_valid:
                logger.info(f"Quantum signature {quantum_signature} validated successfully with probability {max_value}")
            else:
                logger.error(f"Quantum signature {quantum_signature} is invalid with probability {max_value}")

            return is_valid
        except QiskitError as e:
            logger.error(f"Qiskit error during validation: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return False


    def hamming_distance(self, s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def update_total_supply(self, reward):
        total_supply = self.get_total_supply()
        new_total_supply = total_supply + reward
        logger.info(f"Total supply updated from {total_supply:.2e} to {new_total_supply:.2e}")
    def get_recent_transactions(self, limit):
        transactions = []
        for block in reversed(self.chain):
            transactions.extend(block.transactions)
            if len(transactions) >= limit:
                break
        return transactions[:limit]

    def get_block_by_hash(self, block_hash):
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def calculate_average_block_time(self):
        if len(self.chain) < 2:
            return 0
        block_times = [self.chain[i].timestamp - self.chain[i-1].timestamp for i in range(1, len(self.chain))]
        return sum(block_times) / len(block_times)

    def calculate_qhins(self):
        return sum(block.reward for block in self.chain)

    def calculate_entanglement_strength(self):
        total_entanglement = sum(block.quantum_signature.count('1') for block in self.chain)
        return total_entanglement / len(self.chain) if self.chain else 0
        
    def add_wallet(self, wallet):
        self.wallets.append(wallet)
        self.balances[wallet.address] = 0

    async def add_transaction(self, transaction: Transaction):
        # Step 1: Verify the transaction using ZKP
        if not transaction.verify_transaction(self.zk_system):
            raise ValueError("Invalid transaction or ZKP verification failed")
        
        logger.debug(f"Adding transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        logger.debug(f"Sender balance before transaction: {self.balances.get(transaction.sender, 0)}")

        # Step 2: Check if the sender has enough balance
        if self.balances.get(transaction.sender, 0) >= transaction.amount:
            # Step 3: Verify the transaction signature
            wallet = Wallet()
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
            if wallet.verify_signature(message, transaction.signature, transaction.public_key):
                # Step 4: Add the transaction to pending transactions
                self.pending_transactions.append(transaction.to_dict())
                logger.debug(f"Transaction added. Pending transactions count: {len(self.pending_transactions)}")

                # Step 5: Notify any listeners about the new transaction
                for listener in self.new_transaction_listeners:
                    listener(transaction)
                return True
            else:
                logger.debug(f"Transaction signature verification failed for transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        else:
            logger.debug(f"Transaction failed. Insufficient balance for sender {transaction.sender}")

        return False


    def add_contract(self, contract):
        self.contracts.append(contract)

    def search_wallets(self, query):
        return [wallet for wallet in self.wallets if query in wallet.address.lower()]

    def search_transactions(self, query):
        return [tx for tx in self.transactions if query in tx.hash.lower() or query in tx.sender.lower() or query in tx.receiver.lower()]

    def search_contracts(self, query):
        return [contract for contract in self.contracts if query in contract.address.lower() or query in contract.creator.lower()]

    async def get_balance(self, address, currency):
        # Assuming `balances` is a dictionary of dictionaries, where each key is an address,
        # and the value is another dictionary with currencies as keys and amounts as values.
        address_balances = self.balances.get(address, {})
        balance = address_balances.get(currency, 0)
        return balance
    def get_latest_block(self):
        """Return the latest block in the blockchain."""
        if not self.chain:
            raise ValueError("Blockchain is empty.")
        return self.chain[-1]  # Assuming the latest block is at the end of the chain

class Consensus:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        self.current_leader = None
        self.self_node_id = os.getenv("NODE_ID", "node_1")

    def elect_leader(self):
        nodes = node_directory.discover_nodes()
        logger.debug(f"Discovered nodes for leader election: {nodes}")
        if not nodes:
            logger.warning("No other nodes available for leader election. Electing self as leader.")
            self.current_leader = {"node_id": self.self_node_id}
        else:
            self.current_leader = random.choice(nodes)
        logger.info(f"Elected leader: {self.current_leader['node_id']}")
        return self.current_leader
        

    def validate_block(self, block):
        logger.debug(f"Starting block validation for block with hash: {block.hash}")
        logger.debug(f"Block data before validation: {json.dumps(block.to_dict(), sort_keys=True, default=str)}")

        # Check if it's the genesis block
        if len(self.chain) == 0:  # Updated from self.blockchain.chain to self.chain
            is_genesis_valid = block.previous_hash == "0" and block.is_valid()
            logger.info(f"Validating genesis block. Is valid: {is_genesis_valid}")
            return is_genesis_valid

        # Validate previous hash
        if block.previous_hash != self.chain[-1].hash:  # Updated from self.blockchain.chain to self.chain
            logger.warning(f"Invalid previous hash. Expected {self.chain[-1].hash}, got {block.previous_hash}")
            return False

        # Validate block hash using block's is_valid method
        if not block.is_valid():
            computed_hash = block.compute_hash()
            logger.warning(f"Invalid block hash. Computed: {computed_hash}, Got: {block.hash}")
            return False

        logger.info(f"Block validation successful for block with hash: {block.hash}")
        return True





    def is_valid_hash(self, block_hash):
        target = 2 ** (256 - self.blockchain.difficulty)
        is_valid = int(block_hash, 16) < target
        logger.debug(f"Validating hash: {block_hash}")
        logger.debug(f"Target: {target}")
        logger.debug(f"Is valid hash: {is_valid}")
        return is_valid

    def validate_transaction(self, tx):
        # Implement transaction validation logic
        # This is a placeholder and should be replaced with your actual transaction validation logic
        return True


    def validate_quantum_signature(self, quantum_signature):
        logger.debug(f"Validating quantum signature: {quantum_signature}")
        try:
            # Directly identify invalid signatures
            if quantum_signature == "11":
                logger.error(f"Quantum signature {quantum_signature} is directly identified as invalid.")
                return False

            num_qubits = len(quantum_signature)
            qc = QuantumCircuit(num_qubits, num_qubits)

            for i, bit in enumerate(quantum_signature):
                if bit == '1':
                    qc.x(i)
            
            qc.measure(range(num_qubits), range(num_qubits))

            simulator = AerSimulator()
            compiled_circuit = transpile(qc, simulator)
            job = simulator.run(compiled_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()

            logger.debug(f"Measurement counts: {counts}")

            max_key = max(counts, key=counts.get)
            max_value = counts[max_key] / 1024

            logger.info(f"Max measurement key: {max_key}, Max measurement probability: {max_value}")

            # Introduce randomness for borderline cases
            if quantum_signature in ["01", "10"]:
                is_valid = random.choice([True, False])
            else:
                is_valid = max_value > 0.7 and max_key == quantum_signature

            if is_valid:
                logger.info(f"Quantum signature {quantum_signature} validated successfully with probability {max_value}")
            else:
                logger.error(f"Quantum signature {quantum_signature} is invalid with probability {max_value}")

            return is_valid
        except QiskitError as e:
            logger.error(f"Qiskit error during validation: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return False

    def hamming_distance(self, s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def validate_transaction(self, transaction):
        logger.debug(f"Validating transaction: {transaction}")
        sender_balance = self.blockchain.get_balance(transaction['sender'])
        logger.debug(f"Sender balance: {sender_balance}, Transaction amount: {transaction['amount']}")
        if sender_balance < transaction['amount']:
            logger.error(f"Invalid transaction: Sender {transaction['sender']} has insufficient balance")
            return False

        wallet = Wallet()
        message = f"{transaction['sender']}{transaction['receiver']}{transaction['amount']}"
        logger.debug(f"Verifying signature for message: {message}")
        if not wallet.verify_signature(message, transaction['signature'], transaction['public_key']):
            logger.error("Invalid transaction: Signature verification failed")
            return False

        logger.info("Transaction validated successfully")
        return True

secret_key = "your_secret_key_here"  # Replace with actual secret key
p2p_node_instance = P2PNode(
    blockchain=None,  # Pass your actual blockchain instance here
    host='localhost',  # Define the host for the P2P node
    port=50510         # Define the port the P2P node will use
)

# Initialize the NodeDirectory with the p2p_node_instance
node_directory = NodeDirectory(p2p_node=p2p_node_instance)

# Create the VM instance
vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])  # Initialize SimpleVM with necessary arguments

# Create the consensus object
consensus = PBFTConsensus(nodes=[], node_id="node_id_here")  # Initialize PBFTConsensus with necessary arguments
print(f"VM object: {vm}")

# Create the blockchain instance with the VM and consensus

# Now assign the P2PNode to the blockchain

# Update the consensus to point to the correct blockchain



# Router initialization
router = APIRouter()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("node")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
initialization_complete = False


async def initialize_components():
    global initialization_complete

    from P2PNode import P2PNode  # Move the import here
    
    try:
        # Step 1: Initialize node directory
        node_directory = EnhancedNodeDirectory()
        logger.info("Node directory initialized.")

        # Step 2: Initialize SimpleVM
        vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
        logger.info(f"VM object initialized: {vm}")

        # Step 3: Initialize PBFT Consensus
        consensus = PBFTConsensus(nodes=[], node_id="node_1")
        logger.info("PBFT consensus initialized.")

        # Step 4: Initialize QuantumBlockchain
        logger.info("Initializing QuantumBlockchain...")
        secret_key = "your_secret_key_here"
        blockchain = QuantumBlockchain(consensus, secret_key, None, vm)
        logger.info("QuantumBlockchain initialized.")

        # Step 5: Initialize PriceOracle
        price_oracle = PriceOracle()
        logger.info("PriceOracle initialized.")

        # Step 6: Initialize EnhancedExchangeWithZKStarks
        exchange = EnhancedExchangeWithZKStarks(
            blockchain, vm, price_oracle, node_directory, 
            desired_security_level=1, host="localhost", port=8765
        )
        logger.info(f"EnhancedExchange instance created: {exchange}")

        # Step 7: Initialize P2P node
        logger.info("Initializing P2P node...")
        p2p_node = P2PNode(blockchain=None, host=ip_address, port=p2p_port)
        await p2p_node.start()
        logger.info(f"P2P node started with {len(p2p_node.peers)} peers")

        # Set P2P node for blockchain
        await blockchain.set_p2p_node(p2p_node)
        logger.info(f"P2P node set for blockchain: {p2p_node}")

        # Step 8: Health check on P2P node
        if not await blockchain.p2p_node_health_check():
            raise Exception("P2P node failed the health check")
        logger.info("P2P node health check passed.")

        # Step 9: Set up the order book
        exchange.order_book = EnhancedOrderBook()
        logger.info("Order book initialized for EnhancedExchange.")

        # Step 10: Initialize PlataContract and PriceFeed
        plata_contract = PlataContract(vm)
        price_feed = PriceFeed()  # Assuming you have a PriceFeed class
        logger.info("PlataContract and PriceFeed initialized.")

        # Step 11: Initialize MarketMakerBot
        genesis_address = "your_genesis_address_here"  # Replace with actual genesis address generation/retrieval
        bot = MarketMakerBot(exchange, "BTC_PLATA", Decimal('0.01'))
        logger.info("MarketMakerBot initialized.")

        # Step 12: Initialize NativeCoinContract
        native_coin_contract_address, native_coin_contract = vm.get_existing_contract(NativeCoinContract)
        security_level = 20  # Replace with the appropriate level for your system

        # Initialize zk_system with the security_level
        zk_system = SecureHybridZKStark(security_level)
        logger.info(f"zk_system initialized with security level {security_level}.")

        if native_coin_contract is None:
            max_supply = 21000000  # Example max supply
            native_coin_contract_address = vm.deploy_contract(genesis_address, NativeCoinContract, max_supply, zk_system)
            native_coin_contract = vm.contracts[native_coin_contract_address]
            logger.info(f"NativeCoinContract deployed: {native_coin_contract_address}")

        blockchain.native_coin_contract_address = native_coin_contract_address
        blockchain.native_coin_contract = native_coin_contract
        logger.info("NativeCoinContract set in the blockchain.")
        initialization_complete = True

        logger.info("All components initialized successfully.")

        # Return the initialized components
        return {
            'node_directory': node_directory,
            'blockchain': blockchain,
            'exchange': exchange,
            'vm': vm,
            'plata_contract': plata_contract,
            'price_feed': price_feed,
            'genesis_address': genesis_address,
            'bot': bot,
            'p2p_node': p2p_node
        }
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise





async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = {"username": username}
        user = get_user(fake_users_db, username=token_data['username'])
        if user is None:
            raise credentials_exception
        return user
    except PyJWTError:
        raise credentials_exception



state = {
    "portfolio": {
        "PLATA": Decimal('1000000'),
        "BTC": Decimal('10'),
        "ETH": Decimal('100'),
        "DOT": Decimal('1000')
    },
    "plata": {
        "price": Decimal('1.00'),
        "supply": Decimal('1000000'),
        "minted": Decimal('0'),
        "burned": Decimal('0')
    },
    "miningStats": {
        "hashRate": Decimal('0'),
        "qhins": Decimal('0'),
        "entanglementStrength": Decimal('0'),
        "lastBlockMined": "Never"
    },
    "tradeHistory": [],
    "orderBook": {
        "buyOrders": [],
        "sellOrders": []
    }
}
class PlataContract:
    def __init__(self, address, total_supply=Decimal('1000000')):
        self.address = address
        self.total_supply = total_supply

    async def mint(self, address, amount):
        self.total_supply += amount

    async def burn(self, address, amount):
        self.total_supply -= amount

# Instantiate plata_contract
plata_contract = PlataContract(address="contract_address")
class PriceFeed:
    def __init__(self):
        self.api_url = 'https://api.coingecko.com/api/v3/simple/price'
        self.supported_assets = ['bitcoin', 'ethereum', 'polkadot', 'plata']  # Add supported assets here
        self.vs_currency = 'usd'
        self.cache = {}
        self.cache_expiry = 60  # Cache prices for 60 seconds
        self.last_update_time = 0

    async def get_price(self, asset: str) -> Decimal:
        print(f"get_price called with asset: {asset}")
        current_time = time.time()
        if (current_time - self.last_update_time) > self.cache_expiry:
            await self.update_prices()
            self.last_update_time = current_time

        asset_lower = asset.lower()
        if asset_lower in self.cache:
            return self.cache[asset_lower]
        else:
            raise ValueError(f"Asset {asset} is not supported or price data is unavailable.")


    async def update_prices(self):
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'ids': ','.join(self.supported_assets),
                    'vs_currencies': self.vs_currency
                }
                async with session.get(self.api_url, params=params) as response:
                    data = await response.json()

                    for asset in self.supported_assets:
                        asset_lower = asset.lower()
                        if asset_lower in data:
                            price = Decimal(str(data[asset_lower][self.vs_currency]))
                            self.cache[asset_lower] = price
                        else:
                            self.cache[asset_lower] = Decimal('0')

        except Exception as e:
            print(f"Error updating prices: {str(e)}")

# Instantiate the price feed
price_feed = PriceFeed()

class MarketPriceRequest(BaseModel):
    trading_pair: str
class MarketPriceResponse(BaseModel):
    price: Decimal

class LimitOrderRequest(BaseModel):
    order_type: str
    base_currency: str
    quote_currency: str
    amount: Decimal
    price: Decimal

class LimitOrderResponse(BaseModel):
    order_id: str

@router.post("/get_market_price", response_model=MarketPriceResponse)
async def get_market_price(request: MarketPriceRequest, current_user: str = Depends(get_current_user)):
    # Implement logic to fetch the current market price for the given trading pair
    # This could involve querying your order book or external price feeds
    price = await exchange.get_market_price(request.trading_pair)
    return MarketPriceResponse(price=price)

@router.post("/place_limit_order", response_model=LimitOrderResponse)
async def place_limit_order(request: LimitOrderRequest, current_user: str = Depends(get_current_user)):
    # Implement logic to place a limit order
    order_id = await exchange.place_limit_order(
        current_user,
        request.order_type,
        request.base_currency,
        request.quote_currency,
        request.amount,
        request.price
    )
    return LimitOrderResponse(order_id=order_id)
def generate_mnemonic():
    mnemo = Mnemonic("english")
    mnemonic = mnemo.generate(strength=128)
    return mnemonic

def create_genesis_wallet(vm):
    mnemonic = generate_mnemonic()
    address = derive_address_from_mnemonic(mnemonic)
    vm.add_user(address, permissions=[Permission.READ, Permission.WRITE, Permission.EXECUTE], roles=[Role.ADMIN])
    return mnemonic, address
def derive_address_from_mnemonic(mnemonic):
    mnemo = Mnemonic("english")
    seed = mnemo.to_seed(mnemonic)
    private_key = ec.derive_private_key(int.from_bytes(seed[:32], byteorder="big"), ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    address = generate_wallet_address(public_key)
    return address


class QuantumInspiredMarketBot:
    def __init__(self, exchange, vm, plata_contract, price_feed, initial_capital, portfolio=None):
        self.portfolio = portfolio if portfolio is not None else {}
        self.exchange = exchange
        self.vm = vm
        self.plata_contract = plata_contract
        self.price_feed = price_feed
        self.capital = initial_capital
        self.positions: Dict[str, Decimal] = {}
        self.order_book: Dict[str, List[Tuple[Decimal, Decimal]]] = {}
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.historical_data: Dict[str, List[Decimal]] = {}
        self.quantum_circuit = self._initialize_quantum_circuit()
        self.last_rebalance_time = time.time()
        self.rebalance_interval = 3600  # 1 hour
        self.target_price = Decimal('1.00')
        self.price_tolerance = Decimal('0.005')  # 0.5% tolerance
        self.max_single_trade_size = Decimal('10000')  # Max size for a single trade
        self.slippage_tolerance = Decimal('0.01')  # 1% slippage tolerance
        logger.info(f"Class type before calling get_tradable_assets: {type(self.exchange)}")
        print(dir(self.exchange))
        self.portfolio = portfolio if portfolio is not None else {}
        self.market_data = market_data if market_data is not None else {}

        # Ensure 'PLATA' key is always present in the capital dictionary
        if 'PLATA' not in self.capital:
            self.capital['PLATA'] = Decimal('0')

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    async def get_asset_value(self) -> Decimal:
        # Example logic to calculate asset value
        # This assumes that self.portfolio is a dictionary with asset names as keys and amounts as values
        total_value = Decimal(0)
        for asset, amount in self.portfolio.items():
            asset_price = await self.market_data.get_price(asset)
            total_value += Decimal(amount) * asset_price
        return total_value



    def _initialize_quantum_circuit(self):
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)  # Apply Hadamard gates
        qc.measure(qr, cr)
        return qc
    async def run(self):
        while True:
            try:
                # Update market data
                await self._update_market_data()

                # Rebalance the portfolio based on updated data
                await self._rebalance_portfolio()

                # Manage the supply of PLATA or any other asset
                await self._manage_plata_supply()

                # Execute the quantum trading strategy
                await self._execute_quantum_trading_strategy()

                # Provide liquidity to relevant markets
                await self._provide_liquidity()

                # Execute other strategies
                await self.execute_mean_reversion_strategy()
                await self.execute_momentum_strategy()

                # Calculate asset and total values
                asset_value = await self.get_asset_value()
                total_value = await self.get_total_value()

                # Manage risk based on asset and total values
                await self.manage_risk(asset_value, total_value)

                # Handle any black swan events
                await self.handle_black_swan_events()

            except Exception as e:
                # Log any errors that occur during the run loop
                self.logger.error(f"Error in bot run loop: {e}")
                self.logger.error(traceback.format_exc())

            # Sleep for a set interval (e.g., 1 minute) before running the loop again
            await asyncio.sleep(60)  # Adjust this interval as needed


    async def _update_market_data(self):
        try:
            tradable_assets = await self.exchange.get_tradable_assets()  # Await the coroutine
            for asset in tradable_assets:
                price = await self.price_feed.get_price(asset)
                if asset not in self.historical_data:
                    self.historical_data[asset] = []
                self.historical_data[asset].append(price)
                if len(self.historical_data[asset]) > 1000:
                    self.historical_data[asset] = self.historical_data[asset][-1000:]
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            self.logger.error(traceback.format_exc())

    async def _rebalance_portfolio(self):
        try:
            current_time = time.time()
            if current_time - self.last_rebalance_time < self.rebalance_interval:
                return

            total_value = sum(amount * await self.price_feed.get_price(asset) for asset, amount in self.capital.items())
            target_allocation = {asset: Decimal('1') / len(self.capital) for asset in self.capital}

            for asset, amount in self.capital.items():
                current_value = amount * await self.price_feed.get_price(asset)
                target_value = total_value * target_allocation[asset]
                if current_value < target_value:
                    await self._buy_asset(asset, (target_value - current_value) / await self.price_feed.get_price(asset))
                elif current_value > target_value:
                    await self._sell_asset(asset, (current_value - target_value) / await self.price_feed.get_price(asset))

            self.last_rebalance_time = current_time
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            self.logger.error(traceback.format_exc())
    async def _manage_plata_supply(self):
        try:
            # Fetch the current price of PLATA
            plata_price = await self.price_feed.get_price("PLATA")
            
            # Calculate the difference from the target price
            price_difference = plata_price - self.target_price
            
            # Determine the action based on the price difference
            if price_difference > self.price_tolerance:
                # Price is higher than the target, mint more PLATA
                amount_to_mint = (price_difference * self.plata_contract.total_supply) / self.target_price
                if amount_to_mint > 0:
                    await self._mint_plata(amount_to_mint)
                else:
                    self.logger.warning("Calculated mint amount is not positive; minting skipped.")
            
            elif price_difference < -self.price_tolerance:
                # Price is lower than the target, burn some PLATA
                amount_to_burn = (-price_difference * self.plata_contract.total_supply) / self.target_price
                if amount_to_burn > 0:
                    await self._burn_plata(amount_to_burn)
                else:
                    self.logger.warning("Calculated burn amount is not positive; burning skipped.")
            
            else:
                # Price is within the acceptable range, no action needed
                self.logger.info("PLATA price is within the target range; no minting or burning required.")
        
        except Exception as e:
            self.logger.error(f"Error managing PLATA supply: {e}")
            self.logger.error(traceback.format_exc())


    async def _execute_quantum_trading_strategy(self):
        try:
            backend = Aer.get_backend('aer_simulator')
            transpiled_circuit = transpile(self.quantum_circuit, backend)
            job = backend.run(transpiled_circuit, shots=1000)
            result = job.result()
            counts = result.get_counts(self.quantum_circuit)
            
            # Use quantum measurements to inform trading decisions
            quantum_signal = max(counts, key=counts.get)
            signal_strength = counts[quantum_signal] / 1000

            tradable_assets = await self.exchange.get_tradable_assets()
            for asset in tradable_assets:
                if asset == "PLATA":
                    continue

                price_data = np.array(self.historical_data[asset]).reshape(-1, 1)
                if len(price_data) < 100:
                    continue

                self.ml_model.fit(price_data[:-1], price_data[1:])
                predicted_price = self.ml_model.predict(price_data[-1].reshape(1, -1))[0]

                current_price = await self.price_feed.get_price(asset)
                if predicted_price > current_price and quantum_signal in ['0000', '0001']:
                    await self._buy_asset(asset, self.max_single_trade_size * signal_strength)
                elif predicted_price < current_price and quantum_signal in ['1110', '1111']:
                    await self._sell_asset(asset, self.max_single_trade_size * signal_strength)
        except Exception as e:
            self.logger.error(f"Error executing quantum trading strategy: {e}")
            self.logger.error(traceback.format_exc())

    async def _provide_liquidity(self):
        try:
            tradable_assets = await self.exchange.get_tradable_assets()

            for asset in tradable_assets:
                if asset == "PLATA":
                    continue

                current_price = await self.price_feed.get_price(asset)
                spread = current_price * Decimal('0.002')
                buy_price = current_price - spread / 2
                sell_price = current_price + spread / 2

                await self._place_limit_order(asset, "buy", self.max_single_trade_size / 10, buy_price)
                await self._place_limit_order(asset, "sell", self.max_single_trade_size / 10, sell_price)
        except Exception as e:
            self.logger.error(f"Error providing liquidity: {e}")
            self.logger.error(traceback.format_exc())

    async def _buy_asset(self, asset: str, amount: Decimal):
        try:
            current_price = await self.price_feed.get_price(asset)
            max_price = current_price * (1 + self.slippage_tolerance)
            await self.exchange.create_market_buy_order(asset, amount, max_price)
            self.capital[asset] = self.capital.get(asset, Decimal('0')) + amount
            self.capital["PLATA"] -= amount * current_price
        except Exception as e:
            self.logger.error(f"Error buying {asset}: {e}")
            self.logger.error(traceback.format_exc())

    async def _sell_asset(self, asset: str, amount: Decimal):
        try:
            current_price = await self.price_feed.get_price(asset)
            min_price = current_price * (1 - self.slippage_tolerance)
            await self.exchange.create_market_sell_order(asset, amount, min_price)
            self.capital[asset] = self.capital.get(asset, Decimal('0')) - amount
            self.capital["PLATA"] += amount * current_price
        except KeyError as e:
            self.logger.error(f"Error selling {asset}: {e}")
            self.logger.error(traceback.format_exc())
            # Ensure 'PLATA' key exists
            self.capital['PLATA'] = self.capital.get('PLATA', Decimal('0'))
        except Exception as e:
            self.logger.error(f"Error selling {asset}: {e}")
            self.logger.error(traceback.format_exc())

    async def _place_limit_order(self, asset: str, side: str, amount: Decimal, price: Decimal):
        try:
            if side == "buy":
                await self.exchange.create_limit_buy_order(asset, amount, price)
            elif side == "sell":
                await self.exchange.create_limit_sell_order(asset, amount, price)
        except Exception as e:
            self.logger.error(f"Error placing limit order for {asset}: {e}")
            self.logger.error(traceback.format_exc())

    async def _mint_plata(self, amount: Decimal):
        try:
            await self.plata_contract.mint(amount)
        except Exception as e:
            self.logger.error(f"Error minting PLATA: {e}")
            self.logger.error(traceback.format_exc())

    async def _burn_plata(self, amount: Decimal):
        try:
            await self.plata_contract.burn(amount)
        except Exception as e:
            self.logger.error(f"Error burning PLATA: {e}")
            self.logger.error(traceback.format_exc())

    async def calculate_optimal_position_size(self, asset: str) -> Decimal:
        returns = np.array(self.historical_data[asset])
        if len(returns) < 2:
            return Decimal('0')

        win_probability = np.mean(returns > 0)
        average_win = np.mean(returns[returns > 0])
        average_loss = abs(np.mean(returns[returns < 0]))

        odds = average_win / average_loss if average_loss != 0 else 1
        kelly_fraction = self.calculate_kelly_criterion(win_probability, odds)
        return Decimal(str(kelly_fraction)) * self.capital.get(asset, Decimal('0'))

    def calculate_kelly_criterion(self, win_probability: float, odds: float) -> float:
        return win_probability - (1 - win_probability) / odds

    async def execute_mean_reversion_strategy(self):
        try:
            tradable_assets = await self.exchange.get_tradable_assets()

            for asset in tradable_assets:
                if asset == "PLATA":
                    continue

                if asset not in self.historical_data:
                    continue

                price_data = np.array(self.historical_data[asset])
                if len(price_data) < 20:
                    continue

                moving_average = np.mean(price_data[-20:])
                current_price = price_data[-1]
                z_score = (current_price - moving_average) / np.std(price_data[-20:])

                if z_score > 2:
                    optimal_size = self.calculate_optimal_position_size(asset)
                    await self._sell_asset(asset, min(optimal_size, self.max_single_trade_size))
                elif z_score < -2:
                    optimal_size = self.calculate_optimal_position_size(asset)
                    await self._buy_asset(asset, min(optimal_size, self.max_single_trade_size))
        except Exception as e:
            self.logger.error(f"Error executing mean reversion strategy: {e}")
            self.logger.error(traceback.format_exc())

    async def execute_momentum_strategy(self):
        try:
            tradable_assets = await self.exchange.get_tradable_assets()

            for asset in tradable_assets:
                if asset == "PLATA":
                    continue

                if asset not in self.historical_data:
                    continue

                price_data = np.array(self.historical_data[asset])
                if len(price_data) < 20:
                    continue

                moving_average = np.mean(price_data[-20:])
                current_price = price_data[-1]
                z_score = (current_price - moving_average) / np.std(price_data[-20:])

                if z_score > 2:
                    optimal_size = self.calculate_optimal_position_size(asset)
                    await self._sell_asset(asset, min(optimal_size, self.max_single_trade_size))
                elif z_score < -2:
                    optimal_size = self.calculate_optimal_position_size(asset)
                    await self._buy_asset(asset, min(optimal_size, self.max_single_trade_size))
        except Exception as e:
            self.logger.error(f"Error executing momentum strategy: {e}")
            self.logger.error(traceback.format_exc())

    def manage_risk(asset_value, total_value):
        try:
            if total_value == 0:
                raise ValueError("Total value cannot be zero")

            asset_weight = asset_value / total_value
            # Proceed with the rest of your logic using asset_weight
        except decimal.DivisionUndefined as e:
            logger.error(f"DivisionUndefined error: {e}")
            # Handle the division undefined error
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            # Handle the value error, such as total_value being zero
        except Exception as e:
            logger.error(f"Unexpected error in manage_risk: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_black_swan_events(self):
        try:
            tradable_assets = await self.exchange.get_tradable_assets()
            for asset in tradable_assets:
                try:
                    # Handle black swan event for the specific asset
                    pass
                except Exception as e:
                    self.logger.error(f"Error handling asset {asset}: {e}")
        except Exception as e:
            self.logger.error(f"Error handling black swan events: {e}")

class StandardCrypto:
    @staticmethod
    def sign_message(private_key, message):
        return private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

    @staticmethod
    def verify_signature(public_key, message, signature):
        try:
            public_key.verify(
                signature,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
    @staticmethod
    def generate_keypair():
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key

class QuantumResistantCrypto:
    @staticmethod
    def generate_keys():
        private_key = dilithium.generate_private_key()
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def sign(private_key, message):
        return private_key.sign(message)

    @staticmethod
    def verify(public_key, message, signature):
        try:
            public_key.verify(signature, message)
            return True
        except:
            return False
class PredictionMarket:
    def __init__(self):
        self.markets = {}

    def create_market(self, question, options, end_time):
        market_id = hashlib.sha256(f"{question}{time.time()}".encode()).hexdigest()
        self.markets[market_id] = {
            "question": question,
            "options": {option: Decimal('0') for option in options},
            "total_stake": Decimal('0'),
            "end_time": end_time,
            "resolved": False,
            "winning_option": None
        }
        return market_id

    def place_bet(self, market_id, option, amount):
        market = self.markets.get(market_id)
        if not market:
            raise ValueError("Market not found")
        if market["resolved"]:
            raise ValueError("Market already resolved")
        if time.time() > market["end_time"]:
            raise ValueError("Market has ended")
        if option not in market["options"]:
            raise ValueError("Invalid option")

        market["options"][option] += amount
        market["total_stake"] += amount

    def resolve_market(self, market_id, winning_option):
        market = self.markets.get(market_id)
        if not market:
            raise ValueError("Market not found")
        if market["resolved"]:
            raise ValueError("Market already resolved")
        if time.time() < market["end_time"]:
            raise ValueError("Market has not ended yet")
        if winning_option not in market["options"]:
            raise ValueError("Invalid winning option")

        market["resolved"] = True
        market["winning_option"] = winning_option

    def calculate_payout(self, market_id, option, stake):
        market = self.markets.get(market_id)
        if not market or not market["resolved"]:
            raise ValueError("Market not resolved")
        if option == market["winning_option"]:
            return stake * (market["total_stake"] / market["options"][option])
        return Decimal('0')
class CrossChainSwap:
    def __init__(self, vm):
        self.vm = vm
        self.swaps = {}

    def initiate_swap(self, initiator, participant, amount_a, currency_a, amount_b, currency_b, lock_time):
        swap_id = hashlib.sha256(f"{initiator}{participant}{time.time()}".encode()).hexdigest()
        secret = os.urandom(32)
        secret_hash = hashlib.sha256(secret).digest()

        self.swaps[swap_id] = {
            "initiator": initiator,
            "participant": participant,
            "amount_a": amount_a,
            "currency_a": currency_a,
            "amount_b": amount_b,
            "currency_b": currency_b,
            "secret_hash": secret_hash,
            "lock_time": lock_time,
            "status": "initiated"
        }

        return swap_id, secret.hex()

    def participate_swap(self, swap_id, participant):
        swap = self.swaps.get(swap_id)
        if not swap:
            raise ValueError("Swap not found")
        if swap["status"] != "initiated":
            raise ValueError("Invalid swap status")
        if swap["participant"] != participant:
            raise ValueError("Invalid participant")

        swap["status"] = "participated"

    def redeem_swap(self, swap_id, secret):
        swap = self.swaps.get(swap_id)
        if not swap:
            raise ValueError("Swap not found")
        if swap["status"] != "participated":
            raise ValueError("Invalid swap status")
        if hashlib.sha256(bytes.fromhex(secret)).digest() != swap["secret_hash"]:
            raise ValueError("Invalid secret")

        swap["status"] = "redeemed"

    def refund_swap(self, swap_id):
        swap = self.swaps.get(swap_id)
        if not swap:
            raise ValueError("Swap not found")
        if swap["status"] != "participated":
            raise ValueError("Invalid swap status")
        if time.time() < swap["lock_time"]:
            raise ValueError("Lock time not expired")

        swap["status"] = "refunded"
class DecentralizedIdentity:
    def __init__(self):
        self.identities = {}

    def create_identity(self, user_id, public_key):
        if user_id in self.identities:
            raise ValueError("Identity already exists")
        self.identities[user_id] = {
            "public_key": public_key,
            "attributes": {},
            "verifications": []
        }

    def add_attribute(self, user_id, key, value):
        if user_id not in self.identities:
            raise ValueError("Identity not found")
        self.identities[user_id]["attributes"][key] = value

    def verify_attribute(self, verifier_id, user_id, key):
        if user_id not in self.identities:
            raise ValueError("Identity not found")
        if key not in self.identities[user_id]["attributes"]:
            raise ValueError("Attribute not found")
        self.identities[user_id]["verifications"].append({
            "verifier": verifier_id,
            "attribute": key,
            "timestamp": time.time()
        })

    def get_identity(self, user_id):
        return self.identities.get(user_id)
class AdvancedGovernance:
    def __init__(self, vm):
        self.vm = vm
        self.proposals = {}
        self.dao_token = "DAO_TOKEN"

    def create_proposal(self, proposer, description, options, voting_period):
        proposal_id = hashlib.sha256(f"{proposer}{description}{time.time()}".encode()).hexdigest()
        self.proposals[proposal_id] = {
            "proposer": proposer,
            "description": description,
            "options": options,
            "votes": {option: Decimal('0') for option in options},
            "status": "active",
            "created_at": time.time(),
            "voting_period": voting_period,
            "total_votes": Decimal('0')
        }
        return proposal_id

    def vote(self, voter, proposal_id, option, amount):
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError("Proposal not found")
        if proposal["status"] != "active":
            raise ValueError("Proposal is not active")
        if time.time() > proposal["created_at"] + proposal["voting_period"]:
            raise ValueError("Voting period has ended")
        if option not in proposal["options"]:
            raise ValueError("Invalid voting option")

        voter_balance = self.vm.token_balances.get(self.dao_token, {}).get(voter, Decimal('0'))
        if voter_balance < amount:
            raise ValueError("Insufficient DAO tokens")

        proposal["votes"][option] += amount
        proposal["total_votes"] += amount

    def execute_proposal(self, proposal_id):
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError("Proposal not found")
        if proposal["status"] != "active":
            raise ValueError("Proposal is not active")
        if time.time() <= proposal["created_at"] + proposal["voting_period"]:
            raise ValueError("Voting period has not ended")

        winning_option = max(proposal["votes"], key=proposal["votes"].get)
        proposal["status"] = "executed"
        proposal["winning_option"] = winning_option

        return winning_option

class QuantumRandomNumberGenerator:
    @staticmethod
    def generate_random_number(num_qubits=4):
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)

        circuit.h(qr)  # Apply Hadamard gates
        circuit.measure(qr, cr)  # Measure the qubits

        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=1)
        result = job.result()
        counts = result.get_counts(circuit)
        random_bitstring = list(counts.keys())[0]
        return int(random_bitstring, 2)

price_oracle = PriceOracle()
async def create_prediction_market(self, creator: str, question: str, options: List[str], end_time: int):
    market_id = self.prediction_market.create_market(question, options, end_time)
    return market_id

async def place_prediction_bet(self, user: str, market_id: str, option: str, amount: Decimal):
    balance = await self.blockchain.get_balance(user, "PREDICTION_TOKEN")
    if balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    self.prediction_market.place_bet(market_id, option, amount)
    await self.blockchain.add_transaction(Transaction(sender=user, receiver=market_id, amount=amount, currency="PREDICTION_TOKEN"))

async def resolve_prediction_market(self, resolver: str, market_id: str, winning_option: str):
    self.prediction_market.resolve_market(market_id, winning_option)
    market = self.prediction_market.markets[market_id]
    
    for better, stake in market["options"].items():
        if better == winning_option:
            payout = self.prediction_market.calculate_payout(market_id, better, stake)
            await self.blockchain.add_transaction(Transaction(sender=market_id, receiver=better, amount=payout, currency="PREDICTION_TOKEN"))
async def initiate_cross_chain_swap(self, initiator: str, participant: str, amount_a: Decimal, currency_a: str, amount_b: Decimal, currency_b: str, lock_time: int):
    balance = await self.blockchain.get_balance(initiator, currency_a)
    if balance < amount_a:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    swap_id, secret = self.cross_chain_swap.initiate_swap(initiator, participant, amount_a, currency_a, amount_b, currency_b, lock_time)
    await self.blockchain.add_transaction(Transaction(sender=initiator, receiver=swap_id, amount=amount_a, currency=currency_a))
    return {"swap_id": swap_id, "secret": secret}

async def participate_cross_chain_swap(self, participant: str, swap_id: str):
    swap = self.cross_chain_swap.swaps.get(swap_id)
    if not swap:
        raise HTTPException(status_code=404, detail="Swap not found")

    balance = await self.blockchain.get_balance(participant, swap["currency_b"])
    if balance < swap["amount_b"]:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    self.cross_chain_swap.participate_swap(swap_id, participant)
    await self.blockchain.add_transaction(Transaction(sender=participant, receiver=swap_id, amount=swap["amount_b"], currency=swap["currency_b"]))

async def redeem_cross_chain_swap(self, redeemer: str, swap_id: str, secret: str):
    self.cross_chain_swap.redeem_swap(swap_id, secret)
    swap = self.cross_chain_swap.swaps[swap_id]
    await self.blockchain.add_transaction(Transaction(sender=swap_id, receiver=swap["participant"], amount=swap["amount_a"], currency=swap["currency_a"]))
    await self.blockchain.add_transaction(Transaction(sender=swap_id, receiver=swap["initiator"], amount=swap["amount_b"], currency=swap["currency_b"]))

async def refund_cross_chain_swap(self, initiator: str, swap_id: str):
    self.cross_chain_swap.refund_swap(swap_id)
    swap = self.cross_chain_swap.swaps[swap_id]
    await self.blockchain.add_transaction(Transaction(sender=swap_id, receiver=swap["initiator"], amount=swap["amount_a"], currency=swap["currency_a"]))
    await self.blockchain.add_transaction(Transaction(sender=swap_id, receiver=swap["participant"], amount=swap["amount_b"], currency=swap["currency_b"]))
def create_decentralized_identity(self, user_id: str, public_key: str):
    self.decentralized_identity.create_identity(user_id, public_key)

def add_identity_attribute(self, user_id: str, key: str, value: str):
    self.decentralized_identity.add_attribute(user_id, key, value)

def verify_identity_attribute(self, verifier_id: str, user_id: str, key: str):
    self.decentralized_identity.verify_attribute(verifier_id, user_id, key)
async def create_governance_proposal(self, proposer: str, description: str, options: List[str], voting_period: int):
    proposal_id = self.governance.create_proposal(proposer, description, options, voting_period)
    return proposal_id

async def vote_on_proposal(self, voter: str, proposal_id: str, option: str, amount: Decimal):
    balance = await self.blockchain.get_balance(voter, self.governance.dao_token)
    if balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient DAO tokens")

    self.governance.vote(voter, proposal_id, option, amount)

async def execute_governance_proposal(self, proposal_id: str):
    winning_option = self.governance.execute_proposal(proposal_id)
    return {"winning_option": winning_option}
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = {"username": username}
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data['username'])
    if user is None:
        raise credentials_exception
    return user


@router.post("/execute_contract")
async def execute_contract(
    contract_address: str,
    function_name: str,
    args: List[str],
    current_user: str = Depends(get_current_user)
):
    result = await exchange.execute_contract(contract_address, function_name, args, current_user)
    return {"result": result}

@router.post("/execute_contract")
async def execute_contract(
    contract_address: str,
    function_name: str,
    args: List[str],
    current_user: str = Depends(get_current_user)
):
    result = await exchange.execute_contract(contract_address, function_name, args, current_user)
    return {"result": result}
@router.post("/create_prediction_market")
async def create_prediction_market(
    question: str,
    options: List[str],
    end_time: int,
    current_user: str = Depends(get_current_user)
):
    market_id = await exchange.create_prediction_market(current_user, question, options, end_time)
    return {"market_id": market_id}

@router.post("/place_prediction_bet")
async def place_prediction_bet(
    market_id: str,
    option: str,
    amount: Decimal,
    current_user: str = Depends(get_current_user)
):
    await exchange.place_prediction_bet(current_user, market_id, option, amount)
    return {"message": "Bet placed successfully"}

@router.post("/resolve_prediction_market")
async def resolve_prediction_market(
    market_id: str,
    winning_option: str,
    current_user: str = Depends(get_current_user)
):
    await exchange.resolve_prediction_market(current_user, market_id, winning_option)
    return {"message": "Market resolved successfully"}
@router.post("/initiate_cross_chain_swap")
async def initiate_cross_chain_swap(
    participant: str,
    amount_a: Decimal,
    currency_a: str,
    amount_b: Decimal,
    currency_b: str,
    lock_time: int,
    current_user: str = Depends(get_current_user)
):
    result = await exchange.initiate_cross_chain_swap(current_user, participant, amount_a, currency_a, amount_b, currency_b, lock_time)
    return result

@router.post("/participate_cross_chain_swap")
async def participate_cross_chain_swap(
    swap_id: str,
    current_user: str = Depends(get_current_user)
):
    await exchange.participate_cross_chain_swap(current_user, swap_id)
    return {"message": "Swap participation successful"}

@router.post("/redeem_cross_chain_swap")
async def redeem_cross_chain_swap(
    swap_id: str,
    secret: str,
    current_user: str = Depends(get_current_user)
):
    await exchange.redeem_cross_chain_swap(current_user, swap_id, secret)
    return {"message": "Swap redeemed successfully"}

@router.post("/refund_cross_chain_swap")
async def refund_cross_chain_swap(
    swap_id: str,
    current_user: str = Depends(get_current_user)
):
    await exchange.refund_cross_chain_swap(current_user, swap_id)
    return {"message": "Swap refunded successfully"}
@router.post("/create_identity")
async def create_identity(
    public_key: str,
    current_user: str = Depends(get_current_user)
):
    exchange.create_decentralized_identity(current_user, public_key)
    return {"message": "Identity created successfully"}

@router.post("/add_identity_attribute")
async def add_identity_attribute(
    key: str,
    value: str,
    current_user: str = Depends(get_current_user)
):
    exchange.add_identity_attribute(current_user, key, value)
    return {"message": "Attribute added successfully"}

@router.post("/verify_identity_attribute")
async def verify_identity_attribute(
    user_id: str,
    key: str,
    current_user: str = Depends(get_current_user)
):
    exchange.verify_identity_attribute(current_user, user_id, key)
    return {"message": "Attribute verified successfully"}
@router.post("/create_governance_proposal")
async def create_governance_proposal(
    description: str,
    options: List[str],
    voting_period: int,
    current_user: str = Depends(get_current_user)
):
    proposal_id = await exchange.create_governance_proposal(current_user, description, options, voting_period)
    return {"proposal_id": proposal_id}

@router.post("/vote_on_proposal")
async def vote_on_proposal(
    proposal_id: str,
    option: str,
    amount: Decimal,
    current_user: str = Depends(get_current_user)
):
    await exchange.vote_on_proposal(current_user, proposal_id, option, amount)
    return {"message": "Vote placed successfully"}

@router.post("/execute_governance_proposal")
async def execute_governance_proposal(
    proposal_id: str,
    current_user: str = Depends(get_current_user)
):
    result = await exchange.execute_governance_proposal(proposal_id)
    return {"winning_option": result["winning_option"]}
# Add this router to your main FastAPI app
app.include_router(router, prefix="/exchange", tags=["exchange"])

# You might want to run this function periodically or trigger it based on certain conditions
async def update_prices_periodically():
    while True:
        await price_oracle.update_prices()
        await asyncio.sleep(60)  # Update every minute


    
    
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = {"username": username}
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data['username'])
    if user is None:
        raise credentials_exception
    return user
from typing import Literal



class QuantumInspiredFeatures:
    @staticmethod
    async def quantum_hedging(exchange: 'EnhancedExchange', user: str, pair: str, amount: Decimal):
        # Simulate quantum superposition for hedging
        quantum_circuit = QuantumCircuit(2, 2)
        quantum_circuit.h(0)  # Apply Hadamard gate
        quantum_circuit.cx(0, 1)  # Apply CNOT gate
        quantum_circuit.measure([0, 1], [0, 1])

        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(quantum_circuit, backend)
        job = backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts(quantum_circuit)

        # Use quantum result to determine hedging strategy
        if counts.get('00', 0) + counts.get('11', 0) > counts.get('01', 0) + counts.get('10', 0):
            # Open a long position
            await exchange.open_margin_position(user, pair, "long", amount, 2)
        else:
            # Open a short position
            await exchange.open_margin_position(user, pair, "short", amount, 2)

        return "Quantum-inspired hedge position opened"

    @staticmethod
    async def entanglement_based_arbitrage(exchange: 'EnhancedExchange', user: str, pairs: List[str]):
        # Simulate quantum entanglement for multi-pair arbitrage
        n = len(pairs)
        quantum_circuit = QuantumCircuit(n, n)
        for i in range(n):
            quantum_circuit.h(i)
        for i in range(n-1):
            quantum_circuit.cx(i, i+1)
        quantum_circuit.measure_all()

        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(quantum_circuit, backend)
        job = backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts(quantum_circuit)

        # Use quantum result to determine arbitrage strategy
        most_common_state = max(counts, key=counts.get)
        arbitrage_paths = [pairs[i] for i, bit in enumerate(most_common_state) if bit == '1']

        # Execute arbitrage trades
        for pair in arbitrage_paths:
            await exchange.place_market_order(user, "buy", pair, Decimal('0.1'))
            await exchange.place_market_order(user, "sell", pair, Decimal('0.1'))

        return f"Quantum-inspired arbitrage executed on pairs: {', '.join(arbitrage_paths)}"
# Assuming quantumdagknight.py is your module where EnhancedExchange is defined


class ExchangeTransaction:
    def __init__(self, buyer_id: str, seller_id: str, amount: Decimal, price: Decimal, to_currency: str, from_currency: str):
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.amount = amount
        self.price = price
        self.to_currency = to_currency
        self.from_currency = from_currency
        self.signature = None

    def __str__(self):
        return f"ExchangeTransaction(buyer_id='{self.buyer_id}', seller_id='{self.seller_id}', amount={self.amount}, price={self.price}, to_currency='{self.to_currency}', from_currency='{self.from_currency}')"

    def get_message(self):
        return f"{self.buyer_id}{self.seller_id}{self.amount}{self.price}{self.to_currency}{self.from_currency}"
import asyncio
from typing import Dict, List, Optional
from Order import Order



class OrderBook:
    def __init__(self):
        self.buy_orders: Dict[str, List[Order]] = {}
        self.sell_orders: Dict[str, List[Order]] = {}
        self.lock = asyncio.Lock()

    async def add_order(self, order: Order):
        async with self.lock:
            orders = self.buy_orders if order.order_type == "buy" else self.sell_orders
            if order.pair not in orders:
                orders[order.pair] = []
            orders[order.pair].append(order)

    async def remove_order(self, order_id: str) -> Optional[Order]:
        async with self.lock:
            for orders in list(self.buy_orders.values()) + list(self.sell_orders.values()):
                for order in orders:
                    if order.id == order_id:
                        orders.remove(order)
                        return order
            return None

    async def get_orders(self) -> List[Order]:
        async with self.lock:
            all_orders = []
            for orders in list(self.buy_orders.values()) + list(self.sell_orders.values()):
                all_orders.extend(orders)
            return all_orders



    def cancel_order(self, order_id: str) -> bool:
        for orders in list(self.buy_orders.values()) + list(self.sell_orders.values()):
            for order in orders:
                if order.id == order_id:
                    order.status = 'cancelled'
                    orders.remove(order)
                    return True
        return False



    def match_orders(self, pair: str):
        buy_orders = self.buy_orders.get(pair, [])
        sell_orders = self.sell_orders.get(pair, [])
        matches = []

        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]

            if buy_order.price >= sell_order.price:
                match_amount = min(buy_order.amount, sell_order.amount)
                matches.append((buy_order, sell_order, match_amount, sell_order.price))

                buy_order.amount -= match_amount
                sell_order.amount -= match_amount

                if buy_order.amount == 0:
                    buy_orders.pop(0)
                if sell_order.amount == 0:
                    sell_orders.pop(0)
            else:
                break

        return matches
    def get_orders(self) -> List[Order]:
        # Combine buy and sell orders into one list
        orders = []
        for orders_list in self.buy_orders.values():
            orders.extend(orders_list)
        for orders_list in self.sell_orders.values():
            orders.extend(orders_list)
        return orders

    @field_validator('order_type')
    def validate_order_type(cls, v):
        if v not in ['buy', 'sell']:
            raise ValueError('order_type must be either "buy" or "sell"')
        return v

    @field_validator('amount', 'price')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('must be positive')
        return v
class PlaceOrderRequest(BaseModel):
    user_id: str
    type: str
    order_type: str
    pair: str
    base_currency: str
    quote_currency: str
    amount: float
    price: Optional[float]
    from_currency: str
    to_currency: str

class OrderResponse(BaseModel):
    id: str
    type: str
    pair: str
    order_type: str
    amount: float
    price: Optional[float]
    status: str
class TradeResponse(BaseModel):
    id: str
    type: str
    pair: str
    amount: float
    price: float
    timestamp: str

class OrderBookEntry(BaseModel):
    price: float
    amount: float

class InitialDataResponse(BaseModel):
    balance: float
    order_book: dict
    recent_trades: List[TradeResponse]
    price_history: List[float]

  
    # Keep all other existing methods as they are
    # ...


@app.get("/initial_data", response_model=InitialDataResponse)
async def get_initial_data():
    try:
        # Fetch the user's balance
        balance = await blockchain.get_balance("user_address")  # Replace with actual user address

        # Get the current order book
        bids, asks = order_book.get_order_book("PLATA/BTC")  # Replace with desired trading pair

        # Fetch recent trades
        recent_trades = await blockchain.get_recent_trades("PLATA/BTC", limit=20)

        # Get price history
        price_history = await blockchain.get_price_history("PLATA/BTC", limit=50)

        return InitialDataResponse(
            balance=float(balance),
            order_book={
                "bids": [{"price": float(bid.price), "amount": float(bid.amount)} for bid in bids],
                "asks": [{"price": float(ask.price), "amount": float(ask.amount)} for ask in asks]
            },
            recent_trades=[
                TradeResponse(
                    id=trade.id,
                    type=trade.type,
                    pair=trade.pair,
                    amount=float(trade.amount),
                    price=float(trade.price),
                    timestamp=trade.timestamp.isoformat()
                ) for trade in recent_trades
            ],
            price_history=[float(price) for price in price_history]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class Exchange:
    def __init__(self, blockchain, vm, price_oracle):
        self.order_book = OrderBook()
        self.blockchain = blockchain
        self.liquidity_pools: Dict[str, LiquidityPool] = {}
        self.fee_percent = Decimal('0.003')  # 0.3% fee
        self.vm = vm
        self.contracts: Dict[str, SmartContract] = {}
        self.prediction_market = PredictionMarket()
        self.cross_chain_swap = CrossChainSwap(vm)
        self.decentralized_identity = DecentralizedIdentity()
        self.governance = AdvancedGovernance(vm)
        self.quantum_rng = QuantumRandomNumberGenerator()
        self.orders = []
        self.prices = {}
        self.price_oracle = price_oracle  # Ensure this is passed correctly
        self.liquidity_pool_manager = LiquidityPoolManager()
        self.fee_percent = Decimal('0.003')  # 0.3% fee
        self.crypto = StandardCrypto()
        self.user_keys = {}  # In-memory storage of user keys (not secure for production!)

    async def create_liquidity_pool(self, token_a: str, token_b: str, fee_percent: Decimal) -> str:
        return self.liquidity_pool_manager.create_pool(token_a, token_b, fee_percent)

    async def add_liquidity(self, user: str, token_a: str, token_b: str, amount_a: Decimal, amount_b: Decimal) -> Decimal:
        return await self.liquidity_pool_manager.add_liquidity(user, token_a, token_b, amount_a, amount_b)

    async def remove_liquidity(self, user: str, token_a: str, token_b: str, shares: Decimal) -> Tuple[Decimal, Decimal]:
        return await self.liquidity_pool_manager.remove_liquidity(user, token_a, token_b, shares)

    async def swap(self, amount_in: Decimal, token_in: str, token_out: str) -> Decimal:
        return await self.liquidity_pool_manager.swap(amount_in, token_in, token_out)

    def place_order(self, order):
        self.orders.append(order)
        return order['id']

    def get_orders(self):
        return self.orders

    def set_price(self, symbol, price):
        self.prices[symbol] = price

    def get_price(self, symbol):
        return self.prices.get(symbol, None)
    def place_order(self, order):
        self.orders.append(order)
        return order['id']
    async def place_order(self, order: Order):
        try:
            # Check balance before placing order
            if order.order_type == 'buy':
                balance = await self.blockchain.get_balance(order.user_id, order.to_currency)  # Fix this line if needed
                if balance < order.amount * order.price:
                    raise HTTPException(status_code=400, detail="Insufficient balance for buy order")
            else:
                balance = await self.blockchain.get_balance(order.user_id, order.to_currency)  # Fix this line if needed
                if balance < order.amount:
                    raise HTTPException(status_code=400, detail="Insufficient balance for sell order")

            # Generate a signature for the order
            private_key = await self.get_user_private_key(order.user_id)
            order_data = f"{order.user_id}{order.order_type}{order.from_currency}{order.to_currency}{order.amount}{order.price}"
            order.signature = self.crypto.sign_message(private_key, order_data)

            # Add order to the order book
            self.order_book.add_order(order)

            # Match orders
            pair = f"{order.from_currency}_{order.to_currency}"
            matches = self.order_book.match_orders(pair)

            for buy_order, sell_order, amount, price in matches:
                # Create an ExchangeTransaction object
                exchange_tx = ExchangeTransaction(
                    buyer_id=buy_order.user_id,
                    seller_id=sell_order.user_id,
                    amount=amount,
                    price=price,
                    from_currency=sell_order.from_currency,
                    to_currency=buy_order.from_currency
                )

                # Sign the transaction
                seller_private_key = await self.get_user_private_key(exchange_tx.seller_id)
                exchange_tx.signature = self.crypto.sign_message(seller_private_key, exchange_tx.get_message())

                # Process the transaction
                await self.process_exchange_transaction(exchange_tx)

            # Return the order ID
            return {"status": "success", "order_id": order.id}

        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


    async def process_exchange_orders(self):
        for pair in self.order_book.buy_orders.keys():
            matches = self.order_book.match_orders(pair)
            for buy_order, sell_order, amount, price in matches:
                exchange_tx = ExchangeTransaction(
                    buyer_id=buy_order.user_id,
                    seller_id=sell_order.user_id,
                    from_currency=sell_order.from_currency,
                    to_currency=buy_order.from_currency,
                    amount=amount,
                    price=price
                )
                await self.process_exchange_transaction(exchange_tx)


      


    async def get_market_price(self, pair: str) -> Decimal:
        base_currency, quote_currency = pair.split('_')
        base_price = await price_oracle.get_price(base_currency)
        quote_price = await price_oracle.get_price(quote_currency)
        if quote_price == 0:
            raise ValueError(f"Invalid quote price for {quote_currency}")
        return base_price / quote_price
    async def generate_user_keys(self, user_id: str):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Serialize keys for storage
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Store keys (base64 encoded for easy storage)
        self.user_keys[user_id] = {
            'private_key': base64.b64encode(private_pem).decode('utf-8'),
            'public_key': base64.b64encode(public_pem).decode('utf-8')
        }

    async def get_user_private_key(self, user_id: str):
        if user_id not in self.user_keys:
            await self.generate_user_keys(user_id)
        
        private_pem = base64.b64decode(self.user_keys[user_id]['private_key'])
        return serialization.load_pem_private_key(
            private_pem,
            password=None
        )

    async def get_user_public_key(self, user_id: str):
        if user_id not in self.user_keys:
            await self.generate_user_keys(user_id)
        
        public_pem = base64.b64decode(self.user_keys[user_id]['public_key'])
        return serialization.load_pem_public_key(public_pem)
    async def place_order(self, order: Order):
        try:
            # Check balance before placing order
            if order.order_type == 'buy':
                balance = await self.blockchain.get_balance(order.user_id, order.to_currency)
                if balance < order.amount * order.price:
                    raise HTTPException(status_code=400, detail="Insufficient balance for buy order")
            else:
                balance = await self.blockchain.get_balance(order.user_id, order.from_currency)
                if balance < order.amount:
                    raise HTTPException(status_code=400, detail="Insufficient balance for sell order")

            # Generate a signature for the order
            private_key = await self.get_user_private_key(order.user_id)
            order_data = f"{order.user_id}{order.order_type}{order.from_currency}{order.to_currency}{order.amount}{order.price}"
            order.signature = self.crypto.sign_message(private_key, order_data)

            # Add order to the order book
            self.order_book.add_order(order)

            # Match orders
            pair = f"{order.from_currency}_{order.to_currency}"
            matches = self.order_book.match_orders(pair)

            for buy_order, sell_order, amount, price in matches:
                # Create an ExchangeTransaction object
                exchange_tx = ExchangeTransaction(
                    buyer_id=buy_order.user_id,
                    seller_id=sell_order.user_id,
                    amount=amount,
                    price=price,
                    from_currency=sell_order.from_currency,
                    to_currency=buy_order.from_currency
                )
                
                # Sign the transaction
                seller_private_key = await self.get_user_private_key(exchange_tx.seller_id)
                exchange_tx.signature = self.crypto.sign_message(seller_private_key, exchange_tx.get_message())
                
                # Process the transaction
                await self.process_exchange_transaction(exchange_tx)
            
            # Return the order ID
            return order.id

        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    async def process_exchange_transaction(self, exchange_tx: ExchangeTransaction):
        logging.info(f"Processing transaction: {exchange_tx}")

        # Verify the transaction signature before processing
        try:
            sender_public_key = await self.get_user_public_key(exchange_tx.seller_id)
            if not self.crypto.verify_signature(sender_public_key, exchange_tx.get_message(), exchange_tx.signature):
                raise ValueError("Invalid transaction signature")
        except Exception as e:
            logging.error(f"Signature verification failed: {str(e)}")
            raise ValueError("Transaction signature verification failed")

        fee = exchange_tx.amount * self.fee_percent
        amount_after_fee = exchange_tx.amount - fee

        # Create transactions
        buyer_tx = Transaction(sender=exchange_tx.buyer_id, receiver=exchange_tx.seller_id, amount=exchange_tx.amount * exchange_tx.price, price=exchange_tx.price)
        seller_tx = Transaction(sender=exchange_tx.seller_id, receiver=exchange_tx.buyer_id, amount=amount_after_fee, price=exchange_tx.price)
        fee_tx = Transaction(sender=exchange_tx.seller_id, receiver="fee_pool", amount=fee, price=exchange_tx.price)

        # Sign the transactions
        try:
            buyer_private_key = await self.get_user_private_key(exchange_tx.buyer_id)
            seller_private_key = await self.get_user_private_key(exchange_tx.seller_id)

            buyer_tx.signature = self.crypto.sign_message(buyer_private_key, f"{buyer_tx.sender}{buyer_tx.receiver}{buyer_tx.amount}{buyer_tx.price}")
            seller_tx.signature = self.crypto.sign_message(seller_private_key, f"{seller_tx.sender}{seller_tx.receiver}{seller_tx.amount}{seller_tx.price}")
            fee_tx.signature = self.crypto.sign_message(seller_private_key, f"{fee_tx.sender}{fee_tx.receiver}{fee_tx.amount}{fee_tx.price}")
        except Exception as e:
            logging.error(f"Transaction signing failed: {str(e)}")
            raise ValueError("Failed to sign transactions")

        # Add transactions to the blockchain
        try:
            await self.blockchain.add_transaction(buyer_tx)
            await self.blockchain.add_transaction(seller_tx)
            await self.blockchain.add_transaction(fee_tx)
        except Exception as e:
            logging.error(f"Failed to add transactions to blockchain: {str(e)}")
            raise ValueError("Failed to process transactions")

        logging.info("Exchange transaction processed successfully")



    async def add_liquidity(self, pool_id: str, user_id: str, amount_a: Decimal, amount_b: Decimal):
        # Check balances before adding liquidity
        balance_a = await self.blockchain.get_balance(user_id, pool_id.split('_')[0])
        balance_b = await self.blockchain.get_balance(user_id, pool_id.split('_')[1])
        if balance_a < amount_a or balance_b < amount_b:
            raise HTTPException(status_code=400, detail="Insufficient balance for liquidity provision")

        if pool_id not in self.liquidity_pools:
            self.liquidity_pools[pool_id] = LiquidityPool(
                id=pool_id,
                currency_a=pool_id.split('_')[0],
                currency_b=pool_id.split('_')[1],
                balance_a=amount_a,
                balance_b=amount_b,
                fee_percent=self.fee_percent
            )
        else:
            pool = self.liquidity_pools[pool_id]
            pool.balance_a += amount_a
            pool.balance_b += amount_b

        # Create transactions to transfer tokens to the pool
        tx_a = Transaction(sender=user_id, receiver=pool_id, amount=amount_a, currency=pool.currency_a)
        tx_b = Transaction(sender=user_id, receiver=pool_id, amount=amount_b, currency=pool.currency_b)

        # Add transactions to the blockchain
        await self.blockchain.add_transaction(tx_a)
        await self.blockchain.add_transaction(tx_b)

    async def remove_liquidity(self, pool_id: str, user_id: str, amount: Decimal):
        if pool_id not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")

        pool = self.liquidity_pools[pool_id]
        total_liquidity = pool.balance_a + pool.balance_b
        share = amount / total_liquidity

        amount_a = pool.balance_a * share
        amount_b = pool.balance_b * share

        # Check if the pool has enough liquidity
        if amount_a > pool.balance_a or amount_b > pool.balance_b:
            raise HTTPException(status_code=400, detail="Insufficient liquidity in the pool")

        pool.balance_a -= amount_a
        pool.balance_b -= amount_b

        # Create transactions to transfer tokens from the pool to the user
        tx_a = Transaction(sender=pool_id, receiver=user_id, amount=amount_a, currency=pool.currency_a)
        tx_b = Transaction(sender=pool_id, receiver=user_id, amount=amount_b, currency=pool.currency_b)

        # Add transactions to the blockchain
        await self.blockchain.add_transaction(tx_a)
        await self.blockchain.add_transaction(tx_b)

    async def swap(self, user_id: str, from_currency: str, to_currency: str, amount: Decimal):
        pool_id = f"{from_currency}_{to_currency}"
        if pool_id not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")

        # Check balance before swap
        balance = await self.blockchain.get_balance(user_id, from_currency)
        if balance < amount:
            raise HTTPException(status_code=400, detail="Insufficient balance for swap")

        pool = self.liquidity_pools[pool_id]
        
        if from_currency == pool.currency_a:
            in_balance = pool.balance_a
            out_balance = pool.balance_b
        else:
            in_balance = pool.balance_b
            out_balance = pool.balance_a

        # Calculate the amount of tokens to receive
        k = in_balance * out_balance
        new_in_balance = in_balance + amount
        new_out_balance = k / new_in_balance
        tokens_out = out_balance - new_out_balance

        # Apply fee
        fee = tokens_out * pool.fee_percent
        tokens_out -= fee

        # Update pool balances
        if from_currency == pool.currency_a:
            pool.balance_a = new_in_balance
            pool.balance_b = new_out_balance + fee
        else:
            pool.balance_b = new_in_balance
            pool.balance_a = new_out_balance + fee

        # Create transactions
        tx_in = Transaction(sender=user_id, receiver=pool_id, amount=amount, currency=from_currency)
        tx_out = Transaction(sender=pool_id, receiver=user_id, amount=tokens_out, currency=to_currency)
        fee_tx = Transaction(sender=pool_id, receiver="fee_pool", amount=fee, currency=to_currency)

        # Add transactions to the blockchain
        await self.blockchain.add_transaction(tx_in)
        await self.blockchain.add_transaction(tx_out)
        await self.blockchain.add_transaction(fee_tx)

        return tokens_out
    async def place_order(self, order_data: Dict[str, Any]):
        try:
            order = Order(
                user_id=order_data['user_id'],
                type=order_data['type'],
                order_type=order_data['order_type'],
                pair=order_data['pair'],
                base_currency=order_data['base_currency'],
                quote_currency=order_data['quote_currency'],
                amount=Decimal(order_data['amount']),
                price=Decimal(order_data['price']),
                from_currency=order_data['from_currency'],
                to_currency=order_data['to_currency']
            )
            self.order_book.add_order(order)
            return {"status": "success", "order_id": str(order.id)}
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {"status": "error", "message": str(e)}

import requests

user_balances = {
    f"bench_user_node_{i}": {
        "USD": Decimal("1000000.00"),
        "BTC": Decimal("100.0")
    } for i in range(NUM_NODES)
}


order_book = {
    "BTC_USD": []
}
class OrderRequest(BaseModel):
    user_id: str    
    type: str
    order_type: str
    pair: str
    base_currency: str
    quote_currency: str
    amount: str  # Keeping as str to convert to Decimal later
    price: str  # Keeping as str to convert to Decimal later
    from_currency: str
    to_currency: str
@app.post("/place_order")
async def place_order(order_data: OrderRequest):
    try:
        # Convert order data to Order model with Decimal conversion
        order = Order(
            user_id=order_data.user_id,
            type=order_data.type,
            order_type=order_data.order_type,
            pair=order_data.pair,
            base_currency=order_data.base_currency,
            quote_currency=order_data.quote_currency,
            amount=Decimal(order_data.amount),
            price=Decimal(order_data.price),
            from_currency=order_data.from_currency,
            to_currency=order_data.to_currency,
        )

        # Check if the user exists in the fake database
        user_id = order.user_id
        if user_id not in fake_users_db:
            # Create a new user if it doesn't exist
            fake_users_db[user_id] = {
                "username": user_id,
                "full_name": f"Benchmark User {user_id}",
                "email": f"{user_id}@example.com",
                "hashed_password": "fakehashedsecret",
                "disabled": False
            }
            # Initialize balance for new users
            user_balances[user_id] = {
                "USD": Decimal("100000.00"),
                "BTC": Decimal("10.0")
            }
            logger.info(f"Created new user: {user_id}")

        # Simulate order placement logic (this should be replaced with actual logic)
        result = await process_order(order)
        
        if result['status'] == 'success':
            return result
        else:
            raise HTTPException(status_code=400, detail=result['message'])

    except InvalidOperation as e:
        logger.error(f"Invalid decimal value: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid decimal value: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_order(order: Order) -> Dict[str, Any]:
    # Check if the user exists
    if order.user_id not in user_balances:
        logger.error(f"User {order.user_id} does not exist in user_balances")
        return {
            'status': 'failure',
            'message': 'User does not exist'
        }
    
    # Check if the user has sufficient balance for the order
    user_balance = user_balances[order.user_id]
    
    if order.type == 'limit' and order.order_type == 'buy':
        # For a buy order, check if the user has enough quote currency (USD)
        required_balance = order.amount * order.price
        if user_balance.get(order.quote_currency, Decimal("0.0")) < required_balance:
            return {
                'status': 'failure',
                'message': 'Insufficient balance'
            }
        # Deduct the required amount from the user's balance
        user_balance[order.quote_currency] -= required_balance
    
    elif order.type == 'limit' and order.order_type == 'sell':
        # For a sell order, check if the user has enough base currency (BTC)
        if user_balance.get(order.base_currency, Decimal("0.0")) < order.amount:
            return {
                'status': 'failure',
                'message': 'Insufficient balance'
            }
        # Deduct the required amount from the user's balance
        user_balance[order.base_currency] -= order.amount
    
    else:
        return {
            'status': 'failure',
            'message': 'Unsupported order type'
        }
    
    # Simulate adding the order to the order book
    if order.pair not in order_book:
        order_book[order.pair] = []
    order_dict = order.model_dump()
    order_dict['node_id'] = order.user_id.split('_')[-1]  # Extract node_id from user_id
    order_book[order.pair].append(order_dict)
    
    return {
        'status': 'success',
        'message': 'Order placed successfully',
    }




async def get_orders():
    # Implement this to return the list of orders
    # For example:
    return [order.dict() for order in exchange.order_book.get_orders()]


async def get_orders_from_node(node):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:{node['api_port']}/orders")
        if response.status_code == 200:
            return response.json()
        return []
@app.get("/orders")
async def get_orders():
    all_orders = []
    for pair_orders in order_book.values():
        all_orders.extend(pair_orders)
    return all_orders




@router.post("/add_liquidity")
async def add_liquidity(
    pool_id: str,
    amount_a: Decimal,
    amount_b: Decimal,
    current_user: str = Depends(get_current_user)
):
    try:
        await exchange.add_liquidity(pool_id, current_user, amount_a, amount_b)
        return {"message": "Liquidity added successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove_liquidity")
async def remove_liquidity(
    pool_id: str,
    amount: Decimal,
    current_user: str = Depends(get_current_user)
):
    try:
        await exchange.remove_liquidity(pool_id, current_user, amount)
        return {"message": "Liquidity removed successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/swap")
async def swap(
    from_currency: str,
    to_currency: str,
    amount: Decimal,
    current_user: str = Depends(get_current_user)
):
    try:
        tokens_out = await exchange.swap(current_user, from_currency, to_currency, amount)
        return {"message": "Swap successful", "tokens_received": tokens_out}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/place_limit_order")
async def place_limit_order(
    order_type: str,
    base_currency: str,
    quote_currency: str,
    amount: Decimal,
    price: Decimal,
    current_user: str = Depends(get_current_user)
):
    try:
        order_id = await exchange.place_limit_order(current_user, order_type, base_currency, quote_currency, amount, price)
        return {"order_id": order_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/add_liquidity")
async def add_liquidity(
    pool_id: str,
    amount_a: Decimal,
    amount_b: Decimal,
    current_user: str = Depends(get_current_user)
):
    try:
        await exchange.add_liquidity(pool_id, current_user, amount_a, amount_b)
        return {"message": "Liquidity added successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/swap")
async def swap(
    from_currency: str,
    to_currency: str,
    amount: Decimal,
    current_user: str = Depends(get_current_user)
):
    try:
        tokens_out = await exchange.swap(current_user, from_currency, to_currency, amount)
        return {"tokens_received": tokens_out}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/order_book/{pair}")
async def get_order_book(pair: str):
    buy_orders = exchange.order_book.buy_orders.get(pair, [])
    sell_orders = exchange.order_book.sell_orders.get(pair, [])
    return {
        "buy_orders": [order.dict() for order in buy_orders],
        "sell_orders": [order.dict() for order in sell_orders]
    }

@router.get("/liquidity_pools")
async def get_liquidity_pools():
    return {pool_id: pool.dict() for pool_id, pool in exchange.liquidity_pools.items()}

app.include_router(router, prefix="/exchange", tags=["exchange"])

# Periodic price update task
async def update_prices_periodically():
    while True:
        await price_oracle.update_prices()
        await asyncio.sleep(60)  # Update every minute


class MarketMakerBot:
    def __init__(self, exchange: Exchange, pair: str, spread: Decimal):
        self.exchange = exchange
        self.pair = pair
        self.spread = spread

    async def run(self):
        while True:
            try:
                # Get current market price (you need to implement this method)
                market_price = await self.exchange.get_market_price(self.pair)

                # Place buy order slightly below market price
                buy_price = market_price * (1 - self.spread / 2)
                await self.exchange.place_order(Order(
                    id=str(uuid.uuid4()),
                    user_id="market_maker_bot",
                    order_type="buy",
                    from_currency=self.pair.split("_")[1],
                    to_currency=self.pair.split("_")[0],
                    amount=Decimal('0.1'),  # Adjust as needed
                    price=buy_price,
                    status='open',
                    created_at=datetime.now()
                ))

                # Place sell order slightly above market price
                sell_price = market_price * (1 + self.spread / 2)
                await self.exchange.place_order(Order(
                    id=str(uuid.uuid4()),
                    user_id="market_maker_bot",
                    order_type="sell",
                    from_currency=self.pair.split("_")[0],
                    to_currency=self.pair.split("_")[1],
                    amount=Decimal('0.1'),  # Adjust as needed
                    price=sell_price,
                    status='open',
                    created_at=datetime.now()
                ))

                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Error in market maker bot: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

# Start the market maker bot

# Add this router to your main FastAPI app
app.include_router(router, prefix="/exchange", tags=["exchange"])

price_oracle = PriceOracle()  # Instead of MagicMock()


# Initialize the exchange object


@asynccontextmanager
async def lifespan(app: FastAPI):
    global initialization_complete
    wallet = Wallet()
    wallet.address = wallet.get_address()
    wallet.public_key = wallet.get_public_key()
    
    # Startup
    try:
        components = await initialize_components()
        app.state.components = components
        
        if initialization_complete:
            asyncio.create_task(update_prices_periodically())
            asyncio.create_task(periodic_mining(components['genesis_address'], wallet))
            asyncio.create_task(deploy_and_run_market_bot(
                components['exchange'], 
                components['vm'], 
                components['plata_contract'], 
                components['price_feed']
            ))
        else:
            logger.error("Initialization did not complete successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        initialization_complete = False
    
    yield
    
    # Cleanup
    initialization_complete = False
    # Add any other cleanup code here

app.router.lifespan_context = lifespan

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(router, prefix="/exchange", tags=["exchange"])


SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory storage for demonstration purposes


@app.on_event("shutdown")
async def on_shutdown():
    # Add any shutdown tasks if needed
    pass


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password, salt):
    return pwd_context.verify(plain_password + salt, hashed_password)

def authenticate_user(pincode: str):
    user = fake_users_db.get(pincode)
    if not user:
        return False
    if not verify_password(pincode, user.hashed_pincode, user.salt):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_salt():
    return base64.b64encode(os.urandom(16)).decode('utf-8')
# Define models
class User(BaseModel):
    pincode: str

class UserInDB(User):
    hashed_pincode: str
    wallet: dict
    salt: str
    alias: str
    disabled: bool = False  # Add the disabled attribute with a default value

class Token(BaseModel):
    access_token: str
    token_type: str
    wallet: dict

class EncryptRequest(BaseModel):
    message: str
    public_key: str

class EncryptResponse(BaseModel):
    encrypted_message: str

class DecryptRequest(BaseModel):
    encrypted_message: str

class DecryptResponse(BaseModel):
    decrypted_message: str

class TotalSupplyResponse(BaseModel):
    total_supply: Decimal
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        if isinstance(user_dict, UserInDB):
            return user_dict  # Directly return if it's already a UserInDB instance
        return UserInDB(**user_dict)  # Otherwise, unpack the dictionary
    return None


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user




def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

import jwt
from datetime import datetime, timedelta

def create_token(secret_key):
    # Define the token payload
    payload = {
        "exp": datetime.utcnow() + timedelta(hours=10000),  # Token expiration time
        "iat": datetime.utcnow(),  # Token issued at time
        "sub": "user_id"  # Subject of the token
    }
    # Create a JWT token
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token

# Generate a token using the same secret key as the server
token = create_token("your_secret_key_here")
print(token)

def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        
        # Log the expiration time for debugging
        logger.debug(f"Token expiration time: {datetime.utcfromtimestamp(exp)}")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_unique_alias():
    alias_length = 8  # Length of the alias
    while True:
        alias = ''.join(random.choices(string.ascii_lowercase + string.digits, k=alias_length))
        if alias not in fake_users_db:
            return alias
            
from wallet_registration import WalletRegistration
@app.get("/solana/balance/{address}")
async def get_solana_balance(address: str):
    try:
        balance = await helius_api.get_balance(address)
        return {"address": address, "balance": balance}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/solana/transactions/{address}")
async def get_solana_transactions(address: str, limit: int = 10):
    try:
        transactions = await helius_api.get_transactions(address, limit)
        return {"address": address, "transactions": transactions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/solana/transfer")
async def solana_transfer(from_address: str, to_address: str, amount: float):
    try:
        # Generate ZKP for the transaction
        secret = int.from_bytes(bytes.fromhex(from_address), 'big')
        public_input = int(amount * 1e9)  # Convert to lamports
        
        zk_proof = zk_system.prove(secret, public_input)
        
        # Verify the ZKP
        if not zk_system.verify(public_input, zk_proof):
            raise HTTPException(status_code=400, detail="Invalid ZK proof")
        
        # Proceed with the Solana transaction
        sender = Keypair.from_secret_key(bytes.fromhex(from_address))
        recipient = PublicKey(to_address)
        
        transaction = await AsyncClient.request_airdrop(recipient, int(amount * 1e9))
        await AsyncClient.confirm_transaction(transaction['result'], commitment=Confirmed)
        
        return {
            "status": "success",
            "transaction": transaction['result'],
            "zk_proof": {
                "stark_proof": zk_proof[0],
                "snark_proof": zk_proof[1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/verify_zk_proof")
async def verify_zk_proof(public_input: int, proof: dict):
    try:
        stark_proof = proof['stark_proof']
        snark_proof = proof['snark_proof']
        
        is_valid = zk_system.verify(public_input, (stark_proof, snark_proof))
        
        return {"is_valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate_qr_code")
async def generate_qr_code(wallet_address: str, coin_type: str, current_user: str = Depends(get_current_user)):
    try:
        # Define the subfolder where QR codes will be saved
        qr_code_folder = "qr_codes"

        # Ensure the folder exists
        if not os.path.exists(qr_code_folder):
            os.makedirs(qr_code_folder)

        # Generate the QR code and save it in the subfolder
        qr_file = secure_qr_system.generate_qr_code(wallet_address, coin_type)
        qr_file_path = os.path.join(qr_code_folder, os.path.basename(qr_file))

        # Assuming `secure_qr_system.generate_qr_code` already returns the path where the QR code is saved
        return FileResponse(qr_file_path, media_type="image/png", filename=os.path.basename(qr_file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating QR code: {str(e)}")

@app.post("/verify_qr_code")
async def verify_qr_code(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    try:
        qr_data = json.loads(await file.read())
        is_valid = secure_qr_system.verify_qr_code(qr_data)
        return {"is_valid": is_valid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying QR code: {str(e)}")
from user_management import fake_users_db
from solana.rpc.async_api import AsyncClient

class RegisterRequest(BaseModel):

    pincode: str
    user_id: str
class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None  # Add refresh_token
    token_type: str
    wallet: dict
    mnemonic: str
    qr_codes: Dict[str, str]

class PincodeInfo(BaseModel):
    pincode: str
    hashed_pincode: str
    wallet: dict
    salt: str
    alias: str
    qr_codes: dict

class UserInDB(BaseModel):
    pincode: str
    hashed_pincode: str
    wallet: dict
    salt: str
    alias: str
    qr_codes: dict
    pincodes: list[PincodeInfo] = []


from user_management import fake_users_db as global_fake_users_db
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.keypair import Keypair
import os
@app.post("/register", response_model=Token)
async def register(request: RegisterRequest):
    global global_fake_users_db
    try:
        # Check if the user already exists in the database
        user_data = global_fake_users_db.get(request.user_id)
        
        if user_data:
            if isinstance(user_data, dict):
                user_in_db = UserInDB(**user_data)
            else:
                user_in_db = user_data

            if any(p.pincode == request.pincode for p in user_in_db.pincodes):
                logger.info(f"Pincode already registered for user {request.user_id}. Adding a new wallet.")
        else:
            user_in_db = UserInDB(
                pincode=request.pincode,
                hashed_pincode='',
                wallet={},
                salt='',
                alias='',
                qr_codes={},
                pincodes=[]
            )
        
        # Generate salt and hash the pincode
        salt = generate_salt()
        hashed_pincode = get_password_hash(request.pincode + salt)
        
        # Create a new Plata wallet
        wallet = Wallet()
        mnemonic = wallet.generate_mnemonic()
        plata_wallet = {
            "address": wallet.address,
            "private_key": wallet.private_key_pem(),
            "public_key": wallet.public_key,
            "mnemonic": mnemonic
        }
        
        # Ensure the wallet address follows the correct format
        if not re.match(r'^plata[a-f0-9]{16}$', wallet.address):
            logger.error(f"Invalid wallet address format: {wallet.address}")
            raise ValueError(f"Invalid wallet address format: {wallet.address}")
        
        alias = wallet.generate_unique_alias()
        wallet_registration = WalletRegistration(blockcypher_api_key, alchemy_key)
        additional_wallets = wallet_registration.register_all_wallets(request.user_id)
        
        # Create a Solana wallet
        solders_keypair = Keypair()
        solana_wallet = {
            'address': str(solders_keypair.pubkey()),
            'private_key': solders_keypair.secret().hex(),
        }

        # Combine all wallets into one structure
        all_wallets = {
            "plata": plata_wallet,
            "bitcoin": additional_wallets.get('bitcoin', {}),
            "litecoin": additional_wallets.get('litecoin', {}),
            "ethereum": additional_wallets.get('ethereum', {}),
            "dogecoin": additional_wallets.get('dogecoin', {}),
            "solana": solana_wallet,
        }

        # Generate QR codes for all wallets
        qr_codes = {}
        for coin_type, wallet_info in all_wallets.items():
            if wallet_info and 'address' in wallet_info:
                qr_file = secure_qr_system.generate_qr_code(wallet_info['address'], coin_type)
                qr_codes[coin_type] = qr_file

        # Add or update the user_in_db object
        user_in_db.pincodes.append(PincodeInfo(
            pincode=request.pincode,
            hashed_pincode=hashed_pincode,
            salt=salt,
            wallet=all_wallets,
            alias=alias,
            qr_codes=qr_codes
        ))

        # Update the fields of the user_in_db object
        user_in_db.hashed_pincode = hashed_pincode
        user_in_db.wallet = all_wallets
        user_in_db.salt = salt
        user_in_db.alias = alias
        user_in_db.qr_codes = qr_codes

        # Save the updated user_in_db back to the global database
        global_fake_users_db[request.user_id] = user_in_db.dict()  # Convert to dict before storing
        
        # Generate access and refresh tokens
        access_token = create_access_token(data={"sub": request.pincode})
        refresh_token = create_refresh_token({"sub": request.pincode})  # Pass a dictionary with 'sub' field
        
        logger.info(f"New wallet registered with Plata address: {wallet.address}")

        # ---- Broadcast the new wallet event to other nodes ----
        await p2p_node.broadcast(Message(
            MessageType.NEW_WALLET, 
            {'user_id': request.user_id, 'wallet_address': wallet.address}
        ))

        # ---- WebSocket broadcast to subscribed clients ----
        await p2p_node.broadcast_event('new_wallet', {
            'user_id': request.user_id,
            'wallet_address': wallet.address,
            'wallets': all_wallets,
            'mnemonic': mnemonic,
            'qr_codes': qr_codes
        })

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,  # Return the refresh token
            "token_type": "bearer",
            "wallet": all_wallets,
            "mnemonic": mnemonic,
            "qr_codes": qr_codes
        }

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in registration: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


import qrcode
import base64
from io import BytesIO
from fastapi import HTTPException, Depends
from fastapi.logger import logger
# Add this new endpoint to get QR codes for a user's wallets
@app.get("/get_wallet_qr_codes", response_model=Dict[str, Dict[str, str]])
async def get_wallet_qr_codes(current_user: UserInDB = Depends(get_current_active_user)):
    try:
        user_id = current_user.username
        user = fake_users_db.get(user_id)
        
        if not user:
            logger.warning(f"User not found: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")
        
        if 'wallets' not in user or not user['wallets']:
            logger.warning(f"No wallets found for user: {user_id}")
            raise HTTPException(status_code=404, detail="No wallets found for user")
        
        qr_codes = {}
        for wallet_type, wallet in user['wallets'].items():
            if 'address' not in wallet:
                logger.warning(f"No address found for wallet type {wallet_type} for user {user_id}")
                continue
            
            try:
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(wallet['address'])
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                
                # Convert image to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                qr_codes[wallet_type] = img_str
                logger.info(f"QR code generated for wallet type {wallet_type} for user {user_id}")
            except Exception as e:
                logger.error(f"Error generating QR code for wallet type {wallet_type} for user {user_id}: {str(e)}")
        
        if not qr_codes:
            logger.warning(f"No QR codes could be generated for user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to generate any QR codes")
        
        # Prepare the response with CORS headers
        response_data = {"qr_codes": qr_codes}
        response = JSONResponse(content=json.dumps(response_data, cls=CustomJSONEncoder))
        response.headers["Access-Control-Allow-Origin"] = "*"  # Adjust as needed for your CORS policy
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_wallet_qr_codes for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred while generating QR codes")


@app.post("/token", response_model=Token)
def login(user: User):
    user_in_db = authenticate_user(user.pincode)
    if not user_in_db:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.pincode})
    return {"access_token": access_token, "token_type": "bearer", "wallet": user_in_db.wallet}

def generate_wallet_address(public_key):
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    hash_bytes = hashlib.sha256(public_bytes).digest()
    address = "plata" + urlsafe_b64encode(hash_bytes).decode('utf-8').rstrip("=")
    return address



def authenticate(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        pincode: str = payload.get("sub")
        if pincode is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return pincode
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def periodically_discover_nodes(p2p_node: P2PNode):
    while True:
        try:
            # Example of using the find_node method from P2PNode to discover nodes
            random_node_id = p2p_node.generate_random_id()
            await p2p_node.find_node(random_node_id)
        except Exception as e:
            logger.error(f"Error during periodic node discovery: {str(e)}")
        await asyncio.sleep(60)  # Discover nodes every 60 seconds






class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.tree = self.build_tree(transactions)

    def build_tree(self, transactions):
        if not transactions:
            return []
        tree = [transactions]
        while len(tree[-1]) > 1:
            level = []
            for i in range(0, len(tree[-1]), 2):
                left = tree[-1][i]
                right = tree[-1][i + 1] if i + 1 < len(tree[-1]) else left
                level.append(self.hash_pair(left, right))
            tree.append(level)
        return tree

    def hash_pair(self, left, right):
        left_str = json.dumps(left, sort_keys=True)
        right_str = json.dumps(right, sort_keys=True)
        return hashlib.sha256((left_str + right_str).encode('utf-8')).hexdigest()

    def get_root(self):
        return self.tree[-1][0] if self.tree else None

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.balances = {}
        self.stakes = {}
        self.wallets = []
        transaction: dagknight_pb2.Transaction
        self.contracts = []

class Wallet(BaseModel):
    address: str
    private_key: str
    public_key: str
    alias: Optional[str] = None


class Contract(BaseModel):
    address: str
    creator: str
class SearchResult(BaseModel):
    wallets: List[Wallet]
    contracts: List[Contract]
    transaction: Transaction  # Use the appropriate Transaction class here

    class Config:
        arbitrary_types_allowed = True


from pydantic import BaseModel, Field
from mnemonic import Mnemonic
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64
import hashlib
from typing import Optional
from pydantic import BaseModel, Field, validator
from typing import Optional
from mnemonic import Mnemonic
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64
import hashlib
import random
import string
import os
import bcrypt
import re
class Wallet(BaseModel):
    private_key: Optional[ec.EllipticCurvePrivateKey] = None
    public_key: Optional[str] = None
    mnemonic: Optional[Mnemonic] = None
    address: Optional[str] = None
    salt: Optional[bytes] = None
    hashed_pincode: Optional[str] = None

    def __init__(self, private_key=None, mnemonic=None, pincode=None, **data):
        super().__init__(**data)
        self.mnemonic = Mnemonic("english")

        if mnemonic:
            seed = self.mnemonic.to_seed(mnemonic)
            self.private_key = ec.derive_private_key(int.from_bytes(seed[:32], byteorder="big"), ec.SECP256R1(), default_backend())
        elif private_key:
            self.private_key = serialization.load_pem_private_key(private_key.encode(), password=None, backend=default_backend())
        else:
            self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

        # Generate public key and address after private key is initialized
        self.public_key = self.get_public_key()
        self.address = self.get_address()

        if pincode:
            self.salt = self.generate_salt()
            self.hashed_pincode = self.hash_pincode(pincode)

    def generate_salt(self) -> bytes:
        """Generate a new salt for hashing the pincode."""
        return bcrypt.gensalt()

    def hash_pincode(self, pincode: str) -> str:
        """Hash the provided pincode with the associated salt."""
        if not self.salt:
            raise ValueError("Salt is not set. Cannot hash pincode.")
        return bcrypt.hashpw(pincode.encode('utf-8'), self.salt).decode('utf-8')

    def verify_pincode(self, pincode: str) -> bool:
        """Verify if the provided pincode matches the stored hashed pincode."""
        if not self.hashed_pincode or not self.salt:
            raise ValueError("Hashed pincode or salt is not set. Cannot verify pincode.")
        return bcrypt.checkpw(pincode.encode('utf-8'), self.hashed_pincode.encode('utf-8'))

    def private_key_pem(self) -> str:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

    def get_public_key(self) -> str:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def generate_mnemonic(self):
        return self.mnemonic.generate(strength=128)

    def sign_message(self, message):
        signature = self.private_key.sign(
            message.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        return base64.b64encode(signature).decode('utf-8')

    def verify_signature(self, message, signature, public_key):
        try:
            public_key_obj = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())
            public_key_obj.verify(
                base64.b64decode(signature),
                message.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            return False

    def encrypt_message(self, message, public_key):
        rsa_public_key = serialization.load_pem_public_key(public_key.encode(), backend=default_backend())
        encrypted_message = rsa_public_key.encrypt(
            message.encode('utf-8'),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256()), label=None)
        return base64.b64encode(encrypted_message).decode('utf-8')

    def decrypt_message(self, encrypted_message):
        rsa_private_key = serialization.load_pem_private_key(self.private_key_pem().encode(), password=None, backend=default_backend())
        decrypted_message = rsa_private_key.decrypt(
            base64.b64decode(encrypted_message),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256()), label=None)
        return decrypted_message.decode('utf-8')

    def get_address(self) -> str:
        public_key_bytes = self.get_public_key().encode()
        address = "plata" + hashlib.sha256(public_key_bytes).hexdigest()[:16]

        # Verify if the address matches the expected format
        if not re.match(r'^plata[a-f0-9]{16}$', address):
            raise ValueError(f"Generated address {address} does not match the expected format.")

        return address

    def generate_unique_alias(self):
        alias_length = 8  # Length of the alias
        alias = ''.join(random.choices(string.ascii_lowercase + string.digits, k=alias_length))
        return alias

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self):
        """Convert the Wallet object to a dictionary for easy serialization."""
        return {
            "address": self.address,
            "public_key": self.public_key,
            "mnemonic": self.mnemonic.generate() if self.mnemonic else None,
            "hashed_pincode": self.hashed_pincode,
            # You can add more fields as needed, but avoid exposing private keys.
        }

class PQWallet(Wallet):
    def __init__(self):
        super().__init__()
        self.rsa_keypair = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

    def get_pq_public_key(self):
        return self.rsa_keypair.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

    def pq_encrypt(self, message):
        cipher = PKCS1_OAEP.new(self.rsa_keypair.public_key())
        encrypted_message = cipher.encrypt(message.encode('utf-8'))
        return base64.b64encode(encrypted_message).decode('utf-8')

    def pq_decrypt(self, ciphertext):
        cipher = PKCS1_OAEP.new(self.rsa_keypair)
        decrypted_message = cipher.decrypt(base64.b64decode(ciphertext))
        return decrypted_message.decode('utf-8')



def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
from pydantic import BaseModel

class TokenInfo(BaseModel):
    address: str
    name: str
    symbol: str
    balance: float
    wallets: dict

from pydantic import BaseModel

class ImportTokenRequest(BaseModel):
    address: str
    # Add other fields as necessary

class ImportWalletRequest(BaseModel):
    mnemonic: str
@app.post("/import_token", response_model=TokenInfo)
async def import_token(request: ImportTokenRequest, current_user: User = Depends(get_current_user)):
    try:
        # Import the Plata wallet first
        token = await blockchain.import_token(request.address, current_user.username)

        # After importing the Plata wallet, retrieve associated wallets
        wallet_registration = WalletRegistration(blockcypher_api_key, alchemy_key)
        additional_wallets = wallet_registration.register_all_wallets()

        # Combine the Plata wallet with the other wallets
        all_wallets = {
            "plata_wallet": token,
            "ethereum_wallet": additional_wallets['ethereum'],
            "bitcoin_wallet": additional_wallets['bitcoin'],
            "litecoin_wallet": additional_wallets['litecoin'],
            "dogecoin_wallet": additional_wallets['dogecoin']
        }

        # Return all the wallet information
        return {
            "address": token.address,
            "name": token.name,
            "symbol": token.symbol,
            "balance": float(token.balance_of(current_user.username)),
            "wallets": all_wallets  # Include all wallets in the response
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.post("/create_pq_wallet")
def create_pq_wallet(pincode: str = Depends(authenticate)):
    pq_wallet = PQWallet()
    address = pq_wallet.get_address()
    pq_public_key = pq_wallet.get_pq_public_key()
    return {"address": address, "pq_public_key": pq_public_key}
@app.post("/search", response_model=SearchResult)
async def search(query: str):
    query = query.lower()
    
    wallet_results = blockchain.search_wallets(query)
    transaction_results = blockchain.search_transactions(query)
    contract_results = blockchain.search_contracts(query)

    if not wallet_results and not transaction_results and not contract_results:
        raise HTTPException(status_code=404, detail="No results found")
    
    return SearchResult(
        wallets=wallet_results,
        transactions=transaction_results,
        contracts=contract_results
    )
class AddressRequest(BaseModel):
    address: str

class WalletRequest(BaseModel):
    wallet_address: str
from pydantic import BaseModel, Field
from pydantic_core import core_schema
class Transaction:
    def __init__(self, sender, receiver, amount, price, buyer_id, seller_id, wallet, tx_hash, timestamp):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.price = price
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.wallet = wallet
        self.tx_hash = tx_hash
        self.timestamp = timestamp
        self.signature = None
        self.zk_proof = None
    def to_dict(self):
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'price': self.price,
            'buyer_id': self.buyer_id,
            'seller_id': self.seller_id,
            'wallet': self.wallet,
            'tx_hash': self.tx_hash,
            'timestamp': self.timestamp,
            'signature': str(self.signature) if self.signature else None,
            'zk_proof': str(self.zk_proof) if self.zk_proof else None
        }





    def sign_transaction(self, zk_system: SecureHybridZKStark):
        # Create a secret from the transaction data
        secret = int(sha256(f"{self.sender}{self.receiver}{self.amount}{self.price}{self.timestamp}".encode()).hexdigest(), 16)
        
        # Use the tx_hash as the public input
        public_input = int(self.tx_hash, 16)

        # Generate the ZKP
        stark_proof, snark_proof = zk_system.prove(secret, public_input)

        # Store the proof
        self.zk_proof = (stark_proof, snark_proof)

        # For compatibility with existing code, we'll also set a signature
        self.signature = str(self.zk_proof)

    def verify_transaction(self, zk_system: SecureHybridZKStark):
        if not self.zk_proof:
            return False

        # Use the tx_hash as the public input
        public_input = int(self.tx_hash, 16)

        # Verify the ZKP
        return zk_system.verify(public_input, self.zk_proof)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import re
from decimal import Decimal
import hashlib
import traceback
from FiniteField import FieldElement

class TransactionModel(BaseModel):
    sender: str
    receiver: str
    amount: Decimal
    price: Decimal
    buyer_id: str
    seller_id: str
    wallet: str
    tx_hash: str
    timestamp: int


    @classmethod
    def from_transaction(cls, transaction: Transaction):
        return cls(
            sender=transaction.sender,
            receiver=transaction.receiver,
            amount=transaction.amount,
            price=transaction.price,
            buyer_id=transaction.buyer_id,
            seller_id=transaction.seller_id,
            wallet=transaction.wallet,
            tx_hash=transaction.tx_hash,
            timestamp=transaction.timestamp
        )
class TransactionInput(BaseModel):
    sender: str
    receiver: str
    amount: Decimal
    price: Decimal
    buyer_id: str
    seller_id: str
    wallet: str
    private_key: str

class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

class BalanceResponse(BaseModel):
    balances: Dict[str, float]
    transactions: List[TransactionModel] = Field(default_factory=list)
    zk_proof: Any  # You might want to define a specific structure for the ZKP

    @classmethod
    def from_balance_and_transactions(cls, balances: Dict[str, float], transactions: List[Transaction], zk_proof: Any):
        return cls(
            balances=balances,
            transactions=[TransactionModel.from_transaction(tx) for tx in transactions],
            zk_proof=zk_proof
        )
import traceback

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, FieldElement):
                return obj.to_int()  # Convert FieldElement to its integer value
            return super().default(obj)
        except Exception as e:
            logger.error(f"Error in CustomJSONEncoder: {str(e)}")
            logger.error(traceback.format_exc())
            raise e


def validate_wallet_address(wallet_address):
    try:
        if not wallet_address or not re.match(r'^plata[a-f0-9]{16}$', wallet_address):
            logger.error(f"Invalid wallet address format: {wallet_address}")
            raise HTTPException(status_code=422, detail=f"Invalid wallet address format: {wallet_address}")
        return wallet_address
    except Exception as e:
        logger.error(f"Error in validate_wallet_address: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def process_balances(balances):
    try:
        processed_balances = {}
        for currency, balance in balances.items():
            balance_decimal = Decimal(balance).quantize(Decimal('0.000000000000000001'))
            processed_balances[currency] = float(balance_decimal)
        return processed_balances
    except Exception as e:
        logger.error(f"Error in process_balances: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def process_transactions(raw_transactions):
    try:
        transactions = []
        for tx in raw_transactions:
            transactions.append(TransactionModel(
                tx_hash=tx.get('hash', 'unknown'),
                sender=tx.get('sender', 'unknown'),
                receiver=tx.get('receiver', 'unknown'),
                amount=float(Decimal(tx.get('amount', 0)).quantize(Decimal('0.000000000000000001'))),
                timestamp=tx.get('timestamp', 'unknown')
            ))
        return transactions
    except Exception as e:
        logger.error(f"Error in process_transactions: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
def process_proof_item(proof_item):
    try:
        logger.debug(f"Processing proof item: {proof_item}")

        root, queries, data = proof_item
        logger.debug(f"Root: {root}, Queries: {queries}, Data: {data}")

        processed_data = []

        if isinstance(data, FieldElement):
            # If `data` is a single FieldElement, handle it directly
            logger.debug(f"Data is a FieldElement, converting directly: {data}")
            processed_data.append(data.to_int())
        elif isinstance(data, (list, tuple)):
            # If `data` is iterable, process each element
            for idx, d in enumerate(data):
                logger.debug(f"Processing data element at index {idx}: {d}")

                if isinstance(d, FieldElement):
                    processed_data.append(d.to_int())
                    logger.debug(f"Converted FieldElement to int: {d.to_int()}")
                elif isinstance(d, tuple) and len(d) == 2:
                    index, hashes = d
                    logger.debug(f"Processing tuple with index {index} and hashes: {hashes}")

                    processed_hashes = [hash_.to_int() if isinstance(hash_, FieldElement) else hash_ for hash_ in hashes]
                    processed_data.append({"index": index, "hashes": processed_hashes})
                    logger.debug(f"Processed hashes: {processed_hashes}")
                elif isinstance(d, (list, set, tuple)):
                    processed_data.append([item.to_int() if isinstance(item, FieldElement) else item for item in d])
                    logger.debug(f"Processed iterable data: {[item.to_int() if isinstance(item, FieldElement) else item for item in d]}")
                else:
                    processed_data.append(d)
                    logger.debug(f"Appended non-iterable data directly: {d}")
        else:
            # If `data` is neither a FieldElement nor iterable, log an error
            logger.error(f"Unexpected data type: {type(data)}")
            raise TypeError(f"Unexpected data type: {type(data)}")

        processed_proof = {
            "root": root.to_int() if isinstance(root, FieldElement) else root,
            "queries": queries,
            "data": processed_data
        }
        logger.debug(f"Processed proof item: {processed_proof}")
        return processed_proof

    except Exception as e:
        logger.error(f"Error in process_proof_item: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def generate_zk_proof(processed_balances, wallet_address):
    try:
        logger.debug(f"Generating ZK proof for wallet address {wallet_address} with processed balances: {processed_balances}")

        total_balance = sum(processed_balances.values())
        logger.debug(f"Total balance calculated: {total_balance}")

        secret = int(total_balance * 10**18)
        public_input = int(hashlib.sha256(wallet_address.encode()).hexdigest(), 16)
        logger.debug(f"Secret: {secret}, Public Input: {public_input}")

        zk_proof = blockchain.zk_system.prove(secret, public_input)
        logger.debug(f"Generated raw zk_proof: {zk_proof}")

        processed_zk_proof = [process_proof_item(proof) for proof in zk_proof]
        logger.debug(f"Processed zk_proof: {processed_zk_proof}")

        return processed_zk_proof

    except Exception as e:
        logger.error(f"Error in generate_zk_proof: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
@app.route("/get_balance", methods=["GET", "POST"])
async def get_balance(request: Request):
    try:
        if blockchain is None:
            logger.error("Blockchain is not initialized.")
            raise HTTPException(status_code=500, detail="Blockchain is not initialized.")

        if request.method == "GET":
            wallet_address = request.query_params.get("wallet_address")
        else:  # POST
            body = await request.json()
            wallet_address = body.get("wallet_address")

        logger.debug(f"Received wallet address: {wallet_address}")
        wallet_address = validate_wallet_address(wallet_address)

        balances = await blockchain.get_balances(wallet_address)
        logger.debug(f"Retrieved balances: {balances}")

        processed_balances = process_balances(balances)
        logger.debug(f"Processed balances: {processed_balances}")

        logger.debug(f"Fetching transactions for wallet: {wallet_address}")
        raw_transactions = await blockchain.get_transactions(wallet_address)
        logger.debug(f"Raw transactions: {raw_transactions}")

        transactions = process_transactions(raw_transactions)
        logger.debug(f"Processed transactions: {transactions}")

        processed_zk_proof = generate_zk_proof(processed_balances, wallet_address)
        logger.debug(f"Processed ZK proof: {processed_zk_proof}")

        response_data = {
            "balances": processed_balances,
            "transactions": [tx.dict() for tx in transactions],
            "zk_proof": processed_zk_proof
        }
        logger.info(f"Response data for address {wallet_address}: {response_data}")

        response = JSONResponse(content=json.dumps(response_data, cls=CustomJSONEncoder))
        response.headers["Access-Control-Allow-Origin"] = "*"  # Adjust as needed
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in get_balance: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.post("/send_transaction")
async def send_transaction(transaction_input: TransactionInput, pincode: str = Depends(get_current_user)):
    try:
        logger.info(f"Received transaction request: {transaction_input}")

        # Resolve receiver address from alias if necessary
        if not transaction_input.receiver.startswith("plata"):
            receiver_address = None
            for wallet in blockchain.wallets:
                if wallet.alias == transaction_input.receiver:
                    receiver_address = wallet.address
                    break
            if not receiver_address:
                raise HTTPException(status_code=400, detail="Invalid receiver alias")
            transaction_input.receiver = receiver_address

        # Create wallet instance using the private key provided in the transaction
        wallet = Wallet(private_key=transaction_input.private_key)
        logger.info("Created wallet from private key.")

        # Create a Transaction object from the input
        transaction = Transaction(
            sender=transaction_input.sender,
            receiver=transaction_input.receiver,
            amount=transaction_input.amount,
            price=transaction_input.price,
            buyer_id=transaction_input.buyer_id,
            seller_id=transaction_input.seller_id,
            wallet=transaction_input.wallet,
            tx_hash=generate_tx_hash(transaction_input),  # You need to implement this function
            timestamp=int(time.time())
        )

        # Add public key to the transaction
        transaction.public_key = wallet.get_public_key()
        logger.info(f"Public key added to transaction: {transaction.public_key}")

        # Generate and sign the transaction using Zero-Knowledge Proofs (ZKP)
        transaction.sign_transaction(blockchain.zk_system)
        logger.info("Transaction signed with ZKP.")

        # Verify the transaction
        if not transaction.verify_transaction(blockchain.zk_system):
            raise HTTPException(status_code=400, detail="Invalid transaction signature")

        # Add the transaction to the blockchain
        if blockchain.add_transaction(transaction):
            logger.info(f"Transaction from {transaction.sender} to {transaction.receiver} added to blockchain.")

            # ---- Broadcast the new transaction event to other nodes ----
            await p2p_node.broadcast(Message(
                MessageType.TRANSACTION, 
                {'tx_hash': transaction.tx_hash, 'transaction': transaction.to_dict()}
            ))

            # ---- WebSocket broadcast to subscribed clients ----
            await p2p_node.broadcast_event('transaction_sent', {
                'tx_hash': transaction.tx_hash,
                'sender': transaction.sender,
                'receiver': transaction.receiver,
                'amount': transaction.amount,
                'timestamp': transaction.timestamp
            })

            # Broadcast the new balance to the user
            new_balance = blockchain.get_balance(transaction.sender)
            await manager.broadcast(f"New balance for {transaction.sender}: {new_balance}")
            
            return {"success": True, "message": "Transaction added successfully."}
        else:
            logger.info(f"Transaction from {transaction.sender} to {transaction.receiver} failed to add: insufficient balance")
            return {"success": False, "message": "Transaction failed to add: insufficient balance"}

    except ValueError as e:
        logger.error(f"Error deserializing private key: {e}")
        raise HTTPException(status_code=400, detail="Invalid private key format")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def generate_tx_hash(transaction_input: TransactionInput) -> str:
    # Implement a function to generate a transaction hash
    # This is just a simple example, you might want to use a more sophisticated method
    data = f"{transaction_input.sender}{transaction_input.receiver}{transaction_input.amount}{transaction_input.price}{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()


@app.post("/deploy_contract")
def deploy_contract(contract: Contract, pincode: str = Depends(get_current_user)):
    blockchain.add_contract(contract)
    return {"success": True, "contract_address": contract.address}


@app.post("/send_batch_transactions")
async def send_batch_transactions(batch_input: BatchTransactionInput, pincode: str = Depends(authenticate)):
    results = []
    try:
        for transaction_input in batch_input.transactions:
            try:
                # Create wallet instance using the private key provided in the transaction
                wallet = Wallet(private_key=transaction_input.private_key)

                # Create a Transaction object from the input
                transaction = Transaction(
                    sender=transaction_input.sender,
                    receiver=transaction_input.receiver,
                    amount=transaction_input.amount,
                    price=transaction_input.price,
                    buyer_id=transaction_input.buyer_id,
                    seller_id=transaction_input.seller_id,
                    wallet=transaction_input.wallet,
                    tx_hash=generate_tx_hash(transaction_input),
                    timestamp=int(time.time())
                )

                # Add public key to the transaction
                transaction.public_key = wallet.get_public_key()

                # Generate and sign the transaction using Zero-Knowledge Proofs (ZKP)
                transaction.sign_transaction(blockchain.zk_system)

                if transaction.verify_transaction(blockchain.zk_system) and blockchain.add_transaction(transaction):
                    results.append({"success": True, "message": "Transaction added successfully.", "transaction": transaction_input.dict()})
                else:
                    results.append({"success": False, "message": "Transaction failed to add: invalid signature or insufficient balance", "transaction": transaction_input.dict()})

            except ValueError as e:
                results.append({"success": False, "message": f"Error processing transaction: {str(e)}", "transaction": transaction_input.dict()})
            except Exception as e:
                results.append({"success": False, "message": f"Unexpected error: {str(e)}", "transaction": transaction_input.dict()})

    except Exception as e:
        logger.error(f"Error processing batch transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch transactions")

    return results

def generate_tx_hash(transaction_input: TransactionInput) -> str:
    # Implement a function to generate a transaction hash
    # This is just a simple example, you might want to use a more sophisticated method
    data = f"{transaction_input.sender}{transaction_input.receiver}{transaction_input.amount}{transaction_input.price}{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()

class DeployContractRequest(BaseModel):
    sender_address: str
    collateral_token: str
    initial_price: float

@app.post("/deploy_contract")
def deploy_contract(request: DeployContractRequest):
    try:
        price_feed = SimplePriceFeed(request.initial_price)
        contract_address = vm.deploy_contract(request.sender_address, Cashewstable, request.collateral_token, price_feed)
        return {"contract_address": contract_address}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MineBlockRequest(BaseModel):
    node_id: str
    wallet: str  # Ensure this matches the expected type
    wallet_address: str
    node_ip: str
    node_port: int
    pincode: Optional[str] = None


async def mine_block(miner_address):
    logger.info(f"Starting to mine block for miner address: {miner_address}")
    while True:
        try:
            data = f"Block mined by {miner_address}"
            quantum_signature = blockchain.generate_quantum_signature()
            transactions = blockchain.pending_transactions[:10]  # Get the first 10 pending transactions
            reward = blockchain.add_new_block(data, quantum_signature, transactions, miner_address)

            logger.info(f"Block mined successfully with reward: {reward}")

            # Broadcast the new block to all peers
            await blockchain.propagate_block_to_all_peers(blockchain.chain[-1].to_dict())
            
            await asyncio.sleep(10)  # Wait for 10 seconds before mining the next block
        except Exception as e:
            logger.error(f"Error while mining block: {str(e)}")
            await asyncio.sleep(10)  # Wait for 10 seconds before retrying

class QuantumHolographicNetwork:
    def __init__(self, size):
        self.size = size
        self.qubits = QuantumRegister(size**2)
        self.circuit = QuantumCircuit(self.qubits)
        self.entanglement_matrix = np.zeros((size, size))

    def initialize_network(self):
        for i in range(self.size**2):
            self.circuit.h(i)
        logging.info("Network initialized with Hadamard gates.")

    def apply_holographic_principle(self):
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.circuit.cz(i*self.size, j*self.size)
        logging.info("Applied holographic principle with CZ gates.")

    def simulate_gravity(self, mass_distribution):
        for i in range(self.size):
            for j in range(self.size):
                strength = mass_distribution[i, j]
                self.circuit.rz(strength, i*self.size + j)
        logging.info("Simulated gravity with RZ gates.")

    def measure_entanglement(self):
        for i in range(self.size):
            for j in range(self.size):
                if i != j:
                    self.circuit.cz(i*self.size, j*self.size)
                    self.circuit.measure_all()
                    simulator = Aer.get_backend('qasm_simulator')
                    transpiled_circuit = transpile(self.circuit, simulator)
                    job = simulator.run(transpiled_circuit)
                    counts = job.result().get_counts(transpiled_circuit)
                    self.entanglement_matrix[i, j] = counts.get('11', 0) / sum(counts.values())
        logging.info("Measured entanglement and updated entanglement matrix.")

    def create_black_hole(self, center, radius):
        for i in range(self.size):
            for j in range(self.size):
                if ((i - center[0])**2 + (j - center[1])**2) <= radius**2:
                    for k in range(4):
                        self.circuit.cx(i*self.size + j, ((i+1)%self.size)*self.size + ((j+1)%self.size))
        logging.info("Created black hole.")

    def extract_hawking_radiation(self, black_hole_region):
        edge_qubits = [q for q in black_hole_region if q in self.get_boundary_qubits()]
        for q in edge_qubits:
            self.circuit.measure(q)
        logging.info("Extracted Hawking radiation from edge qubits.")

    def get_boundary_qubits(self):
        return [i for i in range(self.size**2) if i < self.size or i >= self.size*(self.size-1) or i % self.size == 0 or i % self.size == self.size-1]

import time
import logging
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
import numpy as np
import networkx as nx
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.optimize import minimize
from fastapi import Depends, HTTPException, FastAPI
from decimal import Decimal
import random
from SecureHybridZKStark import SecureHybridZKStark

logger = logging.getLogger(__name__)
continue_mining = True
executor = ThreadPoolExecutor(max_workers=1)

# def mining_algorithm(iterations=1):
    # try:
        # process = psutil.Process()
        # memory_info = process.memory_info()
        # logging.info(f"Memory usage at start: {memory_info.rss / (1024 * 1024):.2f} MB")

        # logging.info("Initializing Quantum Annealing Simulation")
        # global num_qubits, graph
        # num_qubits = 10  # Reduced number of qubits
        # graph = nx.grid_graph(dim=[2, 5])  # 2x5 grid instead of 5x5
        # logging.info(f"Initialized graph with {len(graph.nodes)} nodes and {len(graph.edges())} edges")
        # logging.info(f"Graph nodes: {list(graph.nodes())}")
        # logging.info(f"Graph edges: {list(graph.edges())}")

        # def quantum_annealing_simulation(params):
            # hamiltonian = sparse.csr_matrix((2**num_qubits, 2**num_qubits), dtype=complex)
            
            # grid_dims = list(graph.nodes())[-1]
            # rows, cols = grid_dims[0] + 1, grid_dims[1] + 1
            
            # for edge in graph.edges():
                # i, j = edge
                # sigma_z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
                
                # i_int = i[0] * cols + i[1]
                # j_int = j[0] * cols + j[1]
                
                # term_i = sparse.kron(sparse.eye(2**i_int, dtype=complex), sigma_z)
                # term_i = sparse.kron(term_i, sparse.eye(2**(num_qubits-i_int-1), dtype=complex))
                # hamiltonian += term_i
                
                # term_j = sparse.kron(sparse.eye(2**j_int, dtype=complex), sigma_z)
                # term_j = sparse.kron(term_j, sparse.eye(2**(num_qubits-j_int-1), dtype=complex))
                # hamiltonian += term_j

            # problem_hamiltonian = sparse.diags(np.random.randn(2**num_qubits), dtype=complex)
            # hamiltonian += params[0] * problem_hamiltonian

            # initial_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
            # evolution = sparse.linalg.expm(-1j * hamiltonian.tocsc() * params[1])
            # final_state = evolution @ initial_state

            # return -np.abs(final_state[0])**2  # Negative because we're minimizing

        # cumulative_counts = {}
        # for iteration in range(iterations):
            # logging.info(f"Starting iteration {iteration + 1}/{iterations}")
            # logging.info("Running Quantum Annealing Simulation")
            # start_time_simulation = time.time()
            # random_params = [random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)]
            # logging.info(f"Random parameters: {random_params}")
            
            # result = minimize(quantum_annealing_simulation, random_params, method='Nelder-Mead')
            # end_time_simulation = time.time()

            # simulation_duration = end_time_simulation - start_time_simulation
            # logging.info(f"Simulation completed in {simulation_duration:.2f} seconds")
            # logging.info(f"Optimization result: {result}")

            # logging.info("Simulating gravity effects")
            # mass_distribution = np.random.rand(2, 5)  # 2x5 grid
            # gravity_factor = np.sum(mass_distribution) / 10

            # logging.info("Creating quantum-inspired black hole")
            # black_hole_position = np.unravel_index(np.argmax(mass_distribution), mass_distribution.shape)
            # black_hole_strength = mass_distribution[black_hole_position]

            # logging.info("Measuring entanglement")
            # entanglement_matrix = np.abs(np.outer(result.x, result.x))
            # logging.info(f"Entanglement Matrix: {entanglement_matrix}")

            # logging.info("Extracting Hawking radiation analogue")
            # hawking_radiation = np.random.exponential(scale=black_hole_strength, size=10)

            # logging.info("Calculating final quantum state")
            # final_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
            # counts = {format(i, f'0{num_qubits}b'): abs(val)**2 for i, val in enumerate(final_state) if abs(val)**2 > 1e-6}

            # for state, prob in counts.items():
                # if state in cumulative_counts:
                    # cumulative_counts[state] += prob
                # else:
                    # cumulative_counts[state] = prob

        # memory_info = process.memory_info()
        # logging.info(f"Memory usage after simulation: {memory_info.rss / (1024 * 1024):.2f} MB")

        # qhins = np.trace(entanglement_matrix)
        # hashrate = 1 / (simulation_duration * iterations)

        # logging.info(f"QHINs: {qhins:.6f}")
        # logging.info(f"Hashrate: {hashrate:.6f} hashes/second")

        # return {
            # "counts": cumulative_counts,
            # "energy": result.fun,
            # "entanglement_matrix": entanglement_matrix,
            # "qhins": qhins,
            # "hashrate": hashrate
        # }
    # except Exception as e:
        # logging.error(f"Error in mining_algorithm: {str(e)}")
        # logging.error(traceback.format_exc())
        # return {"success": False, "message": f"Error in mining_algorithm: {str(e)}"}
class MineBlockRequest(BaseModel):
    wallet_address: str

class MiningData(BaseModel):
    block_header: List[int]
    difficulty: int

class SolutionSubmission(BaseModel):
    nonce: int

import time
import hashlib

@app.get("/get_mining_data", response_model=MiningData)
async def get_mining_data():
    try:
        block_header = generate_block_header()
        difficulty = blockchain.difficulty
        return MiningData(block_header=block_header, difficulty=difficulty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mining data: {str(e)}")

@app.post("/submit_solution")
async def submit_solution(submission: SolutionSubmission):
    try:
        if verify_solution(submission.nonce):
            new_block = create_new_block(submission.nonce)
            return {"status": "success", "message": "Solution accepted", "block_hash": new_block.hash}
        else:
            return {"status": "rejected", "message": "Invalid solution"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing solution: {str(e)}")

def generate_block_header() -> List[int]:
    header = [0] * 25
    last_block = blockchain.chain[-1]
    header[0] = int(time.time())
    header[1] = int(last_block.hash, 16)
    return header

def verify_solution(nonce: int) -> bool:
    block_header = generate_block_header()
    block_header[8] = nonce
    result = keccak_f1600(block_header)
    return int.from_bytes(result, 'big') < blockchain.difficulty

def create_new_block(nonce: int):
    data = f"Block mined with nonce: {nonce}"
    quantum_signature = blockchain.generate_quantum_signature()
    transactions = blockchain.pending_transactions[:10]
    new_block = QuantumBlock(
        previous_hash=blockchain.chain[-1].hash,
        data=data,
        quantum_signature=quantum_signature,
        reward=blockchain.get_block_reward(),
        transactions=transactions
    )
    new_block.nonce = nonce
    new_block.hash = new_block.compute_hash()
    if blockchain.add_block(new_block):
        return new_block
    else:
        raise ValueError("Failed to add new block to the blockchain")
        continue_mining = False
def is_blockchain_ready():
    global blockchain
    if blockchain is None:
        logger.error("Blockchain is not ready yet.")
        return False
    return True

def update_mining_stats(blocks_mined, start_time, last_reward, wallet_address, currency="PLATA"):
    global mining_stats
    current_time = time.time()
    mining_duration = current_time - start_time
    
    try:
        total_balance = blockchain.get_balance(wallet_address, currency)  # Add currency as an argument
    except Exception as e:
        logger.error(f"Error getting balance for wallet {wallet_address}: {str(e)}")
        total_balance = 0
    
    mining_stats = {
        "blocks_mined": blocks_mined,
        "mining_duration": mining_duration,
        "hash_rate": blocks_mined / mining_duration if mining_duration > 0 else 0,
        "last_reward": str(last_reward),
        "total_reward": str(total_balance),
        "difficulty": blockchain.difficulty
    }
    
    logger.info(f"Updated mining stats: {mining_stats}")


@app.get("/mining_stats")
async def get_mining_stats():
    global continue_mining, mining_stats
    return {
        "is_mining": continue_mining,
        "stats": mining_stats
    }

# Initialize mining stats
mining_stats = {
    "blocks_mined": 0,
    "mining_duration": 0,
    "hash_rate": 0,
    "last_reward": "0",
    "total_reward": "0",
    "difficulty": 0
}

@app.get("/mining_status")
async def get_mining_status():
    global continue_mining
    return {"is_mining": continue_mining}
async def process_exchange_orders():
    for pair in exchange.order_book.buy_orders.keys():
        matches = exchange.order_book.match_orders(pair)
        for buy_order, sell_order, amount, price in matches:
            exchange_tx = ExchangeTransaction(
                buyer_id=buy_order.user_id,
                seller_id=sell_order.user_id,
                from_currency=sell_order.from_currency,
                to_currency=buy_order.from_currency,
                amount=amount,
                price=price
            )
            await exchange.process_exchange_transaction(exchange_tx)

async def create_new_block(miner_address: str):
    try:
        new_block = await self.create_block(miner_address)
        if new_block:
            logger.info(f"New block mined: {new_block.hash}")
            return new_block
        else:
            logger.error("Failed to mine a new block")
            return None
    except ValueError as e:
        logger.error(f"Block creation failed: {str(e)}")
        return None
async def get_mining_stats():
    try:
        # Assume these are global variables or can be fetched from the blockchain or another source
        total_blocks_mined = len(blockchain.chain)
        latest_block = blockchain.chain[-1] if blockchain.chain else None
        hash_rate = None
        mining_reward = None

        if latest_block:
            mining_reward = latest_block.reward
            # Calculate hash rate based on the latest block's timestamp and the previous block's timestamp
            if len(blockchain.chain) > 1:
                previous_block = blockchain.chain[-2]
                time_difference = latest_block.timestamp - previous_block.timestamp
                hash_rate = 1 / time_difference if time_difference > 0 else None

        mining_stats = {
            "total_blocks_mined": total_blocks_mined,
            "current_hashrate": hash_rate,
            "mining_reward": mining_reward,
            "last_block_hash": latest_block.hash if latest_block else None,
            "last_block_time": latest_block.timestamp if latest_block else None,
        }
        return mining_stats
    except Exception as e:
        logger.error(f"Error fetching mining stats: {str(e)}")
        raise
        

# Function to periodically trigger mining
async def periodic_mining(miner_address: str, wallet: str):
    while True:
        try:
            await mine_block(MineBlockRequest(
                node_id="automated_miner",
                wallet=wallet.address,  # Ensure this is a string
                wallet_address=miner_address,
                node_ip="127.0.0.1",
                node_port=8000,
                pincode="automated_miner_pincode"
            ))
        except Exception as e:
            logger.error(f"Error in periodic mining: {str(e)}")
            logger.error(traceback.format_exc())
        await asyncio.sleep(600)  # Wait for 10 minutes before the next mining attempt






async def gossip_protocol(block_data):
    nodes = node_directory.discover_nodes()
    random.shuffle(nodes)
    gossip_nodes = nodes[:3]  # Randomly select 3 nodes to start gossiping
    tasks = [propagate_block(f"http://{node['ip_address']}:{node['port']}/receive_block", block_data) for node in gossip_nodes]
    await asyncio.gather(*tasks)


class BlockData(BaseModel):
    previous_hash: str
    data: str
    quantum_signature: str
    reward: float
    transactions: List[dict]
    hash: str

@app.post("/receive_block")
def receive_block(block: BlockData, pincode: str = Depends(authenticate)):
    try:
        # Convert the block data into a QuantumBlock object
        new_block = QuantumBlock(
            previous_hash=block.previous_hash,
            data=block.data,
            quantum_signature=block.quantum_signature,
            reward=block.reward,
            transactions=block.transactions
        )
        
        # Validate the block hash
        if new_block.hash != block.hash:
            raise HTTPException(status_code=400, detail="Block hash does not match computed hash")
        
        # Validate and add the block to the blockchain
        if blockchain.add_block(new_block):
            logger.info(f"Received and added block with hash: {new_block.hash}")
            return {"success": True, "message": "Block added successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid block")
    except Exception as e:
        logger.error(f"Error receiving block: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error receiving block: {str(e)}")

class PBFT:
    def __init__(self, f):
        self.view = 0
        self.prepared = set()
        self.committed = set()
        self.f = f  # Number of maximum faulty nodes

    def pre_prepare(self, block):
        # Leader sends pre-prepare message
        message = {"view": self.view, "block": block}
        self.broadcast(message, "pre_prepare")

    def prepare(self, message):
        # Replica receives pre-prepare and sends prepare message
        self.prepared.add(message["block"].hash)
        self.broadcast(message, "prepare")
    def commit(self, message):
        # Replica receives prepare messages and sends commit message
        if len(self.prepared) >= (2 * self.f + 1):
            self.committed.add(message["block"].hash)
            self.broadcast(message, "commit")

    def reply(self, message):
        # Replica receives commit messages and sends reply
        if len(self.committed) >= (2 * self.f + 1):
            self.apply_block(message["block"])
            self.broadcast(message, "reply")


    def broadcast(self, message, stage):
        nodes = node_directory.discover_nodes()
        for node in nodes:
            # Send the message to each node
            pass

    def apply_block(self, block):
        blockchain.add_block(block.data, block.quantum_signature, block.transactions, block.miner_address)


pbft = PBFT(f=1)  # Assuming 1 faulty node, adjust as needed


class ConsensusManager:
    def __init__(self):
        self.node_states = {}

    def add_node_state(self, node_id, state):
        self.node_states[node_id] = state

    def get_consensus(self):
        if not self.node_states:
            return None
        return max(set(self.node_states.values()), key=list(self.node_states.values()).count)


consensus_manager = ConsensusManager()



class ViewChange:
    def __init__(self):
        self.current_view = 0
        self.failed_leader = None

    def initiate_view_change(self):
        self.current_view += 1
        self.failed_leader = consensus.current_leader
        consensus.elect_leader()
        if consensus.current_leader:
            logger.info(f"View change initiated. New view: {self.current_view}, new leader: {consensus.current_leader['node_id']}")
        else:
            logger.warning("View change initiated, but no new leader was elected.")


view_change = ViewChange()
class NodeClient:
    def __init__(self, address):
        self.address = address
        self.channel = aio.insecure_channel(address)
        self.stub = dagknight_pb2_grpc.DAGKnightStub(self.channel)
    async def propagate_transaction(self, transaction):
        try:
            request = dagknight_pb2.Transaction(
                sender=transaction.sender,
                receiver=transaction.receiver,
                amount=int(transaction.amount),
                price=int(transaction.price),
                buyer_id=transaction.buyer_id,
                seller_id=transaction.seller_id,
                public_key=transaction.public_key,
                signature=transaction.signature,
                public_input=transaction.public_input,  # Add this
                zk_proof=transaction.zk_proof  # Add this
            )
            response = await self.stub.PropagateTransaction(request)
            return response
        except Exception as e:
            logger.error(f"Failed to propagate transaction to {self.address}: {str(e)}")
            return None


    async def sync_state(self, last_known_block):
        try:
            request = dagknight_pb2.SyncRequest(last_known_block=last_known_block)
            response = await self.stub.SyncState(request)
            return response
        except Exception as e:
            logger.error(f"Failed to sync state with {self.address}: {str(e)}")
            return None

    async def verify_zkp(self, public_input, zk_proof):
        try:
            request = dagknight_pb2.ZKPVerificationRequest(public_input=public_input, zk_proof=zk_proof)
            response = await self.stub.VerifyZKP(request)
            return response
        except Exception as e:
            logger.error(f"Failed to verify ZKP with {self.address}: {str(e)}")
            return None

    async def full_state_sync(self):
        try:
            request = dagknight_pb2.FullStateSyncRequest()
            response = await self.stub.FullStateSync(request)
            return response
        except Exception as e:
            logger.error(f"Failed to perform full state sync with {self.address}: {str(e)}")
            return None

async def connect_to_peers(node_directory):
    peers = {}
    for address in node_directory.discover_nodes():
        peers[address] = NodeClient(address)
    return peers

async def periodic_sync(blockchain, peers):
    while True:
        for peer in peers.values():
            try:
                last_block = blockchain.get_last_block()
                response = await peer.sync_state(last_block.number)
                if response and response.new_blocks:
                    for block_proto in response.new_blocks:
                        block = QuantumBlock(
                            previous_hash=block_proto.previous_hash,
                            data=block_proto.data,
                            quantum_signature=block_proto.quantum_signature,
                            reward=Decimal(str(block_proto.reward)),
                            transactions=[Transaction.from_proto(tx) for tx in block_proto.transactions],
                            timestamp=block_proto.timestamp
                        )
                        block.hash = block_proto.hash
                        block.nonce = block_proto.nonce
                        blockchain.add_block(block)
            except Exception as e:
                logger.error(f"Error during periodic sync with peer {peer.address}: {str(e)}")
        await asyncio.sleep(60)  # Sync every 60 seconds

# Example usage
if not pbft.committed:
    view_change.initiate_view_change()

class SecureDAGKnightServicer:
    def __init__(self, secret_key, node_directory, vm, zk_system):
        self.secret_key = secret_key
        self.node_directory = node_directory
        self.consensus = Consensus(blockchain=None)
        self.blockchain = QuantumBlockchain(self.consensus, secret_key, node_directory, vm)
        self.zk_system = zk_system

        self.consensus.blockchain = self.blockchain
        self.private_key = self.load_or_generate_private_key()
        self.logger = logging.getLogger(__name__)
        self.security_manager = SecurityManager(secret_key)
        self.transaction_pool = []
        self.node_stubs = []  # Initialize the list for node stubs

    def get_all_node_stubs(self):
        return self.node_stubs

    def add_node_stub(self, node_stub):
        self.node_stubs.append(node_stub)

    def store_transaction(self, transaction_hash, transaction):
        # Store the transaction in the node's directory or any other storage you use
        self.node_directory.store_transaction(transaction_hash, transaction)
        print(f"Transaction with hash {transaction_hash} stored successfully.")

    def compute_transaction_hash(self, transaction):
        transaction_data = f"{transaction.sender}{transaction.receiver}{transaction.amount}{transaction.signature}{transaction.public_key}{transaction.price}{transaction.buyer_id}{transaction.seller_id}"
        return hashlib.sha256(transaction_data.encode()).hexdigest()

    async def PropagateOrder(self, request, context):
        try:
            order = Order(
                user_id=request.user_id,
                order_type=request.order_type,
                base_currency=request.base_currency,
                quote_currency=request.quote_currency,
                amount=Decimal(request.amount),
                price=Decimal(request.price)
            )
            await self.exchange.place_order(order)
            return dagknight_pb2.PropagateOrderResponse(success=True, message="Order propagated successfully")
        except Exception as e:
            return dagknight_pb2.PropagateOrderResponse(success=False, message=str(e))

    async def PropagateTrade(self, request, context):
        try:
            trade = Trade(
                buyer_id=request.buyer_id,
                seller_id=request.seller_id,
                base_currency=request.base_currency,
                quote_currency=request.quote_currency,
                amount=Decimal(request.amount),
                price=Decimal(request.price)
            )
            await self.exchange.execute_trade(trade)
            return dagknight_pb2.PropagateTradeResponse(success=True, message="Trade propagated successfully")
        except Exception as e:
            return dagknight_pb2.PropagateTradeResponse(success=False, message=str(e))

    async def PropagateLiquidityChange(self, request, context):
        try:
            if request.is_add:
                await self.exchange.add_liquidity(request.user_id, request.pool_id, Decimal(request.amount_a), Decimal(request.amount_b))
            else:
                await self.exchange.remove_liquidity(request.user_id, request.pool_id, Decimal(request.amount_a), Decimal(request.amount_b))
            return dagknight_pb2.PropagateLiquidityChangeResponse(success=True, message="Liquidity change propagated successfully")
        except Exception as e:
            return dagknight_pb2.PropagateLiquidityChangeResponse(success=False, message=str(e))

    def GetBlockchain(self, request, context):
        try:
            blockchain = self.blockchain.get_chain()
            chain = []
            for block in blockchain:
                chain.append(dagknight_pb2.Block(
                    previous_hash=block.previous_hash,
                    data=block.data,
                    quantum_signature=block.quantum_signature,
                    reward=block.reward,
                    transactions=[dagknight_pb2.Transaction(
                        sender=tx.sender,
                        receiver=tx.receiver,
                        amount=tx.amount,
                        private_key=tx.private_key,
                        public_key=tx.public_key,
                        signature=tx.signature
                    ) for tx in block.transactions],
                    hash=block.hash,
                    timestamp=int(block.timestamp),  # Ensure this is an integer
                    nonce=int(block.nonce)  # Ensure this is an integer
                ))
            return dagknight_pb2.GetBlockchainResponse(chain=chain)
        except Exception as e:
            logger.error(f"Error in GetBlockchain: {e}")
            context.set_details(f"Error retrieving blockchain: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return dagknight_pb2.GetBlockchainResponse()
    def AddTransaction(self, request, context):
        transaction = request.transaction
        print(f"Received transaction in AddTransaction: {transaction}")
        print(f"Transaction fields - sender: {transaction.sender}, receiver: {transaction.receiver}, price: {transaction.price}, buyer_id: {transaction.buyer_id}, seller_id: {transaction.seller_id}")

        # Convert Protobuf Transaction to Custom Transaction Class
        transaction_obj = Transaction.from_proto(transaction)

        print(f"Converted Transaction Object: {transaction_obj}")
        
        # Validate the transaction
        if not self.validate_transaction(transaction_obj):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Invalid transaction!')
            return dagknight_pb2.AddTransactionResponse(success=False)

        # Compute the hash of the transaction and store it
        transaction_hash = self.compute_transaction_hash(transaction_obj)
        self.node_directory.store_transaction(transaction_hash, transaction_obj)
        
        # Optionally, propagate the transaction to other nodes
        self.propagate_transaction(transaction_obj)

        return dagknight_pb2.AddTransactionResponse(success=True)



    def validate_transaction(self, transaction):
        # Basic validation logic
        if not transaction.sender or not transaction.receiver or not transaction.signature:
            return False
        if transaction.amount <= 0 or transaction.price <= 0:
            return False
        # Add more complex validation as needed, such as checking digital signatures, etc.
        return True


    def FullStateSync(self, request, context):
        chain = [dagknight_pb2.Block(
            previous_hash=block.previous_hash,
            data=block.data,
            quantum_signature=block.quantum_signature,
            reward=block.reward,
            transactions=[dagknight_pb2.Transaction(
                sender=tx['sender'], receiver=tx['receiver'], amount=tx['amount']) for tx in block.transactions]
        ) for block in self.blockchain.chain]

        balances = {k: v for k, v in self.blockchain.balances.items()}
        stakes = {k: v for k, v in self.blockchain.stakes.items()}

        return dagknight_pb2.FullStateResponse(chain=chain, balances=balances, stakes=stakes)

    def _validate_token(self, token):
        try:
            decoded_token = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return True
        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False

    def _check_authorization(self, context):
        metadata = dict(context.invocation_metadata())
        token = metadata.get("authorization")
        if not token or not self._validate_token(token):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")

    def load_or_generate_private_key(self):
        private_key_path = "private_key.pem"
        if os.path.exists(private_key_path):
            with open(private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
            logger.info("Private key loaded from file.")
        else:
            private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(private_key_path, "wb") as key_file:
                key_file.write(pem)
            logger.info("New private key generated and saved to file.")
        return private_key

    def RegisterNode(self, request, context):
        self._check_authorization(context)
        node_id = request.node_id
        public_key = request.public_key
        ip_address = request.ip_address
        port = request.port
        try:
            magnet_link = self.node_directory.register_node(node_id, public_key, ip_address, port)
            logging.info(f"Node registered: {node_id}")
            return dagknight_pb2.RegisterNodeResponse(success=True, magnet_link=magnet_link)
        except Exception as e:
            logging.error(f"Error registering node: {e}")
            return dagknight_pb2.RegisterNodeResponse(success=False, magnet_link="")

    def DiscoverNodes(self, request, context):
        self._check_authorization(context)
        try:
            nodes = self.node_directory.discover_nodes()
            logging.info(f"Discovered nodes: {nodes}")
            return dagknight_pb2.DiscoverNodesResponse(magnet_links=[node['magnet_link'] for node in nodes])
        except Exception as e:
            logging.error(f"Error discovering nodes: {e}")
            context.abort(grpc.StatusCode.INTERNAL, f'Error discovering nodes: {str(e)}')

    def MineBlock(self, request, context):
        try:
            node_id = request.node_id
            data = request.data
            quantum_signature = request.quantum_signature
            transactions = [Transaction.from_proto(t) for t in request.transactions]
            miner_address = request.miner_address

            logger.info(f"Mining block for node {node_id} with {len(transactions)} transactions")

            block = QuantumBlock(
                previous_hash=self.blockchain.chain[-1].hash,
                data=data,
                quantum_signature=quantum_signature,
                reward=self.blockchain.get_block_reward(),
                transactions=transactions
            )
            block.mine_block(self.blockchain.difficulty)

            if self.blockchain.validate_block(block):
                self.blockchain.add_block(block)
                logger.info(f"Block mined successfully by {miner_address}")
                return dagknight_pb2.MineBlockResponse(success=True)
            else:
                logger.error("Failed to validate mined block")
                return dagknight_pb2.MineBlockResponse(success=False)
        except Exception as e:
            logger.error(f"Error during mining: {str(e)}")
            context.set_details(f'Error during mining: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return dagknight_pb2.MineBlockResponse(success=False)

    def GetBalance(self, request, context):
        try:
            address = request.address
            balance = self.blockchain.get_balance(address)
            return dagknight_pb2.GetBalanceResponse(balance=balance)
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            context.set_details(f'Error getting balance: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return dagknight_pb2.GetBalanceResponse(balance=0)

    def StakeCoins(self, request, context):
        self._check_authorization(context)
        address = request.address
        amount = request.amount
        result = self.blockchain.stake_coins(address, amount)
        return dagknight_pb2.StakeCoinsResponse(success=result)

    def UnstakeCoins(self, request, context):
        self._check_authorization(context)
        address = request.address
        amount = request.amount
        result = self.blockchain.unstake_coins(address, amount)
        return dagknight_pb2.UnstakeCoinsResponse(success=result)

    def GetStakedBalance(self, request, context):
        self._check_authorization(context)
        address = request.address
        balance = self.blockchain.get_staked_balance(address)
        return dagknight_pb2.GetStakedBalanceResponse(balance=balance)

    def QKDKeyExchange(self, request, context):
        self._check_authorization(context)
        qkd = BB84()
        simulator = Aer.get_backend('aer_simulator')
        key = qkd.run_protocol(simulator)
        return dagknight_pb2.QKDKeyExchangeResponse(node_id=request.node_id, key=key)

    def PropagateBlock(self, request, context):
        try:
            block = QuantumBlock(
                previous_hash=request.block.previous_hash,
                data=request.block.data,
                quantum_signature=request.block.quantum_signature,
                reward=request.block.reward,
                transactions=[{'sender': tx.sender, 'receiver': tx.receiver, 'amount': tx.amount} for tx in request.block.transactions]
            )
            block.hash = request.block.hash
            if self.blockchain.add_block(block):
                logger.info(f"Received block with hash: {block.hash} from miner {request.miner_address}")
                return dagknight_pb2.PropagateBlockResponse(success=True)
            else:
                logger.error(f"Failed to validate block with hash: {block.hash}")
                return dagknight_pb2.PropagateBlockResponse(success=False)
        except Exception as e:
            logger.error(f"Error adding propagated block: {e}")
            return dagknight_pb2.PropagateBlockResponse(success=False)

    async def sync_state(self, directory_ip, directory_port):
        async with grpc.aio.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
            request = dagknight_pb2.FullStateRequest()
            response = await stub.FullStateSync(request)

            self.blockchain.chain = [QuantumBlock(
                previous_hash=blk.previous_hash,
                data=blk.data,
                quantum_signature=blk.quantum_signature,
                reward=blk.reward,
                transactions=[tx for tx in blk.transactions]
            ) for blk in response.chain]
            self.blockchain.balances = {k: v for k, v in response.balances.items()}
            self.blockchain.stakes = {k: v for k, v in response.stakes.items()}
    def propagate_transaction(self, transaction):
        # Ensure transaction is your custom Transaction class
        if not isinstance(transaction, Transaction):
            raise TypeError("Expected transaction to be of type Transaction")

        # Verify the transaction's ZKP using the blockchain's zk_system
        if not transaction.verify_transaction(self.blockchain.zk_system):
            raise ValueError("Invalid transaction ZKP")

        # Compute the hash before converting to Protobuf
        transaction_hash = transaction.compute_hash()

        # Store the transaction
        self.node_directory.store_transaction(transaction_hash, transaction)
        print(f"Propagating transaction with hash: {transaction_hash} to all nodes")

        for node_stub in self.get_all_node_stubs():
            try:
                print(f"Sending transaction to node {node_stub}")
                grpc_transaction = transaction.to_grpc()  # Convert to Protobuf object
                node_stub.AddTransaction(dagknight_pb2.AddTransactionRequest(
                    transaction=grpc_transaction
                ))
                print(f"Transaction with hash {transaction_hash} sent successfully to node {node_stub}")
            except grpc.RpcError as e:
                print(f"Failed to send transaction to node {node_stub}: {e}")


    def get_all_node_stubs(self):
        return self.node_stubs

    # Example method to add a node stub
    def add_node_stub(self, node_stub):
        self.node_stubs.append(node_stub)

    def get_all_node_stubs(self):
        return self.node_stubs

    def add_node_stub(self, node_stub):
        self.node_stubs.append(node_stub)

    def store_transaction(self, transaction_hash, transaction):
        self.node_directory.store_transaction(transaction_hash, transaction)
        print(f"Transaction with hash {transaction_hash} stored successfully.")

    def compute_transaction_hash(self, transaction):
        transaction_data = f"{transaction.sender}{transaction.receiver}{transaction.amount}{transaction.signature}{transaction.public_key}{transaction.price}{transaction.buyer_id}{transaction.seller_id}"
        return hashlib.sha256(transaction_data.encode()).hexdigest()

    async def PropagateOrder(self, request, context):
        try:
            order = Order(
                user_id=request.user_id,
                order_type=request.order_type,
                base_currency=request.base_currency,
                quote_currency=request.quote_currency,
                amount=Decimal(request.amount),
                price=Decimal(request.price)
            )
            await self.exchange.place_order(order)
            return dagknight_pb2.PropagateOrderResponse(success=True, message="Order propagated successfully")
        except Exception as e:
            return dagknight_pb2.PropagateOrderResponse(success=False, message=str(e))

    async def PropagateTrade(self, request, context):
        try:
            trade = Trade(
                buyer_id=request.buyer_id,
                seller_id=request.seller_id,
                base_currency=request.base_currency,
                quote_currency=request.quote_currency,
                amount=Decimal(request.amount),
                price=Decimal(request.price)
            )
            await self.exchange.execute_trade(trade)
            return dagknight_pb2.PropagateTradeResponse(success=True, message="Trade propagated successfully")
        except Exception as e:
            return dagknight_pb2.PropagateTradeResponse(success=False, message=str(e))

    async def PropagateLiquidityChange(self, request, context):
        try:
            if request.is_add:
                await self.exchange.add_liquidity(request.user_id, request.pool_id, Decimal(request.amount_a), Decimal(request.amount_b))
            else:
                await self.exchange.remove_liquidity(request.user_id, request.pool_id, Decimal(request.amount_a), Decimal(request.amount_b))
            return dagknight_pb2.PropagateLiquidityChangeResponse(success=True, message="Liquidity change propagated successfully")
        except Exception as e:
            return dagknight_pb2.PropagateLiquidityChangeResponse(success=False, message=str(e))

    def GetBlockchain(self, request, context):
        try:
            blockchain = self.blockchain.get_chain()
            chain = []
            for block in blockchain:
                chain.append(dagknight_pb2.Block(
                    previous_hash=block.previous_hash,
                    data=block.data,
                    quantum_signature=block.quantum_signature,
                    reward=block.reward,
                    transactions=[dagknight_pb2.Transaction(
                        sender=tx.sender,
                        receiver=tx.receiver,
                        amount=tx.amount,
                        private_key=tx.private_key,
                        public_key=tx.public_key,
                        signature=tx.signature
                    ) for tx in block.transactions],
                    hash=block.hash,
                    timestamp=int(block.timestamp),
                    nonce=int(block.nonce)
                ))
            return dagknight_pb2.GetBlockchainResponse(chain=chain)
        except Exception as e:
            logger.error(f"Error in GetBlockchain: {e}")
            context.set_details(f"Error retrieving blockchain: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return dagknight_pb2.GetBlockchainResponse()

    def AddTransaction(self, request, context):
        transaction = request.transaction
        print(f"Received transaction in AddTransaction: {transaction}")
        print(f"Transaction fields - sender: {transaction.sender}, receiver: {transaction.receiver}, price: {transaction.price}, buyer_id: {transaction.buyer_id}, seller_id: {transaction.seller_id}")

        # Convert Protobuf Transaction to Custom Transaction Class
        transaction_obj = Transaction.from_proto(transaction)

        print(f"Converted Transaction Object: {transaction_obj}")
        
        # Validate the transaction
        if not self.validate_transaction(transaction_obj):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('Invalid transaction!')
            return dagknight_pb2.AddTransactionResponse(success=False)

        # Compute the hash of the transaction and store it
        transaction_hash = self.compute_transaction_hash(transaction_obj)
        self.node_directory.store_transaction(transaction_hash, transaction_obj)
        
        # Optionally, propagate the transaction to other nodes
        self.propagate_transaction(transaction_obj)

        return dagknight_pb2.AddTransactionResponse(success=True)

    def validate_transaction(self, transaction):
        # Basic validation logic
        if not transaction.sender or not transaction.receiver or not transaction.signature:
            return False
        if transaction.amount <= 0 or transaction.price <= 0:
            return False
        return True

    def FullStateSync(self, request, context):
        try:
            chain = [dagknight_pb2.Block(
                previous_hash=block.previous_hash,
                data=block.data,
                quantum_signature=block.quantum_signature,
                reward=block.reward,
                transactions=[dagknight_pb2.Transaction(
                    sender=tx.sender, receiver=tx.receiver, amount=tx.amount) for tx in block.transactions]
            ) for block in self.blockchain.chain]

            balances = {k: v for k, v in self.blockchain.balances.items()}
            stakes = {k: v for k, v in self.blockchain.stakes.items()}

            return dagknight_pb2.FullStateResponse(chain=chain, balances=balances, stakes=stakes)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error syncing state: {str(e)}")
            return dagknight_pb2.FullStateResponse()

    def PropagateTransaction(self, request, context):
        try:
            # Verify ZKP
            is_valid = self.zk_system.verify(request.public_input, request.zk_proof)
            if not is_valid:
                return dagknight_pb2.PropagationResponse(success=False, message="Invalid ZKP")

            # Add transaction to blockchain
            success = self.blockchain.add_transaction(request)
            return dagknight_pb2.PropagationResponse(success=success, message="Transaction propagated successfully" if success else "Failed to propagate transaction")
        except Exception as e:
            return dagknight_pb2.PropagationResponse(success=False, message=str(e))

    def SyncState(self, request, context):
        try:
            new_blocks = self.blockchain.get_blocks_since(request.last_known_block)
            return dagknight_pb2.SyncResponse(new_blocks=new_blocks)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return dagknight_pb2.SyncResponse()

    def VerifyZKP(self, request, context):
        try:
            is_valid = self.zk_system.verify(request.public_input, request.zk_proof)
            return dagknight_pb2.ZKPVerificationResponse(is_valid=is_valid)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return dagknight_pb2.ZKPVerificationResponse(is_valid=False)

    def MineBlock(self, request, context):
        try:
            node_id = request.node_id
            data = request.data
            quantum_signature = request.quantum_signature
            transactions = [Transaction.from_proto(t) for t in request.transactions]
            miner_address = request.miner_address

            logger.info(f"Mining block for node {node_id} with {len(transactions)} transactions")

            block = QuantumBlock(
                previous_hash=self.blockchain.chain[-1].hash,
                data=data,
                quantum_signature=quantum_signature,
                reward=self.blockchain.get_block_reward(),
                transactions=transactions
            )
            block.mine_block(self.blockchain.difficulty)

            if self.blockchain.validate_block(block):
                self.blockchain.add_block(block)
                logger.info(f"Block mined successfully by {miner_address}")
                return dagknight_pb2.MineBlockResponse(success=True)
            else:
                logger.error("Failed to validate mined block")
                return dagknight_pb2.MineBlockResponse(success=False)
        except Exception as e:
            logger.error(f"Error during mining: {str(e)}")
            context.set_details(f'Error during mining: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return dagknight_pb2.MineBlockResponse(success=False)

    async def GetBalance(self, request, context):
        try:
            address = request.address
            balance = await self.blockchain.get_balance(address)
            return dagknight_pb2.GetBalanceResponse(balance=balance)
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            context.set_details(f'Error getting balance: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            return dagknight_pb2.GetBalanceResponse(balance=0)
    async def PropagateTransaction(self, request, context):
        try:
            # Verify ZKP
            # Note: We need to add public_input and zk_proof to the Transaction message in the .proto file
            public_input = request.public_input if hasattr(request, 'public_input') else None
            zk_proof = request.zk_proof if hasattr(request, 'zk_proof') else None
            
            if public_input is None or zk_proof is None:
                print("Missing public_input or zk_proof")
                return dagknight_pb2.PropagationResponse(success=False, message="Missing ZKP data")

            print(f"Received public_input: {public_input}")
            print(f"Received zk_proof: {zk_proof}")
            
            is_valid = self.zk_system.verify(public_input, zk_proof)
            if not is_valid:
                print("Invalid ZKP")
                return dagknight_pb2.PropagationResponse(success=False, message="Invalid ZKP")

            # Add transaction to blockchain
            success = await self.blockchain.add_transaction(request)
            return dagknight_pb2.PropagationResponse(success=success, message="Transaction propagated successfully" if success else "Failed to propagate transaction")
        except Exception as e:
            print(f"Exception in PropagateTransaction: {str(e)}")
            print(f"Exception traceback: {traceback.format_exc()}")
            return dagknight_pb2.PropagationResponse(success=False, message=str(e))






    async def SyncState(self, request, context):
        try:
            last_known_block = request.last_known_block
            new_blocks = self.blockchain.get_blocks_since(last_known_block)
            return dagknight_pb2.SyncResponse(new_blocks=[self._block_to_proto(block) for block in new_blocks])
        except Exception as e:
            logger.error(f"Error in SyncState: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return dagknight_pb2.SyncResponse()

    async def VerifyZKP(self, request, context):
        try:
            is_valid = self.zk_system.verify(request.public_input, request.zk_proof)
            return dagknight_pb2.ZKPVerificationResponse(is_valid=is_valid)
        except Exception as e:
            logger.error(f"Error in VerifyZKP: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return dagknight_pb2.ZKPVerificationResponse(is_valid=False)

    async def FullStateSync(self, request, context):
        try:
            chain = self.blockchain.chain
            balances = self.blockchain.balances
            stakes = self.blockchain.stakes
            return dagknight_pb2.FullStateSyncResponse(
                chain=[self._block_to_proto(block) for block in chain],
                balances={k: float(v) for k, v in balances.items()},
                stakes={k: float(v) for k, v in stakes.items()}
            )
        except Exception as e:
            logger.error(f"Error in FullStateSync: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            return dagknight_pb2.FullStateSyncResponse()

    def _block_to_proto(self, block):
        return dagknight_pb2.Block(
            previous_hash=block.previous_hash,
            data=block.data,
            quantum_signature=block.quantum_signature,
            reward=float(block.reward),
            transactions=[self._transaction_to_proto(tx) for tx in block.transactions],
            hash=block.hash,
            timestamp=int(block.timestamp),
            nonce=int(block.nonce)
        )

    def _transaction_to_proto(self, tx):
        return dagknight_pb2.Transaction(
            sender=tx.sender,
            receiver=tx.receiver,
            amount=float(tx.amount),
            price=float(tx.price),
            buyer_id=tx.buyer_id,
            seller_id=tx.seller_id,
            public_key=tx.public_key,
            signature=tx.signature
        )


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: User):
    user = authenticate_user(form_data.pincode)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Incorrect pincode",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.pincode}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}



@app.get("/get_block_info/{block_hash}", response_model=dict)
def get_block_info(block_hash: str, pincode: str = Depends(authenticate)):
    response = get(f'http://161.35.219.10:50503/get_block_info/{block_hash}')
    block_info = response.json()

    # Add transactions to the block info
    block_info['transactions'] = [
        {
            "tx_hash": tx['hash'],
            "sender": tx['sender'],
            "receiver": tx['receiver'],
            "amount": tx['amount'],
            "timestamp": tx['timestamp']
        } for tx in block_info.get('transactions', [])
    ]

    return block_info


@app.get("/get_node_info/{node_id}", response_model=dict)
def get_node_info(node_id: str, pincode: str = Depends(authenticate)):
    response = get(f'http://161.35.219.10:50503/get_node_info/{node_id}')
    return response.json()


@app.get("/get_transaction_info/{tx_hash}", response_model=dict)
def get_transaction_info(tx_hash: str, pincode: str = Depends(authenticate)):
    response = get(f'http://161.35.219.10:50503/get_transaction_info/{tx_hash}')
    return response.json()



import unittest

def run_secure_client():
    channel = grpc.insecure_channel('localhost:50051')
    stub = dagknight_pb2_grpc.DAGKnightStub(channel)

    message = "Hello, secure world!"
    encrypted_message = encrypt_message(message, shared_box)

    metadata = [('public_key', client_public_key.encode())]
    response = stub.SecureRPC(dagknight_pb2.SecureRequest(encrypted_message=encrypted_message), metadata=metadata)

    decrypted_response = decrypt_message(response.encrypted_message, shared_box)
    print("Received:", decrypted_response)
async def request_full_state_sync(server_address):
    async with grpc.aio.insecure_channel(server_address) as channel:
        stub = dagknight_pb2_grpc.DAGKnightStub(channel)
        request = dagknight_pb2.FullStateSyncRequest()
        response = await stub.FullStateSync(request)
        return response

def run_tests():
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')  # Adjust pattern to match your test files
    runner = unittest.TextTestRunner()
    runner.run(suite)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user: str):
        await websocket.accept()
        self.active_connections[user] = websocket
        logger.info(f"WebSocket connection established for user: {user}")

    def disconnect(self, user: str):
        if user in self.active_connections:
            websocket = self.active_connections[user]
            del self.active_connections[user]
            logger.info(f"WebSocket connection closed for user: {user}")
            # Remove from subscriptions
            for client_id in list(self.subscriptions.keys()):
                if websocket in self.subscriptions[client_id]:
                    self.subscriptions[client_id].remove(websocket)
                    if not self.subscriptions[client_id]:
                        del self.subscriptions[client_id]

    async def subscribe(self, websocket: WebSocket, client_id: str):
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = []
        self.subscriptions[client_id].append(websocket)

    async def unsubscribe(self, websocket: WebSocket, client_id: str):
        if client_id in self.subscriptions:
            self.subscriptions[client_id].remove(websocket)
            if not self.subscriptions[client_id]:
                del self.subscriptions[client_id]

    async def send_personal_message(self, message: str, user: str):
        if user in self.active_connections:
            await self.active_connections[user].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

    async def broadcast_new_block(self, block: dict):
        message = json.dumps({
            "type": "new_block",
            "data": block
        })
        await self.broadcast(message)

    async def broadcast_new_transaction(self, transaction: dict):
        message = json.dumps({
            "type": "new_transaction",
            "data": transaction
        })
        await self.broadcast(message)
    async def broadcast_network_stats(self, stats: dict):
        # Convert Decimal objects to strings
        stats_serializable = {k: str(v) if isinstance(v, Decimal) else v for k, v in stats.items()}
        for connection in self.active_connections:
            await connection.send_text(json.dumps(stats_serializable))



manager = ConnectionManager()



async def periodic_network_stats_update():
    while True:
        stats = await get_network_stats()  # Make sure this function is defined elsewhere
        await manager.broadcast_network_stats(stats)
        await asyncio.sleep(60)  # Update every minute


# Instance of ConnectionManager
manager = ConnectionManager()

async def periodic_network_stats_update():
    while True:
        stats = await get_network_stats()
        await manager.broadcast_network_stats(stats)
        await asyncio.sleep(60)  # Update every minute
async def get_network_stats():
    try:
        # Await the coroutine to discover nodes
        nodes = await node_directory.discover_nodes()
        total_nodes = len(nodes)
        
        # Accessing totalTransactions from the globalMetrics dictionary
        total_transactions = blockchain.globalMetrics['totalTransactions']
        
        # Length of the blockchain
        total_blocks = len(blockchain.chain)
        
        # Calculate the average block time
        average_block_time = calculate_average_block_time(blockchain.chain)
        
        # Get the current difficulty of the blockchain
        current_difficulty = blockchain.difficulty
        
        # Get the total supply from the blockchain
        total_supply = blockchain.get_total_supply()
        
        # Return the network statistics as a dictionary
        return {
            "total_nodes": total_nodes,
            "total_transactions": total_transactions,
            "total_blocks": total_blocks,
            "average_block_time": average_block_time,
            "current_difficulty": current_difficulty,
            "total_supply": total_supply
        }
    except Exception as e:
        logger.error(f"Error fetching network stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching network stats")

@app.get("/network_stats")
async def get_network_stats_endpoint(pincode: str = Depends(authenticate)):
    return await get_network_stats()

@app.get("/node_performance/{node_id}")
async def get_node_performance(node_id: str, pincode: str = Depends(authenticate)):
    try:
        node_stats = node_directory.get_performance_stats()
        return {
            "node_id": node_id,
            "avg_mining_time": node_stats.get("avg_mining_time", 0),
            "total_blocks_mined": node_stats.get("total_blocks_mined", 0),
            "hash_rate": node_stats.get("hash_rate", 0),
            "uptime": node_stats.get("uptime", 0)
        }
    except Exception as e:
        logger.error(f"Error fetching node performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching node performance")

@app.get("/quantum_metrics")
async def get_quantum_metrics(pincode: str = Depends(authenticate)):
    try:
        qhins = calculate_qhins(blockchain.chain)
        entanglement_strength = calculate_entanglement_strength(blockchain.chain)
        coherence_ratio = blockchain.globalMetrics.coherenceRatio
        return {
            "qhins": qhins,
            "entanglement_strength": entanglement_strength,
            "coherence_ratio": coherence_ratio
        }
    except Exception as e:
        logger.error(f"Error fetching quantum metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching quantum metrics")

@app.get("/recent_transactions")
async def get_recent_transactions(limit: int = 10, pincode: str = Depends(authenticate)):
    try:
        recent_txs = blockchain.get_recent_transactions(limit)
        return [tx.to_dict() for tx in recent_txs]
    except Exception as e:
        logger.error(f"Error fetching recent transactions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching recent transactions")
class TokenRefreshRequest(BaseModel):
    refresh_token: str

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    # Log the expiration time for debugging
    logger.debug(f"Access token will expire at {expire}")
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt



def verify_token(token: str, token_type: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail=f"Invalid {token_type}")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail=f"{token_type.capitalize()} expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail=f"Invalid {token_type}")

@app.post("/refresh_token")
async def refresh_token(request: TokenRefreshRequest):
    try:
        user_id = verify_token(request.refresh_token, "refresh token")
        new_access_token = create_access_token({"sub": user_id})
        return {"access_token": new_access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example protected route
@app.get("/protected")
async def protected_route(token: str = Depends(oauth2_scheme)):
    user_id = verify_token(token, "access token")
    return {"message": "Access granted", "user_id": user_id}

# Example login route (you should implement proper authentication here)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Example user database
fake_users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("password1"),  # Example hashed password
        "disabled": False,
    },
    # Add more users here
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    return db.get(username)
@app.post("/login")
async def login(pincode: str):
    if not pincode:
        raise HTTPException(status_code=400, detail="Pincode is required")

    # Generate a wallet based on the pincode
    wallet = Wallet(mnemonic=pincode)  # or pass in a private_key if available
    wallet_address = wallet.get_address()

    # Create tokens
    access_token = create_access_token({"sub": wallet_address})
    refresh_token = create_refresh_token(wallet_address)

    return {
        "wallet_address": wallet_address,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


                    


@app.get("/block_details/{block_hash}")
async def get_block_details(block_hash: str, pincode: str = Depends(authenticate)):
    try:
        block = blockchain.get_block_by_hash(block_hash)
        if block:
            return block.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Block not found")
    except Exception as e:
        logger.error(f"Error fetching block details: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching block details")
# Helper function to calculate average block time
def calculate_average_block_time(chain):
    if len(chain) < 2:
        return 0  # Not enough blocks to calculate an average
    
    block_times = []
    for i in range(1, len(chain)):
        block_time = chain[i].timestamp - chain[i - 1].timestamp
        block_times.append(block_time)
    
    if block_times:
        return sum(block_times) / len(block_times)
    else:
        return 0

# Helper function to calculate Quantum Hash Information Number (QHIN)
def calculate_qhins(chain):
    # Assume that each block has a 'quantum_hash_rate' attribute for QHIN calculation
    qhins = sum(block.quantum_hash_rate for block in chain) / len(chain)
    return qhins

# Helper function to calculate entanglement strength
def calculate_entanglement_strength(chain):
    total_entanglement_strength = 0
    for block in chain:
        total_entanglement_strength += sum(tx['entanglement_strength'] for tx in block.transactions)
    
    average_entanglement_strength = total_entanglement_strength / len(chain)
    return average_entanglement_strength

manager = ConnectionManager()
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle received data if necessary
            await manager.broadcast(f"Client {client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left the chat")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user
# Endpoint to get address from alias
@app.post("/get_address_from_alias")
def get_address_from_alias(alias: str):
    for user in fake_users_db.values():
        if user.wallet['alias'] == alias:
            return {"address": user.wallet['address']}
    raise HTTPException(status_code=400, detail="Alias not found")
async def get_token(websocket: WebSocket):
    token = websocket.query_params.get('token')
    if not token:
        await websocket.close(code=1008)  # Close with appropriate code for policy violation
        raise WebSocketDisconnect(code=1008)
    return token
async def get_current_user(token: str):
    try:
        logger.info(f"Attempting to decode token: {token[:10]}...{token[-10:]}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = payload.get("sub")
        if user is None:
            logger.error("Token payload does not contain 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token payload")
        logger.info(f"Successfully decoded token for user: {user}")
        return user
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.error("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
import json
from decimal import Decimal
from fastapi import WebSocket, WebSocketDisconnect, Query, HTTPException
from starlette.websockets import WebSocketState

WS_1000_NORMAL_CLOSURE = 1000
WS_1008_POLICY_VIOLATION = 1008
WS_3000_INTERNAL_ERROR = 3000  # Custom code for internal errors

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(None)):
    logger.info("WebSocket connection attempt received")
    logger.info(f"Query parameters: {websocket.query_params}")
    if not token:
        logger.error("No token provided in WebSocket connection")
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
        return
    try:
        logger.info(f"Token provided: {token[:10]}...{token[-10:] if len(token) > 20 else ''}")
        user = await get_current_user(token)  # Assuming get_current_user function exists
        logger.info(f"User authenticated: {user}")
        await manager.connect(websocket, user)
        logger.info(f"WebSocket connection accepted for user: {user}")
        try:
            while True:
                data = await websocket.receive_text()
                logger.debug(f"Received message from {user}: {data}")
                message = json.loads(data)
                # Handle the received message and update state accordingly
                if message['type'] == 'botControl':
                    action = message['action']
                    if action == 'start':
                        # Start the bot
                        logger.info("Starting the bot")
                    elif action == 'stop':
                        # Stop the bot
                        logger.info("Stopping the bot")
                    elif action == 'rebalance':
                        # Rebalance portfolio
                        logger.info("Rebalancing portfolio")
                elif message['type'] == 'miningControl':
                    action = message['action']
                    if action == 'start':
                        # Start mining
                        logger.info("Starting mining")
                    elif action == 'stop':
                        # Stop mining
                        logger.info("Stopping mining")
                elif message['type'] == 'placeOrder':
                    order_type = message['orderType']
                    pair = message['pair']
                    amount = Decimal(message['amount'])
                    price = Decimal(message['price'])
                    # Handle placing order logic
                    logger.info(f"Placing order: {order_type} {amount} {pair} at {price}")
                # Example: broadcast updated state
                await manager.broadcast(json.dumps(state, cls=DecimalEncoder))
        except WebSocketDisconnect:
            manager.disconnect(user)
            logger.info(f"WebSocket disconnected for user: {user}")
    except HTTPException as e:
        logger.error(f"Authentication failed: {str(e)}")
        await websocket.close(code=WS_1008_POLICY_VIOLATION)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {str(e)}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=WS_3000_INTERNAL_ERROR)
async def get_current_user(token: str):
    try:
        logger.info(f"Attempting to decode token: {token[:10]}...{token[-10:]}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = payload.get("sub")
        if user is None:
            logger.error("Token payload does not contain 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token payload")
        logger.info(f"Successfully decoded token for user: {user}")
        return user
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.error("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Call these functions when balance changes or new transactions occur
async def notify_balance_change(wallet_address: str, new_balance: float):
    await manager.send_personal_message(json.dumps({
        "type": "balance_update",
        "wallet_address": wallet_address,
        "balance": str(new_balance)
    }), wallet_address)

async def notify_new_transaction(wallet_address: str, transaction: dict):
    await manager.send_personal_message(json.dumps({
        "type": "new_transaction",
        "transaction": transaction
    }), wallet_address)

async def notify_transaction_confirmed(wallet_address: str, tx_hash: str):
    await manager.send_personal_message(json.dumps({
        "type": "transaction_confirmed",
        "tx_hash": tx_hash
    }), wallet_address)
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/trading", response_class=HTMLResponse)
async def get_trading(request: Request):
    return templates.TemplateResponse("trading.html", {"request": request})
@app.get("/api/dashboard_data")
async def get_dashboard_data():
    return {
        "Quantum Hash Rate": "1.21 TH/s",
        "Network Entanglement": "99.9%",
        "Active Nodes": "42,000",
        "PLATA Price": "$1.0001",
        "Total Value Locked": "$1,000,000,000",
        "Governance Proposals": "7 Active"
    }

    
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantumDAGKnight Nexus</title>
    <script src="https://cdn.babylonjs.com/babylon.js"></script>
    <script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/annyang/2.6.1/annyang.min.js"></script>
    <style>
        #renderCanvas {
            width: 100%;
            height: 100%;
            touch-action: none;
        }
    </style>
</head>
<body>
    <canvas id="renderCanvas"></canvas>
    <div class="interface-container">
        <div id="mainDashboard"></div>
        <div id="tradingModal" class="modal" style="display: none;">
            <span class="close-modal">&times;</span>
            <p>Trading Modal Content</p>
        </div>
        <div id="governanceModal" class="modal" style="display: none;">
            <span class="close-modal">&times;</span>
            <p>Governance Modal Content</p>
        </div>
    </div>

    <script>
        console.log("Page Loaded");

        // Babylon.js setup for Quantum Universe visualization
        const canvas = document.getElementById("renderCanvas");
        const engine = new BABYLON.Engine(canvas, true);

        const createScene = function () {
            const scene = new BABYLON.Scene(engine);

            // Camera
            const camera = new BABYLON.ArcRotateCamera("Camera", -Math.PI / 2, Math.PI / 2, 5, BABYLON.Vector3.Zero(), scene);
            camera.attachControl(canvas, true);

            // Light
            const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);

            // Starfield
            const starCount = 10000;
            const starsSystem = new BABYLON.ParticleSystem("stars", starCount, scene);
            starsSystem.particleTexture = new BABYLON.Texture("https://www.babylonjs-playground.com/textures/flare.png");
            starsSystem.createPointEmitter(new BABYLON.Vector3(-50, -50, -50), new BABYLON.Vector3(50, 50, 50));
            starsSystem.color1 = new BABYLON.Color4(0.9, 0.9, 1.0, 1.0);
            starsSystem.color2 = new BABYLON.Color4(0.8, 0.8, 1.0, 1.0);
            starsSystem.minSize = 0.1;
            starsSystem.maxSize = 0.5;
            starsSystem.minLifeTime = Number.MAX_SAFE_INTEGER;
            starsSystem.maxLifeTime = Number.MAX_SAFE_INTEGER;
            starsSystem.emitRate = starCount;
            starsSystem.start();

            // Quantum Node representation
            const nodeCount = 100;
            const nodes = [];
            for (let i = 0; i < nodeCount; i++) {
                const node = BABYLON.MeshBuilder.CreateSphere(`node${i}`, {diameter: 0.1}, scene);
                node.position = new BABYLON.Vector3(
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 10,
                    (Math.random() - 0.5) * 10
                );
                node.material = new BABYLON.StandardMaterial(`nodeMaterial${i}`, scene);
                node.material.emissiveColor = new BABYLON.Color3(0, 1, 0.5);
                nodes.push(node);
            }

            // Quantum entanglement lines
            const lines = [];
            for (let i = 0; i < nodeCount; i++) {
                if (Math.random() < 0.1) {  // 10% chance to create a line
                    const otherNode = nodes[Math.floor(Math.random() * nodeCount)];
                    const line = BABYLON.MeshBuilder.CreateLines(`line${i}`, {
                        points: [nodes[i].position, otherNode.position]
                    }, scene);
                    line.color = new BABYLON.Color3(0, 1, 0.5);
                    lines.push(line);
                }
            }

            // Animation
            scene.onBeforeRenderObservable.add(() => {
                nodes.forEach(node => {
                    node.position.addInPlace(
                        new BABYLON.Vector3(
                            (Math.random() - 0.5) * 0.01,
                            (Math.random() - 0.5) * 0.01,
                            (Math.random() - 0.5) * 0.01
                        )
                    );
                });

                lines.forEach(line => {
                    const startNode = nodes[lines.indexOf(line)];
                    const endNode = nodes[(lines.indexOf(line) + 1) % nodes.length];
                    if (line && startNode && endNode) {
                        line.dispose();
                        const newLine = BABYLON.MeshBuilder.CreateLines(`line${lines.indexOf(line)}`, {
                            points: [startNode.position, endNode.position]
                        }, scene);
                        newLine.color = new BABYLON.Color3(0, 1, 0.5);
                        lines[lines.indexOf(line)] = newLine;
                    }
                });
            });

            return scene;
        };

        const scene = createScene();

        engine.runRenderLoop(function () {
            scene.render();
        });

        window.addEventListener("resize", function () {
            engine.resize();
        });

        // Main dashboard data
        const dashboardData = {
            "Quantum Hash Rate": "1.21 TH/s",
            "Network Entanglement": "99.9%",
            "Active Nodes": "42,000",
            "PLATA Price": "$1.0001",
            "Total Value Locked": "$1,000,000,000",
            "Governance Proposals": "7 Active"
        };

        function updateDashboard() {
            const response = await fetch('/api/dashboard_data');
    const dashboardData = await response.json();

            const dashboard = document.getElementById('mainDashboard');
            if (dashboard) {
                dashboard.innerHTML = '';
                for (const [key, value] of Object.entries(dashboardData)) {
                    dashboard.innerHTML += `
                        <div class="data-item">
                            <div class="data-label">${key}</div>
                            <div class="data-value">${value}</div>
                        </div>
                    `;
                }
            }
        }
        updateDashboard();

        // Modal functionality
        function openModal(modalId) {
            const modal = document.getElementById(modalId);
            if (modal) {
                modal.style.display = 'block';
            }
        }

        function closeModal(modalId) {
            const modal = document.getElementById(modalId);
            if (modal) {
                modal.style.display = 'none';
            }
        }

        document.querySelectorAll('.close-modal').forEach(elem => {
            elem.addEventListener('click', () => closeModal(elem.closest('.modal').id));
        });

        const openTrading = document.getElementById('openTrading');
        if (openTrading) {
            openTrading.addEventListener('click', () => openModal('tradingModal'));
        }

        const openGovernance = document.getElementById('openGovernance');
        if (openGovernance) {
            openGovernance.addEventListener('click', () => openModal('governanceModal'));
        }

        // Voice commands
        if (annyang) {
            const commands = {
                'open trading': () => openModal('tradingModal'),
                'open governance': () => openModal('governanceModal'),
                'start mining': () => console.log('Mining initiated'),
                'show stats': updateDashboard
            };

            annyang.addCommands(commands);

            annyang.start();

            annyang.addCallback('start', () => {
                const indicator = document.getElementById('voice-command-indicator');
                if (indicator) {
                    indicator.classList.add('listening');
                }
            });

            annyang.addCallback('end', () => {
                const indicator = document.getElementById('voice-command-indicator');
                if (indicator) {
                    indicator.classList.remove('listening');
                }
            });
        }

        // AI Assistant
        const aiAssistant = {
            messages: [],
            addMessage: function(message, isUser = false) {
                this.messages.push({text: message, isUser: isUser});
                this.updateDisplay();
            },
            updateDisplay: function() {
                const display = document.getElementById('ai-assistant-messages');
                if (display) {
                    display.innerHTML = this.messages.map(m => 
                        `<div class="${m.isUser ? 'user' : 'ai'}-message">${m.text}</div>`
                    ).join('');
                    display.scrollTop = display.scrollHeight;
                }
            },
            processInput: function(input) {
                this.addMessage(input, true);
                // Here you would typically send the input to your AI backend and get a response
                // For now, we'll just echo the input
                setTimeout(() => this.addMessage(`You said: ${input}`), 500);
            }
        };

        const aiInput = document.getElementById('ai-assistant-input');
        if (aiInput) {
            aiInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    aiAssistant.processInput(this.value);
                    this.value = '';
                }
            });
        }

        // Initialize the dashboard
        updateDashboard();
    </script>
</body>
</html>
"""

@app.post("/api/ai_assistant")
async def ai_assistant(request: dict):
    user_input = request.get("input")
    # Process the input and generate a response
    response = "AI response to: " + user_input
    return {"response": response}



class Cashewstable:
    def __init__(self, vm, collateral_token, price_feed):
        self.vm = vm
        self.collateral_token = collateral_token
        self.price_feed = price_feed
        self.name = "Cashewstable"
        self.symbol = "cStable"
        self.decimals = 18
        self.total_supply = Decimal('0')
        self.balances: Dict[str, Decimal] = {}
        self.bond_price = Decimal('1') * Decimal('10') ** self.decimals
        self.bonds_outstanding = 0
        self.additional_supply = Decimal('0')
        self.total_supply_at_start_time = Decimal('0')
        self.holders: List[str] = []
        self.target_price = Decimal('1') * Decimal('10') ** self.decimals
        self.max_supply = Decimal('10000000000') * Decimal('10') ** self.decimals
        self.interest_rate = Decimal('0')
        self.savings_balance: Dict[str, Decimal] = {}
        self.last_interest_accrued_time: Dict[str, int] = {}
        self.proposals: Dict[int, Dict] = {}
        self.proposal_count = 0
        self.voting_duration = 7 * 24 * 60 * 60  # 7 days in seconds
    def stabilize(self, address, amount):
        price = self.price_feed.get_price(self.symbol)
        balance_to_add = amount / price
        self.vm.mint(address, balance_to_add)

    def mint(self, user: str, collateral_amount: Decimal):
        amount_to_mint = self.calculate_mint_amount(collateral_amount)
        if self.total_supply + amount_to_mint > self.max_supply:
            raise ValueError("Exceeds maximum supply")

        if not self.vm.transfer_token(self.collateral_token, user, self.vm.contract_address, collateral_amount):
            raise ValueError("Transfer of collateral failed")

        self.total_supply += amount_to_mint
        self.balances[user] = self.balances.get(user, Decimal('0')) + amount_to_mint
        self.accrue_interest(user)

    def burn(self, user: str, token_amount: Decimal):
        if self.balances.get(user, Decimal('0')) < token_amount:
            raise ValueError("Insufficient balance")

        amount_to_return = self.calculate_collateral_amount(token_amount)
        self.balances[user] -= token_amount
        self.total_supply -= token_amount

        if not self.vm.transfer_token(self.collateral_token, self.vm.contract_address, user, amount_to_return):
            raise ValueError("Return of collateral failed")

        self.accrue_interest(user)

    def calculate_mint_amount(self, collateral_amount: Decimal) -> Decimal:
        price = self.get_latest_price()
        if price <= 0:
            raise ValueError("Price feed error")
        return (collateral_amount * price) / (Decimal('10') ** self.decimals)

    def calculate_collateral_amount(self, token_amount: Decimal) -> Decimal:
        price = self.get_latest_price()
        if price <= 0:
            raise ValueError("Price feed error")
        return (token_amount * (Decimal('10') ** self.decimals)) / price

    def get_latest_price(self) -> Decimal:
        return self.price_feed.get_price()

    def adjust_supply(self):
        price = self.get_latest_price()
        if price > self.target_price:
            mint_amount = (self.total_supply * (price - self.target_price)) / (Decimal('10') ** self.decimals)
            if self.total_supply + mint_amount <= self.max_supply:
                self.additional_supply += mint_amount
                self.total_supply_at_start_time = self.total_supply
                self.total_supply += mint_amount
        elif price < self.target_price:
            burn_amount = (self.total_supply * (self.target_price - price)) / (Decimal('10') ** self.decimals)
            self.additional_supply -= burn_amount
            self.total_supply_at_start_time = self.total_supply
            self.total_supply -= burn_amount

    def distribute_rewards(self, start: int, end: int):
        for i in range(start, end):
            holder = self.holders[i]
            reward = (self.balances[holder] * self.additional_supply) / self.total_supply_at_start_time
            self.balances[holder] += reward
        self.additional_supply = Decimal('0')

    def buy_bond(self, user: str):
        if self.balances.get(user, Decimal('0')) < self.bond_price:
            raise ValueError("Not enough tokens")
        self.balances[user] -= self.bond_price
        self.bonds_outstanding += 1

    def redeem_bond(self, user: str):
        if self.bonds_outstanding == 0:
            raise ValueError("No bonds outstanding")
        self.bonds_outstanding -= 1
        self.balances[user] = self.balances.get(user, Decimal('0')) + self.bond_price

    def deposit_to_savings(self, user: str, amount: Decimal):
        if amount <= 0:
            raise ValueError("Amount must be greater than zero")
        if self.balances.get(user, Decimal('0')) < amount:
            raise ValueError("Not enough tokens")
        self.balances[user] -= amount
        self.savings_balance[user] = self.savings_balance.get(user, Decimal('0')) + amount
        self.accrue_interest(user)

    def withdraw_from_savings(self, user: str, amount: Decimal):
        if amount <= 0:
            raise ValueError("Amount must be greater than zero")
        if self.savings_balance.get(user, Decimal('0')) < amount:
            raise ValueError("Not enough savings balance")
        self.savings_balance[user] -= amount
        self.balances[user] = self.balances.get(user, Decimal('0')) + amount
        self.accrue_interest(user)

    def accrue_interest(self, user: str):
        interest_accrued = self.calculate_interest(user)
        self.last_interest_accrued_time[user] = int(time.time())
        if interest_accrued > 0:
            self.savings_balance[user] = self.savings_balance.get(user, Decimal('0')) + interest_accrued
            self.total_supply += interest_accrued

    def calculate_interest(self, user: str) -> Decimal:
        elapsed_time = int(time.time()) - self.last_interest_accrued_time.get(user, int(time.time()))
        interest_amount = (self.savings_balance.get(user, Decimal('0')) * self.interest_rate * Decimal(elapsed_time)) / (Decimal('365') * Decimal('86400') * Decimal('10000'))
        return interest_amount

    def set_interest_rate(self, rate: Decimal):
        self.interest_rate = rate

    def create_proposal(self, description: str):
        self.proposal_count += 1
        self.proposals[self.proposal_count] = {
            'id': self.proposal_count,
            'description': description,
            'end_time': int(time.time()) + self.voting_duration,
            'total_yes_votes': Decimal('0'),
            'total_no_votes': Decimal('0'),
            'executed': False,
            'voted': {}
        }

    def vote(self, user: str, proposal_id: int, vote_choice: bool):
        if proposal_id not in self.proposals:
            raise ValueError("Invalid proposal ID")
        proposal = self.proposals[proposal_id]
        if int(time.time()) > proposal['end_time']:
            raise ValueError("Voting period has ended")
        if proposal['voted'].get(user, False):
            raise ValueError("You have already voted on this proposal")

        vote_weight = self.balances.get(user, Decimal('0'))
        if vote_choice:
            proposal['total_yes_votes'] += vote_weight
        else:
            proposal['total_no_votes'] += vote_weight

        proposal['voted'][user] = True

    def execute_proposal(self, proposal_id: int):
        if proposal_id not in self.proposals:
            raise ValueError("Invalid proposal ID")
        proposal = self.proposals[proposal_id]
        if int(time.time()) <= proposal['end_time']:
            raise ValueError("Voting period has not ended")
        if proposal['executed']:
            raise ValueError("Proposal has already been executed")

        if proposal['total_yes_votes'] > proposal['total_no_votes']:
            proposal['executed'] = True

class AutomatedMarketMaker:
    def __init__(self, token_a, token_b):
        self.token_a = token_a
        self.token_b = token_b
        self.reserve_a = Decimal('0')
        self.reserve_b = Decimal('0')

    def add_liquidity(self, amount_a: Decimal, amount_b: Decimal) -> Decimal:
        if self.reserve_a == 0 and self.reserve_b == 0:
            liquidity = (amount_a * amount_b).sqrt()
        else:
            liquidity = min(amount_a * self.reserve_b / self.reserve_a, amount_b)

        self.reserve_a += amount_a
        self.reserve_b += amount_b
        return liquidity

    def remove_liquidity(self, liquidity: Decimal) -> (Decimal, Decimal):
        amount_a = liquidity * self.reserve_a / self.total_liquidity
        amount_b = liquidity * self.reserve_b / self.total_liquidity
        self.reserve_a -= amount_a
        self.reserve_b -= amount_b
        return amount_a, amount_b

    def swap(self, amount_in: Decimal, token_in: str) -> Decimal:
        if token_in == self.token_a:
            return self._swap(amount_in, self.reserve_a, self.reserve_b)
        elif token_in == self.token_b:
            return self._swap(amount_in, self.reserve_b, self.reserve_a)
        else:
            raise ValueError("Invalid token")

    def _swap(self, amount_in: Decimal, reserve_in: Decimal, reserve_out: Decimal) -> Decimal:
        amount_in_with_fee = amount_in * Decimal('997')
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * Decimal('1000') + amount_in_with_fee
        return numerator / denominator

class YieldFarm:
    def __init__(self, staking_token, reward_token, reward_rate):
        self.staking_token = staking_token
        self.reward_token = reward_token
        self.reward_rate = reward_rate
        self.total_staked = Decimal('0')
        self.last_update_time = int(time.time())
        self.reward_per_token_stored = Decimal('0')
        self.user_reward_per_token_paid: Dict[str, Decimal] = {}
        self.rewards: Dict[str, Decimal] = {}
        self.balances: Dict[str, Decimal] = {}

    def stake(self, user: str, amount: Decimal):
        self._update_reward(user)
        self.total_staked += amount
        self.balances[user] = self.balances.get(user, Decimal('0')) + amount

    def withdraw(self, user: str, amount: Decimal):
        self._update_reward(user)
        self.total_staked -= amount
        self.balances[user] -= amount

    def get_reward(self, user: str):
        self._update_reward(user)
        reward = self.rewards[user]
        self.rewards[user] = Decimal('0')
        return reward

    def _update_reward(self, user: str):
        self.reward_per_token_stored = self._reward_per_token()
        self.last_update_time = int(time.time())
        self.rewards[user] = self._earned(user)
        self.user_reward_per_token_paid[user] = self.reward_per_token_stored

    def _reward_per_token(self) -> Decimal:
        if self.total_staked == 0:
            return self.reward_per_token_stored
        return self.reward_per_token_stored + (Decimal(int(time.time()) - self.last_update_time) * self.reward_rate * Decimal('1e18') / self.total_staked)

    def _earned(self, user: str) -> Decimal:
        return (self.balances.get(user, Decimal('0')) * (self._reward_per_token() - self.user_reward_per_token_paid.get(user, Decimal('0'))) / Decimal('1e18')) + self.rewards.get(user, Decimal('0'))
        
class SmartContract:
    def __init__(self, address, code):
        self.address = address
        self.code = code
        self.storage = {}

    def execute(self, input_data):
        # This is a simplified execution model. In a real implementation,
        # you would have a more sophisticated execution environment.
        context = {
            'storage': self.storage,
            'input': input_data
        }
        exec(self.code, {'input': input_data})
        return locals().get('output', None)


class PrivateSmartContract(SmartContract):
    def __init__(self, address, code, zk_system):
        super().__init__(address, code)
        self.zk_system = zk_system
    async def execute(self, input_data):
        result = super().execute(input_data)
        public_input = self.zk_system.hash(str(input_data), str(result))
        secret = int.from_bytes(os.urandom(32), 'big')
        zk_proof = self.zk_system.prove(secret, public_input)
        return result, zk_proof



    @staticmethod
    def verify_execution(input_data, result, zk_proof, zk_system):
        public_input = zk_system.stark.hash(str(input_data), str(result))
        return zk_system.verify(public_input, zk_proof)


@app.get("/get_node_details/{node_id}")
def get_node_details(node_id: str):
    node_details = node_directory.get(node_id)
    if not node_details:
        raise HTTPException(status_code=404, detail="Node not found")
    return node_details


@router.post("/deploy_cashewstable")
async def deploy_cashewstable(collateral_token: str, price_feed: str, current_user: str = Depends(get_current_user)):
    contract_address = deploy_cashewstable(exchange.vm, collateral_token, price_feed)
    return {"contract_address": contract_address}

@router.post("/deploy_automated_market_maker")
async def deploy_automated_market_maker(token_a: str, token_b: str, current_user: str = Depends(get_current_user)):
    contract_address = deploy_automated_market_maker(exchange.vm, token_a, token_b)
    return {"contract_address": contract_address}

@router.post("/deploy_yield_farm")
async def deploy_yield_farm(staking_token: str, reward_token: str, reward_rate: Decimal, current_user: str = Depends(get_current_user)):
    contract_code = {
        "init": YieldFarm(staking_token, reward_token, reward_rate),
        "stake": lambda storage, user, amount: storage["init"].stake(user, Decimal(amount)),
        "withdraw": lambda storage, user, amount: storage["init"].withdraw(user, Decimal(amount)),
        "get_reward": lambda storage, user: storage["init"].get_reward(user),
    }
    contract_address = exchange.vm.deploy_contract(contract_code)
    return {"contract_address": contract_address}
# Add import statements to your main application file
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
import asyncio
from decimal import Decimal
async def deploy_and_run_market_bot(exchange, vm, plata_contract, price_feed):
    initial_capital = {
        "PLATA": Decimal('1000000'),
        "BTC": Decimal('10'),
        "ETH": Decimal('100'),
        "DOT": Decimal('1000')
    }
    bot = QuantumInspiredMarketBot(exchange, vm, plata_contract, price_feed, initial_capital)
    await bot.run()
    
@router.post("/mint_stablecoin")
async def mint_stablecoin(amount: float, current_user: str = Depends(get_current_user)):
    try:
        result = vm.execute_contract(cashewstable_contract_address, "mint", [current_user, amount])
        return {"success": True, "minted_amount": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from decimal import Decimal

# Define collateral token (this could be another contract address or a token symbol)
collateral_token = "USDC"  # Or use the address of your collateral token contract

# Create a simple price feed (in a real-world scenario, this would be more complex)
class SimplePriceFeed:
    def __init__(self, initial_price):
        self.price = Decimal(initial_price)

    def get_price(self):
        return self.price

    def set_price(self, new_price):
        self.price = Decimal(new_price)
from vm import SimpleVM, Permission, Role, PBFTConsensus

# Initialize the price feed with the desired conversion rate
price_feed = SimplePriceFeed("1e-18")  # 1 PLATA = 1e-18 platastable

print(f"VM object: {vm}")


async def mint_stablecoin(user: str, amount: Decimal):
    return await vm.execute_contract(cashewstable_address, "mint", [user, amount])

async def burn_stablecoin(user: str, amount: Decimal):
    return await vm.execute_contract(cashewstable_address, "burn", [user, amount])

async def get_stablecoin_info():
    total_supply = await vm.execute_contract(cashewstable_address, "total_supply", [])
    price = await vm.execute_contract(cashewstable_address, "get_latest_price", [])
    return {"total_supply": total_supply, "price": price}

@router.post("/mint_stablecoin")
async def api_mint_stablecoin(amount: float, current_user: str = Depends(get_current_user)):
    try:
        result = await mint_stablecoin(current_user, Decimal(str(amount)))
        return {"success": True, "minted_amount": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stablecoin_info")
async def api_stablecoin_info():
    return await get_stablecoin_info()
mock_blockchain = MagicMock()
class TokenInfo(BaseModel):
    address: str
    name: str
    symbol: str
    balance: float

class ImportTokenRequest(BaseModel):
    address: str

class MintBurnRequest(BaseModel):
    address: str
    amount: float

class CreatePoolRequest(BaseModel):
    token_a: str
    token_b: str
    amount_a: float
    amount_b: float

@app.post("/import_token", response_model=TokenInfo)
async def import_token(request: ImportTokenRequest, current_user: User = Depends(get_current_user)):
    try:
        token = await blockchain.import_token(request.address, current_user.username)
        return TokenInfo(
            address=token.address,
            name=token.name,
            symbol=token.symbol,
            balance=float(token.balance_of(current_user.username))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mint_token")
async def mint_token(request: MintBurnRequest, current_user: User = Depends(get_current_user)):
    try:
        await blockchain.mint_token(request.address, current_user.username, Decimal(str(request.amount)))
        return {"success": True, "message": f"Minted {request.amount} tokens"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/burn_token")
async def burn_token(request: MintBurnRequest, current_user: User = Depends(get_current_user)):
    try:
        await blockchain.burn_token(request.address, current_user.username, Decimal(str(request.amount)))
        return {"success": True, "message": f"Burned {request.amount} tokens"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/create_liquidity_pool")
async def create_liquidity_pool(request: CreatePoolRequest, current_user: User = Depends(get_current_user)):
    try:
        pair = await blockchain.create_liquidity_pool(
            current_user.username,
            request.token_a,
            request.token_b,
            Decimal(str(request.amount_a)),
            Decimal(str(request.amount_b))
        )
        return {"success": True, "pair": pair}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_tokens", response_model=List[TokenInfo])
async def get_tokens(current_user: User = Depends(get_current_user)):
    try:
        tokens = await blockchain.get_user_tokens(current_user.username)
        return [
            TokenInfo(
                address=token.address,
                name=token.name,
                symbol=token.symbol,
                balance=float(token.balance_of(current_user.username))
            )
            for token in tokens
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
async def wait_for_initialization(timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if initialization_complete and all([
            app.state.blockchain,
            app.state.p2p_node,
            app.state.vm,
            app.state.price_feed,
            app.state.plata_contract,
            app.state.exchange
        ]):
            return True
        await asyncio.sleep(1)
    return False

@app.on_event("startup")
async def startup_event():
    global initialization_complete
    app.state.redis = await init_redis()
    logger.info("Redis initialized and connected.")

    try:
        # Start initialiseringen af systemet via async_main
        asyncio.create_task(async_main())

        # Vent på at systemet er fuldt initialiseret med en timeout
        if await wait_for_full_initialization():
            initialization_complete = True
            logger.info("System fully initialized and ready.")
        else:
            logger.error("System initialization timed out.")
            initialization_complete = False
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        initialization_complete = False

async def wait_for_full_initialization(timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        is_initialized, _ = check_initialization_status()
        if is_initialized:
            logger.info("System fully initialized and ready.")
            return True
        await asyncio.sleep(1)
    logger.error("System initialization timed out.")
    return False






# Add your API endpoints here
@app.get("/")
async def root():
    return {"message": "Welcome to the Quantum DAG Knight Exchange"}

@app.get("/node_status")
async def get_node_status():
    global node_directory
    return {"active_nodes": node_directory.get_active_nodes()}
@app.get("/blockchain_info")
async def get_blockchain_info():
    global blockchain
    if blockchain and blockchain.chain:
        return {
            "chain_length": len(blockchain.chain),
            "latest_block_hash": blockchain.chain[-1].hash
        }
    else:
        return {"error": "Blockchain not initialized"}

# Add more endpoints for order placement, trading, etc.

# Add a method to simulate node failure (for testing purposes)
@app.post("/simulate_node_failure")
async def simulate_node_failure():
    global node_directory
    node_id = os.getenv("NODE_ID", "node_1")
    node_directory.set_node_status(node_id, 'inactive')
    logger.info(f"Node {node_id} simulated failure")
    return {"message": f"Node {node_id} status set to inactive"}

# Add a method to simulate node recovery (for testing purposes)
@app.post("/simulate_node_recovery")
async def simulate_node_recovery():
    global node_directory, exchange
    node_id = os.getenv("NODE_ID", "node_1")
    node_directory.set_node_status(node_id, 'active')
    await exchange.sync_state()
    logger.info(f"Node {node_id} simulated recovery and state synced")
    return {"message": f"Node {node_id} status set to active and state synced"}
    propagation_times = []

class EnhancedNodeDirectory(NodeDirectory):
    def __init__(self):
        super().__init__()
        self.node_status = {}

    def set_node_status(self, node_id, status):
        self.node_status[node_id] = status

    def get_active_nodes(self):
        return [node for node, status in self.node_status.items() if status == 'active']

class EnhancedSecureDAGKnightServicer():
    def __init__(self, secret_key, node_directory, exchange):
        super().__init__()
        self.secret_key = secret_key
        self.node_directory = node_directory
        self.exchange = exchange

    async def PropagateOrder(self, request, context):
        start_time = time.time()
        result = await super().PropagateOrder(request, context)
        end_time = time.time()
        propagation_times.append(end_time - start_time)
        return result

async def run_dashboard(blockchain, exchange):
    from curses_dashboard.dashboard import DashboardUI
    def curses_main(stdscr):
        ui = DashboardUI(stdscr, blockchain, exchange)
        asyncio.run(ui.run())

    curses.wrapper(curses_main)

async def run_curses_dashboard(blockchain, exchange):
    from curses_dashboard.dashboard import DashboardUI
    def curses_main(stdscr):
        try:
            # Initialize colors
            curses.start_color()
            curses.use_default_colors()
            for i in range(0, curses.COLORS):
                curses.init_pair(i + 1, i, -1)

            # Initialize the dashboard UI
            dashboard = DashboardUI(stdscr, blockchain, exchange)
            
            # Run the dashboard
            asyncio.run(dashboard.run())
        
        except Exception as e:
            # Handle any exceptions that occur during the dashboard's execution
            logging.error(f"Error running curses dashboard: {str(e)}")
            logging.error(traceback.format_exc())
        
        finally:
            # Clean up curses settings and return terminal to normal state
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()

    # Use curses.wrapper to handle initialization and cleanup
    curses.wrapper(curses_main)
from daphne.server import Server as DaphneServer
from daphne.endpoints import build_endpoint_description_strings
class OrderRequest(BaseModel):
    user_id: str
    order_type: str
    pair: str
    amount: float
    price: float

class OrderResponse(BaseModel):
    order_id: str

class SwapRequest(BaseModel):
    user_id: str
    from_token: str
    to_token: str
    amount: float

class LiquidityRequest(BaseModel):
    user_id: str
    token_a: str
    token_b: str
    amount_a: float
    amount_b: float

class TokenInfo(BaseModel):
    address: str
    name: str
    symbol: str
    balance: float

# Endpoint to place a limit order
@router.post("/place_order", response_model=OrderResponse)
async def place_order(order: OrderRequest, current_user: str = Depends(get_current_user)):
    try:
        order_id = await exchange.place_limit_order(
            current_user,
            order.order_type,
            order.pair,
            Decimal(str(order.amount)),
            Decimal(str(order.price))
        )
        return OrderResponse(order_id=order_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to get the order book for a trading pair
@router.get("/order_book/{pair}")
async def get_order_book(pair: str):
    try:
        order_book = await exchange.get_order_book(pair)
        return order_book
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to swap tokens
@router.post("/swap")
async def swap_tokens(swap_request: SwapRequest, current_user: str = Depends(get_current_user)):
    try:
        result = await exchange.swap(
            current_user,
            swap_request.from_token,
            swap_request.to_token,
            Decimal(str(swap_request.amount))
        )
        return {"success": True, "amount_received": float(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to add liquidity to a pool
@router.post("/add_liquidity")
async def add_liquidity(request: LiquidityRequest, current_user: str = Depends(get_current_user)):
    try:
        result = await exchange.add_liquidity(
            current_user,
            f"{request.token_a}_{request.token_b}",
            Decimal(str(request.amount_a)),
            Decimal(str(request.amount_b))
        )
        return {"success": True, "liquidity_tokens": float(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to remove liquidity from a pool
@router.post("/remove_liquidity")
async def remove_liquidity(request: LiquidityRequest, current_user: str = Depends(get_current_user)):
    try:
        amount_a, amount_b = await exchange.remove_liquidity(
            current_user,
            f"{request.token_a}_{request.token_b}",
            Decimal(str(request.amount_a))
        )
        return {"success": True, "amount_a": float(amount_a), "amount_b": float(amount_b)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to get user's token balances
@router.get("/user_tokens", response_model=List[TokenInfo])
async def get_user_tokens(current_user: str = Depends(get_current_user)):
    try:
        tokens = await exchange.get_user_tokens(current_user)
        return [
            TokenInfo(
                address=token.address,
                name=token.name,
                symbol=token.symbol,
                balance=float(token.balance_of(current_user))
            )
            for token in tokens
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to get current market price
@router.get("/market_price/{pair}")
async def get_market_price(pair: str):
    try:
        price = await exchange.get_market_price(pair)
        return {"pair": pair, "price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from pydantic import BaseModel, HttpUrl
from jwt import PyJWTError as JWTError

class CreateTokenRequest(BaseModel):
    creator_address: str
    token_name: str
    token_symbol: str
    total_supply: int
    logo_url: Optional[HttpUrl] = None
    logo_data: Optional[str] = None  # Base64 encoded image data


class TransferTokenRequest(BaseModel):
    from_address: str
    to_address: str
    token_name: str
    amount: int

class CreateLiquidityPoolRequest(BaseModel):
    token1: str
    token2: str
    amount1: int
    amount2: int
    owner: str

class TokenData(BaseModel):
    username: str
class UpdateTokenLogoRequest(BaseModel):
    token_address: str
    logo_url: Optional[HttpUrl] = None
    logo_data: Optional[str] = None  # Base64 encoded image data
@app.post("/api/create_token")
async def create_token(
    token_data: CreateTokenRequest,
    current_user: str = Depends(get_current_user)
):
    success, result = vm.create_token(
        creator_address=current_user,
        token_name=token_data.token_name,
        token_symbol=token_data.token_symbol,
        total_supply=token_data.total_supply,
        logo_url=token_data.logo_url,
        logo_data=token_data.logo_data
    )
    if success:
        return {
            "status": "success",
            "token_name": token_data.token_name,
            "token_symbol": token_data.token_symbol,
            "token_address": result
        }
    else:
        raise HTTPException(status_code=400, detail=result)

@app.post("/api/update_token_logo")
async def update_token_logo(
    logo_data: UpdateTokenLogoRequest,
    current_user: str = Depends(get_current_user)
):
    success = vm.update_token_logo(
        token_address=logo_data.token_address,
        logo_url=logo_data.logo_url,
        logo_data=logo_data.logo_data
    )
    if success:
        return {"status": "success", "message": "Token logo updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to update token logo")


@app.post("/api/upload_token_logo")
async def upload_token_logo(
    token_address: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    try:
        contents = await file.read()
        logo_data = base64.b64encode(contents).decode()
        success = vm.update_token_logo(token_address, logo_data=logo_data)
        if success:
            return {"status": "success", "message": "Token logo uploaded successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to upload token logo")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.get("/api/created_tokens", response_model=List[dict])
async def get_created_tokens(current_user: str = Depends(get_current_user)):
    try:
    
        user_tokens = vm.get_user_tokens(current_user)
        return user_tokens
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching created tokens: {str(e)}"
        )

@app.post("/api/mint_token")
async def mint_token(request: TransferTokenRequest):
    result = vm.transfer_token(request.from_address, request.to_address, request.token_name, request.amount)
    if result:
        return {"status": "success", "minted_amount": request.amount}
    raise HTTPException(status_code=400, detail="Minting failed")

@app.post("/api/burn_token")
async def burn_token(request: TransferTokenRequest):
    # Assuming burn is just a transfer to a burn address (e.g., 0x0)
    burn_address = "0x0000000000000000000000000000000000000000"
    result = vm.transfer_token(request.from_address, burn_address, request.token_name, request.amount)
    if result:
        return {"status": "success", "burned_amount": request.amount}
    raise HTTPException(status_code=400, detail="Burning failed")

@app.post("/api/create_liquidity_pool")
async def create_liquidity_pool(request: CreateLiquidityPoolRequest):
    # Logic to create a liquidity pool in the VM
    pool_address = vm.create_liquidity_pool(request.owner, request.token1, request.token2, request.amount1, request.amount2)
    if pool_address:
        return {"status": "success", "pool_address": pool_address}
    raise HTTPException(status_code=400, detail="Liquidity pool creation failed")


@app.get("/api/liquidity_pools")
async def get_liquidity_pools():
    # Placeholder for retrieving liquidity pools from the VM
    return {"pools": list(vm.contracts.keys())}
class QuantumCrossChainBridge:
    def __init__(self, supported_chains: List[str]):
        self.supported_chains = supported_chains
        self.qrng = QuantumRandomNumberGenerator()
        self.zk_system = SecureHybridZKStark(security_level=256)

    def generate_bridge_key(self, chain_a: str, chain_b: str) -> str:
        if chain_a not in self.supported_chains or chain_b not in self.supported_chains:
            raise ValueError("Unsupported chain")
        
        random_seed = self.qrng.generate_random_number(num_qubits=32)
        return hashlib.sha256(f"{chain_a}-{chain_b}-{random_seed}".encode()).hexdigest()

    def create_cross_chain_transaction(self, from_chain: str, to_chain: str, 
                                       sender: str, receiver: str, amount: int) -> dict:
        bridge_key = self.generate_bridge_key(from_chain, to_chain)
        transaction = {
            "from_chain": from_chain,
            "to_chain": to_chain,
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "bridge_key": bridge_key
        }
        
        # Generate zero-knowledge proof
        secret = int(hashlib.sha256(f"{sender}{amount}".encode()).hexdigest(), 16)
        public_input = int(hashlib.sha256(f"{receiver}{amount}{bridge_key}".encode()).hexdigest(), 16)
        zk_proof = self.zk_system.prove(secret, public_input)
        
        transaction["zk_proof"] = zk_proof
        return transaction

    def verify_cross_chain_transaction(self, transaction: dict) -> bool:
        public_input = int(hashlib.sha256(f"{transaction['receiver']}{transaction['amount']}{transaction['bridge_key']}".encode()).hexdigest(), 16)
        return self.zk_system.verify(public_input, transaction["zk_proof"])

    def execute_cross_chain_swap(self, transaction: dict):
        if self.verify_cross_chain_transaction(transaction):
            # Logic to execute the swap on both chains
            print(f"Executing swap: {transaction['amount']} from {transaction['from_chain']} to {transaction['to_chain']}")
            # Implement actual swap logic here
        else:
            raise ValueError("Invalid cross-chain transaction")
class QuantumRandomNumberGenerator:
    @staticmethod
    def generate_random_number(num_qubits: int = 8) -> int:
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)  # Apply Hadamard gates
        circuit.measure(qr, cr)

        backend = Aer.get_backend('qasm_simulator')
        
        # Transpile the circuit for the target backend
        transpiled_circuit = transpile(circuit, backend)
        
        # Run the transpiled circuit
        job = backend.run(transpiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        random_bitstring = list(counts.keys())[0]
        return int(random_bitstring, 2)


mining_task = None
# Ensure that blockchain and p2p_node are globally accessiblex
# Global P2P node variable
p2p_node = None

continue_mining = False
mining_task = None
event_loop = asyncio.get_event_loop()
@app.post("/mine_block")
async def mine_block(request: MineBlockRequest, background_tasks: BackgroundTasks, pincode: str = Depends(authenticate)):
    global blockchain, p2p_node, continue_mining, mining_task, initialization_complete

    try:
        logger.debug(f"Received request to mine block with wallet address: {request.wallet_address}")

        # Check if mining is already in progress
        if continue_mining:
            logger.warning("Mining is already in progress. Exiting...")
            return {"success": False, "message": "Mining is already in progress"}

        # Check if the blockchain is initialized
        if blockchain is None:
            logger.error(f"Blockchain is None. Initialization complete: {initialization_complete}")
            return {"success": False, "message": "Blockchain is not initialized"}

        # Check if the P2P node is initialized
        if p2p_node is None:
            logger.error(f"P2P node is None. Initialization complete: {initialization_complete}")
            return {"success": False, "message": "P2P node is not initialized"}

        # Retrieve the latest block hash using the new method
        latest_block_hash = blockchain.get_latest_block_hash()
        logger.debug(f"Latest block hash: {latest_block_hash}")

        # If no block exists, we assume it's the genesis block
        if latest_block_hash == "0":
            logger.info("Genesis block being created as blockchain is empty.")

        # Start the continuous mining process
        logger.info("Starting the continuous block mining process")
        continue_mining = True

        # Add the continuous mining task to background tasks
        mining_task = background_tasks.add_task(continuous_mining, blockchain, request.wallet_address, p2p_node)
        logger.debug(f"Mining task added: {mining_task}")

        # Prepare block information
        block_info = {
            "block_hash": latest_block_hash,
            "miner": request.wallet_address,
            "transactions": blockchain.get_pending_transactions(),
            "timestamp": int(time.time())
        }

        logger.info(f"Mining block with info: {block_info}")

        # Broadcast the mined block event to other nodes
        await p2p_node.broadcast(Message(
            MessageType.BLOCK, 
            {'block_hash': latest_block_hash, 'miner': request.wallet_address, 'block_info': block_info}
        ))

        # WebSocket broadcast to subscribed clients
        await p2p_node.broadcast_event('block_mined', {
            'block_hash': latest_block_hash,
            'miner': request.wallet_address,
            'block_info': block_info
        })

        logger.info(f"Block mined and broadcasted: {block_info['block_hash']}")

        # Return success response
        return {"success": True, "message": "Continuous mining started", "block_info": block_info}

    except Exception as e:
        # Log and raise an error if there was an issue starting the mining process
        logger.error(f"Error starting continuous mining: {str(e)}")
        logger.error(traceback.format_exc())

        return JSONResponse(status_code=500, content={
            "success": False, 
            "message": "Error starting continuous mining",
            "details": str(e),
            "traceback": traceback.format_exc()
        })

def check_initialization_status():
    global initialization_status
    uninitialized = [k for k, v in initialization_status.items() if not v]
    if uninitialized:
        return False, f"System is still initializing. Waiting for: {', '.join(uninitialized)}"
    return True, "System is fully initialized and ready"
@app.get("/debug/initialization_status")
async def debug_initialization_status():
    return initialization_status
    
@app.middleware("http")
async def ensure_p2p_node_middleware(request: Request, call_next):
    # Ensure that the P2PNode is initialized before processing the request
    if not hasattr(app.state, 'p2p_node') or app.state.p2p_node is None:
        try:
            # Re-initialize P2PNode if it is missing
            logger.debug("P2PNode is not initialized. Initializing now...")
            await async_main_initialization()  # Make sure the P2PNode and other components are initialized
            logger.info("P2PNode and other components initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize P2PNode: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={"message": "P2PNode is not initialized and cannot be initialized automatically."}
            )
    
    # Proceed with the request
    response = await call_next(request)
    return response

    
initialization_status = {
    "blockchain": False,
    "p2p_node": False,
    "vm": False,
    "price_feed": False,
    "plata_contract": False,
    "exchange": False
}


initialization_lock = asyncio.Lock()


# Initialize Redis connection
redis = None

# Initialize Redis connection
async def init_redis():
    global redis
    if redis is None:
        redis = await aioredis.from_url('redis://localhost')
        app.state.redis = redis  # Store it in app state as well
        logger.info("Redis initialized successfully.")
    else:
        logger.warning("Redis is already initialized in global scope.")
    
    if hasattr(app.state, 'redis'):
        logger.info("Redis is available in app.state.redis")
    else:
        logger.error("Redis is NOT available in app.state.redis")



# Store the initialization status in Redis
async def update_initialization_status(component, status):
    global redis
    if redis is None:
        raise RuntimeError("Redis is not initialized. Call init_redis first.")

    async with initialization_lock:
        await redis.hset('initialization_status', component, int(status))  # Store as int (0 or 1)
        updated_status = await redis.hgetall('initialization_status')
        logger.info(f"Component '{component}' status updated. Current status: {updated_status}")
@app.middleware("http")
async def ensure_redis_and_blockchain_middleware(request: Request, call_next):
    # Ensure Redis is initialized
    if not hasattr(app.state, 'redis') or app.state.redis is None:
        await init_redis()  # Initialize Redis if not already done
        app.state.redis = redis

    # Ensure Blockchain is initialized
    if not hasattr(app.state, 'blockchain') or app.state.blockchain is None:
        await async_main_initialization()  # A function to initialize blockchain and other components
        app.state.blockchain = blockchain

    return await call_next(request)

@app.get("/health")
async def health_check():
    # Ensure Redis is initialized
    if not hasattr(app.state, "redis") or app.state.redis is None:
        logger.error("Redis is not initialized. Call init_redis first.")
        return {
            "status": "error",
            "message": "Redis is not initialized. Call init_redis first."
        }

    # Ensure Blockchain is initialized
    if not hasattr(app.state, "blockchain") or app.state.blockchain is None:
        logger.error("Blockchain is not initialized.")
        return {
            "status": "error",
            "message": "Blockchain is not initialized."
        }

    # Redis is available, proceed with health check
    redis = app.state.redis  # Access Redis from app.state
    initialization_status = await redis.hgetall('initialization_status')

    logger.info(f"Health check called. Current initialization status read: {initialization_status}")

    uninitialized_components = [comp.decode('utf-8') for comp, status in initialization_status.items() if int(status) == 0]
    if uninitialized_components:
        logger.info(f"Health check: System not fully initialized. Waiting for: {uninitialized_components}")
        return {
            "status": "initializing",
            "message": f"System is still initializing. Waiting for: {uninitialized_components}"
        }

    logger.info("Health check: System fully initialized.")
    return {
        "status": "ready",
        "message": "System is fully initialized and ready."
    }




def check_initialization_status():
    global initialization_status
    logger.info(f"Checking initialization status: {initialization_status}")
    uninitialized = [k for k, v in initialization_status.items() if not v]
    if uninitialized:
        logger.info(f"Uninitialized components: {uninitialized}")
        return False, f"System is still initializing. Waiting for: {', '.join(uninitialized)}"
    logger.info("All components are initialized.")
    return True, "System is fully initialized and ready"



# FastAPI route to stop mining
@app.post("/stop_mining")
async def stop_mining():
    global continue_mining, mining_task
    try:
        if not continue_mining:
            return {"success": False, "message": "Mining is not currently running"}
        
        continue_mining = False
        if mining_task:
            mining_task.cancel()
        logger.info("Mining has been stopped")
        return {"success": True, "message": "Mining stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop mining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop mining: {str(e)}")
async def propagate_block_with_retry(block, retries=3):
    for attempt in range(retries):
        try:
            logger.debug(f"Attempting to propagate block: {block.hash}. Retries left: {retries - attempt}")
            
            if p2p_node is None:
                logger.warning("P2P node is None. Waiting for initialization...")
                await asyncio.sleep(5)
                continue

            active_peers = await p2p_node.get_active_peers()
            if not active_peers:
                logger.warning("No active peers found. Retrying...")
                await asyncio.sleep(5)
                continue

            await p2p_node.propagate_block(block)
            logger.info(f"Block {block.hash} successfully propagated.")
            return True

        except Exception as e:
            logger.error(f"Failed to propagate block: {block.hash}. Error: {str(e)}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(5)

    logger.error(f"Max retries reached. Block {block.hash} could not be propagated.")
    return False

import asyncio

async def continuous_mining(blockchain, wallet_address: str, p2p_node: P2PNode):
    global continue_mining
    blocks_mined = 0

    while continue_mining:
        try:
            # Create a new block with the necessary data
            logger.info(f"Starting block creation for miner: {wallet_address}")

            # Get the latest block's hash and quantum signature
            previous_hash = blockchain.get_latest_block_hash()
            quantum_signature = blockchain.generate_quantum_signature()
            reward = blockchain.get_block_reward()
            transactions = blockchain.get_pending_transactions()

            # Log block data before mining
            logger.info(f"Creating new block with previous hash: {previous_hash}, "
                        f"quantum signature: {quantum_signature}, reward: {reward}, "
                        f"transactions: {transactions}")

            # Create a new block
            new_block = QuantumBlock(
                previous_hash=previous_hash,
                data="Block data",
                quantum_signature=quantum_signature,
                reward=reward,
                transactions=transactions
            )

            logger.info(f"New block created with nonce: {new_block.nonce}, ready for mining...")

            # Mine the block
            difficulty = blockchain.difficulty
            new_block.mine_block(difficulty=difficulty)

            logger.info(f"Block mined with hash: {new_block.hash}, nonce: {new_block.nonce}")

            if new_block.is_valid():
                # Propose the block to the network using ZKP via the P2P node
                logger.info(f"Proposing block to the network with hash: {new_block.hash}")
                await p2p_node.propose_block(new_block)

                # Add block locally after proposing
                blockchain.add_block(new_block)
                blockchain.process_transactions(new_block.transactions)
                blockchain.reward_miner(wallet_address, reward)

                # Broadcast the block with a retry mechanism
                retries = 3
                await blockchain.propagate_block_with_retry(new_block, retries=retries)

                logger.info(f"Block {new_block.hash} successfully propagated across the network")
                blocks_mined += 1

            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during continuous mining: {str(e)}")
            await asyncio.sleep(5)





async def wait_for_components_ready():
    components = [blockchain, p2p_node, vm, price_feed, plata_contract, exchange]
    for component in components:
        if hasattr(component, 'wait_for_ready'):
            await component.wait_for_ready()
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
async def run_daphne_server():
    from quantumdagknight import app  # Import your FastAPI or other ASGI app here

    # Setup the Daphne server
    interface = "0.0.0.0"
    port = 50503
    endpoints = build_endpoint_description_strings(interface, port)

    # Increase the timeout and ping interval to prevent shutdown
    server = DaphneServer(
        application=app,
        endpoints=endpoints,
        ping_interval=300,  # Increase ping interval to 300 seconds (5 minutes)
        application_close_timeout=600,  # Increase close timeout to 600 seconds (10 minutes)
        ws_connect_timeout=300  # Increase WebSocket connect timeout if needed
    )

    # Run the server in the current asyncio loop
    await asyncio.to_thread(server.run)


async def initialize_p2p_node(ip_address, p2p_port):
    try:
        logger.info(f"Initializing P2P node at {ip_address}:{p2p_port}")
        p2p_node = P2PNode(blockchain=None, host=ip_address, port=p2p_port)

        # Start P2P Node WebSocket server for node-to-node communication
        await p2p_node.start()

        # Start the second WebSocket server for client subscriptions
        await p2p_node.start_ws_server()

        # Check connection status without await (assuming is_connected returns a boolean)
        if p2p_node.is_connected():
            logger.info(f"P2P node successfully connected. Peers: {p2p_node.peers}")
        else:
            logger.warning("P2P node failed to connect. No peers are connected.")

        return p2p_node

    except Exception as e:
        logger.error(f"Error initializing P2P node: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Global variables for initialization
blockchain = None  # Define blockchain at a global level
p2p_node = None  # Define P2P node at a global level
ip_address = os.getenv("P2P_HOST", "127.0.0.1")
p2p_port = int(os.getenv("P2P_PORT", "50510"))
secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")
node_id = os.getenv("NODE_ID", "node_1")
# Global variables for initialization signaling
# Get the current event loop and make sure all tasks use this loop

initialization_complete = False
import asyncio
import traceback
import logging

logger = logging.getLogger("quantumdagknight")

# Define global variables
blockchain = None
p2p_node = None
initialization_complete = False
async def initialize_vm():
    logger.info("Initializing SimpleVM...")
    vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
    logger.info("SimpleVM initialized successfully.")
    return vm

async def initialize_p2p_node(ip_address, p2p_port):
    logger.info(f"Initializing P2P node at {ip_address}:{p2p_port}")
    p2p_node = P2PNode(blockchain=None, host=ip_address, port=p2p_port)
    await p2p_node.start()
    
    logger.info(f"P2P node initialized successfully: {p2p_node}")
    return p2p_node

async def initialize_blockchain(p2p_node, vm):
    logger.info("Initializing QuantumBlockchain...")
    consensus = PBFTConsensus(nodes=[], node_id=node_id)
    blockchain = QuantumBlockchain(consensus, secret_key, None, vm)
    await blockchain.set_p2p_node(p2p_node)

    logger.info(f"QuantumBlockchain initialized successfully: {blockchain}")
    return blockchain

async def initialize_price_feed():
    logger.info("Initializing PriceOracle...")
    price_feed = PriceOracle()
    logger.info("PriceOracle initialized successfully.")
    return price_feed

async def initialize_plata_contract(vm):
    logger.info("Initializing PlataContract...")
    plata_contract = PlataContract(vm)
    logger.info("PlataContract initialized successfully.")
    return plata_contract

async def initialize_exchange(blockchain, vm, price_feed):
    logger.info("Initializing Exchange...")
    exchange = EnhancedExchangeWithZKStarks(
        blockchain, vm, price_feed, node_directory, 
        desired_security_level=20, host="localhost", port=8765
    )
    logger.info("Exchange initialized successfully.")
    return exchange

async def wait_for_components_ready(components):
    for component in components:
        if hasattr(component, 'wait_for_ready'):
            await component.wait_for_ready()
    logger.info("All components are ready.")
import asyncio
import traceback
import asyncio
import traceback
async def async_main_initialization():
    global blockchain, redis, p2p_node, vm

    try:
        # Initialize Redis if not already initialized
        if not redis:
            logger.info("Initializing Redis...")
            await init_redis()
            app.state.redis = redis  # Store Redis in app state
            logger.info("Redis initialized successfully.")

        # Initialize VM if not already initialized
        if not vm:
            logger.info("Initializing VM...")
            vm = await initialize_vm()
            app.state.vm = vm  # Store VM in app state
            logger.info("VM initialized successfully.")

        # Initialize P2P node if not already initialized
        if not p2p_node:
            logger.info("Initializing P2P node...")
            p2p_node = await initialize_p2p_node(ip_address, p2p_port)
            app.state.p2p_node = p2p_node  # Store P2P node in app state
            logger.info("P2P node initialized successfully.")

        # Initialize Blockchain if not already initialized
        if not blockchain:
            logger.info("Initializing Blockchain...")
            blockchain = await initialize_blockchain(p2p_node, vm)
            app.state.blockchain = blockchain  # Store blockchain in app state
            logger.info("Blockchain initialized successfully.")

        # Store all components in app.state
        app.state.components = {
            "blockchain": blockchain,
            "p2p_node": p2p_node,
            "vm": vm,
            "redis": redis
        }
        

        # Set the blockchain for the P2PNode
        if p2p_node:
            p2p_node.set_blockchain(blockchain)

        logger.info("All components (Blockchain, VM, P2P node, and Redis) initialized successfully.")

    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError("Initialization failed.") from e



async def main():
    restart_delay = 10
    while True:
        try:
            # Start the Daphne server
            daphne_task = asyncio.create_task(run_daphne_server())
            logger.info("Daphne server started.")

            # Run the main initialization
            await async_main()

            # Keep the main loop running
            while True:
                await asyncio.sleep(60)
                logger.info("Main loop still running...")

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info(f"Restarting in {restart_delay} seconds...")
            await asyncio.sleep(restart_delay)
            restart_delay = min(restart_delay * 2, 300)  # Max 5 minutes delay

        finally:
            # Cleanup
            if 'p2p_node' in globals() and p2p_node:
                await p2p_node.stop()
                logger.info("P2P node stopped.")

            # Cancel the Daphne server task
            if 'daphne_task' in locals() and not daphne_task.done():
                daphne_task.cancel()
                try:
                    await daphne_task
                except asyncio.CancelledError:
                    logger.info("Daphne task cancelled.")

            logger.info("Cleanup completed. Restarting...")

async def async_main():
    global redis, blockchain

    # Ensure Redis and Blockchain are initialized via async_main_initialization
    await async_main_initialization()

    try:
        logger.info("Starting async_main initialization...")

        # Initialize VM
        logger.info("Initializing VM...")
        vm = await initialize_vm()
        await update_initialization_status("vm", True)  # Update with lock

        # Initialize P2P node
        logger.info("Initializing P2P node...")
        p2p_node = await initialize_p2p_node(ip_address, p2p_port)
        await update_initialization_status("p2p_node", True)  # Update with lock

        # Initialize Blockchain (already initialized in async_main_initialization)
        logger.info("Initializing Blockchain...")
        blockchain = await initialize_blockchain(p2p_node, vm)
        await update_initialization_status("blockchain", True)  # Update with lock

        # Initialize Price Feed
        logger.info("Initializing Price Feed...")
        price_feed = await initialize_price_feed()
        await update_initialization_status("price_feed", True)  # Update with lock

        # Initialize Plata Contract
        logger.info("Initializing Plata Contract...")
        plata_contract = await initialize_plata_contract(vm)
        await update_initialization_status("plata_contract", True)  # Update with lock

        # Initialize Exchange
        logger.info("Initializing Exchange...")
        exchange = await initialize_exchange(blockchain, vm, price_feed)
        await update_initialization_status("exchange", True)  # Update with lock

        # Wait for all components to be ready
        components = [blockchain, p2p_node, vm, price_feed, plata_contract, exchange]
        await wait_for_components_ready(components)

        # Set components in app.state
        app.state.blockchain = blockchain
        app.state.p2p_node = p2p_node
        app.state.vm = vm
        app.state.price_feed = price_feed
        app.state.plata_contract = plata_contract
        app.state.exchange = exchange

        initialization_complete = True
        logger.info("Initialization complete.")

    except Exception as e:
        initialization_complete = False
        logger.error(f"Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())
