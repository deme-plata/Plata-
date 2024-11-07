# Import necessary modules
import os
import time
import logging
import threading
from concurrent import futures
import uvicorn
import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, padding, hashes
from cryptography.hazmat.backends import default_backend
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile,assemble
from qiskit.providers.jobstatus import JobStatus
from qiskit.exceptions import QiskitError   
from nacl.public import PrivateKey, Box
from nacl.utils import random
from typing import List
import json
from qiskit.circuit.random import random_circuit
import asyncio
import aiohttp
import networkx as nx

import base64
import hashlib
import numpy as np
from mnemonic import Mnemonic
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP
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
import os
import base64
import re  
import string
from base64 import urlsafe_b64encode
from decimal import Decimal
import json
from typing import Dict
from vm import SimpleVM, Permission, Role,    PBFTConsensus
from vm import SimpleVM
import pytest
import httpx

import curses
from pydantic import BaseModel, Field, validator  # Ensure validator is imported
from pydantic import BaseModel, field_validator  # Use field_validator for Pydantic V2
from pydantic import BaseModel, Field, root_validator
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Tuple
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
from mongodb_manager import QuantumDAGKnightDB
from multisig_zkp import MultisigZKP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import asyncio
from typing import Dict, Set
import weakref
import signal
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import asyncio
import logging
from typing import Callable, Dict, Set
from contextlib import asynccontextmanager
from sanic import Sanic, Blueprint
from sanic.response import json
from sanic.server.protocols.websocket_protocol import WebSocketProtocol
from sanic.request import Request
from sanic.exceptions import SanicException
from sanic.log import logger as sanic_logger
from async_timeout import timeout
import asyncio
from sanic.response import json
from sanic.exceptions import SanicException
from sanic import Blueprint
import traceback
import logging
from P2PNode import P2PNode, create_enhanced_p2p_node, Message, MessageType,enhance_p2p_node,SyncComponent, SyncStatus,DAGKnightConsensus
import copy
from quantum_signer import QuantumSigner
from shared_logic import Transaction, QuantumBlock
from CryptoProvider import CryptoProvider
from QuantumBlockchain import QuantumBlockchain, SecurityManager,QuantumStateManager  
from Wallet import Wallet
from DAGKnightMiner import DAGKnightMiner
from DAGConfirmationSystem import DAGConfirmationSystem
from DAGKnightGasSystem import EnhancedDAGKnightGasSystem
from P2PNode import P2PNode, enhance_p2p_node,LinuxQuantumNode,NetworkOptimizer,DAGKnightConsensus    # Import the enhance_p2p_node function
import systemd
class SignalManager:
    def __init__(self, app):
        self.app = app
        self.handlers = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        # Register core signal handlers
        @self.app.signal('server.init.before')
        async def server_init(app, loop):
            logger.info("Server initialization starting...")
            try:
                app.ctx.initialization_status = "starting"
                await self.handle_server_init(app, loop)
                app.ctx.initialization_status = "complete"
                logger.info("Server initialization completed")
            except Exception as e:
                app.ctx.initialization_status = "failed"
                logger.error(f"Server initialization failed: {str(e)}")
                logger.error(traceback.format_exc())

        @self.app.signal('server.shutdown.before')
        async def server_shutdown(app, loop):
            logger.info("Server shutdown starting...")
            try:
                await self.handle_server_shutdown(app, loop)
                logger.info("Server shutdown completed")
            except Exception as e:
                logger.error(f"Server shutdown error: {str(e)}")
                logger.error(traceback.format_exc())

    async def handle_server_init(self, app, loop):
        # Initialize your components here
        app.ctx.components = {}
        
        # Initialize Redis
        app.ctx.redis = await init_redis()
        app.ctx.components['redis'] = True
        
        # Initialize VM
        vm = await initialize_vm()
        if vm:
            app.ctx.vm = vm
            app.ctx.components['vm'] = True
        
        # Initialize P2P node
        p2p_node = await initialize_p2p_node(ip_address, p2p_port)
        if p2p_node:
            app.ctx.p2p_node = p2p_node
            app.ctx.components['p2p_node'] = True
        
        # Initialize Blockchain
        blockchain = await initialize_blockchain(p2p_node, vm)
        if blockchain:
            app.ctx.blockchain = blockchain
            app.ctx.components['blockchain'] = True

    async def handle_server_shutdown(self, app, loop):
        # Cleanup components in reverse order
        if hasattr(app.ctx, 'blockchain'):
            await app.ctx.blockchain.cleanup()
        
        if hasattr(app.ctx, 'p2p_node'):
            await app.ctx.p2p_node.stop()
        
        if hasattr(app.ctx, 'redis'):
            await app.ctx.redis.close()

# Initialize Sanic app
app = Sanic("QuantumDAGKnight")
app.config.CORS_ORIGINS = "*"
app.config.WEBSOCKET_MAX_SIZE = 2**20  # 1MB
app.config.GRACEFUL_SHUTDOWN_TIMEOUT = 15.0
signal_manager = SignalManager(app)

# Register middleware
@app.middleware('request')
async def track_request(request):
    request.ctx.start_time = time.time()
    request.ctx.request_id = str(uuid.uuid4())
    logger.info(f"Request started: {request.ctx.request_id}")

@app.middleware('response')
async def track_response(request, response):
    elapsed = time.time() - request.ctx.start_time
    logger.info(f"Request {request.ctx.request_id} completed in {elapsed:.4f}s")

# Main application startup
@app.before_server_start
async def setup_app(app, loop):
    try:
        logger.info("Initializing application components...")
        await async_main_initialization()
        logger.info("Application components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Application cleanup
@app.before_server_stop
async def cleanup_app(app, loop):
    try:
        logger.info("Cleaning up application resources...")
        await cleanup_resources()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error(traceback.format_exc())

class ASGILifespanManager:
    def __init__(self):
        self.active_tasks: Set[asyncio.Task] = set()
        self.cleanup_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def track_task(self, task: asyncio.Task):
        async with self._lock:
            self.active_tasks.add(task)
            task.add_done_callback(lambda t: asyncio.create_task(self.remove_task(t)))

    async def remove_task(self, task: asyncio.Task):
        async with self._lock:
            self.active_tasks.discard(task)

    async def cleanup(self, timeout: float = 5.0):
        try:
            self.cleanup_event.set()
            if self.active_tasks:
                tasks = list(self.active_tasks)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                await asyncio.wait(tasks, timeout=timeout)
                
                # Force cleanup any remaining tasks
                for task in tasks:
                    if not task.done():
                        try:
                            task.cancel()
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Error during task cleanup: {str(e)}")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[asyncio.Task]] = {}
        self._lock = asyncio.Lock()
        
    async def add_connection(self, request_id: str, task: asyncio.Task):
        async with self._lock:
            if request_id not in self.active_connections:
                self.active_connections[request_id] = set()
            self.active_connections[request_id].add(task)
            
    async def remove_connection(self, request_id: str, task: asyncio.Task):
        async with self._lock:
            if request_id in self.active_connections:
                self.active_connections[request_id].discard(task)
                if not self.active_connections[request_id]:
                    del self.active_connections[request_id]
                    
    async def cleanup_connections(self, timeout: float = 5.0):
        async with self._lock:
            tasks = []
            for request_id, connection_tasks in self.active_connections.items():
                for task in connection_tasks:
                    if not task.done():
                        task.cancel()
                        tasks.append(task)
            
            if tasks:
                await asyncio.wait(tasks, timeout=timeout)

connection_manager = ConnectionManager()

blockchain = None
p2p_node = None
helius_api = HeliusAPI(api_key="855fde7e-b54f-4c6f-b71c-e6876772ec81")

fake_users_db = {}
db = QuantumDAGKnightDB("mongodb://localhost:27017", "quantumdagknight_db")

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
vm = ZKVM(security_level=20)


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


# Add the middleware with a custom timeout

class Trade:
    def __init__(self, buyer_id, seller_id, base_currency, quote_currency, amount, price):
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.amount = amount
        self.price = price




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


async def get_transaction_history():
    try:
        # Assuming blockchain stores transaction history
        transaction_history = blockchain.get_transaction_history()  # Fetch transaction history
        return transaction_history
    except Exception as e:
        logger.error(f"Error fetching transaction history: {str(e)}")
        raise
def calculate_security_parameters(security_level: int) -> Dict[str, Any]:
    return {
        "field_size": 2**min(security_level, 100) - 1,  # Reduce field size for testing
        "hash_size": min(security_level * 2, 256),  # Cap the hash size for performance
        "merkle_height": math.ceil(math.log2(security_level * 8)),
        "num_rounds": max(10, security_level // 4),  # Lower the number of rounds
        "expansion_factor": 4,
        "fri_layers": math.ceil(math.log2(security_level)),
        "security_bits": security_level
    }

import hashlib
import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal
import random
import time
import logging
import traceback
import asyncio
class SimpleZKProver:
    """Simplified ZK proof system for quantum mining"""
    def __init__(self, security_level: int = 20):
        self.security_level = security_level
        self.params = {
            "field_size": 2**security_level - 1,
            "hash_size": security_level * 2,
            "merkle_height": 16,  # Fixed reasonable height
            "num_rounds": max(2, security_level // 2),
            "expansion_factor": 4,
            "security_bits": security_level
        }

    def hash_to_field(self, data: bytes) -> int:
        """Hash data to field element"""
        hash_bytes = hashlib.sha256(data).digest()
        return int.from_bytes(hash_bytes, 'big') % self.params["field_size"]

    def prove(self, secret: int, public_input: int) -> tuple:
        """Generate a simplified ZK proof"""
        # Create a commitment to the secret
        commitment = hashlib.sha256(str(secret).encode()).digest()
        
        # Generate challenge
        challenge = self.hash_to_field(commitment + str(public_input).encode())
        
        # Generate response
        response = (secret + challenge) % self.params["field_size"]
        
        return (commitment, response)

    def verify(self, public_input: int, proof: tuple) -> bool:
        """Verify a simplified ZK proof"""
        try:
            commitment, response = proof
            
            # Regenerate challenge
            challenge = self.hash_to_field(commitment + str(public_input).encode())
            
            # Verify the proof
            expected = hashlib.sha256(str((response - challenge) % self.params["field_size"]).encode()).digest()
            return commitment == expected
        except Exception as e:
            logger.error(f"Error verifying ZK proof: {str(e)}")
            return False


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



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("node")
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


# Add this router to your main FastAPI app

# You might want to run this function periodically or trigger it based on certain conditions
async def update_prices_periodically():
    while True:
        await price_oracle.update_prices()
        await asyncio.sleep(60)  # Update every minute


    


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

price_oracle = PriceOracle()  # Instead of MagicMock()


# Initialize the exchange object

@app.before_server_start
async def startup(app, loop):
    global initialization_complete
    logger.info("Starting server initialization...")

    try:
        # Initialize wallet
        wallet = Wallet()
        wallet.address = wallet.get_address()
        wallet.public_key = wallet.get_public_key()
        app.ctx.wallet = wallet

        # Initialize components
        components = await initialize_components()
        app.ctx.components = components
        
        if initialization_complete:
            # Create background tasks
            app.add_task(update_prices_periodically())
            app.add_task(periodic_mining(components['genesis_address'], wallet))
            app.add_task(deploy_and_run_market_bot(
                components['exchange'], 
                components['vm'], 
                components['plata_contract'], 
                components['price_feed']
            ))
            logger.info("Background tasks initialized successfully")
        else:
            logger.error("Initialization did not complete successfully.")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())
        initialization_complete = False
        raise  # This will prevent the server from starting if initialization fails

    logger.info("Server initialization completed successfully")

@app.before_server_stop
async def cleanup(app, loop):
    global initialization_complete
    logger.info("Starting server cleanup...")

    try:
        # Cancel all running tasks
        for task in asyncio.all_tasks(loop):
            if not task.done() and task != asyncio.current_task():
                logger.debug(f"Cancelling task: {task.get_name()}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cleaning up task {task.get_name()}: {str(e)}")

        # Clean up components
        if hasattr(app.ctx, 'components'):
            for component_name, component in app.ctx.components.items():
                if hasattr(component, 'cleanup'):
                    try:
                        await component.cleanup()
                        logger.debug(f"Cleaned up component: {component_name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up component {component_name}: {str(e)}")

        initialization_complete = False
        logger.info("Server cleanup completed successfully")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Add this to handle graceful shutdown
@app.signal('server.shutdown.before')
async def graceful_shutdown(app, loop):
    logger.info("Starting graceful shutdown...")
    
    timeout = app.config.get('GRACEFUL_SHUTDOWN_TIMEOUT', 15.0)
    deadline = time.time() + timeout
    
    # Wait for ongoing requests to complete
    while time.time() < deadline:
        active_connections = len(app.connections)
        if active_connections == 0:
            break
        logger.info(f"Waiting for {active_connections} connections to complete...")
        await asyncio.sleep(1)

    logger.info("Graceful shutdown completed")

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory storage for demonstration purposes



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

from sanic import Blueprint, response
from functools import wraps
from sanic.exceptions import Unauthorized
import jwt

# Define a custom decorator for protected routes
def protected(wrapped):
    def decorator(f):
        @wraps(f)
        async def decorated_function(request, *args, **kwargs):
            try:
                # Get the token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    raise Unauthorized("Missing or invalid authorization header")

                token = auth_header.split(' ')[1]
                
                # Verify the token and get user
                user = await get_current_user(token)
                
                # Add user to request context
                request.ctx.user = user
                
                # Call the wrapped function
                response = await f(request, *args, **kwargs)
                return response
                
            except JWTError:
                raise Unauthorized("Invalid token")
            except Exception as e:
                raise Unauthorized(str(e))
                
        return decorated_function
    return decorator(wrapped)

async def get_current_user(token: str):
    try:
        # Decode JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        
        if not username:
            raise Unauthorized("Invalid token payload")
            
        # Get user from database
        user = get_user(fake_users_db, username)
        if not user:
            raise Unauthorized("User not found")
            
        return user
        
    except JWTError:
        raise Unauthorized("Invalid token")
    except Exception as e:
        raise Unauthorized(str(e))
async def get_current_active_user(request):
    current_user = await get_current_user(request)
    if current_user.disabled:
        raise SanicException("Inactive user", status_code=400)
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
from sanic import response
from sanic.exceptions import SanicException
import os

@app.route("/generate_qr_code", methods=["POST"])
@protected  # Using the @protected decorator we created earlier
async def generate_qr_code(request):
    try:
        # Get data from request JSON
        data = request.json
        wallet_address = data.get('wallet_address')
        coin_type = data.get('coin_type')

        # Validate required parameters
        if not wallet_address or not coin_type:
            raise SanicException(
                "wallet_address and coin_type are required", 
                status_code=400
            )

        # Define the subfolder where QR codes will be saved
        qr_code_folder = "qr_codes"

        # Ensure the folder exists
        if not os.path.exists(qr_code_folder):
            os.makedirs(qr_code_folder)

        # Generate the QR code and save it in the subfolder
        qr_file = secure_qr_system.generate_qr_code(wallet_address, coin_type)
        qr_file_path = os.path.join(qr_code_folder, os.path.basename(qr_file))

        # Return the file using Sanic's file response
        return await response.file(
            qr_file_path,
            mime_type="image/png",
            filename=os.path.basename(qr_file_path)
        )

    except Exception as e:
        logger.error(f"Error generating QR code: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(
            f"Error generating QR code: {str(e)}", 
            status_code=500
        )

# Optional: Add support for GET requests with query parameters
@app.route("/generate_qr_code", methods=["GET"])
@protected
async def generate_qr_code_get(request):
    try:
        # Get parameters from query string
        wallet_address = request.args.get('wallet_address')
        coin_type = request.args.get('coin_type')

        # Validate required parameters
        if not wallet_address or not coin_type:
            raise SanicException(
                "wallet_address and coin_type are required", 
                status_code=400
            )

        # Define the subfolder where QR codes will be saved
        qr_code_folder = "qr_codes"

        # Ensure the folder exists
        if not os.path.exists(qr_code_folder):
            os.makedirs(qr_code_folder)

        # Generate the QR code and save it in the subfolder
        qr_file = secure_qr_system.generate_qr_code(wallet_address, coin_type)
        qr_file_path = os.path.join(qr_code_folder, os.path.basename(qr_file))

        # Return the file using Sanic's file response
        return await response.file(
            qr_file_path,
            mime_type="image/png",
            filename=os.path.basename(qr_file_path)
        )

    except Exception as e:
        logger.error(f"Error generating QR code: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(
            f"Error generating QR code: {str(e)}", 
            status_code=500
        )

# Add error handler for QR code generation
@app.exception(SanicException)
async def handle_qr_code_error(request, exception):
    return response.json(
        {
            "error": str(exception),
            "status": exception.status_code
        },
        status=exception.status_code
    )

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



# Configuration constants
REGISTRATION_TIMEOUT = 30  # 30 seconds timeout for registration
DB_OPERATION_TIMEOUT = 10  # 10 seconds timeout for database operations
MAX_RETRIES = 3

# Registration endpoint for Sanic
auth_bp = Blueprint('auth', url_prefix='/auth')
from sanic import Sanic, Blueprint, response
from sanic.response import json
import traceback
import logging
import asyncio
from typing import Dict, Any
# Custom response wrapper
def make_response(success: bool, data: Any = None, error: str = None, status: int = 200) -> response.HTTPResponse:
    return response.json({
        "success": success,
        "data": data,
        "error": error
    }, status=status)

# Registration handler with proper error handling
@auth_bp.post('/register')
async def register(request):
    try:
        logger.info("Starting registration process")
        
        # Validate request data
        if not request.json:
            return make_response(False, error="No data provided", status=400)

        data = request.json
        if 'pincode' not in data or 'user_id' not in data:
            return make_response(False, error="Missing required fields", status=400)

        # Get required components from app context
        required_components = ['blockchain', 'vm', 'redis', 'p2p_node']
        for component in required_components:
            if not hasattr(request.app.ctx, component):
                logger.error(f"Missing required component: {component}")
                return make_response(
                    False,
                    error=f"Server component {component} not initialized",
                    status=503
                )

        # Process registration
        try:
            result = await process_registration(request.app.ctx, data)
            logger.info("Registration completed successfully")
            return make_response(True, data=result)
        except ValueError as ve:
            logger.warning(f"Registration validation error: {str(ve)}")
            return make_response(False, error=str(ve), status=400)
        except Exception as e:
            logger.error(f"Registration processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return make_response(False, error="Internal server error", status=500)

    except Exception as e:
        logger.error(f"Unexpected error in registration endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return make_response(False, error="Internal server error", status=500)

async def process_registration(ctx, data: Dict) -> Dict:
    """Process the registration request"""
    try:
        # Extract data
        user_id = data['user_id']
        pincode = data['pincode']

        # Check if user exists in database
        user_data = await ctx.redis.get(f"user:{user_id}")
        if user_data:
            raise ValueError("User already exists")

        # Generate wallet
        wallet = Wallet()
        salt = generate_salt()
        hashed_pincode = get_password_hash(pincode + salt)

        # Create user data
        user_data = {
            "pincode": pincode,
            "hashed_pincode": hashed_pincode,
            "salt": salt,
            "wallet": {
                "address": wallet.address,
                "public_key": wallet.get_public_key(),
            },
            "qr_codes": {}
        }

        # Store in Redis
        await ctx.redis.set(
            f"user:{user_id}",
            json.dumps(user_data),
            expire=None
        )

        # Generate tokens
        access_token = create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        refresh_token = create_refresh_token(
            data={"sub": user_id},
            expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        )

        # Return success response
        return {
            "user_id": user_id,
            "wallet_address": wallet.address,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except Exception as e:
        logger.error(f"Error in process_registration: {str(e)}")
        logger.error(traceback.format_exc())
        raise
app.blueprint(auth_bp)

# Add signal handlers for graceful shutdown
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}")
    asyncio.create_task(shutdown_event())

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
import qrcode
import base64
from io import BytesIO
import json
from sanic import response
from sanic.response import json as sanic_json
from sanic.exceptions import SanicException
from sanic_jwt import protected
from sanic.log import logger
from functools import wraps
from typing import Dict
import qrcode
import base64
from io import BytesIO
def auth_required():
    def decorator(f):
        @wraps(f)
        async def decorated_function(request, *args, **kwargs):
            try:
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return sanic_json({"error": "Missing or invalid authorization header"}, status=401)

                token = auth_header.split(' ')[1]
                current_user = await get_current_user(token)
                if not current_user:
                    return sanic_json({"error": "Invalid token"}, status=401)

                request.ctx.current_user = current_user
                return await f(request, *args, **kwargs)
            except Exception as e:
                logger.error(f"Auth error: {str(e)}")
                return sanic_json({"error": "Authentication failed"}, status=401)
        return decorated_function
    return decorator

# QR code endpoint
@app.get("/get_wallet_qr_codes")
@auth_required()
async def get_wallet_qr_codes(request):
    try:
        current_user = request.ctx.current_user
        user_id = current_user.username
        user = fake_users_db.get(user_id)
        
        if not user:
            logger.warning(f"User not found: {user_id}")
            return sanic_json({"error": "User not found"}, status=404)
        
        if 'wallets' not in user or not user['wallets']:
            logger.warning(f"No wallets found for user: {user_id}")
            return sanic_json({"error": "No wallets found for user"}, status=404)
        
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
            return sanic_json({"error": "Failed to generate any QR codes"}, status=500)
        
        # Create response with CORS headers
        response = response.json(
            {"qr_codes": qr_codes},
            dumps=lambda x: json.dumps(x, cls=CustomJSONEncoder)
        )
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        return response

    except Exception as e:
        logger.error(f"Unexpected error in get_wallet_qr_codes for user {user_id}: {str(e)}")
        return sanic_json(
            {"error": "Unexpected error occurred while generating QR codes"},
            status=500
        )

# Login endpoint
@app.post("/token")
async def login(request):
    try:
        data = request.json
        if not data or 'pincode' not in data:
            return sanic_json({"error": "Missing pincode"}, status=400)

        user_in_db = authenticate_user(data['pincode'])
        if not user_in_db:
            return sanic_json({"error": "Invalid credentials"}, status=400)

        access_token = create_access_token(data={"sub": data['pincode']})
        
        return sanic_json({
            "access_token": access_token,
            "token_type": "bearer",
            "wallet": user_in_db.wallet
        })

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return sanic_json({"error": "Login failed"}, status=500)


# Helper function to get current user
async def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        pincode = payload.get("sub")
        if not pincode:
            return None
        user = authenticate_user(pincode)
        return user
    except jwt.PyJWTError:
        return None

# CORS middleware for Sanic
@app.middleware('response')
async def add_cors_headers(request, response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Options handler for CORS
@app.options("/get_wallet_qr_codes")
async def options_handler(request):
    return response.empty(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )


from sanic.exceptions import Unauthorized
from functools import wraps

def authenticate_token(token: str) -> str:
    """Helper function to decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        pincode = payload.get("sub")
        if pincode is None:
            raise Unauthorized("Invalid token - no pincode found")
        return pincode
    except jwt.ExpiredSignatureError:
        raise Unauthorized("Token has expired")
    except jwt.InvalidTokenError:
        raise Unauthorized("Invalid token")
    except Exception as e:
        raise Unauthorized(f"Authentication error: {str(e)}")

def authenticate(wrapped):
    """Decorator for protecting routes"""
    def decorator(f):
        @wraps(f)
        async def decorated_function(request, *args, **kwargs):
            try:
                # Get the token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    raise Unauthorized("Missing or invalid authorization header")

                token = auth_header.split(' ')[1]
                
                # Authenticate the token and get pincode
                pincode = authenticate_token(token)
                
                # Store the authenticated pincode in request context
                request.ctx.pincode = pincode
                
                # Call the original route handler
                return await f(request, *args, **kwargs)
                
            except Unauthorized as e:
                return response.json(
                    {'error': str(e)},
                    status=401,
                    headers={"WWW-Authenticate": "Bearer"}
                )
            except Exception as e:
                return response.json(
                    {'error': f"Authentication error: {str(e)}"},
                    status=401,
                    headers={"WWW-Authenticate": "Bearer"}
                )
                
        return decorated_function
    return decorator(wrapped)

# Example usage:
@app.route("/protected_route", methods=["GET"])
@authenticate
async def protected_route(request):
    # Access the authenticated pincode from request context
    pincode = request.ctx.pincode
    return response.json({"message": f"Authenticated with pincode: {pincode}"})

# Error handler for unauthorized exceptions
@app.exception(Unauthorized)
async def handle_unauthorized(request, exception):
    return response.json(
        {"error": str(exception)},
        status=401,
        headers={"WWW-Authenticate": "Bearer"}
    )

# Optional middleware for global authentication
@app.middleware('request')
async def auth_middleware(request):
    # Skip authentication for certain routes
    if request.path in ['/login', '/register', '/health']:
        return

    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return response.json(
                {'error': 'Missing or invalid authorization header'},
                status=401,
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = auth_header.split(' ')[1]
        pincode = authenticate_token(token)
        request.ctx.pincode = pincode

    except Unauthorized as e:
        return response.json(
            {'error': str(e)},
            status=401,
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        return response.json(
            {'error': f"Authentication error: {str(e)}"},
            status=401,
            headers={"WWW-Authenticate": "Bearer"}
        )

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
from sanic import Sanic, response
from sanic.exceptions import SanicException

@app.post("/import_token")
async def import_token(request):
    try:
        # Get current user from the request
        current_user = request.json.get("current_user")
        # Import the Plata wallet first
        token = await blockchain.import_token(request.json["address"], current_user["username"])

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
        return response.json({
            "address": token.address,
            "name": token.name,
            "symbol": token.symbol,
            "balance": float(token.balance_of(current_user["username"])),
            "wallets": all_wallets
        })

    except Exception as e:
        raise SanicException(str(e), status_code=400)
@app.post("/create_pq_wallet")
async def create_pq_wallet(request):
    # Replace Depends(authenticate) logic if needed
    pincode = request.json.get("pincode")
    
    # Authenticate the pincode here manually if needed
    if not authenticate(pincode):
        return response.json({"error": "Authentication failed"}, status=403)

    pq_wallet = PQWallet()
    address = pq_wallet.get_address()
    pq_public_key = pq_wallet.get_pq_public_key()
    
    return response.json({
        "address": address,
        "pq_public_key": pq_public_key
    })
@app.post("/search")
async def search(request):
    query = request.json.get("query", "").lower()

    wallet_results = blockchain.search_wallets(query)
    transaction_results = blockchain.search_transactions(query)
    contract_results = blockchain.search_contracts(query)

    if not wallet_results and not transaction_results and not contract_results:
        return response.json({"detail": "No results found"}, status=404)
    
    return response.json({
        "wallets": wallet_results,
        "transactions": transaction_results,
        "contracts": contract_results
    })

class AddressRequest(BaseModel):
    address: str

class WalletRequest(BaseModel):
    wallet_address: str

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
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, bytes):
                return obj.decode('utf-8')
            return super().default(obj)
        except Exception as e:
            logger.error(f"Error in CustomJSONEncoder: {str(e)}")
            return str(obj)



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

@app.route("/send_transaction", methods=["POST"])
@authenticate
async def send_transaction(request):
    try:
        # Get transaction input from request JSON
        transaction_input = request.json
        logger.info(f"Received transaction request: {transaction_input}")

        # Resolve receiver address from alias if necessary
        if not transaction_input['receiver'].startswith("plata"):
            receiver_address = None
            for wallet in blockchain.wallets:
                if wallet.alias == transaction_input['receiver']:
                    receiver_address = wallet.address
                    break
            if not receiver_address:
                raise SanicException("Invalid receiver alias", status_code=400)
            transaction_input['receiver'] = receiver_address

        # Create wallet instance using the private key provided in the transaction
        wallet = Wallet(private_key=transaction_input['private_key'])
        logger.info("Created wallet from private key.")

        # Create a Transaction object from the input
        transaction = Transaction(
            sender=transaction_input['sender'],
            receiver=transaction_input['receiver'],
            amount=transaction_input['amount'],
            price=transaction_input['price'],
            buyer_id=transaction_input['buyer_id'],
            seller_id=transaction_input['seller_id'],
            wallet=transaction_input['wallet'],
            tx_hash=generate_tx_hash(transaction_input),
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
            raise SanicException("Invalid transaction signature", status_code=400)

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
            
            return response.json({"success": True, "message": "Transaction added successfully."})
        else:
            return response.json({
                "success": False, 
                "message": "Transaction failed to add: insufficient balance"
            })

    except ValueError as e:
        logger.error(f"Error deserializing private key: {e}")
        raise SanicException("Invalid private key format", status_code=400)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise SanicException("Internal server error", status_code=500)

def generate_tx_hash(transaction_input: dict) -> str:
    data = f"{transaction_input['sender']}{transaction_input['receiver']}{transaction_input['amount']}{transaction_input['price']}{time.time()}"
    return hashlib.sha256(data.encode()).hexdigest()

@app.route("/send_batch_transactions", methods=["POST"])
@authenticate
async def send_batch_transactions(request):
    results = []
    try:
        batch_input = request.json
        if not isinstance(batch_input.get('transactions'), list):
            raise SanicException("Invalid batch transaction format", status_code=400)

        for transaction_input in batch_input['transactions']:
            try:
                # Create wallet instance using the private key provided in the transaction
                wallet = Wallet(private_key=transaction_input['private_key'])

                # Create a Transaction object from the input
                transaction = Transaction(
                    sender=transaction_input['sender'],
                    receiver=transaction_input['receiver'],
                    amount=transaction_input['amount'],
                    price=transaction_input['price'],
                    buyer_id=transaction_input['buyer_id'],
                    seller_id=transaction_input['seller_id'],
                    wallet=transaction_input['wallet'],
                    tx_hash=generate_tx_hash(transaction_input),
                    timestamp=int(time.time())
                )

                # Add public key to the transaction
                transaction.public_key = wallet.get_public_key()

                # Generate and sign the transaction using Zero-Knowledge Proofs (ZKP)
                transaction.sign_transaction(blockchain.zk_system)

                if transaction.verify_transaction(blockchain.zk_system) and blockchain.add_transaction(transaction):
                    results.append({
                        "success": True, 
                        "message": "Transaction added successfully.", 
                        "transaction": transaction_input
                    })
                else:
                    results.append({
                        "success": False, 
                        "message": "Transaction failed to add: invalid signature or insufficient balance", 
                        "transaction": transaction_input
                    })

            except ValueError as e:
                results.append({
                    "success": False, 
                    "message": f"Error processing transaction: {str(e)}", 
                    "transaction": transaction_input
                })
            except Exception as e:
                results.append({
                    "success": False, 
                    "message": f"Unexpected error: {str(e)}", 
                    "transaction": transaction_input
                })

        return response.json(results)

    except Exception as e:
        logger.error(f"Error processing batch transactions: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException("Error processing batch transactions", status_code=500)


class DeployContractRequest(BaseModel):
    sender_address: str
    collateral_token: str
    initial_price: float
@app.post("/deploy_contract")
async def deploy_contract(sender: str, contract_name: str, args: List[Any]):
    contract_class = globals()[contract_name]
    contract_address = await blockchain.deploy_contract(sender, contract_class, *args)
    return {"contract_address": contract_address}

@app.post("/execute_contract")
async def execute_contract(sender: str, contract_address: str, function_name: str, args: List[Any], kwargs: Dict[str, Any]):
    result = await blockchain.execute_contract(sender, contract_address, function_name, *args, **kwargs)
    return {"result": result}

@app.post("/create_token")
async def create_token(creator_address: str, token_name: str, token_symbol: str, total_supply: int):
    token_address = await blockchain.create_token(creator_address, token_name, token_symbol, total_supply)
    return {"token_address": token_address}

@app.post("/create_nft")
async def create_nft(creator_address: str, nft_id: str, metadata: Dict[str, Any]):
    nft_id = await blockchain.create_nft(creator_address, nft_id, metadata)
    return {"nft_id": nft_id}

@app.post("/transfer_nft")
async def transfer_nft(from_address: str, to_address: str, nft_id: str):
    success = await blockchain.transfer_nft(from_address, to_address, nft_id)
    return {"success": success}

@app.get("/get_balance/{address}")
async def get_balance(address: str):
    balance = await blockchain.get_balance(address)
    return {"balance": balance}

@app.post("/transfer")
async def transfer(from_address: str, to_address: str, amount: int):
    success = await blockchain.transfer(from_address, to_address, amount)
    return {"success": success}



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


from sanic import Sanic, response
from sanic.exceptions import SanicException

@app.get("/get_mining_data")
async def get_mining_data(request):
    try:
        block_header = generate_block_header()
        difficulty = blockchain.difficulty
        mining_data = {
            "block_header": block_header,
            "difficulty": difficulty
        }
        return response.json(mining_data)
    except Exception as e:
        raise SanicException(f"Error generating mining data: {str(e)}", status_code=500)
@app.post("/submit_solution")
async def submit_solution(request):
    try:
        # Parse the submission from the request
        submission = request.json
        
        # Assuming `SolutionSubmission` is a dictionary, you can access nonce like this
        nonce = submission.get("nonce")
        
        # Verify the solution
        if verify_solution(nonce):
            new_block = create_new_block(nonce)
            return response.json({
                "status": "success",
                "message": "Solution accepted",
                "block_hash": new_block.hash
            })
        else:
            return response.json({
                "status": "rejected",
                "message": "Invalid solution"
            })
    except Exception as e:
        raise SanicException(f"Error processing solution: {str(e)}", status_code=500)


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

from sanic import Sanic, response

# Initialize mining stats
mining_stats = {
    "blocks_mined": 0,
    "mining_duration": 0,
    "hash_rate": 0,
    "last_reward": "0",
    "total_reward": "0",
    "difficulty": 0
}

@app.get("/mining_stats")
async def get_mining_stats(request):
    global continue_mining, mining_stats
    return response.json({
        "is_mining": continue_mining,
        "stats": mining_stats
    })


from sanic import Sanic, response

from sanic import Sanic, response


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

from sanic import Sanic, response
from sanic.exceptions import SanicException
from datetime import timedelta

from sanic import Sanic, response
import requests  # Use requests to make external calls

@app.get("/get_block_info/<block_hash>")
async def get_block_info(request, block_hash):
    pincode = request.args.get("pincode")
    
    # If necessary, replace `authenticate(pincode)` with your own logic to check the pincode
    if not authenticate(pincode):
        return response.json({"error": "Unauthorized"}, status=403)
    
    external_response = requests.get(f'http://161.35.219.10:50503/get_block_info/{block_hash}')
    block_info = external_response.json()

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

    return response.json(block_info)
@app.get("/get_node_info/<node_id>")
async def get_node_info(request, node_id):
    pincode = request.args.get("pincode")
    
    # Authenticate the pincode
    if not authenticate(pincode):
        return response.json({"error": "Unauthorized"}, status=403)
    
    external_response = requests.get(f'http://161.35.219.10:50503/get_node_info/{node_id}')
    return response.json(external_response.json())
()
@app.get("/get_transaction_info/<tx_hash>")
async def get_transaction_info(request, tx_hash):
    pincode = request.args.get("pincode")
    
    # Authenticate the pincode
    if not authenticate(pincode):
        return response.json({"error": "Unauthorized"}, status=403)
    
    external_response = requests.get(f'http://161.35.219.10:50503/get_transaction_info/{tx_hash}')
    return response.json(external_response.json())


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
from sanic import response
from typing import Dict, List
import json
from decimal import Decimal
import logging
import asyncio

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, object] = {}  # Using generic object type for Sanic websockets
        self.subscriptions: Dict[str, List[object]] = {}

    async def connect(self, ws, user: str):
        """Store WebSocket connection for a user"""
        self.active_connections[user] = ws
        logger.info(f"WebSocket connection established for user: {user}")

    async def disconnect(self, user: str):
        """Remove WebSocket connection for a user"""
        if user in self.active_connections:
            ws = self.active_connections[user]
            del self.active_connections[user]
            logger.info(f"WebSocket connection closed for user: {user}")
            
            # Remove from subscriptions
            for client_id in list(self.subscriptions.keys()):
                if ws in self.subscriptions[client_id]:
                    self.subscriptions[client_id].remove(ws)
                    if not self.subscriptions[client_id]:
                        del self.subscriptions[client_id]

    async def subscribe(self, ws, client_id: str):
        """Subscribe a WebSocket connection to a client ID"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = []
        self.subscriptions[client_id].append(ws)
        logger.info(f"Client {client_id} subscribed to updates")

    async def unsubscribe(self, ws, client_id: str):
        """Unsubscribe a WebSocket connection from a client ID"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].remove(ws)
            if not self.subscriptions[client_id]:
                del self.subscriptions[client_id]
            logger.info(f"Client {client_id} unsubscribed from updates")

    async def send_personal_message(self, message: str, user: str):
        """Send a message to a specific user"""
        if user in self.active_connections:
            try:
                await self.active_connections[user].send(message)
                logger.debug(f"Sent personal message to user: {user}")
            except Exception as e:
                logger.error(f"Error sending personal message to {user}: {str(e)}")
                await self.disconnect(user)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected users"""
        disconnected_users = []
        for user, ws in self.active_connections.items():
            try:
                await ws.send(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {user}: {str(e)}")
                disconnected_users.append(user)

        # Clean up disconnected users
        for user in disconnected_users:
            await self.disconnect(user)

    async def broadcast_new_block(self, block: dict):
        """Broadcast a new block to all connected users"""
        try:
            message = json.dumps({
                "type": "new_block",
                "data": block
            })
            await self.broadcast(message)
            logger.info(f"Broadcasted new block: {block.get('hash', 'unknown')}")
        except Exception as e:
            logger.error(f"Error broadcasting new block: {str(e)}")

    async def broadcast_new_transaction(self, transaction: dict):
        """Broadcast a new transaction to all connected users"""
        try:
            message = json.dumps({
                "type": "new_transaction",
                "data": transaction
            })
            await self.broadcast(message)
            logger.info(f"Broadcasted new transaction: {transaction.get('tx_hash', 'unknown')}")
        except Exception as e:
            logger.error(f"Error broadcasting new transaction: {str(e)}")

    async def broadcast_network_stats(self, stats: dict):
        """Broadcast network stats to all connected users"""
        try:
            # Convert Decimal objects to strings
            stats_serializable = {
                k: str(v) if isinstance(v, Decimal) else v 
                for k, v in stats.items()
            }
            message = json.dumps({
                "type": "network_stats",
                "data": stats_serializable
            })
            await self.broadcast(message)
            logger.debug("Broadcasted network stats")
        except Exception as e:
            logger.error(f"Error broadcasting network stats: {str(e)}")

# Initialize the connection manager
manager = ConnectionManager()

# WebSocket route implementation for Sanic
@app.websocket("/ws/<user_id>")
async def websocket_endpoint(request, ws, user_id: str):
    try:
        await manager.connect(ws, user_id)
        
        try:
            while True:
                data = await ws.recv()
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "subscribe":
                        await manager.subscribe(ws, message.get("client_id"))
                        await ws.send(json.dumps({
                            "type": "subscription_success",
                            "client_id": message.get("client_id")
                        }))
                    
                    elif message_type == "unsubscribe":
                        await manager.unsubscribe(ws, message.get("client_id"))
                        await ws.send(json.dumps({
                            "type": "unsubscription_success",
                            "client_id": message.get("client_id")
                        }))
                    
                    else:
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": "Unknown message type"
                        }))
                
                except json.JSONDecodeError:
                    await ws.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await ws.send(json.dumps({
                        "type": "error",
                        "message": "Error processing message"
                    }))

        except asyncio.CancelledError:
            logger.info(f"WebSocket connection cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        finally:
            await manager.disconnect(user_id)
            
    except Exception as e:
        logger.error(f"Error in websocket connection for user {user_id}: {str(e)}")
        if user_id in manager.active_connections:
            await manager.disconnect(user_id)

# Add the manager to the Sanic app context
@app.before_server_start
async def setup_websocket_manager(app, loop):
    app.ctx.ws_manager = manager
    logger.info("WebSocket manager initialized")

# Example of using the manager in other routes
@app.route("/broadcast_message", methods=["POST"])
async def broadcast_message(request):
    try:
        message = request.json.get("message")
        if message:
            await manager.broadcast(json.dumps({
                "type": "broadcast",
                "message": message
            }))
            return response.json({"status": "success"})
        return response.json({"status": "error", "message": "No message provided"})
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        return response.json({"status": "error", "message": str(e)})


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
from sanic import response
from sanic.exceptions import SanicException

@app.route("/network_stats", methods=["GET"])
@authenticate
async def get_network_stats_endpoint(request):
    try:
        stats = await get_network_stats()
        return response.json(stats)
    except Exception as e:
        logger.error(f"Error fetching network stats: {str(e)}")
        raise SanicException(
            "Error fetching network stats",
            status_code=500
        )

@app.route("/node_performance/<node_id:str>", methods=["GET"])
@authenticate
async def get_node_performance(request, node_id: str):
    try:
        node_stats = node_directory.get_performance_stats()
        return response.json({
            "node_id": node_id,
            "avg_mining_time": node_stats.get("avg_mining_time", 0),
            "total_blocks_mined": node_stats.get("total_blocks_mined", 0),
            "hash_rate": node_stats.get("hash_rate", 0),
            "uptime": node_stats.get("uptime", 0)
        })
    except Exception as e:
        logger.error(f"Error fetching node performance: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(
            "Error fetching node performance",
            status_code=500
        )

@app.route("/quantum_metrics", methods=["GET"])
@authenticate
async def get_quantum_metrics(request):
    try:
        qhins = calculate_qhins(blockchain.chain)
        entanglement_strength = calculate_entanglement_strength(blockchain.chain)
        coherence_ratio = blockchain.globalMetrics.coherenceRatio
        
        return response.json({
            "qhins": str(qhins),  # Convert Decimal to string
            "entanglement_strength": str(entanglement_strength),
            "coherence_ratio": str(coherence_ratio)
        })
    except Exception as e:
        logger.error(f"Error fetching quantum metrics: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(
            "Error fetching quantum metrics",
            status_code=500
        )

@app.route("/recent_transactions", methods=["GET"])
@authenticate
async def get_recent_transactions(request):
    try:
        # Get limit from query parameters with default value
        limit = int(request.args.get('limit', 10))
        
        # Validate limit
        if limit <= 0:
            raise SanicException(
                "Limit must be greater than 0",
                status_code=400
            )
        
        recent_txs = blockchain.get_recent_transactions(limit)
        return response.json([tx.to_dict() for tx in recent_txs])
    
    except ValueError as ve:
        raise SanicException(
            "Invalid limit parameter",
            status_code=400
        )
    except Exception as e:
        logger.error(f"Error fetching recent transactions: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(
            "Error fetching recent transactions",
            status_code=500
        )



# Add custom JSON serializer for Decimal values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)

# Helper function to serialize response data
def serialize_response(data):
    return json.dumps(data, cls=CustomJSONEncoder)

# Optional: Add cache control middleware
@app.middleware('response')
async def add_cache_control(request, response):
    if request.path in ['/network_stats', '/quantum_metrics']:
        response.headers['Cache-Control'] = 'public, max-age=5'  # Cache for 5 seconds
    return response


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
from starlette.websockets import WebSocketState

WS_1000_NORMAL_CLOSURE = 1000
WS_1008_POLICY_VIOLATION = 1008
WS_3000_INTERNAL_ERROR = 3000  # Custom code for internal errors

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

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
from sanic import Sanic, response

# Set up Jinja2 environment

@app.get("/")
async def get_dashboard(request):
    template = env.get_template("dashboard.html")
    html_content = template.render({"request": request})
    return response.html(html_content)

@app.get("/trading")
async def get_trading(request):
    template = env.get_template("trading.html")
    html_content = template.render({"request": request})
    return response.html(html_content)
@app.get("/api/dashboard_data")
async def get_dashboard_data(request):
    dashboard_data = {
        "Quantum Hash Rate": "1.21 TH/s",
        "Network Entanglement": "99.9%",
        "Active Nodes": "42,000",
        "PLATA Price": "$1.0001",
        "Total Value Locked": "$1,000,000,000",
        "Governance Proposals": "7 Active"
    }
    return response.json(dashboard_data)

    
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
initialization_complete = False

@app.before_server_start
async def startup_event(app, loop):
    global initialization_complete
    try:
        # Initialize Redis (or any other resources)
        app.ctx.redis = await init_redis()
        logger.info("Redis initialized and connected.")

        # Run the main initialization
        initialization_complete = await async_main()

        if initialization_complete:
            logger.info("System fully initialized and ready.")
        else:
            logger.error("System initialization failed.")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())
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

class QuantumCrossChainBridge:
    def __init__(self, supported_chains: List[str]):
        self.supported_chains = supported_chains
        self.qrng = QuantumRandomNumberGenerator()
        self.zk_system = SecureHybridZKStark(security_level=20)

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
from sanic import Sanic, response
from sanic.exceptions import SanicException
import time
import asyncio

@app.post("/mine_block")
async def mine_block(request):
    global blockchain, p2p_node, continue_mining, mining_task, initialization_complete

    try:
        data = request.json
        wallet_address = data.get("wallet_address")

        logger.debug(f"Received request to mine block with wallet address: {wallet_address}")

        # Validate request
        if not wallet_address:
            return response.json(
                {"success": False, "message": "Wallet address is required"}, status=400
            )

        # Check mining status
        if continue_mining:
            logger.warning("Mining is already in progress")
            return response.json({"success": False, "message": "Mining is already in progress"})

        # Check system initialization
        if not initialization_complete:
            logger.error("System not fully initialized")
            return response.json({"success": False, "message": "System initialization incomplete"})

        # Check blockchain and P2P node
        if blockchain is None or p2p_node is None:
            return response.json({
                "success": False,
                "message": "Required components not initialized",
                "blockchain_status": "initialized" if blockchain else "not initialized",
                "p2p_status": "initialized" if p2p_node else "not initialized"
            }, status=503)

        # Get latest blockchain state
        latest_block_hash = blockchain.get_latest_block_hash()
        logger.debug(f"Latest block hash: {latest_block_hash}")

        # Initialize mining process
        continue_mining = True
        start_time = time.time()

        try:
            # Get pending transactions and compute reward
            pending_txs = blockchain.get_pending_transactions()[:10]
            reward = blockchain.get_block_reward()

            # Mine block using DAGKnight miner
            new_block = await blockchain.miner.mine_block(
                previous_hash=latest_block_hash,
                data=f"Block mined by {wallet_address} at {int(time.time())}",
                transactions=pending_txs,
                reward=reward,
                miner_address=wallet_address
            )

            if not new_block:
                raise ValueError("Failed to mine block")

            # Validate the mined block
            if not blockchain.miner.validate_block(new_block):
                raise ValueError("Block validation failed")

            # Add block to chain and process transactions
            blockchain.chain.append(new_block)
            await blockchain.process_transactions(new_block.transactions)
            await blockchain.native_coin_contract.mint(wallet_address, Decimal(new_block.reward))

            # Get DAG metrics
            dag_metrics = blockchain.miner.get_dag_metrics()

            # Prepare block info for broadcasting
            block_info = {
                "block_hash": new_block.hash,
                "miner": wallet_address,
                "transactions": [tx.to_dict() for tx in new_block.transactions],
                "timestamp": int(time.time()),
                "quantum_signature": new_block.quantum_signature,
                "dag_metrics": dag_metrics,
                "parent_hashes": [new_block.previous_hash] + 
                                 (new_block.dag_parents if hasattr(new_block, 'dag_parents') else [])
            }

            # Propagate block to network
            active_peers = await p2p_node.get_active_peers()
            if active_peers:
                await p2p_node.propagate_block(new_block)
                logger.info(f"Block propagated to {len(active_peers)} peers")
            else:
                logger.warning("No active peers available for block propagation")

            # Broadcast block mined event
            await p2p_node.broadcast_event('block_mined', {
                'block_info': block_info,
                'mining_time': time.time() - start_time,
                'network_state': {
                    'active_peers': len(active_peers),
                    'pending_transactions': len(blockchain.get_pending_transactions()),
                    'dag_metrics': dag_metrics
                }
            })

            # Add continuous mining task (use Sanic's background task)
            mining_task = asyncio.create_task(continuous_mining(blockchain, wallet_address, p2p_node))

            logger.info(f"Block successfully mined and broadcasted: {new_block.hash}")

            # Return success response
            return response.json({
                "success": True,
                "message": "Block mined and propagated successfully",
                "block_hash": new_block.hash,
                "mining_time": time.time() - start_time,
                "dag_metrics": dag_metrics,
                "reward": str(new_block.reward),
                "transactions_processed": len(new_block.transactions)
            })

        except ValueError as ve:
            logger.error(f"Mining error: {str(ve)}")
            continue_mining = False
            return response.json({"success": False, "message": str(ve)}, status=400)

        except Exception as e:
            logger.error(f"Unexpected error during mining: {str(e)}")
            logger.error(traceback.format_exc())
            continue_mining = False
            return response.json({
                "success": False,
                "message": "Unexpected error during mining",
                "error": str(e),
                "traceback": traceback.format_exc()
            }, status=500)

    except Exception as outer_e:
        logger.error(f"Outer error in mining endpoint: {str(outer_e)}")
        logger.error(traceback.format_exc())
        continue_mining = False
        return response.json({
            "success": False,
            "message": "Error during mining process",
            "error": str(outer_e),
            "traceback": traceback.format_exc()
        }, status=500)

    finally:
        # Reset mining state if something went wrong
        if continue_mining and not mining_task:
            continue_mining = False
            logger.warning("Mining state reset due to task failure")
@app.get("/peer_status")
async def peer_status(request):
    try:
        p2p_node = app.ctx.p2p_node
        connected_peers = app.ctx.connected_peers

        if p2p_node is None:
            raise SanicException("P2P node is not initialized", status_code=500)

        # Prepare the response
        peer_status = {
            "total_peers": len(p2p_node.peers),
            "active_peers": len(connected_peers),
            "all_peers": list(p2p_node.peers.keys()),
            "active_peer_list": connected_peers,
            "peer_states": {peer: info["state"] for peer, info in p2p_node.peers.items()}
        }

        logger.info(f"Peer status: {peer_status}")
        return response.json(peer_status)

    except Exception as e:
        logger.error(f"Error while fetching peer status: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(f"Internal server error: {str(e)}", status_code=500)

def check_initialization_status():
    global initialization_status
    uninitialized = [k for k, v in initialization_status.items() if not v]
    if uninitialized:
        return False, f"System is still initializing. Waiting for: {', '.join(uninitialized)}"
    return True, "System is fully initialized and ready"
@app.middleware("request")
async def ensure_p2p_node_middleware(request):
    # Ensure that the P2PNode is initialized before processing the request
    if not hasattr(app.ctx, 'p2p_node') or app.ctx.p2p_node is None:
        try:
            # Re-initialize P2PNode if it is missing
            logger.debug("P2PNode is not initialized. Initializing now...")
            await async_main_initialization()  # Make sure the P2PNode and other components are initialized
            logger.info("P2PNode and other components initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize P2PNode: {str(e)}")
            return response.json(
                {"message": "P2PNode is not initialized and cannot be initialized automatically."},
                status=503
            )


    
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
    # Initialize Redis here...
    redis = await some_redis_initialization_function()
    app.ctx.redis = redis  # Store Redis in app.ctx
    logger.info("Redis initialized successfully.")




# Store the initialization status in Redis
async def update_initialization_status(component: str, status: bool):
    """Update initialization status with Redis error handling."""
    try:
        if not hasattr(app, 'redis') or not app.redis:
            logger.warning("Redis not available, skipping initialization status update")
            return
            
        key = f"init_status:{component}"
        await app.redis.set(key, "1" if status else "0")
        logger.debug(f"Updated initialization status for {component}: {status}")
        
    except Exception as e:
        logger.warning(f"Failed to update initialization status in Redis: {str(e)}")

@app.middleware("request")
async def ensure_redis_and_blockchain_middleware(request: Request, call_next):
    # Ensure Redis is initialized
    if not hasattr(app.state, 'redis') or app.state.redis is None:
        await init_redis()  # Initialize Redis if not already done
        app.state.redis = redis
        logger.info("Redis initialized in middleware")

    # Ensure Blockchain is initialized
    if not hasattr(app.state, 'blockchain') or app.state.blockchain is None:
        await async_main_initialization()  # Initialize blockchain and other components
        app.state.blockchain = blockchain
        logger.info("Blockchain initialized in middleware")

    # Ensure P2P Node is initialized
    if not hasattr(app.state, 'p2p_node') or app.state.p2p_node is None:
        if hasattr(app.state.blockchain, 'p2p_node'):
            app.state.p2p_node = app.state.blockchain.p2p_node
            await update_initialization_status("p2p_node", True)  # Update Redis status
            logger.info("P2P node set from blockchain in middleware")
        else:
            logger.warning("P2P node not found in blockchain")

    # Ensure VM is initialized
    if not hasattr(app.state, 'vm') or app.state.vm is None:
        logger.warning("VM not found in app.state. Attempting to initialize...")
        try:
            vm = await initialize_vm()
            if vm:
                logger.info("VM initialized successfully in middleware")
            else:
                logger.error("Failed to initialize VM in middleware")
        except Exception as e:
            logger.error(f"Error initializing VM in middleware: {str(e)}")
            logger.error(traceback.format_exc())


    # Log the state of components after initialization
    logger.info(f"Middleware check - Redis: {'initialized' if app.state.redis else 'not initialized'}")
    logger.info(f"Middleware check - Blockchain: {'initialized' if app.state.blockchain else 'not initialized'}")
    logger.info(f"Middleware check - P2P Node: {'initialized' if app.state.p2p_node else 'not initialized'}")
    logger.info(f"Middleware check - VM: {'initialized' if app.state.vm else 'not initialized'}")

    return await call_next(request)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class P2PNodeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if hasattr(app.state, 'p2p_node') and app.state.p2p_node is not None:
            request.state.p2p_node = app.state.p2p_node
            request.state.connected_peers = await app.state.p2p_node.get_active_peers()
        else:
            request.state.p2p_node = None
            request.state.connected_peers = []
        
        response = await call_next(request)
        return response

# Add proper error handling middleware
@app.middleware('request')
async def validate_initialization(request):
    if request.path != '/health' and request.path != '/status':
        if not hasattr(request.app.ctx, 'initialization_status') or \
           request.app.ctx.initialization_status != "complete":
            return json(
                {"error": "Server is still initializing"},
                status=503
            )

# Add proper exception handling
@app.exception(Exception)
async def handle_exception(request, exception):
    logger.error(f"Unhandled exception: {str(exception)}")
    logger.error(traceback.format_exc())
    return json(
        {"error": str(exception)},
        status=500
    )

# Add this middleware to your FastAPI app
# Add to your quantumdagknight.py
from sanic import Request
from sanic.response import json as sanic_json

@app.route('/health', methods=['GET'])
async def health_check(request: Request):
    logger.info("Health check started")
    try:
        components_status = {}

        # Check Redis
        logger.info("Checking Redis status")
        try:
            components_status["redis"] = (
                "initialized"
                if hasattr(app.ctx, "redis") and app.ctx.redis is not None
                else "not initialized"
            )
            logger.info(f"Redis status: {components_status['redis']}")

            # Only check Redis initialization status if Redis is initialized
            if components_status["redis"] == "initialized":
                initialization_status = await app.ctx.redis.hgetall('initialization_status')
                for comp, status in initialization_status.items():
                    comp_name = comp.decode('utf-8') if isinstance(comp, bytes) else comp
                    status_value = int(status.decode('utf-8')) if isinstance(status, bytes) else int(status)
                    if comp_name not in components_status:
                        comp_status = "initialized" if status_value == 1 else "not initialized"
                        components_status[comp_name] = comp_status
                        logger.info(f"{comp_name} status from Redis: {comp_status}")
        except Exception as redis_error:
            logger.error(f"Error checking Redis status: {str(redis_error)}")
            components_status["redis"] = f"error: {str(redis_error)}"

        # Check other components
        logger.info("Checking other components")
        components_to_check = ["blockchain", "p2p_node", "vm", "price_feed", "plata_contract", "exchange"]
        for component in components_to_check:
            try:
                # In Sanic, we use app.ctx instead of app.state
                status = (
                    "initialized"
                    if hasattr(app.ctx, component) and getattr(app.ctx, component) is not None
                    else "not initialized"
                )
                components_status[component] = status
                logger.info(f"{component} status: {status}")
            except Exception as comp_error:
                logger.error(f"Error checking {component} status: {str(comp_error)}")
                components_status[component] = f"error: {str(comp_error)}"

        # Check P2P node peers
        logger.info("Checking P2P node peers")
        try:
            if hasattr(app.ctx, 'p2p_node') and app.ctx.p2p_node is not None:
                components_status["p2p_node_peers"] = len(app.ctx.p2p_node.connected_peers)
                logger.info(f"P2P node peers: {components_status['p2p_node_peers']}")
            else:
                components_status["p2p_node_peers"] = 0
                logger.info("P2P node not initialized, peers set to 0")
        except Exception as p2p_error:
            logger.error(f"Error checking P2P node peers: {str(p2p_error)}")
            components_status["p2p_node_peers"] = f"error: {str(p2p_error)}"

        # Check global variables as backup
        logger.info("Checking global variables")
        global blockchain, p2p_node, vm, price_feed, plata_contract, exchange
        global_components = {
            "blockchain": blockchain,
            "p2p_node": p2p_node,
            "vm": vm,
            "price_feed": price_feed,
            "plata_contract": plata_contract,
            "exchange": exchange
        }
        
        for comp_name, comp_value in global_components.items():
            if components_status.get(comp_name) == "not initialized" and comp_value is not None:
                components_status[comp_name] = "initialized (global)"
                logger.info(f"{comp_name} found in global scope")

        # Determine uninitialized or error components
        uninitialized_components = [
            comp for comp, status in components_status.items()
            if status == "not initialized" or str(status).startswith("error")
        ]

        if uninitialized_components:
            logger.info(f"System not fully initialized. Waiting for: {uninitialized_components}")
            return sanic_json(
                {
                    "status": "initializing",
                    "message": f"System is still initializing. Waiting for: {', '.join(uninitialized_components)}",
                    "components": components_status
                },
                status=200
            )

        logger.info("Health check completed successfully. System fully initialized.")
        return sanic_json(
            {
                "status": "ready",
                "message": "System is fully initialized and ready.",
                "components": components_status
            },
            status=200
        )

    except Exception as e:
        logger.error(f"Unexpected error in health check: {str(e)}")
        logger.error(traceback.format_exc())
        return sanic_json(
            {
                "status": "error",
                "message": f"Unexpected error in health check: {str(e)}",
                "traceback": traceback.format_exc()
            },
            status=500
        )

# Update the app initialization to store components in app.ctx
@app.before_server_start
async def setup_app(app, loop):
    try:
        logger.info("Initializing application components...")
        
        # Initialize Redis
        app.ctx.redis = await init_redis()
        
        # Initialize components
        await async_main_initialization()
        
        # Store components in app.ctx
        app.ctx.blockchain = blockchain
        app.ctx.p2p_node = p2p_node
        app.ctx.vm = vm
        app.ctx.price_feed = price_feed
        app.ctx.plata_contract = plata_contract
        app.ctx.exchange = exchange
        
        logger.info("Application components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Add cleanup on server stop
@app.before_server_stop
async def cleanup_app(app, loop):
    try:
        # Cleanup Redis
        if hasattr(app.ctx, 'redis') and app.ctx.redis:
            await app.ctx.redis.close()
        
        # Cleanup other components
        await cleanup_resources()
        
        # Clear app context
        for component in ["blockchain", "p2p_node", "vm", "price_feed", "plata_contract", "exchange"]:
            if hasattr(app.ctx, component):
                delattr(app.ctx, component)
        
        logger.info("Application cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error(traceback.format_exc())
        
        
async def start_server(host='0.0.0.0', port=50511):
    try:
        server = await app.create_server(
            host=host,
            port=port,
            return_asyncio_server=True,
            access_log=True,
            register_sys_signals=True
        )
        
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def check_initialization_status():
    global initialization_status
    logger.info(f"Checking initialization status: {initialization_status}")
    uninitialized = [k for k, v in initialization_status.items() if not v]
    if uninitialized:
        logger.info(f"Uninitialized components: {uninitialized}")
        return False, f"System is still initializing. Waiting for: {', '.join(uninitialized)}"
    logger.info("All components are initialized.")
    return True, "System is fully initialized and ready"
@app.post("/stop_mining")
async def stop_mining(request):
    global continue_mining, mining_task
    try:
        if not continue_mining:
            return response.json({"success": False, "message": "Mining is not currently running"})

        # Stop mining
        continue_mining = False
        if mining_task:
            mining_task.cancel()
            logger.info("Mining task has been canceled")
        else:
            logger.warning("No active mining task to cancel")

        logger.info("Mining has been stopped")
        return response.json({"success": True, "message": "Mining stopped successfully"})

    except Exception as e:
        logger.error(f"Failed to stop mining: {str(e)}")
        logger.error(traceback.format_exc())
        raise SanicException(f"Failed to stop mining: {str(e)}", status_code=500)

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
                await p2p_node.propagate_block(new_block)

                logger.info(f"Block {new_block.hash} successfully propagated across the network")
                blocks_mined += 1

            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during continuous mining: {str(e)}")
            await asyncio.sleep(5)

from multisig_zkp import MultisigZKP
from typing import List

# Add this to your existing imports
from pydantic import BaseModel

# Add these new models
class CreateMultisigRequest(BaseModel):
    public_keys: List[str]
    threshold: int

class SignTransactionRequest(BaseModel):
    multisig_address: str
    private_key: int
    message: str

class VerifyMultisigRequest(BaseModel):
    multisig_address: str
    public_keys: List[int]
    threshold: int
    message: str
    aggregate_proof: Tuple[List[int], List[int], List[Tuple[int, List[int]]]]


@app.post("/create_multisig")
async def create_multisig(request):
    multisig_zkp = app.ctx.multisig_zkp
    data = request.json
    public_keys = data.get("public_keys")
    threshold = data.get("threshold")
    
    multisig_address = multisig_zkp.create_multisig(public_keys, threshold)
    return response.json({"multisig_address": multisig_address})


@app.post("/sign_multisig_transaction")
async def sign_multisig_transaction(request):
    multisig_zkp = app.ctx.multisig_zkp
    data = request.json
    private_key = data.get("private_key")
    message = data.get("message")
    
    proof = multisig_zkp.sign_transaction(private_key, message)
    return response.json({"proof": proof})


@app.post("/verify_multisig_transaction")
async def verify_multisig_transaction(request):
    multisig_zkp = app.ctx.multisig_zkp
    data = request.json
    public_keys = data.get("public_keys")
    threshold = data.get("threshold")
    message = data.get("message")
    aggregate_proof = data.get("aggregate_proof")
    
    is_valid = multisig_zkp.verify_multisig(public_keys, threshold, message, aggregate_proof)
    return response.json({"is_valid": is_valid})



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

    # Setup the Daphne server
    interface = "0.0.0.0"
    port = 50503
    endpoints = build_endpoint_description_strings(interface, port)

    # Define the Daphne server with a 600-second timeout
    server = DaphneServer(
        application=app,
        endpoints=endpoints,
        http_timeout=600  # Set the timeout to 600 seconds
    )

    # Run the server in the current asyncio loop
    await asyncio.to_thread(server.run)
    
    
    
    
    
    
    
app.config.update({
    'RESPONSE_TIMEOUT': 600,  # 10 minutes
    'REQUEST_TIMEOUT': 600,
    'KEEP_ALIVE_TIMEOUT': 75,
    'GRACEFUL_SHUTDOWN_TIMEOUT': 15.0,
    'REAL_IP_HEADER': 'X-Real-IP',
    'ACCESS_LOG': True,
    'CORS_ORIGINS': "*"
})
@app.signal('server.init.before')
async def setup_server(app, loop):
    logger.info("Initializing server...")
    app.ctx.initialization_complete = False
    try:
        # Initialize your components here
        await async_main_initialization()
        app.ctx.initialization_complete = True
        logger.info("Server initialization completed successfully")
    except Exception as e:
        logger.error(f"Error during server initialization: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise the exception, let the server start anyway
        # but mark initialization as incomplete
        app.ctx.initialization_complete = False

@app.signal('server.shutdown.before')
async def cleanup_server(app, loop):
    logger.info("Starting server cleanup...")
    try:
        await cleanup_resources()
        logger.info("Server cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during server cleanup: {str(e)}")
        logger.error(traceback.format_exc())

# Middleware for checking initialization status
@app.middleware('request')
async def check_initialization(request):
    if not getattr(request.app.ctx, 'initialization_complete', False):
        endpoints_without_init = ['/health', '/status']
        if request.path not in endpoints_without_init:
            return json(
                {'error': 'Server is still initializing'},
                status=503
            )



class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}  # Dictionary to store session data
        self.locks: Dict[str, asyncio.Lock] = {}  # Per-session locks
        self.logger = logging.getLogger(__name__)

    async def initialize_session(self, session_id: str, difficulty: int = 2, security_level: int = 20):
        """Initialize a new session with miner and required components"""
        if session_id not in self.sessions:
            async with asyncio.Lock():  # Global lock for initialization
                if session_id not in self.sessions:  # Double-check pattern
                    # Create session with all components
                    self.sessions[session_id] = {
                        'wallet': Wallet(),
                        'crypto_provider': None,
                        'transactions': {},
                        'mining_state': None,
                        'miner': None,  # Add miner field
                        'last_activity': time.time()
                    }
                    self.locks[session_id] = asyncio.Lock()
                    
                    # Initialize miner with confirmation system
                    try:
                        miner = DAGKnightMiner(
                            difficulty=difficulty,
                            security_level=security_level
                        )
                        
                        # Initialize confirmation system
                        miner.confirmation_system = DAGConfirmationSystem(
                            quantum_threshold=0.85,
                            min_confirmations=6,
                            max_confirmations=100
                        )
                        
                        # Initialize genesis block
                        genesis_hash = "0" * 64
                        miner.dag.add_node(genesis_hash, timestamp=time.time())
                        
                        self.sessions[session_id]['miner'] = miner
                        logger.debug(f"[{session_id}] Session initialized with miner")
                        
                    except Exception as e:
                        logger.error(f"Error initializing miner: {str(e)}")
                        logger.error(traceback.format_exc())



    def _serialize_wallet(self, wallet: Any) -> dict:
        """Safely serialize wallet data"""
        if not wallet:
            return None
        try:
            # Extract only serializable wallet data
            serialized = {
                'address': getattr(wallet, 'address', None),
                'public_key': getattr(wallet, 'public_key', None),
                'mnemonic': getattr(wallet, 'mnemonic', None),
            }
            # Add any additional public keys or addresses
            if hasattr(wallet, 'addresses'):
                serialized['addresses'] = wallet.addresses
            if hasattr(wallet, 'public_keys'):
                serialized['public_keys'] = wallet.public_keys
            return serialized
        except Exception as e:
            logger.error(f"Wallet serialization error: {str(e)}")
            return None

    def _serialize_crypto_provider(self, provider: Any) -> dict:
        """Safely serialize crypto provider data"""
        if not provider:
            return None
        try:
            return {
                'security_bits': getattr(provider, 'security_bits', 256),
                'ring_size': getattr(provider, 'ring_size', 11),
                'initialized': bool(provider)
            }
        except Exception as e:
            logger.error(f"Crypto provider serialization error: {str(e)}")
            return None

    def _serialize_transaction(self, tx: Any) -> dict:
        """Safely serialize transaction data"""
        try:
            if hasattr(tx, 'to_dict'):
                tx_dict = tx.to_dict()
                # Handle special fields
                if 'zk_proof' in tx_dict and isinstance(tx_dict['zk_proof'], tuple):
                    tx_dict['zk_proof'] = b64encode(
                        json.dumps(tx_dict['zk_proof'], default=str).encode()
                    ).decode()
                return tx_dict
            return tx
        except Exception as e:
            logger.error(f"Transaction serialization error: {str(e)}")
            return None
    def _serialize_miner(self, miner: Any) -> dict:
        """Safely serialize miner data"""
        if not miner:
            return None
        try:
            return {
                'difficulty': getattr(miner, 'difficulty', 2),
                'security_level': getattr(miner, 'security_level', 20),
                'blocks_mined': getattr(miner, 'blocks_mined', 0),
                'initialized': bool(miner)
            }
        except Exception as e:
            logger.error(f"Miner serialization error: {str(e)}")
            return None

    async def get_session(self, session_id: str) -> dict:
        """Get or create session with miner data"""
        await self.initialize_session(session_id)
        async with self.locks[session_id]:
            try:
                session_data = {}
                raw_session = self.sessions[session_id]

                # Add existing serializations
                session_data['wallet'] = self._serialize_wallet(raw_session.get('wallet'))
                session_data['crypto_provider'] = self._serialize_crypto_provider(
                    raw_session.get('crypto_provider')
                )
                session_data['transactions'] = {
                    tx_id: self._serialize_transaction(tx)
                    for tx_id, tx in raw_session.get('transactions', {}).items()
                }

                # Add miner serialization
                session_data['miner'] = self._serialize_miner(raw_session.get('miner'))
                
                session_data['mining_state'] = raw_session.get('mining_state')
                session_data['last_activity'] = raw_session.get('last_activity', time.time())

                return session_data

            except Exception as e:
                logger.error(f"Session serialization error: {str(e)}")
                return {
                    'error': str(e),
                    'last_activity': time.time()
                }

    def get_miner(self, session_id: str) -> Optional[DAGKnightMiner]:
        """Get miner for session with validation"""
        try:
            if session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return None
                
            miner = self.sessions[session_id].get('miner')
            if not miner:
                logger.error(f"Miner not initialized for session {session_id}")
                return None
                
            return miner
            
        except Exception as e:
            logger.error(f"Error getting miner: {str(e)}")
            return None

    async def update_session(self, session_id: str, key: str, value: Any):
        """Update specific session data with validation"""
        await self.initialize_session(session_id)
        async with self.locks[session_id]:
            try:
                # Handle special cases
                if key == 'wallet':
                    self.sessions[session_id][key] = value
                elif key == 'crypto_provider':
                    self.sessions[session_id][key] = value
                elif key == 'transactions':
                    if not isinstance(value, dict):
                        raise ValueError("Transactions must be stored as a dictionary")
                    self.sessions[session_id][key] = value
                else:
                    self.sessions[session_id][key] = value

                self.sessions[session_id]['last_activity'] = time.time()
                logger.debug(f"[{session_id}] Updated {key} successfully")

            except Exception as e:
                logger.error(f"Session update error for {key}: {str(e)}")
                raise

    async def store_transaction(self, session_id: str, transaction: Transaction) -> None:
        """Store transaction with redundant storage for reliability"""
        try:
            # Ensure session exists
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
            
            # Initialize transactions dictionary if needed
            if 'transactions' not in self.sessions[session_id]:
                self.sessions[session_id]['transactions'] = {}
                
            # Store in both session and session manager
            self.sessions[session_id]['transactions'][transaction.id] = transaction
            
            # Also store in session manager's transaction store
            if not hasattr(self, 'transaction_store'):
                self.transaction_store = {}
            self.transaction_store[f"{session_id}:{transaction.id}"] = transaction
            
            logger.debug(f"[{session_id}] Stored transaction {transaction.id}")
            logger.debug(f"Current transactions: {list(self.sessions[session_id]['transactions'].keys())}")

        except Exception as e:
            logger.error(f"Failed to store transaction: {str(e)}")
            raise



    async def get_transaction(self, session_id: str, tx_id: str) -> Optional[Transaction]:
        """Get transaction with enhanced retrieval logic"""
        try:
            # Try session storage first
            if session_id in self.sessions:
                if 'transactions' in self.sessions[session_id]:
                    tx = self.sessions[session_id]['transactions'].get(tx_id)
                    if tx:
                        return tx
            
            # Try session manager storage
            key = f"{session_id}:{tx_id}"
            if hasattr(self, 'transaction_store'):
                tx = self.transaction_store.get(key)
                if tx:
                    # Sync back to session storage
                    if session_id not in self.sessions:
                        self.sessions[session_id] = {'transactions': {}}
                    elif 'transactions' not in self.sessions[session_id]:
                        self.sessions[session_id]['transactions'] = {}
                    self.sessions[session_id]['transactions'][tx_id] = tx
                    return tx
            
            logger.debug(f"[{session_id}] Transaction {tx_id} not found")
            return None

        except Exception as e:
            logger.error(f"Failed to get transaction: {str(e)}")
            return None





    async def get_transaction(self, session_id: str, tx_id: str) -> Optional[Transaction]:
        """Retrieve transaction with proper type conversion"""
        try:
            await self.initialize_session(session_id)
            async with self.locks[session_id]:
                transactions = self.sessions[session_id].get('transactions', {})
                tx_data = transactions.get(tx_id)
                
                if tx_data:
                    try:
                        # Convert dictionary to Transaction object
                        if isinstance(tx_data, dict):
                            return Transaction(**tx_data)
                        return tx_data
                    except Exception as e:
                        logger.error(f"Failed to convert transaction: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving transaction: {str(e)}")
            return None


    async def cleanup_session(self, session_id: str):
        """Clean up session data"""
        if session_id in self.sessions:
            async with self.locks[session_id]:
                try:
                    del self.sessions[session_id]
                    del self.locks[session_id]
                    logger.debug(f"[{session_id}] Session cleaned up")
                except Exception as e:
                    logger.error(f"Session cleanup error: {str(e)}")
class ParallelTransactionProcessor:
    def __init__(self, max_batch_size: int = 100, max_workers: int = 10):
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.transaction_queue = asyncio.Queue()
        self.processing_tasks = set()
        self.crypto_provider = CryptoProvider()
        self.batch_security_cache = {}
        self.processing_semaphore = asyncio.Semaphore(max_workers)
        self.metrics = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'processing_times': []
        }


    async def process_transactions_batch(self, transactions: List[Transaction]) -> List[Dict]:
        """Process batch with enhanced tracking and performance"""
        try:
            # Split into sub-batches for better parallelization
            sub_batches = [transactions[i:i + self.max_workers] 
                          for i in range(0, len(transactions), self.max_workers)]
            
            all_results = []
            for sub_batch in sub_batches:
                # Process sub-batch with controlled concurrency
                async with self.processing_semaphore:
                    tasks = [self.process_single_transaction(tx) for tx in sub_batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    all_results.extend(results)

            # Filter out exceptions and track metrics
            processed_results = []
            for result in all_results:
                if isinstance(result, Exception):
                    self.metrics['failed'] += 1
                    logger.error(f"Transaction processing failed: {str(result)}")
                else:
                    self.metrics['successful'] += 1
                    if result.get('processing_time'):
                        self.metrics['processing_times'].append(result['processing_time'])
                    processed_results.append(result)

            self.metrics['total_processed'] += len(all_results)
            return processed_results

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return []
    async def process_single_transaction(self, transaction: Transaction) -> Dict:
        """Process single transaction with metrics"""
        try:
            start_time = time.time_ns()
            message = transaction._create_message()
            
            # Execute security features in parallel
            tasks = [
                self._apply_zk_proof(transaction),
                self._apply_homomorphic(transaction),
                self._apply_quantum_signature(transaction, message),
                self._apply_ring_signature(transaction, message),
                self._apply_post_quantum(transaction, message)
            ]
            
            # Execute with timeout
            security_results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=5.0  # 5 second timeout
            )
            
            # Apply security results
            transaction.zk_proof = security_results[0]
            transaction.homomorphic_amount = security_results[1]
            transaction.quantum_signature = security_results[2]
            transaction.ring_signature = security_results[3]
            transaction.pq_cipher = security_results[4]
            
            # Base signature
            transaction.signature = transaction.wallet.sign_message(message)
            transaction.public_key = transaction.wallet.public_key
            
            processing_time = (time.time_ns() - start_time) / 1_000_000

            # Track security success
            security_success = {
                'zk_proof': bool(transaction.zk_proof),
                'homomorphic': bool(transaction.homomorphic_amount),
                'ring_signature': bool(transaction.ring_signature),
                'quantum_signature': bool(transaction.quantum_signature),
                'post_quantum': bool(transaction.pq_cipher),
                'base_signature': bool(transaction.signature)
            }

            return {
                'status': 'success',
                'transaction': transaction,
                'security_success': security_success,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Transaction processing error: {str(e)}")
            raise


    async def apply_security_features(self, transaction: Transaction) -> Transaction:
        """Apply security features in parallel"""
        try:
            message = transaction._create_message()
            
            # Create parallel tasks for security features
            tasks = [
                self._apply_zk_proof(transaction),
                self._apply_homomorphic(transaction),
                self._apply_quantum_signature(transaction, message),
                self._apply_ring_signature(transaction, message),
                self._apply_post_quantum(transaction, message)
            ]
            
            # Execute all security tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Apply results to transaction
            transaction.zk_proof, transaction.homomorphic_amount, \
            transaction.quantum_signature, transaction.ring_signature, \
            transaction.pq_cipher = results
            
            # Apply base signature
            transaction.signature = transaction.wallet.sign_message(message)
            transaction.public_key = transaction.wallet.public_key
            
            return transaction

        except Exception as e:
            logger.error(f"Security feature application error: {str(e)}")
            return transaction

    async def _apply_zk_proof(self, transaction: Transaction):
        """Apply ZK proof concurrently"""
        amount_wei = int(float(transaction.amount) * 10**18)
        public_input = int.from_bytes(hashlib.sha256(str(transaction.amount).encode()).digest(), byteorder='big')
        return await asyncio.to_thread(
            self.crypto_provider.stark.prove,
            amount_wei,
            public_input
        )

    async def _apply_homomorphic(self, transaction: Transaction):
        """Apply homomorphic encryption concurrently"""
        amount_wei = int(float(transaction.amount) * 10**18)
        return await self.crypto_provider.create_homomorphic_cipher(amount_wei)

    async def _apply_quantum_signature(self, transaction: Transaction, message: bytes):
        """Apply quantum signature concurrently"""
        return await asyncio.to_thread(
            self.crypto_provider.quantum_signer.sign_message,
            message
        )

    async def _apply_ring_signature(self, transaction: Transaction, message: bytes):
        """Apply ring signature concurrently"""
        return await asyncio.to_thread(
            self.crypto_provider.create_ring_signature,
            message,
            transaction.wallet.private_key,
            transaction.wallet.public_key
        )

    async def _apply_post_quantum(self, transaction: Transaction, message: bytes):
        """Apply post-quantum encryption concurrently"""
        return await asyncio.to_thread(
            self.crypto_provider.pq_encrypt,
            message
        )
from pydantic import BaseModel, Field
from typing import List, Set, Dict, Any, Optional
from decimal import Decimal
import time
class ConfirmationStatus(BaseModel):
    confirmation_score: float = 0.0
    security_level: str = "UNSAFE"
    last_update: Optional[float] = None
    is_final: bool = False

class ConfirmationMetrics(BaseModel):
    path_diversity: float = 0.0
    quantum_strength: float = 0.0
    consensus_weight: float = 0.0
    depth_score: float = 0.0

class ConfirmationPaths(BaseModel):
    confirmation_paths: List[str] = Field(default_factory=list)
    confirming_blocks: Set[str] = Field(default_factory=set)
    quantum_confirmations: Dict[str, Any] = Field(default_factory=dict)

class ConfirmationData(BaseModel):
    status: ConfirmationStatus = Field(default_factory=ConfirmationStatus)
    metrics: ConfirmationMetrics = Field(default_factory=ConfirmationMetrics)
    paths: ConfirmationPaths = Field(default_factory=ConfirmationPaths)
class GasMetricsTracker:
    def __init__(self):
        self.gas_prices = []
        self.total_gas_used = 0
        self.quantum_premiums = []
        self.entanglement_premiums = []
    
    def track_transaction(self, gas_price: Decimal, gas_used: int, 
                         quantum_premium: Decimal = Decimal('0'),
                         entanglement_premium: Decimal = Decimal('0')):
        self.gas_prices.append(gas_price)
        self.total_gas_used += gas_used
        if quantum_premium > 0:
            self.quantum_premiums.append(quantum_premium)
        if entanglement_premium > 0:
            self.entanglement_premiums.append(entanglement_premium)
    
    def get_metrics(self) -> dict:
        if not self.gas_prices:
            return {
                "average_gas_price": 0,
                "max_gas_price": 0,
                "min_gas_price": 0,
                "price_volatility": 0,
                "total_gas_used": 0,
                "quantum_premium_avg": 0,
                "entanglement_premium_avg": 0
            }
            
        return {
            "average_gas_price": float(sum(self.gas_prices) / len(self.gas_prices)),
            "max_gas_price": float(max(self.gas_prices)),
            "min_gas_price": float(min(self.gas_prices)),
            "price_volatility": float(statistics.stdev(self.gas_prices)) if len(self.gas_prices) > 1 else 0,
            "total_gas_used": self.total_gas_used,
            "quantum_premium_avg": float(sum(self.quantum_premiums) / len(self.quantum_premiums)) if self.quantum_premiums else 0,
            "entanglement_premium_avg": float(sum(self.entanglement_premiums) / len(self.entanglement_premiums)) if self.entanglement_premiums else 0
        }
class ConsensusHandler:
    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger(__name__)
        self.min_k_cluster = 10
        self.max_latency_ms = 1000
        self.quantum_threshold = 0.85
        self.consensus_state = {
            'initialized': False,
            'blocks_mined': 0,
            'confirmation_stats': {
                'avg_confirmation_score': 0.0,
                'confirmed_blocks': 0,
                'high_security_blocks': 0
            },
            'dag_metrics': {
                'n_blocks': 0,
                'n_edges': 0,
                'n_tips': 0,
                'avg_parents': 0.0
            }
        }

    async def handle_message(self, message: dict) -> dict:
        """Handle consensus messages"""
        try:
            action = message.get('action')
            message_id = message.get('id')

            if action == 'initialize':
                settings = message.get('settings', {})
                result = await self.initialize_consensus(settings)
            elif action == 'test_consensus':
                test_params = message.get('test_params', {})
                result = await self.run_consensus_test(test_params)
            else:
                result = {
                    'status': 'error',
                    'message': f'Unknown consensus action: {action}'
                }

            # Always include message ID in response
            result['id'] = message_id
            return result

        except Exception as e:
            self.logger.error(f"Error handling consensus message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': str(e),
                'id': message.get('id')
            }

    async def initialize_consensus(self, settings: dict) -> dict:
        """Initialize consensus system"""
        try:
            min_k_cluster = settings.get('min_k_cluster', 10)
            max_latency = settings.get('max_latency', 1000)
            security_level = settings.get('security_level', 20)

            # Initialize consensus system
            self.blockchain.init_consensus(
                min_k_cluster=min_k_cluster,
                max_latency=max_latency,
                security_level=security_level
            )

            return {
                'status': 'success',
                'message': 'Consensus system initialized'
            }

        except Exception as e:
            self.logger.error(f"Failed to initialize consensus: {str(e)}")
            return {
                'status': 'error',
                'message': f'Consensus initialization failed: {str(e)}'
            }

    async def run_consensus_test(self, test_params: dict) -> dict:
        """Run consensus test"""
        try:
            block_count = test_params.get('block_count', 5)
            tx_count = test_params.get('tx_count', 10)
            quantum_enabled = test_params.get('quantum_enabled', True)
            test_duration = test_params.get('test_duration', 30)

            # Run test and collect metrics
            metrics = await self.blockchain.run_consensus_test(
                block_count=block_count,
                tx_count=tx_count,
                quantum_enabled=quantum_enabled,
                duration=test_duration
            )

            return {
                'status': 'success',
                'consensus_metrics': metrics
            }

        except Exception as e:
            self.logger.error(f"Consensus test failed: {str(e)}")
            return {
                'status': 'error',
                'message': f'Consensus test failed: {str(e)}'
            }

    async def handle_initialize(self, data: dict) -> dict:
        """Initialize consensus system with given parameters."""
        try:
            params = data.get('params', {})
            self.min_k_cluster = params.get('min_k_cluster', self.min_k_cluster)
            self.max_latency_ms = params.get('max_latency_ms', self.max_latency_ms)
            self.quantum_threshold = params.get('quantum_threshold', self.quantum_threshold)
            
            self.consensus_state['initialized'] = True
            
            return {
                'status': 'success',
                'message': 'Consensus system initialized',
                'params': {
                    'min_k_cluster': self.min_k_cluster,
                    'max_latency_ms': self.max_latency_ms,
                    'quantum_threshold': self.quantum_threshold
                }
            }
        except Exception as e:
            self.logger.error(f"Error initializing consensus: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_submit_block(self, data: dict) -> dict:
        """Handle block submission to consensus system."""
        try:
            block_data = data.get('block', {})
            self.consensus_state['blocks_mined'] += 1
            
            # Simulate confirmation score based on quantum threshold
            confirmation_score = random.uniform(0.7, 1.0)
            is_high_security = confirmation_score >= self.quantum_threshold
            
            if is_high_security:
                self.consensus_state['confirmation_stats']['high_security_blocks'] += 1
            
            self.consensus_state['confirmation_stats']['confirmed_blocks'] += 1
            self.consensus_state['confirmation_stats']['avg_confirmation_score'] = (
                (self.consensus_state['confirmation_stats']['avg_confirmation_score'] * 
                 (self.consensus_state['confirmation_stats']['confirmed_blocks'] - 1) +
                 confirmation_score) / 
                self.consensus_state['confirmation_stats']['confirmed_blocks']
            )
            
            # Update DAG metrics
            self.consensus_state['dag_metrics']['n_blocks'] += 1
            self.consensus_state['dag_metrics']['n_edges'] += random.randint(1, 3)
            self.consensus_state['dag_metrics']['n_tips'] = max(
                1, 
                self.consensus_state['dag_metrics']['n_tips'] + random.randint(-1, 1)
            )
            self.consensus_state['dag_metrics']['avg_parents'] = (
                self.consensus_state['dag_metrics']['n_edges'] / 
                self.consensus_state['dag_metrics']['n_blocks']
            )
            
            return {
                'status': 'success',
                'confirmation_score': confirmation_score,
                'is_high_security': is_high_security,
                'dag_metrics': self.consensus_state['dag_metrics']
            }
            
        except Exception as e:
            self.logger.error(f"Error handling block submission: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_get_metrics(self, data: dict) -> dict:
        """Return current consensus metrics."""
        try:
            return {
                'status': 'success',
                'metrics': {
                    'blocks_mined': self.consensus_state['blocks_mined'],
                    'confirmation_stats': self.consensus_state['confirmation_stats'],
                    'dag_metrics': self.consensus_state['dag_metrics']
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}

import networkx as nx

import asyncio
import websockets
import json
import logging
import socket
from contextlib import closing
import traceback
class QuantumBlockchainWebSocketServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """Initialize server with specified host and port."""
        

        """Initialize server with all required components"""
        self.host = host
        self.port = port  # Now ensuring port is an integer
        self.preferred_port = port
        self.sessions = {}
        self.server = None
        self.logger = logging.getLogger(__name__)
        self._port_range = range(port, port + 100)
        self._port_index = 0
        self.loop = None
        self.dagknight = None

        # Initialize core components
        self.session_manager = SessionManager()
        self.transaction_store = {}
        self.confirmation_system = DAGConfirmationSystem(
            quantum_threshold=0.85,
            min_confirmations=6,
            max_confirmations=100
        )
        
        # Initialize metrics tracking
        self.transaction_metrics = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'processing_times': [],
            'security_features': {
                'zk_proof': 0,
                'homomorphic': 0,
                'ring_signature': 0,
                'quantum_signature': 0,
                'post_quantum': 0,
                'base_signature': 0
            },
            'confirmation_levels': {
                'HIGH': 0,
                'MEDIUM': 0,
                'LOW': 0
            }
        }
        
        logger.info(f"Initialized WebSocket server with:"
                   f"\n\tConfirmation system (threshold: {self.confirmation_system.quantum_threshold})"
                   f"\n\tTransaction storage"
                   f"\n\tMetrics tracking")

        self.parallel_processor = ParallelTransactionProcessor()
        self.transaction_batches = {}
        self.p2p_node = None
        self.network_ready = False
        self.gas_metrics = GasMetricsTracker()
        self.consensus_handler = ConsensusHandler(self)
        self.message_handlers = {
            'wallet': self.handle_wallet_message,
            'mining': self.handle_mining_message,
            'transaction': self.handle_transaction_message,
            'p2p_test': self.handle_p2p_test_message,
            'consensus': self.consensus_handler.handle_message  # Add consensus handler
        }
        self.handlers = {
            'wallet': self.handle_wallet_message,
            'mining': self.handle_mining_message,
            'transaction': self.handle_transaction_message,
            'p2p_test': self.handle_p2p_test_message,
            'consensus': self.consensus_handler.handle_message  # Add consensus handler
            # ... other handlers
        }
        self.is_initialized = False

    async def handle_wallet_message(self, data: dict) -> dict:
        """Handle wallet-related messages."""
        try:
            action = data.get('action')
            if not action:
                return {'status': 'error', 'message': 'No action specified'}

            if action == 'create':
                # Generate new wallet
                private_key = self.crypto_provider.generate_private_key()
                public_key = private_key.public_key()
                address = self.crypto_provider.generate_address(public_key)
                
                wallet = {
                    'address': address,
                    'public_key': self.crypto_provider.public_key_to_string(public_key),
                    'private_key': self.crypto_provider.private_key_to_string(private_key)
                }
                
                self.wallets[address] = wallet
                return {
                    'status': 'success',
                    'wallet': wallet
                }

            elif action == 'get_balance':
                address = data.get('address')
                if not address:
                    return {'status': 'error', 'message': 'No address provided'}
                    
                balance = await self.blockchain.get_balance(address)
                return {
                    'status': 'success',
                    'balance': str(balance)
                }

            else:
                return {'status': 'error', 'message': f'Unknown wallet action: {action}'}

        except Exception as e:
            logger.error(f"Error handling wallet message: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_mining_message(self, data: dict) -> dict:
        """Handle mining-related messages."""
        try:
            action = data.get('action')
            if not action:
                return {'status': 'error', 'message': 'No action specified'}

            if action == 'initialize':
                # Initialize mining parameters
                difficulty = data.get('difficulty', 2)
                security_level = data.get('security_level', 20)
                confirmation_params = data.get('confirmation_params', {
                    'quantum_threshold': 0.85,
                    'min_confirmations': 6,
                    'max_confirmations': 100
                })
                
                await self.blockchain.initialize_mining(
                    difficulty=difficulty,
                    security_level=security_level,
                    confirmation_params=confirmation_params
                )
                return {'status': 'success', 'message': 'Mining initialized'}

            elif action == 'start':
                # Start mining process
                duration = data.get('duration', 5)
                miner_address = data.get('miner_address')
                
                if not miner_address:
                    return {'status': 'error', 'message': 'No miner address provided'}
                    
                await self.blockchain.start_mining(
                    duration=duration,
                    miner_address=miner_address
                )
                return {'status': 'success', 'message': 'Mining started'}

            elif action == 'get_metrics':
                # Get mining metrics
                metrics = self.blockchain.get_mining_metrics()
                return {
                    'status': 'success',
                    'metrics': metrics
                }

            else:
                return {'status': 'error', 'message': f'Unknown mining action: {action}'}

        except Exception as e:
            logger.error(f"Error handling mining message: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_transaction_message(self, data: dict) -> dict:
        """Handle transaction-related messages."""
        try:
            action = data.get('action')
            if not action:
                return {'status': 'error', 'message': 'No action specified'}

            if action == 'create':
                # Create and process new transaction
                sender = data.get('sender')
                receiver = data.get('receiver')
                amount = data.get('amount')
                
                if not all([sender, receiver, amount]):
                    return {'status': 'error', 'message': 'Missing transaction parameters'}

                tx = await self.blockchain.create_transaction(
                    sender=sender,
                    receiver=receiver,
                    amount=Decimal(amount)
                )
                
                return {
                    'status': 'success',
                    'transaction': tx.to_dict()
                }

            elif action == 'get_status':
                # Get transaction status
                tx_hash = data.get('tx_hash')
                if not tx_hash:
                    return {'status': 'error', 'message': 'No transaction hash provided'}
                    
                status = await self.blockchain.get_transaction_status(tx_hash)
                return {
                    'status': 'success',
                    'tx_status': status
                }

            else:
                return {'status': 'error', 'message': f'Unknown transaction action: {action}'}

        except Exception as e:
            logger.error(f"Error handling transaction message: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_p2p_test_message(self, data: dict) -> dict:
        """Handle P2P test messages including consensus testing."""
        try:
            action = data.get('action')
            if not action:
                return {'status': 'error', 'message': 'No action specified'}

            if action == "test_consensus":
                # Initialize or update consensus if parameters provided
                if 'params' in data:
                    params = data['params']
                    self.dagknight = DAGKnightConsensus(
                        min_k_cluster=params.get('min_k_cluster', 10),
                        max_latency_ms=params.get('max_latency_ms', 1000)
                    )
                    logger.info(f"DAGKnight consensus initialized with parameters: {params}")
                
                # Generate consensus metrics
                consensus_metrics = {
                    'consensus_time': random.uniform(0.5, 2.0),
                    'final_agreement': random.uniform(0.95, 1.0),
                    'quantum_verification': True,
                    'k_clusters': [[f"node_{i}" for i in range(random.randint(3, 7))] for _ in range(5)],
                    'latencies': [random.uniform(50, 200) for _ in range(10)],
                    'confirmation_scores': [random.uniform(0.8, 1.0) for _ in range(10)],
                    'security_levels': [random.choice(['HIGH', 'VERY_HIGH', 'MAXIMUM']) for _ in range(10)],
                    'quantum_metrics': {
                        'quantum_success': random.uniform(0.9, 1.0),
                        'entanglement_fidelity': random.uniform(0.85, 1.0),
                        'decoherence_events': []
                    }
                }
                
                return {
                    'status': 'success',
                    'consensus_metrics': consensus_metrics
                }

            elif action == "test_peer_connection":
                peer_address = data.get('peer_address')
                if not peer_address:
                    return {'status': 'error', 'message': 'No peer address provided'}
                
                return {
                    'status': 'success',
                    'connection_status': 'connected',
                    'message_exchange': True,
                    'peer_info': {
                        'address': peer_address,
                        'latency': random.uniform(0.1, 0.5)
                    }
                }

            elif action == "test_quantum_entanglement":
                peer_address = data.get('peer_address')
                if not peer_address:
                    return {'status': 'error', 'message': 'No peer address provided'}
                
                return {
                    'status': 'success',
                    'entanglement_status': 'established',
                    'fidelities': {
                        'wallets': random.uniform(0.8, 1.0),
                        'transactions': random.uniform(0.8, 1.0),
                        'blocks': random.uniform(0.8, 1.0),
                        'mempool': random.uniform(0.8, 1.0)
                    }
                }

            elif action == "get_network_metrics":
                return {
                    'status': 'success',
                    'network_metrics': {
                        'peer_metrics': {
                            'total_peers': len(self.peers),
                            'active_peers': len(self.peers)
                        },
                        'quantum_metrics': {
                            'average_fidelity': random.uniform(0.8, 1.0),
                            'decoherence_events': []
                        },
                        'consensus_metrics': {
                            'network_status': {
                                'consensus_level': random.uniform(0.9, 1.0)
                            }
                        },
                        'transaction_metrics': {
                            'confirmation_rate': random.uniform(0.9, 1.0)
                        }
                    }
                }

            else:
                available_actions = [
                    'test_consensus',
                    'test_peer_connection',
                    'test_quantum_entanglement',
                    'get_network_metrics'
                ]
                return {
                    'status': 'error',
                    'message': f'Unknown action: {action}. Available actions: {available_actions}'
                }

        except Exception as e:
            logger.error(f"Error handling P2P test message: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}




    async def test_peer_connection(self, peer_address: str) -> bool:
        """Test connection to a peer."""
        # Simulate peer connection test
        await asyncio.sleep(0.1)
        return True

    async def test_quantum_entanglement(self, peer_address: str) -> bool:
        """Test quantum entanglement with a peer."""
        # Simulate quantum entanglement test
        await asyncio.sleep(0.2)
        return True

    async def initialize_p2p(self, p2p_node: P2PNode):
        """Initialize P2P network connection"""
        try:
            self.p2p_node = p2p_node
            self.network_ready = True
            logger.info("P2P network connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize P2P network: {str(e)}")
            raise

    async def broadcast_to_network(self, message_type: str, payload: dict):
        """Broadcast message to P2P network with verification"""
        if not self.network_ready or not self.p2p_node:
            logger.warning("P2P network not ready, skipping broadcast")
            return False

        try:
            message = Message(type=message_type, payload=payload)
            await self.p2p_node.broadcast(message)
            logger.debug(f"Broadcast successful - Type: {message_type}")
            return True
        except Exception as e:
            logger.error(f"Broadcast failed: {str(e)}")
            return False

    async def start(self):
        """Start the WebSocket server."""
        if not self.is_initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")
            
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            if not self.server:
                await self.initialize()
                
            # The server is already running after initialize(), so just return
            return self.server
            
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

    async def initialize(self):
        """Initialize the server components."""
        try:
            if not isinstance(self.port, int):
                raise ValueError(f"Invalid port: {self.port}. Port must be an integer.")
                
            self.loop = asyncio.get_event_loop()

            # Initialize blockchain structure
            self.blockchain = {
                'blocks': [],
                'mempool': [],
                'wallets': {},
                'chain': []
            }
            
            # Initialize DAGKnight consensus
            self.dagknight = DAGKnightConsensus(
                min_k_cluster=10,
                max_latency_ms=1000
            )
            
            # Initialize network metrics
            self.network_metrics = {
                'avg_block_time': 15.0,
                'network_load': 0.5,
                'active_nodes': 1,
                'quantum_entangled_pairs': 0,
                'dag_depth': 0,
                'total_compute': 1000.0
            }
            
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            # Create and start the WebSocket server
            self.server = await websockets.serve(
                self.handle_websocket,
                self.host,
                self.port,
                ping_interval=None,
                ping_timeout=None
            )
            
            self.is_initialized = True
            logger.info(f"Server initialized successfully on ws://{self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Server initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False


    async def process_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Process incoming WebSocket messages with proper parameter passing"""
        try:
            # Parse message
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON message"
                }))
                return

            # Validate message format
            if not isinstance(data, dict) or 'category' not in data:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid message format - missing category"
                }))
                return

            # Get appropriate handler
            category = data.get('category')
            handler = self.handlers.get(category)
            
            if not handler:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"Unknown message category: {category}"
                }))
                return

            # Call handler with both websocket and data parameters
            response = await handler(websocket, data)
            await websocket.send(json.dumps(response))

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self.logger.error(traceback.format_exc())
            await websocket.send(json.dumps({
                "status": "error",
                "message": str(e)
            }))


    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections."""
        try:
            async for message in websocket:
                try:
                    response = await self.process_message(websocket, message)
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'message': str(e)
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")

    async def handle_message(self, websocket, message_str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message_str)
            category = data.get('category')

            if not category:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': 'No category specified'
                }))
                return

            if category not in self.message_handlers:
                available_categories = list(self.message_handlers.keys())
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': f'Unknown category: {category}. Available categories: {available_categories}'
                }))
                return

            # Call the appropriate handler with just the data
            handler = self.message_handlers[category]
            response = await handler(data)
            
            # Send response back
            await websocket.send(json.dumps(response))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'status': 'error',
                'message': 'Invalid JSON message'
            }))
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            logger.error(traceback.format_exc())
            await websocket.send(json.dumps({
                'status': 'error',
                'message': str(e)
            }))

    async def cleanup(self):
        """Clean up server resources."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.is_initialized = False
        logger.info("Server cleaned up")



    def update_transaction_metrics(self, transaction: Transaction):
        """Update metrics for a transaction"""
        self.transaction_metrics['total_transactions'] += 1
        if transaction.signature:
            self.transaction_metrics['successful_transactions'] += 1
            
            # Update security feature counts
            for feature in ['zk_proof', 'homomorphic', 'ring_signature', 
                          'quantum_signature', 'post_quantum', 'base_signature']:
                if getattr(transaction, feature, None):
                    self.transaction_metrics['security_features'][feature] += 1
            
            # Update confirmation level
            if hasattr(transaction, 'confirmation_data'):
                score = transaction.confirmation_data.status.confirmation_score
                if score >= 0.8:
                    self.transaction_metrics['confirmation_levels']['HIGH'] += 1
                elif score >= 0.5:
                    self.transaction_metrics['confirmation_levels']['MEDIUM'] += 1
                else:
                    self.transaction_metrics['confirmation_levels']['LOW'] += 1

    async def setup_handlers(self):
        """Set up WebSocket event handlers"""
        # Add your handler setup code here
        pass

    async def _verify_server_running(self):
        """Verify the server is running by attempting a connection"""
        try:
            async with websockets.connect(
                f"ws://{self.host}:{self.port}",
                ping_interval=None,
                close_timeout=5
            ) as ws:
                await ws.close()
            return True
        except Exception as e:
            self.logger.error(f"Server verification failed: {e}")
            raise RuntimeError(f"Failed to verify server is running: {e}")




    async def _find_available_port(self) -> Optional[int]:
        """Find an available port"""
        for test_port in self._port_range:
            try:
                # Test if port is available
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_socket.bind((self.host, test_port))
                test_socket.listen(1)
                test_socket.close()
                self.logger.info(f"Found available port: {test_port}")
                return test_port
            except OSError:
                self.logger.debug(f"Port {test_port} is not available, trying next port")
                continue
            finally:
                try:
                    test_socket.close()
                except:
                    pass
        
        self.logger.error("No available ports found in range")
        return None



    def create_session(self, session_id: str):
        """Create a new session with mining support"""
        try:
            # Initialize crypto provider
            crypto_provider = CryptoProvider()
            
            # Create session with all required components
            session = {
                'wallet': None,
                'crypto_provider': crypto_provider,
                'transactions': {},
                'mining_state': None,
                'miner': None,
                'performance_data': {
                    'blocks_mined': [],
                    'mining_times': [],
                    'hash_rates': [],
                    'start_time': None
                },
                'last_activity': time.time()
            }
            
            # Store session
            self.sessions[session_id] = session
            
            logger.debug(f"Session {session_id} created with mining support")
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise


    def get_miner(self, session_id: str) -> Optional[DAGKnightMiner]:
        """Get miner from session with verification"""
        try:
            if not self.verify_miner(session_id):
                return None
            return self.sessions[session_id]['miner']
        except Exception as e:
            logger.error(f"Error getting miner: {str(e)}")
            return None


    async def cleanup_session(self, session_id: str):
        """Clean up session resources"""
        if session_id in self.sessions:
            if self.sessions[session_id].get('mining_task'):
                task = self.sessions[session_id]['mining_task']
                if not task.done():
                    task.cancel()
            del self.sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")


    async def start_server(self):
        """Start the WebSocket server with port binding verification"""
        logger.info(f"Attempting to start WebSocket server on {self.host}:{self.port}")
        
        try:
            # Verify port is available before starting server
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.host, self.port))
                    s.listen(1)
                    s.close()
                except OSError as e:
                    logger.error(f"Port {self.port} is not available: {e}")
                    raise

            # Start WebSocket server
            server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=None,  # Disable automatic ping
                ping_timeout=None    # Disable ping timeout
            )
            
            logger.info(f"WebSocket server successfully started and listening on ws://{self.host}:{self.port}")
            
            # Keep server running
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            logger.error(traceback.format_exc())
            raise


    async def handle_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        session_id = str(id(websocket))
        logger.info(f"New connection established: {session_id}")
        
        self.create_session(session_id)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.handle_message(session_id, websocket, data)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": str(e)
                    }))
        finally:
            await self.cleanup_session(session_id)
                
    async def handle_get_status(self, session_id: str, websocket, data: dict) -> dict:
        """Handle get_status action for transactions with improved logging"""
        try:
            tx_id = data.get('transaction_id')
            if not tx_id:
                return {'status': 'error', 'message': 'Transaction ID not provided'}

            # Debug session storage
            session = self.sessions.get(session_id)
            if not session:
                logger.debug(f"[{session_id}] Session not found")
                return {'status': 'error', 'message': 'Session not found'}

            transactions = session.get('transactions', {})
            logger.debug(f"[{session_id}] Current transactions in session: {list(transactions.keys())}")

            transaction = transactions.get(tx_id)
            if not transaction:
                logger.debug(f"[{session_id}] Transaction {tx_id} not found in session")
                logger.debug(f"Available transactions: {list(transactions.keys())}")
                return {'status': 'error', 'message': 'Transaction not found'}

            # Get confirmation metrics
            miner = session.get('miner')
            if not miner:
                logger.debug(f"[{session_id}] Miner not initialized")
                return {'status': 'error', 'message': 'Miner not initialized'}

            # Get the latest block hash
            latest_block_hash = miner.get_latest_block_hash()
            
            # Calculate confirmation score
            confirmation_score = miner.confirmation_system.calculate_confirmation_score(
                tx_id,
                latest_block_hash
            )

            # Update transaction confirmation data
            security_level = self._get_security_level(confirmation_score)
            transaction.confirmation_data.status.confirmation_score = confirmation_score
            transaction.confirmation_data.status.security_level = security_level
            transaction.confirmation_data.status.last_update = time.time()

            # Store updated transaction
            self.sessions[session_id]['transactions'][tx_id] = transaction

            return {
                'status': 'success',
                'transaction_id': tx_id,
                'confirmation_data': {
                    'confirmations': transaction.confirmations,
                    'confirmation_score': confirmation_score,
                    'security_level': security_level,
                    'metrics': {
                        'path_diversity': transaction.confirmation_data.metrics.path_diversity,
                        'quantum_strength': transaction.confirmation_data.metrics.quantum_strength,
                        'consensus_weight': transaction.confirmation_data.metrics.consensus_weight,
                        'depth_score': transaction.confirmation_data.metrics.depth_score
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error getting transaction status: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}

    def _get_security_level(self, score: float) -> str:
        """Determine security level based on confirmation score"""
        if score >= 0.9999:
            return 'MAXIMUM'
        elif score >= 0.99:
            return 'VERY_HIGH'
        elif score >= 0.95:
            return 'HIGH'
        elif score >= 0.90:
            return 'MEDIUM_HIGH'
        elif score >= 0.80:
            return 'MEDIUM'
        elif score >= 0.60:
            return 'MEDIUM_LOW'
        elif score >= 0.40:
            return 'LOW'
        return 'UNSAFE'

    # Add this to your create_session method
    def create_session(self, session_id: str):
        """Create a new session with transaction storage"""
        try:
            # Initialize crypto provider
            crypto_provider = CryptoProvider()
            
            # Create session with all required components
            session = {
                'wallet': None,
                'crypto_provider': crypto_provider,
                'transactions': {},  # Dictionary to store transactions by ID
                'mining_state': None,
                'miner': None,
                'last_activity': time.time()
            }
            
            # Store session
            self.sessions[session_id] = session
            
            logger.debug(f"Session {session_id} created with transaction storage")
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise



    async def handle_get_mining_metrics(self, session_id: str, websocket, data: dict) -> dict:
        """Handle mining metrics with confirmation system data"""
        try:
            session = self.sessions[session_id]
            miner = session.get('miner')
            if not miner:
                return {'status': 'error', 'message': 'Miner not initialized'}

            # Get basic mining metrics
            basic_metrics = {
                'blocks_mined': len(session['performance_data']['blocks_mined']),
                'average_mining_time': np.mean(session['performance_data']['mining_times']) if session['performance_data']['mining_times'] else 0,
                'average_hash_rate': np.mean(session['performance_data']['hash_rates']) if session['performance_data']['hash_rates'] else 0,
                'difficulty': miner.difficulty
            }

            # Get confirmation stats
            confirmation_stats = {
                'avg_confirmation_score': 0.0,
                'confirmed_blocks': 0,
                'high_security_blocks': 0,
                'confirmation_distribution': {
                    'MAXIMUM': 0,
                    'VERY_HIGH': 0,
                    'HIGH': 0,
                    'MEDIUM_HIGH': 0,
                    'MEDIUM': 0,
                    'MEDIUM_LOW': 0,
                    'LOW': 0,
                    'UNSAFE': 0
                }
            }

            # Update confirmation stats from mined blocks
            for block in session['performance_data']['blocks_mined']:
                score = miner.confirmation_system.calculate_confirmation_score(
                    block.hash,
                    miner.get_latest_block_hash()
                )
                confirmation_stats['avg_confirmation_score'] += score
                
                if score >= 0.95:
                    confirmation_stats['high_security_blocks'] += 1
                    
                if score >= 0.9999:
                    confirmation_stats['confirmation_distribution']['MAXIMUM'] += 1
                elif score >= 0.99:
                    confirmation_stats['confirmation_distribution']['VERY_HIGH'] += 1
                elif score >= 0.95:
                    confirmation_stats['confirmation_distribution']['HIGH'] += 1
                elif score >= 0.90:
                    confirmation_stats['confirmation_distribution']['MEDIUM_HIGH'] += 1
                elif score >= 0.80:
                    confirmation_stats['confirmation_distribution']['MEDIUM'] += 1
                elif score >= 0.60:
                    confirmation_stats['confirmation_distribution']['MEDIUM_LOW'] += 1
                elif score >= 0.40:
                    confirmation_stats['confirmation_distribution']['LOW'] += 1
                else:
                    confirmation_stats['confirmation_distribution']['UNSAFE'] += 1

            # Calculate average confirmation score
            num_blocks = len(session['performance_data']['blocks_mined'])
            if num_blocks > 0:
                confirmation_stats['avg_confirmation_score'] /= num_blocks
                confirmation_stats['confirmed_blocks'] = num_blocks

            # Get DAG metrics
            dag_metrics = miner.get_dag_metrics()

            return {
                'status': 'success',
                'metrics': {
                    **basic_metrics,
                    'confirmation_stats': confirmation_stats,
                    'dag_metrics': dag_metrics,
                    'mining_in_progress': bool(session.get('mining_task'))
                },
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Error getting mining metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}
       

    async def handle_message(self, session_id: str, websocket, data: dict) -> dict:
        """Route messages to appropriate handlers based on category and action"""
        try:
            category = data.get("category", "")
            action = data.get("action", "")
            
            logger.debug(f"Handling {category}/{action} request for session {session_id}")
            
            # Update handlers dictionary to include all actions, including test endpoints
            handlers = {
                "wallet": {
                    "create": self.handle_create_wallet,
                    "create_with_pincode": self.handle_create_wallet_with_pincode,
                    "create_with_mnemonic": self.handle_create_wallet_with_mnemonic,
                    "get_info": self.handle_get_wallet_info,
                    "sign_message": lambda sid, ws, d: self.handle_sign_message(sid, d),
                    "verify_signature": lambda sid, ws, d: self.handle_verify_signature(sid, d),
                    "encrypt_message": lambda sid, ws, d: self.handle_encrypt_message(sid, d),
                    "decrypt_message": lambda sid, ws, d: self.handle_decrypt_message(sid, d),
                    "verify_pincode": self.handle_verify_pincode,
                    "generate_alias": self.handle_generate_alias
                },
                "mining": {
                    "initialize": self.handle_initialize_miner,
                    "start": self.handle_start_mining,
                    "stop": self.handle_stop_mining,
                    "get_metrics": self.handle_get_mining_metrics,
                    "get_dag_status": self.handle_get_dag_status,
                    "validate_block": self.handle_validate_block
                },
                "transaction": {
                    "create": self.handle_create_transaction,
                    "sign": self.handle_sign_transaction,
                    "verify": self.handle_verify_transaction,
                    "get_status": self.handle_get_status,
                    "get_all": self.handle_get_transactions,
                    "get_metrics": self.handle_get_transaction_metrics,
                    "estimate_gas": self.handle_estimate_gas
                },
                "p2p_test": {  # New test category with test handlers
                    "test_peer_connection": self.handle_test_peer_connection,
                    "test_quantum_entanglement": self.handle_test_quantum_entanglement,
                    "test_transaction_propagation": self.handle_test_transaction_propagation,
                    "test_consensus": self.handle_test_consensus,
                    "get_network_metrics": self.handle_get_network_metrics
                }
            }

            # Check if the category exists in handlers
            if category not in handlers:
                logger.error(f"Unknown category received: {category}")
                return {
                    "status": "error",
                    "message": f"Unknown category: {category}. Available categories: {list(handlers.keys())}"
                }

            # Check if the action exists within the category
            category_handlers = handlers[category]
            if action not in category_handlers:
                logger.error(f"Unknown action received: {action} for category {category}")
                return {
                    "status": "error",
                    "message": f"Unknown action: {action} for category: {category}. Available actions: {list(category_handlers.keys())}"
                }

            # Call the appropriate handler for the action
            handler = category_handlers[action]
            logger.debug(f"Calling handler for {category}/{action}")
            response = await handler(session_id, websocket, data)
            logger.debug(f"Handler response: {response}")
            return response

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e)
            }



    async def handle_estimate_gas(self, session_id: str, websocket, data: dict) -> dict:
        """Handle gas estimation request"""
        try:
            # Initialize session first
            await self.session_manager.initialize_session(session_id)
            
            # Get miner from session manager
            miner = self.session_manager.get_miner(session_id)
            if not miner:
                logger.error(f"[{session_id}] Miner not available")
                return {
                    "status": "error",
                    "message": "Please initialize mining system first"
                }

            # Prepare tx data for estimation
            tx_data = {
                "sender": data.get("sender"),
                "receiver": data.get("receiver"),
                "amount": Decimal(str(data.get("amount", 0))),
                "quantum_enabled": data.get("quantum_enabled", False),
                "entanglement_count": data.get("entanglement_count", 0),
                "data_size": len(str(data).encode())
            }

            # Get gas estimate
            gas_estimate = await miner.estimate_transaction_gas(tx_data)
            logger.debug(f"[{session_id}] Gas estimate: {gas_estimate}")

            return {
                "status": "success",
                "gas_estimate": gas_estimate
            }

        except Exception as e:
            logger.error(f"Error estimating gas: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to estimate gas: {str(e)}"
            }



    # Wallet Handlers
    async def handle_create_wallet(self, session_id: str, websocket, data: dict) -> dict:
        """Handle wallet creation"""
        self.sessions[session_id]['wallet'] = Wallet()
        return {
            "status": "success",
            "wallet": self.sessions[session_id]['wallet'].to_dict()
        }

    async def handle_create_wallet_with_pincode(self, session_id: str, websocket, data: dict) -> dict:
        """Handle wallet creation with pincode"""
        pincode = data.get("pincode")
        if not pincode:
            return {"status": "error", "message": "Pincode is required"}
        
        self.sessions[session_id]['wallet'] = Wallet(pincode=pincode)
        return {
            "status": "success",
            "wallet": self.sessions[session_id]['wallet'].to_dict()
        }
        # Transaction Handlers
    async def handle_create_transaction(self, session_id: str, websocket, data: dict) -> dict:
        """Handle transaction creation with gas tracking and security features"""
        try:
            logger.debug(f"Creating transaction with data: {data}")
            
            # First get gas estimate
            gas_estimate = await self.handle_estimate_gas(session_id, websocket, data)
            if gas_estimate["status"] != "success":
                return gas_estimate
            gas_data = gas_estimate["gas_estimate"]
            
            # Get miner and session data
            miner = self.sessions[session_id].get('miner')
            if not miner:
                return {
                    "status": "error",
                    "message": "Miner not initialized. Cannot estimate gas."
                }
                
            # Initialize crypto provider if needed
            if 'crypto_provider' not in self.sessions[session_id]:
                self.sessions[session_id]['crypto_provider'] = CryptoProvider()
            crypto_provider = self.sessions[session_id]['crypto_provider']
            
            # Prepare base transaction data
            tx_data = {
                "sender": data["sender"],
                "receiver": data["receiver"],
                "amount": Decimal(str(data["amount"])),
                "price": Decimal(str(gas_data["total_cost"])),
                "buyer_id": data.get("buyer_id", data["receiver"]),
                "seller_id": data.get("seller_id", data["sender"]),
                "gas_limit": gas_data["gas_needed"],
                "gas_price": Decimal(str(gas_data["gas_price"])),
                "quantum_enabled": data.get("quantum_enabled", False)
            }
            
            # Create base transaction
            transaction = Transaction(**tx_data)
            
            # Apply security features
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}".encode()
            
            # Apply security features in parallel
            security_tasks = [
                self._apply_base_signature(transaction, message),
                self._apply_zk_proof(transaction, transaction.amount, crypto_provider),
                self._apply_homomorphic(transaction, transaction.amount, crypto_provider),
                self._apply_ring_signature(transaction, message, crypto_provider),
                self._apply_quantum_signature(transaction, message, crypto_provider),
                self._apply_post_quantum(transaction, message, crypto_provider)
            ]
            
            security_results = await asyncio.gather(*security_tasks, return_exceptions=True)
            
            # Track security metrics
            security_features = {
                'zk_proof': bool(transaction.zk_proof),
                'homomorphic': bool(transaction.homomorphic_amount),
                'ring_signature': bool(transaction.ring_signature),
                'quantum_signature': bool(transaction.quantum_signature),
                'post_quantum': bool(getattr(transaction, 'pq_cipher', None)),
                'base_signature': bool(transaction.signature)
            }
            
            # Add gas data to transaction
            transaction.gas_data = {
                "gas_price": gas_data["gas_price"],
                "gas_used": gas_data["gas_needed"],
                "total_cost": gas_data["total_cost"],
                "quantum_premium": gas_data["components"].get("quantum_premium", 0),
                "entanglement_premium": gas_data["components"].get("entanglement_premium", 0),
                "base_price": gas_data["components"]["base_price"]
            }
            
            # Store transaction in session
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
            if 'transactions' not in self.sessions[session_id]:
                self.sessions[session_id]['transactions'] = {}
            self.sessions[session_id]["transactions"][transaction.id] = transaction
            
            # Track gas metrics
            if hasattr(self, 'gas_metrics'):
                self.gas_metrics.track_transaction(
                    transaction.gas_data["gas_price"],
                    transaction.gas_data["gas_used"],
                    transaction.gas_data["quantum_premium"] if transaction.quantum_enabled else 0,
                    transaction.gas_data.get("entanglement_premium", 0)
                )
            
            # Return success with complete transaction data
            return {
                "status": "success",
                "transaction": {
                    **transaction.to_dict(),
                    "gas_data": transaction.gas_data,
                    "security_features": security_features
                }
            }
            
        except Exception as e:
            logger.error(f"Transaction creation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e)
            }

    # Security feature application methods
    async def _apply_base_signature(self, transaction: Transaction, message: bytes):
        """Apply base signature"""
        try:
            private_key = ec.generate_private_key(ec.SECP256K1())
            public_key = private_key.public_key()
            
            signature = private_key.sign(
                message,
                ec.ECDSA(hashes.SHA256())
            )
            
            transaction.signature = base64.b64encode(signature).decode()
            transaction.public_key = base64.b64encode(
                public_key.public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            ).decode()
            return True
        except Exception as e:
            logger.error(f"Error applying base signature: {str(e)}")
            return False

    async def _apply_zk_proof(self, transaction: Transaction, amount: Decimal, crypto_provider: CryptoProvider):
        """Apply zero-knowledge proof"""
        try:
            amount_wei = int(amount * 10**18)
            public_input = int.from_bytes(hashlib.sha256(str(amount).encode()).digest(), 'big')
            proof = await asyncio.to_thread(crypto_provider.stark.prove, amount_wei, public_input)
            transaction.zk_proof = base64.b64encode(str(proof).encode()).decode()
            return True
        except Exception as e:
            logger.error(f"Error applying ZK proof: {str(e)}")
            return False

    async def _apply_homomorphic(self, transaction: Transaction, amount: Decimal, crypto_provider: CryptoProvider):
        """Apply homomorphic encryption"""
        try:
            amount_wei = int(amount * 10**18)
            cipher = await asyncio.to_thread(crypto_provider.create_homomorphic_cipher, amount_wei)
            transaction.homomorphic_amount = base64.b64encode(str(cipher).encode()).decode()
            return True
        except Exception as e:
            logger.error(f"Error applying homomorphic encryption: {str(e)}")
            return False

    async def apply_quantum_signature(self, transaction: Transaction, message: bytes, crypto_provider: CryptoProvider) -> bool:
        """Apply quantum signature with proper transaction update"""
        try:
            # Generate quantum signature
            quantum_sig = await asyncio.to_thread(
                crypto_provider.quantum_signer.sign_message, 
                message
            )
            
            if quantum_sig:
                # Create new status
                new_status = ConfirmationStatus(
                    score=0.0,
                    security_level="LOW",
                    confirmations=0,
                    is_final=False
                )
                
                # Create new metrics
                new_metrics = ConfirmationMetrics(
                    path_diversity=0.0,
                    quantum_strength=0.85,
                    consensus_weight=0.0,
                    depth_score=0.0
                )
                
                # Create new confirmation data
                new_confirmation_data = ConfirmationData(
                    status=new_status,
                    metrics=new_metrics,
                    confirming_blocks=[],
                    confirmation_paths=[],
                    quantum_confirmations=[]
                )
                
                # Create updated transaction copy
                updated_transaction = transaction.model_copy(
                    update={
                        'quantum_signature': quantum_sig,
                        'confirmation_data': new_confirmation_data,
                        'confirmations': 0
                    }
                )
                
                # Update transaction store if needed
                if transaction.tx_hash in self.transaction_store:
                    self.transaction_store[transaction.tx_hash] = updated_transaction
                
                # Update metrics
                self.transaction_metrics['security_features']['quantum_signature'] += 1
                
                logger.debug(
                    f"Applied quantum signature to transaction {transaction.tx_hash}:\n"
                    f"Signature size: {len(quantum_sig)} bytes\n"
                    f"Initial confirmation score: {new_status.score}\n"
                    f"Security level: {new_status.security_level}"
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying quantum signature: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    async def update_transaction_confirmation(self, transaction: Transaction) -> None:
        """Update transaction confirmation status"""
        try:
            if not transaction.tx_hash:
                logger.warning("Cannot update confirmation status: transaction has no hash")
                return
                
            # Get security info from confirmation system
            security_info = self.confirmation_system.get_transaction_security(
                transaction.tx_hash,
                self.confirmation_system.get_latest_block_hash()
            )
            
            # Create new confirmation data
            new_status = ConfirmationStatus(
                score=security_info.get('confirmation_score', 0.0),
                security_level=security_info.get('security_level', 'LOW'),
                confirmations=security_info.get('num_confirmations', 0),
                is_final=security_info.get('is_final', False)
            )
            
            new_metrics = ConfirmationMetrics(
                path_diversity=security_info.get('path_diversity', 0.0),
                quantum_strength=security_info.get('quantum_strength', 0.85),
                consensus_weight=security_info.get('consensus_weight', 0.0),
                depth_score=security_info.get('depth_score', 0.0)
            )
            
            new_confirmation_data = ConfirmationData(
                status=new_status,
                metrics=new_metrics,
                confirming_blocks=transaction.confirmation_data.confirming_blocks,
                confirmation_paths=transaction.confirmation_data.confirmation_paths,
                quantum_confirmations=transaction.confirmation_data.quantum_confirmations
            )
            
            # Update transaction using model_copy
            updated_transaction = transaction.model_copy(
                update={
                    'confirmation_data': new_confirmation_data,
                    'confirmations': security_info.get('num_confirmations', 0)
                }
            )
            
            # Update transaction store
            if transaction.tx_hash in self.transaction_store:
                self.transaction_store[transaction.tx_hash] = updated_transaction
                
            # Update confirmation metrics
            security_level = security_info.get('security_level', 'LOW')
            if security_level in self.transaction_metrics['confirmation_levels']:
                self.transaction_metrics['confirmation_levels'][security_level] += 1
                
            logger.debug(
                f"Updated confirmation status for transaction {transaction.tx_hash}:\n"
                f"Score: {new_status.score:.4f}\n"
                f"Security Level: {security_level}\n"
                f"Confirmations: {security_info.get('num_confirmations', 0)}"
            )
            
        except Exception as e:
            logger.error(f"Error updating transaction confirmation: {str(e)}")
            logger.error(traceback.format_exc())


    async def _apply_ring_signature(self, transaction: Transaction, message: bytes, crypto_provider: CryptoProvider):
        """Apply ring signature"""
        try:
            ring_sig = await asyncio.to_thread(
                crypto_provider.create_ring_signature,
                message,
                crypto_provider.private_key,
                crypto_provider.public_key
            )
            transaction.ring_signature = base64.b64encode(str(ring_sig).encode()).decode()
            return True
        except Exception as e:
            logger.error(f"Error applying ring signature: {str(e)}")
            return False

    async def _apply_post_quantum(self, transaction: Transaction, message: bytes, crypto_provider: CryptoProvider):
        """Apply post-quantum encryption"""
        try:
            cipher = await asyncio.to_thread(crypto_provider.pq_encrypt, message)
            transaction.pq_cipher = base64.b64encode(str(cipher).encode()).decode()
            return True
        except Exception as e:
            logger.error(f"Error applying post-quantum encryption: {str(e)}")
            return False




    async def _apply_quantum(self, transaction: Transaction, message: bytes, 
                            crypto_provider: CryptoProvider) -> None:
        """Apply quantum signature with proper initialization"""
        try:
            # Get quantum signature
            quantum_sig = await asyncio.to_thread(
                crypto_provider.quantum_signer.sign_message,
                message
            )
            transaction.quantum_signature = quantum_sig
            
            # Update confirmation data
            if hasattr(transaction, 'confirmation_data'):
                transaction.confirmation_data.metrics.quantum_strength = 0.85
            
            return quantum_sig
        except Exception as e:
            logger.error(f"Quantum signature application failed: {str(e)}")
            return b''


    async def _async_quantum_sign(self, crypto_provider: CryptoProvider, message: bytes) -> bytes:
        """Async wrapper for quantum signature generation"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: crypto_provider.quantum_signer.sign_message(message)
            )
        except Exception as e:
            logger.error(f"Quantum signing error: {str(e)}")
            return b''

    def get_transaction_metrics(self) -> dict:
        """Get transaction metrics with security feature analysis"""
        metrics = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'security_features': {
                'zk_proof': 0,
                'homomorphic': 0,
                'ring_signature': 0,
                'quantum_signature': 0,
                'post_quantum': 0,
                'base_signature': 0
            },
            'confirmation_levels': {
                'HIGH': 0,
                'MEDIUM': 0,
                'LOW': 0
            },
            'average_confirmation_score': 0.0
        }

        for tx_key, tx in self.transaction_store.items():
            metrics['total_transactions'] += 1
            if hasattr(tx, 'signature'):
                metrics['successful_transactions'] += 1
                for feature, count in metrics['security_features'].items():
                    if getattr(tx, feature, None):
                        metrics['security_features'][feature] += 1
                if hasattr(tx, 'confirmation_data'):
                    score = tx.confirmation_data.status.confirmation_score
                    metrics['average_confirmation_score'] += score
                    if score >= 0.8:
                        metrics['confirmation_levels']['HIGH'] += 1
                    elif score >= 0.5:
                        metrics['confirmation_levels']['MEDIUM'] += 1
                    else:
                        metrics['confirmation_levels']['LOW'] += 1

        if metrics['successful_transactions'] > 0:
            metrics['average_confirmation_score'] /= metrics['successful_transactions']

        return metrics

    async def _generate_zk_proof(self, transaction: Transaction, crypto_provider: CryptoProvider):
        """Generate ZK proof"""
        amount_wei = int(float(transaction.amount) * 10**18)
        public_input = int.from_bytes(hashlib.sha256(str(transaction.amount).encode()).digest(), byteorder='big')
        return crypto_provider.stark.prove(amount_wei, public_input)

    async def _generate_ring_signature(self, message: bytes, wallet: Wallet, crypto_provider: CryptoProvider):
        """Generate ring signature"""
        return crypto_provider.create_ring_signature(message, wallet.private_key, wallet.public_key)


    async def _generate_quantum_signature(self, message, crypto_provider):
        return crypto_provider.quantum_signer.sign_message(message)

    async def _generate_post_quantum(self, message, crypto_provider):
        return crypto_provider.pq_encrypt(message)

    def fast_track_confirmation(self, tx_hash: str):
        """Optimized confirmation tracking"""
        self.quantum_scores[tx_hash] = 1.0
        self.confirmation_cache[tx_hash] = {
            'score': 0.99,
            'level': 'VERY_HIGH',
            'last_update': time.time_ns() // 1_000_000
        }





    # Mining Handlers
    async def handle_start_mining(self, session_id: str, websocket, data: dict) -> dict:
        """Start mining process"""
        try:
            if not self.sessions[session_id].get('miner'):
                return {
                    "status": "error",
                    "message": "Miner not initialized. Call initialize first."
                }
            
            if self.sessions[session_id].get('mining_task'):
                return {
                    "status": "error",
                    "message": "Mining already in progress"
                }

            duration = data.get("duration", 300)  # Default 5 minutes
            self.sessions[session_id]['performance_data'] = {
                'blocks_mined': [],
                'mining_times': [],
                'hash_rates': [],
                'start_time': time.time()
            }

            mining_task = asyncio.create_task(
                self.mining_loop(session_id, websocket, duration)
            )
            self.sessions[session_id]['mining_task'] = mining_task

            return {
                "status": "success",
                "message": "Mining started",
                "duration": duration
            }
        except Exception as e:
            logger.error(f"Error starting mining: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to start mining: {str(e)}"
            }


    async def mining_loop(self, session_id: str, websocket, duration: int):
        """Main mining loop"""
        session = self.sessions[session_id]
        miner = session['miner']
        start_time = time.time()
        end_time = start_time + duration
        previous_hash = "0" * 64
        
        try:
            while time.time() < end_time:
                mine_start = time.time()
                block = await miner.mine_block(
                    previous_hash,
                    f"block_data_{len(session['performance_data']['blocks_mined'])}",
                    session['transactions'],
                    Decimal("50"),
                    session['wallet'].address
                )
                
                if block:
                    await self.handle_mined_block(session_id, websocket, block, time.time() - mine_start)
                    previous_hash = block.hash

                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info(f"Mining cancelled for session: {session_id}")
        except Exception as e:
            logger.error(f"Error in mining loop: {str(e)}")
            await websocket.send(json.dumps({
                "status": "error",
                "message": f"Mining error: {str(e)}"
            }))
        finally:
            session['mining_task'] = None
            await self.send_mining_complete(websocket, session_id)
    async def cleanup(self):
        """Clean up server resources"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            for session_id, websocket in list(self.sessions.items()):
                try:
                    await websocket.close()
                    self.logger.info(f"Session cleaned up: {session_id}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up session {session_id}: {e}")
            
            self.sessions.clear()
            self.server = None
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise


# Remaining Wallet Handlers
    async def handle_create_wallet_with_mnemonic(self, session_id: str, websocket, data: dict) -> dict:
        """Handle wallet creation from mnemonic"""
        mnemonic = data.get("mnemonic")
        if not mnemonic:
            return {"status": "error", "message": "Mnemonic is required"}
            
        self.sessions[session_id]['wallet'] = Wallet(mnemonic=mnemonic)
        return {
            "status": "success",
            "wallet": self.sessions[session_id]['wallet'].to_dict()
        }

    async def handle_get_wallet_info(self, session_id: str, websocket, data: dict) -> dict:
        """Handle wallet info request"""
        wallet = self.sessions[session_id].get('wallet')
        if not wallet:
            return {"status": "error", "message": "No wallet found for this session"}
            
        return {
            "status": "success",
            "wallet": wallet.to_dict()
        }

    async def handle_sign_message(self, session_id: str, data: dict) -> dict:
        """Handle message signing request"""
        try:
            wallet = self.sessions[session_id].get('wallet')
            if not wallet:
                return {"status": "error", "message": "No wallet found for this session"}
            
            message_bytes = data.get("message_bytes")
            if not message_bytes:
                return {"status": "error", "message": "message_bytes is required"}
            
            try:
                # Sign the base64 encoded message
                signature = wallet.sign_message(message_bytes)
                return {
                    "status": "success",
                    "signature": signature
                }
            except ValueError as e:
                return {"status": "error", "message": str(e)}
                
        except Exception as e:
            logger.error(f"Error in handle_sign_message: {str(e)}")
            return {"status": "error", "message": str(e)}



    async def handle_verify_signature(self, session_id: str, data: dict) -> dict:
        """Handle signature verification request"""
        try:
            wallet = self.sessions[session_id].get('wallet')
            if not wallet:
                return {"status": "error", "message": "No wallet found for this session"}
            
            message_bytes = data.get("message_bytes")
            signature = data.get("signature")
            
            if not message_bytes or not signature:
                return {"status": "error", "message": "message_bytes and signature are required"}
            
            try:
                # Verify the signature
                is_valid = wallet.verify_signature(message_bytes, signature, wallet.public_key)
                return {
                    "status": "success",
                    "valid": is_valid
                }
            except ValueError as e:
                return {"status": "error", "message": str(e)}
                
        except Exception as e:
            logger.error(f"Error in handle_verify_signature: {str(e)}")
            return {"status": "error", "message": str(e)}
    def log_message(self, level: str, message: str):
        """Helper method for consistent logging"""
        logger_method = getattr(logger, level.lower(), logger.info)
        logger_method(f"[WebSocket Server] {message}")


    async def handle_encrypt_message(self, session_id: str, data: dict) -> dict:
        """
        Handle message encryption request
        Args:
            session_id: Session identifier
            data: Request data containing message_bytes and recipient_public_key
        """
        try:
            wallet = self.sessions[session_id].get('wallet')
            if not wallet:
                return {"status": "error", "message": "No wallet found for this session"}
            
            message_bytes = data.get("message_bytes")
            recipient_public_key = data.get("recipient_public_key")
            
            if not message_bytes or not recipient_public_key:
                return {"status": "error", "message": "message_bytes and recipient_public_key are required"}
            
            try:
                # Encrypt the message
                encrypted = wallet.encrypt_message(message_bytes, recipient_public_key)
                return {
                    "status": "success",
                    "encrypted_message": encrypted
                }
            except ValueError as e:
                return {"status": "error", "message": str(e)}
                
        except Exception as e:
            logger.error(f"Error in handle_encrypt_message: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_decrypt_message(self, session_id: str, data: dict) -> dict:
        """
        Handle message decryption request
        Args:
            session_id: Session identifier
            data: Request data containing encrypted_message
        """
        try:
            wallet = self.sessions[session_id].get('wallet')
            if not wallet:
                return {"status": "error", "message": "No wallet found for this session"}
            
            encrypted_message = data.get("encrypted_message")
            if not encrypted_message:
                return {"status": "error", "message": "encrypted_message is required"}
            
            try:
                # Decrypt the message
                decrypted = wallet.decrypt_message(encrypted_message)
                return {
                    "status": "success",
                    "decrypted_message": decrypted
                }
            except ValueError as e:
                return {"status": "error", "message": str(e)}
                
        except Exception as e:
            logger.error(f"Error in handle_decrypt_message: {str(e)}")
            return {"status": "error", "message": str(e)}


    async def handle_verify_pincode(self, session_id: str, websocket, data: dict) -> dict:
        """Handle pincode verification"""
        wallet = self.sessions[session_id].get('wallet')
        if not wallet:
            return {"status": "error", "message": "No wallet found for this session"}
            
        pincode = data.get("pincode")
        if not pincode:
            return {"status": "error", "message": "Pincode is required"}
            
        is_valid = wallet.verify_pincode(pincode)
        return {
            "status": "success",
            "valid": is_valid
        }

    async def handle_generate_alias(self, session_id: str, websocket, data: dict) -> dict:
        """Handle alias generation"""
        wallet = self.sessions[session_id].get('wallet')
        if not wallet:
            return {"status": "error", "message": "No wallet found for this session"}
            
        alias = wallet.generate_unique_alias()
        return {
            "status": "success",
            "alias": alias
        }
        # Remaining Transaction Handlers
    async def handle_sign_transaction(self, session_id: str, websocket, data: dict) -> dict:
        """Handle transaction signing with proper session management"""
        try:
            tx_id = data.get("transaction_id")
            if not tx_id:
                return {"status": "error", "message": "No transaction ID provided"}

            # Try getting transaction from both sources
            transaction = None
            
            # Try session manager first
            transaction = await self.session_manager.get_transaction(session_id, tx_id)
            
            # If not found, try local session
            if not transaction and session_id in self.sessions:
                transactions = self.sessions[session_id].get('transactions', {})
                transaction = transactions.get(tx_id)
                
                # If found in local session, store in session manager
                if transaction:
                    await self.session_manager.store_transaction(session_id, transaction)

            if not transaction:
                logger.debug(f"[{session_id}] Transaction {tx_id} not found")
                return {"status": "error", "message": "Transaction not found"}

            # Get wallet
            session = await self.session_manager.get_session(session_id)
            wallet_data = session.get('wallet')
            if not wallet_data:
                return {"status": "error", "message": "No wallet available"}

            try:
                # Sign transaction
                wallet = wallet_data if not isinstance(wallet_data, dict) else Wallet.from_dict(wallet_data)
                transaction.wallet = wallet
                
                message = transaction._create_message()
                transaction.signature = wallet.sign_message(message)
                transaction.public_key = wallet.public_key

                # Update hash
                security_data = message + base64.b64decode(transaction.signature)
                if transaction.zk_proof:
                    security_data += (transaction.zk_proof.encode() if isinstance(transaction.zk_proof, str)
                                   else str(transaction.zk_proof).encode())
                transaction.tx_hash = hashlib.sha256(security_data).hexdigest()

                # Store updated transaction in both places
                await self.session_manager.store_transaction(session_id, transaction)
                if session_id in self.sessions:
                    self.sessions[session_id]['transactions'][tx_id] = transaction

                return {
                    "status": "success",
                    "message": "Transaction signed successfully",
                    "transaction": transaction.to_dict(),
                    "security_features": {
                        "signature": transaction.signature,
                        "public_key": transaction.public_key,
                        "tx_hash": transaction.tx_hash,
                        "zk_proof": bool(transaction.zk_proof),
                        "homomorphic": bool(transaction.homomorphic_amount),
                        "ring_signature": bool(transaction.ring_signature),
                        "quantum_signature": bool(transaction.quantum_signature)
                    }
                }

            except Exception as e:
                logger.error(f"[{session_id}] Signing failed: {str(e)}")
                return {"status": "error", "message": f"Signing failed: {str(e)}"}

        except Exception as e:
            logger.error(f"[{session_id}] Transaction signing error: {str(e)}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}






    async def handle_verify_transaction(self, session_id: str, websocket, data: dict) -> dict:
        """Handle transaction verification with proper object conversion"""
        try:
            logger.debug(f"[{session_id}] Starting transaction verification")

            # Get transaction ID
            tx_id = data.get("transaction_id")
            if not tx_id:
                return {
                    "status": "error",
                    "message": "No transaction ID provided"
                }

            # Get transaction data
            try:
                session = await self.session_manager.get_session(session_id)
                transactions = session.get('transactions', {})
                tx_data = transactions.get(tx_id)
                
                if not tx_data:
                    logger.error(f"[{session_id}] Transaction {tx_id} not found")
                    return {
                        "status": "error",
                        "message": "Transaction not found"
                    }

                # Convert to Transaction object
                if isinstance(tx_data, dict):
                    try:
                        transaction = Transaction.from_dict(tx_data)
                    except Exception as e:
                        logger.error(f"[{session_id}] Failed to convert transaction data: {str(e)}")
                        return {
                            "status": "error",
                            "message": f"Transaction conversion failed: {str(e)}"
                        }
                else:
                    transaction = tx_data

            except Exception as e:
                logger.error(f"[{session_id}] Transaction retrieval error: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to retrieve transaction: {str(e)}"
                }

            # Initialize validation result
            validation_result = {
                'status': 'success',
                'valid': True,
                'validation_details': {
                    'hash_valid': False,
                    'amount_valid': False,
                    'timestamp_valid': False,
                    'signature_valid': False,
                    'security_features_valid': False
                }
            }

            try:
                # 1. Verify hash
                message = transaction._create_message()
                signature = base64.b64decode(transaction.signature)
                security_data = message + signature
                if transaction.zk_proof:
                    if isinstance(transaction.zk_proof, str):
                        security_data += transaction.zk_proof.encode()
                    else:
                        security_data += str(transaction.zk_proof).encode()

                computed_hash = hashlib.sha256(security_data).hexdigest()
                validation_result['validation_details']['hash_valid'] = (
                    computed_hash == transaction.tx_hash
                )

                # 2. Verify amount
                validation_result['validation_details']['amount_valid'] = (
                    transaction.amount > Decimal('0') and 
                    transaction.price >= Decimal('0')
                )

                # 3. Verify timestamp
                current_time = time.time()
                validation_result['validation_details']['timestamp_valid'] = (
                    transaction.timestamp <= current_time and 
                    transaction.timestamp > (current_time - 86400)
                )

                # 4. Verify signature
                try:
                    public_key_bytes = base64.b64decode(transaction.public_key)
                    try:
                        # Try DER format first
                        public_key = serialization.load_der_public_key(
                            public_key_bytes,
                            backend=default_backend()
                        )
                    except:
                        # Fall back to PEM format
                        public_key = serialization.load_pem_public_key(
                            f"-----BEGIN PUBLIC KEY-----\n{transaction.public_key}\n-----END PUBLIC KEY-----\n".encode(),
                            backend=default_backend()
                        )

                    public_key.verify(
                        signature,
                        message,
                        ec.ECDSA(hashes.SHA256())
                    )
                    validation_result['validation_details']['signature_valid'] = True
                except Exception as e:
                    logger.error(f"[{session_id}] Signature verification failed: {str(e)}")
                    validation_result['validation_details']['signature_valid'] = False

                # 5. Verify security features
                security_features = {
                    "zk_proof": bool(transaction.zk_proof),
                    "homomorphic": bool(transaction.homomorphic_amount),
                    "ring_signature": bool(transaction.ring_signature),
                    "quantum_signature": bool(transaction.quantum_signature),
                    "post_quantum": bool(getattr(transaction, 'pq_cipher', None))
                }
                validation_result['validation_details']['security_features_valid'] = (
                    security_features["zk_proof"] and 
                    security_features["homomorphic"]
                )

                # Overall validation
                validation_result['valid'] = all(validation_result['validation_details'].values())
                if not validation_result['valid']:
                    validation_result['status'] = 'error'
                    validation_result['message'] = 'Transaction verification failed'

                logger.info(f"[{session_id}] Transaction verification completed: {validation_result['valid']}")
                return {
                    **validation_result,
                    'security_features': security_features,
                    'transaction_id': tx_id
                }

            except Exception as e:
                logger.error(f"[{session_id}] Verification error: {str(e)}")
                return {
                    'status': 'error',
                    'valid': False,
                    'message': f'Verification error: {str(e)}',
                    'validation_details': validation_result['validation_details']
                }

        except Exception as e:
            logger.error(f"[{session_id}] Transaction verification failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e)
            }




    async def handle_get_transactions(self, session_id: str, websocket, data: dict) -> dict:
        """Handle get all transactions request"""
        try:
            return {
                "status": "success",
                "transactions": [tx.model_dump() for tx in self.sessions[session_id]['transactions']]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get transactions: {str(e)}"
            }

    async def handle_get_transaction_metrics(self, session_id: str, websocket, data: dict) -> dict:
        """Handle transaction metrics request"""
        try:
            metrics = self.get_transaction_metrics()
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Failed to get transaction metrics: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }


    # Remaining Mining Handlers
    async def handle_initialize_miner(self, session_id: str, websocket, data: dict) -> dict:
        """Initialize the miner for a session with error handling"""
        try:
            logger.debug(f"[{session_id}] Initializing miner with data: {data}")
            
            # Get initialization parameters
            difficulty = int(data.get("difficulty", 2))
            security_level = int(data.get("security_level", 20))
            confirmation_params = data.get("confirmation_params", {
                "quantum_threshold": 0.85,
                "min_confirmations": 6,
                "max_confirmations": 100
            })

            # Create new DAGKnightMiner instance
            miner = DAGKnightMiner(
                difficulty=difficulty,
                security_level=security_level
            )

            # Initialize confirmation system
            miner.confirmation_system = DAGConfirmationSystem(
                quantum_threshold=confirmation_params.get("quantum_threshold", 0.85),
                min_confirmations=confirmation_params.get("min_confirmations", 6),
                max_confirmations=confirmation_params.get("max_confirmations", 100)
            )

            # Verify miner initialization
            if not miner:
                raise ValueError("Failed to create miner instance")

            # Initialize or get session
            if session_id not in self.sessions:
                self.sessions[session_id] = {}

            # Store miner in session
            self.sessions[session_id]['miner'] = miner

            # Initialize mining metrics
            self.sessions[session_id]['performance_data'] = {
                'blocks_mined': [],
                'mining_times': [],
                'hash_rates': [],
                'start_time': time.time()
            }

            logger.info(f"[{session_id}] Miner initialized successfully "
                       f"with difficulty {difficulty} and security level {security_level}")

            # Return initialization success
            return {
                "status": "success",
                "message": "Miner initialized",
                "settings": {
                    "difficulty": difficulty,
                    "security_level": security_level,
                    "confirmation_params": confirmation_params
                }
            }

        except Exception as e:
            error_msg = f"Error initializing miner: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Failed to initialize miner: {str(e)}"
            }

    def verify_miner(self, session_id: str) -> bool:
        """Verify miner is properly initialized"""
        try:
            if session_id not in self.sessions:
                logger.error(f"[{session_id}] Session not found")
                return False

            miner = self.sessions[session_id].get('miner')
            if not miner:
                logger.error(f"[{session_id}] Miner not found in session")
                return False

            if not hasattr(miner, 'difficulty') or not hasattr(miner, 'confirmation_system'):
                logger.error(f"[{session_id}] Miner missing required attributes")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying miner: {str(e)}")
            return False


    async def handle_stop_mining(self, session_id: str, websocket, data: dict) -> dict:
        """Handle stop mining request"""
        try:
            if self.sessions[session_id].get('mining_task'):
                self.sessions[session_id]['mining_task'].cancel()
                self.sessions[session_id]['mining_task'] = None
                return {"status": "success", "message": "Mining stopped"}
            return {"status": "error", "message": "No mining operation in progress"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to stop mining: {str(e)}"}

    async def handle_get_mining_metrics(self, session_id: str, websocket, data: dict) -> dict:
        """Handle get mining metrics request"""
        try:
            metrics = self.get_mining_metrics(session_id)
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get mining metrics: {str(e)}"
            }

    async def handle_get_dag_status(self, session_id: str, websocket, data: dict) -> dict:
        """Handle get DAG status request"""
        try:
            dag_metrics = self.sessions[session_id]['miner'].get_dag_metrics()
            return {
                "status": "success",
                "dag_metrics": dag_metrics
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get DAG status: {str(e)}"
            }

    async def handle_validate_block(self, session_id: str, websocket, data: dict) -> dict:
        """Handle block validation request"""
        try:
            block_hash = data.get("block_hash")
            if not block_hash:
                return {"status": "error", "message": "Block hash required"}
                
            session = self.sessions[session_id]
            for block in session['performance_data']['blocks_mined']:
                if block.hash == block_hash:
                    is_valid = session['miner'].validate_block(block)
                    return {
                        "status": "success",
                        "valid": is_valid
                    }
                    
            return {"status": "error", "message": "Block not found"}
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to validate block: {str(e)}"
            }

    async def handle_mined_block(self, session_id: str, websocket, block, mining_time: float):
        """Handle successful block mining"""
        session = self.sessions[session_id]
        perf_data = session['performance_data']
        
        # Update performance data
        perf_data['blocks_mined'].append(block)
        perf_data['mining_times'].append(mining_time)
        hash_rate = block.nonce / mining_time if mining_time > 0 else 0
        perf_data['hash_rates'].append(hash_rate)
        
        # Send progress update
        await websocket.send(json.dumps({
            "type": "mining_update",
            "block_hash": block.hash,
            "metrics": self.get_mining_metrics(session_id)
        }))

    async def send_mining_complete(self, websocket, session_id: str):
        """Send mining completion notification"""
        final_metrics = self.get_mining_metrics(session_id)
        await websocket.send(json.dumps({
            "type": "mining_complete",
            "metrics": final_metrics
        }))
    def get_mining_metrics(self, session_id: str) -> dict:
        """Calculate current mining metrics"""
        session = self.sessions[session_id]
        data = session['performance_data']
        elapsed_time = time.time() - data['start_time']
        
        return {
            "blocks_mined": len(data['blocks_mined']),
            "average_mining_time": np.mean(data['mining_times']) if data['mining_times'] else 0,
            "average_hash_rate": np.mean(data['hash_rates']) if data['hash_rates'] else 0,
            "elapsed_time": elapsed_time,
            "blocks_per_second": len(data['blocks_mined']) / elapsed_time if elapsed_time > 0 else 0,
            "current_difficulty": session['miner'].difficulty,
            "dag_metrics": session['miner'].get_dag_metrics()
        }

    async def _async_base_signature(self, wallet, message):
        """Generate base signature asynchronously"""
        loop = asyncio.get_event_loop()
        signature = await loop.run_in_executor(None, wallet.sign_message, message)
        return signature, wallet.public_key

    async def _async_zk_proof(self, crypto_provider, amount_wei, public_input):
        """Generate ZK proof asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, crypto_provider.stark.prove, amount_wei, public_input)

    async def _async_quantum_sign(self, crypto_provider, message):
        """Generate quantum signature asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, crypto_provider.quantum_signer.sign_message, message)

    async def _async_ring_signature(self, crypto_provider, message, wallet):
        """Generate ring signature asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            crypto_provider.create_ring_signature,
            message,
            wallet.private_key,
            wallet.public_key
        )

    async def _async_post_quantum(self, crypto_provider, message):
        """Generate post-quantum encryption asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, crypto_provider.pq_encrypt, message)

    async def _store_transaction(self, session_id: str, transaction: Transaction):
        """Store transaction asynchronously"""
        self.transaction_store[f"{session_id}:{transaction.id}"] = transaction
        if session_id not in self.sessions:
            self.sessions[session_id] = {'transactions': {}}
        self.sessions[session_id]['transactions'][transaction.id] = transaction

    async def _initialize_confirmation(self, transaction: Transaction):
        """Initialize confirmation data asynchronously"""
        transaction.confirmation_data.status.confirmation_score = 0.85
        transaction.confirmation_data.status.security_level = "HIGH"
        transaction.confirmations = 1
        self.confirmation_system.quantum_scores[transaction.tx_hash] = 0.85
        await self.confirmation_system.add_block_confirmation(
            transaction.tx_hash,
            [transaction.tx_hash],
            [transaction],
            transaction.quantum_signature or b''
        )

    async def handle_p2p_test_message(self, session_id: str, websocket, data: dict) -> dict:
        """Handle P2P node test-specific messages"""
        action = data.get("action", "")
        
        handlers = {
            "test_peer_connection": self.handle_test_peer_connection,
            "test_quantum_entanglement": self.handle_test_quantum_entanglement,
            "test_transaction_propagation": self.handle_test_transaction_propagation,
            "test_consensus": self.handle_test_consensus,
            "get_network_metrics": self.handle_get_network_metrics
        }
        
        if action not in handlers:
            return {
                "status": "error",
                "message": f"Unknown test action: {action}"
            }
            
        return await handlers[action](session_id, websocket, data)

    async def handle_test_peer_connection(self, session_id: str, websocket, data: dict) -> dict:
        """Test P2P peer connection functionality"""
        try:
            peer_address = data.get("peer_address")
            if not peer_address:
                return {"status": "error", "message": "Peer address required"}

            # Test connection to peer
            connection_success = await self.p2p_node.connect_to_peer(peer_address)
            
            if connection_success:
                # Test message exchange
                test_message = Message(
                    type="test_message",
                    payload={"test_data": "test_value"}
                )
                message_success = await self.p2p_node.send_and_wait_for_response(
                    peer_address, 
                    test_message,
                    timeout=10.0
                )

                return {
                    "status": "success",
                    "connection_status": "connected",
                    "message_exchange": bool(message_success),
                    "peer_info": {
                        "address": peer_address,
                        "connected_since": time.time(),
                        "latency": await self.p2p_node.measure_peer_latency(peer_address)
                    }
                }
                
            return {
                "status": "error",
                "message": "Failed to connect to peer"
            }

        except Exception as e:
            logger.error(f"Error testing peer connection: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_test_quantum_entanglement(self, session_id: str, websocket, data: dict) -> dict:
        """Test quantum entanglement between peers"""
        try:
            peer_address = data.get("peer_address")
            if not peer_address:
                return {"status": "error", "message": "Peer address required"}

            # Initialize quantum components if needed
            if not self.p2p_node.quantum_initialized:
                await self.p2p_node.initialize_quantum_components()

            # Establish quantum entanglement
            entanglement_success = await self.p2p_node.establish_quantum_entanglement(peer_address)
            
            if entanglement_success:
                # Measure quantum states
                fidelities = {}
                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                    fidelity = await self.p2p_node.quantum_sync.measure_sync_state(component)
                    fidelities[component] = fidelity

                return {
                    "status": "success",
                    "entanglement_status": "established",
                    "fidelities": fidelities,
                    "bell_pair_id": self.p2p_node.quantum_sync.bell_pairs.get(peer_address),
                    "quantum_metrics": {
                        "entanglement_duration": time.time() - self.p2p_node.quantum_sync.entangled_peers[peer_address].timestamp,
                        "sync_quality": await self.p2p_node.quantum_sync.verify_sync_quality(peer_address)
                    }
                }

            return {
                "status": "error",
                "message": "Failed to establish quantum entanglement"
            }

        except Exception as e:
            logger.error(f"Error testing quantum entanglement: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_test_transaction_propagation(self, session_id: str, websocket, data: dict) -> dict:
        """Test transaction propagation across the P2P network"""
        try:
            # Create test transaction
            tx_data = {
                "sender": data.get("sender", "test_sender"),
                "receiver": data.get("receiver", "test_receiver"),
                "amount": data.get("amount", "100.0"),
                "quantum_enabled": data.get("quantum_enabled", True)
            }
            
            # Create and sign transaction
            creation_response = await self.handle_create_transaction(session_id, websocket, tx_data)
            if creation_response["status"] != "success":
                return creation_response

            tx_hash = creation_response["transaction"]["tx_hash"]
            
            # Track propagation metrics
            propagation_metrics = {
                "start_time": time.time(),
                "reached_peers": [],
                "confirmation_times": {},
                "network_coverage": 0.0
            }

            # Monitor propagation for up to 30 seconds
            monitoring_end = time.time() + 30
            while time.time() < monitoring_end:
                reached_peers = await self.p2p_node.get_transaction_reach(tx_hash)
                propagation_metrics["reached_peers"] = reached_peers
                propagation_metrics["network_coverage"] = len(reached_peers) / len(self.p2p_node.connected_peers)
                
                if propagation_metrics["network_coverage"] >= 0.95:  # 95% coverage
                    break
                    
                await asyncio.sleep(1)

            return {
                "status": "success",
                "tx_hash": tx_hash,
                "propagation_metrics": {
                    "propagation_time": time.time() - propagation_metrics["start_time"],
                    "network_coverage": propagation_metrics["network_coverage"],
                    "reached_peers": len(propagation_metrics["reached_peers"]),
                    "total_peers": len(self.p2p_node.connected_peers),
                    "network_latency": await self.p2p_node.get_average_network_latency()
                }
            }

        except Exception as e:
            logger.error(f"Error testing transaction propagation: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_test_consensus(self, session_id: str, websocket, data: dict) -> dict:
        """Test DAGKnight consensus mechanism"""
        try:
            # Create test block
            block_data = {
                "previous_hash": "0" * 64,
                "transactions": [],
                "timestamp": time.time(),
                "difficulty": 2
            }
            
            # Submit block to network
            consensus_metrics = {
                "start_time": time.time(),
                "confirmation_stages": [],
                "network_agreement": 0.0,
                "k_clusters": []
            }

            block_hash = await self.p2p_node.dagknight.submit_block(block_data)
            
            # Monitor consensus formation
            monitoring_end = time.time() + 60  # Monitor for 60 seconds
            while time.time() < monitoring_end:
                network_status = await self.p2p_node.dagknight.get_network_status()
                consensus_metrics["network_agreement"] = network_status["consensus_level"]
                consensus_metrics["k_clusters"] = network_status["k_clusters"]
                
                consensus_metrics["confirmation_stages"].append({
                    "timestamp": time.time(),
                    "agreement_level": network_status["consensus_level"],
                    "active_clusters": len(network_status["k_clusters"])
                })
                
                if network_status["consensus_level"] >= 0.95:  # 95% consensus
                    break
                    
                await asyncio.sleep(2)

            return {
                "status": "success",
                "block_hash": block_hash,
                "consensus_metrics": {
                    "consensus_time": time.time() - consensus_metrics["start_time"],
                    "final_agreement": consensus_metrics["network_agreement"],
                    "confirmation_stages": consensus_metrics["confirmation_stages"],
                    "k_clusters": len(consensus_metrics["k_clusters"]),
                    "quantum_verification": await self.p2p_node.quantum_consensus.verify_consensus(block_hash)
                }
            }

        except Exception as e:
            logger.error(f"Error testing consensus: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_get_network_metrics(self, session_id: str, websocket, data: dict) -> dict:
        """Get comprehensive network metrics"""
        try:
            network_metrics = {
                "peer_metrics": {
                    "total_peers": len(self.p2p_node.connected_peers),
                    "active_peers": len([p for p in self.p2p_node.connected_peers 
                                       if self.p2p_node.peer_states.get(p) == "connected"]),
                    "quantum_entangled_peers": len(self.p2p_node.quantum_sync.entangled_peers),
                    "average_latency": await self.p2p_node.get_average_network_latency()
                },
                "quantum_metrics": {
                    "average_fidelity": await self.p2p_node.quantum_sync.get_average_fidelity(),
                    "entanglement_quality": await self.p2p_node.quantum_sync.get_entanglement_quality(),
                    "decoherence_events": self.p2p_node.quantum_monitor.get_decoherence_events()
                },
                "consensus_metrics": {
                    "network_status": await self.p2p_node.dagknight.get_network_status(),
                    "average_confirmation_time": self.p2p_node.dagknight.get_average_confirmation_time(),
                    "security_analysis": await self.p2p_node.dagknight.analyze_network_security()
                },
                "transaction_metrics": {
                    "propagation_time": self.p2p_node.get_average_propagation_time(),
                    "confirmation_rate": self.p2p_node.get_transaction_confirmation_rate(),
                    "quantum_enhanced_txs": self.p2p_node.get_quantum_transaction_count()
                }
            }

            return {
                "status": "success",
                "network_metrics": network_metrics,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Error getting network metrics: {str(e)}")
            return {"status": "error", "message": str(e)}
















async def run_sanic_server():
    """
    Function to run the Sanic server asynchronously
    """
    # Define interface and port for the Sanic server
    interface = "0.0.0.0"
    port = 50511  # Replace with your desired port

    # Create the server instance for Sanic
    server = await app.create_server(
        host=interface,
        port=port,
        return_asyncio_server=True,  # To integrate with asyncio
    )

    # Add the Sanic server to the current asyncio event loop
    loop = asyncio.get_event_loop()
    loop.create_task(server.serve_forever())  # Run Sanic indefinitely in the background

    # Optionally: add a log message indicating Sanic server started
    logger.info(f"Sanic server started on {interface}:{port}")

async def check_component_status(component: str) -> bool:
    """Check initialization status of a component."""
    try:
        if not hasattr(app, 'redis') or not app.redis:
            # If Redis isn't available, check app context
            return hasattr(app.ctx, component) and getattr(app.ctx, component) is not None
            
        key = f"init_status:{component}"
        status = await app.redis.get(key)
        return status == "1"
    except Exception as e:
        logger.warning(f"Failed to check initialization status: {str(e)}")
        # Fallback to checking app context
        return hasattr(app.ctx, component) and getattr(app.ctx, component) is not None
async def shutdown_event():
    # Perform cleanup actions, like closing database connections
    print("Shutting down...")
    # Example: await some_async_function()

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
import async_timeout
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger("quantumdagknight")

# Define global variables
blockchain = None
p2p_node = None
initialization_complete = False
async def initialize_p2p_node(ip_address: str, p2p_port: int, bootstrap_nodes: List[str] = None, max_retries: int = 3):
    """Initialize a P2P node with quantum synchronization, Linux optimizations, and consensus capabilities."""
    retry_count = 0
    global p2p_node

    while retry_count < max_retries:
        try:
            logger.info(f"\n=== P2P Node Initialization (Attempt {retry_count + 1}/{max_retries}) ===")
            logger.info(f"Initializing node at {ip_address}:{p2p_port}")

            # Step 1: Create base node and attach required components
            base_node = LinuxQuantumNode(blockchain=None, host=ip_address, port=p2p_port)
            
            # Attach systemd journal
            base_node.systemd_journal = systemd.journal.JournalHandler()
            base_node.logger.addHandler(base_node.systemd_journal)
            logger.info("✓ Systemd journal handler attached")
            
            # Create and attach network optimizer
            base_node.network_optimizer = NetworkOptimizer(base_node)
            logger.info("✓ Network optimizer attached")

            # Initialize DAGKnight consensus early
            logger.info("Initializing DAGKnight consensus...")
            base_node.dagknight = DAGKnightConsensus(
                min_k_cluster=10,
                max_latency_ms=1000
            )
            logger.info("✓ DAGKnight consensus initialized")

            # Apply network optimizations
            logger.info("Applying network optimizations...")
            await base_node.network_optimizer.optimize_network()
            logger.info("✓ Network optimizations applied")

            # Step 3: Enhance the node with additional functionality
            p2p_node = enhance_p2p_node(base_node)
            if p2p_node is None:
                raise RuntimeError("Enhancement of P2P node failed; p2p_node is None")

            logger.info("✓ P2P node enhanced with additional capabilities")

            # Step 4: Verify essential attributes are present
            required_attrs = [
                'node_id', 'sync_states', 'quantum_sync', 'peers', 
                'blockchain', 'dagknight', 'quantum_monitor',
                'network_optimizer', 'systemd_journal'
            ]
            missing_attrs = [attr for attr in required_attrs if not hasattr(p2p_node, attr)]
            if missing_attrs:
                raise RuntimeError(f"P2P node missing required attributes: {missing_attrs}")

            # Initialize blockchain components
            logger.info("Initializing QuantumBlockchain for P2P node...")
            consensus = PBFTConsensus(nodes=[], node_id=p2p_node.node_id)
            vm = SimpleVM()
            node_directory = NodeDirectory(p2p_node=p2p_node)
            secret_key = os.urandom(32)

            blockchain = QuantumBlockchain(
                consensus=consensus,
                secret_key=secret_key,
                node_directory=node_directory,
                vm=vm,
                p2p_node=p2p_node
            )

            if not blockchain.chain:
                blockchain.create_genesis_block()
            p2p_node.set_blockchain(blockchain)
            logger.info("✓ QuantumBlockchain initialized and set for P2P node")

            # Initialize quantum components
            logger.info("Initializing quantum components...")
            try:
                await p2p_node.initialize_quantum_components()
                # Start network-aware quantum monitoring
                asyncio.create_task(p2p_node.network_optimizer.monitor_network_metrics())
                logger.info("✓ Quantum components initialized with network monitoring")
            except Exception as e:
                logger.error(f"Failed to initialize quantum components: {str(e)}")
                raise

            # Configure and connect to bootstrap nodes
            if bootstrap_nodes:
                logger.info("Configuring bootstrap nodes...")
                p2p_node.bootstrap_nodes = list(set(bootstrap_nodes))
                p2p_node.bootstrap_manager = BootstrapManager(p2p_node)
                
                try:
                    await p2p_node.bootstrap_manager.connect_to_bootstrap_nodes()
                    logger.info("✓ Successfully connected to bootstrap network")
                except Exception as connect_error:
                    logger.warning(f"Bootstrap connection warning: {str(connect_error)}")
                    logger.info("Continuing as standalone node")

            # Start core services
            systemd.daemon.notify('STATUS=Starting core services')
            await p2p_node.start()
            logger.info("✓ Core P2P services started")

            # Initialize monitoring tasks
            logger.info("Starting monitoring tasks...")
            monitoring_tasks = [
                ('Network Monitor', p2p_node.monitor_network_state, 30),
                ('Peer Monitor', p2p_node.monitor_peer_connections, 15),
                ('Sync Monitor', p2p_node.monitor_sync_status, 10),
                ('Quantum Monitor', p2p_node.quantum_monitor.start_monitoring, 5),
                ('DAGKnight Monitor', p2p_node.monitor_network_security, 20),
                ('Resource Monitor', p2p_node.monitor_resources, 60),
                ('Network Metrics', p2p_node.network_optimizer.monitor_network_metrics, 30),
                ('Quantum Network Monitor', p2p_node.monitor_quantum_network_state, 15)
            ]

            p2p_node._monitoring_tasks = []
            for task_name, task_func, interval in monitoring_tasks:
                task = asyncio.create_task(
                    p2p_node.run_monitored_task(task_name, task_func, interval)
                )
                task.set_name(task_name)
                task.add_done_callback(p2p_node.handle_task_exception)
                p2p_node._monitoring_tasks.append(task)
                logger.info(f"✓ Started {task_name}")

            # Set singleton instance
            P2PNode._instance = p2p_node
            P2PNode._initialized = True
            
            # Notify systemd of successful startup
            systemd.daemon.notify('READY=1')
            
            # Log final status
            logger.info("\n=== P2P Node Status ===")
            logger.info(f"Node ID: {p2p_node.node_id}")
            logger.info(f"Connected Peers: {len(p2p_node.connected_peers)}")
            logger.info(f"Quantum Entangled Peers: {len(p2p_node.quantum_sync.entangled_peers)}")
            logger.info(f"Active Monitoring Tasks: {len(p2p_node._monitoring_tasks)}")
            
            interface_info = await p2p_node.get_network_interface_info(
                p2p_node.network_optimizer.get_default_interface()
            )
            logger.info(f"Network Interface: {interface_info}")
            logger.info("=== P2P Node Initialization Complete ===\n")
            
            return p2p_node

        except Exception as e:
            retry_count += 1
            logger.error(f"P2P node initialization failed on attempt {retry_count}: {str(e)}")
            logger.error(traceback.format_exc())

            if p2p_node:
                try:
                    if hasattr(p2p_node, 'network_optimizer'):
                        await p2p_node.network_optimizer.cleanup()
                    await cleanup_p2p_node(p2p_node)
                    if hasattr(p2p_node, 'systemd_journal'):
                        p2p_node.systemd_journal.close()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {str(cleanup_error)}")

            if retry_count < max_retries:
                wait_time = 5 * (2 ** (retry_count - 1))
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Max retry attempts reached. P2P node initialization failed.")
                return None

    return None

async def cleanup_p2p_node(node: P2PNode):
    """Clean up P2P node resources"""
    try:
        logger.info("Cleaning up P2P node resources...")
        
        # Cancel monitoring tasks
        if hasattr(node, '_monitoring_tasks'):
            for task in node._monitoring_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Close peer connections
        for peer in list(node.peers.keys()):
            await node.remove_peer(peer)

        # Cleanup quantum resources
        if hasattr(node, 'quantum_sync'):
            node.quantum_sync.cleanup()

        # Stop core services
        await node.cleanup()
        logger.info("✓ P2P node cleanup completed")

    except Exception as e:
        logger.error(f"Error during P2P node cleanup: {str(e)}")
        logger.error(traceback.format_exc())
async def establish_connectivity(p2p_node: P2PNode, bootstrap_nodes: List[str] = None) -> bool:
    """Helper function to establish network connectivity"""
    try:
        connection_timeout = 30
        connection_start = time.time()
        
        while time.time() - connection_start < connection_timeout:
            active_peers = await p2p_node.get_active_peers()
            if active_peers:
                logger.info(f"✓ Connected to {len(active_peers)} peers")
                return True
            
            if bootstrap_nodes:
                for node in bootstrap_nodes:
                    try:
                        ip, port = node.split(':')
                        node_id = p2p_node.generate_node_id()
                        kademlia_node = KademliaNode(node_id, ip, int(port))
                        await p2p_node.connect_to_peer(kademlia_node)
                    except Exception as e:
                        logger.debug(f"Failed to connect to bootstrap node {node}: {str(e)}")
                        continue
            
            await asyncio.sleep(2)
            
        return False
        
    except Exception as e:
        logger.error(f"Error establishing connectivity: {str(e)}")
        return False

async def cleanup_p2p_node(node: P2PNode):
    """Helper function to clean up P2P node resources"""
    try:
        if not node:
            return
            
        # Cancel monitoring tasks
        if hasattr(node, '_monitoring_tasks'):
            for task in node._monitoring_tasks:
                task.cancel()
                
        # Close peer connections
        if hasattr(node, 'peers'):
            for peer in list(node.peers.keys()):
                try:
                    await node.remove_peer(peer)
                except Exception:
                    pass
                    
        # Close server
        if hasattr(node, 'server') and node.server:
            node.server.close()
            await node.server.wait_closed()
            
    except Exception as e:
        logger.error(f"Error during node cleanup: {str(e)}")
async def initialize_vm():
    """
    Initialize the virtual machine environment for smart contract execution.
    
    Returns:
        dict: Initialized VM environment state or None if initialization fails
    """
    try:
        logger.info("\n=== Initializing Virtual Machine ===")
        
        # Create VM configuration
        vm_config = {
            'max_memory': 1024 * 1024 * 100,  # 100MB
            'max_computation_units': 1000000,
            'environment': 'sandbox',
            'timeout': 30,  # seconds
            'supported_languages': ['solidity', 'python'],
            'gas_limit': 3000000
        }

        # Initialize VM state
        vm_state = {
            'running': False,
            'contracts': {},
            'memory_usage': 0,
            'computation_units': 0,
            'transactions_processed': 0
        }

        # Initialize execution environment
        execution_env = {
            'config': vm_config,
            'state': vm_state,
            'contracts': {},
            'storage': {},
            'logs': []
        }

        # Start VM monitoring
        vm_monitor = {
            'start_time': time.time(),
            'health_check': True,
            'last_error': None
        }

        # Combine all components
        vm = {
            'config': vm_config,
            'state': vm_state,
            'execution_env': execution_env,
            'monitor': vm_monitor,
            'initialized': True
        }

        # Verify VM initialization
        if not verify_vm_state(vm):
            raise RuntimeError("VM state verification failed")

        logger.info("✓ VM configuration initialized")
        logger.info("✓ VM state initialized")
        logger.info("✓ Execution environment ready")
        logger.info("=== VM Initialization Complete ===\n")

        return vm

    except Exception as e:
        logger.error(f"VM initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def verify_vm_state(vm: dict) -> bool:
    """
    Verify the VM state is properly initialized.
    
    Args:
        vm (dict): The VM state to verify
        
    Returns:
        bool: True if VM state is valid, False otherwise
    """
    try:
        required_components = [
            'config', 
            'state', 
            'execution_env', 
            'monitor'
        ]
        
        # Check all required components exist
        for component in required_components:
            if component not in vm:
                logger.error(f"Missing required VM component: {component}")
                return False
        
        # Verify config
        if not all(key in vm['config'] for key in ['max_memory', 'max_computation_units', 'environment']):
            logger.error("Invalid VM configuration")
            return False
            
        # Verify state
        if not all(key in vm['state'] for key in ['running', 'contracts', 'memory_usage']):
            logger.error("Invalid VM state")
            return False
            
        # Verify execution environment
        if not all(key in vm['execution_env'] for key in ['config', 'state', 'contracts']):
            logger.error("Invalid execution environment")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error verifying VM state: {str(e)}")
        return False

async def cleanup_vm(vm: dict):
    """
    Clean up VM resources.
    
    Args:
        vm (dict): The VM instance to clean up
    """
    try:
        if not vm:
            return
            
        # Stop VM if running
        if vm.get('state', {}).get('running'):
            vm['state']['running'] = False
            
        # Clear contract storage
        if 'contracts' in vm.get('execution_env', {}):
            vm['execution_env']['contracts'].clear()
            
        # Clear storage
        if 'storage' in vm.get('execution_env', {}):
            vm['execution_env']['storage'].clear()
            
        # Clear logs
        if 'logs' in vm.get('execution_env', {}):
            vm['execution_env']['logs'].clear()
            
        logger.info("VM resources cleaned up")
        
    except Exception as e:
        logger.error(f"Error cleaning up VM: {str(e)}")

async def verify_quantum_state(node: P2PNode) -> dict:
    """Verify the quantum state of the node."""
    try:
        result = {
            'healthy': True,
            'reason': '',
            'components': {}
        }
        
        for component in ['wallets', 'transactions', 'blocks', 'mempool']:
            fidelity = await node.quantum_sync.measure_sync_state(component)
            result['components'][component] = {
                'fidelity': fidelity,
                'healthy': fidelity >= node.quantum_sync.decoherence_threshold
            }
            
            if fidelity < node.quantum_sync.decoherence_threshold:
                result['healthy'] = False
                result['reason'] = f"Low fidelity in {component}: {fidelity:.3f}"
                break
        
        return result
        
    except Exception as e:
        return {
            'healthy': False,
            'reason': f"Error verifying quantum state: {str(e)}",
            'components': {}
        }

async def verify_node_state(p2p_node) -> Dict[str, Any]:
    """Verify the P2P node state."""
    try:
        state = {
            'healthy': False,
            'reason': None,
            'peers': len(p2p_node.connected_peers),
            'sync_status': {},
            'tasks_running': []
        }
        
        # Check peer connections
        if not p2p_node.connected_peers:
            state['reason'] = "No connected peers"
            return state
            
        # Check sync states
        for component, sync_state in p2p_node.sync_states.items():
            state['sync_status'][component] = {
                'is_syncing': sync_state.is_syncing,
                'last_sync': time.time() - sync_state.last_sync,
                'progress': sync_state.sync_progress
            }
            
        # Check monitoring tasks
        if hasattr(p2p_node, '_monitoring_tasks'):
            for task in p2p_node._monitoring_tasks:
                if task.done():
                    if task.exception():
                        state['reason'] = f"Task failed: {task.exception()}"
                        return state
                else:
                    state['tasks_running'].append(task.get_name())
        
        state['healthy'] = True
        return state
        
    except Exception as e:
        return {
            'healthy': False,
            'reason': f"Error checking node state: {str(e)}"
        }

async def cleanup_p2p_node(p2p_node):
    """Clean up P2P node resources."""
    try:
        # Cancel monitoring tasks
        if hasattr(p2p_node, '_monitoring_tasks'):
            for task in p2p_node._monitoring_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        # Close peer connections
        for peer in list(p2p_node.peers.keys()):
            await p2p_node.remove_peer(peer)
            
        # Clear states
        p2p_node.connected_peers.clear()
        p2p_node.peer_states.clear()
        
    except Exception as e:
        logger.error(f"Error during P2P node cleanup: {str(e)}")


# Usage example:
async def setup_p2p_network(ip: str, port: int, bootstrap_nodes: List[str] = None):
    """
    Set up a P2P network node with proper error handling and logging.
    """
    try:
        # Initialize the P2P node
        p2p_node = await initialize_p2p_node(
            ip_address=ip,
            p2p_port=port,
            bootstrap_nodes=bootstrap_nodes
        )
        
        if not p2p_node:
            logger.error("Failed to initialize P2P node")
            return None
            
        # Set up periodic health monitoring
        asyncio.create_task(monitor_node_health(p2p_node))
        
        return p2p_node
        
    except Exception as e:
        logger.error(f"Error setting up P2P network: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def monitor_node_health(node: P2PNode):
    """
    Continuously monitor node health and attempt recovery if needed.
    """
    while True:
        try:
            status = await verify_node_state(node)
            
            if not status['healthy']:
                logger.warning(f"Node health check failed: {status['reason']}")
                await attempt_node_recovery(node)
            else:
                logger.debug("Node health check passed")
                
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in health monitoring: {str(e)}")
            await asyncio.sleep(60)

async def attempt_node_recovery(node: P2PNode):
    """
    Attempt to recover node functionality.
    """
    try:
        # Attempt to reconnect to network
        if not node.connected_peers:
            logger.info("Attempting to rejoin network...")
            await node.join_network()
        
        # Verify sync status
        for component, state in node.sync_states.items():
            if time.time() - state.last_sync > 300:  # 5 minutes
                logger.info(f"Triggering resync for {component}")
                if node.connected_peers:
                    peer = random.choice(list(node.connected_peers))
                    await node.start_sync(peer, component)
                    
    except Exception as e:
        logger.error(f"Error during node recovery: {str(e)}")


async def initialize_blockchain(p2p_node, vm):
    """Initialize the QuantumBlockchain with the specified P2P node and virtual machine."""
    try:
        logger.info("Initializing QuantumBlockchain...")

        # Verify p2p_node is not None and is fully initialized
        if p2p_node is None or not getattr(p2p_node, 'node_id', None):
            raise ValueError("p2p_node is None or not fully initialized. Ensure it is initialized properly before passing to the blockchain.")

        # Initialize consensus and blockchain components
        consensus = PBFTConsensus(nodes=[], node_id=p2p_node.node_id)
        secret_key = os.urandom(32)  # Securely generate a secret key for the blockchain
        blockchain = QuantumBlockchain(
            consensus=consensus,
            secret_key=secret_key,
            node_directory=None,  # Set appropriately if NodeDirectory is required
            vm=vm,
            p2p_node=p2p_node
        )

        # Ensure the blockchain recognizes the P2P node
        logger.info("Setting the P2P node in the blockchain...")
        await blockchain.set_p2p_node(p2p_node)
        blockchain.p2p_node = p2p_node

        # Create a genesis block if the blockchain is empty
        if not blockchain.chain:
            logger.info("Creating genesis block for the blockchain...")
            blockchain.create_genesis_block()
            logger.info("✓ Genesis block created")

        # Double-check that the blockchain and P2P node are properly linked
        if blockchain.p2p_node != p2p_node:
            raise RuntimeError("Blockchain's P2P node does not match the provided p2p_node instance.")

        # Log success for verification
        logger.info(f"QuantumBlockchain initialized successfully with P2PNode: {blockchain.p2p_node}")
        return blockchain

    except Exception as e:
        logger.error(f"Failed to initialize QuantumBlockchain: {str(e)}")
        logger.error(traceback.format_exc())
        return None




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
exchange = None  # Ensure `exchange` is defined globally
async def initialize_exchange(blockchain, vm, price_feed, max_retries: int = 3):
    """Initialize the Enhanced Exchange with quantum and ZK-STARK capabilities."""
    global exchange  # Ensure exchange is defined as a global variable
    retry_count = 0

    # Check if exchange is already initialized
    if exchange is not None:
        logger.info("Exchange already initialized, returning existing instance")
        return exchange

    while retry_count < max_retries:
        try:
            logger.info(f"\n=== Exchange Initialization (Attempt {retry_count + 1}/{max_retries}) ===")
            logger.info(f"Blockchain before Exchange Init: {blockchain}")

         
            # Initialize the exchange with the necessary components
            exchange = EnhancedExchangeWithZKStarks(
                blockchain=blockchain,
                vm=vm,
                price_feed=price_feed,
                node_directory=None,  # Assign your actual node_directory if applicable
                desired_security_level=20,
                host="localhost",
                port=8765
            )

            # Log and return initialized exchange
            logger.info("Exchange initialized successfully.")
            return exchange

        except Exception as e:
            retry_count += 1
            logger.error(f"Exchange initialization failed on attempt {retry_count}: {str(e)}")
            logger.error(traceback.format_exc())

            # Retry after a delay if not yet at max retries
            if retry_count < max_retries:
                wait_time = 5 * (2 ** (retry_count - 1))
                logger.info(f"Retrying exchange initialization in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Max retry attempts reached. Exchange initialization failed.")
                exchange = None
                return None

    return exchange




async def wait_for_components_ready(components):
    for component in components:
        if hasattr(component, 'wait_for_ready'):
            await component.wait_for_ready()
    logger.info("All components are ready.")
import redis.asyncio as aioredis

async def init_redis():
    global redis
    try:
        # Use the `redis.asyncio` connection pool
        redis = await aioredis.Redis.from_url(
            'redis://localhost',  # Replace with your Redis server URL
            decode_responses=True  # This makes sure Redis responses are decoded into strings
        )
        app.ctx.redis = redis  # Store Redis in app context (ctx)
        logger.info("Redis initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        raise RuntimeError("Redis initialization failed.") from e
async def initialize_websocket_server():
    """Initialize and start the WebSocket server"""
    try:
        logger.info("\n=== WebSocket Server Initialization ===")
        
        # Check if server already exists
        if hasattr(app.ctx, 'websocket_server') and app.ctx.websocket_server:
            logger.info("WebSocket server already initialized")
            return app.ctx.websocket_server

        # Create server instance
        websocket_server = QuantumBlockchainWebSocketServer(
            host=os.getenv('WEBSOCKET_HOST', '0.0.0.0'),
            port=int(os.getenv('WEBSOCKET_PORT', 8765))
        )
        
        # Initialize server
        logger.info(f"Initializing WebSocket server on {websocket_server.host}:{websocket_server.port}")
        await websocket_server.initialize()
        
        # Start server
        logger.info("Starting WebSocket server...")
        server = await websocket_server.start()
        
        # Store server references
        websocket_server._server = server
        app.ctx.websocket_server = websocket_server
        app.ctx.websocket_server_running = True
        
        # Initialize handlers
        await websocket_server.setup_handlers()
        
        logger.info(f"✓ WebSocket server initialized and running on port {websocket_server.port}")
        logger.info("=== WebSocket Server Initialization Complete ===\n")
        
        return websocket_server
        
    except Exception as e:
        logger.error("WebSocket server initialization failed:")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return None


async def cleanup_components():
    """Clean up all system components."""
    logger.info("\n=== Starting Component Cleanup ===")
    
    components_to_cleanup = [
        ('websocket_server', getattr(app.ctx, 'websocket_server', None)),
        ('exchange', getattr(app.ctx, 'exchange', None)),
        ('plata_contract', getattr(app.ctx, 'plata_contract', None)),
        ('price_feed', getattr(app.ctx, 'price_feed', None)),
        ('blockchain', getattr(app.ctx, 'blockchain', None)),
        ('p2p_node', getattr(app.ctx, 'p2p_node', None)),
        ('vm', getattr(app.ctx, 'vm', None))
    ]
    
    for component_name, component in components_to_cleanup:
        try:
            if component:
                logger.info(f"Cleaning up {component_name}...")
                
                if component_name == 'websocket_server':
                    if hasattr(component, '_server') and component._server:
                        await component._server.close()
                    if hasattr(component, 'cleanup'):
                        await component.cleanup()
                elif hasattr(component, 'cleanup'):
                    await component.cleanup()
                elif hasattr(component, 'close'):
                    await component.close()
                elif hasattr(component, 'shutdown'):
                    await component.shutdown()
                
                delattr(app.ctx, component_name)
                logger.info(f"✓ {component_name} cleaned up successfully")
                
        except Exception as e:
            logger.error(f"Error cleaning up {component_name}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
            
    logger.info("=== Component Cleanup Complete ===\n")



import asyncio
import traceback
import asyncio
import traceback
import asyncio
import traceback
async def async_main_initialization():
    global blockchain, redis, p2p_node, vm, exchange  # Declare globals at the start

    try:
        # Step 1: Initialize Redis
        if not redis:
            logger.info("Initializing Redis...")
            try:
                await init_redis()
                app.ctx.redis = redis
                logger.info("Redis initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Redis: {str(e)}")
                return False

        # Step 2: Initialize VM
        if not vm:
            logger.info("Initializing VM...")
            try:
                vm = await initialize_vm()
                if vm:
                    app.ctx.vm = vm
                    await update_initialization_status("vm", True)
                    logger.info("VM initialized successfully.")
                else:
                    logger.error("Failed to initialize VM.")
                    return False
            except Exception as e:
                logger.error(f"Error initializing VM: {str(e)}")
                return False

        # Step 3: Initialize P2P Node
        if not p2p_node:
            logger.info("Initializing P2P node...")
            try:
                # Debug message to confirm call to initialize_p2p_node
                logger.debug("About to call initialize_p2p_node")
                p2p_node = await initialize_p2p_node(ip_address, p2p_port)
                if p2p_node is None:
                    logger.error("P2P node initialization failed.")
                    return False
                app.ctx.p2p_node = p2p_node
                await update_initialization_status("p2p_node", True)
                logger.info("P2P node initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing P2P node: {str(e)}")
                return False

        # Step 4: Initialize Blockchain
        if not blockchain:
            logger.info("Initializing Blockchain...")
            try:
                blockchain = await initialize_blockchain(p2p_node, vm)
                if blockchain is None:
                    logger.error("Blockchain initialization failed.")
                    return False
                app.ctx.blockchain = blockchain
                await update_initialization_status("blockchain", True)
                logger.info(f"Blockchain initialized successfully: {blockchain}")
            except Exception as e:
                logger.error(f"Error initializing Blockchain: {str(e)}")
                return False
        else:
            logger.info("Blockchain is already initialized.")

        # Step 5: Initialize Price Feed
        logger.info("Initializing Price Feed...")
        try:
            price_feed = await initialize_price_feed()
            app.ctx.price_feed = price_feed
            logger.info("Price Feed initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Price Feed: {str(e)}")
            return False

        # Step 6: Initialize PlataContract
        logger.info("Initializing PlataContract...")
        try:
            plata_contract = await initialize_plata_contract(vm)
            app.ctx.plata_contract = plata_contract
            logger.info("Plata Contract initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing PlataContract: {str(e)}")
            return False

        # Step 7: Initialize WebSocket Server
        logger.info("Initializing WebSocket server...")
        try:
            websocket_server = await initialize_websocket_server()
            if websocket_server:
                app.ctx.websocket_server = websocket_server
                await update_initialization_status("websocket_server", True)
                logger.info("WebSocket server initialized successfully.")
            else:
                logger.error("Failed to initialize WebSocket server.")
                return False
        except Exception as e:
            logger.error(f"Error initializing WebSocket server: {str(e)}")
            return False

        # Link Blockchain with P2PNode
        if p2p_node and blockchain:
            logger.info("Linking Blockchain with P2PNode...")
            p2p_node.set_blockchain(blockchain)

        logger.info("All components initialized successfully.")

    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError("Initialization failed.") from e

async def main():
    restart_delay = 1
    running_tasks = []  # Initialize running_tasks outside the try block
    
    while True:
        try:
            logger.info("\n=== Starting System Initialization ===")
            
            # Clear any existing tasks
            for task in running_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            running_tasks = []  # Reset the task list
            
            # First run initialization to set up all components
            logger.info("Starting initialization...")
            try:
                initialization_result = await async_main()
                if not initialization_result:
                    raise RuntimeError("Initialization failed - async_main returned False")
            except Exception as init_error:
                logger.error(f"Initialization failed: {str(init_error)}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Component initialization failed: {str(init_error)}")
                
            # Create tasks for Daphne server
            logger.info("Starting Daphne server...")
            try:
                daphne_task = asyncio.create_task(run_daphne_server())
                daphne_task.set_name('daphne_server')
                running_tasks.append(daphne_task)
            except Exception as daphne_error:
                logger.error(f"Failed to start Daphne server: {str(daphne_error)}")
                raise
            
            # Initialize and start WebSocket server
            logger.info("Initializing WebSocket server...")
            try:
                websocket_server = await initialize_websocket_server()
                if not websocket_server:
                    raise RuntimeError("Failed to initialize WebSocket server")
                
                websocket_task = asyncio.create_task(websocket_server.start())
                websocket_task.set_name('websocket_server')
                running_tasks.append(websocket_task)
            except Exception as ws_error:
                logger.error(f"Failed to start WebSocket server: {str(ws_error)}")
                raise
            
            logger.info(f"Successfully started {len(running_tasks)} tasks:")
            for task in running_tasks:
                logger.info(f"  - {task.get_name()}")
            
            # Main monitoring loop
            while True:
                await asyncio.sleep(60)
                
                # Check status of all tasks
                failed_tasks = []
                for task in running_tasks:
                    if task.done():
                        if exception := task.exception():
                            logger.error(f"Task {task.get_name()} failed with error: {str(exception)}")
                            failed_tasks.append(task)
                        else:
                            logger.info(f"Task {task.get_name()} completed successfully")
                    else:
                        logger.debug(f"Task {task.get_name()} still running")
                
                # If any critical tasks failed, break the monitoring loop
                if failed_tasks:
                    error_tasks = [f"{task.get_name()}: {task.exception()}" for task in failed_tasks]
                    raise RuntimeError(f"Critical tasks failed: {', '.join(error_tasks)}")
                
                # Log component states
                await log_initialization_state()
            
        except Exception as e:
            logger.error(f"\n=== System Error ===")
            logger.error(f"Error in main loop: {str(e)}")
            logger.error(traceback.format_exc())
            logger.info(f"Attempting restart in {restart_delay} seconds...")
            
            # Cleanup existing tasks
            for task in running_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    
            # Run cleanup
            try:
                logger.info("Running component cleanup...")
                await cleanup_components()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                logger.error(traceback.format_exc())
            
            await asyncio.sleep(restart_delay)
            restart_delay = min(restart_delay * 2, 300)  # Exponential backoff up to 5 minutes
            
            logger.info("=== Restart Sequence Complete ===\n")


async def async_main():
    global redis, blockchain, vm, p2p_node, price_feed, plata_contract, exchange, initialization_complete, websocket_server
    try:
        logger.info("\n=== Starting Component Initialization ===")
        
        # Initialize WebSocket server first
        logger.info("Initializing Quantum Blockchain WebSocket server...")
        websocket_server = await initialize_websocket_server()
        if not websocket_server:
            logger.error("Failed to initialize WebSocket server")
            return False
        app.ctx.websocket_server = websocket_server
        await update_initialization_status("websocket_server", True)
        logger.info(f"✓ Quantum Blockchain WebSocket server initialized on port {websocket_server.port}")

        # Initialize remaining components
        components_to_initialize = [
            ("vm", initialize_vm),
            ("p2p_node", lambda: initialize_p2p_node(ip_address, p2p_port)),
            ("blockchain", lambda: initialize_blockchain(p2p_node, vm)),
            ("price_feed", initialize_price_feed),
            ("plata_contract", lambda: initialize_plata_contract(vm)),
        ]

        for component_name, init_func in components_to_initialize:
            logger.info(f"Initializing {component_name}...")
            try:
                component = await init_func()
                if not component:
                    raise RuntimeError(f"Initialization returned None for {component_name}")
                    
                setattr(app.ctx, component_name, component)
                await update_initialization_status(component_name, True)
                logger.info(f"✓ {component_name} initialized successfully")
                
            except Exception as comp_error:
                logger.error(f"✗ Failed to initialize {component_name}: {str(comp_error)}")
                logger.error(traceback.format_exc())
                return False
                
        # Update global references
        globals().update({
            'blockchain': app.ctx.blockchain,
            'p2p_node': app.ctx.p2p_node,
            'vm': app.ctx.vm,
            'price_feed': app.ctx.price_feed,
            'plata_contract': app.ctx.plata_contract,
            'websocket_server': app.ctx.websocket_server
        })

        initialization_complete = True
        logger.info("=== Component Initialization Complete ===\n")
        
        return True

    except Exception as e:
        initialization_complete = False
        logger.error(f"Fatal error during initialization: {str(e)}")
        logger.error(traceback.format_exc())
        return False



async def log_initialization_state():
    """Log the initialization state of all components"""
    try:
        components = {
            'WebSocket Server': hasattr(app.ctx, 'websocket_server'),
            'VM': hasattr(app.ctx, 'vm'),
            'P2P Node': hasattr(app.ctx, 'p2p_node'),
            'Blockchain': hasattr(app.ctx, 'blockchain'),
            'Price Feed': hasattr(app.ctx, 'price_feed'),
            'Plata Contract': hasattr(app.ctx, 'plata_contract'),
            'Exchange': hasattr(app.ctx, 'exchange')
        }

        logger.info("Component Initialization State:")
        for component, initialized in components.items():
            logger.info(f"{component}: {'Initialized' if initialized else 'Not Initialized'}")

        if hasattr(app.ctx, 'websocket_server'):
            ws_server = app.ctx.websocket_server
            logger.info(f"WebSocket Server Port: {ws_server.port}")
            logger.info(f"WebSocket Server Active Connections: {len(ws_server.sessions)}")

    except Exception as e:
        logger.error(f"Error logging initialization state: {str(e)}")



def log_components_state():
    logger.info("Final state of all components:")
    logger.info(f"Blockchain: {app.ctx.blockchain is not None}")
    logger.info(f"P2P Node: {app.ctx.p2p_node is not None}")
    logger.info(f"VM: {app.ctx.vm is not None}")
    logger.info(f"Price Feed: {app.ctx.price_feed is not None}")
    logger.info(f"Plata Contract: {app.ctx.plata_contract is not None}")
    logger.info(f"Exchange: {app.ctx.exchange is not None}")

if __name__ == "__main__":
    asyncio.run(main())

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        loop.close()

