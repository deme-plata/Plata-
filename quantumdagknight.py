# Import necessary modules
import os
import time
import logging
import threading
from concurrent import futures
import grpc
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
import dagknight_pb2
import dagknight_pb2_grpc
from dagknight_pb2 import *
from dagknight_pb2_grpc import DAGKnightStub
import base64
import hashlib
from grpc_reflection.v1alpha import reflection
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
import grpc
import uvicorn
import dagknight_pb2_grpc
from dagknight_pb2 import *
from dagknight_pb2_grpc import DAGKnightStub
from dagknight_pb2_grpc import DAGKnightStub, add_DAGKnightServicer_to_server
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
from common import QuantumBlock, Transaction, NodeState
from P2PNode import P2PNode
import curses

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
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
 
templates = Jinja2Templates(directory="templates")

def initialize_dashboard_ui():
    from curses_dashboard.dashboard import DashboardUI
    return DashboardUI
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



class NodeDirectory:
    def __init__(self):
        self.nodes = {}  # Dictionary to store nodes
        self.transactions = {}  # Dictionary to store transactions
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.register_times = []
        self.discover_times = []
    def store_transaction(self, transaction_hash, transaction):
        self.transactions[transaction_hash] = transaction
        print(f"Transaction {transaction_hash} stored in directory.")

    def register_node(self, node_id, node_stub):
        """Register a new node with its corresponding stub."""
        self.nodes[node_id] = node_stub

    def get_all_node_stubs(self):
        """Return a list of all node stubs."""
        return list(self.nodes.values())

    def store_transaction(self, transaction_hash, transaction):
        """Store a transaction in the directory."""
        self.transactions[transaction_hash] = transaction
        print(f"Transaction with hash {transaction_hash} stored successfully.")

    def get_transaction(self, transaction_hash):
        """Retrieve a transaction by its hash."""
        return self.transactions.get(transaction_hash)

    def get_all_transactions(self):
        """Retrieve all transactions."""
        return self.transactions

    def get_node_details(self, node_id):
        """Retrieve details of a specific node."""
        with self.lock:
            return self.nodes.get(node_id)

    def discover_nodes(self):
        """Discover and return all nodes."""
        start_time = time.time()
        with self.lock:
            nodes = [{"node_id": node_id, **info} for node_id, info in self.nodes.items()]
        end_time = time.time()
        self.discover_times.append(end_time - start_time)
        return nodes

    def get_performance_stats(self):
        """Get performance statistics for node registration and discovery."""
        return {
            "avg_register_time": statistics.mean(self.register_times) if self.register_times else 0,
            "max_register_time": max(self.register_times) if self.register_times else 0,
            "avg_discover_time": statistics.mean(self.discover_times) if self.discover_times else 0,
            "max_discover_time": max(self.discover_times) if self.discover_times else 0,
        }

    def generate_magnet_link(self, node_id, public_key, ip_address, port):
        """Generate a magnet link for a node."""
        info = f"{node_id}:{public_key}:{ip_address}:{port}"
        hash = hashlib.sha1(info.encode()).hexdigest()
        magnet_link = f"magnet:?xt=urn:sha1:{hash}&dn={node_id}&pk={base64.urlsafe_b64encode(public_key.encode()).decode()}&ip={ip_address}&port={port}"
        return magnet_link

    def register_node_with_grpc(self, node_id, public_key, ip_address, port, directory_ip, directory_port):
        """Register a node with gRPC."""
        try:
            with grpc.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
                stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                request = dagknight_pb2.RegisterNodeRequest(node_id=node_id, public_key=public_key, ip_address=ip_address, port=port)
                response = stub.RegisterNode(request)
                self.logger.info(f"Registered node with magnet link: {response.magnet_link}")
                return response.magnet_link
        except grpc.RpcError as e:
            self.logger.error(f"gRPC error when registering node: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error when registering node: {str(e)}")
            raise

    def discover_nodes_with_grpc(self, directory_ip, directory_port):
        """Discover nodes with gRPC."""
        try:
            with grpc.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
                stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                request = dagknight_pb2.DiscoverNodesRequest()
                response = stub.DiscoverNodes(request)
                self.logger.info(f"Discovered nodes: {response.magnet_links}")
                return response.magnet_links
        except grpc.RpcError as e:
            self.logger.error(f"gRPC error when discovering nodes: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error when discovering nodes: {str(e)}")
            raise

# Initialize an instance of NodeDirectory
node_directory = NodeDirectory()


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
    def __init__(self, consensus, secret_key, node_directory, vm):
        self.globalMetrics = {
                'totalTransactions': 0,
                'totalBlocks': 0,
                # Add more metrics as needed
            }
        self.initial_reward = 50.0
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

        # Event listeners
        self.new_block_listeners = []
        self.new_transaction_listeners = []

        # Genesis wallet address
        self.genesis_wallet_address = "genesis_wallet"

        # Ensure NativeCoinContract is deployed or retrieve existing
        self.initialize_native_coin_contract()

        self.create_genesis_block()
        self.balances = {}  # Initialize balances as a dictionary
        self.wallets = []
        self.transactions = []
        self.contracts = []
        self.tokens: Dict[str, Token] = {}
        self.liquidity_pool_manager = LiquidityPoolManager()
        self.zk_system = SecureHybridZKStark(security_level=2)  # Adjust security level as needed
        self.p2p_node = None  # Will be initialized later
        self.start_time = time.time()
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


    async def set_p2p_node(self, p2p_node):
        from P2PNode import P2PNode  # Move the import here

        self.p2p_node = p2p_node

    async def initialize_p2p(self, host, port):
        logger.debug(f"Initializing P2P node on {host}:{port}")
        self.p2p_node = P2PNode(host, port, self)
        try:
            await asyncio.wait_for(self.p2p_node.start(), timeout=10)
            logger.debug("P2P node started successfully")
        except asyncio.TimeoutError:
            logger.error("Timeout while starting P2P node")
            raise



    def get_blocks_since(self, last_known_block_index):
        """
        Returns the blocks added since the given block index.
        """
        return self.chain[last_known_block_index + 1:]

    async def mine_block(self, miner_address):
        # Implementation of block mining
        # This is a placeholder, implement according to your needs
        block = QuantumBlock(
            previous_hash=self.chain[-1].hash if self.chain else "0",
            data="Mined block",
            quantum_signature=self.generate_quantum_signature(),
            reward=self.get_block_reward(),
            transactions=self.pending_transactions[:10],
            timestamp=time.time()
        )
        
        # Generate ZK proof for the block
        block_data = f"{block.previous_hash}{block.data}{block.quantum_signature}{block.reward}"
        public_input = self.zk_system.hash(block_data)
        block.zk_proof = self.zk_system.prove(int(block.timestamp), public_input)

        if self.consensus.validate_block(block):
            self.chain.append(block)
            return block.reward
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
        peers = self.node_directory.discover_nodes()

        async def send_block(peer):
            async with grpc.aio.insecure_channel(f"{peer['ip_address']}:{peer['port']}") as channel:
                stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                grpc_block = dagknight_pb2.Block(
                    previous_hash=block.previous_hash,
                    data=block.data,
                    quantum_signature=block.quantum_signature,
                    reward=float(block.reward),
                    transactions=[
                        dagknight_pb2.Transaction(
                            sender=t.sender,
                            receiver=t.receiver,
                            amount=float(t.amount),
                            private_key=t.private_key,
                            public_key=t.public_key,
                            signature=t.signature
                        ) for t in block.transactions
                    ],
                    hash=block.hash,
                    timestamp=int(block.timestamp),  # Ensure this is an integer
                    nonce=int(block.nonce)  # Ensure this is an integer
                )
                try:
                    response = await stub.PropagateBlock(dagknight_pb2.PropagateBlockRequest(block=grpc_block, miner_address=block.miner_address))
                    if response.success:
                        logger.info(f"Block successfully propagated to {peer['node_id']}")
                    else:
                        logger.error(f"Failed to propagate block to {peer['node_id']}. Message: {response.message}")
                except grpc.RpcError as e:
                    logger.error(f"gRPC error: {e}")

        await asyncio.gather(*[send_block(peer) for peer in peers])

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
        logger.info(f"Validating block: {block.to_dict()}")

        if len(self.chain) == 0:
            return block.previous_hash == "0" and block.hash == block.compute_hash()

        if block.previous_hash != self.chain[-1].hash:
            logger.warning(f"Invalid previous hash. Expected {self.chain[-1].hash}, got {block.previous_hash}")
            return False

        if block.hash != block.compute_hash():
            logger.warning(f"Invalid block hash. Computed {block.compute_hash()}, got {block.hash}")
            return False

        current_time = int(time.time())
        if block.timestamp > current_time + 300:
            logger.warning(f"Block timestamp too far in the future. Current time: {current_time}, Block time: {block.timestamp}")
            return False

        logger.info("Block validation successful")
        return True
    def add_block(self, block):
        current_time = time.time()
        if block.timestamp > current_time + 300:  # Allow for a 5-minute future timestamp window
            logger.warning(f"Block timestamp too far in the future. Current time: {current_time}, Block time: {block.timestamp}")
            return False

        if not self.validate_block(block):
            logger.warning("Block validation failed. Block not added.")
            return False

        self.chain.append(block)
        self.blocks_since_last_adjustment += 1

        if self.blocks_since_last_adjustment >= self.adjustment_interval:
            self.adjust_difficulty()
            self.blocks_since_last_adjustment = 0

        # Asynchronously propagate the block to the P2P network
        asyncio.create_task(self.p2p_node.propagate_block(block))

        for listener in self.new_block_listeners:
            listener(block)
        return True


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

    async def propagate_block(self, node_url, block_data, retries=5):
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(node_url, json=block_data) as response:
                        if response.status == 200:
                            logger.info(f"Block successfully propagated to {node_url}")
                            return True
                        else:
                            logger.warning(f"Failed to propagate block to {node_url}. Status: {response.status}")
            except Exception as e:
                logger.error(f"Error propagating block to {node_url}: {str(e)}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

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

    def get_transactions(self, address):
        transactions = []
        for block in self.chain:
            for tx in block.transactions:
                if tx['sender'] == address or tx['receiver'] == address:
                    transactions.append(tx)
        return transactions

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
        logger.info(f"Validating block: {block.to_dict()}")

        if len(self.blockchain.chain) == 0:
            return block.previous_hash == "0" and block.hash == block.compute_hash()
        if block.previous_hash != self.blockchain.chain[-1].hash:
            logger.warning(f"Invalid previous hash. Expected {self.blockchain.chain[-1].hash}, got {block.previous_hash}")
            return False

        if block.hash != block.compute_hash():
            logger.warning(f"Invalid block hash. Computed {block.compute_hash()}, got {block.hash}")
            return False
        if not self.blockchain.batch_verify_transactions(block.transactions):
            logger.warning("Batch transaction verification failed")
            return False

        current_time = int(time.time())
        if block.timestamp > current_time + 300:
            logger.warning(f"Block timestamp too far in the future. Current time: {current_time}, Block time: {block.timestamp}")
            return False

        logger.info("Block validation successful")
        return True

    def is_valid_hash(self, block_hash):
        logger.info(f"Validating hash against target. Current difficulty: {self.blockchain.difficulty}, Target: {self.blockchain.target}")
        valid = int(block_hash, 16) < self.blockchain.target
        logger.debug(f"Is valid hash: {valid}")
        return valid

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
node_directory = NodeDirectory()  # Initialize the node directory

# Create the VM instance
vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])  # Initialize SimpleVM with necessary arguments

# Create the consensus object
consensus = PBFTConsensus(nodes=[], node_id="node_id_here")  # Initialize PBFTConsensus with necessary arguments
print(f"VM object: {vm}")

# Create the blockchain instance with the VM and consensus
blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)

# Update the consensus to point to the correct blockchain
consensus.blockchain = blockchain

# FastAPI initialization
app = FastAPI(
    title="Plata Network",
    description="A Quantum Blockchain Network",
    version="0.1.0"
)

# Router initialization
router = APIRouter()



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("node")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

async def initialize_components():
    from P2PNode import P2PNode  # Move the import here

    global node_directory, blockchain, exchange, vm, plata_contract, price_feed, genesis_address, bot

    try:
        node_directory = EnhancedNodeDirectory()
        vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
        
        if vm is None:
            raise ValueError("Failed to initialize SimpleVM")

        consensus = PBFTConsensus(nodes=[], node_id="node_1")
        secret_key = "your_secret_key_here"
        print(f"VM object: {vm}")

        blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)
        
        if blockchain is None:
            raise ValueError("Failed to initialize QuantumBlockchain")

        price_oracle = PriceOracle()
        logger.info(f"Instantiating EnhancedExchange: {EnhancedExchange}")

        exchange = EnhancedExchangeWithZKStarks(blockchain, vm, price_oracle, node_directory, desired_security_level=1, host="localhost", port=8765)
        await exchange.start_p2p()
        await blockchain.set_p2p_node(exchange.p2p_node)
        logger.info(f"EnhancedExchange instance created: {exchange}")

        exchange.order_book = EnhancedOrderBook()



        plata_contract = PlataContract(vm)
        price_feed = PriceFeed()  # Assuming you have a PriceFeed class

        genesis_address = "your_genesis_address_here"  # Replace with actual genesis address generation/retrieval

        bot = MarketMakerBot(exchange, "BTC_PLATA", Decimal('0.01'))
        native_coin_contract_address, native_coin_contract = vm.get_existing_contract(NativeCoinContract)
        security_level = 20  # Replace with the appropriate level for your system

        # Initialize zk_system with the security_level
        zk_system = SecureHybridZKStark(security_level)

        if native_coin_contract is None:
            max_supply = 21000000  # Example max supply
            native_coin_contract_address = vm.deploy_contract(genesis_address, NativeCoinContract, max_supply, zk_system)
            native_coin_contract = vm.contracts[native_coin_contract_address]

        blockchain.native_coin_contract_address = native_coin_contract_address
        blockchain.native_coin_contract = native_coin_contract

        print("All components initialized successfully")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
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
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict
    return None


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
exchange = Exchange(blockchain, vm, price_oracle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    wallet = Wallet()  # Or however the wallet is initialized
    wallet.address = wallet.get_address()
    wallet.public_key = wallet.get_public_key()

    # Startup
    await initialize_components()
    asyncio.create_task(update_prices_periodically())
    asyncio.create_task(periodic_mining(genesis_address, wallet))
    asyncio.create_task(deploy_and_run_market_bot(exchange, vm, plata_contract, price_feed))
    yield


    # Clean up resources, close database connections, etc.
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
fake_users_db = {
    "user1": {"username": "user1", "full_name": "User One", "email": "user1@example.com", "hashed_password": "fakehashedsecret1", "disabled": False},
    "user2": {"username": "user2", "full_name": "User Two", "email": "user2@example.com", "hashed_password": "fakehashedsecret2", "disabled": True},
}
@app.on_event("startup")
async def on_startup():
    wallet = Wallet()  # Or however the wallet is initialized
    wallet.address = wallet.get_address()
    wallet.public_key = wallet.get_public_key()

    # Startup
    asyncio.create_task(update_prices_periodically())
    asyncio.create_task(periodic_mining(genesis_address, wallet))
    asyncio.create_task(deploy_and_run_market_bot(exchange, vm, plata_contract, price_feed))
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
        "exp": datetime.utcnow() + timedelta(hours=1),  # Token expiration time
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
        pincode: str = payload.get("sub")
        if pincode is None:
            raise HTTPException(status_code=401, detail="Invalid token")
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

@app.post("/register", response_model=Token)
def register(user: User):
    if user.pincode in fake_users_db:
        raise HTTPException(status_code=400, detail="Pincode already registered")
    salt = generate_salt()
    hashed_pincode = get_password_hash(user.pincode + salt)
    wallet = {"address": "0x123456", "private_key": "abcd1234"}
    alias = generate_unique_alias()  # Generate a unique alias for the user
    user_in_db = UserInDB(pincode=user.pincode, hashed_pincode=hashed_pincode, salt=salt, wallet=wallet, alias=alias)
    fake_users_db[user.pincode] = user_in_db
    access_token = create_access_token(data={"sub": user.pincode})
    return {"access_token": access_token, "token_type": "bearer", "wallet": wallet}

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



def periodically_discover_nodes(directory_ip, directory_port):
    while True:
        try:
            node_directory.discover_nodes_with_grpc(directory_ip, directory_port)
        except Exception as e:
            logger.error(f"Error during periodic node discovery: {str(e)}")
        time.sleep(60)  # Discover nodes every 60 seconds





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
    transaction: dagknight_pb2.Transaction

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
class Wallet(BaseModel):
    private_key: Optional[ec.EllipticCurvePrivateKey] = None
    public_key: Optional[str] = None
    mnemonic: Optional[Mnemonic] = None
    address: Optional[str] = None

    def __init__(self, private_key=None, mnemonic=None, **data):
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
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        return base64.b64encode(encrypted_message).decode('utf-8')

    def decrypt_message(self, encrypted_message):
        rsa_private_key = serialization.load_pem_private_key(self.private_key_pem().encode(), password=None, backend=default_backend())
        decrypted_message = rsa_private_key.decrypt(
            base64.b64decode(encrypted_message),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )
        return decrypted_message.decode('utf-8')

    def get_address(self) -> str:
        public_key_bytes = self.get_public_key().encode()
        address = "plata" + hashlib.sha256(public_key_bytes).hexdigest()
        return address

    def generate_unique_alias(self):
        alias_length = 8  # Length of the alias
        alias = ''.join(random.choices(string.ascii_lowercase + string.digits, k=alias_length))
        return alias

    class Config:
        arbitrary_types_allowed = True



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


class ImportWalletRequest(BaseModel):
    mnemonic: str
@app.post("/import_wallet")
def import_wallet(request: ImportWalletRequest, pincode: str = Depends(get_current_user)):
    try:
        mnemonic = Mnemonic("english")
        seed = mnemonic.to_seed(request.mnemonic)

        private_key = ec.derive_private_key(
            int.from_bytes(seed[:32], byteorder="big"),
            ec.SECP256R1(),
            default_backend()
        )
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        public_key = private_key.public_key()
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        address = "plata" + hashlib.sha256(public_key_pem.encode()).hexdigest()
        alias = generate_unique_alias()

        wallet = Wallet(address=address, private_key=private_key_pem, public_key=public_key_pem, alias=alias)
        blockchain.add_wallet(wallet)
        return {"address": address, "private_key": private_key_pem, "public_key": public_key_pem, "alias": alias}
    except Exception as e:
        logger.error(f"Error importing wallet: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")





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
@app.post("/create_wallet")
async def create_wallet(pincode: str = Depends(get_current_user)):
    try:
        # Generate mnemonic phrase
        mnemo = Mnemonic("english")
        mnemonic_phrase = mnemo.generate(strength=256)

        # Derive seed from mnemonic phrase
        seed = mnemo.to_seed(mnemonic_phrase)

        # Generate private key from seed
        private_key = ec.derive_private_key(
            int.from_bytes(seed[:32], byteorder='big'),
            ec.SECP256R1(), default_backend()
        )
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        public_key = private_key.public_key()
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        address = "plata" + hashlib.sha256(public_key_pem.encode()).hexdigest()
        alias = generate_unique_alias()

        # Create Wallet object with all necessary attributes
        wallet = Wallet(address=address, private_key=private_key_pem, public_key=public_key_pem, alias=alias, mnemonic=mnemonic_phrase)
        blockchain.add_wallet(wallet)
        return {"address": address, "private_key": private_key_pem, "public_key": public_key_pem, "alias": alias, "mnemonic": mnemonic_phrase}
    except Exception as e:
        logger.error(f"Error creating wallet: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


class AddressRequest(BaseModel):
    address: str
class WalletRequest(BaseModel):
    wallet_address: str
class BalanceResponse(BaseModel):
    balance: float  # Change to float for serialization
    transactions: List[Transaction] = Field([], description="List of transactions")
    
@app.post("/get_balance", response_model=BalanceResponse)
async def get_balance(request: WalletRequest):
    wallet_address = request.wallet_address
    try:
        # Ensure the wallet address is valid
        if not wallet_address or not re.match(r'^plata[a-f0-9]{64}$', wallet_address):
            logger.error(f"Invalid wallet address format: {wallet_address}")
            raise HTTPException(status_code=422, detail="Invalid wallet address format")

        # Retrieve the balance for the given wallet address
        balance = await blockchain.get_balance(wallet_address)
        # Ensure balance is a Decimal with 18 decimal places
        balance = Decimal(balance).quantize(Decimal('0.000000000000000001'))
        
        # Convert balance to float for JSON serialization
        balance_float = float(balance)

        # Retrieve the transaction history for the wallet
        transactions = [
            Transaction(
                tx_hash=tx.get('hash', 'unknown'),
                sender=tx.get('sender', 'unknown'),
                receiver=tx.get('receiver', 'unknown'),
                amount=float(Decimal(tx.get('amount', 0)).quantize(Decimal('0.000000000000000001'))),  # Convert to float
                timestamp=tx.get('timestamp', 'unknown')
            )
            for tx in await blockchain.get_transactions(wallet_address)
        ]
        
        # Generate a ZKP for the balance
        secret = int(balance * 10**18)  # Convert Decimal to integer
        public_input = int(hashlib.sha256(wallet_address.encode()).hexdigest(), 16)
        zk_proof = blockchain.zk_system.prove(secret, public_input)

        logger.info(f"Balance and ZKP retrieved for address {wallet_address}: {balance_float}")
        return BalanceResponse(balance=balance_float, transactions=transactions, zk_proof=zk_proof)
    except KeyError as e:
        logger.error(f"Missing key in transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: missing key {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/send_transaction")
async def send_transaction(transaction: Transaction, pincode: str = Depends(get_current_user)):
    try:
        logger.info(f"Received transaction request: {transaction}")

        # Resolve receiver address from alias if necessary
        if transaction.receiver.startswith("plata"):
            receiver_address = transaction.receiver
        else:
            receiver_alias = transaction.receiver
            receiver_address = None
            for wallet in blockchain.wallets:
                if wallet.alias == receiver_alias:
                    receiver_address = wallet.address
                    break
            if not receiver_address:
                raise HTTPException(status_code=400, detail="Invalid receiver alias")

        transaction.receiver = receiver_address

        # Create wallet instance using the private key provided in the transaction
        wallet = Wallet(private_key=transaction.private_key)
        logger.info("Created wallet from private key.")

        # Create the message to sign
        message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"

        # Sign the transaction
        transaction.signature = wallet.sign_message(message)
        logger.info(f"Transaction signed with signature: {transaction.signature}")

        # Add public key to the transaction
        transaction.public_key = wallet.get_public_key()
        logger.info(f"Public key added to transaction: {transaction.public_key}")

        # Generate and sign the transaction using Zero-Knowledge Proofs (ZKP)
        transaction.sign_transaction(blockchain.zk_system)

        # Add the transaction to the blockchain
        if blockchain.add_transaction(transaction):
            logger.info(f"Transaction from {transaction.sender} to {transaction.receiver} added to blockchain.")

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
    except InvalidSignature as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/deploy_contract")
def deploy_contract(contract: Contract, pincode: str = Depends(get_current_user)):
    blockchain.add_contract(contract)
    return {"success": True, "contract_address": contract.address}



@app.post("/send_batch_transactions")
def send_batch_transactions(transactions: List[Transaction], pincode: str = Depends(authenticate)):
    results = []
    try:
        for transaction in transactions:
            wallet = Wallet(private_key=transaction.private_key)
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
            transaction.signature = wallet.sign_message(message)

            if blockchain.add_transaction(transaction):
                results.append({"success": True, "message": "Transaction added successfully.", "transaction": transaction.dict()})
            else:
                results.append({"success": False, "message": "Transaction failed to add: insufficient balance", "transaction": transaction.dict()})
    except ValueError as e:
        logger.error(f"Error deserializing private key: {e}")
        raise HTTPException(status_code=400, detail="Invalid private key format")
    except InvalidSignature as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return results
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
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
import numpy as np
import networkx as nx
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.optimize import minimize
from fastapi import Depends, HTTPException, FastAPI

continue_mining = True

def mining_algorithm(iterations=1):
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        logging.info(f"Memory usage at start: {memory_info.rss / (1024 * 1024):.2f} MB")

        logging.info("Initializing Quantum Annealing Simulation")
        global num_qubits, graph
        num_qubits = 10  # Reduced number of qubits
        graph = nx.grid_graph(dim=[2, 5])  # 2x5 grid instead of 5x5
        logging.info(f"Initialized graph with {len(graph.nodes)} nodes and {len(graph.edges())} edges")
        logging.info(f"Graph nodes: {list(graph.nodes())}")
        logging.info(f"Graph edges: {list(graph.edges())}")


        def quantum_annealing_simulation(params):
            hamiltonian = sparse.csr_matrix((2**num_qubits, 2**num_qubits), dtype=complex)
            
            # Get the dimensions of the grid graph
            grid_dims = list(graph.nodes())[-1]
            rows, cols = grid_dims[0] + 1, grid_dims[1] + 1
            
            for edge in graph.edges():
                i, j = edge
                sigma_z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
                
                # Convert tuple indices to integers
                i_int = i[0] * cols + i[1]
                j_int = j[0] * cols + j[1]
                
                term_i = sparse.kron(sparse.eye(2**i_int, dtype=complex), sigma_z)
                term_i = sparse.kron(term_i, sparse.eye(2**(num_qubits-i_int-1), dtype=complex))
                hamiltonian += term_i
                
                term_j = sparse.kron(sparse.eye(2**j_int, dtype=complex), sigma_z)
                term_j = sparse.kron(term_j, sparse.eye(2**(num_qubits-j_int-1), dtype=complex))
                hamiltonian += term_j

            problem_hamiltonian = sparse.diags(np.random.randn(2**num_qubits), dtype=complex)
            hamiltonian += params[0] * problem_hamiltonian

            initial_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
            evolution = sparse.linalg.expm(-1j * hamiltonian.tocsc() * params[1])
            final_state = evolution @ initial_state

            return -np.abs(final_state[0])**2  # Negative because we're minimizing



        cumulative_counts = {}
        for iteration in range(iterations):
            logging.info(f"Starting iteration {iteration + 1}/{iterations}")
            logging.info("Running Quantum Annealing Simulation")
            start_time_simulation = time.time()
            random_params = [random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)]
            logging.info(f"Random parameters: {random_params}")
            
            result = minimize(quantum_annealing_simulation, random_params, method='Nelder-Mead')
            end_time_simulation = time.time()

            simulation_duration = end_time_simulation - start_time_simulation
            logging.info(f"Simulation completed in {simulation_duration:.2f} seconds")
            logging.info(f"Optimization result: {result}")



            logging.info("Simulating gravity effects")
            mass_distribution = np.random.rand(2, 5)  # 2x5 grid
            gravity_factor = np.sum(mass_distribution) / 10

            logging.info("Creating quantum-inspired black hole")
            black_hole_position = np.unravel_index(np.argmax(mass_distribution), mass_distribution.shape)
            black_hole_strength = mass_distribution[black_hole_position]

            logging.info("Measuring entanglement")
            entanglement_matrix = np.abs(np.outer(result.x, result.x))
            logging.info(f"Entanglement Matrix: {entanglement_matrix}")

            logging.info("Extracting Hawking radiation analogue")
            hawking_radiation = np.random.exponential(scale=black_hole_strength, size=10)

            logging.info("Calculating final quantum state")
            final_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
            counts = {format(i, f'0{num_qubits}b'): abs(val)**2 for i, val in enumerate(final_state) if abs(val)**2 > 1e-6}

            for state, prob in counts.items():
                if state in cumulative_counts:
                    cumulative_counts[state] += prob
                else:
                    cumulative_counts[state] = prob

        memory_info = process.memory_info()
        logging.info(f"Memory usage after simulation: {memory_info.rss / (1024 * 1024):.2f} MB")

        # Realistic Calculation of QHINs and Hashrate
        qhins = np.trace(entanglement_matrix)  # Quantum Hash Information Number based on the trace of the entanglement matrix
        hashrate = 1 / (simulation_duration * iterations)  # Hashrate as the inverse of the total simulation duration

        logging.info(f"QHINs: {qhins:.6f}")
        logging.info(f"Hashrate: {hashrate:.6f} hashes/second")

        return cumulative_counts, result.fun, entanglement_matrix, qhins, hashrate
    except Exception as e:
        logging.error(f"Error in mining_algorithm: {str(e)}")
        logging.error(traceback.format_exc())  # Log the traceback
        return {"success": False, "message": f"Error in mining_algorithm: {str(e)}"}
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
async def create_block(self, miner_address: str):
    # ... (existing block creation logic)
    for tx in block.transactions:
        if not tx.verify_transaction(self.zk_system):
            raise ValueError(f"Invalid transaction in block: {tx}")

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
@app.post("/mine_block")
async def mine_block(request: MineBlockRequest, pincode: str = Depends(authenticate)):
    global continue_mining
    node_id = request.node_id
    wallet_address = request.wallet_address
    node_ip = request.node_ip
    node_port = request.node_port
    wallet = request.wallet

    try:
        logger.info(f"Received mining request: {request.dict()}")
    except Exception as e:
        logger.error(f"Error parsing request: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid request format")

    try:
        logger.info(f"Starting mining process for node {node_id} with wallet {wallet_address} at {node_ip}:{node_port}")
        iteration_count = 0
        max_iterations = 20

        while continue_mining and iteration_count < max_iterations:
            start_time = time.time()

            # Process exchange orders
            await process_exchange_orders()

            # Perform quantum annealing simulation
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(mining_algorithm, iterations=2)
                try:
                    result = future.result(timeout=300)
                    logger.info("Mining algorithm completed within timeout")
                except TimeoutError:
                    logger.error("Mining algorithm timed out after 5 minutes")
                    continue

            if isinstance(result, dict) and not result.get("success", True):
                logger.error(f"Mining algorithm failed: {result.get('message')}")
                continue

            counts, energy, entanglement_matrix, qhins, hashrate = result["counts"], result["energy"], result["entanglement_matrix"], result["qhins"], result["hashrate"]
            end_time = time.time()

            logger.info(f"Quantum Annealing Simulation completed in {end_time - start_time:.2f} seconds")
            logger.info("Checking mining conditions")

            max_state = max(counts, key=counts.get)
            max_prob = counts[max_state]

            top_n = 10
            sorted_probs = sorted(counts.values(), reverse=True)
            cumulative_prob = sum(sorted_probs[:top_n])

            if cumulative_prob > 0.01:
                logger.info(f"Mining condition met with cumulative probability of top {top_n} states: {cumulative_prob:.6f}, generating quantum signature")
                quantum_signature = blockchain.generate_quantum_signature()

                logger.info("Adding block to blockchain")

                # Create transaction objects with ZKP integration
                transactions = [
                    Transaction(
                        sender='sender_address',
                        receiver='receiver_address',
                        amount=Decimal(10),
                        price=Decimal(100),
                        buyer_id='buyer_id_value',
                        seller_id='seller_id_value',
                        wallet=wallet
                    )
                ]

                # Sign and verify each transaction using ZKP
                zk_system = SecureHybridZKStark()
                for tx in transactions:
                    tx.sign_transaction(zk_system)
                    if not tx.verify_transaction(zk_system):
                        logger.error(f"Transaction verification failed for transaction {tx.id}")
                        raise HTTPException(status_code=400, detail=f"Invalid transaction: {tx.id}")

                new_block = QuantumBlock(
                    previous_hash=blockchain.chain[-1].hash,
                    data="Some data to be included in the block",
                    quantum_signature=quantum_signature,
                    reward=blockchain.get_block_reward(),
                    transactions=transactions,
                    timestamp=time.time()
                )
                new_block.mine_block(blockchain.difficulty)

                if blockchain.consensus.validate_block(new_block):
                    blockchain.chain.append(new_block)
                    blockchain.process_transactions(new_block.transactions)

                    try:
                        blockchain.native_coin_contract.mint(wallet_address, Decimal(new_block.reward))
                        logger.info(f"Reward of {new_block.reward} QuantumDAGKnight Coins added to wallet {wallet_address}")
                    except Exception as e:
                        logger.error(f"Error minting reward: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"Error minting reward: {str(e)}")

                    blockchain.adjust_difficulty()

                    logger.info(f"Node {node_id} mined a block and earned {new_block.reward} QuantumDAGKnight Coins")
                    
                    # Use P2P node to propagate the new block
                    await blockchain.p2p_node.propagate_block(new_block)

                    hash_rate = 1 / (end_time - start_time)
                    logger.info(f"Mining Successful. Hash Rate: {hash_rate:.2f} hashes/second")
                    logger.info(f"Mining Time: {end_time - start_time:.2f} seconds")
                    logger.info(f"Quantum State Probabilities: {counts}")
                    logger.info(f"Entanglement Matrix: {entanglement_matrix.tolist()}")
                    logger.info(f"Quantum Hash Information Number: {qhins:.6f}")

                    updated_balance = blockchain.get_balance(wallet_address)
                    
                    # Use P2P node to broadcast balance update
                    balance_update_message = Message(
                        type=MessageType.BALANCE_UPDATE.value,
                        payload={
                            "wallet_address": wallet_address,
                            "balance": float(updated_balance)
                        }
                    )
                    await blockchain.p2p_node.broadcast(balance_update_message)

                    return {
                        "success": True,
                        "message": f"Block mined successfully. Reward: {new_block.reward} QuantumDAGKnight Coins",
                        "hash_rate": hash_rate,
                        "mining_time": end_time - start_time,
                        "quantum_state_probabilities": counts,
                        "entanglement_matrix": entanglement_matrix.tolist(),
                        "qhins": qhins,
                        "hashrate": hashrate
                    }
                else:
                    logger.error("Failed to create a new block")
            else:
                logger.warning(f"Mining failed. Condition not met. Highest probability state: {max_state} with probability: {max_prob:.6f}")
                iteration_count += 1
                if iteration_count >= max_iterations:
                    logger.error(f"Maximum number of iterations ({max_iterations}) reached. Stopping mining.")
                    break
                continue

        return {"success": False, "message": "Mining failed after maximum iterations."}

    except Exception as e:
        logger.error(f"Error during mining: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during mining: {str(e)}")
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





async def propagate_block(node_url, block_data, retries=5):
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(node_url, json=block_data) as response:
                    if response.status == 200:
                        logger.info(f"Block successfully propagated to {node_url}")
                        return True
                    else:
                        logger.warning(f"Failed to propagate block to {node_url}. Status: {response.status}")
        except Exception as e:
            logger.error(f"Error propagating block to {node_url}: {str(e)}")
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return False

async def propagate_block_to_all_peers(block_data):
    nodes = node_directory.discover_nodes()
    tasks = [propagate_block(f"http://{node['ip_address']}:{node['port']}/receive_block", block_data) for node in nodes]
    results = await asyncio.gather(*tasks)
    successful_propagations = sum(results)
    logger.info(f"Successfully propagated block to {successful_propagations}/{len(nodes)} peers")




async def gossip_protocol(block_data):
    nodes = node_directory.discover_nodes()
    random.shuffle(nodes)
    gossip_nodes = nodes[:3]  # Randomly select 3 nodes to start gossiping
    tasks = [propagate_block(f"http://{node['ip_address']}:{node['port']}/receive_block", block_data) for node in gossip_nodes]
    await asyncio.gather(*tasks)


# Example usage
import asyncio
import grpc
import dagknight_pb2
import dagknight_pb2_grpc
from grpc.experimental import aio

async def propagate_block_to_single_peer(node, data, quantum_signature, transactions, miner_address):
    try:
        async with aio.insecure_channel(f"{node['ip_address']}:{node['port']}") as channel:
            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
            block = dagknight_pb2.Block(
                previous_hash=blockchain.chain[-1].hash,
                data=data,
                quantum_signature=quantum_signature,
                reward=blockchain.current_reward(),
                transactions=[dagknight_pb2.Transaction(
                    sender=tx['sender'], receiver=tx['receiver'], amount=tx['amount']) for tx in transactions]
            )
            request = dagknight_pb2.PropagateBlockRequest(block=block, miner_address=miner_address)
            response = await stub.PropagateBlock(request)
            if response.success:
                logger.info(f"Node {node['node_id']} received block with hash: {block.hash}")
                return True
            else:
                logger.error(f"Node {node['node_id']} failed to receive the block.")
                return False
    except Exception as e:
        logger.error(f"Error propagating block to node {node['node_id']}: {str(e)}")
        return False

async def propagate_block_to_peers(data, quantum_signature, transactions, miner_address, max_retries=3):
    nodes = node_directory.discover_nodes()
    logger.info(f"Propagating block to {len(nodes)} nodes")

    async def propagate_with_retry(node):
        for attempt in range(max_retries):
            success = await propagate_block_to_single_peer(node, data, quantum_signature, transactions, miner_address)
            if success:
                return True
            logger.warning(f"Retry {attempt + 1}/{max_retries} for node {node['node_id']}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

    tasks = [propagate_with_retry(node) for node in nodes]
    results = await asyncio.gather(*tasks)

    successful_propagations = sum(results)
    logger.info(f"Successfully propagated block to {successful_propagations}/{len(nodes)} peers")
    return successful_propagations, len(nodes)

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

async def run_node():
    # Initialize components
    secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")
    node_directory = NodeDirectory()
    vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
    consensus = PBFTConsensus(nodes=[], node_id="node_1")
    blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)
    zk_system = SecureHybridZKStark(security_level=20)

    # Start gRPC server
    server, servicer = await serve(secret_key, node_directory, vm, blockchain, zk_system)

    # Connect to peers
    peers = await connect_to_peers(node_directory)

    # Start periodic sync
    sync_task = asyncio.create_task(periodic_sync(blockchain, peers))

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down node...")
        sync_task.cancel()
        await server.stop(0)

# Example usage
if not pbft.committed:
    view_change.initiate_view_change()

class SecureDAGKnightServicer(dagknight_pb2_grpc.DAGKnightServicer):
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

def serve_directory_service(node_directory):
    class DirectoryServicer(dagknight_pb2_grpc.DAGKnightServicer):
        def RegisterNode(self, request, context):
            node_id = request.node_id
            public_key = request.public_key
            ip_address = request.ip_address
            port = request.port
            try:
                magnet_link = node_directory.register_node(node_id, public_key, ip_address, port)
                return dagknight_pb2.RegisterNodeResponse(success=True, magnet_link=magnet_link)
            except Exception as e:
                context.abort(grpc.StatusCode.INTERNAL, f'Error registering node: {str(e)}')

        def DiscoverNodes(self, request, context):
            try:
                nodes = node_directory.discover_nodes()
                return dagknight_pb2.DiscoverNodesResponse(magnet_links=[node['magnet_link'] for node in nodes])
            except AttributeError as e:
                logger.error(f"Error in DiscoverNodes: {str(e)}")
                logger.error(traceback.format_exc())
                context.abort(grpc.StatusCode.INTERNAL, f'Error discovering nodes: {str(e)}')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dagknight_pb2_grpc.add_DAGKnightServicer_to_server(DirectoryServicer(), server)
    directory_port = int(os.getenv("DIRECTORY_PORT", 50501))
    server.add_insecure_port(f'[::]:{directory_port}')
    server.start()
    logger.info(f"Directory service started on port {directory_port}")
    server.wait_for_termination()
async def serve(secret_key, node_directory, vm, blockchain, zk_system):
    print("Entering serve function")
    try:
        print("Creating gRPC server")
        server = aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024)
            ]
        )
        print("gRPC server created")

        print("Initializing SecureDAGKnightServicer")
        servicer = SecureDAGKnightServicer(secret_key, node_directory, vm, zk_system)
        print("Adding servicer to server")
        dagknight_pb2_grpc.add_DAGKnightServicer_to_server(servicer, server)
        print("Servicer added to server")

        print("Enabling server reflection")
        service_names = (
            dagknight_pb2.DESCRIPTOR.services_by_name['DAGKnight'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        print("Server reflection enabled")

        grpc_port = int(os.getenv("GRPC_PORT", 50502))
        server_address = f'[::]:{grpc_port}'
        print(f"Adding insecure port {server_address}")
        server.add_insecure_port(server_address)
        print(f"Insecure port {server_address} added")

        print("Starting server")
        await server.start()
        print(f"gRPC server started on {server_address}")

        print("Waiting for server termination")
        await server.wait_for_termination()
    except Exception as e:
        print(f"Exception in serve function: {e}")
        print(f"Exception traceback: {traceback.format_exc()}")
    finally:
        if 'server' in locals():
            await server.stop(0)
            print("gRPC server shutdown complete")





def register_node_with_grpc(node_id, public_key, ip_address, port, directory_ip, directory_port):
    with grpc.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
        stub = dagknight_pb2_grpc.DAGKnightStub(channel)
        request = dagknight_pb2.RegisterNodeRequest(node_id=node_id, public_key=public_key, ip_address=ip_address, port=port)
        response = stub.RegisterNode(request)
        logger.info(f"Registered node with magnet link: {response.magnet_link}")


def discover_nodes_with_grpc(directory_ip, directory_port):
    try:
        with grpc.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
            request = dagknight_pb2.DiscoverNodesRequest()
            response = stub.DiscoverNodes(request)
            logger.info(f"Discovered nodes: {response.magnet_links}")
            return response.magnet_links
    except grpc.RpcError as e:
        logger.error(f"gRPC error when discovering nodes: {e.code()}: {e.details()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error when discovering nodes: {str(e)}")
        raise


def periodically_discover_nodes(directory_ip, directory_port, retry_interval=60, max_retries=5):
    retries = 0
    while True:
        try:
            logger.info(f"Attempting to discover nodes (Attempt {retries + 1})...")
            discover_nodes_with_grpc(directory_ip, directory_port)
            retries = 0  # Reset retries after a successful attempt
        except Exception as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"Max retries reached. Failed to discover nodes: {str(e)}")
                break
            logger.warning(f"Error discovering nodes: {str(e)}. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        else:
            logger.info(f"Node discovery successful. Next discovery in {retry_interval} seconds...")
            time.sleep(retry_interval)

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
        total_nodes = len(node_directory.discover_nodes())
        total_transactions = blockchain.globalMetrics['totalTransactions']  # Accessing totalTransactions from the globalMetrics dictionary
        total_blocks = len(blockchain.chain)
        average_block_time = calculate_average_block_time(blockchain.chain)
        current_difficulty = blockchain.difficulty
        total_supply = blockchain.get_total_supply()
        
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
@app.post("/refresh_token")
async def refresh_token(refresh_token: str):
    user = verify_refresh_token(refresh_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    new_access_token = create_access_token({"sub": user})
    return {"access_token": new_access_token}


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

WS_1000_NORMAL_CLOSURE = 1000
WS_1008_POLICY_VIOLATION = 1008
WS_1011_INTERNAL_ERROR = 1011
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(None)):
    logger.info("WebSocket connection attempt received")
    logger.info(f"Query parameters: {websocket.query_params}")

    if not token:
        logger.error("No token provided in WebSocket connection")
        await websocket.close(code=1008)
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
                await manager.broadcast(json.dumps(state))
        except WebSocketDisconnect:
            manager.disconnect(user)
            logger.info(f"WebSocket disconnected for user: {user}")
    except HTTPException as e:
        logger.error(f"Authentication failed: {str(e)}")
        await websocket.close(code=1008)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {str(e)}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011)


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
        
node_directory = NodeDirectory()
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

blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)

# Assign the genesis wallet address to sender_address
sender_address = blockchain.genesis_wallet_address


cashewstable_contract = vm.deploy_contract(sender_address, Cashewstable, collateral_token, price_feed)
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
        
@app.on_event("startup")
async def startup_event():
    await startup()

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

class EnhancedSecureDAGKnightServicer(dagknight_pb2_grpc.DAGKnightServicer):
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

# Helper function to create a test node
def create_test_node(node_id, grpc_port, api_port):
    node_directory = EnhancedNodeDirectory()
    price_oracle = PriceOracle()
    vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
    consensus = PBFTConsensus(nodes=[], node_id=f"node_{node_id}")
    blockchain = QuantumBlockchain(consensus, "test_secret_key", node_directory, vm)
    exchange = EnhancedExchange(blockchain, vm, price_oracle, node_directory)
    exchange.order_book = EnhancedOrderBook()



    # Create gRPC server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dagknight_servicer = EnhancedSecureDAGKnightServicer("test_secret_key", node_directory, exchange)
    dagknight_pb2_grpc.add_DAGKnightServicer_to_server(dagknight_servicer, grpc_server)
    grpc_server.add_insecure_port(f'[::]:{grpc_port}')
    grpc_server.start()
    
    # Create FastAPI app
    fastapi_app = TestClient(app)
    
    node_directory.set_node_status(f"node_{node_id}", 'active')
    
    return {
        'node_id': f"node_{node_id}",
        'grpc_port': grpc_port,
        'api_port': api_port,
        'exchange': exchange,
        'grpc_server': grpc_server,
        'fastapi_app': fastapi_app,
        'node_directory': node_directory,
        'fastapi_app': app,  # Make sure this is the FastAPI app instance

    }

# Create test nodes
@pytest.fixture(scope="module")
def test_nodes():
    nodes = []
    for i in range(NUM_NODES):
        node = create_test_node(i, BASE_GRPC_PORT + i, BASE_API_PORT + i)
        nodes.append(node)
    yield nodes
    # Cleanup
    for node in nodes:
        node['grpc_server'].stop(0)

# Test node failure and recovery
@pytest.mark.asyncio
async def test_node_failure_and_recovery(test_nodes):
    # Place initial order
    order = {
        'user_id': 'test_user',
        'type': 'limit',  # Order type
        'order_type': 'buy',
        'pair': 'BTC_USD',  # Currency pair
        'base_currency': 'BTC',
        'quote_currency': 'USD',
        'amount': Decimal('1.0'),
        'price': Decimal('50000'),
        'from_currency': 'USD',  # Add this line
        'to_currency': 'BTC'  # Add this line
    }
    response = test_nodes[0]['fastapi_app'].post('/place_order', json=order)
    logger.info(f"Initial order placement response: {response.status_code} - {response.text}")
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}. Response: {response.text}"
    
    # Simulate node failure
    failed_node = random.choice(test_nodes)
    failed_node['node_directory'].set_node_status(failed_node['node_id'], 'inactive')
    logger.info(f"Node {failed_node['node_id']} has failed")
    
    # Place another order
    order['amount'] = Decimal('0.5')
    response = test_nodes[0]['fastapi_app'].post('/place_order', json=order)
    logger.info(f"Second order placement response: {response.status_code} - {response.text}")
    assert response.status_code == 200
    
    # Allow time for propagation
    await asyncio.sleep(2)
    
    # Check that the order is not in the failed node
    failed_node_orders = failed_node['exchange'].order_book.get_orders()
    assert len(failed_node_orders) == 1, f"Expected 1 order, but got {len(failed_node_orders)}"
    
    # Simulate node recovery
    failed_node['node_directory'].set_node_status(failed_node['node_id'], 'active')
    logger.info(f"Node {failed_node['node_id']} has recovered")
    
    # Trigger state synchronization
    await failed_node['exchange'].sync_state()
    
    # Verify that the recovered node has all orders
    recovered_node_orders = failed_node['exchange'].order_book.get_orders()
    assert len(recovered_node_orders) == 2, f"Expected 2 orders, but got {len(recovered_node_orders)}"
    
    # Place a new order to verify the recovered node is fully functional
    order['amount'] = Decimal('0.25')
    response = failed_node['fastapi_app'].post('/place_order', json=order)
    logger.info(f"Third order placement response: {response.status_code} - {response.text}")
    assert response.status_code == 200
    
    # Verify the new order is propagated to all nodes
    await asyncio.sleep(1)
    for node in test_nodes:
        node_orders = node['exchange'].order_book.get_orders()
        assert len(node_orders) == 3, f"Expected 3 orders for node {node['node_id']}, but got {len(node_orders)}"

# Benchmark order placement and propagation with detailed metrics
@pytest.mark.asyncio
async def test_order_placement_benchmark_with_metrics(test_nodes):
    num_orders = 100
    start_time = time.time()
    
    # Place orders concurrently
    async def place_order(node):
        order = {
            'user_id': f'bench_user_{node["node_id"]}',
            'type': 'limit',
            'order_type': 'buy',
            'pair': 'BTC_USD',
            'base_currency': 'BTC',
            'quote_currency': 'USD',
            'amount': Decimal('0.1'),
            'price': Decimal('50000'),
            'from_currency': 'USD',
            'to_currency': 'BTC'
        }
        try:
            await node['exchange'].place_order(order)
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
    
    tasks = [place_order(node) for node in test_nodes for _ in range(num_orders // NUM_NODES)]
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    orders_per_second = num_orders / total_time
    
    # Calculate propagation time metrics
    if propagation_times:
        avg_propagation_time = sum(propagation_times) / len(propagation_times)
        max_propagation_time = max(propagation_times)
        min_propagation_time = min(propagation_times)
    else:
        logger.warning("No propagation times recorded")
        avg_propagation_time = max_propagation_time = min_propagation_time = 0
    
    logger.info(f"Benchmark results:")
    logger.info(f"Total orders placed: {num_orders}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Orders per second: {orders_per_second:.2f}")
    logger.info(f"Average propagation time: {avg_propagation_time:.4f} seconds")
    logger.info(f"Max propagation time: {max_propagation_time:.4f} seconds")
    logger.info(f"Min propagation time: {min_propagation_time:.4f} seconds")
    
    # Verify order propagation
    await asyncio.sleep(2)  # Allow time for propagation
    for node in test_nodes:
        order_book = node['exchange'].order_book.get_orders()
        assert len(order_book) == num_orders, f"Expected {num_orders} orders, but got {len(order_book)} for node {node['node_id']}"
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



def start_grpc_server(secret_key, node_directory, vm, blockchain, zk_system):
    print("Entering start_grpc_server function")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        print("About to run serve function")
        loop.run_until_complete(serve(secret_key, node_directory, vm, blockchain, zk_system))
    except Exception as e:
        print(f"Exception in start_grpc_server: {e}")
        print(f"Exception traceback: {traceback.format_exc()}")
    finally:
        print("Closing event loop")
        loop.close()
async def run_uvicorn_server():
    config = uvicorn.Config("quantumdagknight:app", host="0.0.0.0", port=50503, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
import socket
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

async def async_main():
    from curses_dashboard.dashboard import DashboardUI
    try:
        # Initialize the VM and create the genesis wallet
        vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
        logger.info("SimpleVM initialized successfully")

        mnemonic, genesis_address = create_genesis_wallet(vm)
        print(f"Genesis Mnemonic: {mnemonic}")
        print(f"Genesis Address: {genesis_address}")

        collateral_token = "USDC"
        initial_price = 0.000000000000000001  # Adjusted price
        price_feed = SimplePriceFeed(initial_price)
        cashewstable_contract_address = vm.deploy_contract(genesis_address, Cashewstable, collateral_token, price_feed)
        print(f"Cashewstable contract deployed at address: {cashewstable_contract_address}")

        node_id = os.getenv("NODE_ID", "node_1")
        public_key = "public_key_example"
        ip_address = os.getenv("IP_ADDRESS", "127.0.0.1")
        grpc_port = find_free_port()  # Dynamically find a free port for gRPC
        api_port = int(os.getenv("API_PORT", 50503))
        directory_ip = os.getenv("DIRECTORY_IP", "127.0.0.1")
        directory_port = int(os.getenv("DIRECTORY_PORT", 50501))
        secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")

        consensus = PBFTConsensus(nodes=[], node_id=node_id)
        node_directory = NodeDirectory()
        blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)

        price_oracle = PriceOracle()
        exchange = EnhancedExchange(blockchain, vm, price_oracle, node_directory)
        exchange.order_book = EnhancedOrderBook()

        # Initialize the P2PNode with a unique port
        free_port = find_free_port()  # Use this function to assign a free port
        p2p_node = P2PNode(ip_address, free_port, blockchain)

        # Start the directory service in a separate thread
        directory_service_thread = threading.Thread(target=serve_directory_service, args=(node_directory,))
        directory_service_thread.start()

        zk_system = SecureHybridZKStark(security_level=2)

        grpc_server_thread = threading.Thread(
            target=start_grpc_server,
            args=(secret_key, node_directory, vm, blockchain, zk_system)
        )
        grpc_server_thread.start()
        await asyncio.sleep(5)

        discovery_thread = threading.Thread(target=periodically_discover_nodes, args=(directory_ip, directory_port))
        discovery_thread.start()

        asyncio.create_task(periodic_network_stats_update())

        blockchain.on_new_block(manager.broadcast_new_block)
        blockchain.on_new_transaction(manager.broadcast_new_transaction)

        def curses_main(stdscr):
            dashboard_ui = DashboardUI(stdscr, blockchain, exchange, p2p_node)
            asyncio.run(dashboard_ui.run())

        uvicorn_task = asyncio.create_task(run_uvicorn_server())
        websocket_task = asyncio.create_task(p2p_node.start())

        await asyncio.gather(
            uvicorn_task,
            websocket_task,
            asyncio.to_thread(curses.wrapper, curses_main)
        )

    except Exception as e:
        logger.error(f"Error in async_main function: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(async_main())
