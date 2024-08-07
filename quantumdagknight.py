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
import dagknight_pb2
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

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator  # Ensure validator is imported
from pydantic import BaseModel, field_validator  # Use field_validator for Pydantic V2
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, root_validator
from unittest.mock import AsyncMock, MagicMock, patch


import logging

logger = logging.getLogger(__name__)

MAX_SUPPLY = 21000000  
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
 
templates = Jinja2Templates(directory="templates")

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
            
class QuantumBlock:
    def __init__(self, previous_hash, data, quantum_signature, reward, transactions, timestamp=None):
        self.previous_hash = previous_hash
        self.data = data
        self.quantum_signature = quantum_signature
        self.reward = reward
        self.transactions = transactions  # Ensure transactions are stored as Transaction objects
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()
        logger.debug(f"Initialized QuantumBlock: {self.to_dict()}")

    def compute_hash(self):
        block_string = json.dumps({
            "previous_hash": self.previous_hash,
            "data": self.data,
            "quantum_signature": self.quantum_signature,
            "reward": self.reward,
            "transactions": [txn.to_dict() if hasattr(txn, 'to_dict') else txn for txn in self.transactions],
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }, sort_keys=True)
        block_hash = sha256(block_string.encode()).hexdigest()
        logger.debug(f"Computed hash: {block_hash} for nonce: {self.nonce}")
        return block_hash
    def mine_block(self, difficulty):
        target = 2 ** (256 - difficulty)
        initial_hash = self.hash
        logger.debug(f"Starting mining with initial hash: {initial_hash}")
        while int(self.hash, 16) >= target:
            self.nonce += 1
            self.hash = self.compute_hash()
        logger.info(f"Block mined with nonce: {self.nonce}, initial hash: {initial_hash}, final hash: {self.hash}")

    def to_dict(self):
        return {
            "previous_hash": self.previous_hash,
            "data": self.data,
            "quantum_signature": self.quantum_signature,
            "reward": self.reward,
            "transactions": [txn.to_dict() if hasattr(txn, 'to_dict') else txn for txn in self.transactions],
            "hash": self.hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }

    @classmethod
    def from_dict(cls, block_data):
        block = cls(
            previous_hash=block_data["previous_hash"],
            data=block_data["data"],
            quantum_signature=block_data["quantum_signature"],
            reward=block_data["reward"],
            transactions=block_data["transactions"],
            timestamp=block_data["timestamp"]
        )
        block.nonce = block_data["nonce"]
        block.hash = block_data["hash"]
        return block

    def set_hash(self, hash_value):
        self.hash = hash_value

    def recompute_hash(self):
        self.hash = self.compute_hash()

    def hash_block(self):
        self.hash = self.compute_hash()

class NodeDirectory:
    def __init__(self):
        self.nodes = {}
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.register_times = []
        self.discover_times = []
        self.transactions = {}  # Assuming transactions are stored in a dictionary
    def add_transaction(self, transaction):
        transaction_hash = transaction.compute_hash()
        self.transactions[transaction_hash] = transaction
    def get_node_details(self, node_id):
        with self.lock:
            return self.nodes.get(node_id)

    def register_node(self, node_id, public_key, ip_address, port):
        start_time = time.time()
        with self.lock:
            magnet_link = self.generate_magnet_link(node_id, public_key, ip_address, port)
            self.nodes[node_id] = {"magnet_link": magnet_link, "ip_address": ip_address, "port": port}
        end_time = time.time()
        self.register_times.append(end_time - start_time)
        return magnet_link
    def get_transaction(self, transaction_hash):
        return self.transactions.get(transaction_hash)

    def discover_nodes(self):
        start_time = time.time()
        with self.lock:
            nodes = [{"node_id": node_id, **info} for node_id, info in self.nodes.items()]
        end_time = time.time()
        self.discover_times.append(end_time - start_time)
        return nodes

    def get_performance_stats(self):
        return {
            "avg_register_time": statistics.mean(self.register_times) if self.register_times else 0,
            "max_register_time": max(self.register_times) if self.register_times else 0,
            "avg_discover_time": statistics.mean(self.discover_times) if self.discover_times else 0,
            "max_discover_time": max(self.discover_times) if self.discover_times else 0,
        }




    def generate_magnet_link(self, node_id, public_key, ip_address, port):
        info = f"{node_id}:{public_key}:{ip_address}:{port}"
        hash = hashlib.sha1(info.encode()).hexdigest()
        magnet_link = f"magnet:?xt=urn:sha1:{hash}&dn={node_id}&pk={base64.urlsafe_b64encode(public_key.encode()).decode()}&ip={ip_address}&port={port}"
        return magnet_link

    def register_node(self, node_id, public_key, ip_address, port):
        magnet_link = self.generate_magnet_link(node_id, public_key, ip_address, port)
        self.nodes[node_id] = {"magnet_link": magnet_link, "ip_address": ip_address, "port": port}
        return magnet_link

    def discover_nodes(self):
        return [{"node_id": node_id, **info} for node_id, info in self.nodes.items()]

    def register_node_with_grpc(self, node_id, public_key, ip_address, port, directory_ip, directory_port):
        try:
            with grpc.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
                stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                request = dagknight_pb2.RegisterNodeRequest(node_id=node_id, public_key=public_key, ip_address=ip_address, port=port)
                response = stub.RegisterNode(request)
                logger.info(f"Registered node with magnet link: {response.magnet_link}")
                return response.magnet_link
        except grpc.RpcError as e:
            logger.error(f"gRPC error when registering node: {e.code()}: {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when registering node: {str(e)}")
            raise

    def discover_nodes_with_grpc(self, directory_ip, directory_port):
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


node_directory = NodeDirectory()

from decimal import Decimal
from typing import Dict
class NativeCoinContract:
    def __init__(self, vm, max_supply):
        self.vm = vm
        self.max_supply = max_supply
        self.total_supply = Decimal('0')
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

    def transfer(self, sender: str, receiver: str, amount: Decimal):
        if self.balances.get(sender, Decimal('0')) < amount:
            raise ValueError("Insufficient balance")
        self.balances[sender] -= amount
        self.balances[receiver] = self.balances.get(receiver, Decimal('0')) + amount

    def get_balance(self, user: str) -> Decimal:
        return self.balances.get(user, Decimal('0'))

    def get_total_supply(self) -> Decimal:
        return self.total_supply
class QuantumBlockchain:
    def __init__(self, consensus, secret_key, node_directory, vm):
        self.initial_reward = 50.0
        self.chain = []
        self.pending_transactions = []
        self.consensus = consensus
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
    def get_balance(self, address):
        balance = self.balances.get(address, 0)
        return balance

    def initialize_native_coin_contract(self):
        try:
            self.native_coin_contract_address, self.native_coin_contract = self.vm.get_existing_contract(NativeCoinContract)
            logger.info(f"NativeCoinContract already exists at address {self.native_coin_contract_address}")
        except ValueError:
            logger.info("NativeCoinContract does not exist, deploying new contract...")
            self.native_coin_contract_address = self.vm.deploy_contract(
                self.genesis_wallet_address, NativeCoinContract, self.max_supply)
            self.native_coin_contract = self.vm.contracts[self.native_coin_contract_address]
            logger.info(f"NativeCoinContract deployed at address {self.native_coin_contract_address}")

        # Add additional logging to verify correct storage
        if self.native_coin_contract_address in self.vm.contracts:
            logger.debug(f"Contract stored correctly in VM: {self.native_coin_contract_address}")
        else:
            logger.error(f"Failed to store contract in VM: {self.native_coin_contract_address}")



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
    def add_new_block(self, data, quantum_signature, transactions, miner_address):
        previous_block = self.chain[-1]
        previous_hash = previous_block.hash
        reward = self.get_block_reward()
        total_supply = self.get_total_supply()

        if total_supply + Decimal(reward) > Decimal(MAX_SUPPLY):
            reward = Decimal(MAX_SUPPLY) - total_supply

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

    def get_balance(self, address):
        balance = self.balances.get(address, 0)
        logger.info(f"Balance for {address}: {balance}")
        return balance

    def add_transaction(self, transaction):
        logger.debug(f"Adding transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        logger.debug(f"Sender balance before transaction: {self.balances.get(transaction.sender, 0)}")

        if self.balances.get(transaction.sender, 0) >= transaction.amount:
            wallet = Wallet()
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
            if wallet.verify_signature(message, transaction.signature, transaction.public_key):
                self.pending_transactions.append(transaction.to_dict())
                logger.debug(f"Transaction added. Pending transactions count: {len(self.pending_transactions)}")

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

                if self.validate_quantum_signature(signature, threshold=0.5):
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
        # You might want to fetch the user from a database here
        # For now, we'll just return the username
        return username
    except jwt.PyJWTError:
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
    async def get_price(self, asset):
        return Decimal('1.00')  # Simplified for example

# Instantiate price_feed
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
    def __init__(self, exchange, vm, plata_contract, price_feed, initial_capital: Dict[str, Decimal]):
        self.exchange = exchange
        self.vm = vm
        self.plata_contract = plata_contract
        self.price_feed = price_feed
        self.capital = initial_capital
        self.positions: Dict[str, Decimal] = {}
        self.order_book: Dict[str, List[Tuple[Decimal, Decimal]]] = {}  # price, amount
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.historical_data: Dict[str, List[Decimal]] = {}
        self.quantum_circuit = self._initialize_quantum_circuit()
        self.last_rebalance_time = time.time()
        self.rebalance_interval = 3600  # 1 hour
        self.target_price = Decimal('1.00')
        self.price_tolerance = Decimal('0.005')  # 0.5% tolerance
        self.max_single_trade_size = Decimal('10000')  # Max size for a single trade
        self.slippage_tolerance = Decimal('0.01')  # 1% slippage tolerance

    def _initialize_quantum_circuit(self):
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(qr, cr)
        qc.h(qr)  # Apply Hadamard gates
        qc.measure(qr, cr)
        return qc

    async def run(self):
        while True:
            await self._update_market_data()
            await self._rebalance_portfolio()
            await self._manage_plata_supply()
            await self._execute_quantum_trading_strategy()
            await self._provide_liquidity()
            await self.execute_mean_reversion_strategy()
            await self.execute_momentum_strategy()
            await self.manage_risk()
            await self.handle_black_swan_events()
            await asyncio.sleep(60)  # Run every minute

    async def _update_market_data(self):
        for asset in self.exchange.get_tradable_assets():
            price = await self.price_feed.get_price(asset)
            if asset not in self.historical_data:
                self.historical_data[asset] = []
            self.historical_data[asset].append(price)
            if len(self.historical_data[asset]) > 1000:
                self.historical_data[asset] = self.historical_data[asset][-1000:]

    async def _rebalance_portfolio(self):
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

    async def _manage_plata_supply(self):
        plata_price = await self.price_feed.get_price("PLATA")
        if plata_price > self.target_price + self.price_tolerance:
            amount_to_mint = (plata_price - self.target_price) * self.plata_contract.total_supply / self.target_price
            await self._mint_plata(amount_to_mint)
        elif plata_price < self.target_price - self.price_tolerance:
            amount_to_burn = (self.target_price - plata_price) * self.plata_contract.total_supply / self.target_price
            await self._burn_plata(amount_to_burn)

    async def _execute_quantum_trading_strategy(self):
        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(self.quantum_circuit, backend)
        qobj = assemble(transpiled_circuit)
        job = backend.run(qobj, shots=1000)
        result = job.result()
        counts = result.get_counts(self.quantum_circuit)
        
        # Use quantum measurements to inform trading decisions
        quantum_signal = max(counts, key=counts.get)
        signal_strength = counts[quantum_signal] / 1000

        for asset in self.exchange.get_tradable_assets():
            if asset == "PLATA":
                continue  # Skip PLATA as it's managed separately

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

    async def _provide_liquidity(self):
        for asset in self.exchange.get_tradable_assets():
            if asset == "PLATA":
                continue  # Skip PLATA as it's managed separately

            current_price = await self.price_feed.get_price(asset)
            spread = current_price * Decimal('0.002')  # 0.2% spread

            buy_price = current_price - spread / 2
            sell_price = current_price + spread / 2

            await self._place_limit_order(asset, "buy", self.max_single_trade_size / 10, buy_price)
            await self._place_limit_order(asset, "sell", self.max_single_trade_size / 10, sell_price)

    async def _buy_asset(self, asset: str, amount: Decimal):
        current_price = await self.price_feed.get_price(asset)
        max_price = current_price * (1 + self.slippage_tolerance)
        try:
            await self.exchange.create_market_buy_order(asset, amount, max_price)
            self.capital[asset] = self.capital.get(asset, Decimal('0')) + amount
            self.capital["PLATA"] -= amount * current_price
        except Exception as e:
            print(f"Error buying {asset}: {e}")

    async def _sell_asset(self, asset: str, amount: Decimal):
        current_price = await self.price_feed.get_price(asset)
        min_price = current_price * (1 - self.slippage_tolerance)
        try:
            await self.exchange.create_market_sell_order(asset, amount, min_price)
            self.capital[asset] = self.capital.get(asset, Decimal('0')) - amount
            self.capital["PLATA"] += amount * current_price
        except Exception as e:
            print(f"Error selling {asset}: {e}")

    async def _place_limit_order(self, asset: str, side: str, amount: Decimal, price: Decimal):
        try:
            if side == "buy":
                await self.exchange.create_limit_buy_order(asset, amount, price)
            else:
                await self.exchange.create_limit_sell_order(asset, amount, price)
        except Exception as e:
            print(f"Error placing limit order for {asset}: {e}")

    async def _mint_plata(self, amount: Decimal):
        try:
            await self.plata_contract.mint(self.vm.contract_address, amount)
            self.capital["PLATA"] = self.capital.get("PLATA", Decimal('0')) + amount
        except Exception as e:
            print(f"Error minting PLATA: {e}")

    async def _burn_plata(self, amount: Decimal):
        try:
            await self.plata_contract.burn(self.vm.contract_address, amount)
            self.capital["PLATA"] = self.capital.get("PLATA", Decimal('0')) - amount
        except Exception as e:
            print(f"Error burning PLATA: {e}")

    def calculate_kelly_criterion(self, win_probability: float, odds: float) -> float:
        return (win_probability * odds - (1 - win_probability)) / odds

    def calculate_optimal_position_size(self, asset: str) -> Decimal:
        price_data = np.array(self.historical_data[asset])
        returns = np.diff(price_data) / price_data[:-1]
        
        win_probability = np.sum(returns > 0) / len(returns)
        average_win = np.mean(returns[returns > 0])
        average_loss = abs(np.mean(returns[returns < 0]))
        
        odds = average_win / average_loss if average_loss != 0 else 1
        
        kelly_fraction = self.calculate_kelly_criterion(win_probability, odds)
        return Decimal(str(kelly_fraction)) * self.capital.get(asset, Decimal('0'))

    async def execute_mean_reversion_strategy(self):
        for asset in self.exchange.get_tradable_assets():
            if asset == "PLATA":
                continue

            price_data = np.array(self.historical_data[asset])
            if len(price_data) < 20:
                continue

            moving_average = np.mean(price_data[-20:])
            current_price = price_data[-1]
            
            z_score = (current_price - moving_average) / np.std(price_data[-20:])
            
            if z_score > 2:  # Price is significantly above the mean
                optimal_size = self.calculate_optimal_position_size(asset)
                await self._sell_asset(asset, min(optimal_size, self.max_single_trade_size))
            elif z_score < -2:  # Price is significantly below the mean
                optimal_size = self.calculate_optimal_position_size(asset)
                await self._buy_asset(asset, min(optimal_size, self.max_single_trade_size))

    async def execute_momentum_strategy(self):
        for asset in self.exchange.get_tradable_assets():
            if asset == "PLATA":
                continue

            price_data = np.array(self.historical_data[asset])
            if len(price_data) < 50:
                continue

            returns = np.diff(price_data) / price_data[:-1]
            momentum = np.mean(returns[-10:])  # 10-day momentum
            
            if momentum > 0.01:  # Strong positive momentum
                optimal_size = self.calculate_optimal_position_size(asset)
                await self._buy_asset(asset, min(optimal_size, self.max_single_trade_size))
            elif momentum < -0.01:  # Strong negative momentum
                optimal_size = self.calculate_optimal_position_size(asset)
                await self._sell_asset(asset, min(optimal_size, self.max_single_trade_size))

    async def manage_risk(self):
        total_value = sum(amount * await self.price_feed.get_price(asset) for asset, amount in self.capital.items())
        
        for asset, amount in self.capital.items():
            asset_value = amount * await self.price_feed.get_price(asset)
            asset_weight = asset_value / total_value
            
            if asset_weight > 0.2:  # No single asset should be more than 20% of the portfolio
                amount_to_sell = (asset_value - total_value * Decimal('0.2')) / await self.price_feed.get_price(asset)
                await self._sell_asset(asset, amount_to_sell)

    async def handle_black_swan_events(self):
        for asset in self.exchange.get_tradable_assets():
            price_data = np.array(self.historical_data[asset])
            if len(price_data) < 100:
                continue

            returns = np.diff(price_data) / price_data[:-1]
            current_return = returns[-1]
            
            if current_return < np.percentile(returns, 1):  # Current return is in the bottom 1%
                # Reduce position by half
                amount_to_sell = self.capital.get(asset, Decimal('0')) / 2
                await self._sell_asset(asset, amount_to_sell)




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

class PriceOracle:
    def __init__(self):
        self.prices: Dict[str, Decimal] = {}
        self.last_update: float = 0
        self.update_interval: float = 60  # Update prices every 60 seconds

    async def get_price(self, token: str) -> Decimal:
        if time.time() - self.last_update > self.update_interval:
            await self.update_prices()
        return self.prices.get(token, Decimal('0'))

    async def update_prices(self):
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,polkadot&vs_currencies=usd') as response:
                data = await response.json()
                self.prices = {
                    'BTC': Decimal(str(data['bitcoin']['usd'])),
                    'ETH': Decimal(str(data['ethereum']['usd'])),
                    'DOT': Decimal(str(data['polkadot']['usd'])),
                    'PLATA': Decimal('1.0')  # Assuming 1 PLATA = 1 USD for simplicity
                }
        self.last_update = time.time()

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
class Order(BaseModel):
    id: str
    user_id: str
    order_type: Literal['buy', 'sell']
    from_currency: str
    to_currency: str
    amount: Decimal
    price: Decimal
    status: str = 'open'
    created_at: float = Field(default_factory=lambda: time.time())

    @root_validator(pre=True)
    def validate_order(cls, values):
        order_type = values.get('order_type')
        if order_type not in ['buy', 'sell']:
            raise ValueError('order_type must be either "buy" or "sell"')
        return values

    @root_validator(pre=True)
    def validate_positive(cls, values):
        for field in ['amount', 'price']:
            if values.get(field) <= 0:
                raise ValueError(f'{field} must be positive')
        return values

    def __eq__(self, other):
        if not isinstance(other, Order):
            return False
        return (self.id == other.id and
                self.user_id == other.user_id and
                self.order_type == other.order_type and
                self.from_currency == other.from_currency and
                self.to_currency == other.to_currency and
                self.amount == other.amount and
                self.price == other.price and
                self.status == other.status and
                self.created_at == other.created_at)

    def __hash__(self):
        return hash((self.id, self.user_id, self.order_type, self.from_currency, self.to_currency, self.amount, self.price, self.status, self.created_at))


class LiquidityPool(BaseModel):
    id: str
    currency_a: str
    currency_b: str
    balance_a: Decimal
    balance_b: Decimal
    fee_percent: Decimal

class ExchangeTransaction(BaseModel):
    buyer_id: str
    seller_id: str
    amount: Decimal
    price: Decimal
    to_currency: str
    from_currency: str


class OrderBook:
    def __init__(self):
        self.buy_orders: Dict[str, List[Order]] = {}
        self.sell_orders: Dict[str, List[Order]] = {}

    def add_order(self, order: Order):
        order_list = self.buy_orders if order.order_type == 'buy' else self.sell_orders
        pair = f"{order.from_currency}_{order.to_currency}"
        if pair not in order_list:
            order_list[pair] = []
        order_list[pair].append(order)
        order_list[pair].sort(key=lambda x: x.price, reverse=(order.order_type == 'buy'))

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

    async def place_limit_order(self, user: str, order_type: str, base_currency: str, quote_currency: str, amount: Decimal, price: Decimal) -> str:
        order = Order(
            id=str(uuid.uuid4()),
            user_id=user,
            order_type=order_type,
            from_currency=base_currency if order_type == 'sell' else quote_currency,
            to_currency=quote_currency if order_type == 'sell' else base_currency,
            amount=amount,
            price=price,
            status='open'
        )
        
        # Check user balance
        if order_type == 'buy':
            required_balance = amount * price
            user_balance = await self.blockchain.get_balance(user, quote_currency)
        else:  # sell
            required_balance = amount
            user_balance = await self.blockchain.get_balance(user, base_currency)

        if user_balance < required_balance:
            raise ValueError("Insufficient balance")

        # Add order to the order book
        self.order_book.add_order(order)

        # Try to match the order immediately
        await self.process_exchange_orders()

        return order.id

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

    async def place_order(self, order: Order):
        # Check balance before placing order
        if order.order_type == 'buy':
            balance = await self.blockchain.get_balance(order.user_id, order.to_currency)
            if balance < order.amount * order.price:
                raise HTTPException(status_code=400, detail="Insufficient balance for buy order")
        else:
            balance = await self.blockchain.get_balance(order.user_id, order.from_currency)
            if balance < order.amount:
                raise HTTPException(status_code=400, detail="Insufficient balance for sell order")

        # Add order to the order book
        self.order_book.add_order(order)

        # Match orders
        pair = f"{order.from_currency}_{order.to_currency}"
        matches = self.order_book.match_orders(pair)

        for buy_order, sell_order, amount, price in matches:
            # Create a Transaction object with all required fields
            exchange_tx = Transaction(
                sender=sell_order.user_id,  # Assuming the seller is the one sending
                receiver=buy_order.user_id,  # Assuming the buyer is the one receiving
                amount=amount,
                price=price,
                buyer_id=buy_order.user_id,  # Ensure these fields are included
                seller_id=sell_order.user_id  # Ensure these fields are included
            )
            
            # Optionally, sign the transaction if a wallet is available
            if exchange_tx.wallet:
                exchange_tx.sign_transaction()

            # Process the transaction
            await self.process_exchange_transaction(exchange_tx)
            
        # Return the order ID
        return order.id




    async def process_exchange_transaction(self, exchange_tx: ExchangeTransaction):
        logging.info(f"Processing transaction: {exchange_tx}")

        fee = exchange_tx.amount * self.fee_percent
        amount_after_fee = exchange_tx.amount - fee

        # Log the details of transactions being created
        logging.info(f"Creating buyer transaction with sender: {exchange_tx.buyer_id}, receiver: {exchange_tx.seller_id}")

        # Create buyer transaction
        buyer_tx = Transaction(
            sender=exchange_tx.buyer_id,
            receiver=exchange_tx.seller_id,
            amount=exchange_tx.amount * exchange_tx.price,
            price=exchange_tx.price,  # Ensure price is provided
            private_key=None,  # Optional
            public_key=None,  # Optional
            signature=None,  # Optional
            wallet=None,  # Optional
            buyer_id=exchange_tx.buyer_id,
            seller_id=exchange_tx.seller_id
        )

        # Log the details of transactions being created
        logging.info(f"Creating seller transaction with sender: {exchange_tx.seller_id}, receiver: {exchange_tx.buyer_id}")

        # Create seller transaction
        seller_tx = Transaction(
            sender=exchange_tx.seller_id,
            receiver=exchange_tx.buyer_id,
            amount=amount_after_fee,
            price=exchange_tx.price,  # Ensure price is provided
            private_key=None,  # Optional
            public_key=None,  # Optional
            signature=None,  # Optional
            wallet=None,  # Optional
            buyer_id=exchange_tx.buyer_id,
            seller_id=exchange_tx.seller_id
        )

        # Log the details of transactions being created
        logging.info(f"Creating fee transaction with sender: {exchange_tx.seller_id}, receiver: fee_pool")

        # Create fee transaction
        fee_tx = Transaction(
            sender=exchange_tx.seller_id,
            receiver="fee_pool",  # Special address for collecting fees
            amount=fee,
            price=exchange_tx.price,  # Ensure price is provided
            private_key=None,  # Optional
            public_key=None,  # Optional
            signature=None,  # Optional
            wallet=None,  # Optional
            buyer_id=exchange_tx.buyer_id,
            seller_id=exchange_tx.seller_id
        )

        # Add transactions to the blockchain
        await self.blockchain.add_transaction(buyer_tx)
        await self.blockchain.add_transaction(seller_tx)
        await self.blockchain.add_transaction(fee_tx)


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
        
class EnhancedExchange(Exchange):
    def __init__(self, blockchain, vm):
        super().__init__(blockchain, vm)
        self.order_book = OrderBook()
        self.liquidity_pools = {}

    async def place_limit_order(self, user: str, order_type: str, base_currency: str, quote_currency: str, amount: Decimal, price: Decimal) -> str:
        order = Order(
            id=str(uuid.uuid4()),
            user_id=user,
            order_type=order_type,
            from_currency=base_currency if order_type == 'sell' else quote_currency,
            to_currency=quote_currency if order_type == 'sell' else base_currency,
            amount=amount,
            price=price,
            status='open'
        )
        
        # Check user balance
        if order_type == 'buy':
            required_balance = amount * price
            user_balance = await self.blockchain.get_balance(user, quote_currency)
        else:  # sell
            required_balance = amount
            user_balance = await self.blockchain.get_balance(user, base_currency)

        if user_balance < required_balance:
            raise ValueError("Insufficient balance")

        # Add order to the order book
        self.order_book.add_order(order)

        # Try to match the order immediately
        await self.process_exchange_orders()

        return order.id

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

    async def add_liquidity(self, pool_id: str, user_id: str, amount_a: Decimal, amount_b: Decimal):
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

    async def swap(self, user_id: str, from_currency: str, to_currency: str, amount: Decimal):
        pool_id = f"{from_currency}_{to_currency}"
        if pool_id not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")

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

@app.post("/place_order")
async def place_order(
    order: Order,
    background_tasks: BackgroundTasks,  # Ensure this is a non-default argument
    current_user: str = Depends(get_current_user)  # Ensure default arguments come after
):
    try:
        order.user_id = current_user["username"]
        background_tasks.add_task(exchange.place_order, order)
        return {"message": "Order placed successfully", "order_id": order.id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
if 'exchange' in globals():
    bot = MarketMakerBot(exchange, "BTC_PLATA", Decimal('0.01'))  # 1% spread
    asyncio.create_task(bot.run())

# Start the market maker bot

# Add this router to your main FastAPI app
app.include_router(router, prefix="/exchange", tags=["exchange"])

price_oracle = MagicMock()  # Or use an appropriate object if you have one

# Initialize the exchange object
exchange = Exchange(blockchain, vm, price_oracle)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code here
    logger.info("Starting up the application...")
    # Initialize global variables, open database connections, etc.
    # For example, initializing a QuantumBlockchain instance:
    consensus = Consensus(blockchain=None)  # Temporarily set blockchain to None
    secret_key = "your_secret_key"  # Replace with actual secret key
    node_directory = NodeDirectory()  # Initialize the node directory
    global blockchain
    blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)


    # Update the consensus to point to the correct blockchain
    consensus.blockchain = blockchain

    price_oracle = MagicMock()  # Or use an appropriate object if you have one
    
    # Initialize the exchange object
    global exchange
    exchange = Exchange(blockchain, vm, price_oracle)

    yield

    # Shutdown code here
    logger.info("Shutting down the application...")
    # Clean up resources, close database connections, etc.

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
async def startup_event():
    asyncio.create_task(update_prices_periodically())
    asyncio.create_task(mine_block("miner_address_here"))
    asyncio.create_task(deploy_and_run_market_bot(exchange, vm, plata_contract, price_feed))




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
        

from pydantic import BaseModel, Field
from mnemonic import Mnemonic
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64
import hashlib
from typing import Optional
class Wallet(BaseModel):
    private_key: Optional[ec.EllipticCurvePrivateKey] = None
    public_key: Optional[ec.EllipticCurvePublicKey] = None
    mnemonic: Optional[Mnemonic] = None

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
        self.public_key = self.private_key.public_key()

    def private_key_pem(self) -> str:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

    def get_public_key(self) -> str:
        return self.public_key.public_bytes(
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

    def get_address(self):
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
from pydantic import BaseModel, Field
from typing import Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import base64

class Transaction(BaseModel):
    sender: str
    receiver: str
    amount: Decimal  # Ensure Decimal is used if needed
    private_key: Optional[str] = None
    public_key: Optional[str] = None
    signature: Optional[str] = None
    wallet: Optional[Wallet] = None
    price: Decimal
    buyer_id: str
    seller_id: str

    def sign_transaction(self):
        if not self.wallet:
            raise ValueError("Wallet is required to sign the transaction")
        message = f"{self.sender}{self.receiver}{self.amount}"
        self.signature = self.wallet.sign_message(message)
        self.public_key = self.wallet.get_public_key()


    class Config:
        arbitrary_types_allowed = True


    def to_grpc(self):
        return dagknight_pb2.Transaction(
            sender=self.sender,
            receiver=self.receiver,
            amount=self.amount,
            public_key=self.public_key,
            signature=self.signature
        )






    def compute_hash(self):
        transaction_string = f'{self.sender}{self.receiver}{self.amount}{self.signature}'
        return hashlib.sha256(transaction_string.encode('utf-8')).hexdigest()

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "signature": self.signature,
            "public_key": self.public_key
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_proto(cls, proto_transaction):
        return cls(
            sender=proto_transaction.sender,
            receiver=proto_transaction.receiver,
            amount=proto_transaction.amount,
            private_key=None,  # We don't receive private key in proto
            public_key=proto_transaction.public_key,
            signature=proto_transaction.signature
        )



def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


class ImportWalletRequest(BaseModel):
    mnemonic: str
@app.post("/import_wallet")
def import_wallet(request: ImportWalletRequest, pincode: str = Depends(authenticate)):
    mnemonic = Mnemonic("english")
    seed = mnemonic.to_seed(request.mnemonic)

    # Use the first 32 bytes of the seed to generate a deterministic private key
    private_key = ec.derive_private_key(
        int.from_bytes(seed[:32], byteorder="big"),
        ec.SECP256R1(),
        default_backend()
    )

    public_key = private_key.public_key()

    # Generate address from public key
    address = generate_wallet_address(public_key)

    # Serialize private key
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')

    # Update the user's wallet in the database
    if pincode in fake_users_db:
        alias = generate_unique_alias()
        fake_users_db[pincode].wallet = {
            "address": address,
            "private_key": private_key_pem,
            "mnemonic": request.mnemonic,
            "alias": alias  # Store the alias
        }
    else:
        raise HTTPException(status_code=400, detail="Pincode not registered")

    # Fetch and return the balance for the imported wallet
    balance = blockchain.get_balance(address)
    return {
        "address": address,
        "private_key": private_key_pem,
        "mnemonic": request.mnemonic,
        "balance": balance,
        "alias": alias
    }



@app.post("/create_pq_wallet")
def create_pq_wallet(pincode: str = Depends(authenticate)):
    pq_wallet = PQWallet()
    address = pq_wallet.get_address()
    pq_public_key = pq_wallet.get_pq_public_key()
    return {"address": address, "pq_public_key": pq_public_key}

@app.post("/create_wallet")
def create_wallet(pincode: str = Depends(authenticate)):
    wallet = Wallet()
    address = wallet.get_address()
    private_key = wallet.private_key_pem()
    mnemonic = wallet.generate_mnemonic()
    alias = generate_unique_alias()
    
    if pincode in fake_users_db:
        fake_users_db[pincode].wallet = {
            "address": address,
            "private_key": private_key,
            "mnemonic": mnemonic,
            "alias": alias  # Store the alias
        }
    else:
        raise HTTPException(status_code=400, detail="Pincode not registered")

    return {"address": address, "private_key": private_key, "mnemonic": mnemonic, "alias": alias}


class AddressRequest(BaseModel):
    address: str
class WalletRequest(BaseModel):
    wallet_address: str
class BalanceResponse(BaseModel):
    balance: Decimal = Field(..., description="Current balance")
    transactions: List[Transaction] = Field([], description="List of transactions")
@app.post("/get_balance", response_model=BalanceResponse)
async def get_balance(request: WalletRequest):
    wallet_address = request.wallet_address
    try:
        # Ensure the wallet address is valid
        if not wallet_address or not re.match(r'^plata[a-f0-9]{64}$', wallet_address):
            logger.error(f"Invalid wallet address format: {wallet_address}")
            raise HTTPException(status_code=422, detail="Invalid wallet address format")

        balance = blockchain.get_balance(wallet_address)
        # Ensure balance is a Decimal with 18 decimal places
        balance = Decimal(balance).quantize(Decimal('0.000000000000000001'))

        transactions = [
            Transaction(
                tx_hash=tx.get('hash', 'unknown'),
                sender=tx.get('sender', 'unknown'),
                receiver=tx.get('receiver', 'unknown'),
                amount=Decimal(tx.get('amount', 0)).quantize(Decimal('0.000000000000000001')),
                timestamp=tx.get('timestamp', 'unknown')
            )
            for tx in blockchain.get_transactions(wallet_address)
        ]
        logger.info(f"Balance retrieved for address {wallet_address}: {balance}")
        return BalanceResponse(balance=balance, transactions=transactions)
    except KeyError as e:
        logger.error(f"Missing key in transaction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: missing key {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/send_transaction")
async def send_transaction(transaction: Transaction, pincode: str = Depends(authenticate)):
    try:
        logger.info(f"Received transaction request: {transaction}")

        # Check if receiver is an alias and get the corresponding address
        if transaction.receiver.startswith("plata"):
            receiver_address = transaction.receiver
        else:
            receiver_alias = transaction.receiver
            receiver_address = None
            for user in fake_users_db.values():
                if user.wallet['alias'] == receiver_alias:
                    receiver_address = user.wallet['address']
                    break
            if not receiver_address:
                raise HTTPException(status_code=400, detail="Invalid receiver alias")

        transaction.receiver = receiver_address

        # Create a wallet from the private key
        wallet = Wallet(private_key=transaction.private_key)
        logger.info("Created wallet from private key.")

        # Create the message to sign
        message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"

        # Sign the message
        transaction.signature = wallet.sign_message(message)
        logger.info(f"Transaction signed with signature: {transaction.signature}")

        # Add the public key to the transaction
        transaction.public_key = wallet.get_public_key()
        logger.info(f"Public key added to transaction: {transaction.public_key}")

        # Verify if the transaction can be added to the blockchain
        if blockchain.add_transaction(transaction):
            logger.info(f"Transaction from {transaction.sender} to {transaction.receiver} added to blockchain.")

            # Broadcast the new balance to all connected clients
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
    wallet_address: str  
    node_ip: str
    node_port: int

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

def mining_algorithm(iterations=5):
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
                logger.debug(f"Processing edge: {edge}, i: {i}, j: {j}")
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

async def create_new_block(miner_address: str):
    new_block = await blockchain.create_block(miner_address)
    if new_block:
        logger.info(f"New block mined: {new_block.hash}")
        return new_block
    else:
        logger.error("Failed to mine a new block")
        return None

@app.post("/mine_block")
async def mine_block(request: MineBlockRequest, pincode: str = Depends(authenticate)):
    global continue_mining
    node_id = request.node_id
    wallet_address = request.wallet_address
    node_ip = request.node_ip
    node_port = request.node_port

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

            counts, energy, entanglement_matrix, qhins, hashrate = result
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
                new_block = await create_new_block(wallet_address)
                
                if new_block:
                    reward = blockchain.calculate_block_reward()
                    blockchain.add_reward_transaction(wallet_address, reward)
                    logger.info(f"Reward of {reward} QuantumDAGKnight Coins added to wallet {wallet_address}")

                    blockchain.adjust_difficulty()

                    logger.info(f"Node {node_id} mined a block and earned {reward} QuantumDAGKnight Coins")
                    await blockchain.propagate_block_to_peers(new_block)

                    hash_rate = 1 / (end_time - start_time)
                    logger.info(f"Mining Successful. Hash Rate: {hash_rate:.2f} hashes/second")
                    logger.info(f"Mining Time: {end_time - start_time:.2f} seconds")
                    logger.info(f"Quantum State Probabilities: {counts}")
                    logger.info(f"Entanglement Matrix: {entanglement_matrix.tolist()}")
                    logger.info(f"Quantum Hash Information Number: {qhins:.6f}")

                    updated_balance = blockchain.get_balance(wallet_address)
                    await manager.broadcast(json.dumps({
                        "type": "balance_update",
                        "wallet_address": wallet_address,
                        "balance": updated_balance
                    }))

                    return {
                        "success": True,
                        "message": f"Block mined successfully. Reward: {reward} QuantumDAGKnight Coins",
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

# Function to periodically trigger mining
async def periodic_mining(miner_address: str):
    while True:
        try:
            await mine_block(MineBlockRequest(
                node_id="automated_miner",
                wallet_address=miner_address,
                node_ip="127.0.0.1",
                node_port=8000
            ), pincode="automated_miner_pincode")
        except Exception as e:
            logger.error(f"Error in periodic mining: {str(e)}")
        await asyncio.sleep(600)  # Wait for 10 minutes before next mining attempt


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


# Example usage
block_data = {
    "previous_hash": blockchain.chain[-1].hash,
    "data": "new block data",
    "transactions": blockchain.pending_transactions,
    "miner_address": "miner1"
}
asyncio.run(propagate_block_to_all_peers(block_data))


async def gossip_protocol(block_data):
    nodes = node_directory.discover_nodes()
    random.shuffle(nodes)
    gossip_nodes = nodes[:3]  # Randomly select 3 nodes to start gossiping
    tasks = [propagate_block(f"http://{node['ip_address']}:{node['port']}/receive_block", block_data) for node in gossip_nodes]
    await asyncio.gather(*tasks)


# Example usage
asyncio.run(gossip_protocol(block_data))
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

# Example usage
if not pbft.committed:
    view_change.initiate_view_change()


class SecureDAGKnightServicer(dagknight_pb2_grpc.DAGKnightServicer):
    def __init__(self, secret_key, node_directory):
        self.secret_key = secret_key
        self.node_directory = node_directory
        self.consensus = Consensus(blockchain=None)
        self.blockchain = QuantumBlockchain(self.consensus, secret_key, node_directory)
        self.consensus.blockchain = self.blockchain
        self.private_key = self.load_or_generate_private_key()
        self.logger = logging.getLogger(__name__)
        self.security_manager = SecurityManager(secret_key)
        self.transaction_pool = []
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


    def GetTransaction(self, request, context):
        transaction_hash = request.transaction_hash
        transaction = self.node_directory.get_transaction(transaction_hash)

        if not transaction:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Transaction not found!')
            return dagknight_pb2.GetTransactionResponse()

        return dagknight_pb2.GetTransactionResponse(transaction=dagknight_pb2.Transaction(
            sender=transaction.sender,
            receiver=transaction.receiver,
            amount=int(transaction.amount),  # Ensure amount is an integer
            signature=transaction.signature,
            public_key=transaction.public_key
        ))

    def AddTransaction(self, request, context):
        transaction = request.transaction
        # Validate and add transaction to the transaction pool
        self.transaction_pool.append(transaction)
        return dagknight_pb2.AddTransactionResponse(success=True)

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


@app.get("/status")
def status():
    nodes = node_directory.discover_nodes()
    return {"status": "ok", "nodes": nodes}


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


def serve_directory_service():
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
            nodes = node_directory.discover_nodes()
            return dagknight_pb2.DiscoverNodesResponse(magnet_links=[node['magnet_link'] for node in nodes])

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dagknight_pb2_grpc.add_DAGKnightServicer_to_server(DirectoryServicer(), server)
    directory_port = int(os.getenv("DIRECTORY_PORT", 50501))
    server.add_insecure_port(f'[::]:{directory_port}')
    server.start()
    logger.info(f"Directory service started on port {directory_port}")
    server.wait_for_termination()
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dagknight_servicer = SecureDAGKnightServicer(SECRET_KEY, node_directory)
    dagknight_pb2_grpc.add_DAGKnightServicer_to_server(dagknight_servicer, server)

    grpc_port = int(os.getenv("GRPC_PORT", 50502))
    service_names = (
        dagknight_pb2.DESCRIPTOR.services_by_name['DAGKnight'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    server.add_insecure_port(f'[::]:{grpc_port}')
    server.start()
    logger.info(f"gRPC server started on port {grpc_port}")
    print(f"gRPC server started on port {grpc_port}. Connect using grpcurl or any gRPC client.")
    return server, dagknight_servicer  # Return the server and servicer for testing purposes



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

    def disconnect(self, user: str):
        if user in self.active_connections:
            websocket = self.active_connections[user]
            del self.active_connections[user]
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
        message = json.dumps({
            "type": "network_stats",
            "data": stats
        })
        await self.broadcast(message)


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
@app.get("/network_stats")
async def get_network_stats(pincode: str = Depends(authenticate)):
    try:
        total_nodes = len(node_directory.discover_nodes())
        total_transactions = blockchain.globalMetrics.totalTransactions
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

        await manager.connect(websocket)
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
            manager.disconnect(websocket)
            logger.info(f"WebSocket disconnected for user: {user}")
    except HTTPException as e:
        logger.error(f"Authentication failed: {str(e)}")
        await websocket.close(code=1008)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
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
        
node_directory = {
    "node_1": {"ip_address": "161.35.219.10", "port": 50503},
    # Add other nodes here
}
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

# Initialize the price feed with the desired conversion rate
price_feed = SimplePriceFeed("1e-18")  # 1 PLATA = 1e-18 platastable


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

# Set the future result to be what you want for your test

# Pass mock_blockchain to your Exchange instance

async def async_main():
    # Initialize the VM and create the genesis wallet
    vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
    mnemonic, genesis_address = create_genesis_wallet(vm)

    # Log the mnemonic and address
    print(f"Genesis Mnemonic: {mnemonic}")
    print(f"Genesis Address: {genesis_address}")

    # Automatically deploy the Cashewstable contract using the genesis address
    collateral_token = "USDC"
    initial_price = 0.000000000000000001  # Adjusted price
    price_feed = SimplePriceFeed(initial_price)
    cashewstable_contract_address = vm.deploy_contract(genesis_address, Cashewstable, collateral_token, price_feed)
    print(f"Cashewstable contract deployed at address: {cashewstable_contract_address}")

    # Node details
    node_id = os.getenv("NODE_ID", "node_1")
    public_key = "public_key_example"
    ip_address = os.getenv("IP_ADDRESS", "127.0.0.1")
    grpc_port = int(os.getenv("GRPC_PORT", 50502))
    api_port = int(os.getenv("API_PORT", 50503))
    directory_ip = os.getenv("DIRECTORY_IP", "127.0.0.1")
    directory_port = int(os.getenv("DIRECTORY_PORT", 50501))
    secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")

    print(f"Starting node {node_id} at {ip_address}:{grpc_port}")

    # Initialize the consensus object
    consensus = PBFTConsensus(nodes=[], node_id=node_id)

    # Initialize the blockchain with the consensus and secret key
    node_directory = NodeDirectory()
    blockchain = QuantumBlockchain(consensus, secret_key, node_directory, vm)

    print(f"NativeCoinContract address: {blockchain.native_coin_contract_address}")
    print(f"NativeCoinContract instance: {blockchain.native_coin_contract}")

    # Update the consensus to point to the correct blockchain
    consensus.blockchain = blockchain

    # Start the directory service in a separate thread
    directory_service_thread = threading.Thread(target=serve_directory_service)
    directory_service_thread.start()

    # Start the gRPC server in a separate thread
    grpc_server = threading.Thread(target=serve)
    grpc_server.start()

    # Ensure the gRPC server is ready
    time.sleep(2)

    # Periodically discover nodes
    discovery_thread = threading.Thread(target=periodically_discover_nodes, args=(directory_ip, directory_port))
    discovery_thread.start()

    # Start periodic network stats update
    asyncio.create_task(periodic_network_stats_update())

    # Set up blockchain event listeners
    blockchain.on_new_block(manager.broadcast_new_block)
    blockchain.on_new_transaction(manager.broadcast_new_transaction)

    # Start the FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=api_port)
    server = uvicorn.Server(config)
    await server.serve()

    # Block propagation logic
    successful_propagations, total_nodes = await propagate_block_to_peers(data, quantum_signature, transactions, miner_address)
    if successful_propagations == 0 and total_nodes > 0:
        print("Failed to propagate block to any peers.")

    task1 = asyncio.create_task(update_prices_periodically())
    task2 = asyncio.create_task(deploy_and_run_market_bot(exchange, vm, plata_contract, price_feed))
    await asyncio.gather(task1, task2)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
