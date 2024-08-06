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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("node")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    # Node details
    node_id = os.getenv("NODE_ID", "node_1")
    public_key = "public_key_example"
    ip_address = os.getenv("IP_ADDRESS", "127.0.0.1")
    grpc_port = int(os.getenv("GRPC_PORT", 50502))
    directory_ip = os.getenv("DIRECTORY_IP", "127.0.0.1")
    directory_port = int(os.getenv("DIRECTORY_PORT", 50501))

    # Register the node with the directory
    try:
        node_directory.register_node_with_grpc(node_id, public_key, ip_address, grpc_port, directory_ip, directory_port)
        logger.info("Node registered successfully.")
    except Exception as e:
        logger.error(f"Failed to register node: {str(e)}")

    # Perform full state sync
    try:
        async with grpc.aio.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
            request = dagknight_pb2.FullStateRequest()
            response = await stub.FullStateSync(request)
            blockchain.chain = [QuantumBlock(
                previous_hash=blk.previous_hash,
                data=blk.data,
                quantum_signature=blk.quantum_signature,
                reward=blk.reward,
                transactions=[tx for tx in blk.transactions]
            ) for blk in response.chain]
            blockchain.balances = {k: v for k, v in response.balances.items()}
            blockchain.stakes = {k: v for k, v in response.stakes.items()}
            logger.info("Blockchain state synchronized successfully.")
    except Exception as e:
        logger.error(f"Failed to synchronize blockchain state: {str(e)}")

    yield

    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

MAX_SUPPLY = 21000000  # Example max supply
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory storage for demonstration purposes
fake_users_db = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def generate_salt():
    return base64.b64encode(os.urandom(16)).decode('utf-8')

class User(BaseModel):
    pincode: str

class UserInDB(User):
    hashed_pincode: str
    wallet: dict
    salt: str
    alias: str  # Add an alias field



class Token(BaseModel):
    access_token: str
    token_type: str
    wallet: dict


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


def periodically_discover_nodes(directory_ip, directory_port):
    while True:
        try:
            node_directory.discover_nodes_with_grpc(directory_ip, directory_port)
        except Exception as e:
            logger.error(f"Error during periodic node discovery: {str(e)}")
        time.sleep(60)  # Discover nodes every 60 seconds


class QuantumStateManager:
    def __init__(self):
        self.shards = {}

    def store_quantum_state(self, shard_id, quantum_state):
        self.shards[shard_id] = quantum_state

    def retrieve_quantum_state(self, shard_id):
        return self.shards.get(shard_id)


quantum_state_manager = QuantumStateManager()


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
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
 
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
class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.balances = {}
        self.stakes = {}
import logging

logger = logging.getLogger(__name__)
class QuantumBlockchain:
    def __init__(self, consensus, secret_key, node_directory):
        self.initial_reward = 50.0
        self.chain = []
        self.pending_transactions = []
        self.balances = {}
        self.stakes = {}
        self.halving_interval = 4 * 365 * 24 * 3600  # 4 years in seconds
        self.start_time = time.time()
        self.difficulty = 1
        self.target_block_time = 600  # 10 minutes in seconds
        self.adjustment_interval = 10
        self.max_supply = MAX_SUPPLY
        self.target = 2**(256 - self.difficulty)
        self.consensus = consensus
        self.create_genesis_block()
        self.miner_address = None
        self.security_manager = SecurityManager(secret_key)
        self.quantum_state_manager = QuantumStateManager()
        self.node_directory = node_directory
        self.blocks_since_last_adjustment = 0

        # Event listeners
        self.new_block_listeners = []
        self.new_transaction_listeners = []

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

        if total_supply + reward > self.max_supply:
            reward = self.max_supply - total_supply

        new_block = QuantumBlock(
            previous_hash=previous_hash,
            data=data,
            quantum_signature=quantum_signature,
            reward=reward,
            transactions=transactions,
            timestamp=time.time()
        )

        new_block.mine_block(self.difficulty)
        logger.info(f"Adding new block: {new_block.to_dict()}")

        if self.consensus.validate_block(new_block):
            self.chain.append(new_block)
            self.process_transactions(transactions)
            self.balances[miner_address] = self.balances.get(miner_address, 0) + reward
            self.update_total_supply(reward)
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

    def validate_quantum_signature(self, quantum_signature, threshold=0.5):
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

            max_count = max(counts.values())
            probability = max_count / 1024

            logger.info(f"Validation probability for signature {quantum_signature}: {probability}")
            return probability >= threshold
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
    amount: int
    private_key: Optional[str] = None
    public_key: Optional[str] = None
    signature: Optional[str] = None
    wallet: Optional[Wallet] = None

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
class Consensus:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        self.current_leader = None

    def elect_leader(self):
        nodes = node_directory.discover_nodes()
        logger.debug(f"Discovered nodes for leader election: {nodes}")
        if not nodes:
            logger.warning("No nodes available for leader election")
            return None
        self.current_leader = random.choice(nodes)
        logger.info(f"Elected leader: {self.current_leader['node_id']}")

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

# Initialize the consensus object and the blockchain
consensus = Consensus(blockchain=None)  # Temporarily set blockchain to None
secret_key = "your_secret_key_here"
blockchain = QuantumBlockchain(consensus, secret_key, node_directory)

# Update the consensus to point to the correct blockchain
consensus.blockchain = blockchain

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
    return {"address": address, "private_key": private_key_pem, "mnemonic": request.mnemonic, "balance": balance, "alias": alias}


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

@app.post("/get_balance")
def get_balance(request: WalletRequest):
    wallet_address = request.wallet_address
    try:
        # Ensure the wallet address is valid
        if not wallet_address or not re.match(r'^plata[a-f0-9]{64}$', wallet_address):
            logger.error(f"Invalid wallet address format: {wallet_address}")
            raise HTTPException(status_code=422, detail="Invalid wallet address format")

        balance = blockchain.get_balance(wallet_address)
        transactions = [
            {
                "tx_hash": tx.get('hash', 'unknown'),
                "sender": tx.get('sender', 'unknown'),
                "receiver": tx.get('receiver', 'unknown'),
                "amount": tx.get('amount', 0.0),
                "timestamp": tx.get('timestamp', 'unknown')
            }
            for tx in blockchain.get_transactions(wallet_address)
        ]
        logger.info(f"Balance retrieved for address {wallet_address}: {balance}")
        return {"balance": balance, "transactions": transactions}
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
    contract_code: str
    constructor_args: list = []


@app.post("/deploy_contract")
def deploy_contract(request: DeployContractRequest):
    try:
        contract_address = vm.deploy_contract(request.sender_address, request.contract_code, request.constructor_args)
        return {"contract_address": contract_address}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MineBlockRequest(BaseModel):
    node_id: str
    wallet_address: str  
    node_ip: str
    node_port: int


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
                reward = blockchain.add_new_block(f"Block mined by {node_id}", quantum_signature, blockchain.pending_transactions, wallet_address)
                blockchain.pending_transactions = []
                logger.info(f"Reward of {reward} QuantumDAGKnight Coins added to wallet {wallet_address}")

                blockchain.adjust_difficulty()

                logger.info(f"Node {node_id} mined a block and earned {reward} QuantumDAGKnight Coins")
                propagate_block_to_peers(f"Block mined by {node_id}", quantum_signature, blockchain.chain[-1].transactions, wallet_address)

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
def propagate_block_to_peers(data, quantum_signature, transactions, miner_address):
    nodes = node_directory.discover_nodes()
    logger.info(f"Propagating block to nodes: {nodes}")
    for node in nodes:
        try:
            with grpc.insecure_channel(f"{node['ip_address']}:{node['port']}") as channel:
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
                response = stub.PropagateBlock(request)
                if response.success:
                    logger.info(f"Node {node['node_id']} received block with hash: {block.hash}")
                else:
                    logger.error(f"Node {node['node_id']} failed to receive the block.")
        except Exception as e:
            logger.error(f"Error propagating block to node {node['node_id']}: {str(e)}")


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
        self.active_connections: List[WebSocket] = []
        self.subscriptions = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
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

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def broadcast_new_block(self, block: QuantumBlock):
        message = json.dumps({
            "type": "new_block",
            "data": block.to_dict()
        })
        await self.broadcast(message)

    async def broadcast_new_transaction(self, transaction: Transaction):
        message = json.dumps({
            "type": "new_transaction",
            "data": transaction.to_dict()
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
@app.get("/get_node_details/{node_id}")
def get_node_details(node_id: str):
    node_details = node_directory.get_node_details(node_id)
    if not node_details:
        raise HTTPException(status_code=404, detail="Node not found")
    return node_details
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


@app.get("/")
def get():
    return {"message": "WebSocket server is running"}

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
        return

    # Node details
    node_id = os.getenv("NODE_ID", "node_1")
    public_key = "public_key_example"
    ip_address = os.getenv("IP_ADDRESS", "127.0.0.1")
    grpc_port = int(os.getenv("GRPC_PORT", 50502))
    api_port = int(os.getenv("API_PORT", 50503))
    directory_ip = os.getenv("DIRECTORY_IP", "127.0.0.1")
    directory_port = int(os.getenv("DIRECTORY_PORT", 50501))
    secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")

    logger.info(f"Starting node {node_id} at {ip_address}:{grpc_port}")

    # Initialize the consensus object
    consensus = Consensus(blockchain=None)

    # Initialize the blockchain with the consensus and secret key
    global blockchain  # Make blockchain global so it can be accessed by endpoints
    blockchain = QuantumBlockchain(consensus, secret_key, node_directory)

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

    # Set up event loop for asynchronous tasks
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start periodic network stats update
    loop.create_task(periodic_network_stats_update())

    # Set up blockchain event listeners
    blockchain.on_new_block(manager.broadcast_new_block)
    blockchain.on_new_transaction(manager.broadcast_new_transaction)

    # Start the FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=api_port)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    main()
