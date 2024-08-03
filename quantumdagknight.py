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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
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


# Initialize the logger
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
    allow_origins=["*"],
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


class User(BaseModel):
    pincode: str


class UserInDB(User):
    hashed_pincode: str
    wallet: dict


class Token(BaseModel):
    access_token: str
    token_type: str
    wallet: dict


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(pincode: str):
    user = fake_users_db.get(pincode)
    if not user:
        return False
    if not verify_password(pincode, user.hashed_pincode):
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


@app.post("/register", response_model=Token)
def register(user: User):
    if user.pincode in fake_users_db:
        raise HTTPException(status_code=400, detail="Pincode already registered")
    hashed_pincode = get_password_hash(user.pincode)
    wallet = {"address": "0x123456", "private_key": "abcd1234"}
    user_in_db = UserInDB(pincode=user.pincode, hashed_pincode=hashed_pincode, wallet=wallet)
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

class QuantumBlockchain:
    def __init__(self, consensus, secret_key):
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
        self.miner_address = None  # Add miner_address attribute
        self.security_manager = SecurityManager(secret_key)
        self.quantum_state_manager = QuantumStateManager()
        self.node_directory = NodeDirectory()
        self.blockchain = Blockchain()  # Initialize Blockchain

    def validate_block(self, block):
        # Check if the previous hash of the block matches the hash of the last block in the chain
        if len(self.chain) > 0 and block.previous_hash != self.chain[-1].hash:
            return False

        # Check if the hash of the block is valid
        if block.hash != block.compute_hash():
            return False

        # Check if the quantum signature is valid
        if not self.validate_quantum_signature(block.quantum_signature):
            return False

    def validate_quantum_signature(self, quantum_signature):
        logger.debug(f"Validating quantum signature: {quantum_signature}")
        try:
            num_qubits = len(quantum_signature)
            qr = QuantumRegister(num_qubits)
            cr = ClassicalRegister(num_qubits)
            qc = QuantumCircuit(qr, cr)

            for i, bit in enumerate(quantum_signature):
                if bit == '1':
                    qc.x(qr[i])
                qc.measure(qr[i], cr[i])

            simulator = AerSimulator()
            transpiled_circuit = transpile(qc, simulator)
            job = simulator.run(transpiled_circuit, shots=1000)
            result = job.result()

            if result.status != 'COMPLETED':
                logger.error(f"Quantum validation job failed: {result.status}")
                return False

            counts = result.get_counts()
            logger.debug(f"Quantum measurement counts: {counts}")

            total_counts = sum(counts.values())
            probabilities = {state: count / total_counts for state, count in counts.items()}
            
            exact_match_prob = probabilities.get(quantum_signature, 0)
            close_match_prob = sum(probabilities[state] for state in probabilities if hamming_distance(state, quantum_signature) <= 1)

            logger.info(f"Exact match probability: {exact_match_prob}")
            logger.info(f"Close match probability: {close_match_prob}")

            is_valid = exact_match_prob > 0.5 or close_match_prob > 0.8

            if is_valid:
                logger.info(f"Quantum signature {quantum_signature} validated successfully")
            else:
                logger.error(f"Quantum signature {quantum_signature} is invalid. Exact probability: {exact_match_prob}, Close probability: {close_match_prob}")
            return is_valid

        except QiskitError as e:
            logger.error(f"Qiskit error during validation: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return False

    def hamming_distance(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def create_genesis_block(self):
        genesis_block = QuantumBlock(
            previous_hash="0",
            data="Genesis Block",
            quantum_signature="00",
            reward=0,
            transactions=[]
        )
        genesis_block.timestamp = time.time()  # Set the timestamp to the current time
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)


    def get_total_supply(self):
        total_supply = sum(self.balances.values())
        logger.info(f"Total supply: {total_supply}")
        return total_supply
    def generate_quantum_signature(self):
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
                raise QiskitError('Quantum job did not complete successfully')

            counts = result.get_counts()
            signature = list(counts.keys())[0]
            logger.info(f"Generated quantum signature: {signature}")

            # Validate the generated signature
            is_valid = self.validate_quantum_signature(signature)
            if not is_valid:
                logger.warning(f"Generated signature {signature} failed validation. Regenerating...")
                return self.generate_quantum_signature()  # Recursively try again

            return signature

        except QiskitError as e:
            logger.error(f"Qiskit error in generate_quantum_signature: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_quantum_signature: {str(e)}")
            raise
    def adjust_difficulty(self):
        if len(self.chain) >= self.adjustment_interval:
            start_block = self.chain[-self.adjustment_interval]
            end_block = self.chain[-1]
            total_time = end_block.timestamp - start_block.timestamp
            avg_time = total_time / (self.adjustment_interval - 1)  # Subtract 1 because we're measuring intervals
            target_time = self.target_block_time

            logger.info(f"Total time: {total_time:.2e}, Avg time: {avg_time:.2e}, Target time: {target_time:.2e}")
            logger.info(f"Current difficulty: {self.difficulty}, Target: {self.target:.2e}")

            # Calculate the adjustment factor
            adjustment_factor = target_time / avg_time

            # Adjust difficulty more gradually
            if adjustment_factor > 1.25:
                self.difficulty = min(int(self.difficulty * 1.25), 256)  # Cap at 256
            elif adjustment_factor < 0.75:
                self.difficulty = max(int(self.difficulty * 0.75), 1)  # Ensure minimum difficulty of 1
            else:
                # Fine-tune difficulty for smaller variations
                self.difficulty = max(int(self.difficulty * adjustment_factor), 1)

            logger.info(f"Adjustment factor: {adjustment_factor:.2f}")
            logger.info(f"New difficulty: {self.difficulty}")

            # Recalculate target based on new difficulty
            self.target = 2**(256 - self.difficulty)
            logger.info(f"New target: {self.target:.2e}")





    def current_reward(self):
        elapsed_time = time.time() - self.start_time
        halvings = int(elapsed_time // self.halving_interval)
        reward = self.initial_reward / (2 ** halvings)
        logger.info(f"Current reward: {reward}")
        return reward
    def get_block_reward(self):
        return self.current_reward()
    def add_block(self, block):
        if not self.validate_block(block):
            logger.error(f"Block validation failed for block with hash: {block.hash}")
            return False
        
        if block.previous_hash != self.chain[-1].hash:
            logger.error(f"Block's previous hash doesn't match the last block in the chain. Expected: {self.chain[-1].hash}, Got: {block.previous_hash}")
            return False
        
        self.chain.append(block)
        logger.info(f"Block added to the chain. New chain length: {len(self.chain)}")
        return True

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
            
            # Process transactions before adding the reward
            self.process_transactions(transactions)
            
            # Add the reward to the miner's balance
            self.balances[miner_address] = self.balances.get(miner_address, 0) + reward
            logger.info(f"Updated balance for miner {miner_address}: {self.balances[miner_address]}")
            
            # Update total supply after adding the reward
            self.update_total_supply(reward)

            total_supply = self.get_total_supply()
            logger.info(f"Total supply after adding block: {total_supply:.2e}")

            return reward
        else:
            logger.error("Block validation failed. Block not added to the chain.")
            raise ValueError("Invalid block")



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
    def process_transactions(self, transactions):
        for tx in transactions:
            if isinstance(tx, dict):
                sender = tx['sender']
                receiver = tx['receiver']
                amount = tx['amount']
            else:  # Assume it's a Transaction object
                sender = tx.sender
                receiver = tx.receiver
                amount = tx.amount
            
            if self.balances.get(sender, 0) >= amount:
                self.balances[sender] = self.balances.get(sender, 0) - amount
                self.balances[receiver] = self.balances.get(receiver, 0) + amount
                logger.info(f"Updated balance for sender {sender}: {self.balances[sender]}")
                logger.info(f"Updated balance for receiver {receiver}: {self.balances[receiver]}")
            else:
                logger.warning(f"Insufficient balance for transaction: {tx}")

        total_supply = self.get_total_supply()
        logger.info(f"Total supply after processing transactions: {total_supply}")





    def add_transaction(self, transaction):
        logger.debug(f"Adding transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        logger.debug(f"Sender balance before transaction: {self.balances.get(transaction.sender, 0)}")

        # Check if the sender has enough balance
        if self.balances.get(transaction.sender, 0) >= transaction.amount:
            # Verify the transaction signature
            wallet = Wallet()
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
            if wallet.verify_signature(message, transaction.signature, transaction.public_key):
                self.pending_transactions.append(transaction.to_dict())
                logger.debug(f"Transaction added. Pending transactions count: {len(self.pending_transactions)}")
                return True
            else:
                logger.debug(f"Transaction signature verification failed for transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")
        else:
            logger.debug(f"Transaction failed. Insufficient balance for sender {transaction.sender}")

        return False

    def get_balance(self, address):
        balance = self.balances.get(address, 0)
        logger.info(f"Balance for {address}: {balance}")
        return balance

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
    def update_total_supply(self, reward):
        total_supply = self.get_total_supply()
        new_total_supply = total_supply + reward
        logger.info(f"Total supply updated from {total_supply:.2e} to {new_total_supply:.2e}")

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
    private_key: Optional[str] = None
    public_key: Optional[str] = None
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

    def get_address(self):
        public_key_bytes = self.public_key.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
        return hashlib.sha256(public_key_bytes).hexdigest()

    def private_key_pem(self):
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
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

    def get_public_key(self):
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')

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
        logger.info(f"Validating block with hash: {block.hash}")

        if len(self.blockchain.chain) > 0:
            last_block_hash = self.blockchain.chain[-1].hash
            logger.debug(f"Last block hash: {last_block_hash}, Block's previous hash: {block.previous_hash}")
            if block.previous_hash != last_block_hash:
                logger.error(f"Invalid block: Previous hash does not match. Expected {last_block_hash}, got {block.previous_hash}")
                return False

        computed_hash = block.compute_hash()
        logger.debug(f"Computed hash: {computed_hash}, Block's hash: {block.hash}")
        if block.hash != computed_hash:
            logger.error(f"Invalid block: Computed hash does not match. Expected {computed_hash} but got {block.hash}")
            return False

        if not self.is_valid_hash(block.hash):
            logger.error(f"Invalid block: Hash does not meet the required criteria. Hash: {block.hash}, Target: {self.blockchain.target}")
            return False

        if not self.validate_quantum_signature(block.quantum_signature):
            logger.error(f"Invalid block: Quantum signature is not valid. Signature: {block.quantum_signature}")
            return False

        for tx in block.transactions:
            if not self.validate_transaction(tx):
                logger.error(f"Invalid block: Contains invalid transaction {tx}")
                return False

        logger.info("Block validated successfully")
        return True


    def is_valid_hash(self, block_hash):
        logger.info(f"Validating hash against target. Current difficulty: {self.blockchain.difficulty}, Target: {self.blockchain.target}")
        valid = int(block_hash, 16) < self.blockchain.target
        logger.debug(f"Is valid hash: {valid}")
        return valid
    def validate_quantum_signature(self, quantum_signature):
        logger.debug(f"Validating quantum signature: {quantum_signature}")
        try:
            num_qubits = len(quantum_signature)
            qr = QuantumRegister(num_qubits)
            cr = ClassicalRegister(num_qubits)
            qc = QuantumCircuit(qr, cr)

            for i, bit in enumerate(quantum_signature):
                if bit == '1':
                    qc.x(qr[i])
                qc.measure(qr[i], cr[i])

            simulator = AerSimulator()
            transpiled_circuit = transpile(qc, simulator)
            job = simulator.run(transpiled_circuit, shots=1000)
            result = job.result()

            if result.status != 'COMPLETED':
                logger.error(f"Quantum validation job failed: {result.status}")
                return False

            counts = result.get_counts()
            logger.debug(f"Quantum measurement counts: {counts}")

            total_counts = sum(counts.values())
            probabilities = {state: count / total_counts for state, count in counts.items()}
            
            exact_match_prob = probabilities.get(quantum_signature, 0)
            close_match_prob = sum(probabilities[state] for state in probabilities if hamming_distance(state, quantum_signature) <= 1)

            logger.info(f"Exact match probability: {exact_match_prob}")
            logger.info(f"Close match probability: {close_match_prob}")

            is_valid = exact_match_prob > 0.5 or close_match_prob > 0.8

            if is_valid:
                logger.info(f"Quantum signature {quantum_signature} validated successfully")
            else:
                logger.error(f"Quantum signature {quantum_signature} is invalid. Exact probability: {exact_match_prob}, Close probability: {close_match_prob}")
            return is_valid

        except QiskitError as e:
            logger.error(f"Qiskit error during validation: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return False

    def hamming_distance(s1, s2):
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
blockchain = QuantumBlockchain(consensus, secret_key)  # Initialize the blockchain with the consensus and secret key

# Update the consensus to point to the correct blockchain
consensus.blockchain = blockchain

class ImportWalletRequest(BaseModel):
    mnemonic: str


@app.post("/import_wallet")
def import_wallet(request: ImportWalletRequest, pincode: str = Depends(authenticate)):
    wallet = Wallet(mnemonic=request.mnemonic)
    address = wallet.get_address()
    private_key = wallet.private_key_pem()
    return {"address": address, "private_key": private_key}


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
    return {"address": address, "private_key": private_key, "mnemonic": mnemonic}


class AddressRequest(BaseModel):
    address: str


@app.post("/get_balance")
def get_balance(request: AddressRequest, pincode: str = Depends(authenticate)):
    address = request.address
    balance = blockchain.get_balance(address)
    transactions = [
        {
            "tx_hash": tx['hash'],
            "sender": tx['sender'],
            "receiver": tx['receiver'],
            "amount": tx['amount'],
            "timestamp": tx['timestamp']
        } for tx in blockchain.get_transactions(address)
    ]
    return {"balance": balance, "transactions": transactions}

@app.post("/send_transaction")
def send_transaction(transaction: Transaction, pincode: str = Depends(authenticate)):
    try:
        logger.info("Received transaction request.")
        
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
def mine_block(request: MineBlockRequest, pincode: str = Depends(authenticate)):
    global continue_mining
    node_id = request.node_id
    miner_address = request.node_id

    try:
        logging.info(f"Starting mining process for node {node_id}")
        iteration_count = 0
        max_iterations = 20  # Set a maximum number of iterations to prevent endless looping

        while continue_mining and iteration_count < max_iterations:
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(mining_algorithm, iterations=5)  # Run the simulation multiple times
                try:
                    result = future.result(timeout=300)  # 5 minutes timeout
                    logging.info("Mining algorithm completed within timeout")
                except TimeoutError:
                    logging.error("Mining algorithm timed out after 5 minutes")
                    continue  # Continue mining even if there's a timeout

            if isinstance(result, dict) and not result.get("success", True):
                logging.error(f"Mining algorithm failed: {result.get('message')}")
                continue  # Continue mining even if there's an error

            counts, energy, entanglement_matrix, qhins, hashrate = result
            end_time = time.time()

            logging.info(f"Quantum Annealing Simulation completed in {end_time - start_time:.2f} seconds")
            logging.info("Checking mining conditions")

            # Log the probabilities of all states
            for state, probability in counts.items():
                logging.info(f"State: {state}, Probability: {probability:.6f}")

            # New condition to check if any probability exceeds a lower threshold
            max_state = max(counts, key=counts.get)
            max_prob = counts[max_state]

            # Cumulative probability of top N states
            top_n = 10
            sorted_probs = sorted(counts.values(), reverse=True)
            cumulative_prob = sum(sorted_probs[:top_n])

            if cumulative_prob > 0.01:  # Adjusted cumulative threshold to be lower
                logging.info(f"Mining condition met with cumulative probability of top {top_n} states: {cumulative_prob:.6f}, generating quantum signature")
                quantum_signature = blockchain.generate_quantum_signature()

                logging.info("Adding block to blockchain")
                reward = blockchain.add_new_block(f"Block mined by {node_id}", quantum_signature, blockchain.pending_transactions, miner_address)
                blockchain.pending_transactions = []

                # Adjust the difficulty after adding a block
                blockchain.adjust_difficulty()

                logging.info(f"Node {node_id} mined a block and earned {reward} QuantumDAGKnight Coins")
                propagate_block_to_peers(f"Block mined by {node_id}", quantum_signature, blockchain.chain[-1].transactions, miner_address)

                hash_rate = 1 / (end_time - start_time)
                logging.info(f"Mining Successful. Hash Rate: {hash_rate:.2f} hashes/second")
                logging.info(f"Mining Time: {end_time - start_time:.2f} seconds")
                logging.info(f"Quantum State Probabilities: {counts}")
                logging.info(f"Entanglement Matrix: {entanglement_matrix.tolist()}")
                logging.info(f"Quantum Hash Information Number: {qhins:.6f}")

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
                logging.warning(f"Mining failed. Condition not met. Highest probability state: {max_state} with probability: {max_prob:.6f}")
                iteration_count += 1
                if iteration_count >= max_iterations:
                    logging.error(f"Maximum number of iterations ({max_iterations}) reached. Stopping mining.")
                    break  # Stop mining if maximum iterations reached
                continue  # Continue mining even if the condition is not met

        return {"success": False, "message": "Mining failed after maximum iterations."}

    except Exception as e:
        logging.error(f"Error during mining: {str(e)}")
        logging.error(traceback.format_exc())  # Log the traceback
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
        self.blockchain = blockchain
        self.private_key = self.load_or_generate_private_key()
        self.logger = logging.getLogger(__name__)
        self.security_manager = SecurityManager(secret_key)
        self.transaction_pool = []
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
        if not token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing token")
        node_id = self.security_manager.verify_token(token)
        if not node_id:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")
        return node_id

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

    def SendQuantumState(self, request, context):
        self._check_authorization(context)
        node_id = request.node_id
        quantum_state = request.quantum_state
        shard_id = request.shard_id

        peer_public_key_bytes = dict(context.invocation_metadata()).get('public_key')
        peer_public_key = nacl.public.PublicKey(peer_public_key_bytes)

        shared_box = Box(self.private_key, peer_public_key)
        decrypted_quantum_state = self.decrypt_message(request.encrypted_quantum_state, shared_box)

        qc = QuantumCircuit.from_qasm_str(decrypted_quantum_state)
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        statevector = result.get_statevector()

        quantum_state_manager.store_quantum_state(shard_id, statevector)
        logger.info(f"Received and stored quantum state from node {node_id} for shard {shard_id}")

        stored_state = quantum_state_manager.retrieve_quantum_state(shard_id)
        if np.allclose(stored_state, statevector):
            logger.info(f"Quantum state for shard {shard_id} verified successfully")
            return dagknight_pb2.QuantumStateResponse(success=True)
        else:
            logger.error(f"Error: Quantum state for shard {shard_id} could not be verified")
            return dagknight_pb2.QuantumStateResponse(success=False)

    def RequestConsensus(self, request, context):
        self._check_authorization(context)
        node_ids = request.node_ids
        network_quality = request.network_quality

        logger.info(f"Consensus requested for nodes {node_ids} with network quality {network_quality}")

        for node_id in node_ids:
            if random.random() < network_quality:
                state = quantum_state_manager.retrieve_quantum_state(node_id)
                if state is not None:
                    consensus_manager.add_node_state(node_id, tuple(state))
                else:
                    logger.warning(f"Warning: No quantum state found for node {node_id}")
            else:
                logger.info(f"Node {node_id} failed to participate due to poor network quality")

        consensus_result = consensus_manager.get_consensus()
        result_str = f"Consensus reached: {consensus_result}" if consensus_result else "No consensus reached"
        logger.info(result_str)
        return dagknight_pb2.ConsensusResponse(consensus_result=result_str)
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
    dagknight_pb2_grpc.add_DAGKnightServicer_to_server(dagknight_servicer, server)  # Corrected line

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
    server.wait_for_termination()




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

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
        return

    # Node details
    node_id = os.getenv("NODE_ID", "node_1")  # Use environment variables or change manually
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
    blockchain = QuantumBlockchain(consensus, secret_key)

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

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=api_port)

if __name__ == "__main__":
    main()
