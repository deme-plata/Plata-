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
from qiskit_aer import Aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, QuantumCircuit
from qiskit.providers.jobstatus import JobStatus
from qiskit.exceptions import QiskitError
from passlib.context import CryptContext
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from nacl.public import PrivateKey, Box
from nacl.utils import random
from fastapi import FastAPI, HTTPException, Depends, status
from cryptography.exceptions import InvalidSignature

import dagknight_pb2
import dagknight_pb2_grpc

from dagknight_pb2 import *
from dagknight_pb2_grpc import DAGKnightStub
import base64
import hashlib
from grpc_reflection.v1alpha import reflection
import numpy as np
import random
from mnemonic import Mnemonic
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
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from nacl.public import PrivateKey, Box
from nacl.utils import random

import dagknight_pb2
import dagknight_pb2_grpc
from dagknight_pb2 import *
from dagknight_pb2_grpc import DAGKnightStub
import base64
import hashlib
from grpc_reflection.v1alpha import reflection
import numpy as np
import random
from mnemonic import Mnemonic
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import List  # Add this import statement


# Import the SimpleVM class
from vm import SimpleVM

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("node")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

MAX_SUPPLY = 21000000  # Example max supply

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory storage for demonstration purposes
fake_users_db = {}
class User(BaseModel):
    pincode: str

class UserInDB(User):
    hashed_pincode: str
    wallet: dict

class Token(BaseModel):
    access_token: str
    token_type: str
    wallet: dict

def constant_time_compare(val1, val2):
    if len(val1) != len(val2):
        return False
    result = 0
    for x, y in zip(val1, val2):
        result |= x ^ y
    return result == 0

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
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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
    def __init__(self, previous_hash, data, quantum_signature, reward, transactions):
        self.previous_hash = previous_hash
        self.data = data
        self.quantum_signature = quantum_signature
        self.reward = reward
        self.transactions = transactions
        self.merkle_tree = MerkleTree(transactions)
        self.merkle_root = self.merkle_tree.get_root()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        merkle_root = self.merkle_root if self.merkle_root else ""
        return hashlib.sha256((str(self.previous_hash) + str(self.data) + str(self.quantum_signature) + str(merkle_root)).encode('utf-8')).hexdigest()

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
                right = tree[-1][i+1] if i+1 < len(tree[-1]) else left
                level.append(self.hash_pair(left, right))
            tree.append(level)
        return tree

    def hash_pair(self, left, right):
        return hashlib.sha256((left + right).encode('utf-8')).hexdigest()
    
    def get_root(self):
        return self.tree[-1][0] if self.tree else None

class QuantumBlockchain:
    def __init__(self):
        self.initial_reward = 50
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.balances = {}
        self.stakes = {}
        self.halving_interval = 4 * 365 * 24 * 3600
        self.start_time = time.time()
        self.difficulty = 1
        self.target = 2**(256 - self.difficulty)
        self.max_supply = MAX_SUPPLY

    def create_genesis_block(self):
        return QuantumBlock("0", "Genesis Block", self.generate_quantum_signature(), self.initial_reward, [])

    def generate_quantum_signature(self):
        num_qubits = 2
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        qc = QuantumCircuit(qr, cr)

        for i in range(num_qubits):
            qc.h(qr[i])
            qc.measure(qr[i], cr[i])

        backend = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(qc, backend)
        job = backend.run(transpiled_circuit)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        signature = list(counts.keys())[0]
        logger.info(f"Generated quantum signature: {signature}")
        return signature

    def current_reward(self):
        elapsed_time = time.time() - self.start_time
        halvings = int(elapsed_time // self.halving_interval)
        reward = self.initial_reward / (2 ** halvings)
        logger.info(f"Current reward: {reward}")
        return reward

    def adjust_difficulty(self):
        if len(self.chain) % 10 == 0:  # Adjust difficulty every 10 blocks
            total_time = sum(block.timestamp - self.chain[i - 1].timestamp for i, block in enumerate(self.chain[1:], start=1))
            avg_time = total_time / len(self.chain)
            target_time = 10 * 60  # Target block time: 10 minutes
            if avg_time < target_time:
                self.difficulty += 1
            elif avg_time > target_time:
                self.difficulty -= 1
            self.target = 2**(256 - self.difficulty)
            logger.info(f"Adjusted difficulty to {self.difficulty} with target {self.target}")

    def get_total_supply(self):
        return sum(self.balances.values())

    def add_block(self, data, quantum_signature, transactions, miner_address):
        previous_block = self.chain[-1]
        reward = self.current_reward()
        total_supply = self.get_total_supply()
        
        # Calculate the actual reward considering the max supply
        if total_supply + reward > self.max_supply:
            reward = self.max_supply - total_supply
        
        new_block = QuantumBlock(previous_block.hash, data, quantum_signature, reward, transactions)
        self.chain.append(new_block)
        self.process_transactions(transactions)

        if reward > 0:
            self.balances[miner_address] = self.balances.get(miner_address, 0) + reward
            logger.info(f"Block added: {new_block.hash}")
            logger.info(f"Updated balance for miner {miner_address}: {self.balances[miner_address]}")
        
        # Log total supply after adding block
        total_supply = self.get_total_supply()
        logger.info(f"Total supply after adding block: {total_supply}")
        
        self.adjust_difficulty()
        return reward

    def process_transactions(self, transactions):
        for tx in transactions:
            transaction = Transaction.from_dict(tx)
            self.balances[transaction.sender] = self.balances.get(transaction.sender, 0) - transaction.amount
            self.balances[transaction.receiver] = self.balances.get(transaction.receiver, 0) + transaction.amount
            logger.info(f"Updated balance for sender {transaction.sender}: {self.balances[transaction.sender]}")
            logger.info(f"Updated balance for receiver {transaction.receiver}: {self.balances[transaction.receiver]}")
            
            # Log total supply after each transaction
            total_supply = self.get_total_supply()
            logger.info(f"Total supply after transaction: {total_supply}")

    def add_transaction(self, transaction):
        if self.balances.get(transaction.sender, 0) >= transaction.amount:
            self.pending_transactions.append(transaction.to_dict())
            return True
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

class Wallet:
    def __init__(self, private_key=None, mnemonic=None):
        self.mnemo = Mnemonic("english")
        if mnemonic:
            seed = self.mnemo.to_seed(mnemonic)
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
        return self.mnemo.generate(strength=128)

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
            PKCS1_OAEP.new(rsa_public_key)
        )
        return base64.b64encode(encrypted_message).decode('utf-8')

    def decrypt_message(self, encrypted_message):
        rsa_private_key = serialization.load_pem_private_key(self.private_key_pem().encode(), password=None, backend=default_backend())
        decrypted_message = rsa_private_key.decrypt(
            base64.b64decode(encrypted_message),
            PKCS1_OAEP.new(rsa_private_key)
        )
        return decrypted_message.decode('utf-8')

class PQWallet(Wallet):
    def __init__(self):
        super().__init__()
        self.rsa_keypair = RSA.generate(2048)

    def get_pq_public_key(self):
        return self.rsa_keypair.publickey().export_key(format='PEM').decode('utf-8')

    def pq_encrypt(self, message):
        cipher = PKCS1_OAEP.new(self.rsa_keypair.publickey())
        encrypted_message = cipher.encrypt(message.encode('utf-8'))
        return base64.b64encode(encrypted_message).decode('utf-8')

    def pq_decrypt(self, ciphertext):
        cipher = PKCS1_OAEP.new(self.rsa_keypair)
        decrypted_message = cipher.decrypt(base64.b64decode(ciphertext))
        return decrypted_message.decode('utf-8')

class Transaction(BaseModel):
    sender: str
    receiver: str
    amount: float
    private_key: str
    signature: Optional[str] = None

    def sign_transaction(self, wallet):
        private_key = wallet.private_key
        message = f'{self.sender}{self.receiver}{self.amount}'.encode('utf-8')
        signature = private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )
        self.signature = base64.b64encode(signature).decode('utf-8')

    def to_dict(self):
        return self.dict()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

blockchain = QuantumBlockchain()

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

# Initialize the SimpleVM instance
simple_vm = SimpleVM()

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
        wallet = Wallet(private_key=transaction.private_key)
        logger.info("Created wallet from private key.")
        message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
        transaction.signature = wallet.sign_message(message)
        logger.info(f"Transaction signed with signature: {transaction.signature}")

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
        num_qubits = 10  # Reduced number of qubits
        graph = nx.grid_graph(dim=[2, 5])  # 2x5 grid instead of 5x5

        def quantum_annealing_simulation(params):
            hamiltonian = sparse.csr_matrix((2**num_qubits, 2**num_qubits), dtype=complex)
            for edge in graph.edges():
                i, j = edge  # Extract nodes from the edge tuple
                sigma_z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
                term = sparse.kron(sparse.eye(2**i[0], dtype=complex), sigma_z)  # Using i[0] for proper dimension
                term = sparse.kron(term, sparse.eye(2**(num_qubits-i[0]-1), dtype=complex))
                hamiltonian += term

            problem_hamiltonian = sparse.diags(np.random.randn(2**num_qubits), dtype=complex)
            hamiltonian += params[0] * problem_hamiltonian

            initial_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
            evolution = sparse.linalg.expm(-1j * hamiltonian.tocsc() * params[1])
            final_state = evolution @ initial_state

            logging.info(f"Shape of final_state: {final_state.shape}")
            logging.info(f"Type of final_state: {type(final_state)}")
            logging.info(f"First element of final_state: {final_state[0]}")

            # Ensure final_state is a 1D array
            if final_state.ndim != 1:
                logging.error(f"final_state is not a 1D array: {final_state.shape}")
                return float('inf')  # Return a large value to indicate an error

            return -np.abs(final_state[0])**2  # Negative because we're minimizing

        cumulative_counts = {}
        for _ in range(iterations):
            logging.info("Running Quantum Annealing Simulation")
            start_time_simulation = time.time()
            # Introduce randomness in the parameters to help produce a wider range of probabilities
            random_params = [random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)]
            result = minimize(quantum_annealing_simulation, random_params, method='Nelder-Mead')
            end_time_simulation = time.time()

            simulation_duration = end_time_simulation - start_time_simulation

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
                reward = blockchain.add_block(f"Block mined by {node_id}", quantum_signature, blockchain.pending_transactions, miner_address)
                blockchain.pending_transactions = []

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

def propagate_block_to_peers(data, quantum_signature, transactions, miner_address):
    nodes = node_directory.discover_nodes()
    logger.info(f"Propagating block to nodes: {nodes}")
    for node in nodes:
        try:
            with grpc.insecure_channel(f"{node['ip_address']}:{node['port']}") as channel:
                stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                block = dagknight_pb2.Block(
                    previous_hash=blockchain.chain[-2].hash,
                    data=data,
                    quantum_signature=quantum_signature,
                    reward=blockchain.current_reward(),
                    transactions=[dagknight_pb2.Transaction(sender=tx['sender'], receiver=tx['receiver'], amount=tx['amount']) for tx in transactions]
                )
                request = dagknight_pb2.PropagateBlockRequest(block=block, miner_address=miner_address)
                response = stub.PropagateBlock(request)
                if response.success:
                    logger.info(f"Node {node['node_id']} received block with hash: {block.previous_hash}")
                else:
                    logger.error(f"Node {node['node_id']} failed to receive the block.")
        except Exception as e:
            logger.error(f"Failed to propagate block to node {node['node_id']}: {e}")

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

class SecureDAGKnightServicer(dagknight_pb2_grpc.DAGKnightServicer):
    def __init__(self, private_key):
        self.private_key = private_key

    def authenticate(self, context):
        metadata = dict(context.invocation_metadata())
        token = metadata.get('authorization')
        if not token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Authorization token is missing')
        try:
            jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Invalid token')

    def decrypt_message(self, encrypted, shared_box):
        decrypted = shared_box.decrypt(encrypted)
        return decrypted.decode()

    def encrypt_message(self, message, shared_box):
        nonce = random(Box.NONCE_SIZE)
        encrypted = shared_box.encrypt(message.encode(), nonce)
        return encrypted

    def SendQuantumState(self, request, context):
        self.authenticate(context)
        node_id = request.node_id
        quantum_state = request.quantum_state
        shard_id = request.shard_id

        # Get the peer's public key from the request metadata
        peer_public_key_bytes = dict(context.invocation_metadata()).get('public_key')
        peer_public_key = nacl.public.PublicKey(peer_public_key_bytes)

        # Create a shared box
        shared_box = Box(self.private_key, peer_public_key)

        # Decrypt the incoming message
        decrypted_quantum_state = self.decrypt_message(request.encrypted_quantum_state, shared_box)

        # Decode the quantum state
        qc = QuantumCircuit.from_qasm_str(decrypted_quantum_state)

        # Simulate the quantum circuit to get the statevector
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        statevector = result.get_statevector()

        # Store the quantum state
        quantum_state_manager.store_quantum_state(shard_id, statevector)

        print(f"Received and stored quantum state from node {node_id} for shard {shard_id}")

        # Verify the stored state
        stored_state = quantum_state_manager.retrieve_quantum_state(shard_id)
        if np.allclose(stored_state, statevector):
            print(f"Quantum state for shard {shard_id} verified successfully")
            return dagknight_pb2.QuantumStateResponse(success=True)
        else:
            print(f"Error: Quantum state for shard {shard_id} could not be verified")
            return dagknight_pb2.QuantumStateResponse(success=False)

    def RequestConsensus(self, request, context):
        self.authenticate(context)
        node_ids = request.node_ids
        network_quality = request.network_quality

        print(f"Consensus requested for nodes {node_ids} with network quality {network_quality}")

        # Simulate each node's state based on network quality
        for node_id in node_ids:
            if random.random() < network_quality:
                # Node successfully participates in consensus
                state = quantum_state_manager.retrieve_quantum_state(node_id)
                if state is not None:
                    consensus_manager.add_node_state(node_id, tuple(state))
                else:
                    print(f"Warning: No quantum state found for node {node_id}")
            else:
                print(f"Node {node_id} failed to participate due to poor network quality")

        # Get the consensus result
        consensus_result = consensus_manager.get_consensus()

        if consensus_result:
            result_str = f"Consensus reached: {consensus_result}"
        else:
            result_str = "No consensus reached"

        print(result_str)
        return dagknight_pb2.ConsensusResponse(consensus_result=result_str)

    def RegisterNode(self, request, context):
        self.authenticate(context)
        node_id = request.node_id
        public_key = request.public_key
        ip_address = request.ip_address
        port = request.port
        try:
            magnet_link = generate_magnet_link(request.node_id, request.public_key, request.ip_address, request.port)
            return dagknight_pb2.RegisterNodeResponse(success=True, magnet_link=magnet_link)
        except Exception as e:
            logger.error(f"Error registering node: {e}")
            return dagknight_pb2.RegisterNodeResponse(success=False, magnet_link="")

    def DiscoverNodes(self, request, context):
        self.authenticate(context)
        try:
            nodes = node_directory.discover_nodes()
            return dagknight_pb2.DiscoverNodesResponse(magnet_links=[node['magnet_link'] for node in nodes])
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f'Error discovering nodes: {str(e)}')

    def MineBlock(self, request, context):
        self.authenticate(context)
        node_id = request.node_id
        try:
            result = mine_block(node_id, None)
            return dagknight_pb2.MineBlockResponse(success=result['success'], message=result['message'])
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f'Error during mining: {str(e)}')

    def StakeCoins(self, request, context):
        self.authenticate(context)
        address = request.address
        amount = request.amount
        result = blockchain.stake_coins(address, amount)
        return dagknight_pb2.StakeCoinsResponse(success=result)

    def UnstakeCoins(self, request, context):
        self.authenticate(context)
        address = request.address
        amount = request.amount
        result = blockchain.unstake_coins(address, amount)
        return dagknight_pb2.UnstakeCoinsResponse(success=result)

    def GetStakedBalance(self, request, context):
        self.authenticate(context)
        address = request.address
        balance = blockchain.get_staked_balance(address)
        return dagknight_pb2.GetStakedBalanceResponse(balance=balance)

    def QKDKeyExchange(self, request, context):
        self.authenticate(context)
        qkd = BB84()
        simulator = Aer.get_backend('aer_simulator')
        key = qkd.run_protocol(simulator)
        return dagknight_pb2.QKDKeyExchangeResponse(node_id=request.node_id, key=key)

    def PropagateBlock(self, request, context):
        block = request.block
        miner_address = request.miner_address
        try:
            blockchain.add_block(
                data=block.data,
                quantum_signature=block.quantum_signature,
                transactions=[{'sender': tx.sender, 'receiver': tx.receiver, 'amount': tx.amount} for tx in block.transactions],
                miner_address=miner_address
            )
            logger.info(f"Received block with hash: {block.previous_hash} from miner {miner_address}")
            return dagknight_pb2.PropagateBlockResponse(success=True)
        except Exception as e:
            logger.error(f"Error adding propagated block: {e}")
            return dagknight_pb2.PropagateBlockResponse(success=False)

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
    dagknight_servicer = SecureDAGKnightServicer(SECRET_KEY)  # Pass the SECRET_KEY here
    dagknight_pb2_grpc.add_DAGKnightServicer_to_server(dagknight_servicer, server)
    grpc_port = int(os.getenv("GRPC_PORT", 50502))  # Use environment variable or default to 50502
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

def run_secure_client():
    channel = grpc.insecure_channel('localhost:50051')
    stub = dagknight_pb2_grpc.DAGKnightStub(channel)

    message = "Hello, secure world!"
    encrypted_message = encrypt_message(message, shared_box)

    metadata = [('public_key', client_public_key.encode())]
    response = stub.SecureRPC(dagknight_pb2.SecureRequest(encrypted_message=encrypted_message), metadata=metadata)

    decrypted_response = decrypt_message(response.encrypted_message, shared_box)
    print("Received:", decrypted_response)

def main():
    # Node details
    node_id = os.getenv("NODE_ID", "node_1")  # Use environment variables or change manually
    public_key = "public_key_example"
    ip_address = os.getenv("IP_ADDRESS", "127.0.0.1")
    grpc_port = int(os.getenv("GRPC_PORT", 50502))
    api_port = int(os.getenv("API_PORT", 50503))
    directory_ip = os.getenv("DIRECTORY_IP", "127.0.0.1")
    directory_port = int(os.getenv("DIRECTORY_PORT", 50501))

    logger.info(f"Starting node {node_id} at {ip_address}:{grpc_port}")

    # Start the directory service in a separate thread
    directory_service_thread = threading.Thread(target=serve_directory_service)
    directory_service_thread.start()

    # Start the gRPC server in a separate thread
    grpc_server = threading.Thread(target=serve)
    grpc_server.start()

    # Ensure the gRPC server is ready
    time.sleep(2)

    # Register the node with the directory
    try:
        node_directory.register_node_with_grpc(node_id, public_key, ip_address, grpc_port, directory_ip, directory_port)
    except Exception as e:
        logger.error(f"Failed to register node: {str(e)}")

    # Periodically discover nodes
    discovery_thread = threading.Thread(target=periodically_discover_nodes, args=(directory_ip, directory_port))
    discovery_thread.start()

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=api_port)

if __name__ == "__main__":
    main()
