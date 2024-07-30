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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import dagknight_pb2
import dagknight_pb2_grpc
import base64
import hashlib
from grpc_reflection.v1alpha import reflection
import numpy as np
import random

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

    def add_block(self, data, quantum_signature, transactions, miner_address):
        previous_block = self.chain[-1]
        reward = self.current_reward()
        new_block = QuantumBlock(previous_block.hash, data, quantum_signature, reward, transactions)
        self.chain.append(new_block)
        self.process_transactions(transactions)
        
        # Update miner's balance with the reward
        self.balances[miner_address] = self.balances.get(miner_address, 0) + reward
        logger.info(f"Block added: {new_block.hash}")
        logger.info(f"Updated balance for {miner_address}: {self.balances[miner_address]}")
        self.adjust_difficulty()
        return reward

    def process_transactions(self, transactions):
        for tx in transactions:
            transaction = Transaction.from_dict(tx)
            self.balances[transaction.sender] = self.balances.get(transaction.sender, 0) - transaction.amount
            self.balances[transaction.receiver] = self.balances.get(transaction.receiver, 0) + transaction.amount

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

class Wallet:
    def __init__(self, private_key=None):
        if private_key:
            self.private_key = serialization.load_pem_private_key(
                private_key.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
        else:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
        self.public_key = self.private_key.public_key()

    def get_address(self):
        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_bytes).hexdigest()

    def private_key_pem(self):
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

class Transaction(BaseModel):
    sender: str
    receiver: str
    amount: float
    private_key: str
    signature: Optional[str] = None

    def sign_transaction(self, wallet):
        private_key = wallet.private_key
        message = f'{self.sender}{self.receiver}{self.amount}'.encode('utf-8')
        self.signature = base64.b64encode(private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )).decode('utf-8')

    def to_dict(self):
        return self.dict()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

blockchain = QuantumBlockchain()

# Initialize the SimpleVM instance
simple_vm = SimpleVM()

@app.post("/create_wallet")
def create_wallet(pincode: str = Depends(authenticate)):
    wallet = Wallet()
    address = wallet.get_address()
    private_key = wallet.private_key_pem()
    return {"address": address, "private_key": private_key}

class AddressRequest(BaseModel):
    address: str

@app.post("/get_balance")
def get_balance(request: AddressRequest, pincode: str = Depends(authenticate)):
    address = request.address
    balance = blockchain.get_balance(address)
    return {"balance": balance}

@app.post("/send_transaction")
def send_transaction(transaction: Transaction, pincode: str = Depends(authenticate)):
    try:
        wallet = Wallet(private_key=transaction.private_key)
        transaction.sign_transaction(wallet)
        if blockchain.add_transaction(transaction):
            return {"success": True, "message": "Transaction added successfully."}
        else:
            return {"success": False, "message": "Transaction failed to add."}
    except ValueError as e:
        logger.error(f"Error deserializing private key: {e}")
        raise HTTPException(status_code=400, detail="Invalid private key format")

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

class AddressRequest(BaseModel):
    address: str

@app.post("/mine_block")
def mine_block(request: MineBlockRequest, pincode: str = Depends(authenticate)):
    node_id = request.node_id
    miner_address = request.node_id

    def mining_algorithm():
        try:
            network_size = 5
            qhn = QuantumHolographicNetwork(network_size)
            qhn.initialize_network()
            qhn.apply_holographic_principle()

            mass_distribution = np.random.rand(network_size, network_size)
            qhn.simulate_gravity(mass_distribution)

            qhn.create_black_hole((2, 2), 1)
            qhn.measure_entanglement()
            entanglement_matrix = qhn.entanglement_matrix
            logging.info("Entanglement Matrix:")
            logging.info(entanglement_matrix)

            black_hole_region = [q for q in range(network_size**2) if ((q // network_size - 2)**2 + (q % network_size - 2)**2) <= 1]
            qhn.extract_hawking_radiation(black_hole_region)

            simulator = Aer.get_backend('qasm_simulator')
            transpiled_circuit = transpile(qhn.circuit, simulator)
            job = simulator.run(transpiled_circuit)
            result = job.result()
            counts = result.get_counts()
            logging.info("QHIN Circuit Measurement Results:")
            logging.info(counts)

            qc = QuantumCircuit(3)
            qc.h([0, 1, 2])
            qc.measure_all()
            return qc, counts, result.time_taken
        except Exception as e:
            logging.error(f"Error in mining_algorithm: {str(e)}")
            raise

    try:
        logging.info(f"Starting mining process for node {node_id}")
        start_time = time.time()
        qc, qhin_counts, mining_time = mining_algorithm()
        end_time = time.time()
        
        simulator = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(qc, simulator)
        job = simulator.run(transpiled_circuit)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        logging.info("Quantum Circuit Measurement Results:")
        logging.info(counts)

        if '111' in counts and '111' in qhin_counts:
            quantum_signature = blockchain.generate_quantum_signature()
            reward = blockchain.add_block(f"Block mined by {node_id}", quantum_signature, blockchain.pending_transactions, miner_address)
            blockchain.pending_transactions = []
            logging.info(f"Node {node_id} mined a block and earned {reward} QuantumDAGKnight Coins")
            propagate_block_to_peers(f"Block mined by {node_id}", quantum_signature, blockchain.chain[-1].transactions, miner_address)
            
            # Calculate hash rate (hashes per second)
            hash_rate = 1 / (end_time - start_time)
            
            return {
                "success": True,
                "message": f"Block mined successfully. Reward: {reward} QuantumDAGKnight Coins",
                "hash_rate": hash_rate,
                "mining_time": end_time - start_time,
                "qhin_counts": qhin_counts,
                "entanglement_matrix": entanglement_matrix.tolist()
            }
        else:
            logging.warning("Mining failed. Condition '111' not met.")
            return {"success": False, "message": "Mining failed. Try again."}
    except Exception as e:
        logging.error(f"Error during mining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during mining: {str(e)}")

@app.get("/research_data")
def get_research_data(pincode: str = Depends(authenticate)):
    # Provide the entanglement matrix and counts from QHIN circuits for researchers
    try:
        network_size = 5
        qhn = QuantumHolographicNetwork(network_size)
        qhn.initialize_network()
        qhn.apply_holographic_principle()

        # Simulate gravity with a simple mass distribution
        mass_distribution = np.random.rand(network_size, network_size)
        qhn.simulate_gravity(mass_distribution)

        # Create a black hole
        qhn.create_black_hole((2, 2), 1)

        # Measure entanglement
        qhn.measure_entanglement()
        entanglement_matrix = qhn.entanglement_matrix

        # Extract Hawking radiation
        black_hole_region = [q for q in range(network_size**2) if ((q // network_size - 2)**2 + (q % network_size - 2)**2) <= 1]
        qhn.extract_hawking_radiation(black_hole_region)

        # Execute the QHIN circuit
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(qhn.circuit, simulator)
        job = simulator.run(transpiled_circuit)
        result = job.result()
        counts = result.get_counts()

        return {"entanglement_matrix": entanglement_matrix.tolist(), "counts": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving research data: {str(e)}")

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
                logger.info(f"Block propagated to node {node['node_id']}: {response.success}")
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

class DAGKnightServicer(dagknight_pb2_grpc.DAGKnightServicer):
    def __init__(self, secret_key):
        self.secret_key = secret_key

    def authenticate(self, context):
        metadata = dict(context.invocation_metadata())
        token = metadata.get('authorization')
        if not token:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Authorization token is missing')
        try:
            jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Invalid token')

    def SendQuantumState(self, request, context):
        self.authenticate(context)
        node_id = request.node_id
        quantum_state = request.quantum_state
        shard_id = request.shard_id

        # Decode the quantum state
        qc = QuantumCircuit.from_qasm_str(quantum_state)

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
            magnet_link = node_directory.register_node(node_id, public_key, ip_address, port)
            return dagknight_pb2.RegisterNodeResponse(magnet_link=magnet_link)
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f'Error registering node: {str(e)}')

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
        self.authenticate(context)
        block = request.block
        miner_address = request.miner_address
        try:
            blockchain.add_block(
                data=block.data,
                quantum_signature=block.quantum_signature,
                transactions=[tx for tx in block.transactions],
                miner_address=miner_address
            )
            return dagknight_pb2.PropagateBlockResponse(success=True)
        except Exception as e:
            logger.error(f"Error adding propagated block: {e}")
            return dagknight_pb2.PropagateBlockResponse(success=False)

@app.get("/status")
def status():
    nodes = node_directory.discover_nodes()
    return {"status": "ok", "nodes": nodes}

def serve_directory_service():
    class DirectoryServicer(dagknight_pb2_grpc.DAGKnightServicer):
        def RegisterNode(self, request, context):
            node_id = request.node_id
            public_key = request.public_key
            ip_address = request.ip_address
            port = request.port
            magnet_link = node_directory.register_node(node_id, public_key, ip_address, port)
            return dagknight_pb2.RegisterNodeResponse(magnet_link=magnet_link)

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
    dagknight_servicer = DAGKnightServicer(SECRET_KEY)  # Pass the SECRET_KEY here
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
    with grpc.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
        stub = dagknight_pb2_grpc.DAGKnightStub(channel)
        request = dagknight_pb2.DiscoverNodesRequest()
        response = stub.DiscoverNodes(request)
        logger.info(f"Discovered nodes: {response.magnet_links}")

def periodically_discover_nodes(directory_ip, directory_port):
    while True:
        discover_nodes_with_grpc(directory_ip, directory_port)
        time.sleep(60)  # Discover nodes every 60 seconds

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