from pydantic import BaseModel, Field
from typing import Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import base64
from decimal import Decimal
from typing import Dict, List, Tuple,Any
from SecureHybridZKStark import SecureHybridZKStark
import time 
import json 
from hashlib import sha256
import logging
from dataclasses import dataclass, field
import hashlib
import threading
import logging 
from pydantic import BaseModel, Field, validator
from typing import Optional, Any, Tuple
from decimal import Decimal
import uuid
import hashlib
import base64
import logging
logger = logging.getLogger(__name__)

class Transaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    amount: Decimal
    price: Decimal
    buyer_id: str
    seller_id: str
    public_key: Optional[str] = None
    signature: Optional[str] = None
    zk_proof: Optional[Tuple[Tuple, Tuple]] = None
    wallet: Optional[Any] = None
    tx_hash: str = Field(default="")  # Initialize with empty string
    timestamp: float = Field(default_factory=time.time)  # Set default to current time

    def generate_hash(self) -> str:
        """Generate transaction hash"""
        message = (f"{self.id}{self.sender}{self.receiver}{self.amount}"
                  f"{self.price}{self.buyer_id}{self.seller_id}{self.timestamp}")
        return hashlib.sha256(message.encode()).hexdigest()

    def __init__(self, **data):
        super().__init__(**data)
        if not self.tx_hash:
            self.tx_hash = self.generate_hash()


    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v

    @validator('price')
    def validate_price(cls, v):
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v

    def sign_transaction(self, zk_system: 'SecureHybridZKStark'):
        """Sign the transaction and generate ZK proof"""
        try:
            if not self.wallet:
                raise ValueError("Wallet is required to sign the transaction")
                
            # Create message from transaction data
            message = self._create_message()
            
            # Sign with wallet
            self.signature = self.wallet.sign_message(message)
            self.public_key = self.wallet.get_public_key()
            
            # Generate ZK proof
            secret = int(self.amount * 10**18)  # Convert Decimal to integer
            public_input = int(hashlib.sha256(message.encode()).hexdigest(), 16)
            self.zk_proof = zk_system.prove(secret, public_input)
            
            # Generate transaction hash
            self.tx_hash = hashlib.sha256(
                f"{self.id}{message}{self.signature}".encode()
            ).hexdigest()
            
            logger.debug(f"Transaction {self.id} signed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error signing transaction: {str(e)}")
            return False

    def verify_transaction(self, zk_system: 'SecureHybridZKStark') -> bool:
        """Verify transaction signature and ZK proof"""
        try:
            if not all([self.signature, self.public_key, self.zk_proof]):
                logger.error("Missing required verification components")
                return False
            
            # Create message and verify signature
            message = self._create_message()
            if not self.wallet.verify_signature(message, self.signature, self.public_key):
                logger.error("Signature verification failed")
                return False
            
            # Verify ZK proof
            public_input = int(hashlib.sha256(message.encode()).hexdigest(), 16)
            if not zk_system.verify(public_input, self.zk_proof):
                logger.error("ZK proof verification failed")
                return False
            
            logger.debug(f"Transaction {self.id} verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying transaction: {str(e)}")
            return False

    def _create_message(self) -> str:
        """Create message string for signing/verification"""
        return f"{self.sender}{self.receiver}{self.amount}{self.price}{self.buyer_id}{self.seller_id}"

    def to_dict(self) -> dict:
        """Convert transaction to dictionary"""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": str(self.amount),
            "price": str(self.price),
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "public_key": self.public_key,
            "signature": self.signature,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp
        }

    def model_dump(self) -> dict:
        """Pydantic v2 compatible model dump"""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        """Create transaction from dictionary"""
        # Handle Decimal conversion
        if 'amount' in data:
            data['amount'] = Decimal(str(data['amount']))
        if 'price' in data:
            data['price'] = Decimal(str(data['price']))
        return cls(**data)

    def to_grpc(self) -> Any:
        """Convert to gRPC message"""
        return dagknight_pb2.Transaction(
            id=self.id,
            sender=self.sender,
            receiver=self.receiver,
            amount=int(self.amount * 10**18),  # Convert to integer
            price=int(self.price * 10**18),    # Convert to integer
            public_key=self.public_key,
            signature=self.signature,
            buyer_id=self.buyer_id,
            seller_id=self.seller_id,
            tx_hash=self.tx_hash,
            timestamp=int(self.timestamp) if self.timestamp else None
        )

    @classmethod
    def from_proto(cls, proto) -> 'Transaction':
        """Create transaction from gRPC message"""
        return cls(
            id=proto.id,
            sender=proto.sender,
            receiver=proto.receiver,
            amount=Decimal(proto.amount) / Decimal(10**18),
            price=Decimal(proto.price) / Decimal(10**18),
            public_key=proto.public_key,
            signature=proto.signature,
            buyer_id=proto.buyer_id,
            seller_id=proto.seller_id,
            tx_hash=proto.tx_hash,
            timestamp=proto.timestamp
        )

    class Config:
        arbitrary_types_allowed = True

class QuantumBlock:
    def __init__(
        self,
        previous_hash,
        data,
        quantum_signature,
        reward,
        transactions,
        miner_address,
        nonce,
        parent_hashes,
        timestamp=None
    ):
        self.previous_hash = previous_hash
        self.data = data
        self.quantum_signature = quantum_signature
        self.reward = reward
        self.transactions = transactions
        self.miner_address = miner_address
        self.nonce = nonce
        self.parent_hashes = parent_hashes
        self.dag_parents = parent_hashes[1:] if len(parent_hashes) > 1 else []
        self.timestamp = timestamp or time.time()
        self.hash = None  # Initialize the hash as None
        logger.debug(f"Initialized QuantumBlock: {self.to_dict()}")


    def to_dict(self):
        return {
            "previous_hash": self.previous_hash,
            "data": self.data,
            "quantum_signature": self.quantum_signature,
            "reward": str(self.reward),  # Convert Decimal to string
            "transactions": [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in self.transactions],
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "hash": self.hash
        }

    def compute_hash(self):
        def decimal_default(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError

        block_data = {
            "previous_hash": self.previous_hash,
            "data": self.data,
            "quantum_signature": self.quantum_signature,
            "reward": self.reward,
            "transactions": [tx if isinstance(tx, dict) else tx.to_dict() for tx in self.transactions],
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True, default=decimal_default)
        logger.debug(f"Computing hash for block data: {block_string}")
        return hashlib.sha256(block_string.encode()).hexdigest()

    def is_valid(self):
        try:
            logger.debug("Starting block validation...")
            computed_hash = self.compute_hash()
            is_valid = self.hash == computed_hash
            logger.debug(f"Computed hash: {computed_hash}, Stored hash: {self.hash}, Is valid: {is_valid}")
            if not is_valid:
                logger.error(f"Block validation failed. Stored hash: {self.hash}, Computed hash: {computed_hash}")
                logger.debug(f"Block data during validation: {json.dumps(self.to_dict(), indent=4)}")
            else:
                logger.info(f"Block validated successfully: {self.hash}")
            return is_valid
        except Exception as e:
            logger.error(f"Exception during block validation: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def mine_block(self, difficulty):
        try:
            target = 2 ** (256 - difficulty)
            self.hash = self.compute_hash()  # Compute initial hash
            logger.debug(f"Initial hash: {self.hash}, Target: {target}")
            while int(self.hash, 16) >= target:
                self.nonce += 1
                logger.debug(f"Trying nonce: {self.nonce}")
                self.hash = self.compute_hash()
                logger.debug(f"New hash: {self.hash} with nonce {self.nonce}")
            logger.info(f"Block mined with nonce: {self.nonce}, hash: {self.hash}")
        except Exception as e:
            logger.error(f"Exception during block mining: {str(e)}")
            logger.error(traceback.format_exc())






    @staticmethod
    def json_serial(obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def transaction_to_dict(self, tx):
        if isinstance(tx, dict):
            return {k: str(v) if isinstance(v, Decimal) else v for k, v in tx.items()}
        elif hasattr(tx, 'to_dict'):
            return tx.to_dict()
        elif hasattr(tx, '__dict__'):
            return {k: str(v) if isinstance(v, Decimal) else v for k, v in tx.__dict__.items()}
        else:
            raise TypeError(f"Cannot convert transaction of type {type(tx)} to dict")


    @classmethod
    def from_dict(cls, data):
        block = cls(
            previous_hash=data["previous_hash"],
            data=data["data"],
            quantum_signature=data["quantum_signature"],
            reward=Decimal(data["reward"]),
            transactions=[Transaction.from_dict(tx) for tx in data["transactions"]]
        )
        block.timestamp = data["timestamp"]
        block.nonce = data["nonce"]
        block.hash = data["hash"]
        return block



    def set_hash(self, hash_value):
        self.hash = hash_value

    def recompute_hash(self):
        self.hash = self.compute_hash()

    def hash_block(self):
        self.hash = self.compute_hash()
@dataclass
class NodeState:
    blockchain_length: int
    latest_block_hash: str
    pending_transactions_count: int
    total_supply: Decimal
    difficulty: int
    mempool_size: int
    connected_peers: int
    active_liquidity_pools: int
    node_uptime: float  # in seconds


class NodeDirectory:
    def __init__(self, p2p_node):

        self.nodes = {}  # Dictionary to store nodes
        self.transactions = {}  # Dictionary to store transactions
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.register_times = []
        self.discover_times = []
        self.p2p_node = p2p_node  # Pass P2PNode from outside

    def store_transaction(self, transaction_hash, transaction):
        self.transactions[transaction_hash] = transaction
        print(f"Transaction {transaction_hash} stored in directory.")
    async def register_node(self, node_id, ip_address, port):
        """Register a new node by adding it to the P2P network."""
        try:
            # Ensure p2p_node is available
            if not self.p2p_node:
                raise ValueError("P2PNode instance is not initialized.")

            # Generate a magnet link for the node
            magnet_link = self.p2p_node.generate_magnet_link(node_id, ip_address, port)
            
            # Create a new KademliaNode instance for the node
            kademlia_node = KademliaNode(id=node_id, ip=ip_address, port=port)

            # Attempt to connect to the peer node
            await self.p2p_node.connect_to_peer(kademlia_node)

            # Register the node in the directory
            self.nodes[node_id] = {
                "ip": ip_address,
                "port": port,
                "magnet_link": magnet_link
            }

            logger.info(f"Node {node_id} registered successfully with IP {ip_address} and port {port}.")

        except ValueError as ve:
            logger.error(f"Failed to register node {node_id}: {ve}")
        except Exception as e:
            logger.error(f"An error occurred while registering node {node_id}: {str(e)}")

    def get_all_node_stubs(self):
        """Return a list of all node information."""
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

    async def discover_nodes(self):
        """Discover and return all nodes using P2P."""
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