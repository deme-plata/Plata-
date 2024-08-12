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

logger = logging.getLogger(__name__)

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



class Transaction(BaseModel):
    id: str = None  # Unique transaction identifier
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

    def __init__(self, **data):
        super().__init__(**data)
        # Generate a unique ID if not provided
        if not self.id:
            self.id = str(uuid.uuid4())

    def sign_transaction(self, zk_system: SecureHybridZKStark):
        if not self.wallet:
            raise ValueError("Wallet is required to sign the transaction")
        message = f"{self.sender}{self.receiver}{self.amount}{self.price}{self.buyer_id}{self.seller_id}"
        self.signature = self.wallet.sign_message(message)
        self.public_key = self.wallet.get_public_key()
        
        secret = int(self.amount * 10**18)  # Convert Decimal to integer
        public_input = int(hashlib.sha256(message.encode()).hexdigest(), 16)
        self.zk_proof = zk_system.prove(secret, public_input)

    def verify_transaction(self, zk_system: SecureHybridZKStark) -> bool:
        if not self.signature or not self.public_key or not self.zk_proof:
            return False
        
        message = f"{self.sender}{self.receiver}{self.amount}{self.price}{self.buyer_id}{self.seller_id}"
        if not self.wallet.verify_signature(message, self.signature, self.public_key):
            return False
        
        public_input = int(hashlib.sha256(message.encode()).hexdigest(), 16)
        return zk_system.verify(public_input, self.zk_proof)

    def compute_hash(self):
        """
        Compute a unique hash for the transaction using its attributes.
        """
        transaction_data = f"{self.sender}{self.receiver}{self.amount}{self.signature}{self.public_key}{self.price}{self.buyer_id}{self.seller_id}"
        return hashlib.sha256(transaction_data.encode()).hexdigest()

    class Config:
        arbitrary_types_allowed = True

    def to_grpc(self):
        return dagknight_pb2.Transaction(
            id=self.id,
            sender=self.sender,
            receiver=self.receiver,
            amount=int(self.amount),  # Convert Decimal to int
            public_key=self.public_key,
            signature=self.signature,
            price=int(self.price),  # Convert Decimal to int
            buyer_id=self.buyer_id,
            seller_id=self.seller_id
        )

    def to_dict(self):
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "signature": self.signature,
            "public_key": self.public_key,
            "price": self.price,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_proto(cls, proto):
        return cls(
            id=proto.id,
            sender=proto.sender,
            receiver=proto.receiver,
            amount=Decimal(proto.amount),
            public_key=proto.public_key,
            signature=proto.signature,
            price=Decimal(proto.price),
            buyer_id=proto.buyer_id,
            seller_id=proto.seller_id
        )

    def is_valid(self, zk_system: SecureHybridZKStark = None) -> bool:
        """
        Validate the transaction with multiple checks, including ZKP if provided.
        """
        # 1. Ensure the amount is positive
        if self.amount <= 0:
            return False

        # 2. Ensure sender and receiver are valid addresses
        if not self.sender or not self.receiver:
            return False

        # 3. Verify that the transaction signature is valid
        if not self.signature or not self.public_key:
            return False

        # Assuming `verify_signature` is a method that checks the signature's validity
        if not self.verify_signature():
            return False

        # 4. If using Zero-Knowledge Proofs, verify the transaction's ZKP
        if zk_system and not self.verify_zkp(zk_system):
            return False

        # If all checks pass, the transaction is valid
        return True

    def verify_signature(self) -> bool:
        """
        Example signature verification logic.
        Ensure that the signature is valid based on the public key and transaction data.
        """
        message = f"{self.sender}{self.receiver}{self.amount}{self.price}{self.buyer_id}{self.seller_id}"
        return self.wallet.verify_message(message, self.signature)

    def verify_zkp(self, zk_system: SecureHybridZKStark) -> bool:
        """
        Example ZKP verification logic.
        Use the provided zk_system to verify the Zero-Knowledge Proof.
        """
        message = f"{self.sender}{self.receiver}{self.amount}{self.price}{self.buyer_id}{self.seller_id}"
        public_input = int(hashlib.sha256(message.encode()).hexdigest(), 16)
        return zk_system.verify(public_input, self.zk_proof)

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
            "reward": str(self.reward),  # Convert Decimal to string for JSON serialization
            "transactions": [txn.to_dict() if hasattr(txn, 'to_dict') else txn for txn in self.transactions],
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }, sort_keys=True, default=self.default_serializer)
        
        block_hash = sha256(block_string.encode()).hexdigest()
        logger.debug(f"Computed hash: {block_hash} for nonce: {self.nonce}")
        return block_hash

    @staticmethod
    def default_serializer(obj):
        if isinstance(obj, Decimal):
            return str(obj)  # Convert Decimal to string
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

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
            "reward": str(self.reward),  # Convert Decimal to string for JSON serialization
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
            reward=Decimal(block_data["reward"]),  # Convert string back to Decimal
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

    def is_valid(self):
        if self.hash != self.compute_hash():
            print(f"Block hash {self.hash} is invalid.")
            return False

        for transaction in self.transactions:
            if not transaction.is_valid():
                print(f"Transaction {transaction.id} in block is invalid.")
                return False

        return True