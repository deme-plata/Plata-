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
import os
import base64
import uuid
import hashlib
import logging
import time
from typing import Optional, Tuple, Any, List
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
from ecdsa import SigningKey, SECP256k1
import base64
import traceback
from typing import Optional, Union, Tuple

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
    zk_proof: Optional[Union[Tuple, str]] = None  # Can be either tuple or string
    homomorphic_amount: Optional[bytes] = None
    ring_signature: Optional[bytes] = None
    quantum_signature: Optional[bytes] = None
    pq_cipher: Optional[bytes] = None
    wallet: Optional[Any] = None
    tx_hash: str = Field(default="")  # Initialize with empty string
    timestamp: float = Field(default_factory=time.time)  # Set default to current time
    private_key: Optional[SigningKey] = None  # For ECDSA private key
    def sign_transaction(self, zk_system) -> bool:
        """
        Sign the transaction with enhanced security features including ZK proofs
        
        Args:
            zk_system: Zero-knowledge proof system instance
            
        Returns:
            bool: True if signing successful, False otherwise
            
        Raises:
            ValueError: If wallet or required data is missing
        """
        try:
            # Validate prerequisites
            if not self.wallet:
                raise ValueError("No wallet available for signing")
                
            if not hasattr(self.wallet, 'sign_message') or not hasattr(self.wallet, 'public_key'):
                raise ValueError("Wallet missing required signing capabilities")
                
            if not zk_system:
                raise ValueError("ZK proof system not provided")
                
            # Create canonical message representation
            try:
                message = self._create_message()
                logger.debug(f"Created message for signing: {message.hex()}")
            except Exception as e:
                logger.error(f"Failed to create message: {str(e)}")
                raise ValueError(f"Message creation failed: {str(e)}")
                
            # Sign message with wallet
            try:
                self.signature = self.wallet.sign_message(message)
                self.public_key = self.wallet.public_key
                
                # Validate signature format
                if not isinstance(self.signature, (str, bytes)):
                    raise ValueError("Invalid signature format")
                    
                # Ensure signature is properly encoded
                if isinstance(self.signature, bytes):
                    self.signature = base64.b64encode(self.signature).decode('utf-8')
                    
                logger.debug(f"Message signed successfully. Signature: {self.signature[:32]}...")
            except Exception as e:
                logger.error(f"Signing failed: {str(e)}")
                raise ValueError(f"Failed to sign message: {str(e)}")
                
            # Generate and verify ZK proof
            try:
                # Convert amount to integer representation
                amount_wei = int(float(self.amount) * 10**18)
                if amount_wei <= 0:
                    raise ValueError("Invalid amount for ZK proof")
                    
                # Generate public input from transaction data
                public_input = int.from_bytes(
                    hashlib.sha256(message).digest(),
                    byteorder='big'
                )
                
                # Generate ZK proof
                self.zk_proof = zk_system.prove(amount_wei, public_input)
                
                # Verify the proof immediately
                if not zk_system.verify(public_input, self.zk_proof):
                    raise ValueError("Generated ZK proof verification failed")
                    
                logger.debug("ZK proof generated and verified successfully")
            except Exception as e:
                logger.error(f"ZK proof generation failed: {str(e)}")
                raise ValueError(f"Failed to generate ZK proof: {str(e)}")
                
            # Update transaction hash with all components
            try:
                # Combine all security elements
                security_data = message + base64.b64decode(self.signature)
                if self.zk_proof:
                    # Convert ZK proof to bytes if it's not already
                    zk_proof_bytes = (
                        self.zk_proof if isinstance(self.zk_proof, bytes)
                        else str(self.zk_proof).encode()
                    )
                    security_data += zk_proof_bytes
                    
                # Generate final hash
                self.tx_hash = hashlib.sha256(security_data).hexdigest()
                logger.debug(f"Generated transaction hash: {self.tx_hash}")
                
            except Exception as e:
                logger.error(f"Hash generation failed: {str(e)}")
                raise ValueError(f"Failed to generate transaction hash: {str(e)}")
                
            return True
                
        except Exception as e:
            logger.error(f"Transaction signing failed: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False



    async def verify_transaction(self, public_key_pem: str) -> dict:
        """Enhanced transaction verification with detailed validation results"""
        try:
            # Create validation result dictionary
            validation_result = {
                'status': 'success',
                'valid': True,
                'validation_details': {
                    'hash_valid': False,
                    'amount_valid': False,
                    'timestamp_valid': False,
                    'signature_valid': False
                }
            }
            
            # 1. Verify transaction hash
            computed_hash = self.generate_hash()
            validation_result['validation_details']['hash_valid'] = (computed_hash == self.tx_hash)
            
            # 2. Verify amount
            try:
                validation_result['validation_details']['amount_valid'] = (
                    self.amount > Decimal('0') and 
                    self.price >= Decimal('0')
                )
            except (TypeError, ValueError):
                validation_result['validation_details']['amount_valid'] = False
            
            # 3. Verify timestamp
            current_time = time.time()
            validation_result['validation_details']['timestamp_valid'] = (
                self.timestamp <= current_time and 
                self.timestamp > (current_time - 86400)  # Within last 24 hours
            )
            
            # 4. Verify signature
            if self.signature and self.public_key:
                try:
                    message = self._create_message()
                    signature_bytes = base64.b64decode(self.signature)
                    public_key_bytes = base64.b64decode(self.public_key)
                    
                    # Create public key object
                    public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                        ec.SECP256R1(),
                        public_key_bytes
                    )
                    
                    # Verify signature
                    public_key.verify(
                        signature_bytes,
                        message,
                        ec.ECDSA(hashes.SHA256())
                    )
                    validation_result['validation_details']['signature_valid'] = True
                except Exception as e:
                    logger.error(f"Signature verification failed: {str(e)}")
                    validation_result['validation_details']['signature_valid'] = False
            
            # Check if all validations passed
            all_valid = all(validation_result['validation_details'].values())
            if not all_valid:
                validation_result['status'] = 'error'
                validation_result['valid'] = False
                validation_result['message'] = 'Transaction verification failed'
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Transaction verification error: {str(e)}")
            return {
                'status': 'error',
                'valid': False,
                'message': f'Verification error: {str(e)}',
                'validation_details': {
                    'hash_valid': False,
                    'amount_valid': False,
                    'timestamp_valid': False,
                    'signature_valid': False
                }
            }


    def _create_message(self) -> bytes:
        """Create standardized message for signing"""
        message = (
            f"{self.id}{self.sender}{self.receiver}"
            f"{str(self.amount)}{str(self.price)}"
            f"{self.buyer_id}{self.seller_id}"
            f"{str(self.timestamp)}"
        ).encode('utf-8')
        return message




    def verify_signature(self):
        """Verify the transaction signature using ECDSA."""
        if not self.signature or not self.public_key:
            raise ValueError("Missing signature or public key for verification")

        # Create a verifying key from the stored public key
        verifying_key = SigningKey.from_string(bytes.fromhex(self.public_key), curve=SECP256k1).get_verifying_key()
        message = f"{self.sender}{self.receiver}{self.amount}{self.price}{self.buyer_id}{self.seller_id}".encode()

        # Verify the signature
        return verifying_key.verify(self.signature, message)



    def generate_hash(self) -> str:
        """Generate transaction hash"""
        message = self._create_message()
        security_data = message
        
        if self.signature:
            try:
                security_data += base64.b64decode(self.signature)
            except:
                security_data += self.signature.encode()

        if self.zk_proof:
            if isinstance(self.zk_proof, str):
                security_data += self.zk_proof.encode()
            else:
                security_data += str(self.zk_proof).encode()

        return hashlib.sha256(security_data).hexdigest()


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
    @validator('zk_proof', pre=True)
    def validate_zk_proof(cls, v):
        if isinstance(v, str):
            try:
                # If it's a base64 string, decode it to tuple
                decoded = base64.b64decode(v).decode()
                return eval(decoded)  # Convert string representation of tuple to actual tuple
            except:
                return v  # Keep as string if can't decode
        return v

    async def apply_enhanced_security(self, crypto_provider: Any) -> bool:
        """Apply enhanced security features with proper error handling and type conversion"""
        try:
            logger.debug("Starting enhanced security application")
            message = self._create_message()
            security_elements = []  # Store all security elements for hash computation
            
            # 1. Generate ZK proof
            try:
                amount_wei = int(float(self.amount) * 10**18)
                public_input = int.from_bytes(hashlib.sha256(message).digest(), byteorder='big')
                self.zk_proof = crypto_provider.stark.prove(amount_wei, public_input)
                security_elements.append(str(self.zk_proof).encode())
                logger.debug("ZK proof generated successfully")
            except Exception as e:
                logger.error(f"ZK proof generation failed: {str(e)}")
                return False

            # 2. Apply homomorphic encryption
            try:
                self.homomorphic_amount = await crypto_provider.create_homomorphic_cipher(
                    int(float(self.amount) * 10**18)
                )
                security_elements.append(self.homomorphic_amount)
                logger.debug("Homomorphic encryption applied successfully")
            except Exception as e:
                logger.error(f"Homomorphic encryption failed: {str(e)}")
                return False

            # 3. Generate ring signature with proper ring size
            try:
                if self.wallet and hasattr(self.wallet, 'private_key'):
                    # Create a ring of public keys (minimum 2)
                    ring_keys = [self.public_key] if self.public_key else []
                    # Add a dummy key if needed to meet minimum size
                    if len(ring_keys) < 2:
                        dummy_key = base64.b64encode(os.urandom(32)).decode('utf-8')
                        ring_keys.append(dummy_key)
                    
                    self.ring_signature = crypto_provider.create_ring_signature(
                        message,
                        self.wallet.private_key,
                        ring_keys
                    )
                    if self.ring_signature:
                        security_elements.append(self.ring_signature)
                    logger.debug("Ring signature created successfully")
            except Exception as e:
                logger.error(f"Ring signature creation failed: {str(e)}")
                # Continue even if ring signature fails

            # 4. Apply post-quantum encryption
            try:
                if not hasattr(crypto_provider, 'kem'):
                    # Initialize post-quantum components if needed
                    crypto_provider.kem = KeyEncapsulation("Kyber768")
                    crypto_provider.pq_public_key, crypto_provider.pq_secret_key = crypto_provider.kem.generate_keypair()
                
                self.pq_cipher = crypto_provider.pq_encrypt(message)
                if self.pq_cipher:
                    security_elements.append(self.pq_cipher)
                logger.debug("Post-quantum encryption applied successfully")
            except Exception as e:
                logger.error(f"Post-quantum encryption failed: {str(e)}")
                # Continue even if PQ encryption fails

            # 5. Generate base signature
            try:
                if not self.signature:
                    if not hasattr(self.wallet, 'private_key'):
                        private_key = ec.generate_private_key(ec.SECP256R1())
                        self.wallet.private_key = private_key
                        public_key = private_key.public_key()
                        self.wallet.public_key = base64.b64encode(
                            public_key.public_bytes(
                                encoding=serialization.Encoding.X962,
                                format=serialization.PublicFormat.UncompressedPoint
                            )
                        ).decode('utf-8')

                    signature = self.wallet.private_key.sign(
                        message,
                        ec.ECDSA(hashes.SHA256())
                    )
                    self.signature = base64.b64encode(signature).decode('utf-8')
                    self.public_key = self.wallet.public_key
                    security_elements.append(signature)
                    logger.debug("Base signature generated successfully")
            except Exception as e:
                logger.error(f"Base signature generation failed: {str(e)}")
                return False

            # Update transaction hash with all security elements
            try:
                # Start with the message
                hash_data = message
                
                # Add each security element, converting to bytes if needed
                for element in security_elements:
                    if isinstance(element, str):
                        hash_data += element.encode()
                    elif isinstance(element, bytes):
                        hash_data += element
                    elif isinstance(element, tuple):
                        hash_data += str(element).encode()
                    else:
                        hash_data += str(element).encode()
                
                self.tx_hash = hashlib.sha256(hash_data).hexdigest()
                logger.debug("Transaction hash updated successfully")
                return True
                
            except Exception as e:
                logger.error(f"Hash update failed: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        except Exception as e:
            logger.error(f"Enhanced security application failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False


    async def verify_enhanced_security(self, crypto_provider: Any) -> bool:
        """Verify all enhanced security features"""
        try:
            message = self._create_message()
            
            # 1. Verify ZK proof
            if self.zk_proof:
                public_input = int.from_bytes(hashlib.sha256(message).digest(), byteorder='big')
                if not crypto_provider.stark.verify(public_input, self.zk_proof):
                    logger.error("ZK proof verification failed")
                    return False

            # 2. Verify homomorphic amount
            if self.homomorphic_amount:
                decrypted_amount = await crypto_provider.decrypt_homomorphic(self.homomorphic_amount)
                expected_amount = int(float(self.amount) * 10**18)
                if abs(decrypted_amount - expected_amount) > 1000:  # Allow small precision differences
                    logger.error("Homomorphic amount verification failed")
                    return False

            # 3. Verify ring signature
            if self.ring_signature:
                if not crypto_provider.verify_ring_signature(
                    message, 
                    self.ring_signature,
                    [self.public_key] if self.public_key else []
                ):
                    logger.error("Ring signature verification failed")
                    return False

            # 4. Verify post-quantum encryption
            if self.pq_cipher:
                try:
                    decrypted_message = crypto_provider.pq_decrypt(self.pq_cipher)
                    if decrypted_message != message:
                        logger.error("Post-quantum encryption verification failed")
                        return False
                except Exception as e:
                    logger.error(f"Post-quantum decryption failed: {str(e)}")
                    return False

            # 5. Verify base signature
            if self.signature and self.public_key:
                try:
                    public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                        ec.SECP256R1(),
                        base64.b64decode(self.public_key)
                    )
                    public_key.verify(
                        base64.b64decode(self.signature),
                        message,
                        ec.ECDSA(hashes.SHA256())
                    )
                except Exception as e:
                    logger.error(f"Base signature verification failed: {str(e)}")
                    return False

            logger.info("All security features verified successfully")
            return True

        except Exception as e:
            logger.error(f"Security verification failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False


    def to_dict(self) -> dict:
        """Convert transaction to dictionary with proper encoding"""
        try:
            data = {
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

            # Handle ZK proof
            if self.zk_proof is not None:
                if isinstance(self.zk_proof, str):
                    data["zk_proof"] = self.zk_proof
                else:
                    data["zk_proof"] = base64.b64encode(
                        str(self.zk_proof).encode()
                    ).decode()

            # Handle binary data
            if self.homomorphic_amount is not None:
                data["homomorphic_amount"] = base64.b64encode(
                    self.homomorphic_amount
                ).decode()

            if self.ring_signature is not None:
                data["ring_signature"] = base64.b64encode(
                    self.ring_signature
                ).decode()

            if self.quantum_signature is not None:
                data["quantum_signature"] = base64.b64encode(
                    self.quantum_signature
                ).decode()

            if self.pq_cipher is not None:
                data["pq_cipher"] = base64.b64encode(
                    self.pq_cipher
                ).decode()

            return data

        except Exception as e:
            logger.error(f"Error converting transaction to dict: {str(e)}")
            raise ValueError(f"Failed to convert transaction to dictionary: {str(e)}")
    def get_security_features(self) -> dict:
        """Get the status of all security features"""
        return {
            "base_signature": bool(self.signature),
            "zk_proof": bool(self.zk_proof),
            "homomorphic": bool(self.homomorphic_amount),
            "ring_signature": bool(self.ring_signature),
            "quantum_signature": bool(self.quantum_signature),
            "post_quantum": bool(self.pq_cipher)
        }

    def _create_message(self) -> bytes:
        """Create standardized message for signing"""
        message = (
            f"{self.id}{self.sender}{self.receiver}"
            f"{str(self.amount)}{str(self.price)}"
            f"{self.buyer_id}{self.seller_id}"
            f"{str(self.timestamp)}"
        ).encode('utf-8')
        return message

    @classmethod
    def from_dict(cls, data: dict) -> 'Transaction':
        """Create a Transaction instance from a dictionary with proper type conversion"""
        try:
            decoded_data = data.copy()

            # Convert decimal strings
            if 'amount' in decoded_data:
                decoded_data['amount'] = Decimal(str(decoded_data['amount']))
            if 'price' in decoded_data:
                decoded_data['price'] = Decimal(str(decoded_data['price']))

            # Handle binary data
            if 'homomorphic_amount' in decoded_data and decoded_data['homomorphic_amount']:
                if isinstance(decoded_data['homomorphic_amount'], str):
                    decoded_data['homomorphic_amount'] = base64.b64decode(
                        decoded_data['homomorphic_amount']
                    )

            # Handle ring signature
            if 'ring_signature' in decoded_data and decoded_data['ring_signature']:
                if isinstance(decoded_data['ring_signature'], str):
                    decoded_data['ring_signature'] = base64.b64decode(
                        decoded_data['ring_signature']
                    )

            # Handle quantum signature
            if 'quantum_signature' in decoded_data and decoded_data['quantum_signature']:
                if isinstance(decoded_data['quantum_signature'], str):
                    decoded_data['quantum_signature'] = base64.b64decode(
                        decoded_data['quantum_signature']
                    )

            # Handle pq cipher
            if 'pq_cipher' in decoded_data and decoded_data['pq_cipher']:
                if isinstance(decoded_data['pq_cipher'], str):
                    decoded_data['pq_cipher'] = base64.b64decode(
                        decoded_data['pq_cipher']
                    )

            # Handle ZK proof
            if 'zk_proof' in decoded_data and decoded_data['zk_proof']:
                if isinstance(decoded_data['zk_proof'], str):
                    try:
                        decoded = base64.b64decode(decoded_data['zk_proof']).decode()
                        decoded_data['zk_proof'] = eval(decoded)  # Convert to tuple
                    except:
                        # Keep as string if conversion fails
                        pass

            return cls(**decoded_data)

        except Exception as e:
            logger.error(f"Error creating transaction from dict: {str(e)}")
            raise ValueError(f"Failed to create transaction from dictionary: {str(e)}")



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