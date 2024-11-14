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
from typing import Optional, Any
from quantum_signer import QuantumSigner
from DAGConfirmationSystem import * 
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from decimal import Decimal
import base64
import json
import time
import logging
import uuid
import hashlib
import traceback
from pydantic import root_validator
from cryptography.hazmat.primitives.asymmetric import ec

# Third-party imports
from pydantic import BaseModel, Field
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature
from ecdsa import SigningKey, SECP256k1
from typing import Optional, Dict, Union, Tuple, Any
from pydantic import BaseModel, Field, validator as model_validator
from decimal import Decimal
import time
import uuid
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from decimal import Decimal
import time
import logging
import traceback
from pydantic import BaseModel, Field, ConfigDict, root_validator
from typing import List, Set, Dict, Any, Optional, Union
from decimal import Decimal
import time
import uuid
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from decimal import Decimal
import base64
import json
import time
import logging
import uuid
import hashlib
import traceback
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    validator,  # Use validator instead of field_validator
    root_validator
)
from confirmation_models import ConfirmationStatus, ConfirmationMetrics, ConfirmationData

logger = logging.getLogger(__name__)
class SecurityFeatures(BaseModel):
    """Security features status for transactions"""
    zk_proof: bool = Field(default=False)
    homomorphic: bool = Field(default=False)
    ring_signature: bool = Field(default=False)
    quantum_signature: bool = Field(default=False)
    post_quantum: bool = Field(default=False)
    base_signature: bool = Field(default=False)

    model_config = ConfigDict(frozen=False)

class ValidationMetrics(BaseModel):
    """Validation metrics for transaction security"""
    strength_score: float = Field(default=0.0)
    complexity_score: float = Field(default=0.0)
    time_valid: bool = Field(default=True)
    signature_valid: bool = Field(default=False)
    amount_valid: bool = Field(default=False)

    model_config = ConfigDict(frozen=False)

class ValidationFeatures(BaseModel):
    """Complete validation features for transactions"""
    metrics: ValidationMetrics = Field(default_factory=ValidationMetrics)
    status: str = Field(default="PENDING")
    timestamp: float = Field(default_factory=time.time)
    validation_count: int = Field(default=0)
    last_validation: Optional[float] = None
    
    model_config = ConfigDict(frozen=False)

class ConfirmationStatus(BaseModel):
    """Status information for transaction confirmations"""
    score: float = Field(default=0.0)
    security_level: str = Field(default="LOW")
    confirmations: int = Field(default=0)
    is_final: bool = Field(default=False)
    
    model_config = ConfigDict(frozen=False)

class ConfirmationMetrics(BaseModel):
    """Metrics for transaction confirmation"""
    path_diversity: float = Field(default=0.0)
    quantum_strength: float = Field(default=0.85)
    consensus_weight: float = Field(default=0.0)
    depth_score: float = Field(default=0.0)
    last_updated: float = Field(default_factory=time.time)
    
    model_config = ConfigDict(frozen=False)

class ConfirmationPaths(BaseModel):
    """Path information for transaction confirmation"""
    confirmation_paths: List[str] = Field(default_factory=list)
    confirming_blocks: set[str] = Field(default_factory=set)
    quantum_confirmations: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(frozen=False)
class ConfirmationData(BaseModel):
    """Complete confirmation data for transactions"""
    status: ConfirmationStatus = Field(default_factory=ConfirmationStatus)
    metrics: ConfirmationMetrics = Field(default_factory=ConfirmationMetrics)
    paths: ConfirmationPaths = Field(default_factory=ConfirmationPaths)

    @property
    def confirmation_paths(self) -> List[List[str]]:
        """Compatibility property for accessing confirmation paths"""
        return self.paths.paths

    @property
    def confirming_blocks(self) -> Set[str]:
        """Compatibility property for accessing confirming blocks"""
        return self.paths.blocks
        
    @property
    def quantum_confirmations(self) -> Dict[str, float]:
        """Compatibility property for accessing quantum confirmations"""
        return self.paths.quantum_confirmations

    model_config = ConfigDict(frozen=False)



class Transaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    amount: Decimal
    price: Optional[Decimal] = Decimal('0')
    buyer_id: Optional[str] = None
    seller_id: Optional[str] = None
    gas_limit: Optional[int] = None
    gas_price: Optional[Decimal] = None
    quantum_enabled: bool = False
    timestamp: Optional[int] = None
    gas_data: Optional[Dict] = None
    wallet: Optional[Any] = Field(default=None, exclude=True)

    # Fields related to cryptographic and confirmation features
    zk_proof: Optional[bytes] = None
    signature: Optional[str] = None
    tx_hash: Optional[str] = None
    public_key: Optional[str] = None
    confirmations: int = Field(default=0)
    confirmation_data: ConfirmationData = Field(default_factory=ConfirmationData)
    homomorphic_amount: Optional[bytes] = None
    quantum_signature: Optional[bytes] = None
    pq_cipher: Optional[bytes] = None
    ring_signature: Optional[bytes] = None
    gas_used: int = Field(default=0)
    security_features: SecurityFeatures = Field(default_factory=SecurityFeatures)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        populate_by_name=True,
        json_encoders={
            Decimal: str,
            bytes: lambda v: base64.b64encode(v).decode('utf-8') if v else None
        }
    )

    @validator('*', pre=True)  # Using validator instead of field_validator
    def set_defaults(cls, v, values):
        """Set default values for fields"""
        # Set buyer_id and seller_id defaults
        if not values.get('buyer_id') and 'receiver' in values:
            values['buyer_id'] = values['receiver']
        if not values.get('seller_id') and 'sender' in values:
            values['seller_id'] = values['sender']
            
        # Set timestamp default
        if not values.get('timestamp'):
            values['timestamp'] = int(time.time() * 1000)
            
        # Initialize confirmation_data if not present
        if not values.get('confirmation_data'):
            values['confirmation_data'] = ConfirmationData(
                status=ConfirmationStatus(
                    score=0.0,
                    security_level="LOW",
                    confirmations=0,
                    is_final=False
                ),
                metrics=ConfirmationMetrics(
                    path_diversity=0.0,
                    quantum_strength=0.85,
                    consensus_weight=0.0,
                    depth_score=0.0
                )
            )
            
        return v




    @property
    def confirmation_score(self) -> float:
        """Get confirmation score from confirmation data"""
        return self.confirmation_data.status.score if self.confirmation_data else 0.0
    @property
    def security_level(self) -> str:
        """Get security level from confirmation data"""
        return self.confirmation_data.status.security_level if self.confirmation_data else "LOW"



    def update_confirmation_status(self, security_info: Dict[str, Any]) -> None:
        """Update transaction's confirmation status properly"""
        try:
            # Create new status with updated values
            new_status = ConfirmationStatus(
                score=security_info.get('confirmation_score', 0.0),
                security_level=security_info.get('security_level', 'LOW'),
                confirmations=security_info.get('num_confirmations', 0),
                is_final=security_info.get('is_final', False)
            )

            # Create new metrics with updated values
            new_metrics = ConfirmationMetrics(
                path_diversity=security_info.get('path_diversity', 0.0),
                quantum_strength=security_info.get('quantum_strength', 0.85),
                consensus_weight=security_info.get('consensus_weight', 0.0),
                depth_score=security_info.get('depth_score', 0.0),
                last_updated=time.time()
            )

            # Create new confirmation data preserving existing paths data
            new_confirmation_data = ConfirmationData(
                status=new_status,
                metrics=new_metrics,
                confirming_blocks=self.confirmation_data.confirming_blocks,
                confirmation_paths=self.confirmation_data.confirmation_paths,
                quantum_confirmations=self.confirmation_data.quantum_confirmations
            )

            # Update using model_copy to ensure Pydantic validation
            updated_transaction = self.model_copy(
                update={
                    'confirmation_data': new_confirmation_data,
                    'confirmations': security_info.get('num_confirmations', 0)
                }
            )

            # Copy updated values back to self
            self.confirmation_data = updated_transaction.confirmation_data
            self.confirmations = updated_transaction.confirmations

        except Exception as e:
            logger.error(f"Error updating confirmation status: {str(e)}")
            logger.error(traceback.format_exc())

    def get_confirmation_status(self) -> Dict[str, Any]:
        """Get current confirmation status"""
        try:
            return {
                'confirmation_score': self.confirmation_data.status.score,
                'security_level': self.confirmation_data.status.security_level,
                'confirmations': self.confirmations,
                'is_final': self.confirmation_data.status.is_final,
                'metrics': {
                    'path_diversity': self.confirmation_data.metrics.path_diversity,
                    'quantum_strength': self.confirmation_data.metrics.quantum_strength,
                    'consensus_weight': self.confirmation_data.metrics.consensus_weight,
                    'depth_score': self.confirmation_data.metrics.depth_score
                }
            }
        except Exception as e:
            logger.error(f"Error getting confirmation status: {str(e)}")
            return {
                'confirmation_score': 0.0,
                'security_level': 'LOW',
                'confirmations': 0,
                'is_final': False,
                'metrics': {
                    'path_diversity': 0.0,
                    'quantum_strength': 0.85,
                    'consensus_weight': 0.0,
                    'depth_score': 0.0
                }
            }


    def is_securely_confirmed(self) -> bool:
        """Check if transaction has reached a secure confirmation state"""
        return (
            self.confirmation_data.status.security_level in ['MAXIMUM', 'VERY_HIGH', 'HIGH'] and
            self.confirmations >= 6 and
            self.confirmation_data.status.confirmation_score >= 0.95
        )

    def compute_hash(self) -> str:
        """Compute transaction hash"""
        data = f"{self.sender}{self.receiver}{self.amount}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()



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
    async def apply_quantum_signature(self, quantum_signer: Any) -> bool:
        """Apply quantum signature to transaction with async support"""
        try:
            # Create message to sign
            message = self._create_message()
            
            # Apply quantum signature
            self.quantum_signature = quantum_signer.sign_message(message)
            
            if self.quantum_signature:
                # Initialize confirmation data if needed
                if not hasattr(self, 'confirmation_data'):
                    self.confirmation_data = ConfirmationData(
                        status=ConfirmationStatus(
                            score=0.0,
                            security_level="LOW",
                            confirmations=0,
                            is_final=False
                        ),
                        metrics=ConfirmationMetrics(
                            path_diversity=0.0,
                            quantum_strength=0.85,
                            consensus_weight=0.0,  
                            depth_score=0.0,
                            last_updated=time.time()
                        ),
                        paths=ConfirmationPaths(
                            confirmation_paths=[],
                            confirming_blocks=set(),
                            quantum_confirmations={}
                        )
                    )
                else:
                    # Update existing confirmation data
                    self.confirmation_data.status.score = 0.0
                    self.confirmation_data.status.security_level = "LOW"
                    self.confirmation_data.status.confirmations = 0
                    self.confirmation_data.status.is_final = False
                    
                    self.confirmation_data.metrics.path_diversity = 0.0
                    self.confirmation_data.metrics.quantum_strength = 0.85
                    self.confirmation_data.metrics.consensus_weight = 0.0
                    self.confirmation_data.metrics.depth_score = 0.0
                    self.confirmation_data.metrics.last_updated = time.time()

                # Reset confirmations counter
                self.confirmations = 0

                # Update security features
                if hasattr(self, 'security_features'):
                    self.security_features.quantum_signature = True

                logger.debug(f"""
                Applied quantum signature:
                - Size: {len(self.quantum_signature)} bytes
                - Message hash: {hashlib.sha256(message).hexdigest()[:16]}...
                - Security level: {self.confirmation_data.status.security_level}
                - Quantum strength: {self.confirmation_data.metrics.quantum_strength}
                """)
                
                # Allow for any async cleanup/verification
                await asyncio.sleep(0)
                return True

            logger.error("Failed to generate quantum signature")
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply quantum signature: {str(e)}")
            logger.error(traceback.format_exc())
            return False





    def verify_quantum_signature(self, quantum_signer: QuantumSigner) -> bool:
        """Verify quantum signature on transaction"""
        try:
            if not self.quantum_signature:
                logger.error("No quantum signature found")
                return False
                
            message = self._create_message()
            result = quantum_signer.verify_signature(message, self.quantum_signature)
            
            if result:
                logger.debug("Quantum signature verified successfully")
            else:
                logger.error("Quantum signature verification failed")
            return result
        except Exception as e:
            logger.error(f"Quantum signature verification error: {str(e)}")
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


    def get_security_level(self) -> str:
        """Get current security level"""
        try:
            if hasattr(self, 'confirmation_data') and self.confirmation_data:
                return self.confirmation_data.status.security_level
            return "LOW"
        except Exception as e:
            logger.error(f"Error getting security level: {str(e)}")
            return "LOW"







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
    @validator('zk_proof', check_fields=False)
    def validate_zk_proof(cls, v):
        """
        Validate zk_proof field to ensure it conforms to the expected encoding or structure.
        """
        if isinstance(v, str):
            try:
                # If it's a base64 string, decode it
                base64.b64decode(v)  # Just to check for valid base64 encoding
            except Exception as e:
                raise ValueError(f"zk_proof must be a valid base64 encoded string. Error: {e}")
        elif isinstance(v, tuple):
            # Additional tuple validation can go here if necessary
            pass
        return v

    async def apply_enhanced_security(self, crypto_provider: Any) -> bool:
        """Apply enhanced security features with comprehensive quantum support"""
        try:
            logger.debug("Starting enhanced security application")
            message = self._create_message()
            security_elements = []  # Store all security elements for hash computation

            # Store current quantum signature state
            current_quantum = {
                'signature': self.quantum_signature,
                'signature_flag': self.security_features.quantum_signature
            }

            # Reset security features while preserving quantum state
            self.security_features = SecurityFeatures()
            if current_quantum['signature_flag'] and current_quantum['signature']:
                self.quantum_signature = current_quantum['signature']
                self.security_features.quantum_signature = True
                security_elements.append(self.quantum_signature)
                logger.debug("Preserved existing quantum signature")

            # 1. Handle ZK proof
            try:
                amount_wei = int(float(self.amount) * 10**18)
                public_input = int.from_bytes(hashlib.sha256(message).digest(), byteorder='big')
                
                if self.zk_proof:
                    # Use existing proof but verify it
                    proof_bytes = self.zk_proof
                    if isinstance(proof_bytes, str):
                        proof_bytes = proof_bytes.encode()
                    # Verify the proof directly without converting to string
                    if crypto_provider.stark.verify(public_input, eval(proof_bytes.decode())):
                        security_elements.append(proof_bytes)
                        self.security_features.zk_proof = True
                        logger.debug("Existing ZK proof verified successfully")
                    else:
                        logger.error("Existing ZK proof verification failed")
                        return False
                else:
                    # Generate new proof
                    new_proof = crypto_provider.stark.prove(amount_wei, public_input)
                    self.zk_proof = str(new_proof).encode()
                    security_elements.append(self.zk_proof)
                    self.security_features.zk_proof = True
                    logger.debug("New ZK proof generated successfully")
                    
            except Exception as e:
                logger.error(f"ZK proof processing failed: {str(e)}")
                logger.error(traceback.format_exc())
                return False

            # 2. Handle homomorphic encryption
            try:
                if self.homomorphic_amount:
                    # Verify existing homomorphic amount
                    if hasattr(crypto_provider, 'decrypt_homomorphic'):
                        decrypted = await crypto_provider.decrypt_homomorphic(self.homomorphic_amount)
                        if abs(decrypted - amount_wei) <= amount_wei * 0.001:
                            security_elements.append(self.homomorphic_amount)
                            self.security_features.homomorphic = True
                            logger.debug("Existing homomorphic encryption verified")
                        else:
                            logger.error(f"Homomorphic verification failed: {decrypted} vs {amount_wei}")
                            return False
                    else:
                        security_elements.append(self.homomorphic_amount)
                        self.security_features.homomorphic = True
                        logger.debug("Existing homomorphic amount trusted (no verification available)")
                else:
                    # Generate new homomorphic encryption
                    cipher = await crypto_provider.create_homomorphic_cipher(amount_wei)
                    if cipher:
                        self.homomorphic_amount = cipher
                        security_elements.append(cipher)
                        self.security_features.homomorphic = True
                        logger.debug("New homomorphic encryption created successfully")
                    else:
                        logger.error("Failed to create homomorphic encryption")
                        return False
                        
            except Exception as e:
                logger.error(f"Homomorphic encryption failed: {str(e)}")
                logger.error(traceback.format_exc())
                return False

            # 3. Handle base signature
            try:
                if self.signature and self.public_key:
                    # Verify existing signature
                    try:
                        public_key = serialization.load_pem_public_key(
                            self.public_key.encode(),
                            backend=default_backend()
                        )
                        signature_bytes = base64.b64decode(self.signature)
                        public_key.verify(
                            signature_bytes,
                            message,
                            ec.ECDSA(hashes.SHA256())
                        )
                        security_elements.append(signature_bytes)
                        self.security_features.base_signature = True
                        logger.debug("Existing signature verified successfully")
                    except Exception as e:
                        logger.error(f"Signature verification failed: {str(e)}")
                        return False
                else:
                    # Generate new signature
                    try:
                        if hasattr(self, 'wallet') and self.wallet and hasattr(self.wallet, 'private_key'):
                            private_key = self.wallet.private_key
                        else:
                            private_key = crypto_provider.get_signing_key()
                            
                        signature = private_key.sign(
                            message,
                            ec.ECDSA(hashes.SHA256())
                        )
                        self.signature = base64.b64encode(signature).decode('utf-8')
                        security_elements.append(signature)
                        self.security_features.base_signature = True
                        logger.debug("New signature generated successfully")
                    except Exception as e:
                        logger.error(f"Signature generation failed: {str(e)}")
                        return False

            except Exception as e:
                logger.error(f"Base signature processing failed: {str(e)}")
                logger.error(traceback.format_exc())
                return False

            # 4. Update transaction hash and verify all required features
            try:
                # Generate final hash
                hash_data = message
                for element in security_elements:
                    if isinstance(element, (str, bytes)):
                        hash_data += element if isinstance(element, bytes) else element.encode()
                    else:
                        hash_data += str(element).encode()
                
                self.tx_hash = hashlib.sha256(hash_data).hexdigest()
                
                # Verify all required features are present
                required_features = {'zk_proof', 'homomorphic', 'base_signature'}
                if self.quantum_enabled:
                    required_features.add('quantum_signature')
                    
                applied_features = {
                    name for name, value in self.security_features.dict().items() 
                    if value and name in required_features
                }
                
                if not required_features.issubset(applied_features):
                    missing = required_features - applied_features
                    logger.error(f"Missing required security features: {missing}")
                    logger.debug(f"Applied features: {applied_features}")
                    logger.debug(f"Security features status: {self.security_features.dict()}")
                    return False
                
                logger.info("Security features applied and verified successfully")
                return True
                
            except Exception as e:
                logger.error(f"Final verification failed: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        except Exception as e:
            logger.error(f"Enhanced security application failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False


    async def verify_enhanced_security(self, crypto_provider: Any, confirmation_system: Optional[DAGConfirmationSystem] = None) -> Dict[str, Any]:
        """Verify all enhanced security features including confirmation status"""
        
        # Initialize verification results outside try block to avoid UnboundLocalError
        verification_results = {
            'overall_status': False,
            'features_verified': {
                'zk_proof': False,
                'homomorphic': False,
                'ring_signature': False,
                'quantum_signature': False,
                'post_quantum': False,
                'base_signature': False,
                'confirmation_status': False
            },
            'security_metrics': {
                'quantum_strength': 0.0,
                'confirmation_score': 0.0,
                'path_diversity': 0.0,
                'consensus_weight': 0.0
            },
            'timestamps': {
                'verification_time': time.time(),
                'last_confirmation': (
                    self.confirmation_data.status.last_update 
                    if hasattr(self.confirmation_data.status, 'last_update') 
                    else None
                )
            }
        }
        
        try:
            message = self._create_message()
            
            # 1. Enhanced ZK Proof Verification
            if self.zk_proof:
                try:
                    public_input = int.from_bytes(hashlib.sha256(message).digest(), byteorder='big')
                    zk_valid = crypto_provider.stark.verify(public_input, self.zk_proof)
                    verification_results['features_verified']['zk_proof'] = zk_valid
                    
                    if not zk_valid:
                        logger.error("ZK proof verification failed")
                        return verification_results
                        
                except Exception as e:
                    logger.error(f"ZK proof verification error: {str(e)}")
                    return verification_results

            # 2. Enhanced Homomorphic Amount Verification
            if self.homomorphic_amount:
                try:
                    decrypted_amount = await crypto_provider.decrypt_homomorphic(self.homomorphic_amount)
                    expected_amount = int(float(self.amount) * 10**18)
                    
                    # Use more sophisticated comparison with tolerance
                    tolerance = min(1000, expected_amount * 0.001)  # 0.1% or 1000 wei, whichever is smaller
                    amount_valid = abs(decrypted_amount - expected_amount) <= tolerance
                    
                    verification_results['features_verified']['homomorphic'] = amount_valid
                    
                    if not amount_valid:
                        logger.error(
                            f"Homomorphic amount verification failed. "
                            f"Expected: {expected_amount}, Got: {decrypted_amount}"
                        )
                        return verification_results
                        
                except Exception as e:
                    logger.error(f"Homomorphic verification error: {str(e)}")
                    return verification_results

            # 3. Enhanced Ring Signature Verification
            if self.ring_signature:
                try:
                    ring_valid = crypto_provider.verify_ring_signature(
                        message,
                        self.ring_signature,
                        [self.public_key] if self.public_key else []
                    )
                    verification_results['features_verified']['ring_signature'] = ring_valid
                    
                    if not ring_valid:
                        logger.error("Ring signature verification failed")
                        return verification_results
                        
                except Exception as e:
                    logger.error(f"Ring signature verification error: {str(e)}")
                    return verification_results

            # 4. Enhanced Post-Quantum Verification
            if self.pq_cipher:
                try:
                    decrypted_message = crypto_provider.pq_decrypt(self.pq_cipher)
                    pq_valid = decrypted_message == message
                    verification_results['features_verified']['post_quantum'] = pq_valid
                    
                    if not pq_valid:
                        logger.error("Post-quantum encryption verification failed")
                        return verification_results
                        
                except Exception as e:
                    logger.error(f"Post-quantum verification error: {str(e)}")
                    return verification_results

            # 5. Enhanced Base Signature Verification
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
                    verification_results['features_verified']['base_signature'] = True
                except Exception as e:
                    logger.error(f"Base signature verification error: {str(e)}")
                    return verification_results

            # Update security metrics based on confirmations
            if confirmation_system and hasattr(self.confirmation_data, 'metrics'):
                verification_results['security_metrics'].update({
                    'quantum_strength': float(self.confirmation_data.metrics.quantum_strength),
                    'confirmation_score': float(self.confirmation_data.status.score),
                    'path_diversity': float(self.confirmation_data.metrics.path_diversity),
                    'consensus_weight': float(self.confirmation_data.metrics.consensus_weight)
                })

            # Calculate overall validation status
            required_verified = all(
                verification_results['features_verified'][feature]
                for feature in ['zk_proof', 'base_signature']
            )
            
            optional_features = {
                'homomorphic': bool(self.homomorphic_amount),
                'ring_signature': bool(self.ring_signature),
                'quantum_signature': bool(self.quantum_signature),
                'post_quantum': bool(self.pq_cipher)
            }
            
            optional_verified = all(
                not required or verification_results['features_verified'][feature]
                for feature, required in optional_features.items()
            )

            verification_results['overall_status'] = required_verified and optional_verified

            # Log verification result
            if verification_results['overall_status']:
                logger.info(
                    f"Enhanced security verification successful for transaction {self.tx_hash}\n"
                    f"\tConfirmation Score: {verification_results['security_metrics']['confirmation_score']:.4f}\n"
                    f"\tQuantum Strength: {verification_results['security_metrics']['quantum_strength']:.4f}\n"
                    f"\tFeatures Verified: {sum(verification_results['features_verified'].values())}"
                )
            else:
                logger.error(f"Enhanced security verification failed for transaction {self.tx_hash}")

            return verification_results

        except Exception as e:
            logger.error(f"Security verification failed: {str(e)}")
            logger.error(traceback.format_exc())
            verification_results['error'] = str(e)
            return verification_results



    def to_dict(self) -> dict:
        """Convert transaction to dictionary with enhanced security features and confirmation data"""
        try:
            # Basic transaction data with validation
            data = {
                "id": self.id,
                "sender": self.sender,
                "receiver": self.receiver,
                "amount": str(self.amount),
                "price": str(self.price) if self.price else "0",
                "buyer_id": self.buyer_id,
                "seller_id": self.seller_id,
                "public_key": self.public_key,
                "signature": self.signature,
                "tx_hash": self.tx_hash,
                "timestamp": self.timestamp,
                "confirmations": self.confirmations,
                "gas_limit": self.gas_limit,
                "gas_price": str(self.gas_price) if self.gas_price else None,
                "gas_used": self.gas_used,
                "quantum_enabled": self.quantum_enabled
            }

            # Enhanced confirmation data
            data["confirmation_data"] = {
                "status": {
                    "confirmation_score": float(self.confirmation_data.status.score),
                    "security_level": self.confirmation_data.status.security_level,
                    "confirmations": self.confirmation_data.status.confirmations,
                    "is_final": self.confirmation_data.status.is_final
                },
                "metrics": {
                    "path_diversity": float(self.confirmation_data.metrics.path_diversity),
                    "quantum_strength": float(self.confirmation_data.metrics.quantum_strength),
                    "consensus_weight": float(self.confirmation_data.metrics.consensus_weight),
                    "depth_score": float(self.confirmation_data.metrics.depth_score),
                    "last_updated": self.confirmation_data.metrics.last_updated
                },
                "paths": {
                    "blocks": list(getattr(self.confirmation_data.paths, 'blocks', set())),
                    "quantum_confirmations": dict(getattr(self.confirmation_data.paths, 'quantum_confirmations', {}))
                }
            }

            # Handle binary fields manually
            binary_fields = {
                "zk_proof": self.zk_proof,
                "homomorphic_amount": self.homomorphic_amount,
                "ring_signature": self.ring_signature,
                "quantum_signature": self.quantum_signature,
                "pq_cipher": self.pq_cipher
            }

            # Custom encoding for binary fields
            for field_name, field_value in binary_fields.items():
                if field_value is not None:
                    try:
                        if isinstance(field_value, bytes):
                            data[field_name] = base64.b64encode(field_value).decode('utf-8')
                        elif isinstance(field_value, str):
                            # If it's already a base64 string, verify and keep it
                            try:
                                base64.b64decode(field_value)
                                data[field_name] = field_value
                            except:
                                # If not valid base64, encode it
                                data[field_name] = base64.b64encode(
                                    field_value.encode('utf-8')
                                ).decode('utf-8')
                        else:
                            # For other types, convert to string and encode
                            data[field_name] = base64.b64encode(
                                str(field_value).encode('utf-8')
                            ).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error encoding {field_name}: {str(e)}")
                        data[field_name] = None

            # Add security features status
            data["security_features"] = {
                "cryptographic": {
                    "zk_proof": bool(self.zk_proof),
                    "homomorphic": bool(self.homomorphic_amount),
                    "ring_signature": bool(self.ring_signature),
                    "quantum_signature": bool(self.quantum_signature),
                    "post_quantum": bool(self.pq_cipher),
                    "base_signature": bool(self.signature)
                },
                "confirmation": {
                    "has_confirmations": self.confirmations > 0,
                    "meets_minimum_depth": self.confirmations >= 6,
                    "is_quantum_verified": bool(self.quantum_signature),
                    # Safely check confirming blocks count instead of paths
                    "has_multiple_confirmations": len(getattr(self.confirmation_data.paths, 'blocks', set())) > 1
                },
                "validation": {
                    "timestamp_valid": abs(time.time() - (self.timestamp or 0)) < 86400,
                    "amount_valid": float(self.amount) > 0,
                    "signature_valid": bool(self.signature and self.public_key)
                }
            }

            # Add gas data if present
            if hasattr(self, 'gas_data') and self.gas_data:
                data["gas_data"] = {
                    str(k): (str(v) if isinstance(v, (Decimal, float)) else v)
                    for k, v in self.gas_data.items()
                }

            return data

        except Exception as e:
            logger.error(f"Error converting transaction to dict: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to convert transaction to dictionary: {str(e)}")

    def initialize_confirmation_data(self) -> None:
        """Initialize confirmation data with proper empty collections"""
        if not hasattr(self, 'confirmation_data') or self.confirmation_data is None:
            self.confirmation_data = ConfirmationData(
                status=ConfirmationStatus(
                    confirmation_score=0.0,
                    security_level="UNSAFE",
                    last_update=None,
                    is_final=False
                ),
                metrics=ConfirmationMetrics(
                    path_diversity=0.0,
                    quantum_strength=0.0,
                    consensus_weight=0.0,
                    depth_score=0.0
                ),
                paths=ConfirmationPaths(
                    confirmation_paths=[],
                    confirming_blocks=set(),
                    quantum_confirmations={}
                )
            )


    def update_confirmation_metrics(self, 
                                  path_diversity: Optional[float] = None,
                                  quantum_strength: Optional[float] = None,
                                  consensus_weight: Optional[float] = None,
                                  depth_score: Optional[float] = None) -> None:
        """Update confirmation metrics with new values"""
        if path_diversity is not None:
            self.confirmation_data.metrics.path_diversity = path_diversity
        if quantum_strength is not None:
            self.confirmation_data.metrics.quantum_strength = quantum_strength
        if consensus_weight is not None:
            self.confirmation_data.metrics.consensus_weight = consensus_weight
        if depth_score is not None:
            self.confirmation_data.metrics.depth_score = depth_score
        
        # Update status
        self.confirmation_data.status.last_update = time.time()
        self.confirmation_data.status.is_final = (
            self.confirmation_data.metrics.quantum_strength >= 0.95 and
            self.confirmation_data.metrics.path_diversity >= 0.8 and
            self.confirmations >= 6
        )


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
        """Create a Transaction instance from a dictionary"""
        try:
            decoded_data = data.copy()
            
            # Handle confirmation data
            confirmation_data = decoded_data.pop('confirmation_data', {})
            
            # Process status data
            status_data = confirmation_data.get('status', {})
            status = ConfirmationStatus(
                confirmation_score=float(status_data.get('confirmation_score', 0.0)),
                security_level=status_data.get('security_level', 'UNSAFE'),
                last_update=status_data.get('last_update'),
                is_final=status_data.get('is_final', False)
            )

            # Process metrics data
            metrics_data = confirmation_data.get('metrics', {})
            metrics = ConfirmationMetrics(
                path_diversity=float(metrics_data.get('path_diversity', 0.0)),
                quantum_strength=float(metrics_data.get('quantum_strength', 0.0)),
                consensus_weight=float(metrics_data.get('consensus_weight', 0.0)),
                depth_score=float(metrics_data.get('depth_score', 0.0))
            )

            # Process paths data - ensure proper collection types
            paths_data = confirmation_data.get('paths', {})
            paths = ConfirmationPaths(
                confirmation_paths=list(paths_data.get('confirmation_paths', [])) if isinstance(paths_data.get('confirmation_paths'), (list, set, tuple)) else [],
                confirming_blocks=set(paths_data.get('confirming_blocks', [])) if isinstance(paths_data.get('confirming_blocks'), (list, set, tuple)) else set(),
                quantum_confirmations=dict(paths_data.get('quantum_confirmations', {})) if isinstance(paths_data.get('quantum_confirmations'), dict) else {}
            )

            # Create complete confirmation data
            decoded_data['confirmation_data'] = ConfirmationData(
                status=status,
                metrics=metrics,
                paths=paths
            )

            # Process decimal values
            for decimal_field in ['amount', 'price']:
                if decimal_field in decoded_data:
                    try:
                        value = str(decoded_data[decimal_field])
                        decoded_data[decimal_field] = Decimal(value)
                    except Exception as e:
                        logger.error(f"Error converting {decimal_field} to Decimal: {str(e)}")
                        raise ValueError(f"Invalid {decimal_field} format")

            # Process binary fields
            binary_fields = {
                'homomorphic_amount': {'required': False, 'max_size': 1024},
                'ring_signature': {'required': False, 'max_size': 2048},
                'quantum_signature': {'required': False, 'max_size': 1024},
                'pq_cipher': {'required': False, 'max_size': 4096}
            }

            for field, constraints in binary_fields.items():
                if field in decoded_data and decoded_data[field]:
                    try:
                        if isinstance(decoded_data[field], str):
                            decoded_value = base64.b64decode(decoded_data[field])
                        elif isinstance(decoded_data[field], bytes):
                            decoded_value = decoded_data[field]
                        else:
                            decoded_value = base64.b64decode(
                                base64.b64encode(str(decoded_data[field]).encode())
                            )
                        
                        if len(decoded_value) > constraints['max_size']:
                            raise ValueError(f"{field} exceeds maximum size of {constraints['max_size']} bytes")
                        
                        decoded_data[field] = decoded_value
                        
                    except Exception as e:
                        logger.error(f"Error decoding {field}: {str(e)}")
                        if constraints['required']:
                            raise ValueError(f"Required field {field} failed to decode")
                        decoded_data[field] = None

            # Create instance with validated data
            instance = cls(**decoded_data)
            
            # Validate the created instance
            try:
                instance.validate_transaction()
            except Exception as e:
                logger.error(f"Transaction validation failed: {str(e)}")
                raise ValueError(f"Invalid transaction data: {str(e)}")

            return instance

        except Exception as e:
            logger.error(f"Error creating transaction from dict: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Failed to create transaction from dictionary: {str(e)}")


    def validate_transaction(self) -> bool:
        """Validate transaction data"""
        try:
            # Validate basic fields
            if not self.id or not self.sender or not self.receiver:
                raise ValueError("Missing required fields")

            # Validate amounts
            if self.amount <= 0 or self.price < 0:
                raise ValueError("Invalid amount or price")

            # Validate confirmation data
            if not isinstance(self.confirmation_data, ConfirmationData):
                raise ValueError("Invalid confirmation data type")

            # Validate paths collections
            if not isinstance(self.confirmation_data.paths.confirmation_paths, list):
                self.confirmation_data.paths.confirmation_paths = []
                
            if not isinstance(self.confirmation_data.paths.confirming_blocks, set):
                self.confirmation_data.paths.confirming_blocks = set()
                
            if not isinstance(self.confirmation_data.paths.quantum_confirmations, dict):
                self.confirmation_data.paths.quantum_confirmations = {}

            return True

        except Exception as e:
            logger.error(f"Transaction validation error: {str(e)}")
            raise ValueError(f"Transaction validation failed: {str(e)}")

    @classmethod
    def default_confirmation_data(cls) -> dict:
        """Provide default confirmation data structure"""
        return {
            'confirmation_score': 0.0,
            'security_level': 'UNSAFE',
            'last_confirmation_update': None,
            'confirmation_metrics': {
                'path_diversity': 0.0,
                'quantum_strength': 0.0,
                'consensus_weight': 0.0,
                'depth_score': 0.0
            },
            'confirmation_paths': [],
            'confirming_blocks': set(),
            'quantum_confirmations': {}
        }


    

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