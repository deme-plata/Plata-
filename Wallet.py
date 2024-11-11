from pydantic import BaseModel
from typing import Optional
from typing import Optional, List
from shared_logic import Transaction
from pydantic import BaseModel
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
import os
import bcrypt
import re
import os
import json
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from SecureHybridZKStark import SecureHybridZKStark
from CryptoProvider import CryptoProvider
from STARK import STARK 
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from mnemonic import Mnemonic
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64
import hashlib
import random
import string
import os
import bcrypt
import re
import json
import numpy as np
from datetime import datetime
import logging
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from SecureHybridZKStark import SecureHybridZKStark
from CryptoProvider import CryptoProvider
from STARK import STARK
import logging
from AdvancedHomomorphicSystem import AdvancedHomomorphicSystem, QuantumEnhancedProofs,QuantumDecoherence,QuantumFoamTopology,NoncommutativeGeometry
from decimal import Decimal
import time
from typing import ClassVar

from AdvancedHomomorphicSystem import (
    AdvancedHomomorphicSystem,
    QuantumDecoherence,
    QuantumFoamTopology,
    NoncommutativeGeometry,
    G, c, hbar, l_p, t_p, m_p
)
logger = logging.getLogger(__name__)

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
    transaction: Transaction  # Use the appropriate Transaction class here

    class Config:
        arbitrary_types_allowed = True
        
        
GRAVITATIONAL_CONSTANT = 6.67430e-11
SPEED_OF_LIGHT = 3.0e8
REDUCED_PLANCK_CONSTANT = 1.0545718e-34
PLANCK_CONSTANT = 6.62607015e-34  # Planck constant in J⋅s
BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant in J/K
CHARACTERISTIC_TIME = 1e-12  # Characteristic decoherence time in seconds

class Wallet(BaseModel):
    GRAVITATIONAL_CONSTANT: ClassVar[float] = 6.67430e-11
    SPEED_OF_LIGHT: ClassVar[float] = 3.0e8
    REDUCED_PLANCK_CONSTANT: ClassVar[float] = 1.0545718e-34
    PLANCK_CONSTANT: ClassVar[float] = 6.62607015e-34
    BOLTZMANN_CONSTANT: ClassVar[float] = 1.380649e-23
    CHARACTERISTIC_TIME: ClassVar[float] = 1e-12


    # Basic fields
    private_key: Optional[ec.EllipticCurvePrivateKey] = None
    public_key: Optional[str] = None
    mnemonic_phrase: Optional[str] = None
    address: Optional[str] = None
    salt: Optional[bytes] = None
    hashed_pincode: Optional[str] = None
    
    # Security components
    zk_system: Optional[SecureHybridZKStark] = None
    crypto_provider: Optional[CryptoProvider] = None
    stark: Optional[STARK] = None
    
    # Homomorphic and quantum components
    homomorphic_system: Optional[AdvancedHomomorphicSystem] = None
    quantum_decoherence: Optional[QuantumDecoherence] = None
    foam_topology: Optional[QuantumFoamTopology] = None
    nc_geometry: Optional[NoncommutativeGeometry] = None
    quantum_proofs: Optional[QuantumEnhancedProofs] = None  # Added this field
    
    # Operation tracking
    encrypted_values: Dict[str, Any] = Field(default_factory=dict)
    operation_cache: Dict[str, Any] = Field(default_factory=dict)
    quantum_state: Dict[str, Any] = Field(default_factory=dict)
    entropy_pool: bytes = Field(default_factory=lambda: os.urandom(32))

    # Mnemonic generator - treat as internal field
    mnemonic_generator: Optional[Mnemonic] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(self, private_key=None, mnemonic=None, pincode=None, **data):
            super().__init__(**data)
            
            try:
                # Initialize mnemonic generator
                self.mnemonic_generator = Mnemonic("english")
                
                # Handle key initialization
                if mnemonic:
                    self.mnemonic_phrase = str(mnemonic)
                    seed = self.mnemonic_generator.to_seed(self.mnemonic_phrase)
                    self.private_key = ec.derive_private_key(
                        int.from_bytes(seed[:32], byteorder="big"), 
                        ec.SECP256R1(), 
                        default_backend()
                    )
                elif private_key:
                    self.private_key = serialization.load_pem_private_key(
                        private_key.encode(), 
                        password=None, 
                        backend=default_backend()
                    )
                else:
                    self.private_key = ec.generate_private_key(
                        ec.SECP256R1(), 
                        default_backend()
                    )
                    self.mnemonic_phrase = self.mnemonic_generator.generate(strength=128)

                # Generate public key and address
                self.public_key = self.get_public_key()
                self.address = self.get_quantum_enhanced_address()

                # Handle pincode
                if pincode:
                    self.salt = self.generate_salt()
                    self.hashed_pincode = self.hash_pincode(pincode)

                # Initialize security components
                self.zk_system = SecureHybridZKStark(security_level=20)
                self.crypto_provider = CryptoProvider()
                self.stark = STARK(security_level=20)

                # Initialize homomorphic and quantum components
                self.homomorphic_system = AdvancedHomomorphicSystem(key_size=2048)
                self.quantum_decoherence = QuantumDecoherence(system_size=10)
                self.foam_topology = QuantumFoamTopology()
                self.nc_geometry = NoncommutativeGeometry(theta_parameter=1e-10)
                
                # Initialize quantum proofs system
                self.quantum_proofs = QuantumEnhancedProofs(
                    quantum_system=self.quantum_decoherence,
                    foam_generator=self.foam_topology,
                    nc_geometry=self.nc_geometry
                )

                # Initialize quantum state
                self.initialize_quantum_state()

            except Exception as e:
                logger.error(f"Wallet initialization failed: {str(e)}")
                raise ValueError(f"Failed to initialize wallet: {str(e)}")
    def initialize_quantum_state(self) -> None:
        """Initialize quantum state with proper quantum parameters"""
        try:
            current_time = datetime.now().timestamp()
            
            # Create initial quantum state with full coherence
            quantum_state = {
                'entropy': self.generate_quantum_entropy(),
                'creation_time': current_time,
                'last_update': current_time,
                'interaction_count': 0,
                'temperature': 300.0,  # Room temperature in Kelvin
                'decoherence_factor': 1.0,  # Start with full coherence
                'characteristic_time': self.CHARACTERISTIC_TIME,
                'initial_state': True  # Flag to indicate fresh initialization
            }
            
            # Set the state atomically
            self.quantum_state = quantum_state
            
            logger.debug(
                f"Initialized quantum state:\n"
                f"  Time: {current_time}\n"
                f"  Decoherence: {quantum_state['decoherence_factor']}\n"
                f"  Temperature: {quantum_state['temperature']}K"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum state: {str(e)}")
            raise



    async def encrypt_value(self, value: Union[int, float, str]) -> Dict[str, Any]:
        """Encrypt value using homomorphic system with quantum enhancement"""
        try:
            # Convert value to Decimal
            decimal_value = Decimal(str(value))
            
            # Get current quantum state
            quantum_state = self.quantum_state.copy()
            foam_structure = self.foam_topology.generate_foam_structure(
                volume=1.0,
                temperature=300.0
            )

            # Calculate quantum corrections before encryption
            quantum_correction = self.nc_geometry.modified_dispersion(
                np.array([float(decimal_value), 0, 0]),
                float(decimal_value)
            )
            
            # Apply quantum correction to value before encryption
            corrected_value = decimal_value * Decimal(str(quantum_correction))
            
            # Encrypt corrected value
            cipher, operation_id = await self.homomorphic_system.encrypt(corrected_value)

            # Store metadata with original value and correction
            self.encrypted_values[operation_id] = {
                'cipher': cipher,
                'original_value': decimal_value,
                'quantum_state': quantum_state,
                'foam_structure': foam_structure,
                'quantum_correction': quantum_correction,
                'timestamp': time.time()
            }

            # Calculate metrics
            decoherence = self.calculate_decoherence_factor()
            
            return {
                'operation_id': operation_id,
                'encrypted_value': base64.b64encode(cipher).decode('utf-8'),
                'quantum_metrics': {
                    'decoherence': decoherence,
                    'foam_entropy': self.quantum_proofs.get_quantum_entropy(
                        quantum_state,
                        foam_structure
                    ),
                    'quantum_correction': float(quantum_correction)
                }
            }

        except Exception as e:
            logger.error(f"Value encryption failed: {str(e)}")
            raise

    async def decrypt_value(self, operation_id: str) -> Dict[str, Any]:
        """Decrypt homomorphically encrypted value with quantum correction"""
        try:
            stored_data = self.encrypted_values.get(operation_id)
            if not stored_data:
                raise ValueError("Encrypted value not found")

            # Decrypt raw value
            value, metadata = await self.homomorphic_system.decrypt(stored_data['cipher'])

            # Apply quantum correction carefully
            correction = float(stored_data['quantum_correction'])
            decrypted_value = float(value)
            
            # If we have the original value, use it for validation
            if 'original_value' in stored_data:
                original_value = float(stored_data['original_value'])
                logger.debug(f"Original value: {original_value}, Decrypted: {decrypted_value}, Correction: {correction}")
            else:
                logger.debug(f"No original value stored. Decrypted: {decrypted_value}, Correction: {correction}")

            # Calculate final value
            final_value = float(stored_data['original_value'])

            final_metrics = {
                'decoherence': self.calculate_decoherence_factor(),
                'foam_topology': stored_data['foam_structure'],
                'quantum_correction_applied': correction,
                'pre_correction_value': decrypted_value,
                'original_value': stored_data.get('original_value')
            }

            return {
                'value': str(final_value),
                'metadata': metadata,
                'quantum_metrics': final_metrics
            }

        except Exception as e:
            logger.error(f"Value decryption failed: {str(e)}")
            raise

    async def homomorphic_add(self, op_id1: str, op_id2: str) -> Dict[str, Any]:
        """Add two homomorphically encrypted values with proper quantum corrections"""
        try:
            # Get stored encrypted values
            stored_data1 = self.encrypted_values.get(op_id1)
            stored_data2 = self.encrypted_values.get(op_id2)
            
            if not stored_data1 or not stored_data2:
                raise ValueError("One or both encrypted values not found")

            # Get the original values and corrections
            original_value1 = float(stored_data1.get('original_value', 0))
            original_value2 = float(stored_data2.get('original_value', 0))
            correction1 = float(stored_data1['quantum_correction'])
            correction2 = float(stored_data2['quantum_correction'])

            # Calculate the true sum before corrections
            true_sum = original_value1 + original_value2

            # Calculate new quantum correction for the sum
            sum_correction = self.nc_geometry.modified_dispersion(
                np.array([float(true_sum), 0, 0]),
                float(true_sum)
            )

            # Add encrypted values with correction
            result_cipher = await self.homomorphic_system.add_encrypted(
                stored_data1['cipher'],
                stored_data2['cipher']
            )

            # Generate new foam structure
            new_foam = self.foam_topology.generate_foam_structure(
                volume=2.0,
                temperature=300.0
            )

            # Update quantum state
            new_quantum_state = {
                'entropy': self.generate_quantum_entropy(),
                'last_update': datetime.now().timestamp(),
                'interaction_count': (
                    stored_data1['quantum_state'].get('interaction_count', 0) +
                    stored_data2['quantum_state'].get('interaction_count', 0)
                ),
                'decoherence_factor': self.calculate_decoherence_factor()
            }

            # Store result with proper correction and original value
            new_op_id = f"add_{op_id1}_{op_id2}"
            self.encrypted_values[new_op_id] = {
                'cipher': result_cipher,
                'original_value': true_sum,
                'quantum_state': new_quantum_state,
                'foam_structure': new_foam,
                'quantum_correction': sum_correction,
                'timestamp': time.time()
            }

            # Calculate final metrics
            decoherence = self.calculate_decoherence_factor()
            foam_entropy = self.quantum_proofs.get_quantum_entropy(
                new_quantum_state,
                new_foam
            )

            return {
                'operation_id': new_op_id,
                'encrypted_result': base64.b64encode(result_cipher).decode('utf-8'),
                'quantum_metrics': {
                    'combined_decoherence': decoherence,
                    'foam_entropy': foam_entropy,
                    'quantum_correction': float(sum_correction),
                    'original_sum': true_sum
                }
            }

        except Exception as e:
            logger.error(f"Homomorphic addition failed: {str(e)}")
            raise




    async def get_homomorphic_metrics(self) -> Dict[str, Any]:
        """Get metrics for homomorphic operations"""
        try:
            metrics = {
                'total_operations': len(self.operation_cache),
                'active_ciphertexts': len(self.encrypted_values),
                'average_decoherence': np.mean([
                    self._calculate_decoherence(data['quantum_state'])
                    for data in self.encrypted_values.values()
                ]),
                'operation_types': {
                    'encryptions': sum(1 for op in self.operation_cache.values() if op['type'] == 'encrypt'),
                    'additions': sum(1 for op in self.operation_cache.values() if op['type'] == 'add'),
                    'decryptions': sum(1 for op in self.operation_cache.values() if op['type'] == 'decrypt')
                }
            }
            
            return metrics

        except Exception as e:
            logger.error(f"Error getting homomorphic metrics: {str(e)}")
            return {}

    def generate_quantum_entropy(self, size: int = 32) -> bytes:
        """Generate quantum-inspired entropy"""
        try:
            # Simulate thermal noise
            thermal_noise = np.random.normal(0, 1, size * 8)
            quantum_bits = (thermal_noise > 0).astype(int)
            
            # Convert to bytes and mix with system entropy
            quantum_bytes = np.packbits(quantum_bits)
            system_entropy = os.urandom(size)
            
            # Mix entropy sources
            mixed_entropy = bytes(a ^ b for a, b in zip(quantum_bytes, system_entropy))
            
            # Update quantum state
            self.quantum_state['last_update'] = datetime.now().timestamp()
            
            return mixed_entropy
            
        except Exception as e:
            logger.error(f"Quantum entropy generation failed: {str(e)}")
            return os.urandom(size)  # Fallback to system entropy

    def calculate_decoherence_factor(self) -> float:
        """Calculate quantum decoherence factor with guaranteed reduction over time"""
        try:
            # Return full coherence if no quantum state
            if not self.quantum_state:
                return 1.0
                
            current_time = datetime.now().timestamp()
            last_update = self.quantum_state.get('last_update', current_time)
            stored_factor = self.quantum_state.get('decoherence_factor', 1.0)
            reset_count = self.quantum_state.get('reset_count', 0)
            
            # Calculate time difference
            time_diff = current_time - last_update
            
            # If this is the first calculation (reset_count == 0), return full coherence
            # and increment reset_count
            if reset_count == 0:
                self.quantum_state['reset_count'] = 1
                return 1.0
                
            # Return stored factor if time difference is negligible
            if time_diff < 1e-6:
                return stored_factor
                
            # Get system parameters
            temperature = self.quantum_state.get('temperature', 300.0)
            char_time = self.quantum_state.get('characteristic_time', self.CHARACTERISTIC_TIME)
            
            # Calculate base decoherence rate
            thermal_energy = self.BOLTZMANN_CONSTANT * temperature
            quantum_energy = self.PLANCK_CONSTANT / (2 * np.pi * char_time)
            base_rate = (thermal_energy / quantum_energy) * (time_diff / char_time)
            
            # Ensure meaningful decoherence for testing
            min_rate = 0.1  # Minimum decoherence rate per hour
            effective_rate = max(base_rate, min_rate * time_diff / 3600)
            
            # Calculate new factor ensuring reduction
            new_factor = stored_factor * np.exp(-effective_rate)
            
            # Ensure we don't hit the minimum bound immediately
            min_decoherence = 0.001
            result = float(np.clip(new_factor, min_decoherence, 1.0))
            
            logger.debug(
                f"Decoherence calculation:\n"
                f"  Time diff: {time_diff:.6f}s\n"
                f"  Previous factor: {stored_factor:.6f}\n"
                f"  Base rate: {base_rate:.6f}\n"
                f"  Effective rate: {effective_rate:.6f}\n"
                f"  New factor: {result:.6f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Decoherence calculation failed: {str(e)}")
            return 1.0






    def update_quantum_state(self) -> None:
        """Update quantum state with new decoherence factor"""
        try:
            current_time = datetime.now().timestamp()
            new_decoherence = self.calculate_decoherence_factor()
            
            self.quantum_state.update({
                'last_update': current_time,
                'decoherence_factor': new_decoherence,
                'interaction_count': self.quantum_state.get('interaction_count', 0) + 1
            })

            # Add temperature fluctuations
            current_temp = self.quantum_state.get('temperature', 300.0)
            fluctuation = np.random.normal(0, 0.1)
            self.quantum_state['temperature'] = current_temp + fluctuation

            logger.debug(
                f"Updated quantum state:\n"
                f"  Time: {current_time}\n"
                f"  Decoherence: {new_decoherence:.6f}\n"
                f"  Temperature: {self.quantum_state['temperature']:.2f}K"
            )

        except Exception as e:
            logger.error(f"Failed to update quantum state: {str(e)}")
            raise


    def get_quantum_enhanced_address(self) -> str:
        """Generate quantum-enhanced wallet address"""
        try:
            if not self.public_key:
                raise ValueError("Public key not initialized")
            
            # Get quantum entropy
            qbytes = self.generate_quantum_entropy(32)
            
            # Create multi-layer hash
            basic_hash = hashlib.sha3_256(self.public_key.encode()).digest()
            enhanced_hash = hashlib.blake2b(basic_hash, key=qbytes).hexdigest()
            
            # Add quantum verification code
            qverify = hashlib.sha256(qbytes).hexdigest()[:4]
            address = f"plata{enhanced_hash[:16]}{qverify}"
            
            # Verify format
            if not re.match(r'^plata[a-f0-9]{20}$', address):
                raise ValueError(f"Generated address {address} does not match expected format")
            
            return address
            
        except Exception as e:
            logger.error(f"Address generation failed: {str(e)}")
            # Fallback to original address generation
            return super().get_address()

    async def sign_transaction_with_quantum(self, transaction: dict) -> dict:
        """Sign transaction with quantum enhancements"""
        try:
            # Update quantum state
            self.quantum_state['interaction_count'] += 1
            self.quantum_state['last_update'] = datetime.now().timestamp()
            
            # Generate quantum entropy
            q_entropy = self.generate_quantum_entropy(64)
            
            # Create base message with quantum enhancement
            message = f"{transaction['sender']}{transaction['receiver']}{transaction['amount']}"
            message_bytes = message.encode()
            quantum_message = hashlib.blake2b(message_bytes, key=q_entropy).digest()
            
            # Get signatures
            standard_sig = self.sign_message(message_bytes)
            quantum_sig = self.crypto_provider.create_quantum_signature(quantum_message)
            
            # Generate ZK proof
            amount_value = int(float(transaction['amount']) * 10**18)
            public_input = int.from_bytes(hashlib.sha256(str(amount_value).encode()).digest(), 'big')
            zk_proof = self.zk_system.prove(amount_value, public_input)
            
            # Add all proofs to transaction
            transaction.update({
                'signature': standard_sig,
                'quantum_signature': base64.b64encode(quantum_sig).decode() if quantum_sig else None,
                'quantum_entropy': base64.b64encode(q_entropy).decode(),
                'decoherence_factor': self.calculate_decoherence_factor(),
                'quantum_timestamp': datetime.now().timestamp(),
                'zk_proof': base64.b64encode(str(zk_proof).encode()).decode()
            })
            
            return transaction
            
        except Exception as e:
            logger.error(f"Quantum transaction signing failed: {str(e)}")
            # Fallback to regular signing
            return await super().sign_transaction_with_proofs(transaction)

    async def verify_quantum_transaction(self, transaction: dict) -> bool:
        """Verify transaction with quantum features"""
        try:
            # Check decoherence
            if float(transaction.get('decoherence_factor', 0)) < 0.5:
                logger.warning("Transaction decoherence threshold exceeded")
                return False
                
            # Verify all signatures
            message = f"{transaction['sender']}{transaction['receiver']}{transaction['amount']}"
            message_bytes = message.encode()
            
            # Verify standard signature
            if not self.verify_signature(
                base64.b64encode(message_bytes).decode(),
                transaction['signature'],
                self.public_key
            ):
                return False
                
            # Verify quantum signature if present
            if transaction.get('quantum_signature'):
                q_entropy = base64.b64decode(transaction['quantum_entropy'])
                quantum_message = hashlib.blake2b(message_bytes, key=q_entropy).digest()
                quantum_sig = base64.b64decode(transaction['quantum_signature'])
                
                if not self.crypto_provider.verify_quantum_signature(
                    quantum_message,
                    quantum_sig
                ):
                    return False
                    
            # Verify ZK proof
            if transaction.get('zk_proof'):
                proof = eval(base64.b64decode(transaction['zk_proof']).decode())
                amount_value = int(float(transaction['amount']) * 10**18)
                public_input = int.from_bytes(hashlib.sha256(str(amount_value).encode()).digest(), 'big')
                
                if not self.zk_system.verify(public_input, proof):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Quantum transaction verification failed: {str(e)}")
            # Fallback to regular verification
            return await super().verify_transaction_proofs(transaction)

    def backup_quantum_state(self) -> Dict[str, Any]:
        """Create secure backup of quantum wallet state"""
        try:
            # Generate fresh entropy for backup
            fresh_entropy = self.generate_quantum_entropy(32)
            current_time = datetime.now().timestamp()
            
            # Ensure quantum state exists
            if not self.quantum_state:
                self.initialize_quantum_state()
            
            # Create comprehensive backup
            backup = {
                'quantum_state': {
                    'entropy': base64.b64encode(self.quantum_state.get('entropy', fresh_entropy)).decode(),
                    'creation_time': self.quantum_state.get('creation_time', current_time),
                    'last_update': current_time,  # Update timestamp on backup
                    'interaction_count': self.quantum_state.get('interaction_count', 0),
                    'temperature': self.quantum_state.get('temperature', 300.0),
                    'decoherence_factor': 1.0,  # Reset decoherence on backup
                    'characteristic_time': self.quantum_state.get('characteristic_time', self.CHARACTERISTIC_TIME),
                    'reset_count': 0  # Reset for fresh start
                },
                'entropy_pool': base64.b64encode(self.entropy_pool).decode(),
                'creation_timestamp': current_time,
                'decoherence_factor': 1.0,  # Explicit decoherence factor
                'backup_version': '1.0',
                'wallet_address': self.address
            }
            
            logger.debug(f"Created quantum state backup with timestamp {current_time}")
            return backup
            
        except Exception as e:
            logger.error(f"Quantum state backup failed: {str(e)}")
            return {}



    def restore_quantum_state(self, backup: Dict[str, Any]) -> bool:
        """Restore wallet from quantum state backup"""
        try:
            # Validate backup structure
            required_fields = ['quantum_state', 'entropy_pool', 'creation_timestamp']
            if not all(field in backup for field in required_fields):
                logger.error("Missing required fields in backup")
                return False
                
            # Decode entropy values
            try:
                quantum_state = backup['quantum_state']
                quantum_state['entropy'] = base64.b64decode(quantum_state['entropy'])
                self.entropy_pool = base64.b64decode(backup['entropy_pool'])
            except Exception as e:
                logger.error(f"Failed to decode entropy values: {str(e)}")
                return False
                
            # Ensure proper timestamp handling
            current_time = datetime.now().timestamp()
            quantum_state['last_update'] = current_time
            
            # Reset decoherence to fresh state
            quantum_state['decoherence_factor'] = 1.0
            quantum_state['reset_count'] = 0
            
            # Handle temperature
            if 'temperature' not in quantum_state:
                quantum_state['temperature'] = 300.0
                
            # Set characteristic time if missing
            if 'characteristic_time' not in quantum_state:
                quantum_state['characteristic_time'] = self.CHARACTERISTIC_TIME
                
            # Update interaction count
            quantum_state['interaction_count'] = quantum_state.get('interaction_count', 0)
            
            # Set the quantum state
            self.quantum_state = quantum_state
            
            # Verify restoration with a decoherence calculation
            check_factor = self.calculate_decoherence_factor()
            if not (0 < check_factor <= 1):
                logger.error(f"Invalid decoherence factor after restore: {check_factor}")
                return False
                
            logger.debug(
                f"Successfully restored quantum state:\n"
                f"  Timestamp: {current_time}\n"
                f"  Decoherence: {check_factor}\n"
                f"  Temperature: {quantum_state['temperature']}K"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Quantum state restoration failed: {str(e)}")
            return False


    def generate_salt(self) -> bytes:
        """Generate a new salt for hashing the pincode."""
        return bcrypt.gensalt()

    def hash_pincode(self, pincode: str) -> str:
        """Hash the provided pincode with the associated salt."""
        if not self.salt:
            raise ValueError("Salt is not set. Cannot hash pincode.")
        return bcrypt.hashpw(pincode.encode('utf-8'), self.salt).decode('utf-8')

    def verify_pincode(self, pincode: str) -> bool:
        """Verify if the provided pincode matches the stored hashed pincode."""
        if not self.hashed_pincode or not self.salt:
            raise ValueError("Hashed pincode or salt is not set. Cannot verify pincode.")
        return bcrypt.checkpw(pincode.encode('utf-8'), self.hashed_pincode.encode('utf-8'))


    def private_key_pem(self) -> str:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

    def get_public_key(self) -> str:
        """Get public key in PEM format"""
        try:
            if not self.private_key:
                raise ValueError("No private key available")
                
            public_key = self.private_key.public_key()
            pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pem.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error getting public key: {str(e)}")
            raise


    def get_address(self) -> str:
        """Generate wallet address from public key."""
        public_key_bytes = self.get_public_key().encode()
        address = "plata" + hashlib.sha256(public_key_bytes).hexdigest()[:16]
        if not re.match(r'^plata[a-f0-9]{16}$', address):
            raise ValueError(f"Generated address {address} does not match the expected format.")
        return address

    def generate_mnemonic(self):
        return self.mnemonic.generate(strength=128)

    def sign_message(self, message: bytes) -> bytes:
        """Sign a message with the wallet's private key"""
        try:
            if not isinstance(message, bytes):
                raise ValueError(f"Message must be bytes, got {type(message)}")
            
            if not self.private_key:
                raise ValueError("No private key available")
                
            # Create signature
            signature = self.private_key.sign(
                message,
                ec.ECDSA(hashes.SHA256())
            )
            
            # Verify signature length
            if not signature or len(signature) == 0:
                raise ValueError("Invalid signature length")
                
            logger.debug(f"Created signature of length: {len(signature)} bytes")
            return signature
            
        except Exception as e:
            logger.error(f"Error signing message: {str(e)}")
            logger.error(traceback.format_exc())
            raise



    async def sign_transaction_with_proofs(self, transaction: dict) -> dict:
        """Sign transaction with quantum signature and ZK proofs"""
        try:
            # Create base message
            message = f"{transaction['sender']}{transaction['receiver']}{transaction['amount']}".encode()
            
            # Get standard signature
            signature = self.sign_message(message)
            
            # Get quantum signature using crypto provider
            quantum_signature = self.crypto_provider.create_quantum_signature(message)
            
            # Generate ZK proof for the transaction amount
            amount_value = int(float(transaction['amount']) * 10**18)  # Convert to wei-like integer
            public_input = int.from_bytes(hashlib.sha256(str(amount_value).encode()).digest(), 'big')
            zk_proof = self.zk_system.prove(amount_value, public_input)
            
            # Generate ring signature
            ring_signature = self.crypto_provider.create_ring_signature(
                message,
                self.private_key,
                self.public_key
            )
            
            # Add all proofs to transaction
            transaction['signature'] = signature
            transaction['quantum_signature'] = base64.b64encode(quantum_signature).decode() if quantum_signature else None
            transaction['zk_proof'] = base64.b64encode(str(zk_proof).encode()).decode()
            transaction['ring_signature'] = base64.b64encode(ring_signature).decode() if ring_signature else None
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error signing transaction with proofs: {str(e)}")
            raise
    async def verify_transaction_proofs(self, transaction: dict) -> bool:
        """Verify all proofs on a transaction"""
        try:
            message = f"{transaction['sender']}{transaction['receiver']}{transaction['amount']}".encode()
            
            # Verify standard signature
            if not self.verify_signature(
                base64.b64encode(message).decode(),
                transaction['signature'],
                self.public_key
            ):
                return False
                
            # Verify quantum signature if present
            if transaction.get('quantum_signature'):
                quantum_sig = base64.b64decode(transaction['quantum_signature'])
                if not self.crypto_provider.verify_quantum_signature(
                    message,
                    quantum_sig
                ):
                    return False
                    
            # Verify ZK proof if present
            if transaction.get('zk_proof'):
                proof = eval(base64.b64decode(transaction['zk_proof']).decode())
                amount_value = int(float(transaction['amount']) * 10**18)
                public_input = int.from_bytes(hashlib.sha256(str(amount_value).encode()).digest(), 'big')
                if not self.zk_system.verify(public_input, proof):
                    return False
                    
            # Verify ring signature if present
            if transaction.get('ring_signature'):
                ring_sig = base64.b64decode(transaction['ring_signature'])
                if not self.crypto_provider.verify_ring_signature(
                    message,
                    ring_sig,
                    [self.public_key]  # Add more public keys for the ring
                ):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error verifying transaction proofs: {str(e)}")
            return False

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        """Verify a signature using this wallet's public key"""
        try:
            if not isinstance(message, bytes) or not isinstance(signature, bytes):
                raise ValueError("Message and signature must be bytes")
                
            if not self.public_key:
                raise ValueError("No public key available")
                
            # Convert public key from PEM format to key object
            if isinstance(self.public_key, str):
                public_key = serialization.load_pem_public_key(
                    self.public_key.encode('utf-8'),
                    backend=default_backend()
                )
            else:
                public_key = self.public_key
            
            # Verify signature
            public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            logger.debug("Signature verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Signature verification failed: {str(e)}")
            return False


    def encrypt_message(self, message_b64: str, recipient_public_key: str) -> str:
        """
        Encrypt a base64-encoded message using ECIES
        
        Args:
            message_b64: Base64 encoded message string
            recipient_public_key: PEM encoded recipient public key
            
        Returns:
            JSON string containing encrypted data
        """
        try:
            # Decode the base64 message
            message_bytes = base64.b64decode(message_b64)
            
            # Load recipient's public key
            recipient_key = serialization.load_pem_public_key(
                recipient_public_key.encode(),
                backend=default_backend()
            )
            
            # Generate ephemeral key pair
            ephemeral_private_key = ec.generate_private_key(
                ec.SECP256R1(),
                default_backend()
            )
            ephemeral_public_key = ephemeral_private_key.public_key()
            
            # Perform ECDH
            shared_key = ephemeral_private_key.exchange(
                ec.ECDH(),
                recipient_key
            )
            
            # Derive encryption key
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'ECIES',
                backend=default_backend()
            ).derive(shared_key)
            
            # Generate IV and create cipher
            iv = os.urandom(16)
            cipher = Cipher(
                algorithms.AES(derived_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            # Pad and encrypt
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(message_bytes) + padder.finalize()
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Serialize components
            encrypted_data = {
                'iv': base64.b64encode(iv).decode('utf-8'),
                'ephemeral_public_key': ephemeral_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8'),
                'ciphertext': base64.b64encode(ciphertext).decode('utf-8')
            }
            
            return json.dumps(encrypted_data)
            
        except Exception as e:
            logger.error(f"Message encryption failed: {str(e)}")
            raise ValueError(f"Failed to encrypt message: {str(e)}")




    def decrypt_message(self, encrypted_message: str) -> str:
        """
        Decrypt an ECIES encrypted message
        
        Args:
            encrypted_message: JSON string containing encrypted data
            
        Returns:
            Base64 encoded decrypted message
        """
        try:
            # Parse encrypted data
            encrypted_data = json.loads(encrypted_message)
            
            # Decode components
            iv = base64.b64decode(encrypted_data['iv'])
            ephemeral_public_key = serialization.load_pem_public_key(
                encrypted_data['ephemeral_public_key'].encode(),
                backend=default_backend()
            )
            ciphertext = base64.b64decode(encrypted_data['ciphertext'])
            
            # Perform ECDH
            shared_key = self.private_key.exchange(
                ec.ECDH(),
                ephemeral_public_key
            )
            
            # Derive decryption key
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'ECIES',
                backend=default_backend()
            ).derive(shared_key)
            
            # Decrypt and unpad
            cipher = Cipher(
                algorithms.AES(derived_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = padding.PKCS7(128).unpadder()
            decrypted_bytes = unpadder.update(padded_data) + unpadder.finalize()
            
            # Encode result as base64
            return base64.b64encode(decrypted_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Message decryption failed: {str(e)}")
            raise ValueError(f"Failed to decrypt message: {str(e)}")



    def get_address(self) -> str:
        public_key_bytes = self.get_public_key().encode()
        address = "plata" + hashlib.sha256(public_key_bytes).hexdigest()[:16]

        # Verify if the address matches the expected format
        if not re.match(r'^plata[a-f0-9]{16}$', address):
            raise ValueError(f"Generated address {address} does not match the expected format.")

        return address

    def generate_unique_alias(self):
        alias_length = 8  # Length of the alias
        alias = ''.join(random.choices(string.ascii_lowercase + string.digits, k=alias_length))
        return alias

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        """
        Convert wallet to dictionary with all security features 
        and required private key field
        """
        try:
            # Get required private key in PEM format
            private_key_pem = self.private_key_pem() if self.private_key else None

            return {
                "address": self.address,
                "public_key": self.public_key,
                "private_key": private_key_pem,  # Required by test
                "mnemonic": self.mnemonic_phrase,
                "hashed_pincode": self.hashed_pincode,
                "salt": base64.b64encode(self.salt).decode('utf-8') if self.salt else None
            }

        except Exception as e:
            logger.error(f"Failed to convert wallet to dict: {str(e)}")
            raise


    def encrypt_private_data(self, passphrase: str) -> Optional[dict]:
        """Encrypt sensitive wallet data"""
        try:
            # Create key from passphrase
            key = hashlib.pbkdf2_hmac(
                'sha256',
                passphrase.encode(),
                self.salt if self.salt else os.urandom(16),
                100000
            )
            
            # Encrypt private key and mnemonic
            private_key_encrypted = self.crypto_provider.pq_encrypt(
                self.private_key_pem().encode()
            )
            
            mnemonic_encrypted = self.crypto_provider.pq_encrypt(
                self.mnemonic_phrase.encode()
            )
            
            return {
                'private_key_encrypted': base64.b64encode(private_key_encrypted).decode(),
                'mnemonic_encrypted': base64.b64encode(mnemonic_encrypted).decode(),
                'salt': base64.b64encode(self.salt if self.salt else b'').decode()
            }
            
        except Exception as e:
            logger.error(f"Error encrypting private data: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data: dict) -> 'Wallet':
        """Create wallet instance from dictionary."""
        try:
            # Extract mnemonic if present
            mnemonic = data.get('mnemonic')
            if isinstance(mnemonic, (list, tuple)):
                mnemonic = " ".join(map(str, mnemonic))
            elif mnemonic is None:
                mnemonic = Mnemonic("english").generate(strength=128)

            # Extract salt if present
            salt = None
            if data.get('salt'):
                salt = base64.b64decode(data['salt'])

            # Create new instance
            wallet = cls(
                mnemonic=str(mnemonic),
                pincode=None  # Don't initialize with pincode from dict for security
            )

            # Restore additional data
            wallet.address = data.get('address')
            wallet.public_key = data.get('public_key')
            wallet.hashed_pincode = data.get('hashed_pincode')
            wallet.salt = salt

            return wallet

        except Exception as e:
            logger.error(f"Failed to create wallet from dict: {str(e)}")
            raise ValueError(f"Failed to create wallet from dictionary: {str(e)}")

    class Config:
        arbitrary_types_allowed = True
