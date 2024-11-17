import asyncio
import websockets
import json
import time
import hashlib
from typing import Dict, Set, List, Tuple, Optional
from asyncio import Lock
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, OrderedDict
import traceback
from SecureHybridZKStark import SecureHybridZKStark
from STARK import calculate_field_size
import logging
import random
import base64
import urllib.parse
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import aiohttp
import miniupnpc
import stun
from typing import Tuple, Optional
import asyncio
import websockets
import json
import time
import hashlib
from typing import Dict, Set, List, Tuple, Optional
from asyncio import Lock
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, OrderedDict
import traceback
from SecureHybridZKStark import SecureHybridZKStark
from STARK import calculate_field_size
import logging
import random
import base64
import urllib.parse
import os
from dotenv import load_dotenv
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from typing import Any, Dict, Set, List, Tuple, Optional,Callable
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from exchange_websocket_updates import ExchangeWebSocketUpdates
from shared_logic import QuantumBlock, Transaction, NodeState, NodeDirectory
from cryptography.exceptions import InvalidSignature
import re
import base64
import logging
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import requests
from typing import Union, Dict, List, Optional
import psutil
import asyncio
import time
import traceback
import concurrent.futures
from decimal import Decimal 
import types 
from typing import TYPE_CHECKING
import types
import asyncio
import websockets
import json
import time
import hashlib
from typing import Dict, Set, List, Tuple, Optional
from asyncio import Lock
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, OrderedDict
import traceback
import logging
import random
import base64
import urllib.parse
import os
from dotenv import load_dotenv
import aiohttp
import socket
import systemd.daemon
import systemd.journal
import socket
import fcntl
import struct
import os
import pwd
import grp
from typing import Dict, Optional
import subprocess
import asyncio
import logging
from typing import Optional
from CryptoProvider import CryptoProvider
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HandshakeState(Enum):
    INIT = 0
    PUBLIC_KEY_SENT = 1
    PUBLIC_KEY_RECEIVED = 2
    CHALLENGE_SENT = 3
    CHALLENGE_RECEIVED = 4
    CHALLENGE_RESPONSE_SENT = 5
    CHALLENGE_RESPONSE_RECEIVED = 6
    COMPLETED = 7

class LoggingWebSocket(websockets.WebSocketClientProtocol):
    async def recv(self):
        message = await super().recv()
        logger.debug(f"Received WebSocket message: {message[:100]}...")  # Log first 100 chars
        return message

    async def send(self, message):
        logger.debug(f"Sending WebSocket message: {message[:100]}...")  # Log first 100 chars
        await super().send(message)

@dataclass
class ComputationTask:
    task_id: str
    function: str
    args: List[Any]
    kwargs: Dict[str, Any]
    result: Any = None
    status: str = "pending"


class DistributedComputationSystem:
    def __init__(self, node: 'P2PNode'):
        self.node = node
        self.tasks: Dict[str, ComputationTask] = {}
        self.available_functions: Dict[str, Callable] = {}

    def register_function(self, func: Callable):
        self.available_functions[func.__name__] = func

    async def submit_task(self, function_name: str, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        task = ComputationTask(task_id, function_name, args, kwargs)
        self.tasks[task_id] = task
        
        # Store the task in the DHT
        await self.node.store(f"task:{task_id}", json.dumps(asdict(task)))
        
        # Broadcast the task to the network
        await self.node.broadcast(Message(
            MessageType.SUBMIT_COMPUTATION.value,
            {"task_id": task_id}
        ))
        
        return task_id

    async def process_task(self, task_id: str):
        task_data = await self.node.get(f"task:{task_id}")
        if not task_data:
            return
        
        task = ComputationTask(**json.loads(task_data))
        if task.function not in self.available_functions:
            return
        
        task.status = "processing"
        await self.node.store(f"task:{task_id}", json.dumps(asdict(task)))
        
        try:
            result = self.available_functions[task.function](*task.args, **task.kwargs)
            task.result = result
            task.status = "completed"
        except Exception as e:
            task.result = str(e)
            task.status = "failed"
        
        await self.node.store(f"task:{task_id}", json.dumps(asdict(task)))
        
        # Broadcast the result
        await self.node.broadcast(Message(
            MessageType.COMPUTATION_RESULT.value,
            {"task_id": task_id, "status": task.status}
        ))

    async def get_task_result(self, task_id: str) -> ComputationTask:
        task_data = await self.node.get(f"task:{task_id}")
        if task_data:
            return ComputationTask(**json.loads(task_data))
        return None


@dataclass
class MagnetLink:
    info_hash: str
    trackers: List[str]
    peer_id: str

    @classmethod
    def from_uri(cls, uri: str):
        parsed = urllib.parse.urlparse(uri)
        params = urllib.parse.parse_qs(parsed.query)
        return cls(
            info_hash=params['xt'][0].split(':')[-1],
            trackers=params.get('tr', []),
            peer_id=params.get('x.peer_id', [None])[0]
        )

    def to_uri(self) -> str:
        query_params = {
            'xt': f'urn:btih:{self.info_hash}',
            'tr': self.trackers
        }
        if self.peer_id:
            query_params['x.peer_id'] = self.peer_id
        return f"magnet:?{urllib.parse.urlencode(query_params, doseq=True)}"
@dataclass  # Use frozen=True to ensure immutability for hashability
class KademliaNode:
    id: str
    ip: str
    port: int
    magnet_link: Optional[MagnetLink] = None
    last_seen: float = field(default_factory=time.time)

    @property
    def address(self):
        return f"{self.ip}:{self.port}"

    # Define a hash function based on the immutable fields (id, ip, and port)
    def __hash__(self):
        return hash((self.id, self.ip, self.port))
    def update_last_seen(self):
        self.last_seen = time.time()



    # __eq__ is automatically generated by the dataclass, no need to redefine


class MessageType(Enum):
    CHALLENGE = "challenge"
    CHALLENGE_RESPONSE = "challenge_response"
    GET_TRANSACTIONS = "get_transactions"
    GET_WALLETS = "get_wallets"
    TRANSACTIONS = "transactions"
    WALLETS = "wallets"
    HANDSHAKE = "handshake"
    GET_LATEST_BLOCK = "get_latest_block"
    GET_BLOCK_HASH = "get_block_hash"
    GET_BLOCK = "get_block"
    GET_MEMPOOL = "get_mempool"
    STATE_REQUEST = "state_request"
    FIND_NODE = "find_node"
    FIND_VALUE = "find_value"
    STORE = "store"
    MAGNET_LINK = "magnet_link"
    PEER_EXCHANGE = "peer_exchange"
    TRANSACTION = "transaction"
    BLOCK = "block"
    CHAIN_REQUEST = "chain_request"
    CHAIN_RESPONSE = "chain_response"
    MEMPOOL_REQUEST = "mempool_request"
    MEMPOOL_RESPONSE = "mempool_response"
    HEARTBEAT = "heartbeat"
    ZK_PROOF = "zk_proof"
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    STORE_FILE = "store_file"
    REQUEST_FILE = "request_file"
    SUBMIT_COMPUTATION = "submit_computation"
    COMPUTATION_RESULT = "computation_result"
    PUBLIC_KEY_EXCHANGE = "public_key_exchange"
    LOGO_UPLOAD = "logo_upload"
    LOGO_REQUEST = "logo_request"
    LOGO_RESPONSE = "logo_response"
    LOGO_SYNC = "logo_sync"
    BLOCK_PROPOSAL = "block_proposal"
    FULL_BLOCK_REQUEST = "full_block_request"
    BLOCK_ACCEPTANCE = "block_acceptance"
    FULL_BLOCK_RESPONSE = "full_block_response"
    NEW_WALLET = "new_wallet"
    GET_ALL_DATA = "get_all_data"
    ALL_DATA = "all_data"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    SYNC_DATA = "sync_data"
    SYNC_STATUS = "sync_status"
    # Add quantum-related message types
    QUANTUM_STATE_UPDATE = "quantum_state_update"
    QUANTUM_STATE_VERIFICATION = "quantum_state_verification"
    QUANTUM_ENTANGLEMENT_REQUEST = "quantum_entanglement_request"
    QUANTUM_ENTANGLEMENT_RESPONSE = "quantum_entanglement_response"
    QUANTUM_RESYNC_REQUEST = "quantum_resync_request"
    QUANTUM_RESYNC_RESPONSE = "quantum_resync_response"
    QUANTUM_HEARTBEAT = "quantum_heartbeat"
    QUANTUM_HEARTBEAT_RESPONSE = "quantum_heartbeat_response"
    QUANTUM_ALERT = "quantum_alert"
    CONSENSUS_INITIALIZE = "consensus_initialize"
    CONSENSUS_SUBMIT = "consensus_submit"
    CONSENSUS_METRICS = "consensus_metrics"
    CONSENSUS_STATUS = "consensus_status"


    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)


class SyncComponent(Enum):
    WALLETS = "wallets"
    TRANSACTIONS = "transactions"
    BLOCKS = "blocks"
    MEMPOOL = "mempool"

class SyncStatus:
    """Track sync status of a component"""
    def __init__(self):
        self.last_sync = time.time()
        self.current_hash = None
        self.is_syncing = False
        self.sync_progress = 0
        self.last_validated = time.time()
        self.retry_count = 0
        self.peers_synced = set()
from dataclasses import dataclass, field, asdict
import json
import time
from typing import Optional
import uuid

@dataclass
class Message:
    type: str
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sender: Optional[str] = None
    receiver: Optional[str] = None  
    challenge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Normalize message type after initialization"""
        if self.type:
            self.type = str(self.type).lower().strip()

    def to_json(self) -> str:
        """Convert message to JSON string with normalized type"""
        data = asdict(self)
        # Ensure type is normalized
        if data['type']:
            data['type'] = str(data['type']).lower().strip()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string with proper type handling"""
        try:
            data = json.loads(json_str)
            
            # Normalize message type
            if 'type' in data:
                data['type'] = str(data['type']).lower().strip()
            
            # Ensure required fields with defaults
            processed_data = {
                'type': data.get('type', "").lower().strip(),
                'payload': data.get('payload', {}),
                'timestamp': data.get('timestamp', time.time()),
                'sender': data.get('sender', None),
                'receiver': data.get('receiver', None),
                'challenge_id': data.get('challenge_id', str(uuid.uuid4())),
                'id': data.get('id', str(uuid.uuid4()))
            }
            
            return cls(**processed_data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error creating message: {str(e)}")

    @property
    def message_type(self) -> str:
        """Get normalized message type"""
        return str(self.type).lower().strip()

    def is_type(self, type_str: str) -> bool:
        """Check if message is of specific type (case-insensitive)"""
        return self.message_type == str(type_str).lower().strip()

    def is_challenge(self) -> bool:
        """Check if message is a challenge message"""
        return self.is_type("challenge")

    def is_challenge_response(self) -> bool:
        """Check if message is a challenge response"""
        return self.is_type("challenge_response")

    def is_public_key_exchange(self) -> bool:
        """Check if message is a public key exchange"""
        return self.is_type("public_key_exchange")
        
@dataclass
class TransactionState:
    tx_hash: str
    timestamp: float
    status: str
    propagation_count: int
    received_by: Set[str]
    confirmed_by: Set[str]

class TransactionTracker:
    def __init__(self):
        self.transactions: Dict[str, TransactionState] = {}
        
    def add_transaction(self, tx_hash: str, sender: str):
        self.transactions[tx_hash] = TransactionState(
            tx_hash=tx_hash,
            timestamp=time.time(),
            status="pending",
            propagation_count=0,
            received_by={sender},
            confirmed_by=set()
        )
        
    def update_transaction(self, tx_hash: str, peer: str, status: str = None):
        if tx_hash in self.transactions:
            tx_state = self.transactions[tx_hash]
            tx_state.received_by.add(peer)
            if status:
                tx_state.status = status
            tx_state.propagation_count += 1
class SyncState:
    """Helper class to track sync state"""
    def __init__(self):
        self.last_update = time.time()
        self.status = "initialized"
        self.hash = None
        self.in_progress = False
        self.last_attempt = None
        self.retry_count = 0
        self.peers_synced = set()

@dataclass
class QuantumHeartbeat:
    """Quantum heartbeat data structure"""
    node_id: str
    timestamp: float
    quantum_states: Dict[str, complex]
    fidelities: Dict[str, float]
    bell_pair_id: str
    nonce: str
    signature: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id,
            'timestamp': self.timestamp,
            'quantum_states': {
                k: {'real': v.real, 'imag': v.imag}
                for k, v in self.quantum_states.items()
            },
            'fidelities': self.fidelities,
            'bell_pair_id': self.bell_pair_id,
            'nonce': self.nonce,
            'signature': self.signature
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QuantumHeartbeat':
        quantum_states = {
            k: complex(v['real'], v['imag'])
            for k, v in data['quantum_states'].items()
        }
        return cls(
            node_id=data['node_id'],
            timestamp=data['timestamp'],
            quantum_states=quantum_states,
            fidelities=data['fidelities'],
            bell_pair_id=data['bell_pair_id'],
            nonce=data['nonce'],
            signature=data.get('signature')
        )
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import numpy as np
import asyncio
import hashlib
import time
import logging
from collections import defaultdict

class QuantumState(Enum):
    """Quantum states for node synchronization"""
    SUPERPOSITION = "SUPERPOSITION"  # Initial state while gathering data
    ENTANGLED = "ENTANGLED"         # Synchronized with peers
    COLLAPSED = "COLLAPSED"         # Local state verified
    DECOHERENT = "DECOHERENT"      # Out of sync

@dataclass
class QubitState:
    """Represents a quantum state for synchronization"""
    value: complex
    peers: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)
    fidelity: float = 1.0

@dataclass
class QuantumRegister:
    """Quantum register for storing entangled states"""
    wallets: QubitState = field(default_factory=lambda: QubitState(complex(1, 0)))
    transactions: QubitState = field(default_factory=lambda: QubitState(complex(1, 0)))
    blocks: QubitState = field(default_factory=lambda: QubitState(complex(1, 0)))
    mempool: QubitState = field(default_factory=lambda: QubitState(complex(1, 0)))

class QuantumEntangledSync:
    """
    Implements quantum-inspired synchronization between P2P nodes
    """
    def __init__(self, node_id: str, decoherence_threshold: float = 0.8):
        self.node_id = node_id
        self.decoherence_threshold = decoherence_threshold
        self.register = QuantumRegister()
        self.state = QuantumState.SUPERPOSITION
        self.entangled_peers: Dict[str, QuantumRegister] = {}
        self.logger = logging.getLogger("QuantumEntangledSync")
        self.bell_pairs: Dict[str, List[complex]] = defaultdict(list)

    async def initialize_quantum_state(self, initial_data: dict):
        """Initialize quantum state with node data"""
        try:
            # Create quantum state from node data
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                data_hash = hashlib.sha256(str(initial_data.get(component, '')).encode()).digest()
                # Convert hash to complex number for quantum state
                real = int.from_bytes(data_hash[:16], 'big') / 2**128
                imag = int.from_bytes(data_hash[16:], 'big') / 2**128
                setattr(self.register, component, QubitState(complex(real, imag)))
            
            self.state = QuantumState.SUPERPOSITION
            self.logger.info(f"Initialized quantum state for node {self.node_id}")
            
        except Exception as e:
            self.logger.error(f"Error initializing quantum state: {str(e)}")
            raise

    async def entangle_with_peer(self, peer_id: str, peer_data: dict):
        """Create quantum entanglement with a peer node"""
        try:
            # Generate Bell pair for entanglement
            bell_pair = self._generate_bell_pair()
            self.bell_pairs[peer_id] = bell_pair
            
            # Create entangled quantum register
            peer_register = QuantumRegister()
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                qubit = getattr(self.register, component)
                # Apply quantum entanglement operation
                entangled_state = self._entangle_states(qubit.value, bell_pair)
                peer_qubit = QubitState(entangled_state)
                peer_qubit.peers.add(peer_id)
                setattr(peer_register, component, peer_qubit)
            
            self.entangled_peers[peer_id] = peer_register
            self.state = QuantumState.ENTANGLED
            
            self.logger.info(f"Established quantum entanglement with peer {peer_id}")
            return peer_register
            
        except Exception as e:
            self.logger.error(f"Error entangling with peer {peer_id}: {str(e)}")
            raise

    def _generate_bell_pair(self) -> List[complex]:
        """Generate a Bell pair for quantum entanglement"""
        # Create maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        return [complex(1/np.sqrt(2), 0), complex(0, 0), 
                complex(0, 0), complex(1/np.sqrt(2), 0)]

    def _entangle_states(self, state1: complex, bell_pair: List[complex]) -> complex:
        """Entangle two quantum states using Bell pair"""
        # Simulate quantum entanglement using Bell state
        return (state1 * bell_pair[0] + state1.conjugate() * bell_pair[3]) / np.sqrt(2)

    async def measure_sync_state(self, component: str) -> float:
        """Measure the synchronization state of a component"""
        try:
            qubit = getattr(self.register, component)
            if not qubit.peers:
                return 1.0

            fidelities = []
            for peer_id in qubit.peers:
                if peer_id in self.entangled_peers:
                    peer_register = self.entangled_peers[peer_id]
                    peer_qubit = getattr(peer_register, component)
                    # Calculate quantum state fidelity
                    fidelity = self._calculate_fidelity(qubit.value, peer_qubit.value)
                    fidelities.append(fidelity)

            avg_fidelity = np.mean(fidelities) if fidelities else 1.0
            qubit.fidelity = avg_fidelity

            if avg_fidelity < self.decoherence_threshold:
                self.state = QuantumState.DECOHERENT
                self.logger.warning(f"Decoherence detected in {component}")
            
            return avg_fidelity
            
        except Exception as e:
            self.logger.error(f"Error measuring sync state: {str(e)}")
            return 0.0

    def _calculate_fidelity(self, state1: complex, state2: complex) -> float:
        """Calculate quantum state fidelity between two states"""
        return abs(state1.conjugate() * state2)

    async def update_component_state(self, component: str, new_data: dict):
        """Update quantum state for a component with new data"""
        try:
            # Calculate new quantum state
            data_hash = hashlib.sha256(str(new_data).encode()).digest()
            real = int.from_bytes(data_hash[:16], 'big') / 2**128
            imag = int.from_bytes(data_hash[16:], 'big') / 2**128
            new_state = complex(real, imag)

            # Update local register
            qubit = getattr(self.register, component)
            qubit.value = new_state
            qubit.timestamp = time.time()

            # Propagate change to entangled peers
            update_tasks = []
            for peer_id, peer_register in self.entangled_peers.items():
                if peer_id in self.bell_pairs:
                    bell_pair = self.bell_pairs[peer_id]
                    entangled_state = self._entangle_states(new_state, bell_pair)
                    peer_qubit = getattr(peer_register, component)
                    peer_qubit.value = entangled_state
                    update_tasks.append(self._notify_peer_update(peer_id, component))

            if update_tasks:
                await asyncio.gather(*update_tasks)
                self.logger.info(f"Updated quantum state for {component} across {len(update_tasks)} peers")
            
        except Exception as e:
            self.logger.error(f"Error updating component state: {str(e)}")
            raise

    async def _notify_peer_update(self, peer_id: str, component: str):
        """Notify peer of state update"""
        # This would be implemented to send actual network messages
        pass

    async def verify_global_consensus(self) -> bool:
        """Verify quantum consensus across all components"""
        try:
            consensus = True
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                fidelity = await self.measure_sync_state(component)
                if fidelity < self.decoherence_threshold:
                    consensus = False
                    break

            if consensus:
                self.state = QuantumState.COLLAPSED
                self.logger.info("Global quantum consensus verified")
            else:
                self.state = QuantumState.DECOHERENT
                self.logger.warning("Global quantum consensus failed")

            return consensus
            
        except Exception as e:
            self.logger.error(f"Error verifying global consensus: {str(e)}")
            return False

    async def monitor_quantum_state(self, interval: float = 1.0):
        """Continuously monitor quantum state and trigger resync if needed"""
        while True:
            try:
                await asyncio.sleep(interval)
                consensus = await self.verify_global_consensus()
                
                if not consensus:
                    components_to_sync = []
                    for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                        fidelity = await self.measure_sync_state(component)
                        if fidelity < self.decoherence_threshold:
                            components_to_sync.append(component)
                    
                    if components_to_sync:
                        self.logger.warning(f"Quantum resync needed for: {components_to_sync}")
                        # Trigger resync for decoherent components
                        
            except Exception as e:
                self.logger.error(f"Error in quantum state monitoring: {str(e)}")
                await asyncio.sleep(interval)
from dataclasses import dataclass
import asyncio
import json
from typing import Optional, Dict, Any
import hashlib
import base64
import time
import logging

@dataclass
class QuantumStateUpdate:
    """Represents a quantum state update message"""
    component: str
    state_value: complex
    timestamp: float
    bell_pair_id: str
    node_id: str
    signature: Optional[str] = None
    nonce: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'component': self.component,
            'state_value': {
                'real': self.state_value.real,
                'imag': self.state_value.imag
            },
            'timestamp': self.timestamp,
            'bell_pair_id': self.bell_pair_id,
            'node_id': self.node_id,
            'signature': self.signature,
            'nonce': self.nonce
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QuantumStateUpdate':
        state_value = complex(
            data['state_value']['real'],
            data['state_value']['imag']
        )
        return cls(
            component=data['component'],
            state_value=state_value,
            timestamp=data['timestamp'],
            bell_pair_id=data['bell_pair_id'],
            node_id=data['node_id'],
            signature=data.get('signature'),
            nonce=data.get('nonce')
        )


class QuantumStateNotifier:
    """Handles network communication for quantum state updates"""
    
    def __init__(self, node, max_retries: int = 3, retry_delay: float = 1.0):
        self.node = node  # Reference to the P2PNode instance
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pending_updates: Dict[str, asyncio.Event] = {}
        self.logger = logging.getLogger("QuantumStateNotifier")
        self.verification_cache: Dict[str, float] = {}
        self.cache_expiry = 60  # seconds

    async def notify_peer_update(self, peer_id: str, component: str, quantum_state: complex,
                               bell_pair_id: str) -> bool:
        """Send quantum state update to peer with retry logic and verification"""
        try:
            # Create quantum state update message
            update = QuantumStateUpdate(
                component=component,
                state_value=quantum_state,
                timestamp=time.time(),
                bell_pair_id=bell_pair_id,
                node_id=self.node.node_id,
                nonce=self._generate_nonce()
            )

            # Sign the update
            update.signature = self._sign_update(update)
            
            # Create verification tracker
            update_id = f"{update.node_id}:{update.nonce}"
            verification_event = asyncio.Event()
            self.pending_updates[update_id] = verification_event

            success = False
            retries = 0
            last_error = None

            while retries < self.max_retries and not success:
                try:
                    # Send update to peer
                    await self._send_update_to_peer(peer_id, update)
                    
                    # Wait for verification with timeout
                    verification_timeout = 5.0  # 5 seconds timeout
                    success = await asyncio.wait_for(
                        verification_event.wait(),
                        timeout=verification_timeout
                    )
                    
                    if success:
                        self.logger.info(
                            f"Quantum state update verified by peer {peer_id} "
                            f"for component {component}"
                        )
                        break
                    
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Verification timeout for peer {peer_id} "
                        f"(attempt {retries + 1}/{self.max_retries})"
                    )
                except Exception as e:
                    last_error = e
                    self.logger.error(
                        f"Error sending update to peer {peer_id}: {str(e)}"
                    )

                retries += 1
                if retries < self.max_retries:
                    # Exponential backoff
                    await asyncio.sleep(self.retry_delay * (2 ** retries))

            # Cleanup
            del self.pending_updates[update_id]

            if not success:
                error_msg = f"Failed to notify peer {peer_id} after {self.max_retries} attempts"
                if last_error:
                    error_msg += f": {str(last_error)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            return success

        except Exception as e:
            self.logger.error(f"Error in notify_peer_update: {str(e)}")
            raise

    async def _send_update_to_peer(self, peer_id: str, update: QuantumStateUpdate):
        """Send quantum state update message to peer"""
        try:
            # Convert update to network message
            message = {
                'type': 'quantum_state_update',
                'payload': update.to_dict()
            }

            # Send via P2P node's message system
            await self.node.send_message(
                peer_id,
                Message(
                    type=MessageType.QUANTUM_STATE_UPDATE.value,
                    payload=message
                )
            )

            self.logger.debug(
                f"Sent quantum state update to peer {peer_id} "
                f"for component {update.component}"
            )

        except Exception as e:
            self.logger.error(f"Error sending update to peer {peer_id}: {str(e)}")
            raise

    async def handle_update_message(self, message: Dict[str, Any], sender: str) -> bool:
        """Handle incoming quantum state update message"""
        try:
            # Parse update message
            update = QuantumStateUpdate.from_dict(message)
            
            # Verify signature
            if not self._verify_signature(update, sender):
                self.logger.warning(f"Invalid signature for update from {sender}")
                return False

            # Check for replay attacks
            if self._is_replay(update):
                self.logger.warning(f"Detected replay attack from {sender}")
                return False

            # Cache the nonce to prevent replay attacks
            self._cache_nonce(update.nonce, update.timestamp)

            # Verify quantum state using Bell pair
            if not await self._verify_quantum_state(update, sender):
                self.logger.warning(
                    f"Quantum state verification failed for update from {sender}"
                )
                return False

            # Apply the quantum state update locally
            await self._apply_quantum_update(update, sender)

            # Send verification acknowledgment
            await self._send_verification(update, sender)

            self.logger.info(
                f"Successfully processed quantum state update from {sender} "
                f"for component {update.component}"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Error handling update message from {sender}: {str(e)}"
            )
            return False

    async def _verify_quantum_state(self, update: QuantumStateUpdate, sender: str) -> bool:
        """Verify quantum state using Bell pair"""
        try:
            # Get Bell pair for this peer
            bell_pair = self.node.quantum_sync.bell_pairs.get(sender)
            if not bell_pair:
                self.logger.warning(f"No Bell pair found for peer {sender}")
                return False

            # Verify Bell pair ID matches
            if update.bell_pair_id != self._get_bell_pair_id(bell_pair):
                self.logger.warning(f"Bell pair ID mismatch for peer {sender}")
                return False

            # Calculate expected quantum state
            local_state = getattr(
                self.node.quantum_sync.register,
                update.component
            ).value
            expected_state = self.node.quantum_sync._entangle_states(
                local_state,
                bell_pair
            )

            # Calculate fidelity between expected and received states
            fidelity = abs(
                expected_state.conjugate() * update.state_value
            )

            # Check if fidelity is above threshold
            return fidelity >= self.node.quantum_sync.decoherence_threshold

        except Exception as e:
            self.logger.error(f"Error verifying quantum state: {str(e)}")
            return False

    async def _apply_quantum_update(self, update: QuantumStateUpdate, sender: str):
        """Apply verified quantum state update"""
        try:
            # Get peer's quantum register
            peer_register = self.node.quantum_sync.entangled_peers[sender]
            
            # Update the quantum state
            qubit = getattr(peer_register, update.component)
            qubit.value = update.state_value
            qubit.timestamp = update.timestamp

            self.logger.debug(
                f"Applied quantum state update for {update.component} "
                f"from peer {sender}"
            )

        except Exception as e:
            self.logger.error(f"Error applying quantum update: {str(e)}")
            raise

    async def _send_verification(self, update: QuantumStateUpdate, sender: str):
        """Send verification acknowledgment to peer"""
        try:
            verification = {
                'type': 'quantum_state_verification',
                'update_id': f"{update.node_id}:{update.nonce}",
                'timestamp': time.time(),
                'verified': True
            }

            await self.node.send_message(
                sender,
                Message(
                    type=MessageType.QUANTUM_STATE_VERIFICATION.value,
                    payload=verification
                )
            )

        except Exception as e:
            self.logger.error(
                f"Error sending verification to {sender}: {str(e)}"
            )
            raise

    async def handle_verification_message(self, message: Dict[str, Any], sender: str):
        """Handle incoming verification acknowledgment"""
        try:
            update_id = message['update_id']
            if update_id in self.pending_updates:
                verification_event = self.pending_updates[update_id]
                verification_event.set()

        except Exception as e:
            self.logger.error(
                f"Error handling verification message from {sender}: {str(e)}"
            )

    def _generate_nonce(self) -> str:
        """Generate unique nonce for update message"""
        return base64.b64encode(os.urandom(16)).decode()

    def _sign_update(self, update: QuantumStateUpdate) -> str:
        """Sign update message using node's private key"""
        message = (
            f"{update.component}:{update.state_value}:{update.timestamp}:"
            f"{update.bell_pair_id}:{update.nonce}"
        ).encode()
        
        signature = self.node.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def _verify_signature(self, update: QuantumStateUpdate, sender: str) -> bool:
        """Verify update signature using sender's public key"""
        try:
            public_key = self.node.peer_public_keys.get(sender)
            if not public_key:
                return False

            message = (
                f"{update.component}:{update.state_value}:{update.timestamp}:"
                f"{update.bell_pair_id}:{update.nonce}"
            ).encode()
            
            signature = base64.b64decode(update.signature)
            
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True

        except Exception:
            return False

    def _is_replay(self, update: QuantumStateUpdate) -> bool:
        """Check if update is a replay attack"""
        return update.nonce in self.verification_cache

    def _cache_nonce(self, nonce: str, timestamp: float):
        """Cache nonce and cleanup expired entries"""
class BootstrapManager:
    """Manages bootstrap node connections with retry logic"""
    
    def __init__(self, node: 'P2PNode', max_retries: int = 5, retry_delay: float = 2.0):
        self.node = node
        self.max_retries = max_retries
        self.base_retry_delay = retry_delay
        self.bootstrap_nodes = self.node.bootstrap_nodes
        self.connected_bootstrap_nodes = set()
        self.logger = logging.getLogger("BootstrapManager")

    async def connect_to_bootstrap_nodes(self):
        """Connect to bootstrap nodes with enhanced retry logic and connection recovery."""
        if not self.bootstrap_nodes:
            self.logger.warning("No bootstrap nodes configured")
            return False

        self.logger.info("\n=== Connecting to Bootstrap Nodes ===")
        self.logger.info(f"Bootstrap nodes: {self.bootstrap_nodes}")
        
        connected = False
        retry_count = 0
        max_retries = 5
        base_delay = 2.0  # Base delay for exponential backoff
        
        while not connected and retry_count < max_retries:
            try:
                for bootstrap_node in self.bootstrap_nodes:
                    try:
                        # Parse bootstrap node address
                        host, port = self.parse_bootstrap_address(bootstrap_node)
                        if not host or not port:
                            continue

                        self.logger.info(f"Attempting connection to bootstrap node {host}:{port}")
                        
                        # Create connection with timeout and keep-alive
                        websocket = await asyncio.wait_for(
                            websockets.connect(
                                f"ws://{host}:{port}",
                                timeout=30,
                                ping_interval=20,  # Enable periodic ping
                                ping_timeout=10,    # Timeout for ping responses
                                close_timeout=5,
                                extra_headers={     # Add version and capabilities
                                    'X-Node-Version': '1.0',
                                    'X-Node-Capabilities': 'quantum,dagknight,zkp'
                                }
                            ),
                            timeout=30
                        )

                        # Add to peers and initialize connection state
                        peer_id = f"{host}:{port}"
                        async with self.peer_lock:
                            self.peers[peer_id] = websocket
                            self.peer_states[peer_id] = "connecting"
                            # Initialize connection tracking
                            self.peer_info[peer_id] = {
                                'connection_attempts': 0,
                                'last_attempt': time.time(),
                                'connected_since': time.time(),
                                'failures': 0
                            }

                        # Perform key exchange with retry logic
                        key_exchange_success = False
                        for key_attempt in range(3):  # Try key exchange up to 3 times
                            try:
                                if await self.exchange_public_keys(peer_id):
                                    key_exchange_success = True
                                    break
                                await asyncio.sleep(1)  # Brief pause between attempts
                            except Exception as ke:
                                self.logger.warning(f"Key exchange attempt {key_attempt + 1} failed: {str(ke)}")
                                continue

                        if key_exchange_success:
                            self.logger.info(f"✓ Connected to bootstrap node {peer_id}")
                            self.connected_peers.add(peer_id)
                            connected = True
                            
                            # Start handlers for the new connection
                            asyncio.create_task(self.handle_messages(websocket, peer_id))
                            asyncio.create_task(self.keep_connection_alive(websocket, peer_id))
                            
                            # Request peer list after successful connection
                            await self.request_peer_list(peer_id)
                            break
                        else:
                            self.logger.warning(f"Failed to exchange keys with {peer_id}")
                            await self.cleanup_failed_connection(peer_id)

                    except asyncio.TimeoutError:
                        self.logger.warning(f"Connection attempt to {bootstrap_node} timed out")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error connecting to {bootstrap_node}: {str(e)}")
                        continue

                if not connected:
                    retry_count += 1
                    retry_delay = base_delay * (2 ** retry_count)  # Exponential backoff
                    self.logger.warning(
                        f"Failed to connect to any bootstrap nodes. "
                        f"Retrying in {retry_delay:.1f}s ({retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                self.logger.error(f"Error in bootstrap connection loop: {str(e)}")
                retry_count += 1
                await asyncio.sleep(base_delay * (2 ** retry_count))

        if connected:
            self.logger.info("✓ Successfully connected to bootstrap network")
        else:
            self.logger.warning("Failed to connect to any bootstrap nodes")

        return connected

    async def cleanup_failed_connection(self, peer_id: str):
        """Clean up resources for a failed connection."""
        try:
            async with self.peer_lock:
                if peer_id in self.peers:
                    websocket = self.peers[peer_id]
                    if not websocket.closed:
                        await websocket.close()
                    self.peers.pop(peer_id)
                    self.peer_states.pop(peer_id, None)
                    self.peer_public_keys.pop(peer_id, None)
                    self.connected_peers.discard(peer_id)
                    self.challenges.pop(peer_id, None)
                    if hasattr(self, 'quantum_sync'):
                        self.quantum_sync.entangled_peers.pop(peer_id, None)
                        self.quantum_sync.bell_pairs.pop(peer_id, None)
        except Exception as e:
            self.logger.error(f"Error cleaning up failed connection to {peer_id}: {str(e)}")

    async def request_peer_list(self):
        """Request peer list from connected bootstrap nodes"""
        try:
            if not self.connected_bootstrap_nodes:
                return
                
            self.logger.info("Requesting peer list from bootstrap nodes...")
            
            for bootstrap_node in self.connected_bootstrap_nodes:
                try:
                    message = Message(
                        type=MessageType.PEER_EXCHANGE.value,
                        payload={"request": "peer_list"}
                    )
                    
                    response = await self.node.send_and_wait_for_response(
                        bootstrap_node,
                        message,
                        timeout=10.0
                    )
                    
                    if response and response.type == MessageType.PEER_EXCHANGE.value:
                        peers = response.payload.get('peers', [])
                        self.logger.info(f"Received {len(peers)} peers from {bootstrap_node}")
                        
                        # Connect to received peers
                        for peer in peers:
                            if peer not in self.node.peers:
                                asyncio.create_task(self.connect_to_peer(peer))
                                
                except Exception as e:
                    self.logger.error(f"Error getting peers from {bootstrap_node}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error requesting peer list: {str(e)}")

    def parse_bootstrap_address(self, address: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse bootstrap node address"""
        try:
            if ':' not in address:
                self.logger.error(f"Invalid bootstrap address format: {address}")
                return None, None
                
            host, port = address.split(':')
            return host.strip(), int(port)
            
        except Exception as e:
            self.logger.error(f"Error parsing bootstrap address {address}: {str(e)}")
            return None, None

    async def connect_to_peer(self, peer_address: str):
        """Connect to a new peer"""
        try:
            host, port = self.parse_bootstrap_address(peer_address)
            if not host or not port:
                return

            websocket = await websockets.connect(
                f"ws://{host}:{port}",
                timeout=10,
                ping_interval=None,
                close_timeout=5
            )
            
            peer_id = f"{host}:{port}"
            self.node.peers[peer_id] = websocket
            self.node.peer_states[peer_id] = "connecting"
            
            if await self.node.exchange_public_keys(peer_id):
                self.logger.info(f"✓ Connected to peer {peer_id}")
                asyncio.create_task(self.node.handle_messages(websocket, peer_id))
            else:
                await self.node.remove_peer(peer_id)
                
        except Exception as e:
            self.logger.error(f"Error connecting to peer {peer_address}: {str(e)}")
class ChallengeRole(Enum):
    """Role in the challenge-response protocol"""
    SERVER = "server"
    CLIENT = "client"

class ChallengeState:
    """State of a challenge with enhanced tracking"""
    def __init__(self, challenge: str, role: ChallengeRole):
        self.challenge = challenge
        self.role = role
        self.timestamp = time.time()
        self.attempts = 0
        self.verified = False
        self.last_attempt = None
        self.responses = []  # Track all received responses

class ChallengeManager:
    """Manages challenge storage and retrieval with thread safety"""
    def __init__(self):
        self.challenges: Dict[str, Dict[str, ChallengeState]] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger("ChallengeManager")
        self.CHALLENGE_TIMEOUT = 30  # seconds
        self.MAX_ATTEMPTS = 3
        self.MAX_CHALLENGES_PER_PEER = 5

    async def store_challenge(self, peer: str, challenge_id: str, challenge: str, role: ChallengeRole) -> bool:
        """
        Store a challenge with validation and limits.
        
        Args:
            peer: Peer address
            challenge_id: Unique challenge identifier
            challenge: Base64 encoded challenge string
            role: Role in the challenge (server/client)
            
        Returns:
            bool: True if challenge was stored successfully
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not all([peer, challenge_id, challenge]):
                raise ValueError("Missing required challenge parameters")

            # Validate base64 challenge
            try:
                base64.b64decode(challenge)
            except Exception:
                raise ValueError("Challenge must be valid base64")

            async with self._lock:
                # Initialize peer dict if needed
                if peer not in self.challenges:
                    self.challenges[peer] = {}
                
                # Check challenge limits
                if len(self.challenges[peer]) >= self.MAX_CHALLENGES_PER_PEER:
                    # Remove oldest challenge
                    oldest = min(
                        self.challenges[peer].items(),
                        key=lambda x: x[1].timestamp
                    )
                    del self.challenges[peer][oldest[0]]
                    self.logger.warning(f"Removed old challenge {oldest[0]} for peer {peer}")

                # Store new challenge
                self.challenges[peer][challenge_id] = ChallengeState(
                    challenge=challenge,
                    role=role
                )
                
                self.logger.debug(
                    f"Stored challenge for peer {peer}:\n"
                    f"  ID: {challenge_id}\n"
                    f"  Role: {role.value}\n"
                    f"  Active Challenges: {len(self.challenges[peer])}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error storing challenge: {str(e)}")
            return False

    async def get_challenge(self, peer: str, challenge_id: str) -> Optional[ChallengeState]:
        """
        Get a stored challenge with validation and tracking.
        
        Args:
            peer: Peer address
            challenge_id: Challenge identifier
            
        Returns:
            Optional[ChallengeState]: Challenge state if found and valid
        """
        try:
            async with self._lock:
                # Get challenge state
                challenge = self.challenges.get(peer, {}).get(challenge_id)
                if not challenge:
                    self.logger.warning(f"No challenge found for {peer} with ID {challenge_id}")
                    return None

                # Check expiration
                if time.time() - challenge.timestamp > self.CHALLENGE_TIMEOUT:
                    await self.remove_challenge(peer, challenge_id)
                    self.logger.warning(f"Challenge {challenge_id} expired for {peer}")
                    return None

                # Check attempts
                if challenge.attempts >= self.MAX_ATTEMPTS:
                    await self.remove_challenge(peer, challenge_id)
                    self.logger.warning(f"Max attempts exceeded for challenge {challenge_id}")
                    return None

                # Update attempt tracking
                challenge.attempts += 1
                challenge.last_attempt = time.time()

                self.logger.debug(
                    f"Retrieved challenge for peer {peer}:\n"
                    f"  ID: {challenge_id}\n"
                    f"  Role: {challenge.role.value}\n"
                    f"  Attempts: {challenge.attempts}/{self.MAX_ATTEMPTS}"
                )
                return challenge

        except Exception as e:
            self.logger.error(f"Error retrieving challenge: {str(e)}")
            return None

    async def remove_challenge(self, peer: str, challenge_id: str) -> None:
        """Safely remove a challenge."""
        async with self._lock:
            if peer in self.challenges and challenge_id in self.challenges[peer]:
                del self.challenges[peer][challenge_id]
                if not self.challenges[peer]:
                    del self.challenges[peer]

    async def cleanup_expired(self) -> None:
        """Clean up expired challenges."""
        try:
            current_time = time.time()
            async with self._lock:
                for peer in list(self.challenges.keys()):
                    for challenge_id, state in list(self.challenges[peer].items()):
                        if current_time - state.timestamp > self.CHALLENGE_TIMEOUT:
                            await self.remove_challenge(peer, challenge_id)
                            self.logger.debug(f"Cleaned up expired challenge {challenge_id} for {peer}")
        except Exception as e:
            self.logger.error(f"Error cleaning up challenges: {str(e)}")

    async def get_active_challenges(self, peer: str) -> Dict[str, ChallengeState]:
        """Get all active challenges for a peer."""
        async with self._lock:
            return self.challenges.get(peer, {}).copy()
class ConnectionState:
    INIT = "server_init"
    KEY_EXCHANGED = "key_exchanged"
    CHALLENGE_SENT = "challenge_sent"
    VERIFIED = "verified"
    CONNECTED = "connected"
class ChallengeVerificationSystem:
    """Enhanced challenge verification system with port mapping and state tracking"""

    def __init__(self, node: 'P2PNode'):
        self.node = node
        self.logger = logging.getLogger("ChallengeVerification")
        self.port_mappings = {}  # Track original:mapped port pairs
        self.active_challenges = {}
        self.verification_lock = asyncio.Lock()

    async def verify_challenge_response(self, peer: str, incoming_challenge_id: str, response_payload: dict) -> bool:
        """Verify challenge response with guaranteed challenge matching."""
        try:
            # Initial logging
            self.logger.info(f"\n[VERIFY] {'='*20} Verifying Response {'='*20}")
            
            # Normalize addresses with proper port handling
            peer_ip, peer_port = peer.split(':')
            client_port = '50510'  # Original client port
            client_address = f"{peer_ip}:{client_port}"
            
            self.logger.debug(f"[VERIFY] Client address: {client_address}")
            self.logger.debug(f"[VERIFY] Original challenge ID: {incoming_challenge_id}")

            # Get latest challenge stored for the client address
            client_challenges = await self.challenge_manager.get_active_challenges(client_address)
            if not client_challenges:
                self.logger.error(f"[VERIFY] No challenges found for client {client_address}")
                return False
                
            # Get most recent challenge
            recent_challenge = None
            recent_id = None
            
            # First try to find the incoming challenge ID
            if incoming_challenge_id in client_challenges:
                recent_challenge = client_challenges[incoming_challenge_id]
                recent_id = incoming_challenge_id
                self.logger.debug(f"[VERIFY] Found exact matching challenge: {incoming_challenge_id}")
            else:
                # Get most recent challenge by timestamp
                sorted_challenges = sorted(
                    client_challenges.items(),
                    key=lambda x: x[1].timestamp if hasattr(x[1], 'timestamp') else 0,
                    reverse=True
                )
                if sorted_challenges:
                    recent_id, recent_challenge = sorted_challenges[0]
                    self.logger.debug(f"[VERIFY] Using most recent challenge: {recent_id}")

            if not recent_challenge:
                self.logger.error("[VERIFY] No valid challenge found")
                self.logger.debug(f"[VERIFY] Available challenges: {list(client_challenges.keys())}")
                return False

            # Log challenge details
            self.logger.debug(f"[VERIFY] Challenge roles:")
            self.logger.debug(f"  - Original role: {recent_challenge.role}")
            self.logger.debug(f"  - Challenge data: {recent_challenge.challenge}")

            # Get public key from client address
            peer_public_key = self.peer_public_keys.get(client_address)
            if not peer_public_key:
                self.logger.error(f"[VERIFY] No public key found for {client_address}")
                return False

            try:
                # Verify signature
                signature = base64.b64decode(response_payload['signature'])
                challenge_bytes = recent_challenge.challenge.encode()
                
                self.logger.debug(f"[VERIFY] Verification details:")
                self.logger.debug(f"  - Signature length: {len(signature)}")
                self.logger.debug(f"  - Challenge bytes length: {len(challenge_bytes)}")
                self.logger.debug(f"  - First 32 bytes of challenge: {recent_challenge.challenge[:32]}")

                peer_public_key.verify(
                    signature,
                    challenge_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )

                # Update states only after successful verification
                async with self.peer_lock:
                    # Update primary client address
                    self.peer_states[client_address] = "verified"
                    self.connected_peers.add(client_address)
                    
                    # Map current peer address to client address if different
                    if peer != client_address:
                        self.peer_address_map[peer] = client_address
                        self.peer_states[peer] = "verified"
                        self.connected_peers.add(peer)

                # Clean up the verified challenge
                await self.challenge_manager.remove_challenge(client_address, recent_id)
                
                self.logger.info(f"[VERIFY] ✓ Challenge verification successful")
                self.logger.info(f"[VERIFY] {'='*50}\n")
                return True

            except InvalidSignature:
                self.logger.error(f"[VERIFY] Invalid signature for challenge {recent_id}")
                self.logger.debug(f"[VERIFY] Challenge being verified: {recent_challenge.challenge}")
                return False

            except Exception as e:
                self.logger.error(f"[VERIFY] Verification error: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False

        except Exception as e:
            self.logger.error(f"[VERIFY] Fatal error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    def get_possible_addresses(self, ip: str, port: int) -> Set[str]:
        """Get all possible address combinations for a peer"""
        addresses = {f"{ip}:{port}"}  # Original address
        
        # Add mapped ports
        for orig_port, mapped_port in self.port_mappings.items():
            if port in (orig_port, mapped_port):
                addresses.add(f"{ip}:{orig_port}")
                addresses.add(f"{ip}:{mapped_port}")

        return addresses

    def add_port_mapping(self, original_port: int, mapped_port: int):
        """Add a new port mapping"""
        self.port_mappings[original_port] = mapped_port
        self.logger.debug(f"Added port mapping: {original_port} -> {mapped_port}")
import logging
class ImprovedChallengeManager:
    """Manages challenges with enhanced port mapping and ID tracking"""

    def __init__(self):
        self.challenges = {}
        self.port_mappings = {}  # original_port -> mapped_port
        self.challenge_map = {}  # challenge_id -> peer_address
        self.logger = logging.getLogger("ChallengeManager")

    def add_port_mapping(self, original_port: int, mapped_port: int):
        """Track port mappings for a peer"""
        self.port_mappings[original_port] = mapped_port
        self.logger.debug(f"Added port mapping: {original_port} -> {mapped_port}")

    def get_all_peer_addresses(self, peer: str) -> set:
        """Get all possible addresses for a peer including port mappings"""
        addresses = {peer}
        peer_ip, peer_port = peer.split(':')
        peer_port = int(peer_port)

        # Add mapped port variations
        if peer_port in self.port_mappings:
            addresses.add(f"{peer_ip}:{self.port_mappings[peer_port]}")
        # Check reverse mapping
        for orig, mapped in self.port_mappings.items():
            if mapped == peer_port:
                addresses.add(f"{peer_ip}:{orig}")

        return addresses

logger = logging.getLogger(__name__)
class P2PNode:
    _instance = None
    _initialized = False
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, blockchain, host='localhost', port=8000, security_level=10, k: int = 20):
        try:
            # Load environment variables first
            load_dotenv()
            self.peer_address_map = {}
            self.peer_ports = {}        # Tracks original connection ports
            self.challenge_verifier = ChallengeVerificationSystem(self)
            self.challenge_manager = ImprovedChallengeManager()


            # Basic node configuration
            self.host = os.getenv('P2P_HOST', host)
            self.port = int(os.getenv('P2P_PORT', port))
            self.blockchain = blockchain
            self.security_level = int(os.getenv('SECURITY_LEVEL', security_level))
            self.max_peers = int(os.getenv('MAX_PEERS', '10'))

            # Generate node_id BEFORE quantum components
            self.node_id = self.generate_node_id()
            
            # Initialize crypto components
            self.crypto_provider = CryptoProvider()  # Create the CryptoProvider instance
            self.peer_crypto_providers = {}  # Track peer crypto providers
            
            # Initialize core components
            self.field_size = calculate_field_size(self.security_level)
            self.zk_system = SecureHybridZKStark(self.security_level)
            self.peers = {}
            self.peer_lock = asyncio.Lock()
            self.message_queue = asyncio.Queue()
            self.last_broadcast = {}
            self.mempool = []
            self.seen_messages = set()
            self.message_expiry = int(os.getenv('MESSAGE_EXPIRY', '300'))
            self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '30'))
            
            # Initialize sync states
            self.sync_queue = asyncio.Queue()
            self.sync_states = {
                SyncComponent.WALLETS: SyncStatus(),
                SyncComponent.TRANSACTIONS: SyncStatus(),
                SyncComponent.BLOCKS: SyncStatus(),
                SyncComponent.MEMPOOL: SyncStatus()
            }
            self.sync_retry_count = {}
            self.pending_sync_operations = {}

            # Now initialize quantum components
            self.quantum_sync = QuantumEntangledSync(self.node_id)
            self.quantum_notifier = QuantumStateNotifier(self)
            self.quantum_state = QuantumState.SUPERPOSITION
            self.quantum_register = QuantumRegister()
            self.quantum_initialized = False
            self.quantum_consensus = QuantumConsensusManager(self)
            self.quantum_monitor = QuantumNetworkMonitor(self)
            self.quantum_heartbeats = {}
            self.last_quantum_heartbeat = {}
            self.quantum_heartbeat_interval = 5.0  # seconds
            self.quantum_heartbeat_timeout = 15.0  # seconds

            # Initialize other components
            self.server = None
            self.last_challenge = {}
            self.pending_challenges = {}
            self.peer_states = {}
            self.peer_tasks = {}
            self.connected_peers = set()
            self.peer_locks = {}
            self.last_activity_time = time.time()

            # Initialize Kademlia DHT components
            self.k = k
            self.buckets = [[] for _ in range(160)]
            self.data_store = OrderedDict()
            self.max_data_store_size = int(os.getenv('MAX_DATA_STORE_SIZE', '1000'))

            # Initialize magnet link
            self.magnet_link = self.generate_magnet_link()

            # Initialize cryptographic components
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.public_key = self.private_key.public_key()
            self.peer_public_keys = {}
            
            # Initialize networking components
            self.ws_updates = ExchangeWebSocketUpdates(
                os.getenv('WS_HOST', 'localhost'),
                int(os.getenv('WS_PORT', '8080'))
            )
            
            # Parse bootstrap and seed nodes
            self.bootstrap_nodes = [node for node in os.getenv('BOOTSTRAP_NODES', '').split(',') if node]
            self.seed_nodes = [node for node in os.getenv('SEED_NODES', '').split(',') if node]

            # Initialize logging
            self.logger = logging.getLogger('P2PNode')
            self.logger.setLevel(logging.DEBUG)
            
            # Initialize transaction tracking
            self.transaction_tracker = TransactionTracker()
            self.is_running = False
            self.target_peer_count = int(os.getenv('TARGET_PEER_COUNT', 5))  # Default to 5 peers
            self.external_ip = os.getenv('EXTERNAL_IP', 'your.node.ip.address')
            self.challenges = {}
            self.peer_info = {}
            self.peer_states = {}
            self.handshake_timeouts = {}
            self.max_handshake_attempts = 3
            self.handshake_timeout = 30  # seconds
            self.health_monitor = ConnectionHealthMonitor(self) 
            self.challenge_manager = ChallengeManager()

            P2PNode._initialized = True

            self.consensus_handler = ConsensusMessageHandler(self)
            self.message_handlers = {
                'mining': self.handle_mining_message,
                'transaction': self.handle_transaction_message,
                'wallet': self.handle_wallet_message,
                'consensus': self.handle_consensus_message
            }

            self.logger.info("P2P Node initialized successfully")
            self.test_handler = QuantumP2PTestHandler(self)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error initializing P2P Node: {str(e)}")
                self.logger.error(traceback.format_exc())
            else:
                print(f"Error initializing P2P Node: {str(e)}")
                print(traceback.format_exc())
            raise
                
                
    def decode_quantum_key(self, quantum_key_b64: str) -> bytes:
        """Decode base64 quantum key back to bytes."""
        try:
            return base64.b64decode(quantum_key_b64)
        except Exception as e:
            self.logger.error(f"Error decoding quantum key: {str(e)}")
            raise ValueError("Invalid quantum key format")

    # Update the key exchange handling code:
        peer_quantum_key_b64 = response.payload.get('quantum_key')
        if not peer_quantum_key_b64:
            raise ValueError("Missing quantum public key")

        try:
            peer_quantum_key = self.decode_quantum_key(peer_quantum_key_b64)
        except Exception as e:
            raise ValueError(f"Invalid quantum key: {str(e)}")

    async def monitor_network_state(self):
        """Monitor network state and connections with detailed metrics."""
        try:
            while self.is_running:
                self.logger.info("\n=== Network State Monitor ===")
                
                # Check peer connections
                connected_peers = len(self.connected_peers)
                total_peers = len(self.peers)
                self.logger.info(f"Peers: {connected_peers}/{total_peers} connected")
                
                # Check quantum state if enabled
                if hasattr(self, 'quantum_sync') and self.quantum_initialized:
                    entangled_peers = len(self.quantum_sync.entangled_peers)
                    self.logger.info(f"Quantum peers: {entangled_peers}")
                    
                    # Check quantum fidelities
                    for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                        try:
                            fidelity = await self.quantum_sync.measure_sync_state(component)
                            self.logger.info(f"{component} fidelity: {fidelity:.3f}")
                        except Exception as e:
                            self.logger.error(f"Error measuring {component} fidelity: {str(e)}")
                
                # Check sync status
                for component, state in self.sync_states.items():
                    self.logger.info(f"{component} sync: {'In Progress' if state.is_syncing else 'Synced'}")
                
                # Network metrics
                if hasattr(self, 'network_optimizer'):
                    metrics = await self.network_optimizer.collect_network_metrics()
                    self.logger.info(f"Network latency: {metrics.get('latency', {}).get('average', 0):.2f}ms")
                    self.logger.info(f"Packet loss: {metrics.get('packet_loss', {}).get('average', 0):.2f}%")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except Exception as e:
            self.logger.error(f"Error in network state monitor: {str(e)}")
            self.logger.error(traceback.format_exc())  
    async def handle_consensus_message(self, message: dict) -> dict:
        """Handle consensus-related messages"""
        try:
            action = message.get('action')
            if action == 'initialize':
                return await self.initialize_consensus(message.get('params', {}))
            elif action == 'test_consensus':
                return await self.run_consensus_test(message.get('params', {}))
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown consensus action: {action}'
                }
        except Exception as e:
            logger.error(f"Error handling consensus message: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    async def handle_mining_message(self, data: dict) -> dict:
        """
        Handle incoming mining-related messages.
        
        Args:
            data (dict): Mining message data containing action and parameters
            
        Returns:
            dict: Response with status and any relevant data
        """
        try:
            action = data.get('action')
            if not action:
                return {'status': 'error', 'message': 'No mining action specified'}

            if action == 'submit_block':
                # Handle new block submission
                block_data = data.get('block')
                if not block_data:
                    return {'status': 'error', 'message': 'No block data provided'}
                
                # Convert block data to QuantumBlock object
                block = QuantumBlock.from_dict(block_data)
                
                # Add block to blockchain
                result = await self.blockchain.handle_new_block(block_data, sender=None)
                
                if result:
                    return {
                        'status': 'success',
                        'message': f'Block {block.hash} accepted',
                        'block_hash': block.hash
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Block {block.hash} rejected'
                    }

            elif action == 'get_mining_metrics':
                # Return current mining metrics
                metrics = self.get_mining_metrics()
                return {
                    'status': 'success',
                    'metrics': metrics
                }

            elif action == 'get_difficulty':
                # Return current mining difficulty
                if hasattr(self.blockchain, 'difficulty'):
                    return {
                        'status': 'success',
                        'difficulty': self.blockchain.difficulty
                    }
                return {
                    'status': 'error',
                    'message': 'Difficulty not available'
                }

            else:
                return {
                    'status': 'error',
                    'message': f'Unknown mining action: {action}'
                }

        except Exception as e:
            self.logger.error(f"Error handling mining message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }
            
            
            
            
            
    async def handle_transaction_message(self, data: dict) -> dict:
        """
        Handle incoming transaction-related messages with enhanced security and validation.
        
        Args:
            data (dict): Transaction message data containing transaction details
            
        Returns:
            dict: Response with status and transaction processing results
        """
        try:
            self.logger.info("[TRANSACTION] Processing transaction message")
            self.logger.debug(f"[TRANSACTION] Data: {data}")

            if not data:
                return {
                    'status': 'error',
                    'message': 'No transaction data provided'
                }

            action = data.get('action', 'process')  # Default action is process
            
            if action == 'process':
                # Check required fields
                required_fields = ['sender', 'receiver', 'amount']
                if not all(field in data for field in required_fields):
                    missing_fields = [field for field in required_fields if field not in data]
                    return {
                        'status': 'error',
                        'message': f'Missing required fields: {missing_fields}'
                    }

                try:
                    # Convert amount to Decimal for precise handling
                    amount = Decimal(str(data['amount']))
                except (InvalidOperation, TypeError):
                    return {
                        'status': 'error',
                        'message': 'Invalid amount format'
                    }

                # Create transaction object
                transaction = Transaction(
                    sender=data['sender'],
                    receiver=data['receiver'],
                    amount=amount,
                    timestamp=data.get('timestamp', int(time.time())),
                    signature=data.get('signature'),
                    public_key=data.get('public_key'),
                    quantum_enabled=data.get('quantum_enabled', False)
                )

                # Add additional data if present
                if 'gas_price' in data:
                    transaction.gas_price = Decimal(str(data['gas_price']))
                if 'gas_limit' in data:
                    transaction.gas_limit = int(data['gas_limit'])
                if 'nonce' in data:
                    transaction.nonce = int(data['nonce'])

                # Validate transaction
                if not await self.blockchain.validate_transaction(transaction):
                    return {
                        'status': 'error',
                        'message': 'Transaction validation failed'
                    }

                # Process transaction
                try:
                    success = await self.blockchain.add_transaction(transaction)
                    if success:
                        # Add to transaction tracker
                        self.transaction_tracker.add_transaction(
                            transaction.hash(),
                            data.get('origin_peer', 'unknown')
                        )

                        # Propagate to other peers if not already propagated
                        if not data.get('propagated'):
                            data['propagated'] = True
                            await self.propagate_transaction(transaction)

                        return {
                            'status': 'success',
                            'message': 'Transaction processed successfully',
                            'tx_hash': transaction.hash(),
                            'timestamp': transaction.timestamp
                        }
                    else:
                        return {
                            'status': 'error',
                            'message': 'Failed to process transaction'
                        }

                except Exception as process_error:
                    self.logger.error(f"Transaction processing error: {str(process_error)}")
                    return {
                        'status': 'error',
                        'message': f'Transaction processing error: {str(process_error)}'
                    }

            elif action == 'get_status':
                # Check for transaction hash
                tx_hash = data.get('tx_hash')
                if not tx_hash:
                    return {
                        'status': 'error',
                        'message': 'Transaction hash not provided'
                    }

                # Get transaction status
                tx_status = await self.blockchain.get_transaction_status(tx_hash)
                return {
                    'status': 'success',
                    'tx_status': tx_status
                }

            elif action == 'get_recent':
                # Get recent transactions
                limit = data.get('limit', 100)  # Default to 100 transactions
                recent_txs = await self.blockchain.get_recent_transactions(limit)
                return {
                    'status': 'success',
                    'transactions': [tx.to_dict() for tx in recent_txs]
                }

            else:
                return {
                    'status': 'error',
                    'message': f'Unknown transaction action: {action}'
                }

        except Exception as e:
            self.logger.error(f"Error handling transaction message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }
            
    async def handle_wallet_message(self, data: dict) -> dict:
        """
        Handle wallet-related messages including creation, updates, and queries.
        
        Args:
            data (dict): Wallet message data containing action and parameters
            
        Returns:
            dict: Response with status and wallet operation results
        """
        try:
            self.logger.info("[WALLET] Processing wallet message")
            self.logger.debug(f"[WALLET] Data: {data}")

            if not data:
                return {
                    'status': 'error',
                    'message': 'No wallet data provided'
                }

            action = data.get('action')
            if not action:
                return {
                    'status': 'error',
                    'message': 'No wallet action specified'
                }

            # Handle different wallet actions
            if action == 'create':
                # Create new wallet
                try:
                    user_id = data.get('user_id')
                    if not user_id:
                        return {
                            'status': 'error',
                            'message': 'User ID required for wallet creation'
                        }

                    wallet_info = await self.blockchain.create_wallet(user_id)
                    
                    # Propagate new wallet to network if successful
                    if wallet_info:
                        await self.broadcast_event('new_wallet', wallet_info)
                        
                        return {
                            'status': 'success',
                            'message': 'Wallet created successfully',
                            'wallet': wallet_info
                        }
                    else:
                        return {
                            'status': 'error',
                            'message': 'Failed to create wallet'
                        }

                except ValueError as ve:
                    return {
                        'status': 'error',
                        'message': str(ve)
                    }

            elif action == 'get_balance':
                # Get wallet balance
                try:
                    address = data.get('address')
                    if not address:
                        return {
                            'status': 'error',
                            'message': 'Wallet address required'
                        }

                    balances = await self.blockchain.get_balances(address)
                    return {
                        'status': 'success',
                        'balances': balances
                    }

                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error getting balance: {str(e)}'
                    }

            elif action == 'get_transactions':
                # Get wallet transactions
                try:
                    address = data.get('address')
                    limit = data.get('limit', 100)  # Default to 100 transactions
                    
                    if not address:
                        return {
                            'status': 'error',
                            'message': 'Wallet address required'
                        }

                    transactions = await self.blockchain.get_transaction_history(address, limit)
                    return {
                        'status': 'success',
                        'transactions': transactions
                    }

                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error getting transactions: {str(e)}'
                    }

            elif action == 'get_wallets':
                # Get all wallets
                try:
                    wallets = self.blockchain.get_wallets()
                    return {
                        'status': 'success',
                        'wallets': [w.to_dict() for w in wallets]
                    }
                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error getting wallets: {str(e)}'
                    }

            elif action == 'update_wallet':
                # Update wallet information
                try:
                    wallet_data = data.get('wallet')
                    if not wallet_data:
                        return {
                            'status': 'error',
                            'message': 'No wallet data provided for update'
                        }

                    wallet = Wallet.from_dict(wallet_data)
                    success = self.blockchain.update_wallet(wallet)
                    
                    if success:
                        # Propagate wallet update
                        await self.broadcast_event('wallet_update', wallet_data)
                        
                        return {
                            'status': 'success',
                            'message': 'Wallet updated successfully'
                        }
                    else:
                        return {
                            'status': 'error',
                            'message': 'Failed to update wallet'
                        }

                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error updating wallet: {str(e)}'
                    }

            elif action == 'validate_address':
                # Validate wallet address
                try:
                    address = data.get('address')
                    if not address:
                        return {
                            'status': 'error',
                            'message': 'No address provided for validation'
                        }

                    is_valid = self.blockchain.validate_wallet_address(address)
                    return {
                        'status': 'success',
                        'is_valid': is_valid
                    }

                except Exception as e:
                    return {
                        'status': 'error',
                        'message': f'Error validating address: {str(e)}'
                    }

            else:
                return {
                    'status': 'error',
                    'message': f'Unknown wallet action: {action}'
                }

        except Exception as e:
            self.logger.error(f"Error handling wallet message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }
                
    async def handle_websocket_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming WebSocket messages with enhanced category support."""
        try:
            data = json.loads(message)
            category = data.get('category')

            if not category:
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': 'No category specified'
                }))
                return

            # Log the incoming message
            self.logger.debug(f"Received {category} message: {data}")

            # Update handler dictionary with consensus support
            self.message_handlers = {
                'wallet': self.handle_wallet_message,
                'mining': self.handle_mining_message,
                'transaction': self.handle_transaction_message,
                'consensus': self.handle_consensus_message,  # Added consensus handler
                'p2p_test': self.handle_p2p_test_message,  # Assuming this function exists
            }

            # Check if category is supported
            if category not in self.message_handlers:
                supported_categories = list(self.message_handlers.keys())
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': f'Unknown category: {category}. Valid categories: {supported_categories}'
                }))
                return

            # Handle the message using appropriate handler
            handler = self.message_handlers[category]
            response = await handler(data)
            
            # Log and send response
            self.logger.debug(f"Handler response: {response}")
            await websocket.send(json.dumps(response))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'status': 'error',
                'message': 'Invalid JSON message'
            }))
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.logger.error(traceback.format_exc())
            await websocket.send(json.dumps({
                'status': 'error',
                'message': str(e)
            }))

    async def handle_p2p_test_message(self, websocket: websockets.WebSocketServerProtocol, data: dict) -> dict:
        """Handle P2P test messages with proper parameter passing"""
        try:
            if not self.p2p_node:
                return {
                    "status": "error",
                    "message": "P2P node not initialized"
                }

            # Ensure test handler exists
            if not hasattr(self.p2p_node, 'test_handler'):
                self.logger.info("Initializing P2P test handler")
                self.p2p_node.test_handler = QuantumP2PTestHandler(self.p2p_node)

            return await self.p2p_node.test_handler.handle_test_message(websocket, data)
            
        except Exception as e:
            self.logger.error(f"Error handling P2P test message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e)
            }


    async def initialize_dagknight(self):
        """Initialize DAGKnight consensus system."""
        self.dagknight = DAGKnightConsensus(
            min_k_cluster=10,
            max_latency_ms=1000
        )
        self.logger.info("DAGKnight consensus system initialized")

    async def handle_new_block_dagknight(self, block: QuantumBlock, sender: str):
        """Handle new block using DAGKnight consensus."""
        try:
            # Measure network latency
            start_time = time.time()
            block_hash = block.hash
            
            # Basic block validation
            if not self.blockchain.validate_block(block):
                self.logger.warning(f"Invalid block received from {sender}: {block_hash}")
                return False
                
            # Calculate network latency
            network_latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Add block to DAGKnight
            confirmable = await self.dagknight.add_block(block, network_latency)
            
            if confirmable:
                # Add confirmed block to blockchain
                success = await self.blockchain.add_block(block)
                if success:
                    self.logger.info(f"Block {block_hash[:8]} added to blockchain (latency: {network_latency:.2f}ms)")
                    
                    # Propagate block to other peers
                    await self.propagate_block(block, exclude_peer=sender)
                    
                    # Update quantum state
                    await self.quantum_sync.update_component_state('blocks', {
                        'block': block.to_dict(),
                        'latency': network_latency
                    })
                
                return success
                
            else:
                self.logger.debug(f"Block {block_hash[:8]} pending confirmation")
                return False
                
        except Exception as e:
            self.logger.error(f"Error handling block with DAGKnight: {str(e)}")
            return False

    async def monitor_network_security(self):
        """Monitor network security using DAGKnight metrics."""
        while self.is_running:
            try:
                # Get network status
                status = await self.dagknight.get_network_status()
                self.logger.info("\n=== DAGKnight Network Status ===")
                self.logger.info(f"Confirmed blocks: {status['confirmed_blocks']}")
                self.logger.info(f"Pending blocks: {status['pending_blocks']}")
                self.logger.info(f"Average latency: {status['average_latency']:.2f}ms")
                
                # Analyze security
                security = await self.dagknight.analyze_network_security()
                self.logger.info("\n=== Security Analysis ===")
                self.logger.info(f"Status: {security['status']}")
                self.logger.info(f"Median latency: {security['median_latency']:.2f}ms")
                
                # Alert on security issues
                if security['status'] != 'secure':
                    await self.broadcast(Message(
                        type=MessageType.NETWORK_ALERT.value,
                        payload={
                            'type': 'security_warning',
                            'analysis': security
                        }
                    ))
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error monitoring network security: {str(e)}")
                await asyncio.sleep(60)



    def find_highest_fidelity_peer(self, component: str):
        """Find the peer with the highest fidelity for a given component."""
        if not self.peers:
            self.logger.warning(f"No peers available to select for {component}.")
            return None
        # Example: Sort by fidelity for the specific component (mock example if fidelity is stored)
        return max(self.peers.values(), key=lambda peer: peer.get(f'{component}_fidelity', 0), default=None)



    async def _notify_peer_update(self, peer_id: str, component: str):
        """Notifies peer of quantum state updates with verification"""
        try:
            # Get quantum state and Bell pair for peer
            qubit = getattr(self.quantum_sync.register, component)
            bell_pair = self.quantum_sync.bell_pairs.get(peer_id)
            
            if not bell_pair:
                self.logger.warning(f"No Bell pair found for peer {peer_id}")
                return

            # Generate Bell pair ID
            bell_pair_id = self.quantum_notifier._get_bell_pair_id(bell_pair)

            # Send update via notifier with verification
            success = await self.quantum_notifier.notify_peer_update(
                peer_id,
                component,
                qubit.value,
                bell_pair_id
            )

            if success:
                self.logger.info(f"Successfully notified {peer_id} of {component} update")
            else:
                self.logger.warning(f"Failed to notify {peer_id} of {component} update")

        except Exception as e:
            self.logger.error(f"Error notifying peer {peer_id}: {str(e)}")
            raise


    async def propagate_multisig_transaction(self, multisig_address: str, sender_public_keys: List[int], threshold: int, receiver: str, amount: Decimal, message: str, aggregate_proof: Tuple[List[int], List[int], List[Tuple[int, List[int]]]]):
        transaction_data = {
            "type": "multisig_transaction",
            "multisig_address": multisig_address,
            "sender_public_keys": sender_public_keys,
            "threshold": threshold,
            "receiver": receiver,
            "amount": str(amount),
            "message": message,
            "aggregate_proof": aggregate_proof
        }
        await self.broadcast(Message(MessageType.TRANSACTION.value, transaction_data))


    async def start_ws_server(self):
        self.ws_server = await websockets.serve(self.handle_ws_connection, self.host, self.port + 1000)
        logger.info(f"WebSocket server started on {self.host}:{self.port + 1000}")

    async def handle_ws_connection(self, websocket, path):
        self.ws_clients.add(websocket)
        try:
            async for message in websocket:
                await self.handle_ws_message(websocket, message)
        finally:
            self.ws_clients.remove(websocket)

    async def handle_ws_message(self, websocket, message):
        data = json.loads(message)
        if data['type'] == 'subscribe':
            topic = data['topic']
            if topic in self.subscription_topics:
                self.subscription_topics[topic].add(websocket)
                await websocket.send(json.dumps({'status': 'subscribed', 'topic': topic}))
            else:
                await websocket.send(json.dumps({'status': 'error', 'message': 'Invalid topic'}))


    async def broadcast_event(self, topic, data):
        # Ensure the topic exists in subscription_topics
        if topic not in self.subscription_topics:
            logger.warning(f"Topic '{topic}' not found. Initializing the topic.")
            self.subscription_topics[topic] = set()  # Initialize the topic as an empty set

        message = json.dumps({'topic': topic, 'data': data})

        # Broadcast the message to all clients subscribed to the topic
        for client in self.subscription_topics[topic]:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                # If the client is no longer connected, remove them from the subscription list
                self.subscription_topics[topic].remove(client)
                logger.info(f"Removed disconnected client from topic '{topic}'")

    async def handle_new_wallet(self, data):
        user_id = data['user_id']
        wallet_address = data['wallet_address']
        self.logger.info(f"Registering new wallet for user {user_id} with address {wallet_address}")
        # Update local state
        self.blockchain.add_wallet(user_id, wallet_address)
        self.logger.debug(f"Wallet added to local blockchain for user {user_id}")
        # Broadcast to other nodes
        await self.broadcast(Message(MessageType.NEW_WALLET, data))
        self.logger.info(f"Broadcasted NEW_WALLET message for {wallet_address} to peers")

    async def handle_new_block(self, block_data: dict, sender: str):
        try:
            new_block = QuantumBlock.from_dict(block_data)
            logger.info(f"Received new block from {sender}: {new_block.hash}")
            
            if self.blockchain.validate_block(new_block):
                added = await self.blockchain.add_block(new_block)
                if added:
                    logger.info(f"Block {new_block.hash} added to blockchain")
                    # Propagate the block to other peers
                    await self.propagate_block(new_block)
                else:
                    logger.warning(f"Block {new_block.hash} not added to blockchain (might be duplicate)")
            else:
                logger.warning(f"Received invalid block from {sender}: {new_block.hash}")
        except Exception as e:
            logger.error(f"Error processing new block from {sender}: {str(e)}")
            logger.error(traceback.format_exc())



    async def handle_private_transaction(self, data):
        tx_hash = data['tx_hash']
        sender = data['sender']
        amount = data['amount']
        # Update local state
        await self.blockchain.add_private_transaction(tx_hash, sender, amount)
        # Broadcast to other nodes
        await self.broadcast(Message(MessageType.PRIVATE_TRANSACTION, data))
        
        # WebSocket broadcast for subscribed clients
        await self.broadcast_event('private_transaction', data)

    async def request_latest_data(self):
        for peer in self.peers:
            try:
                response = await self.send_and_wait_for_response(peer, Message(MessageType.REQUEST_LATEST_DATA, {}))
                if response and response.type == MessageType.LATEST_DATA:
                    await self.update_local_data(response.payload)
            except Exception as e:
                logger.error(f"Failed to request latest data from {peer}: {str(e)}")

    async def update_local_data(self, data):
        for wallet in data['wallets']:
            self.blockchain.add_wallet(wallet['user_id'], wallet['address'])
        for tx in data['private_transactions']:
            await self.blockchain.add_private_transaction(tx['hash'], tx['sender'], tx['amount'])

    async def handle_request_latest_data(self, sender: str):
        wallets = self.blockchain.get_all_wallets()
        private_txs = self.blockchain.get_private_transactions()
        await self.send_message(sender, Message(MessageType.LATEST_DATA, {
            'wallets': wallets,
            'private_transactions': private_txs
        }))
    async def handle_private_transaction(self, data):
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        await self.broadcast(Message(MessageType.PRIVATE_TRANSACTION, {'encrypted': encrypted_data.decode()}))

    async def process_private_transaction(self, encrypted_data):
        decrypted_data = json.loads(self.fernet.decrypt(encrypted_data.encode()).decode())
        tx_hash = decrypted_data['tx_hash']
        sender = decrypted_data['sender']
        amount = decrypted_data['amount']
        
        # Verify ZKP
        proof = decrypted_data['zkp']
        if not self.verify_zkp(proof, sender, amount):
            logger.warning(f"Invalid ZKP for private transaction from {sender}")
            return

        await self.blockchain.add_private_transaction(tx_hash, sender, amount)

    def verify_zkp(self, proof, sender, amount):
        # Implement ZKP verification logic here
        pass


    def generate_encryption_key(self) -> bytes:
        password = os.getenv('ENCRYPTION_PASSWORD', 'default_password').encode()
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    def set_blockchain(self, blockchain):
        self.blockchain = blockchain
        self.logger.info("Blockchain set for P2PNode")

    async def ensure_blockchain(self):
        if self.blockchain is None:
            self.logger.warning("Blockchain is not set. Waiting for initialization...")
            while self.blockchain is None:
                await asyncio.sleep(1)
            self.logger.info("Blockchain is now available")
    async def periodic_sync(self):
        while True:
            try:
                active_peers = await self.get_active_peers()
                for peer in active_peers:
                    await self.sync_wallets(peer)
                    await self.sync_transactions(peer)
                    await self.sync_blocks(peer)
                    await self.sync_with_peer(peer)

                await asyncio.sleep(1)  # Sync every 5 minutes
            except Exception as e:
                logger.error(f"Error during periodic sync: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Wait a minute before retrying
    async def sync_with_peer(self, peer):
        try:
            latest_block = await self.request_latest_block(peer)
            if latest_block.index > self.blockchain.get_latest_block().index:
                await self.sync_blocks(peer, self.blockchain.get_latest_block().index + 1, latest_block.index)
        except Exception as e:
            logger.error(f"Error syncing with peer {peer}: {str(e)}")
    async def sync_with_connected_peers(self):
        logger.info("Starting sync with connected peers")
        for peer in self.peers.keys():
            try:
                logger.info(f"Attempting to sync with peer: {peer}")
                peer_data = await self.send_and_wait_for_response(peer, Message(MessageType.GET_ALL_DATA.value, {}))
                if peer_data and isinstance(peer_data.payload, dict):
                    await self.blockchain.sync_with_peer(peer_data.payload)
                    logger.info(f"Successfully synced with peer: {peer}")
                else:
                    logger.warning(f"Received invalid data from peer: {peer}")
            except Exception as e:
                logger.error(f"Error syncing with peer {peer}: {str(e)}")
                logger.error(traceback.format_exc())
        logger.info("Finished sync with connected peers")



    async def sync_wallets(self, peer):
        try:
            logger.info(f"Starting wallet sync with peer {peer}")
            message = Message(type=MessageType.GET_WALLETS.value, payload={})
            response = await self.send_and_wait_for_response(peer, message)
            
            if response and response.type == MessageType.WALLETS.value:
                new_wallets = response.payload.get('wallets', [])
                for wallet_data in new_wallets:
                    wallet = Wallet.from_dict(wallet_data)
                    self.blockchain.add_wallet(wallet)
                logger.info(f"Synced {len(new_wallets)} wallets from peer {peer}")
            else:
                logger.warning(f"Failed to receive wallet data from peer {peer}")
        except Exception as e:
            logger.error(f"Error syncing wallets from peer {peer}: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_get_wallets(self, sender):
        try:
            wallets = self.blockchain.get_wallets()
            wallet_data = [wallet.to_dict() for wallet in wallets]
            response = Message(type=MessageType.WALLETS.value, payload={"wallets": wallet_data})
            await self.send_message(sender, response)
        except Exception as e:
            logger.error(f"Error handling get_wallets request: {str(e)}")
            logger.error(traceback.format_exc())
    async def propagate_block(self, block: QuantumBlock) -> bool:
        try:
            logger.info(f"Propagating block with hash: {block.hash}")
            self.logger.debug(f"Current connected peers: {self.connected_peers}")

            
            message = Message(
                type=MessageType.BLOCK.value,
                payload=block.to_dict()
            )
            
            logger.debug(f"Created block message: {message.to_json()}")
            
            if not self.connected_peers:
                logger.warning("No active peers available for block propagation")
                return False

            logger.info(f"Attempting to propagate block to {len(self.connected_peers)} active peers")
            
            await self.broadcast(message)
            logger.info(f"Block {block.hash} propagated to peers")
            return True

        except Exception as e:
            logger.error(f"Error propagating block {block.hash}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    async def send_block_to_peer(self, peer: str, message: Message) -> bool:
        try:
            await self.send_message(peer, message)
            logger.debug(f"Block sent to peer: {peer}")
            return True
        except Exception as e:
            logger.error(f"Failed to send block to peer {peer}: {str(e)}")
            return False

    def set_blockchain(self, blockchain):
        self.blockchain = blockchain
    def normalize_peer_address(self, peer_ip: str, peer_port: Optional[int] = None) -> str:
        """
        Normalize the peer address to a consistent format 'ip:port'.

        Args:
            peer_ip (str): The IP address of the peer or combined 'ip:port' string.
            peer_port (int, optional): The port number of the peer.

        Returns:
            str: Normalized peer address in 'ip:port' format.
        """
        if peer_port is None:
            # If peer_port is not provided, assume peer_ip is a combined 'ip:port' string
            peer_ip, peer_port = peer_ip.split(':')
            peer_port = int(peer_port)
        
        return f"{peer_ip.strip().lower()}:{peer_port}"


    def encrypt_message(self, message: str, recipient_public_key) -> str:
        try:
            # Check if the public key is passed as a string and convert it
            if isinstance(recipient_public_key, str):
                self.logger.debug(f"Recipient public key is provided as string, converting to public key object.")
                recipient_public_key = serialization.load_pem_public_key(
                    recipient_public_key.encode(),
                    backend=default_backend()
                )

            # Log the recipient's public key type and additional details
            self.logger.debug(f"Recipient public key type: {type(recipient_public_key)}")

            # Encode the message to bytes and log the length
            message_bytes = message.encode('utf-8')
            self.logger.debug(f"Message length: {len(message_bytes)} bytes")

            # Log part of the message being encrypted (first 100 characters for privacy)
            self.logger.debug(f"Encrypting message (first 100 chars): {message[:100]}...")

            # Split the message into manageable chunks for RSA encryption
            chunk_size = 190  # RSA 2048-bit key with OAEP padding supports up to ~190 bytes
            chunks = [message_bytes[i:i + chunk_size] for i in range(0, len(message_bytes), chunk_size)]

            # Log chunk details
            self.logger.debug(f"Message split into {len(chunks)} chunks of size {chunk_size} bytes.")

            # Encrypt each chunk and collect them
            encrypted_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    encrypted_chunk = recipient_public_key.encrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    encrypted_chunks.append(encrypted_chunk)
                    self.logger.debug(f"Successfully encrypted chunk {i + 1} of size {len(chunk)} bytes.")
                except Exception as chunk_error:
                    self.logger.error(f"Failed to encrypt chunk {i + 1}: {str(chunk_error)}")
                    raise

            # Combine encrypted chunks and encode them in base64
            encrypted_message = base64.b64encode(b''.join(encrypted_chunks)).decode('utf-8')
            self.logger.debug(f"Final encrypted message length: {len(encrypted_message)} bytes.")

            # Return the base64-encoded encrypted message
            return encrypted_message

        except Exception as e:
            # Log the full error with traceback for debugging
            self.logger.error(f"Encryption failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise ValueError(f"Encryption failed: {str(e)}")





    async def get_peer_public_key(self, peer: str):
        normalized_peer = self.normalize_peer_address(peer)
        if normalized_peer not in self.peer_public_keys:
            await self.exchange_public_keys(peer)
        return self.peer_public_keys.get(normalized_peer)
    async def maintain_peer_connections(self):
        while True:
            try:
                current_peers = list(self.peers.keys())
                for peer in current_peers:
                    if not await self.is_peer_connected(peer):
                        self.logger.warning(f"Peer {peer} is not responsive, removing and attempting to find new peers")
                        await self.remove_peer(peer)
                
                if len(self.peers) < self.target_peer_count:
                    await self.find_new_peers()
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in maintain_peer_connections: {str(e)}")
                await asyncio.sleep(60)


    async def find_new_peers(self):
        try:
            discovered_nodes = await self.node_directory.discover_nodes()
            for node in discovered_nodes:
                if node['address'] not in self.peers and len(self.peers) < self.target_peer_count:
                    await self.connect_to_peer(KademliaNode(node['node_id'], node['ip'], node['port']))
        except Exception as e:
            self.logger.error(f"Error finding new peers: {str(e)}")
    async def complete_connection(self, peer: str) -> bool:
        """Complete connection setup with proper state verification."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.info(f"[COMPLETE] Completing connection with peer {peer_normalized}")

            # Wait for verified state with timeout
            start_time = time.time()
            timeout = 30.0  # 30 seconds timeout
            
            while time.time() - start_time < timeout:
                async with self.peer_lock:
                    state = self.peer_states.get(peer_normalized)
                    self.logger.debug(f"[COMPLETE] Current state: {state}")
                    
                    if state == "verified":
                        # Update peer state
                        self.peer_states[peer_normalized] = "connected"
                        if peer_normalized not in self.connected_peers:
                            self.connected_peers.add(peer_normalized)
                        
                        # Start message handler for this peer
                        websocket = self.peers.get(peer_normalized)
                        if websocket:
                            asyncio.create_task(self.handle_messages(websocket, peer_normalized))
                            asyncio.create_task(self.keep_connection_alive(websocket, peer_normalized))

                        # Initialize quantum components if available
                        if hasattr(self, 'quantum_sync') and self.quantum_initialized:
                            await self.establish_quantum_entanglement(peer_normalized)

                        self.logger.info(f"[COMPLETE] ✓ Connection completed with peer {peer_normalized}")
                        return True
                    
                    elif state == "challenge_response_sent":
                        # Still waiting for verification
                        await asyncio.sleep(0.1)
                        continue
                        
                    else:
                        self.logger.warning(f"[COMPLETE] Cannot complete connection - invalid state: {state}")
                        return False

            self.logger.error(f"[COMPLETE] Timeout waiting for verification from {peer_normalized}")
            return False

        except Exception as e:
            self.logger.error(f"[COMPLETE] Error completing connection: {str(e)}")
            return False
    async def sync_transactions(self, peer):
        try:
            # Request transactions from the peer
            message = Message(type=MessageType.GET_TRANSACTIONS.value, payload={})
            response = await self.send_and_wait_for_response(peer, message)
            
            new_transactions = []  # Initialize new_transactions
            if response and response.type == MessageType.TRANSACTIONS.value:
                new_transactions = response.payload.get('transactions', [])
            
            # Process and add new transactions to the blockchain
            for tx_data in new_transactions:
                tx = Transaction.from_dict(tx_data)
                if await self.blockchain.validate_transaction(tx):
                    await self.blockchain.add_transaction(tx)
            
            logger.info(f"Synced {len(new_transactions)} transactions from peer {peer}")
        except Exception as e:
            logger.error(f"Error syncing transactions from peer {peer}: {str(e)}")
    async def sync_wallets(self, peer):
        try:
            # Request wallet data from the peer
            message = Message(type=MessageType.GET_WALLETS.value, payload={})
            response = await self.send_and_wait_for_response(peer, message)
            
            new_wallets = []  # Initialize new_wallets
            if response and response.type == MessageType.WALLETS.value:
                new_wallets = response.payload.get('wallets', [])
            
            # Process and add new wallets to the blockchain
            for wallet_data in new_wallets:
                wallet = Wallet.from_dict(wallet_data)
                self.blockchain.add_wallet(wallet)
            
            logger.info(f"Synced {len(new_wallets)} wallets from peer {peer}")
        except Exception as e:
            logger.error(f"Error syncing wallets from peer {peer}: {str(e)}")
    def start_peer_tasks(self, peer):
        # Start any background tasks specific to this peer
        # For example, you might want to start a task to periodically check the peer's status
        asyncio.create_task(self.periodic_peer_check())  # Remove the peer argument
    async def find_and_connect_to_new_peers(self):
        try:
            self.logger.info("[FIND_PEERS] Attempting to find and connect to new peers")
            nodes = await self.find_node(self.node_id)
            for node in nodes:
                if node.id not in self.peers and len(self.peers) < self.target_peer_count:
                    self.logger.info(f"[FIND_PEERS] Attempting to connect to new peer: {node.id}")
                    await self.connect_to_peer(node)
            self.logger.info(f"[FIND_PEERS] Finished attempt to find new peers. Current peer count: {len(self.peers)}")
        except Exception as e:
            self.logger.error(f"[FIND_PEERS] Error finding new peers: {str(e)}")
            self.logger.error(traceback.format_exc())
    async def periodic_peer_check(self):
        self.logger.info("[PEER_CHECK] Periodic peer check started")

        while True:
            try:
                self.logger.info("[PEER_CHECK] Starting periodic peer check")
                async with self.peer_lock:
                    for peer in list(self.peers.keys()):
                        peer_state = self.peer_states.get(peer)
                        self.logger.info(f"[PEER_CHECK] Checking peer {peer}, current state: {peer_state}")
                        
                        if peer_state != "connected":
                            if await self.is_peer_connected(peer):
                                self.logger.info(f"[PEER_CHECK] Peer {peer} is responsive but not marked as connected. Finalizing connection.")
                                await self.finalize_connection(peer)
                            else:
                                self.logger.warning(f"[PEER_CHECK] Peer {peer} is not responsive. Removing.")
                                await self.remove_peer(peer)
                        else:
                            # Even for connected peers, perform a quick check
                            if not await self.is_peer_connected(peer):
                                self.logger.warning(f"[PEER_CHECK] Connected peer {peer} is not responsive. Removing.")
                                await self.remove_peer(peer)

                # After checking all peers, attempt to connect to new ones if needed
                if len(self.peers) < self.target_peer_count:
                    self.logger.info(f"[PEER_CHECK] Current peer count ({len(self.peers)}) is below target ({self.target_peer_count}). Attempting to find new peers.")
                    await self.find_and_connect_to_new_peers()

                self.log_peer_status()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"[PEER_CHECK] Error in periodic peer check: {str(e)}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(30)
    async def send_handshake(self, peer: str):
        """Send handshake message with proper identification."""
        try:
            handshake_data = {
                "node_id": self.node_id,
                "version": "1.0",
                "blockchain_height": len(self.blockchain.chain) if self.blockchain else 0,
                "capabilities": ["quantum", "dagknight", "zkp"],
                "timestamp": time.time(),
                "role": "server" if peer in self.peers else "client"
            }

            message = Message(
                type=MessageType.HANDSHAKE.value,
                payload=handshake_data,
                sender=self.node_id  # Add sender ID
            )

            await self.send_raw_message(peer, message)
            self.logger.info(f"[HANDSHAKE] Handshake sent to {peer}")
            
            # Update peer state
            async with self.peer_lock:
                self.peer_states[peer] = "handshake_sent"

        except Exception as e:
            self.logger.error(f"Error sending handshake to {peer}: {str(e)}")
            raise


    def validate_public_key(self, public_key: str, expected_length: int = 294) -> bool:
        try:
            # Strip any surrounding whitespace
            public_key = public_key.strip()

            # Check for valid PEM format headers and footers
            if not public_key.startswith("-----BEGIN PUBLIC KEY-----") or not public_key.endswith("-----END PUBLIC KEY-----"):
                logger.error("Public key does not have valid PEM format headers/footers.")
                return False

            # Extract the base64-encoded content between the headers
            key_content = public_key.replace("-----BEGIN PUBLIC KEY-----", "").replace("-----END PUBLIC KEY-----", "").strip()

            # Clean up the base64 content by removing any unintended newlines or extra spaces
            key_content = key_content.replace("\n", "").replace("\r", "").strip()

            # Log the cleaned base64 content for debugging
            logger.debug(f"Cleaned Base64 key content: {key_content}")

            # Check for valid base64 characters
            if not re.match(r'^[A-Za-z0-9+/=]+$', key_content):
                logger.error("Public key contains invalid base64 characters.")
                return False

            # Try decoding the base64 content to verify it's valid
            decoded_key = base64.b64decode(key_content, validate=True)

            # Optionally, check for key length
            if len(decoded_key) < expected_length:
                logger.error(f"Public key is shorter than the expected length: {len(decoded_key)} < {expected_length}.")
                return False

            return True

        except (ValueError, base64.binascii.Error) as e:
            logger.error(f"Public key validation error: {str(e)}")
            return False

    async def exchange_public_keys(self, peer):
        """Exchange public keys with enhanced message handling and error recovery."""
        try:
            self.logger.info(f"\n[KEY_EXCHANGE] {'='*20} Starting {'='*20}")
            self.logger.info(f"[KEY_EXCHANGE] Peer: {peer}")

            # Parse and normalize peer address
            try:
                peer_ip, peer_port = peer.split(':')
                peer_port = int(peer_port)
                peer_normalized = self.normalize_peer_address(peer_ip, peer_port)
                self.logger.debug(f"[KEY_EXCHANGE] Normalized address: {peer_normalized}")
            except ValueError:
                self.logger.error(f"[KEY_EXCHANGE] Invalid peer address format: {peer}")
                return False

            # Get websocket
            websocket = self.peers.get(peer_normalized)
            if not websocket or websocket.closed:
                self.logger.error(f"[KEY_EXCHANGE] No active connection for {peer_normalized}")
                return False

            # Check if this is a bootstrap node
            is_bootstrap = peer_normalized in self.bootstrap_nodes
            role = "client" if is_bootstrap else "server"
            self.logger.info(f"[KEY_EXCHANGE] Role: {role} (Bootstrap: {is_bootstrap})")

            # Initialize message processing state
            exchange_complete = False
            key_received = asyncio.Event()
            
            # Prepare public key
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()

            key_message = Message(
                type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                payload={
                    "public_key": public_key_pem,
                    "node_id": self.node_id,
                    "role": role
                }
            )

            try:
                if role == "client":
                    # Send our key first
                    self.logger.debug("[KEY_EXCHANGE] Client sending public key")
                    await self.send_raw_message(peer_normalized, key_message)
                    self.logger.debug("[KEY_EXCHANGE] Client sent public key")

                    # Wait for server's key
                    while not exchange_complete:
                        try:
                            raw_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            message = Message.from_json(raw_response)
                            self.logger.debug(f"[KEY_EXCHANGE] Client received message type: {message.type}")

                            if message.type == MessageType.PUBLIC_KEY_EXCHANGE.value:
                                if await self.handle_public_key_exchange(message, peer_normalized):
                                    exchange_complete = True
                                    self.logger.info("[KEY_EXCHANGE] ✓ Client exchange completed")
                                    key_received.set()
                                    break
                            elif message.type == MessageType.HANDSHAKE.value:
                                # Process handshake messages immediately
                                await self.handle_handshake(peer_normalized, message.payload)
                            else:
                                # Process other messages through normal handler
                                await self.handle_message(message, peer_normalized)

                        except asyncio.TimeoutError:
                            self.logger.error("[KEY_EXCHANGE] Client timeout waiting for server key")
                            return False

                else:  # Server role
                    # Wait for client's key first
                    while not exchange_complete:
                        try:
                            raw_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                            message = Message.from_json(raw_response)
                            self.logger.debug(f"[KEY_EXCHANGE] Server received message type: {message.type}")

                            if message.type == MessageType.PUBLIC_KEY_EXCHANGE.value:
                                if await self.handle_public_key_exchange(message, peer_normalized):
                                    # Send our key
                                    await self.send_raw_message(peer_normalized, key_message)
                                    exchange_complete = True
                                    self.logger.info("[KEY_EXCHANGE] ✓ Server exchange completed")
                                    key_received.set()
                                    break
                            elif message.type == MessageType.HANDSHAKE.value:
                                # Process handshake messages immediately
                                await self.handle_handshake(peer_normalized, message.payload)
                            else:
                                # Process other messages through normal handler
                                await self.handle_message(message, peer_normalized)

                        except asyncio.TimeoutError:
                            self.logger.error("[KEY_EXCHANGE] Server timeout waiting for client key")
                            return False

                # Wait for key exchange to complete
                try:
                    await asyncio.wait_for(key_received.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    self.logger.error(f"[KEY_EXCHANGE] Key exchange timeout for {peer_normalized}")
                    return False

                return exchange_complete

            except websockets.exceptions.ConnectionClosed as e:
                self.logger.error(f"[KEY_EXCHANGE] Connection closed: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"[KEY_EXCHANGE] Fatal error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

        finally:
            self.logger.info(f"[KEY_EXCHANGE] {'='*50}\n")
    async def send_public_key(self, peer: str):
        """Send initial public key exchange with proper role assignment."""
        try:
            # Convert the public key to PEM format
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()

            # Create the public key exchange message with role
            is_server = peer in self.peers  # If peer exists in self.peers, we're responding to their connection
            message = Message(
                type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                payload={
                    "public_key": public_key_pem,
                    "node_id": self.node_id,
                    "role": "server" if is_server else "client"
                }
            )

            # Send the message
            await self.send_raw_message(peer, message)
            self.logger.info(f"Public key sent to {peer} as {message.payload['role']}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send public key to {peer}: {str(e)}")
            return False
    async def send_challenge(self, peer: str) -> Optional[str]:
        """Send challenge with single challenge enforcement."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            if not peer_normalized:
                raise ValueError(f"Invalid peer address: {peer}")

            self.logger.info(f"\n[CHALLENGE] {'='*20} Sending Challenge {'='*20}")
            self.logger.info(f"[CHALLENGE] Peer: {peer_normalized}")

            # Check existing challenges
            async with self.peer_lock:
                current_state = self.peer_states.get(peer_normalized)
                
                # Return existing challenge if one exists
                if current_state == "challenge_sent":
                    active_challenges = await self.challenge_manager.get_active_challenges(peer_normalized)
                    if active_challenges:
                        challenge_id = next(iter(active_challenges.keys()))
                        self.logger.info(f"[CHALLENGE] Using existing challenge: {challenge_id}")
                        return challenge_id

                # Verify state is valid for new challenge
                if current_state not in ['connecting', 'key_exchanged']:
                    self.logger.error(f"[CHALLENGE] Invalid state for challenge: {current_state}")
                    return None

                self.peer_states[peer_normalized] = "processing_challenge"

            # Create and store new challenge
            try:
                challenge_id, challenge = await self.create_challenge()
                if not challenge_id or not challenge:
                    raise RuntimeError("Failed to create challenge")

                stored = await self.challenge_manager.store_challenge(
                    peer_normalized,
                    challenge_id,
                    challenge,
                    ChallengeRole.SERVER
                )
                
                if not stored:
                    raise RuntimeError("Failed to store challenge")

                self.logger.debug(f"[CHALLENGE] New challenge created: {challenge_id}")

            except Exception as e:
                self.logger.error(f"[CHALLENGE] Challenge creation failed: {str(e)}")
                async with self.peer_lock:
                    self.peer_states[peer_normalized] = "key_exchanged"
                return None

            # Send challenge message
            message = Message(
                type=MessageType.CHALLENGE.value,
                payload={
                    'challenge': f"{challenge_id}:{challenge}",
                    'role': ChallengeRole.SERVER.value
                },
                challenge_id=challenge_id
            )

            # Send with retries
            for attempt in range(3):
                try:
                    await asyncio.wait_for(
                        self.send_raw_message(peer_normalized, message),
                        timeout=10.0
                    )
                    
                    async with self.peer_lock:
                        self.peer_states[peer_normalized] = "challenge_sent"

                    # Start timeout monitor
                    timeout_task = asyncio.create_task(
                        self.monitor_challenge_timeout(peer_normalized, challenge_id)
                    )
                    
                    self.logger.info(f"[CHALLENGE] ✓ Challenge sent successfully")
                    return challenge_id

                except Exception as e:
                    self.logger.warning(f"[CHALLENGE] Send attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            # If all retries failed
            await self.cleanup_failed_challenge(peer_normalized, challenge_id)
            return None

        except Exception as e:
            self.logger.error(f"[CHALLENGE] Error sending challenge: {str(e)}")
            if peer_normalized and challenge_id:
                await self.cleanup_failed_challenge(peer_normalized, challenge_id)
            return None

    async def cleanup_failed_challenge(self, peer: str, challenge_id: str):
        """Clean up resources for failed challenge."""
        try:
            # Remove challenge state
            await self.challenge_manager.remove_challenge(peer, challenge_id)
            
            # Cancel timeout task if exists
            if hasattr(self, 'challenge_timeout_tasks') and challenge_id in self.challenge_timeout_tasks:
                task = self.challenge_timeout_tasks.pop(challenge_id)
                if not task.done():
                    task.cancel()
                    
            # Update peer state if needed
            async with self.peer_lock:
                if self.peer_states.get(peer) == "challenge_sent":
                    self.peer_states[peer] = "key_exchanged"
                    
        except Exception as e:
            self.logger.error(f"[CLEANUP] Error cleaning up failed challenge: {str(e)}")

    async def monitor_challenge_timeout(self, peer: str, challenge_id: str):
        """Monitor challenge response timeout with proper async management."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            await asyncio.sleep(30)  # 30 second timeout
            
            # Get challenge state
            challenge = await self.challenge_manager.get_challenge(peer_normalized, challenge_id)
            
            if challenge and not challenge.verified:
                self.logger.warning(f"[CHALLENGE] Timeout for peer {peer_normalized}")
                # Remove challenge
                await self.challenge_manager.remove_challenge(peer_normalized, challenge_id)
                # Remove peer connection
                await self.remove_peer(peer_normalized)
                
        except Exception as e:
            self.logger.error(f"Error monitoring challenge timeout: {str(e)}")


    async def handle_challenge_timeout(self, peer: str, challenge_id: str):
        """Handle challenge timeout."""
        try:
            # Clean up challenge data
            if peer in self.challenges:
                self.challenges[peer].pop(challenge_id, None)
                if not self.challenges[peer]:
                    del self.challenges[peer]

            # Update peer state
            async with self.peer_lock:
                if self.peer_states.get(peer) == "challenge_sent":
                    self.peer_states[peer] = "timeout"

            # Remove peer if necessary
            await self.remove_peer(peer)
            
        except Exception as e:
            self.logger.error(f"Error handling challenge timeout: {str(e)}")



    def generate_node_id(self) -> str:
        """Generate a unique node ID."""
        if hasattr(self, 'host') and hasattr(self, 'port'):
            return hashlib.sha1(f"{self.host}:{self.port}".encode()).hexdigest()
        return hashlib.sha1(os.urandom(20)).hexdigest()


    def generate_magnet_link(self) -> MagnetLink:
        info_hash = self.node_id
        peer_id = base64.b64encode(f"{self.host}:{self.port}".encode()).decode()
        return MagnetLink(info_hash, [], peer_id)
    async def start(self):
        """Enhanced start method with quantum component initialization, bootstrap management, and detailed logging."""
        try:
            self.logger.info("\n=== Starting P2P Node ===")

            # Step 1: Initialize Quantum Components
            try:
                self.logger.info("[QUANTUM] Initializing quantum components...")

                if not hasattr(self, 'quantum_sync') or self.quantum_sync is None:
                    self.quantum_sync = QuantumEntangledSync(self.node_id)
                
                # Initialize with blockchain data
                initial_data = {
                    "wallets": [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else [],
                    "transactions": [tx.to_dict() for tx in self.blockchain.get_recent_transactions(limit=100)] if self.blockchain else [],
                    "blocks": [block.to_dict() for block in self.blockchain.chain] if self.blockchain else [],
                    "mempool": [tx.to_dict() for tx in self.blockchain.mempool] if hasattr(self.blockchain, '_mempool') else []
                }

                await self.quantum_sync.initialize_quantum_state(initial_data)
                self.quantum_initialized = True
                self.logger.info("[QUANTUM] ✓ Quantum components initialized successfully")

            except Exception as e:
                self.logger.error("[QUANTUM] Quantum component initialization failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise RuntimeError("Quantum component initialization failed") from e

            # Step 2: Initialize Bootstrap Manager
            self.bootstrap_manager = BootstrapManager(self)
            
            # Step 3: Attempt to Connect to Bootstrap Nodes
            if self.bootstrap_nodes:
                try:
                    self.logger.info("Connecting to Bootstrap Nodes")
                    bootstrap_success = await self.bootstrap_manager.connect_to_bootstrap_nodes()
                    if not bootstrap_success:
                        self.logger.warning("Failed to connect to bootstrap network, continuing as standalone node")
                except Exception as e:
                    self.logger.error("Error connecting to bootstrap nodes: {str(e)}")
                    self.logger.error(traceback.format_exc())

            # Additional steps (e.g., WebSocket server, background tasks) here...

            self.logger.info("✓ Node started successfully")

        except Exception as e:
            self.logger.error(f"Fatal error during node startup: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise


    async def cleanup(self):
        """Cleanup resources before shutdown."""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.logger.info("WebSocket server closed")
            
            self.is_running = False
            
            # Close all peer connections
            for peer in list(self.peers.keys()):
                await self.remove_peer(peer)
                
            self.logger.info("P2P node cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during P2P node cleanup: {str(e)}")
    async def shutdown(self, sig=None):
        """Gracefully shutdown the node."""
        if sig:
            self.logger.info(f"Received exit signal {sig.name}...")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        
        # Cleanup resources
        await self.cleanup()
        
        self.logger.info("Node shutdown complete")
    async def monitor_connections(self):
        """Monitor connection health and status."""
        while self.is_running:
            try:
                active_peers = len(self.connected_peers)
                total_peers = len(self.peers)
                
                self.logger.info(f"\n=== Connection Status ===")
                self.logger.info(f"Active peers: {active_peers}/{total_peers}")
                self.logger.info(f"Connected peers: {list(self.connected_peers)}")
                
                # Check quantum entanglement status
                if hasattr(self, 'quantum_sync'):
                    entangled_peers = len(self.quantum_sync.entangled_peers)
                    self.logger.info(f"Quantum entangled peers: {entangled_peers}")
                    
                    # Check quantum state fidelities
                    for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                        try:
                            fidelity = await self.quantum_sync.measure_sync_state(component)
                            self.logger.debug(f"{component} quantum fidelity: {fidelity:.3f}")
                        except Exception as e:
                            self.logger.error(f"Error measuring {component} fidelity: {str(e)}")
                
                # Trigger reconnection if needed
                if active_peers < self.target_peer_count:
                    self.logger.info("Below target peer count, triggering peer discovery")
                    asyncio.create_task(self.find_and_connect_to_new_peers())
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in connection monitoring: {str(e)}")
                await asyncio.sleep(60)
        # Add periodic quantum heartbeat sender
        async def periodic_quantum_heartbeats(self):
            """Send quantum heartbeats periodically to all peers"""
            while self.is_running:
                try:
                    self.logger.info("[QUANTUM] Starting quantum heartbeat round")
                    for peer in self.quantum_sync.entangled_peers:
                        try:
                            await self.send_quantum_heartbeats(peer)
                            self.logger.info(f"✓ Sent quantum heartbeat to {peer}")
                        except Exception as e:
                            self.logger.error(f"Failed to send quantum heartbeat to {peer}: {str(e)}")
                    
                    await asyncio.sleep(self.quantum_heartbeat_interval)  # Default 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in quantum heartbeat loop: {str(e)}")
                    await asyncio.sleep(self.quantum_heartbeat_interval)
    async def send_quantum_heartbeat(self, peer: str) -> bool:
        """Send quantum heartbeat to a peer with detailed logging"""
        try:
            self.logger.info(f"\n[QUANTUM] {'='*20} Quantum Heartbeat {'='*20}")
            self.logger.info(f"[QUANTUM] Sending quantum heartbeat to peer {peer}")

            # Get current quantum states
            quantum_states = {}
            fidelities = {}
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                try:
                    qubit = getattr(self.quantum_sync.register, component)
                    quantum_states[component] = qubit.value
                    fidelity = await self.quantum_sync.measure_sync_state(component)
                    fidelities[component] = fidelity
                    self.logger.debug(f"[QUANTUM] {component} state: {qubit.value}")
                    self.logger.debug(f"[QUANTUM] {component} fidelity: {fidelity:.3f}")
                except Exception as e:
                    self.logger.error(f"[QUANTUM] Error getting state for {component}: {str(e)}")
                    return False

            # Get Bell pair
            bell_pair = self.quantum_sync.bell_pairs.get(peer)
            if not bell_pair:
                self.logger.warning(f"[QUANTUM] No Bell pair found for peer {peer}")
                return False
            
            bell_pair_id = self.quantum_notifier._get_bell_pair_id(bell_pair)
            self.logger.debug(f"[QUANTUM] Using Bell pair ID: {bell_pair_id[:16]}...")

            # Create heartbeat
            heartbeat = QuantumHeartbeat(
                node_id=self.node_id,
                timestamp=time.time(),
                quantum_states=quantum_states,
                fidelities=fidelities,
                bell_pair_id=bell_pair_id,
                nonce=os.urandom(16).hex()
            )

            # Sign the heartbeat
            message_str = (
                f"{heartbeat.node_id}:{heartbeat.timestamp}:{heartbeat.bell_pair_id}:"
                f"{heartbeat.nonce}"
            ).encode()
            
            signature = self.private_key.sign(
                message_str,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            heartbeat.signature = base64.b64encode(signature).decode()

            # Create and send message
            message = Message(
                type=MessageType.QUANTUM_HEARTBEAT.value,
                payload=heartbeat.to_dict()
            )
            await self.send_message(peer, message)
            
            # Update tracking
            self.last_quantum_heartbeat[peer] = time.time()
            
            self.logger.info(f"[QUANTUM] ✓ Quantum heartbeat sent successfully to {peer}")
            self.logger.debug(f"[QUANTUM] Current fidelities: {fidelities}")
            self.logger.info(f"[QUANTUM] {'='*50}\n")
            
            return True

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error sending quantum heartbeat to {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    async def periodic_quantum_heartbeats(self):
        """Continuously send quantum heartbeats to all entangled peers."""
        while self.is_running:
            try:
                for peer in list(self.quantum_sync.entangled_peers.keys()):
                    try:
                        await self.send_quantum_heartbeat(peer)
                        await asyncio.sleep(1)  # Small delay between peers
                    except Exception as e:
                        self.logger.error(f"Error sending heartbeat to {peer}: {str(e)}")
                
                await asyncio.sleep(self.quantum_heartbeat_interval)  # Wait before next round
                
            except Exception as e:
                self.logger.error(f"Error in quantum heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause on error

    async def send_quantum_heartbeats(self, peer: str):
        """Send quantum heartbeat to a specific peer"""
        try:
            self.logger.info(f"Initiating quantum heartbeat send to peer {peer}")

            # Step 1: Get current quantum states and fidelities for each component
            quantum_states = {}
            fidelities = {}
            
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                try:
                    # Retrieve quantum state
                    qubit = getattr(self.quantum_sync.register, component)
                    quantum_states[component] = qubit.value
                    self.logger.debug(f"Quantum state for {component}: {qubit.value}")

                    # Measure fidelity for synchronization state
                    fidelities[component] = await self.quantum_sync.measure_sync_state(component)
                    self.logger.debug(f"Fidelity for {component}: {fidelities[component]}")

                except AttributeError as ae:
                    self.logger.error(f"Failed to retrieve quantum state or fidelity for {component}: {str(ae)}")
                    raise

            # Step 2: Get Bell pair ID for the peer
            try:
                bell_pair_id = self.quantum_notifier._get_bell_pair_id(self.quantum_sync.bell_pairs[peer])
                self.logger.debug(f"Retrieved Bell pair ID for peer {peer}: {bell_pair_id}")
            except KeyError as ke:
                self.logger.error(f"Bell pair ID not found for peer {peer}: {str(ke)}")
                raise

            # Step 3: Create QuantumHeartbeat instance
            heartbeat = QuantumHeartbeat(
                node_id=self.node_id,
                timestamp=time.time(),
                quantum_states=quantum_states,
                fidelities=fidelities,
                bell_pair_id=bell_pair_id,
                nonce=self.quantum_notifier._generate_nonce()
            )
            self.logger.debug(f"Quantum heartbeat created: {heartbeat}")

            # Step 4: Sign the heartbeat
            try:
                message_str = (
                    f"{heartbeat.node_id}:{heartbeat.timestamp}:{heartbeat.bell_pair_id}:"
                    f"{heartbeat.nonce}"
                ).encode()
                
                signature = self.private_key.sign(
                    message_str,
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                heartbeat.signature = base64.b64encode(signature).decode()
                self.logger.debug(f"Quantum heartbeat signed with signature: {heartbeat.signature}")
            except Exception as sign_error:
                self.logger.error(f"Error signing quantum heartbeat for peer {peer}: {str(sign_error)}")
                raise

            # Step 5: Prepare and send the quantum heartbeat message
            message = Message(
                type=MessageType.QUANTUM_HEARTBEAT.value,
                payload=heartbeat.to_dict()
            )
            self.logger.debug(f"Quantum heartbeat message prepared for peer {peer}: {message}")

            await self.send_message(peer, message)
            self.logger.info(f"Quantum heartbeat successfully sent to peer {peer}")

        except Exception as e:
            self.logger.error(f"Error sending quantum heartbeat to {peer}: {str(e)}")


    async def run_quantum_heartbeats(self):
        """Dedicated quantum heartbeat loop"""
        while self.is_running:
            try:
                if not self.quantum_initialized:
                    self.logger.warning("[QUANTUM] Quantum components not initialized, skipping heartbeats")
                    await asyncio.sleep(5)
                    continue

                self.logger.info(f"\n[QUANTUM] {'='*20} Quantum Heartbeat Round {'='*20}")
                peers = [p for p in self.connected_peers if p in self.quantum_sync.entangled_peers]
                self.logger.info(f"[QUANTUM] Sending quantum heartbeats to {len(peers)} entangled peers")

                for peer in peers:
                    try:
                        await self.send_quantum_heartbeat(peer)
                        await asyncio.sleep(1)  # Small delay between peers
                    except Exception as e:
                        self.logger.error(f"[QUANTUM] Error sending heartbeat to {peer}: {str(e)}")

                self.logger.info(f"[QUANTUM] {'='*50}\n")
                await asyncio.sleep(self.quantum_heartbeat_interval)

            except Exception as e:
                self.logger.error(f"[QUANTUM] Error in quantum heartbeat loop: {str(e)}")
                await asyncio.sleep(5)


    async def handle_quantum_heartbeat(self, message: Message, sender: str):
        """Handle incoming quantum heartbeat with detailed logging"""
        try:
            self.logger.info(f"\n[QUANTUM] {'='*20} Received Heartbeat {'='*20}")
            self.logger.info(f"[QUANTUM] Processing heartbeat from {sender}")

            # Parse heartbeat data
            heartbeat = QuantumHeartbeat.from_dict(message.payload)
            self.logger.debug(f"[QUANTUM] Heartbeat timestamp: {heartbeat.timestamp}")
            
            # Verify signature
            if not await self.verify_quantum_heartbeat(heartbeat, sender):
                self.logger.warning(f"[QUANTUM] ✗ Invalid heartbeat signature from {sender}")
                return False
            self.logger.debug(f"[QUANTUM] ✓ Signature verified")

            # Verify Bell pair
            bell_pair = self.quantum_sync.bell_pairs.get(sender)
            if not bell_pair:
                self.logger.warning(f"[QUANTUM] ✗ No Bell pair for {sender}")
                return False
            
            local_bell_pair_id = self.quantum_notifier._get_bell_pair_id(bell_pair)
            if heartbeat.bell_pair_id != local_bell_pair_id:
                self.logger.warning(f"[QUANTUM] ✗ Bell pair ID mismatch from {sender}")
                return False
            self.logger.debug(f"[QUANTUM] ✓ Bell pair verified")

            # Check quantum states and fidelities
            decoherent_components = []
            for component, remote_state in heartbeat.quantum_states.items():
                local_state = getattr(self.quantum_sync.register, component).value
                fidelity = self.quantum_sync._calculate_fidelity(local_state, remote_state)
                
                self.logger.debug(f"[QUANTUM] {component}:")
                self.logger.debug(f"  - Local state: {local_state}")
                self.logger.debug(f"  - Remote state: {remote_state}")
                self.logger.debug(f"  - Fidelity: {fidelity:.3f}")
                
                if fidelity < self.quantum_sync.decoherence_threshold:
                    decoherent_components.append(component)
                    self.logger.warning(
                        f"[QUANTUM] Low fidelity detected in {component}: {fidelity:.3f}"
                    )

            # Handle decoherence if detected
            if decoherent_components:
                self.logger.warning(
                    f"[QUANTUM] Decoherence detected in components: {decoherent_components}"
                )
                await self.handle_quantum_decoherence(sender, decoherent_components)
            else:
                self.logger.info(f"[QUANTUM] ✓ All components synchronized")

            # Update tracking
            self.quantum_heartbeats[sender] = heartbeat
            self.last_quantum_heartbeat[sender] = time.time()
            
            # Send response
            await self.send_quantum_heartbeat_response(sender, heartbeat)
            
            self.logger.info(f"[QUANTUM] ✓ Heartbeat processed successfully")
            self.logger.info(f"[QUANTUM] {'='*50}\n")
            
            return True

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error handling heartbeat from {sender}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False



    async def send_quantum_heartbeat_response(self, sender: str, received_heartbeat: QuantumHeartbeat):
        """Send response to quantum heartbeat"""
        try:
            response = Message(
                type=MessageType.QUANTUM_HEARTBEAT_RESPONSE.value,
                payload={
                    'original_nonce': received_heartbeat.nonce,
                    'timestamp': time.time(),
                    'node_id': self.node_id
                }
            )
            await self.send_message(sender, response)
            
        except Exception as e:
            self.logger.error(f"Error sending heartbeat response to {sender}: {str(e)}")

    async def verify_quantum_heartbeat(self, heartbeat: QuantumHeartbeat, sender: str) -> bool:
        """Verify quantum heartbeat signature"""
        try:
            # Get sender's public key
            public_key = self.peer_public_keys.get(sender)
            if not public_key:
                return False

            # Verify signature
            message_str = (
                f"{heartbeat.node_id}:{heartbeat.timestamp}:{heartbeat.bell_pair_id}:"
                f"{heartbeat.nonce}"
            ).encode()
            
            signature = base64.b64decode(heartbeat.signature)
            
            public_key.verify(
                signature,
                message_str,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True

        except Exception:
            return False

    async def monitor_quantum_heartbeats(self):
        """Monitor quantum heartbeats and handle timeouts"""
        while True:
            try:
                current_time = time.time()
                peers_to_check = list(self.quantum_sync.entangled_peers.keys())
                
                for peer in peers_to_check:
                    last_heartbeat = self.last_quantum_heartbeat.get(peer, 0)
                    if current_time - last_heartbeat > self.quantum_heartbeat_timeout:
                        self.logger.warning(f"Quantum heartbeat timeout for peer {peer}")
                        await self.handle_quantum_heartbeat_timeout(peer)

                await asyncio.sleep(self.quantum_heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring quantum heartbeats: {str(e)}")
                await asyncio.sleep(self.quantum_heartbeat_interval)

    async def handle_quantum_heartbeat_timeout(self, peer: str):
        """Handle quantum heartbeat timeout"""
        try:
            self.logger.warning(f"Initiating quantum resync due to heartbeat timeout: {peer}")
            
            # Request resync for all components
            components = ['wallets', 'transactions', 'blocks', 'mempool']
            await self.request_quantum_resync(peer, components)
            
            # If resync fails, remove quantum entanglement
            if not await self.verify_quantum_sync(peer):
                self.logger.error(f"Quantum resync failed for {peer}, removing entanglement")
                await self.remove_quantum_entanglement(peer)

        except Exception as e:
            self.logger.error(f"Error handling quantum heartbeat timeout: {str(e)}")

    async def handle_heartbeat_decoherence(self, peer: str, components: List[str]):
        """Handle decoherence detected in heartbeat"""
        try:
            self.logger.warning(f"Handling heartbeat decoherence for {peer}: {components}")
            
            # Request quantum resync for decoherent components
            await self.request_quantum_resync(peer, components)
            
            # Verify sync was successful
            if not await self.verify_quantum_sync(peer):
                self.logger.error(f"Failed to restore quantum sync with {peer}")
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Error handling heartbeat decoherence: {str(e)}")
            return False

    async def handle_subscribe_orderbook(self, websocket, data):
        # Logic to handle orderbook subscription
        pair = data.get('pair')
        # Subscribe the client to orderbook updates for the specified pair
        # You might want to store this subscription information

    async def handle_subscribe_trades(self, websocket, data):
        # Logic to handle trades subscription
        pair = data.get('pair')
        # Subscribe the client to trade updates for the specified pair
        # You might want to store this subscription information

    async def broadcast_orderbook_update(self, pair, orderbook):
        await self.ws_updates.broadcast({
            'type': 'orderbook_update',
            'pair': pair,
            'data': orderbook
        })

    async def broadcast_trade_update(self, pair, trade):
        await self.ws_updates.broadcast({
            'type': 'trade_update',
            'pair': pair,
            'data': trade
        })
    async def handle_place_order(self, order_data, sender):
        # ... (existing order placement logic) ...

        # After successfully placing the order, broadcast the update
        await self.broadcast_orderbook_update(order_data['pair'], updated_orderbook)

    async def handle_trade(self, trade_data, sender):
        # ... (existing trade handling logic) ...

        # After processing the trade, broadcast the update
        await self.broadcast_trade_update(trade_data['pair'], trade_data)
    async def start(self):
        """Start the P2P node and connect to bootstrap network."""
        await self.initialize_dagknight()

        try:
            self.logger.info("\n=== Starting P2P Node ===")
            
            # Initialize quantum components first
            try:
                self.logger.info("[QUANTUM] Initializing quantum components...")
                
                if not hasattr(self, 'quantum_sync') or self.quantum_sync is None:
                    self.quantum_sync = QuantumEntangledSync(self.node_id)
                    
                # Initialize with blockchain data
                initial_data = {
                    "wallets": [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else [],
                    "transactions": [tx.to_dict() for tx in self.blockchain.get_recent_transactions(limit=100)] if self.blockchain else [],  
                    "blocks": [block.to_dict() for block in self.blockchain.chain] if self.blockchain else [],
                    "mempool": [] if not hasattr(self.blockchain, '_mempool') else [tx.to_dict() for tx in self.blockchain.mempool]
                }

                await self.quantum_sync.initialize_quantum_state(initial_data)
                self.quantum_initialized = True
                self.logger.info("[QUANTUM] ✓ Quantum components initialized")

                # Find available port
                max_port_attempts = 10
                current_port = self.port
                server_started = False
                
                for port_attempt in range(max_port_attempts):
                    try:
                        self.server = await websockets.serve(
                            self.handle_connection, 
                            self.host, 
                            current_port,
                            ping_interval=None
                        )
                        self.port = current_port  # Update port if successful
                        self.logger.info(f"✓ WebSocket server started on {self.host}:{current_port}")
                        server_started = True
                        break
                    except OSError as e:
                        if e.errno == 98:  # Address already in use
                            current_port += 1
                            self.logger.warning(f"Port {current_port-1} in use, trying port {current_port}")
                            if port_attempt == max_port_attempts - 1:
                                raise RuntimeError(f"Could not find available port after {max_port_attempts} attempts")
                        else:
                            raise

                if not server_started:
                    raise RuntimeError("Failed to start WebSocket server")

                # Join network before starting tasks
                self.is_running = True
                network_joined = await self.join_network()
                if network_joined:
                    self.logger.info("✓ Successfully joined P2P network")
                else:
                    self.logger.warning("Failed to join P2P network, running as standalone node")

                # Start background tasks
                tasks = [
                    asyncio.create_task(self.send_heartbeats(), name="heartbeat"),
                    asyncio.create_task(self.run_quantum_heartbeats(), name="quantum_heartbeat"),
                    asyncio.create_task(self.periodic_cleanup(), name="cleanup"),
                    # Update the line in the start method (or relevant section) as follows
                    asyncio.create_task(self.maintain_peer_connections(), name="connections"),
                    asyncio.create_task(self.quantum_monitor.start_monitoring(), name="quantum_monitor"),
                    asyncio.create_task(self.update_active_peers(), name="peer_updates"),
                    asyncio.create_task(self.monitor_network_security())


                ]

                # Log started tasks
                self.logger.info("Started background tasks:")
                for task in tasks:
                    self.logger.info(f"  - {task.get_name()}")

                # Monitor tasks for failures
                asyncio.create_task(self.monitor_background_tasks(tasks))
                
                self.logger.info("✓ All background tasks started")

            except Exception as e:
                self.logger.error(f"[QUANTUM] Failed to initialize quantum components: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise

        except Exception as e:
            self.logger.error(f"Fatal error starting P2P node: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    async def monitor_background_tasks(self, tasks: List[asyncio.Task]):
        """Monitor background tasks and restart them if they fail."""
        while self.is_running:
            for task in tasks[:]:  # Copy list to avoid modification during iteration
                if task.done():
                    try:
                        await task
                    except Exception as e:
                        self.logger.error(f"Task {task.get_name()} failed with error: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        # Restart the task
                        if task.get_name() == "heartbeat":
                            new_task = asyncio.create_task(self.send_heartbeats(), name="heartbeat")
                        elif task.get_name() == "quantum_heartbeat":
                            new_task = asyncio.create_task(self.run_quantum_heartbeats(), name="quantum_heartbeat")
                        elif task.get_name() == "cleanup":
                            new_task = asyncio.create_task(self.periodic_cleanup(), name="cleanup")
                        elif task.get_name() == "connections":
                            new_task = asyncio.create_task(self.maintain_peer_connections(), name="connections")
                        elif task.get_name() == "quantum_monitor":
                            new_task = asyncio.create_task(self.quantum_monitor.start_monitoring(), name="quantum_monitor")
                        elif task.get_name() == "peer_updates":
                            new_task = asyncio.create_task(self.update_active_peers(), name="peer_updates")
                        
                        tasks.remove(task)
                        tasks.append(new_task)
                        self.logger.info(f"Restarted failed task: {task.get_name()}")
                        
            await asyncio.sleep(5)  # Check every 5 seconds


    async def connect_to_peer_list(self, peer_list: List[str], max_connections: int = 5):
        """Connect to a list of peers with limit."""
        connected_count = 0
        for peer_addr in peer_list:
            if connected_count >= max_connections:
                break
                
            try:
                peer_ip, peer_port = peer_addr.split(':')
                peer_node = KademliaNode(
                    id=self.generate_node_id(),
                    ip=peer_ip,
                    port=int(peer_port)
                )
                
                if await self.connect_to_peer(peer_node):
                    connected_count += 1
                    self.logger.info(f"Connected to peer {peer_addr}")
                
            except Exception as e:
                self.logger.error(f"Error connecting to peer {peer_addr}: {str(e)}")
                continue

        return connected_count

    def load_bootstrap_nodes(self):
        """Load bootstrap nodes from environment variables."""
        bootstrap_nodes = os.getenv('BOOTSTRAP_NODES', '').split(',')
        return [node.strip() for node in bootstrap_nodes if node.strip()]


    async def update_active_peers(self):
        while True:
            try:
                async with self.peer_lock:
                    for peer in list(self.peers.keys()):
                        if await self.is_peer_connected(peer):
                            if peer not in self.connected_peers:
                                self.connected_peers.add(peer)
                                self.logger.info(f"Added {peer} to active peers list")
                        else:
                            if peer in self.connected_peers:
                                self.connected_peers.remove(peer)
                                self.logger.info(f"Removed {peer} from active peers list")
                
                self.logger.info(f"Updated active peers list. Current active peers: {list(self.connected_peers)}")
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"Error updating active peers: {str(e)}")
                await asyncio.sleep(60)

    async def bootstrap_from_connected_peers(self):
        logger.info("Bootstrapping from connected peers...")
        for peer in list(self.peers.keys()):  # Create a copy of the keys to avoid modification during iteration
            try:
                peer_list = await self.request_peer_list(peer)
                logger.info(f"Received peer list from {peer}: {peer_list}")
                for peer_address in peer_list:
                    if peer_address not in self.peers and len(self.peers) < self.max_peers:
                        try:
                            peer_ip, peer_port = peer_address.split(':')
                            peer_node = KademliaNode(self.generate_node_id(), peer_ip, int(peer_port))
                            await self.connect_to_peer(peer_node)
                        except Exception as e:
                            logger.error(f"Failed to connect to peer {peer_address}: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to bootstrap from peer {peer}: {str(e)}")

        logger.info(f"Bootstrap complete. Now connected to {len(self.peers)} peers.")
        self.log_connected_peers()
    async def connect_to_seed_node(self, node: KademliaNode):
        try:
            await self.connect_to_peer(node)
            peer_list = await self.request_peer_list(node.address)
            for peer_address in peer_list:
                peer_ip, peer_port = peer_address.split(':')
                peer_node = KademliaNode(self.generate_node_id(), peer_ip, int(peer_port))
                await self.connect_to_peer(peer_node)
        except Exception as e:
            logger.error(f"Failed to connect to seed node {node.address}: {str(e)}")
    async def request_peer_list(self, peer: str) -> List[str]:
        """Request peer list from a connected peer."""
        try:
            self.logger.debug(f"Requesting peer list from {peer}")
            message = Message(
                type=MessageType.PEER_LIST_REQUEST.value,
                payload={}
            )
            
            response = await self.send_and_wait_for_response(peer, message, timeout=10.0)
            if response and response.type == MessageType.PEER_LIST_RESPONSE.value:
                peer_list = response.payload.get('peers', [])
                self.logger.debug(f"Received {len(peer_list)} peers from {peer}")
                return peer_list
            return []
        except Exception as e:
            self.logger.error(f"Error requesting peer list from {peer}: {str(e)}")
            return []


    async def handle_peer_list_request(self, sender: str):
        active_peers = await self.get_active_peers()
        response = Message(MessageType.PEER_LIST_RESPONSE.value, {"peers": active_peers})
        await self.send_message(sender, response)

    async def is_peer_active(self, peer: str) -> bool:
        try:
            await self.ping_node(peer)
            return True
        except:
            return False


    async def periodic_tasks(self):
        logger.info("Starting periodic tasks...")
        while True:
            try:
                await asyncio.gather(
                    self.refresh_buckets(),
                    self.republish_data(),
                    asyncio.to_thread(self.cleanup_data_store),
                    self.send_heartbeats(),
                    asyncio.to_thread(self.log_connected_peers),
                    self.find_new_peers_if_needed()
                )
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                logger.info("Periodic tasks cancelled. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in periodic tasks: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

        logger.info("Periodic tasks stopped.")

    async def find_new_peers_if_needed(self):
        if len(self.peers) < self.max_peers:
            logger.info(f"Current peer count ({len(self.peers)}) is below maximum ({self.max_peers}). Searching for new peers...")
            await self.find_new_peers()
        else:
            logger.debug(f"Peer count is at maximum ({self.max_peers}). No need to find new peers.")

    async def refresh_buckets(self):
        try:
            for i in range(len(self.buckets)):
                if not self.buckets[i]:
                    random_id = self.generate_random_id_in_bucket(i)
                    await self.find_node(random_id)
        except Exception as e:
            logger.error(f"Failed to refresh buckets: {str(e)}")
            logger.error(traceback.format_exc())

    def generate_random_id_in_bucket(self, bucket_index: int) -> str:
        return format(random.getrandbits(160) | (1 << (159 - bucket_index)), '040x')

    async def republish_data(self):
        try:
            for key, value in self.data_store.items():
                await self.store(key, value)
        except Exception as e:
            logger.error(f"Failed to republish data: {str(e)}")
            logger.error(traceback.format_exc())

    async def cleanup_data_store(self):
        try:
            while len(self.data_store) > self.max_data_store_size:
                self.data_store.popitem(last=False)
        except Exception as e:
            logger.error(f"Failed to clean up data store: {str(e)}")
            logger.error(traceback.format_exc())

    def calculate_distance(self, node_id1: str, node_id2: str) -> int:
        return int(node_id1, 16) ^ int(node_id2, 16)

    def get_bucket_index(self, node_id: str) -> int:
        distance = self.calculate_distance(self.node_id, node_id)
        return (distance.bit_length() - 1) if distance > 0 else 0

    async def add_node_to_bucket(self, node: KademliaNode):
        try:
            bucket_index = self.get_bucket_index(node.id)
            bucket = self.buckets[bucket_index]
            if node not in bucket:
                if len(bucket) < self.k:
                    bucket.append(node)
                else:
                    # Ping the least recently seen node
                    oldest_node = min(bucket, key=lambda n: n.last_seen)
                    if not await self.ping_node(oldest_node):
                        bucket.remove(oldest_node)
                        bucket.append(node)
            node.update_last_seen()
            
            # Update peer_states
            self.peer_states[node.address] = "connected"
            
            self.logger.debug(f"Added/updated node {node.id} in bucket {bucket_index}")
            self.log_peer_status()  # Log updated peer status
        except Exception as e:
            self.logger.error(f"Failed to add node to bucket: {str(e)}")
            self.logger.error(traceback.format_exc())
    def xor_distance(a: str, b: str) -> int:
        """
        Calculate the XOR distance between two node IDs (expected to be hexadecimal strings).
        """
        try:
            if not isinstance(a, str) or not isinstance(b, str):
                raise ValueError(f"Invalid types for XOR distance calculation: a={a} ({type(a)}), b={b} ({type(b)})")

            if len(a) != 40 or len(b) != 40:
                raise ValueError(f"Node IDs must be 40-character hexadecimal strings: a={a}, b={b}")

            # Convert the hex strings to integers and XOR them
            return int(a, 16) ^ int(b, 16)

        except Exception as e:
            logger.error(f"Error in XOR distance calculation: {str(e)}")
            raise ValueError(f"Failed to calculate XOR distance for node IDs a={a}, b={b}")
    def get_peer_id_from_public_key(self, public_key):
        try:
            if isinstance(public_key, rsa.RSAPublicKey):
                # Serialize the public key and hash it to get a unique identifier
                public_bytes = public_key.public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                hash_object = hashes.Hash(hashes.SHA1())
                hash_object.update(public_bytes)
                return hash_object.finalize().hex()[:40]
            elif isinstance(public_key, str):
                # If it's already a string, just return it (assuming it's a valid node ID)
                return public_key
            else:
                # If it's neither an RSAPublicKey nor a string, log an error and return a placeholder
                self.logger.error(f"Unexpected type for public key: {type(public_key)}")
                return "0" * 40  # Return a placeholder ID
        except Exception as e:
            self.logger.error(f"Error in get_peer_id_from_public_key: {str(e)}")
            return None



    def select_closest_peers(self, target_node_id: str, k: int):
        try:
            if not isinstance(target_node_id, str) or len(target_node_id) != 40:
                raise ValueError(f"Invalid target node ID: {target_node_id}")

            peers_with_distance = []
            for peer, public_key in self.peer_public_keys.items():
                peer_id = self.get_peer_id_from_public_key(public_key)
                if peer_id is None:
                    self.logger.warning(f"Failed to get peer ID for {peer}")
                    continue
                
                # Calculate the XOR distance
                distance = int(peer_id, 16) ^ int(target_node_id, 16)
                peers_with_distance.append((distance, peer))

            # Sort peers by XOR distance
            peers_with_distance.sort(key=lambda x: x[0])

            # Select the top k closest peers
            closest_peers = [{'ip': peer.split(':')[0], 'port': int(peer.split(':')[1])}
                             for _, peer in peers_with_distance[:k]]

            self.logger.debug(f"Selected {len(closest_peers)} closest peers for node ID {target_node_id}")
            return closest_peers

        except Exception as e:
            self.logger.error(f"Error selecting closest peers for node ID {target_node_id}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []


    async def find_node(self, node_id: str, k: int = 5) -> Optional[List[KademliaNode]]:
        """Find nodes closest to the given node ID."""
        try:
            self.logger.debug(f"Finding nodes closest to {node_id}")
            closest_peers = self.select_closest_peers(node_id, k)
            
            if not closest_peers:
                self.logger.warning("No peers available for node lookup")
                return []

            nodes = []
            for peer in closest_peers:
                try:
                    response = await self.send_find_node(node_id, peer)
                    if response and isinstance(response.payload, dict) and 'nodes' in response.payload:
                        new_nodes = response.payload['nodes']
                        if isinstance(new_nodes, list):
                            nodes.extend(new_nodes)
                            self.logger.debug(f"Received {len(new_nodes)} nodes from peer {peer}")
                except Exception as e:
                    self.logger.error(f"Error getting nodes from peer {peer}: {str(e)}")
                    continue

            if not nodes:
                self.logger.warning("No nodes found during find_node operation")
                return []

            self.logger.debug(f"Found {len(nodes)} total nodes")
            return nodes

        except Exception as e:
            self.logger.error(f"Error in find_node: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def get_closest_nodes(self, node_id: str) -> List[KademliaNode]:
        try:
            nodes = [node for bucket in self.buckets for node in bucket]
            return sorted(nodes, key=lambda n: self.calculate_distance(n.id, node_id))[:self.k]
        except Exception as e:
            logger.error(f"Failed to get closest nodes: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def store(self, key: str, value: str):
        try:
            nodes = await self.find_node(key)
            store_operations = [self.send_store(node, key, value) for node in nodes]
            await asyncio.gather(*store_operations)
        except Exception as e:
            logger.error(f"Failed to store value: {str(e)}")
            logger.error(traceback.format_exc())

    async def get(self, key: str) -> Optional[str]:
        try:
            if key in self.data_store:
                return self.data_store[key]
            nodes = await self.find_node(key)
            for node in nodes:
                value = await self.send_find_value(node, key)
                if value:
                    await self.store(key, value)
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get value: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    async def upload_token_logo(self, token_address: str, logo_data: str, logo_format: str):
        """
        Upload a logo for a token and distribute it to the network.
        """
        logo_info = {
            'data': logo_data,
            'format': logo_format
        }
        self.token_logos[token_address] = logo_info
        await self.distribute_logo(token_address, logo_info)

    async def distribute_logo(self, token_address: str, logo_info: dict):
        """
        Distribute the logo to all nodes in the network.
        """
        message = Message(
            MessageType.LOGO_UPLOAD.value,
            {
                'token_address': token_address,
                'logo_info': logo_info
            }
        )
        await self.broadcast(message)

    async def request_logo(self, token_address: str):
        """
        Request a logo from the network.
        """
        message = Message(
            MessageType.LOGO_REQUEST.value,
            {'token_address': token_address}
        )
        await self.broadcast(message)

    async def sync_logos(self):
        """
        Synchronize logos across all nodes.
        """
        message = Message(
            MessageType.LOGO_SYNC.value,
            {'logos': self.token_logos}
        )
        await self.broadcast(message)

    async def handle_logo_upload(self, data: dict, sender: str):
        """
        Handle incoming logo upload message.
        """
        token_address = data['token_address']
        logo_info = data['logo_info']
        self.token_logos[token_address] = logo_info
        logger.info(f"Received logo for token {token_address} from {sender}")

    async def handle_logo_request(self, data: dict, sender: str):
        """
        Handle incoming logo request message.
        """
        token_address = data['token_address']
        if token_address in self.token_logos:
            response = Message(
                MessageType.LOGO_RESPONSE.value,
                {
                    'token_address': token_address,
                    'logo_info': self.token_logos[token_address]
                }
            )
            await self.send_message(sender, response)

    async def handle_logo_response(self, data: dict, sender: str):
        """
        Handle incoming logo response message.
        """
        token_address = data['token_address']
        logo_info = data['logo_info']
        self.token_logos[token_address] = logo_info
        logger.info(f"Received requested logo for token {token_address} from {sender}")

    async def handle_logo_sync(self, data: dict, sender: str):
        """
        Handle incoming logo sync message.
        """
        received_logos = data['logos']
        self.token_logos.update(received_logos)
        logger.info(f"Synced logos with {sender}")
    async def propose_block(self, block: QuantumBlock, zkp_timeout: int = 6000):
        """
        Proposes a block to peers with Zero-Knowledge Proof (ZKP) generation and broadcasting.
        Logs the process and tracks resource usage (CPU, memory).
        """
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().percent

        try:
            self.logger.info(f"Starting block proposal for block {block.hash}")

            # Ensure that peers are connected before proceeding
            if not self.connected_peers:
                self.logger.warning(f"No active peers available for block proposal. Block proposal aborted.")
                return
            
            # Log current peer information
            self.logger.info("Current peer status at block proposal start:")
            self.logger.info(f"Connected peers: {self.connected_peers}")

            # Generate secret data from the block
            self.logger.debug(f"Generating secret data for block {block.hash}")
            secret = self.get_block_secret(block)
            self.logger.debug(f"Secret data generated for block {block.hash}: {secret}")

            # Generate public input from the block
            self.logger.debug(f"Generating public input for block {block.hash}")
            public_input = self.get_block_public_input(block)
            self.logger.debug(f"Public input for block {block.hash}: {public_input}")

            # Start ZKP generation as a background task
            zkp_start_time = time.time()
            self.logger.debug(f"Starting ZKP generation for block {block.hash} with a timeout of {zkp_timeout} seconds")

            # Create an async task for ZKP generation and allow other operations to proceed in parallel
            zkp_task = asyncio.create_task(self.generate_proof(secret, public_input))
            try:
                # Await the result of the ZKP generation with a timeout
                proof = await asyncio.wait_for(zkp_task, timeout=zkp_timeout)
                self.logger.debug(f"ZKP generated for block {block.hash} (Time taken: {time.time() - zkp_start_time:.2f}s)")
            except asyncio.TimeoutError:
                self.logger.error(f"ZKP generation for block {block.hash} timed out after {zkp_timeout} seconds.")
                return  # Handle the timeout gracefully
            except Exception as zkp_error:
                self.logger.error(f"Error during ZKP generation for block {block.hash}: {str(zkp_error)}")
                self.logger.error(traceback.format_exc())
                return

            # Create a block proposal message
            self.logger.debug(f"Creating proposal message for block {block.hash}")
            proposal_message = Message(
                type=MessageType.BLOCK_PROPOSAL.value,
                payload={
                    "proof": self.serialize_proof(proof),
                    "public_input": public_input,
                    "block_hash": block.hash,
                    "block": block.to_dict(),  # Include the full block data
                    "proposer": self.node_id
                }
            )
            self.logger.debug(f"Proposal message created for block {block.hash}")

            # Log connected peers before broadcasting
            self.logger.info(f"Current active peers: {self.connected_peers}")

            # Broadcast the block proposal to peers
            broadcast_start_time = time.time()
            self.logger.debug(f"Broadcasting block proposal for block {block.hash}")
            await self.broadcast(proposal_message)
            broadcast_time = time.time() - broadcast_start_time
            self.logger.info(f"Block {block.hash} proposal broadcasted. (Broadcast Time: {broadcast_time:.2f}s)")

            # Store the proposed block locally in pending proposals
            self.logger.debug(f"Storing block {block.hash} in pending block proposals.")
            self.pending_block_proposals[block.hash] = block
            self.logger.debug(f"Block {block.hash} stored in pending block proposals.")

            # Log total time and resource usage
            total_time = time.time() - start_time
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().percent

            self.logger.info(f"Block proposal process completed for {block.hash}:")
            self.logger.info(f"  Total time: {total_time:.2f}s")
            self.logger.info(f"  ZKP generation time: {time.time() - zkp_start_time:.2f}s")
            self.logger.info(f"  Broadcast time: {broadcast_time:.2f}s")
            self.logger.info(f"  CPU usage: {end_cpu - start_cpu:.2f}%")
            self.logger.info(f"  Memory usage change: {end_memory - start_memory:.2f}%")

        except Exception as e:
            self.logger.error(f"Error proposing block {block.hash}: {str(e)}")
            self.logger.error(traceback.format_exc())

        finally:
            # Log final resource usage
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().percent
            self.logger.info(f"Resource usage for block proposal {block.hash}:")
            self.logger.info(f"  Total time: {end_time - start_time:.2f}s")
            self.logger.info(f"  CPU usage: {end_cpu - start_cpu:.2f}%")
            self.logger.info(f"  Memory usage change: {end_memory - start_memory:.2f}%")



    async def handle_block_proposal(self, payload: dict, sender: str):
        try:
            proof = self.deserialize_proof(payload["proof"])
            public_input = payload["public_input"]
            block_hash = payload["block_hash"]
            block = QuantumBlock.from_dict(payload["block"])
            proposer = payload["proposer"]
            
            logger.info(f"Received block proposal from {proposer} for block {block_hash}")
            
            # Verify the ZKP
            is_valid = await self.verify_proof(public_input, proof)
            
            if is_valid:
                logger.info(f"Received valid block proposal {block_hash} from {proposer}")
                
                # If we're the chosen validator, request the full block
                if self.is_chosen_validator(block_hash):
                    await self.request_full_block(block_hash, proposer)
            else:
                logger.warning(f"Received invalid block proposal {block_hash} from {proposer}")
        except Exception as e:
            logger.error(f"Error handling block proposal: {str(e)}")
            logger.error(traceback.format_exc())


    def is_chosen_validator(self, block_hash: str) -> bool:
        # Implement your validator selection logic here
        # This could be based on stake, reputation, or other factors
        return hash(f"{self.node_id}{block_hash}") % 10 == 0  # Example: 10% chance of being chosen

    async def request_full_block(self, block_hash: str, proposer: str):
        try:
            request_message = Message(
                type=MessageType.FULL_BLOCK_REQUEST.value,
                payload={"block_hash": block_hash}
            )
            response = await self.send_and_wait_for_response(proposer, request_message)
            
            if response and response.type == MessageType.FULL_BLOCK_RESPONSE.value:
                full_block = QuantumBlock.from_dict(response.payload["block"])
                
                # Validate the full block
                if self.validate_full_block(full_block):
                    await self.add_block_to_blockchain(full_block)
                    await self.broadcast_block_acceptance(block_hash)
                else:
                    logger.warning(f"Received invalid full block {block_hash} from {proposer}")
            else:
                logger.warning(f"Failed to receive full block {block_hash} from {proposer}")
        except Exception as e:
            logger.error(f"Error requesting full block: {str(e)}")
            logger.error(traceback.format_exc())



    def get_block_secret(self, block: QuantumBlock) -> int:
        # This method should return a secret integer derived from the block
        # For example, you could hash the block's transactions and convert to int
        transactions_str = ''.join(tx.to_json() for tx in block.transactions)
        return int(hashlib.sha256(transactions_str.encode()).hexdigest(), 16)

    def get_block_public_input(self, block: QuantumBlock) -> int:
        # This method should return a public integer derived from the block
        # For example, you could use the block's timestamp and nonce
        return int(f"{int(block.timestamp)}{block.nonce}")


    def is_chosen_validator(self, block_hash: str) -> bool:
        # Implement your validator selection logic here
        # This could be based on stake, reputation, or other factors
        return hash(f"{self.node_id}{block_hash}") % 10 == 0  # Example: 10% chance of being chosen

    def validate_full_block(self, block: QuantumBlock) -> bool:
        # Implement full block validation logic here
        # This should check the block's structure, transactions, etc.
        return self.blockchain.validate_block(block)

    async def add_block_to_blockchain(self, block: QuantumBlock):
        await self.blockchain.add_block(block)
        self.logger.info(f"Added block {block.hash} to blockchain")

    async def broadcast_block_acceptance(self, block_hash: str):
        acceptance_message = Message(
            type=MessageType.BLOCK_ACCEPTANCE.value,
            payload={"block_hash": block_hash, "validator": self.node_id}
        )
        await self.broadcast(acceptance_message)

    async def handle_full_block_request(self, message: Message, sender: str):
        try:
            block_hash = message.payload["block_hash"]
            if block_hash in self.pending_block_proposals:
                full_block = self.pending_block_proposals[block_hash]
                response = Message(
                    type=MessageType.FULL_BLOCK_RESPONSE.value,
                    payload={"block": full_block.to_dict()}
                )
                await self.send_message(sender, response)
            else:
                self.logger.warning(f"Received request for unknown block {block_hash} from {sender}")
        except Exception as e:
            self.logger.error(f"Error handling full block request: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def handle_block_acceptance(self, message: Message, sender: str):
        try:
            block_hash = message.payload["block_hash"]
            validator = message.payload["validator"]
            # Implement your block acceptance logic here
            # This could involve collecting a certain number of acceptances before finalizing the block
            self.logger.info(f"Received block acceptance for {block_hash} from validator {validator}")
        except Exception as e:
            self.logger.error(f"Error handling block acceptance: {str(e)}")
            self.logger.error(traceback.format_exc())
    async def handle_message(self, message: Message, sender: str):
        """Enhanced message handler with consistent type comparison, quantum state updates, verification, and detailed logging."""
        try:
            # Extract message type and standardize it for comparison
            msg_type = str(message.type).lower() if message.type else ""
            self.logger.debug(f"\n[MESSAGE] Received message from {sender}")
            self.logger.debug(f"[MESSAGE] Type: {msg_type}")
            self.logger.debug(f"[MESSAGE] Payload: {message.payload}")
            response = None

            # Handle challenge messages first with direct string comparison
            if msg_type == "challenge":
                self.logger.info(f"[CHALLENGE] Processing challenge from {sender}")
                challenge_payload = message.payload.get('challenge')
                if not challenge_payload:
                    self.logger.error(f"[CHALLENGE] Invalid challenge payload from {sender}")
                    return

                # Parse challenge data
                try:
                    if ':' in challenge_payload:
                        challenge_id, challenge = challenge_payload.split(':', 1)
                    else:
                        challenge_id = message.challenge_id
                        challenge = challenge_payload

                    self.logger.debug(f"[CHALLENGE] Challenge ID: {challenge_id}")
                    self.logger.debug(f"[CHALLENGE] Challenge Data: {challenge[:32]}...")
                    await self.handle_challenge(sender, challenge, challenge_id)
                    return

                except Exception as e:
                    self.logger.error(f"[CHALLENGE] Error parsing challenge: {str(e)}")
                    return

            elif msg_type == "challenge_response":
                self.logger.info(f"[CHALLENGE_RESPONSE] Received response from {sender}")
                await self.handle_challenge_response(sender, message.payload)
                return

            # Quantum state update messages with verification
            elif msg_type == str(MessageType.QUANTUM_STATE_UPDATE.value).lower():
                success = await self.quantum_notifier.handle_update_message(message.payload, sender)
                if success:
                    await self.quantum_notifier._send_verification(QuantumStateUpdate.from_dict(message.payload), sender)
                return

            elif msg_type == str(MessageType.QUANTUM_HEARTBEAT.value).lower():
                self.logger.info(f"[QUANTUM] Received heartbeat from {sender}")
                await self.handle_quantum_heartbeat(message, sender)
                return

            elif msg_type == str(MessageType.QUANTUM_STATE_VERIFICATION.value).lower():
                await self.quantum_notifier.handle_verification_message(message.payload, sender)
                return

            # Quantum sync messages
            elif msg_type == str(MessageType.QUANTUM_ENTANGLEMENT_REQUEST.value).lower():
                await self.handle_entanglement_request(message, sender)
                return

            elif msg_type == str(MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value).lower():
                await self.handle_entanglement_response(message, sender)
                return

            elif msg_type == str(MessageType.QUANTUM_RESYNC_REQUEST.value).lower():
                await self.handle_quantum_resync_request(message, sender)
                return

            elif msg_type == str(MessageType.QUANTUM_RESYNC_RESPONSE.value).lower():
                await self.handle_quantum_resync_response(message, sender)
                return

            # Fast path for transactions and blocks - handle these with quantum state updates and peer notification
            elif msg_type == str(MessageType.TRANSACTION.value).lower():
                self.logger.info(f"[TRANSACTION] Received TRANSACTION message from {sender}. Details: {message.payload}")
                result = await self.handle_transaction(message.payload, sender)

                if result:
                    await self.quantum_sync.update_component_state('transactions', message.payload)
                    await self._notify_peer_update(sender, 'transactions')
                return

            elif msg_type == str(MessageType.BLOCK.value).lower():
                self.logger.info(f"[BLOCK] Received BLOCK message from {sender}. Details: {message.payload}")
                result = await self.handle_block(message.payload, sender)

                if result:
                    await self.quantum_sync.update_component_state('blocks', message.payload)
                    await self._notify_peer_update(sender, 'blocks')

                    # Update mempool quantum state and notify peers
                    mempool_data = [tx.to_dict() for tx in self.blockchain.mempool]
                    await self.quantum_sync.update_component_state('mempool', mempool_data)
                    await self._notify_peer_update(sender, 'mempool')
                return

            # Handle sync-related messages with priority
            elif msg_type in [
                str(MessageType.SYNC_STATUS.value).lower(),
                str(MessageType.SYNC_REQUEST.value).lower(),
                str(MessageType.SYNC_RESPONSE.value).lower(),
                str(MessageType.SYNC_DATA.value).lower(),
            ]:
                await self.handle_sync_message(message, sender)
                return

            # Additional message types with logging
            elif msg_type == str(MessageType.GET_WALLETS.value).lower():
                self.logger.info(f"[WALLETS] Received GET_WALLETS request from {sender}")
                await self.handle_get_wallets(sender)

            elif msg_type == str(MessageType.WALLETS.value).lower():
                self.logger.info(f"[WALLETS] Received WALLETS message from {sender}. Details: {message.payload}")
                await self.handle_wallets(message.payload, sender)

            elif msg_type == str(MessageType.PUBLIC_KEY_EXCHANGE.value).lower():
                self.logger.info(f"[KEY_EXCHANGE] Received PUBLIC_KEY_EXCHANGE message from {sender}. Data: {message.payload}")
                await self.handle_public_key_exchange(message, sender)

            elif msg_type == str(MessageType.FIND_NODE.value).lower():
                self.logger.info(f"[FIND_NODE] Received FIND_NODE request from {sender}. Details: {message.payload}")
                response = await self.handle_find_node(message.payload, sender)

            elif msg_type == str(MessageType.FIND_VALUE.value).lower():
                self.logger.info(f"[FIND_VALUE] Received FIND_VALUE request from {sender}. Details: {message.payload}")
                response = await self.handle_find_value(message.payload, sender)

            elif msg_type == str(MessageType.STORE.value).lower():
                self.logger.info(f"[STORE] Received STORE message from {sender}. Details: {message.payload}")
                await self.handle_store(message.payload, sender)

            elif msg_type == str(MessageType.BLOCK_PROPOSAL.value).lower():
                self.logger.info(f"[BLOCK_PROPOSAL] Received BLOCK_PROPOSAL message from {sender}. Details: {message.payload}")
                await self.handle_block_proposal(message, sender)

            elif msg_type == str(MessageType.FULL_BLOCK_REQUEST.value).lower():
                self.logger.info(f"[FULL_BLOCK_REQUEST] Received FULL_BLOCK_REQUEST from {sender}. Details: {message.payload}")
                await self.handle_full_block_request(message, sender)

            elif msg_type == str(MessageType.FULL_BLOCK_RESPONSE.value).lower():
                self.logger.info(f"[FULL_BLOCK_RESPONSE] Received FULL_BLOCK_RESPONSE from {sender}. Block details: {message.payload}")
                await self.handle_full_block_response(message, sender)

            elif msg_type == str(MessageType.BLOCK_ACCEPTANCE.value).lower():
                self.logger.info(f"[BLOCK_ACCEPTANCE] Received BLOCK_ACCEPTANCE message from {sender}. Details: {message.payload}")
                await self.handle_block_acceptance(message, sender)

            elif msg_type == str(MessageType.STATE_REQUEST.value).lower():
                self.logger.info(f"[STATE_REQUEST] Received STATE_REQUEST from {sender}")
                state = self.blockchain.get_node_state()
                response = Message(type=MessageType.STATE_RESPONSE.value, payload=asdict(state))

            elif msg_type == str(MessageType.GET_ALL_DATA.value).lower():
                self.logger.info(f"[GET_ALL_DATA] Received GET_ALL_DATA request from {sender}")
                all_data = await self.blockchain.get_all_data()
                response = Message(MessageType.ALL_DATA.value, all_data)

            elif msg_type == str(MessageType.LOGO_UPLOAD.value).lower():
                self.logger.info(f"[LOGO_UPLOAD] Received LOGO_UPLOAD message from {sender}. Logo details: {message.payload}")
                await self.handle_logo_upload(message.payload, sender)

            elif msg_type == str(MessageType.GET_TRANSACTIONS.value).lower():
                self.logger.info(f"[GET_TRANSACTIONS] Received GET_TRANSACTIONS message from {sender}")
                await self.handle_get_transactions(sender)

            # Handle unknown message types
            else:
                self.logger.warning(f"Received unknown message type from {sender}: {message.type}")

            # If a response is generated, send it back to the sender
            if response:
                response.id = message.id  # Set the response ID to match the request ID
                await self.send_message(sender, response)

        except Exception as e:
            self.logger.error(f"Failed to handle message from {sender}: {str(e)}")
            self.logger.error(f"Message content: {message.to_json()}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")


    async def handle_entanglement_request(self, message: Message, sender: str):
        """Handle incoming quantum entanglement request"""
        try:
            self.logger.info(f"[QUANTUM] Received entanglement request from {sender}")

            # Get our current state
            our_state = {
                'wallets': [w.to_dict() for w in self.blockchain.get_wallets()],
                'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions()],
                'blocks': [block.to_dict() for block in self.blockchain.chain],
                'mempool': [tx.to_dict() for tx in self.blockchain.mempool]
            }

            # Initialize quantum sync if needed
            if not self.quantum_initialized:
                await self.initialize_quantum_sync(our_state)

            # Create quantum entanglement
            peer_state = message.payload.get('state_data', {})
            peer_register = await self.quantum_sync.entangle_with_peer(sender, peer_state)
            
            if not peer_register:
                self.logger.error(f"[QUANTUM] Failed to create quantum register with {sender}")
                return

            # Generate and store Bell pair
            bell_pair = self.quantum_sync._generate_bell_pair()
            self.quantum_sync.bell_pairs[sender] = bell_pair

            # Send response
            response = Message(
                type=MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value,
                payload={
                    'node_id': self.node_id,
                    'state_data': our_state,
                    'timestamp': time.time()
                }
            )

            await self.send_message(sender, response)
            
            self.logger.info(f"[QUANTUM] ✓ Quantum entanglement established with {sender}")

            # Start quantum monitoring
            asyncio.create_task(self.monitor_peer_quantum_state(sender))

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error handling entanglement request: {str(e)}")


    def parse_challenge_message(self, message: Message) -> Tuple[str, str]:
        """Parse challenge message and return (challenge_id, challenge_data)"""
        if not isinstance(message.payload, dict):
            raise ValueError("Invalid challenge payload format")
            
        challenge_str = message.payload.get('challenge')
        if not challenge_str:
            raise ValueError("Missing challenge in payload")
            
        if ':' in challenge_str:
            challenge_id, challenge = challenge_str.split(':', 1)
        else:
            challenge_id = message.challenge_id
            challenge = challenge_str
            
        if not challenge_id or not challenge:
            raise ValueError("Invalid challenge format")
            
        return challenge_id, challenge

    async def periodic_consensus_check(self):
        """Periodically check quantum consensus"""
        while True:
            try:
                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                    consensus = await self.quantum_consensus.check_quantum_consensus(
                        component
                    )
                    if not consensus:
                        self.logger.warning(
                            f"Quantum consensus lost for {component}"
                        )
                        await self.handle_consensus_loss(component)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in consensus check: {str(e)}")
                await asyncio.sleep(5)

    async def handle_consensus_loss(self, component: str):
        """Handle loss of quantum consensus"""
        try:
            self.logger.info(f"Handling consensus loss for {component}")
            
            # Get highest fidelity peer
            best_peer = await self.find_highest_fidelity_peer(component)
            
            if best_peer:
                # Request state update from best peer
                await self.request_verified_state_update(component, best_peer)
                
                # Verify consensus is restored
                consensus = await self.quantum_consensus.check_quantum_consensus(
                    component
                )
                if consensus:
                    self.logger.info(f"Restored consensus for {component}")
                else:
                    self.logger.warning(
                        f"Failed to restore consensus for {component}"
                    )
            
        except Exception as e:
            self.logger.error(f"Error handling consensus loss: {str(e)}")

    async def monitor_quantum_health(self, interval: float = 5.0):
        """Monitor quantum state health and trigger recovery if needed"""
        while True:
            try:
                await asyncio.sleep(interval)
                
                for peer in self.quantum_sync.entangled_peers:
                    for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                        fidelity = await self.quantum_sync.measure_sync_state(component)
                        
                        if fidelity < self.quantum_sync.decoherence_threshold:
                            self.logger.warning(
                                f"Low fidelity detected: {component} with {peer} "
                                f"(fidelity: {fidelity:.3f})"
                            )
                            await self.handle_quantum_decoherence_recovery(
                                component, peer)
                            
            except Exception as e:
                self.logger.error(f"Error in quantum health monitoring: {str(e)}")
                await asyncio.sleep(interval)
    async def cleanup_quantum_resources(self, peer: str):
        """Clean up quantum resources for a peer"""
        try:
            # Remove from entangled peers
            self.quantum_sync.entangled_peers.pop(peer, None)
            # Remove Bell pairs
            self.quantum_sync.bell_pairs.pop(peer, None)
            # Clear verification cache
            nonces_to_remove = [
                nonce for nonce in self.quantum_notifier.verification_cache
                if nonce.startswith(f"{peer}:")
            ]
            for nonce in nonces_to_remove:
                self.quantum_notifier.verification_cache.pop(nonce, None)
            
            self.logger.info(f"Cleaned up quantum resources for peer {peer}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up quantum resources: {str(e)}")

    async def handle_quantum_state_verification(self, message: Dict[str, Any], sender: str):
        """Handle incoming quantum state verification messages"""
        await self.quantum_notifier.handle_verification_message(message, sender)

    async def get_bell_pair_id_for_peer(self, peer: str) -> Optional[str]:
        """Get Bell pair ID for a peer"""
        bell_pair = self.quantum_sync.bell_pairs.get(peer)
        if bell_pair:
            return self.quantum_notifier._get_bell_pair_id(bell_pair)
        return None
    async def verify_and_update_quantum_state(self, component: str, new_data: dict, sender: str) -> bool:
        """Verify and update quantum state with security checks"""
        try:
            # Get Bell pair ID
            bell_pair_id = await self.get_bell_pair_id_for_peer(sender)
            if not bell_pair_id:
                self.logger.warning(f"No Bell pair found for peer {sender}")
                return False

            # Create state update
            update = QuantumStateUpdate(
                component=component,
                state_value=self.quantum_sync.register.compute_state_value(new_data),
                timestamp=time.time(),
                bell_pair_id=bell_pair_id,
                node_id=self.node_id,
                nonce=self.quantum_notifier._generate_nonce()
            )
            
            # Verify and apply update
            if await self.quantum_notifier._verify_quantum_state(update, sender):
                await self.quantum_notifier._apply_quantum_update(update, sender)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error in verify_and_update_quantum_state: {str(e)}")
            return False

    async def handle_quantum_decoherence_recovery(self, component: str, peer: str):
        """Handle recovery from quantum decoherence"""
        try:
            self.logger.info(f"[QUANTUM] Starting decoherence recovery for {component} with {peer}")
            
            # Request verified state
            response = await self.request_verified_state_update(component, peer)
            if not response:
                return False
                
            # Verify and apply update
            success = await self.verify_and_update_quantum_state(
                component, 
                response.payload['state_data'],
                peer
            )
            
            if success:
                self.logger.info(f"[QUANTUM] Successfully recovered {component} state with {peer}")
            else:
                self.logger.warning(f"[QUANTUM] Failed to recover {component} state with {peer}")
                
            return success

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error in decoherence recovery: {str(e)}")
            return False




    async def monitor_quantum_state(self, interval: float = 5.0):
        """Monitor quantum state health and trigger recovery if needed"""
        while True:
            try:
                await asyncio.sleep(interval)
                
                if not self.quantum_initialized:
                    continue
                    
                for peer in self.quantum_sync.entangled_peers:
                    for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                        fidelity = await self.quantum_sync.measure_sync_state(component)
                        
                        if fidelity < self.quantum_sync.decoherence_threshold:
                            self.logger.warning(
                                f"[QUANTUM] Low fidelity detected for {component} "
                                f"with peer {peer} (fidelity: {fidelity:.3f})"
                            )
                            await self.handle_quantum_decoherence_recovery(component, peer)
                            
            except Exception as e:
                self.logger.error(f"[QUANTUM] Error in quantum state monitoring: {str(e)}")
                await asyncio.sleep(interval)


    async def request_verified_state_update(self, component: str):
        """Request verified quantum state update from peers"""
        try:
            # Get entangled peers for this component
            qubit = getattr(self.quantum_sync.register, component)
            peers = list(qubit.peers)
            
            if not peers:
                self.logger.warning(f"No entangled peers for {component}")
                return

            # Request update from random peer
            peer_id = random.choice(peers)
            
            # Create resync request
            resync_message = Message(
                type=MessageType.QUANTUM_RESYNC_REQUEST.value,
                payload={
                    'component': component,
                    'node_id': self.node_id,
                    'timestamp': time.time()
                }
            )

            # Wait for verified response
            response = await self.send_and_wait_for_response(peer_id, resync_message)
            
            if response and response.type == MessageType.QUANTUM_RESYNC_RESPONSE.value:
                update = QuantumStateUpdate.from_dict(response.payload)
                
                # Verify and apply update
                if await self.quantum_notifier._verify_quantum_state(update, peer_id):
                    await self.quantum_notifier._apply_quantum_update(update, peer_id)
                    self.logger.info(f"Successfully resynced {component} with {peer_id}")
                else:
                    self.logger.warning(f"Failed to verify quantum state from {peer_id}")
            
        except Exception as e:
            self.logger.error(f"Error requesting state update: {str(e)}")
            raise
    async def update_component_state(self, component: str, new_data: dict):
        """Update quantum state with verified peer notifications"""
        try:
            await self.quantum_sync.update_component_state(component, new_data)
            
            # Notify all entangled peers with verification
            update_tasks = []
            for peer_id in self.quantum_sync.entangled_peers:
                update_tasks.append(self._notify_peer_update(peer_id, component))
            
            if update_tasks:
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                successful = sum(1 for r in results if r and not isinstance(r, Exception))
                self.logger.info(
                    f"Notified {successful}/{len(update_tasks)} peers of {component} update"
                )
                
        except Exception as e:
            self.logger.error(f"Error updating component state: {str(e)}")
            raise



    async def handle_sync_message(self, message: Message, sender: str):
        try:
            if message.type == MessageType.SYNC_STATUS.value:
                await self.handle_sync_status(sender, message.payload)
            elif message.type == MessageType.SYNC_REQUEST.value:
                await self.handle_sync_request(sender, message.payload)
            elif message.type == MessageType.SYNC_RESPONSE.value:
                await self.handle_sync_response(sender, message.payload)
            elif message.type == MessageType.SYNC_DATA.value:
                await self.handle_sync_data(sender, message.payload)
        except Exception as e:
            self.logger.error(f"Error handling sync message: {str(e)}")

    async def sync_new_peer(self, peer: str):
        """Sync state with a new peer"""
        # Get our current state hashes
        our_state = {
            SyncComponent.WALLETS.value: await self.calculate_state_hash(SyncComponent.WALLETS),
            SyncComponent.TRANSACTIONS.value: await self.calculate_state_hash(SyncComponent.TRANSACTIONS),
            SyncComponent.BLOCKS.value: await self.calculate_state_hash(SyncComponent.BLOCKS),
            SyncComponent.MEMPOOL.value: await self.calculate_state_hash(SyncComponent.MEMPOOL)
        }

        # Send sync request
        await self.send_message(peer, Message(
            type=MessageType.SYNC_REQUEST.value,
            payload={"state_hashes": our_state}
        ))

    async def handle_get_transactions(self, sender):
        try:
            await self.ensure_blockchain()
            transactions = self.blockchain.get_recent_transactions(limit=100)
            tx_data = [tx.to_dict() for tx in transactions]
            response = Message(type=MessageType.TRANSACTIONS.value, payload={"transactions": tx_data})
            await self.send_message(sender, response)
        except Exception as e:
            self.logger.error(f"Error handling get_transactions request: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def handle_get_wallets(self, sender):
        try:
            if self.blockchain is None:
                logger.warning("Blockchain is not initialized. Cannot get wallets.")
                wallet_data = []
            else:
                if hasattr(self.blockchain, 'get_wallets'):
                    wallets = self.blockchain.get_wallets()
                    wallet_data = [wallet.to_dict() for wallet in wallets]
                else:
                    logger.warning("Blockchain does not have a get_wallets method.")
                    wallet_data = []

            response = Message(type=MessageType.WALLETS.value, payload={"wallets": wallet_data})
            await self.send_message(sender, response)
        except Exception as e:
            logger.error(f"Error handling get_wallets request: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_transactions(self, payload, sender):
        new_transactions = payload.get('transactions', [])
        for tx_data in new_transactions:
            tx = Transaction.from_dict(tx_data)
            if await self.blockchain.validate_transaction(tx):
                await self.blockchain.add_transaction(tx)
        logger.info(f"Processed {len(new_transactions)} transactions from {sender}")

    async def handle_wallets(self, payload, sender):
        new_wallets = payload.get('wallets', [])
        for wallet_data in new_wallets:
            wallet = Wallet.from_dict(wallet_data)
            self.blockchain.add_wallet(wallet)
        logger.info(f"Processed {len(new_wallets)} wallets from {sender}")
    async def handle_handshake(self, peer: str, payload: dict):
        """Handle incoming handshake message."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.info(f"[HANDSHAKE] Processing handshake from {peer_normalized}")
            
            # Update peer info
            self.peer_info[peer_normalized].update({
                'node_id': payload.get('node_id'),
                'version': payload.get('version', '1.0'),
                'blockchain_height': payload.get('blockchain_height', 0),
                'capabilities': payload.get('capabilities', []),
                'timestamp': payload.get('timestamp', time.time()),
                'last_activity': time.time()
            })
            
            # Update peer state
            async with self.peer_lock:
                self.peer_states[peer_normalized] = "handshake_complete"
                
            self.logger.info(f"[HANDSHAKE] ✓ Handshake completed for {peer_normalized}")
            return True
            
        except Exception as e:
            self.logger.error(f"[HANDSHAKE] Error handling handshake: {str(e)}")
            return False


    async def sync_blockchain(self, peer):
        try:
            logger.info(f"Starting blockchain sync with peer {peer}")

            # Step 1: Get the peer's latest block
            latest_block_message = await self.send_and_wait_for_response(
                peer, 
                Message(type=MessageType.GET_LATEST_BLOCK.value, payload={})
            )
            peer_latest_block = QuantumBlock.from_dict(latest_block_message.payload['block'])

            # Step 2: Compare the peer's latest block with our latest block
            our_latest_block = self.blockchain.get_latest_block()
            if peer_latest_block.index <= our_latest_block.index:
                logger.info(f"Our blockchain is up to date. No sync needed with {peer}")
                return

            # Step 3: Find the common ancestor block
            common_ancestor = await self.find_common_ancestor(peer, our_latest_block)

            # Step 4: Request blocks from the common ancestor to the peer's latest block
            blocks_to_add = await self.get_blocks_from_peer(peer, common_ancestor.index + 1, peer_latest_block.index)

            # Step 5: Validate and add the new blocks
            for block in blocks_to_add:
                if self.blockchain.consensus.validate_block(block):
                    success = await self.blockchain.add_block(block)
                    if not success:
                        logger.error(f"Failed to add block {block.index} from peer {peer}")
                        break
                else:
                    logger.error(f"Invalid block {block.index} received from peer {peer}")
                    break

            # Step 6: Sync mempool (unconfirmed transactions)
            await self.sync_mempool(peer)

            logger.info(f"Blockchain sync with peer {peer} completed. Current height: {self.blockchain.get_latest_block().index}")

        except Exception as e:
            logger.error(f"Error during blockchain sync with peer {peer}: {str(e)}")
            logger.error(traceback.format_exc())

    async def find_common_ancestor(self, peer, our_latest_block):
        current_height = our_latest_block.index
        step = 10  # We'll check every 10th block to speed up the process

        while current_height > 0:
            block_hash_message = await self.send_and_wait_for_response(
                peer,
                Message(type=MessageType.GET_BLOCK_HASH.value, payload={'height': current_height})
            )
            peer_block_hash = block_hash_message.payload['hash']

            our_block_hash = self.blockchain.get_block_hash_at_height(current_height)

            if peer_block_hash == our_block_hash:
                return self.blockchain.get_block_at_height(current_height)

            current_height -= step

        # If we couldn't find a common ancestor, return the genesis block
        return self.blockchain.get_block_at_height(0)

    async def get_blocks_from_peer(self, peer, start_height, end_height):
        blocks = []
        for height in range(start_height, end_height + 1):
            block_message = await self.send_and_wait_for_response(
                peer,
                Message(type=MessageType.GET_BLOCK.value, payload={'height': height})
            )
            block = QuantumBlock.from_dict(block_message.payload['block'])
            blocks.append(block)
        return blocks

    async def sync_mempool(self, peer):
        mempool_message = await self.send_and_wait_for_response(
            peer,
            Message(type=MessageType.GET_MEMPOOL.value, payload={})
        )
        peer_transactions = [Transaction.from_dict(tx) for tx in mempool_message.payload['transactions']]

        for tx in peer_transactions:
            if tx not in self.blockchain.mempool and self.blockchain.validate_transaction(tx):
                self.blockchain.add_to_mempool(tx)

        logger.info(f"Synchronized mempool with peer {peer}. Added {len(peer_transactions)} new transactions.")
        
        
    async def handle_get_latest_block(self, sender):
        latest_block = self.blockchain.get_latest_block()
        response = Message(type=MessageType.LATEST_BLOCK.value, payload={'block': latest_block.to_dict()})
        await self.send_message(sender, response)

    async def handle_get_block_hash(self, payload, sender):
        height = payload['height']
        block_hash = self.blockchain.get_block_hash_at_height(height)
        response = Message(type=MessageType.BLOCK_HASH.value, payload={'hash': block_hash})
        await self.send_message(sender, response)

    async def handle_get_block(self, payload, sender):
        height = payload['height']
        block = self.blockchain.get_block_at_height(height)
        response = Message(type=MessageType.BLOCK.value, payload={'block': block.to_dict()})
        await self.send_message(sender, response)

    async def handle_get_mempool(self, sender):
        mempool_transactions = [tx.to_dict() for tx in self.blockchain.mempool]
        response = Message(type=MessageType.MEMPOOL.value, payload={'transactions': mempool_transactions})
        await self.send_message(sender, response)


    def validate_and_convert_public_key(self, peer: str) -> bool:
        normalized_peer = self.normalize_peer_address(peer)
        stored_key = self.peer_public_keys.get(normalized_peer)
        
        if isinstance(stored_key, str):
            logger.warning(f"Public key for {normalized_peer} is stored as a string. Attempting to convert.")
            try:
                converted_key = serialization.load_pem_public_key(
                    stored_key.encode(),
                    backend=default_backend()
                )
                if isinstance(converted_key, rsa.RSAPublicKey):
                    self.peer_public_keys[normalized_peer] = converted_key
                    logger.info(f"Successfully converted public key for {normalized_peer} to RSAPublicKey object.")
                    return True
                else:
                    logger.error(f"Converted key for {normalized_peer} is not an RSA public key.")
                    return False
            except Exception as e:
                logger.error(f"Failed to convert public key for {normalized_peer}: {str(e)}")
                return False
        elif isinstance(stored_key, rsa.RSAPublicKey):
            return True
        else:
            logger.error(f"Invalid public key type for {normalized_peer}: {type(stored_key)}")
            return False

    async def handle_submit_computation(self, data: dict, sender: str):
        task_id = data['task_id']
        await self.computation_system.process_task(task_id)
    async def handle_computation_result(self, data: dict, sender: str):
        task_id = data['task_id']
        status = data['status']
        # You might want to do something with this information,
        # such as updating a local cache or notifying waiting clients

    async def submit_computation(self, function_name: str, *args, **kwargs) -> str:
        return await self.computation_system.submit_task(function_name, *args, **kwargs)

    async def get_computation_result(self, task_id: str) -> ComputationTask:
        return await self.computation_system.get_task_result(task_id)


    async def handle_find_node(self, data: dict, sender: str):
        try:
            node_id = data['node_id']
            closest_nodes = self.get_closest_nodes(node_id)
            await self.send_message(sender, Message(MessageType.FIND_NODE.value, {"nodes": [asdict(node) for node in closest_nodes]}))
        except Exception as e:
            logger.error(f"Failed to handle find node: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_find_value(self, data: dict, sender: str):
        try:
            key = data['key']
            if key in self.data_store:
                await self.send_message(sender, Message(MessageType.FIND_VALUE.value, {"value": self.data_store[key]}))
            else:
                await self.handle_find_node({"node_id": key}, sender)
        except Exception as e:
            logger.error(f"Failed to handle find value: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_store(self, data: dict, sender: str):
        try:
            key, value = data['key'], data['value']
            self.data_store[key] = value
            if len(self.data_store) > self.max_data_store_size:
                self.data_store.popitem(last=False)
        except Exception as e:
            logger.error(f"Failed to handle store: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_magnet_link(self, data: dict, sender: str):
        try:
            magnet_link = MagnetLink.from_uri(data['magnet_link'])
            sender_node = next((node for node in self.get_closest_nodes(magnet_link.info_hash) if node.address == sender), None)
            if sender_node:
                sender_node.magnet_link = magnet_link
            await self.add_node_to_bucket(KademliaNode(magnet_link.info_hash, sender.split(':')[0], int(sender.split(':')[1]), magnet_link))
        except Exception as e:
            logger.error(f"Failed to handle magnet link: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_peer_exchange(self, data: dict, sender: str):
        try:
            new_peers = set(data['peers']) - set(self.peers.keys())
            for peer in new_peers:
                await self.connect_to_peer(KademliaNode(self.generate_node_id(), *peer.split(':')))
        except Exception as e:
            logger.error(f"Failed to handle peer exchange: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_transaction(self, transaction_data: dict, sender: str):
        """Handle incoming transaction with real-time propagation and detailed logging."""
        try:
            start_time = time.time()
            tx_hash = None
            
            # Log incoming transaction
            self.logger.info(f"[TRANSACTION] Received transaction from {sender}")
            self.logger.debug(f"[TRANSACTION] Transaction data: {transaction_data}")

            # Perform fast format validation
            if not self.validate_transaction_format(transaction_data):
                self.logger.warning(f"[TRANSACTION] Invalid transaction format from {sender}")
                return

            try:
                # Determine transaction type and process accordingly
                if transaction_data.get('type') == 'multisig_transaction':
                    self.logger.info(f"[TRANSACTION] Processing multisig transaction")
                    tx_hash = await self.process_multisig_transaction(transaction_data)
                else:
                    self.logger.info(f"[TRANSACTION] Processing regular transaction")
                    tx_hash = await self.process_regular_transaction(transaction_data)

                # Exit if processing fails
                if not tx_hash:
                    return

                # Update sync state after processing
                self.sync_states[SyncComponent.MEMPOOL].current_hash = await self.calculate_state_hash(SyncComponent.MEMPOOL)

                # Prepare for propagation to peers
                self.logger.info(f"[TRANSACTION] Propagating transaction {tx_hash} to peers")
                active_peers = self.connected_peers - {sender}
                propagation_tasks = [self.send_message(peer, Message(
                    type=MessageType.TRANSACTION.value,
                    payload=transaction_data
                )) for peer in active_peers]

                # Execute propagation with timeout and log results
                if propagation_tasks:
                    self.logger.info(f"[TRANSACTION] Starting propagation to {len(propagation_tasks)} peers")
                    done_tasks, pending_tasks = await asyncio.wait(propagation_tasks, timeout=1.0)
                    
                    successful_propagations = len([t for t in done_tasks if not t.exception()])
                    self.logger.info(f"[TRANSACTION] Propagation completed: {successful_propagations}/{len(propagation_tasks)} successful")
                    
                    if pending_tasks:
                        self.logger.warning(f"[TRANSACTION] {len(pending_tasks)} propagations timed out")
                else:
                    self.logger.warning("[TRANSACTION] No peers available for propagation")

                # Log transaction processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                self.logger.info(f"[TRANSACTION] {tx_hash} processed and propagated in {processing_time:.2f}ms")

                # Record and log average propagation metrics if available
                if hasattr(self, 'tx_metrics'):
                    self.tx_metrics.record_propagation(processing_time)
                    avg_time = self.tx_metrics.get_average_propagation()
                    self.logger.info(f"[TRANSACTION] Average propagation time: {avg_time:.2f}ms")

                # Notify WebSocket subscribers about the new transaction
                await self.broadcast_event('new_transaction', {
                    'tx_hash': tx_hash,
                    'type': transaction_data.get('type', 'regular'),
                    'timestamp': time.time(),
                    'propagation_time': processing_time
                })

            except Exception as e:
                self.logger.error(f"[TRANSACTION] Processing error: {str(e)}")
                self.logger.error(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"[TRANSACTION] Fatal error: {str(e)}")
            self.logger.error(traceback.format_exc())

        # Final transaction state logging
        finally:
            if tx_hash:
                mempool_size = len(self.blockchain.mempool) if self.blockchain else 0
                self.logger.info(f"[TRANSACTION] Current mempool size: {mempool_size}")
                self.logger.info(f"[TRANSACTION] Transaction {tx_hash} handling completed")
    async def log_network_transaction_state(self):
        """Log the state of transactions across the network."""
        while True:
            try:
                if self.transaction_tracker.transactions:
                    self.logger.info("\n=== Transaction Network State ===")
                    for tx_hash, state in self.transaction_tracker.transactions.items():
                        self.logger.info(f"Transaction: {tx_hash}")
                        self.logger.info(f"  Status: {state.status}")
                        self.logger.info(f"  Age: {time.time() - state.timestamp:.2f}s")
                        self.logger.info(f"  Propagation count: {state.propagation_count}")
                        self.logger.info(f"  Received by peers: {len(state.received_by)}")
                        self.logger.info(f"  Peers: {state.received_by}")
                        self.logger.info("---")
                await asyncio.sleep(5)  # Log every 5 seconds
            except Exception as e:
                self.logger.error(f"Error logging transaction state: {str(e)}")
                await asyncio.sleep(5)
    async def log_transaction_state(self):
        """Log current transaction state across the network."""
        try:
            local_mempool = len(self.blockchain.mempool) if self.blockchain else 0
            peer_states = {}
            
            for peer in self.connected_peers:
                try:
                    response = await self.send_message(peer, Message(
                        type=MessageType.GET_MEMPOOL.value,
                        payload={}
                    ))
                    if response and response.payload:
                        peer_states[peer] = len(response.payload.get('transactions', []))
                except Exception as e:
                    self.logger.error(f"Error getting mempool from peer {peer}: {str(e)}")
                    peer_states[peer] = "error"

            self.logger.info(f"[TRANSACTION_STATE] Local mempool: {local_mempool}")
            self.logger.info(f"[TRANSACTION_STATE] Peer mempools: {peer_states}")
            
        except Exception as e:
            self.logger.error(f"Error logging transaction state: {str(e)}")

    async def verify_transaction_propagation(self, tx_hash: str):
        """Verify that a transaction has propagated to all peers."""
        try:
            propagation_status = {}
            
            for peer in self.connected_peers:
                try:
                    response = await self.send_message(peer, Message(
                        type=MessageType.VERIFY_TRANSACTION.value,
                        payload={"tx_hash": tx_hash}
                    ))
                    propagation_status[peer] = response.payload.get('exists', False)
                except Exception as e:
                    self.logger.error(f"Error verifying transaction with peer {peer}: {str(e)}")
                    propagation_status[peer] = "error"

            self.logger.info(f"[PROPAGATION] Transaction {tx_hash} status across network:")
            self.logger.info(f"[PROPAGATION] Status by peer: {propagation_status}")
            return propagation_status
            
        except Exception as e:
            self.logger.error(f"Error verifying transaction propagation: {str(e)}")
            return {}
    async def process_multisig_transaction(self, transaction_data: dict) -> Optional[str]:
        """Process a multisig transaction"""
        try:
            # Validate required fields
            required_fields = ['multisig_address', 'sender_public_keys', 'threshold', 
                             'receiver', 'amount', 'message', 'aggregate_proof']
            if not all(field in transaction_data for field in required_fields):
                logger.warning("Missing required fields in multisig transaction")
                return None

            # Process the transaction
            tx_hash = await self.blockchain.add_multisig_transaction(
                transaction_data['multisig_address'],
                transaction_data['sender_public_keys'],
                transaction_data['threshold'],
                transaction_data['receiver'],
                Decimal(transaction_data['amount']),
                transaction_data['message'],
                transaction_data['aggregate_proof']
            )
            
            logger.info(f"Multisig transaction added: {tx_hash}")
            return tx_hash

        except Exception as e:
            logger.error(f"Error processing multisig transaction: {str(e)}")
            raise

    async def process_regular_transaction(self, transaction_data: dict) -> Optional[str]:
        """Process a regular transaction"""
        try:
            transaction = Transaction.from_dict(transaction_data)
            
            # Validate the transaction
            if not await self.blockchain.validate_transaction(transaction):
                logger.warning(f"Invalid transaction: {transaction_data}")
                return None
                
            # Add to blockchain
            tx_hash = await self.blockchain.add_transaction(transaction)
            logger.info(f"Regular transaction added: {tx_hash}")
            return tx_hash

        except Exception as e:
            logger.error(f"Error processing regular transaction: {str(e)}")
            raise

    def validate_transaction_format(self, transaction_data: dict) -> bool:
        """Validate the basic format of a transaction"""
        try:
            # Check if it's a dictionary
            if not isinstance(transaction_data, dict):
                return False

            # Check transaction type
            if transaction_data.get('type') == 'multisig_transaction':
                required_fields = ['multisig_address', 'sender_public_keys', 'threshold', 
                                 'receiver', 'amount', 'message', 'aggregate_proof']
            else:
                required_fields = ['sender', 'receiver', 'amount', 'signature']

            # Validate required fields
            if not all(field in transaction_data for field in required_fields):
                return False

            # Validate amount format
            try:
                if isinstance(transaction_data['amount'], str):
                    Decimal(transaction_data['amount'])
                elif not isinstance(transaction_data['amount'], (int, float, Decimal)):
                    return False
            except:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating transaction format: {str(e)}")
            return False




    async def handle_block(self, block_data: dict, sender: str):
        block = QuantumBlock.from_dict(block_data)
        await self.handle_new_block_dagknight(block, sender)



    async def handle_zk_proof(self, data: dict, sender: str):
        try:
            transaction = Transaction.from_dict(data['transaction'])
            proof = data['proof']
            public_input = int(transaction.hash(), 16)
            if self.zk_system.verify(public_input, proof):
                if await self.blockchain.add_transaction(transaction):
                    await self.broadcast(Message(MessageType.TRANSACTION.value, data['transaction']), exclude=sender)
        except Exception as e:
            logger.error(f"Failed to handle zk proof: {str(e)}")
            logger.error(traceback.format_exc())

    async def send_find_node(self, node_id: str, node: Union[Dict, str]):
        """
        Send a FIND_NODE message to a specific peer and wait for a response.
        
        Args:
            node_id (str): The ID of the node to find.
            node (Union[Dict, str]): The node dictionary containing 'ip' and 'port', or a string in 'ip:port' format.
        
        Returns:
            Optional[Message]: The response message if received, else None.
        """
        try:
            if isinstance(node, dict):
                peer_address = self.normalize_peer_address(node['ip'], node['port'])
            elif isinstance(node, str):
                peer_address = self.normalize_peer_address(node)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")

            response = await self.send_and_wait_for_response(peer_address, Message(MessageType.FIND_NODE.value, {"node_id": node_id}))
            return response
        except Exception as e:
            self.logger.error(f"Failed to send find node: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    async def periodic_transaction_monitoring(self):
        """Monitor transaction propagation periodically."""
        while True:
            try:
                await self.log_transaction_state()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in transaction monitoring: {str(e)}")
                await asyncio.sleep(5)

    # Add to your task definitions in run():

    async def send_find_value(self, node: KademliaNode, key: str) -> Optional[str]:
        try:
            response = await self.send_and_wait_for_response(node.address, Message(MessageType.FIND_VALUE.value, {"key": key}))
            return response.payload.get('value')
        except Exception as e:
            logger.error(f"Failed to send find value: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def send_store(self, node: KademliaNode, key: str, value: str):
        try:
            await self.send_message(node.address, Message(MessageType.STORE.value, {"key": key, "value": value}))
        except Exception as e:
            logger.error(f"Failed to send store: {str(e)}")
            logger.error(traceback.format_exc())

    async def ping_node(self, node_address: str) -> bool:
        try:
            if node_address not in self.peers:
                logger.error(f"Invalid node address: {node_address}. Not in peers list.")
                return False

            # Ping the node
            await self.send_message(node_address, Message(MessageType.HEARTBEAT.value, {"timestamp": time.time()}))
            return True
        except Exception as e:
            logger.error(f"Failed to ping node: {str(e)}")
            return False
    async def broadcast(self, message: Message, max_retries: int = 3, retry_delay: float = 2.0):
        """Broadcast a message to all active peers with enhanced logging."""
        self.logger.info("[BROADCAST] Starting message broadcast")
        self.logger.debug(f"[BROADCAST] Message type: {message.type}")
        
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            self.logger.debug(f"[BROADCAST] Broadcast attempt {retries + 1}/{max_retries}")
            
            if not self.connected_peers:
                self.logger.warning("[BROADCAST] No active peers available")
                return False
                
            self.logger.info(f"[BROADCAST] Broadcasting to {len(self.connected_peers)} peers")
            
            successful_broadcasts = 0
            failed_peers = []
            
            for peer_address in self.connected_peers.copy():
                try:
                    self.logger.debug(f"[BROADCAST] Sending to peer: {peer_address}")
                    await self.send_message(peer_address, message)
                    successful_broadcasts += 1
                    self.logger.debug(f"[BROADCAST] Successfully sent to {peer_address}")
                except Exception as e:
                    self.logger.error(f"[BROADCAST] Failed to send to {peer_address}: {str(e)}")
                    failed_peers.append(peer_address)
                    self.connected_peers.remove(peer_address)
            
            success_rate = successful_broadcasts / len(self.connected_peers) if self.connected_peers else 0
            self.logger.info(f"[BROADCAST] Broadcast completed - Success rate: {success_rate:.1%}")
            self.logger.info(f"[BROADCAST] Successful: {successful_broadcasts}, Failed: {len(failed_peers)}")
            
            if successful_broadcasts > 0:
                success = True
            else:
                retries += 1
                if retries < max_retries:
                    self.logger.warning(f"[BROADCAST] Retrying broadcast in {retry_delay} seconds")
                    await asyncio.sleep(retry_delay)
        
        return success


    async def is_peer_connected(self, peer):
        
        websocket = self.peers.get(peer)
        if websocket and websocket.open:
            await asyncio.wait_for(websocket.ping(), timeout=5)
            return True
        return False
    def cleanup_challenges(self):
        """Safely cleanup old challenges"""
        try:
            current_time = time.time()
            for peer_id in list(self.challenges.keys()):
                peer_challenges = self.challenges[peer_id]
                for ch_id in list(peer_challenges.keys()):
                    challenge_data = peer_challenges[ch_id]
                    if isinstance(challenge_data, dict) and current_time - challenge_data.get('timestamp', 0) > 300:
                        del peer_challenges[ch_id]
                if not peer_challenges:
                    del self.challenges[peer_id]
        except Exception as e:
            self.logger.error(f"Error cleaning up challenges: {str(e)}")

    async def periodic_cleanup(self):
        while True:
            self.cleanup_challenges()
            await asyncio.sleep(60)  # Run every minute
    async def send_message(self, peer: str, message: Message):
        """Send message with comprehensive state checking and encryption handling."""
        try:
            # Normalize peer address and validate
            peer_normalized = self.normalize_peer_address(peer)
            if not peer_normalized:
                raise ValueError(f"Invalid peer address: {peer}")

            self.logger.debug(f"[SEND] Preparing to send {message.type} to {peer_normalized}")

            # Verify connection state with timeout
            try:
                if not await asyncio.wait_for(
                    self.is_peer_connected(peer_normalized),
                    timeout=5.0
                ):
                    raise ConnectionError(f"No active connection for {peer_normalized}")
            except asyncio.TimeoutError:
                raise ConnectionError(f"Connection check timeout for {peer_normalized}")

            # Define allowed message types that don't require verification
            allowed_unverified_types = {
                MessageType.CHALLENGE_RESPONSE.value,
                MessageType.PUBLIC_KEY_EXCHANGE.value,
                MessageType.CHALLENGE.value,
                MessageType.HANDSHAKE.value
            }

            # Get current peer state safely
            async with self.peer_lock:
                try:
                    websocket = self.peers.get(peer_normalized)
                    if not websocket or websocket.closed:
                        raise ConnectionError(f"No active websocket for {peer_normalized}")
                    
                    current_state = self.peer_states.get(peer_normalized)
                    if not current_state:
                        raise ValueError(f"No state found for peer {peer_normalized}")

                    self.logger.debug(f"[SEND] Peer state: {current_state}")

                    # Check verification requirements
                    if (current_state != "verified" and 
                        message.type not in allowed_unverified_types):
                        self.logger.warning(
                            f"Cannot send {message.type} to unverified peer {peer_normalized}"
                        )
                        return False

                    # Handle message encryption
                    try:
                        if (current_state == "verified" and 
                            message.type not in allowed_unverified_types):
                            # Get and validate encryption key
                            recipient_public_key = self.peer_public_keys.get(peer_normalized)
                            if not recipient_public_key:
                                raise ValueError(f"No public key for {peer_normalized}")

                            # Convert message to JSON and encrypt
                            message_json = message.to_json()
                            self.logger.debug(f"[SEND] Encrypting message of size {len(message_json)}")
                            
                            encrypted_message = self.encrypt_message(
                                message_json, 
                                recipient_public_key
                            )
                            
                            # Send encrypted message
                            await websocket.send(encrypted_message)
                            self.logger.debug(
                                f"[SEND] Sent encrypted {message.type} message "
                                f"to {peer_normalized}"
                            )
                        else:
                            # Send unencrypted message for special types
                            message_json = message.to_json()
                            await websocket.send(message_json)
                            self.logger.debug(
                                f"[SEND] Sent unencrypted {message.type} message "
                                f"to {peer_normalized}"
                            )

                        return True

                    except websockets.exceptions.ConnectionClosed as conn_error:
                        self.logger.warning(
                            f"Connection closed while sending to {peer_normalized}: "
                            f"{str(conn_error)}"
                        )
                        raise

                except Exception as inner_error:
                    self.logger.error(
                        f"Error in message sending critical section: {str(inner_error)}"
                    )
                    raise

        except Exception as e:
            self.logger.error(f"Error sending message to {peer_normalized}: {str(e)}")
            self.logger.error(traceback.format_exc())
            await self.remove_peer(peer_normalized)
            return False

        finally:
            # Update last activity time if still connected
            try:
                async with self.peer_lock:
                    if (peer_normalized in self.peer_info and 
                        peer_normalized in self.peers):
                        self.peer_info[peer_normalized]['last_activity'] = time.time()
            except Exception as update_error:
                self.logger.error(f"Error updating activity time: {str(update_error)}")



    async def ensure_peer_connection(self, peer_normalized: str) -> bool:
        """Ensure that a connection exists with the peer, establishing one if necessary."""
        if peer_normalized in self.peers and await self.is_peer_connected(peer_normalized):
            return True

        try:
            # If the peer is not connected, try to establish a new connection
            websocket = await websockets.connect(f"ws://{peer_normalized}", timeout=10)
            self.peers[peer_normalized] = websocket
            
            # Perform handshake and key exchange
            if not await self.perform_handshake(peer_address, websocket):  # Note the argument order
                self.logger.warning(f"Handshake failed with peer {peer_normalized}")
                await self.remove_peer(peer_normalized)
                return False

            self.logger.info(f"Established new connection with peer {peer_normalized}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to establish connection with peer {peer_normalized}: {str(e)}")
            return False

    async def send_raw_message(self, peer: str, message: Message):
        """Send raw message with enhanced error handling."""
        try:
            if peer not in self.peers:
                raise ValueError(f"Peer {peer} not found")

            # Convert any bytes in payload to base64 strings
            if isinstance(message.payload, dict):
                encoded_payload = {}
                for key, value in message.payload.items():
                    if isinstance(value, bytes):
                        encoded_payload[key] = base64.b64encode(value).decode('utf-8')
                    else:
                        encoded_payload[key] = value
                message.payload = encoded_payload

            message_json = message.to_json()
            self.logger.debug(f"Raw message sent to peer {peer}: {message_json[:100]}...")
            
            websocket = self.peers[peer]
            if not websocket.open:
                raise ValueError(f"Connection to peer {peer} is closed")
                
            await websocket.send(message_json)

        except Exception as e:
            self.logger.error(f"Error sending raw message to {peer}: {str(e)}")
            raise

    async def decode_message_payload(self, message: Message) -> Message:
        """Decode base64-encoded values in message payload."""
        try:
            if isinstance(message.payload, dict):
                decoded_payload = {}
                for key, value in message.payload.items():
                    if isinstance(value, str) and key.endswith('_key'):
                        try:
                            decoded_payload[key] = base64.b64decode(value)
                        except:
                            decoded_payload[key] = value
                    else:
                        decoded_payload[key] = value
                message.payload = decoded_payload
            return message
        except Exception as e:
            self.logger.error(f"Error decoding message payload: {str(e)}")
            return message


    async def send_and_wait_for_response(self, peer: str, message: Message, timeout: float = 300.0) -> Optional[Message]:
        try:
            message.id = str(uuid.uuid4())  # Add a unique ID to the message
            await self.send_message(peer, message)
            self.logger.debug(f"Sent message of type {message.type} to peer {peer}")
            
            start_time = time.time()
            while True:  # Keep waiting indefinitely
                if time.time() - start_time > timeout:
                    self.logger.warning(f"Long wait for response from {peer}. Retrying...")
                    # Optionally, you can resend the message here
                    await self.send_message(peer, message)
                    start_time = time.time()  # Reset the timer
                
                response = await self.receive_message(peer, timeout=10.0)  # Short timeout for each receive attempt
                if response and response.id == message.id:
                    return response
                elif response:
                    self.logger.debug(f"Received unrelated message, continuing to wait for response to {message.id}")
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error communicating with peer {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Don't remove the peer here, just log the error
            # Optionally, you can implement a retry mechanism here
            return None

                        
    async def wait_for_challenge_response(self, peer: str, challenge_id: str, timeout: float = 30) -> bool:
        """Wait for and verify challenge response with better error handling"""
        peer_normalized = self.normalize_peer_address(peer)
        start_time = time.time()
        
        try:
            self.logger.info(f"\n[CHALLENGE] Waiting for response from {peer_normalized}")
            self.logger.info(f"[CHALLENGE] Challenge ID: {challenge_id}")

            while time.time() - start_time < timeout:
                try:
                    # Check if peer connection still exists
                    if peer_normalized not in self.peers:
                        self.logger.warning(f"[CHALLENGE] Lost connection to {peer_normalized}")
                        return False

                    # Wait for message with smaller timeout
                    message = await self.receive_message(peer_normalized, timeout=5.0)
                    if not message:
                        continue

                    if message.type.lower() == "challenge_response":
                        self.logger.debug(f"[CHALLENGE] Received response for challenge {message.challenge_id}")
                        
                        # Verify the challenge ID matches
                        if message.challenge_id == challenge_id:
                            verification_result = await self.verify_challenge_response(
                                peer_normalized, 
                                challenge_id, 
                                message.payload
                            )
                            
                            if verification_result:
                                self.logger.info(f"[CHALLENGE] ✓ Challenge response verified for {peer_normalized}")
                                
                                # Move to quantum initialization
                                self.logger.info(f"[QUANTUM] Challenge verified, proceeding with quantum initialization")
                                self.peer_states[peer_normalized] = "verified"
                                return True
                            else:
                                self.logger.error(f"[CHALLENGE] ✗ Challenge verification failed for {peer_normalized}")
                                return False
                                    
                    else:
                        self.logger.debug(f"[CHALLENGE] Received non-challenge message: {message.type}")
                        
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning(f"[CHALLENGE] Connection closed by {peer_normalized}")
                    return False
                    
                except Exception as e:
                    self.logger.error(f"[CHALLENGE] Error processing response: {str(e)}")
                    await asyncio.sleep(1)
                    
            self.logger.error(f"[CHALLENGE] ✗ Challenge response timeout for {peer_normalized}")
            return False
                
        except Exception as e:
            self.logger.error(f"[CHALLENGE] Error in wait_for_challenge_response: {str(e)}")
            return False


    async def reconnect_to_peer(self, peer: str) -> bool:
        """Attempt to reconnect to a peer"""
        try:
            self.logger.info(f"[RECONNECT] Attempting to reconnect to {peer}")
            
            # Remove old connection
            await self.remove_peer(peer)
            
            # Get peer IP and port
            ip, port = peer.split(':')
            port = int(port)
            
            # Create new connection
            websocket = await websockets.connect(
                f"ws://{ip}:{port}",
                timeout=10,
                close_timeout=5
            )
            
            # Store new connection
            self.peers[peer] = websocket
            
            # Re-establish security
            if await self.exchange_public_keys(peer):
                self.logger.info(f"[RECONNECT] Successfully reconnected to {peer}")
                return True
            else:
                self.logger.error(f"[RECONNECT] Failed to exchange keys with {peer}")
                await self.remove_peer(peer)
                return False
                
        except Exception as e:
            self.logger.error(f"[RECONNECT] Failed to reconnect to {peer}: {str(e)}")
            return False

    async def handle_connection_closed(self, peer: str):
        """Handle websocket connection closure"""
        try:
            self.logger.warning(f"[CONNECTION] Connection closed with {peer}")
            
            # Check if we're in the middle of a challenge
            active_challenges = [cid for cid, data in self.challenges.items() if 'peer' in data and data['peer'] == peer]
            
            if active_challenges:
                self.logger.info(f"[CONNECTION] Active challenges found for {peer}, attempting reconnect")
                if await self.reconnect_to_peer(peer):
                    return True
            
            await self.remove_peer(peer)
            return False
            
        except Exception as e:
            self.logger.error(f"[CONNECTION] Error handling closed connection: {str(e)}")
            return False


    async def receive_message(self, peer: str, timeout: float = 300.0) -> Optional[Message]:
        """
        Receive and process messages with improved encryption and binary data handling.
        """
        peer_normalized = self.normalize_peer_address(peer)
        if peer_normalized not in self.peer_locks:
            self.peer_locks[peer_normalized] = Lock()

        async with self.peer_locks[peer_normalized]:
            if peer_normalized not in self.peers:
                return None
            try:
                websocket = self.peers[peer_normalized]
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                
                if not raw_message:
                    self.logger.warning(f"Empty message from {peer_normalized}")
                    return None

                try:
                    # First try to parse as unencrypted JSON
                    try:
                        message_data = json.loads(raw_message)
                        message = Message.from_json(raw_message)
                        message_type = message.type.lower() if message.type else ""
                        
                        # Handle unencrypted message types
                        if message_type in ["public_key_exchange", "challenge", "challenge_response"]:
                            self.logger.debug(f"Received unencrypted {message_type} message")
                            # Decode any base64 encoded binary data in payload
                            if isinstance(message.payload, dict):
                                decoded_payload = {}
                                for key, value in message.payload.items():
                                    if isinstance(value, str) and key.endswith(('_key', '_bytes')):
                                        try:
                                            decoded_payload[key] = base64.b64decode(value)
                                        except:
                                            decoded_payload[key] = value
                                    else:
                                        decoded_payload[key] = value
                                message.payload = decoded_payload
                            return message

                    except json.JSONDecodeError:
                        # Not valid JSON, likely encrypted
                        if peer_normalized not in self.peer_public_keys:
                            self.logger.error(f"No public key for {peer_normalized}")
                            return None
                        
                        try:
                            # Decrypt the message
                            decrypted_message = self.decrypt_message(raw_message)
                            if not decrypted_message:
                                self.logger.error(f"Decryption failed for message from {peer_normalized}")
                                return None

                            # Parse the decrypted message
                            message = Message.from_json(decrypted_message)
                            
                            # Decode any base64 encoded binary data in encrypted payload
                            if isinstance(message.payload, dict):
                                decoded_payload = {}
                                for key, value in message.payload.items():
                                    if isinstance(value, str) and key.endswith(('_key', '_bytes')):
                                        try:
                                            decoded_payload[key] = base64.b64decode(value)
                                        except:
                                            decoded_payload[key] = value
                                    else:
                                        decoded_payload[key] = value
                                message.payload = decoded_payload
                                
                            self.logger.debug(f"Successfully decrypted and parsed message of type: {message.type}")
                            return message

                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse decrypted message: {str(e)}")
                            return None
                        except Exception as e:
                            self.logger.error(f"Error processing encrypted message: {str(e)}")
                            return None

                except Exception as e:
                    self.logger.error(f"Error processing message from {peer_normalized}: {str(e)}")
                    return None

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"Connection closed by {peer_normalized}")
                await self.remove_peer(peer_normalized)
                return None
            except asyncio.TimeoutError:
                # Send keepalive instead of closing
                await self.send_keepalive(peer_normalized)
                return None
            except Exception as e:
                self.logger.error(f"Error in receive_message for {peer_normalized}: {str(e)}")
                await self.remove_peer(peer_normalized)
                return None

            return None

    async def verify_peer(self, peer: str) -> bool:
        """Verify peer connection state."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            async with self.peer_lock:
                # Update peer state
                self.peer_states[peer_normalized] = "verified"
                if peer_normalized not in self.connected_peers:
                    self.connected_peers.add(peer_normalized)
                
                self.logger.info(f"Peer {peer_normalized} verified")
                return True
                
        except Exception as e:
            self.logger.error(f"Error verifying peer {peer}: {str(e)}")
            return False

    async def is_peer_connected(self, peer: str) -> bool:
        """Check if peer is connected with connection test."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            websocket = self.peers.get(peer_normalized)
            
            if not websocket or websocket.closed:
                return False

            try:
                pong_waiter = await websocket.ping()
                await asyncio.wait_for(pong_waiter, timeout=2.0)
                return True
            except:
                return False
                
        except Exception:
            return False

    # Helper function to send keep-alive messages
    async def send_keepalive(self, peer: str):
        try:
            keepalive_message = Message(type=MessageType.HEARTBEAT.value, payload={})
            await self.send_message(peer, keepalive_message)
            self.logger.debug(f"Sent keep-alive message to {peer}")
        except Exception as e:
            self.logger.error(f"Failed to send keep-alive to {peer}: {str(e)}")


    def is_self_connection(self, ip: str, port: int) -> bool:
        own_ips = [self.host]
        if self.external_ip:
            own_ips.append(self.external_ip)
        is_self = ip in own_ips and port == self.port
        logger.debug(f"Checking if {ip}:{port} is self: {is_self}")
        return is_self

    async def handle_public_key_exchange(self, message: Message, peer: str) -> bool:
        """Handle public key exchange with enhanced key handling."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.info(f"\n[KEY_EXCHANGE] {'='*20} Processing Key Exchange {'='*20}")
            self.logger.info(f"[KEY_EXCHANGE] Peer: {peer_normalized}")
            self.logger.debug(f"[KEY_EXCHANGE] Current peer state: {self.peer_states.get(peer_normalized, 'unknown')}")

            # Extract and validate payload data
            payload = message.payload
            if not payload:
                raise ValueError("Empty payload in key exchange message")
                
            # Handle both public_key and rsa_key fields for compatibility
            peer_key_pem = payload.get('rsa_key') or payload.get('public_key')
            if not peer_key_pem:
                raise ValueError("Missing RSA key in exchange message")

            # Load and validate the public key
            try:
                self.logger.debug("[KEY_EXCHANGE] Loading peer public key...")
                peer_public_key = serialization.load_pem_public_key(
                    peer_key_pem.encode(),
                    backend=default_backend()
                )
                if not isinstance(peer_public_key, rsa.RSAPublicKey):
                    raise ValueError("Invalid RSA key type")
                self.logger.debug("[KEY_EXCHANGE] Successfully loaded peer RSA key")
            except Exception as e:
                raise ValueError(f"Invalid RSA key format: {str(e)}")

            # Handle quantum key if present
            peer_quantum_key = payload.get('quantum_key')
            if peer_quantum_key:
                try:
                    if isinstance(peer_quantum_key, str):
                        quantum_key_bytes = base64.b64decode(peer_quantum_key)
                    else:
                        quantum_key_bytes = peer_quantum_key
                        
                    if peer_normalized not in self.peer_crypto_providers:
                        self.peer_crypto_providers[peer_normalized] = CryptoProvider()
                    
                    self.logger.debug("[KEY_EXCHANGE] Quantum key processed successfully")
                    
                except Exception as e:
                    self.logger.error(f"[KEY_EXCHANGE] Error processing quantum key: {str(e)}")
                    # Continue with classical only if quantum fails

            # Store the peer's public key
            async with self.peer_lock:
                self.peer_public_keys[peer_normalized] = peer_public_key
                self.peer_info[peer_normalized] = {
                    'node_id': payload.get('node_id'),
                    'version': payload.get('version', '1.0'),
                    'quantum_capable': bool(peer_quantum_key),
                    'last_activity': time.time(),
                    'key_exchange_complete': True
                }

                # Update peer state if not already verified
                if self.peer_states.get(peer_normalized) != "verified":
                    self.peer_states[peer_normalized] = "key_exchanged"

            # Respond with our key if this is a client-initiated exchange
            role = payload.get('role', '')
            if role == 'client':
                try:
                    self.logger.debug("[KEY_EXCHANGE] Preparing server response...")
                    # Convert our public key to PEM format
                    rsa_public_key = self.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ).decode()
                    
                    # Prepare quantum key if available
                    quantum_public_key = None
                    if hasattr(self, 'crypto_provider'):
                        quantum_public_key = base64.b64encode(
                            self.crypto_provider.pq_public_key
                        ).decode('utf-8')

                    # Send our response
                    response = Message(
                        type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                        payload={
                            "rsa_key": rsa_public_key,
                            "quantum_key": quantum_public_key,
                            "node_id": self.node_id,
                            "role": "server",
                            "timestamp": time.time()
                        }
                    )
                    await self.send_raw_message(peer_normalized, response)
                    self.logger.debug("[KEY_EXCHANGE] Server response sent")

                    # Send challenge after key exchange
                    challenge_id = await self.send_challenge(peer_normalized)
                    if not challenge_id:
                        raise ValueError("Failed to send challenge")
                    self.logger.debug("[KEY_EXCHANGE] Challenge sent")

                except Exception as e:
                    self.logger.error(f"[KEY_EXCHANGE] Error in server response: {str(e)}")
                    raise

            self.logger.info(f"[KEY_EXCHANGE] ✓ Public key exchange completed with {peer_normalized}")
            self.logger.info(f"[KEY_EXCHANGE] {'='*50}\n")
            return True

        except Exception as e:
            self.logger.error(f"[KEY_EXCHANGE] Error handling public key exchange: {str(e)}")
            self.logger.error(traceback.format_exc())
            await self.remove_peer(peer_normalized)
            return False





    def decrypt_message(self, encrypted_message: str) -> Optional[str]:
        """Decrypt message with improved error handling."""
        try:
            # Decode base64 if needed
            if isinstance(encrypted_message, str):
                try:
                    encrypted_bytes = base64.b64decode(encrypted_message)
                except:
                    encrypted_bytes = encrypted_message.encode()
            else:
                encrypted_bytes = encrypted_message

            self.logger.debug(f"Decrypting message of length: {len(encrypted_bytes)} bytes")

            # Split into chunks for RSA decryption
            chunk_size = 256  # RSA 2048 decryption chunk size
            chunks = [encrypted_bytes[i:i + chunk_size] 
                     for i in range(0, len(encrypted_bytes), chunk_size)]

            self.logger.debug(f"Message split into {len(chunks)} chunks, each of size {chunk_size}")

            # Decrypt each chunk
            decrypted_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    decrypted_chunk = self.private_key.decrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    decrypted_chunks.append(decrypted_chunk)
                    self.logger.debug(f"Successfully decrypted chunk {i + 1} of size {len(chunk)} bytes")
                except Exception as e:
                    self.logger.error(f"Failed to decrypt chunk {i + 1}: {str(e)}")
                    return None

            # Combine decrypted chunks and decode
            decrypted_data = b''.join(decrypted_chunks)
            decrypted_message = decrypted_data.decode('utf-8')
            
            self.logger.debug(f"Successfully decrypted message (first 100 chars): {decrypted_message[:100]}...")
            return decrypted_message

        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            return None

        except Exception as e:
            # Log the error and the full base64-encoded encrypted message for debugging
            self.logger.error(f"Decryption failed: {str(e)}")
            self.logger.error(f"Full encrypted message (base64): {encrypted_message}")
            raise ValueError(f"Decryption failed: {str(e)}")
    async def remove_peer(self, peer: str):
        """Remove peer with comprehensive cleanup."""
        try:
            async with self.peer_lock:
                if peer in self.peers:
                    self.logger.info(f"[CLEANUP] Removing peer {peer}")
                    
                    # Close websocket if needed
                    websocket = self.peers[peer]
                    if websocket and not websocket.closed:
                        try:
                            await websocket.close()
                        except Exception as e:
                            self.logger.debug(f"Error closing websocket for {peer}: {str(e)}")

                    # Clean up all peer-related state
                    self.peers.pop(peer, None)
                    self.peer_states.pop(peer, None)
                    self.peer_public_keys.pop(peer, None)
                    self.connected_peers.discard(peer)
                    self.challenges.pop(peer, None)
                    
                    # Clean up quantum state if applicable
                    if hasattr(self, 'quantum_sync'):
                        self.quantum_sync.entangled_peers.pop(peer, None)
                        self.quantum_sync.bell_pairs.pop(peer, None)
                    
                    self.logger.info(f"Peer {peer} removed successfully")

        except Exception as e:
            self.logger.error(f"Error removing peer {peer}: {str(e)}")


    async def attempt_reconnection(self, peer: str):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting to reconnect to {peer} (Attempt {attempt + 1}/{max_retries})")
                ip, port = peer.split(':')
                node = KademliaNode(id=self.generate_node_id(), ip=ip, port=int(port))
                
                if await self.connect_to_peer(node):
                    self.logger.info(f"Successfully reconnected to {peer}")
                    return
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
            
            await asyncio.sleep(5 * (2 ** attempt))  # Exponential backoff
        
        self.logger.warning(f"Failed to reconnect to {peer} after {max_retries} attempts")

    def log_connected_peers(self):
        connected_peers = [peer for peer, state in self.peer_states.items() if state == "connected"]
        logger.info(f"Connected peers: {', '.join(connected_peers)}")


    def log_connected_peers(self):
        connected_peers = [peer for peer, state in self.peer_states.items() if state == "connected"]
        logger.info(f"Connected peers: {', '.join(connected_peers)}")
        
            
    async def republish_data(self):
        try:
            for key, value in self.data_store.items():
                await self.store(key, value)
        except Exception as e:
            logger.error(f"Failed to republish data: {str(e)}")
            logger.error(traceback.format_exc())
    async def create_challenge_response(self, peer: str, challenge_id: str) -> Message:
        try:
            # Normalize the peer address
            peer_normalized = self.normalize_peer_address(peer)

            # Check if the peer has any pending challenges
            if peer_normalized not in self.pending_challenges:
                raise ValueError(f"No pending challenges found for peer {peer_normalized}")

            # Check if the specific challenge_id exists for the peer
            if challenge_id not in self.pending_challenges[peer_normalized]:
                raise KeyError(f"Challenge ID {challenge_id} not found for peer {peer_normalized}")

            # Retrieve the challenge
            challenge = self.pending_challenges[peer_normalized][challenge_id]

            # Generate signature for the challenge
            signature = self.private_key.sign(
                challenge.encode(),
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            # Generate proof (e.g., hash of the challenge)
            proof = hashes.Hash(hashes.SHA256())
            proof.update(challenge.encode())
            proof_value = proof.finalize().hex()  # Convert to hexadecimal string

            # Return the message with both the signature and proof
            return Message(
                type=MessageType.CHALLENGE_RESPONSE.value,
                payload={
                    "signature": signature.hex(),
                    "proof": proof_value  # Add the proof here
                },
                challenge_id=challenge_id
            )

        except KeyError as e:
            logger.error(f"Error in create_challenge_response: {str(e)}")
            logger.debug(f"Pending challenges for peer {peer_normalized}: {self.pending_challenges.get(peer_normalized, {})}")
            return None  # Or handle error more gracefully

        except Exception as e:
            logger.error(f"Unexpected error in create_challenge_response: {str(e)}")
            logger.error(traceback.format_exc())
            return None  # Or handle error more gracefully
    def get_status(self):
        return {
            "is_connected": self.is_connected(),
            "num_peers": len(self.peers),
            "peers": list(self.peers.keys())
        }
    async def check_connections(self):
        while True:
            logger.info(f"Current connection status: Connected peers: {len(self.peers)}")
            logger.info(f"Peer list: {', '.join(self.peers.keys())}")
            await asyncio.sleep(60)  # Check every minute
            
    async def wait_for_connection(self, timeout=30):
        retries = timeout // 5
        while retries > 0:
            if self.is_connected():  # Assuming you have an `is_connected` method
                logger.info("P2PNode connected.")
                return True
            else:
                logger.warning("No peers connected, retrying in 5 seconds...")
            retries -= 1
            await asyncio.sleep(5)
        
        logger.error("Timed out waiting for P2PNode to connect.")
        return False
    async def check_peer_connection(self, peer_id: str) -> bool:
        """
        Check if a specific peer is still connected and responsive.
        
        Args:
            peer_id (str): The ID of the peer to check
            
        Returns:
            bool: True if peer is connected and responsive, False otherwise
        """
        try:
            if peer_id not in self.peers:
                return False

            websocket = self.peers[peer_id]
            if not websocket or not websocket.open:
                return False

            try:
                pong_waiter = await websocket.ping()
                await asyncio.wait_for(pong_waiter, timeout=5)
                return True
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                return False

        except Exception as e:
            self.logger.debug(f"Error checking peer {peer_id} connection: {str(e)}")
            return False

    async def is_connected(self) -> bool:
        """
        Asynchronously check if the P2P node is properly connected to the network.
        
        Returns:
            bool: True if connected to peers, False otherwise
        """
        try:
            # First check if we have any peers
            if len(self.peers) == 0:
                self.logger.debug("No peers connected")
                return False

            # Check each peer's connection status
            active_peers = 0
            for peer_id, peer in self.peers.items():
                try:
                    if await self.check_peer_connection(peer_id):
                        active_peers += 1
                except Exception as e:
                    self.logger.debug(f"Error checking peer {peer_id}: {str(e)}")
                    continue

            # Log connection status
            self.logger.debug(f"Active peers: {active_peers}/{len(self.peers)}")
            return active_peers > 0

        except Exception as e:
            self.logger.error(f"Error checking connection status: {str(e)}")
            return False


    def is_peer_active(self, peer):
        try:
            return peer.ws.open  # Assuming peer has a WebSocket and ws.open indicates an active connection
        except Exception as e:
            logger.error(f"Error checking peer activity: {e}")
            return False

    def is_peer_active(self, peer):
        try:
            return peer.ws.open  # Assuming peer has a WebSocket and ws.open indicates an active connection
        except Exception as e:
            logger.error(f"Error checking peer activity: {e}")
            return False
    async def get_active_peers(self) -> List[str]:
        async with self.peer_lock:
            active_peers = list(self.connected_peers)
        self.logger.debug(f"Active peers: {active_peers}")
        self.logger.debug(f"All peers: {list(self.peers.keys())}")
        self.logger.debug(f"Peer states: {self.peer_states}")
        return active_peers

    async def handle_keepalive(self, message: Message, sender: str):
        logger.debug(f"Received keepalive from {sender}")
        # You can implement additional logic here if needed
    async def periodic_connection_check(self):
        while True:
            try:
                active_peers = await self.get_active_peers()
                if not active_peers:
                    logger.warning("No active peers, attempting to rejoin network")
                    await self.join_network()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in periodic connection check: {str(e)}")
                await asyncio.sleep(60)
                    
    async def join_network(self):
        """Join the P2P network using bootstrap nodes from .env."""
        try:
            self.logger.info("\n=== Joining Network ===")
            bootstrap_nodes = self.bootstrap_nodes
            self.logger.info(f"Bootstrap nodes: {bootstrap_nodes}")

            if not bootstrap_nodes:
                self.logger.warning("No bootstrap nodes configured")
                return False

            connected = False
            for bootstrap_node in bootstrap_nodes:
                if not bootstrap_node.strip():
                    continue

                try:
                    self.logger.info(f"Attempting to connect to bootstrap node: {bootstrap_node}")
                    host, port = bootstrap_node.strip().split(':')
                    port = int(port)

                    # Create KademliaNode instance
                    node_id = hashlib.sha1(f"{host}:{port}".encode()).hexdigest()
                    bootstrap_kademlia = KademliaNode(
                        id=node_id,
                        ip=host,
                        port=port,
                        magnet_link=None
                    )

                    # Try to connect with timeout
                    try:
                        connection_success = await asyncio.wait_for(
                            self.connect_to_peer(bootstrap_kademlia),
                            timeout=30
                        )
                        if connection_success:
                            self.logger.info(f"Successfully connected to bootstrap node {bootstrap_node}")
                            connected = True
                            break
                        else:
                            self.logger.warning(f"Failed to connect to bootstrap node {bootstrap_node}")
                    except asyncio.TimeoutError:
                        self.logger.error(f"Connection attempt to {bootstrap_node} timed out")

                except Exception as e:
                    self.logger.error(f"Error connecting to bootstrap node {bootstrap_node}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    continue

            if not connected:
                self.logger.warning("Failed to connect to any bootstrap nodes")
                return False

            self.logger.info(f"Successfully joined network with {len(self.connected_peers)} peers")
            # Initiate state sync with connected peers
            await self.sync_with_connected_peers()
            return True

        except Exception as e:
            self.logger.error(f"Error joining network: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    async def request_peer_list(self, peer: str) -> List[str]:
        """Request peer list from a connected peer."""
        try:
            self.logger.debug(f"Requesting peer list from {peer}")
            message = Message(
                type=MessageType.PEER_LIST_REQUEST.value,
                payload={}
            )
            
            response = await self.send_and_wait_for_response(peer, message, timeout=10.0)
            if response and response.type == MessageType.PEER_LIST_RESPONSE.value:
                peer_list = response.payload.get('peers', [])
                self.logger.debug(f"Received {len(peer_list)} peers from {peer}")
                return peer_list
            return []
        except Exception as e:
            self.logger.error(f"Error requesting peer list from {peer}: {str(e)}")
            return []
    def is_self_connection(self, ip: str, port: int) -> bool:
        """
        Check if a given IP and port combination represents this node itself.
        
        Args:
            ip (str): IP address to check
            port (int): Port number to check
            
        Returns:
            bool: True if this is a self-connection, False otherwise
        """
        own_ips = {self.host, self.external_ip, 'localhost', '127.0.0.1'}
        if ip in own_ips and port == self.port:
            self.logger.debug(f"Detected self-connection attempt: {ip}:{port}")
            return True
        return False

    async def cleanup_invalid_connections(self):
        """Clean up invalid or stale connections."""
        self.logger.info("Starting cleanup of invalid connections")
        try:
            async with self.peer_lock:
                # Track peers to remove
                peers_to_remove = set()
                
                # Check for self-connections
                for peer in list(self.peers.keys()):
                    try:
                        ip, port = peer.split(':')
                        port = int(port)
                        if self.is_self_connection(ip, port):
                            self.logger.warning(f"Found self-connection to remove: {peer}")
                            peers_to_remove.add(peer)
                    except ValueError:
                        self.logger.warning(f"Invalid peer address format: {peer}")
                        peers_to_remove.add(peer)

                # Check for stale 'connecting' states
                current_time = time.time()
                for peer, state in self.peer_states.items():
                    if state == 'connecting' and peer in self.peers:
                        websocket = self.peers[peer]
                        if not websocket.open or current_time - getattr(websocket, 'last_activity', 0) > 60:
                            self.logger.warning(f"Found stale connecting state: {peer}")
                            peers_to_remove.add(peer)

                # Remove invalid peers
                for peer in peers_to_remove:
                    await self.remove_peer(peer)
                    
                self.logger.info(f"Cleanup complete. Removed {len(peers_to_remove)} invalid connections")
                
                # Log current state
                self.log_peer_status()

        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {str(e)}")
            self.logger.error(traceback.format_exc())
    async def cleanup_connections(self):
        """Clean up stale and invalid connections periodically."""
        while self.is_running:
            try:
                self.logger.info("\n=== Starting Connection Cleanup ===")
                async with self.peer_lock:
                    peers_to_remove = set()
                    
                    # Get current peer addresses and states
                    current_peers = list(self.peers.keys())
                    self.logger.info(f"Current peer count: {len(current_peers)}")
                    
                    for peer in current_peers:
                        try:
                            ip, port = peer.split(':')
                            port = int(port)
                            
                            # Check for self-connections
                            if self.is_self_connection(ip, port):
                                self.logger.warning(f"Found self-connection to remove: {peer}")
                                peers_to_remove.add(peer)
                                continue
                                
                            # Check for stale 'connecting' states
                            if self.peer_states.get(peer) == 'connecting':
                                if time.time() - self.last_activity_time > 60:  # 60 seconds timeout
                                    self.logger.warning(f"Found stale connecting state: {peer}")
                                    peers_to_remove.add(peer)
                                    continue
                            
                            # Check for duplicate connections to same IP with different ports
                            for other_peer in current_peers:
                                if peer != other_peer and other_peer.split(':')[0] == ip:
                                    if self.peer_states.get(peer) == 'connecting':
                                        self.logger.warning(f"Found duplicate connection: {peer}")
                                        peers_to_remove.add(peer)
                                        break

                            # Verify connection is still alive
                            if not await self.is_connection_alive(peer):
                                self.logger.warning(f"Found dead connection: {peer}")
                                peers_to_remove.add(peer)
                                
                        except ValueError:
                            self.logger.warning(f"Invalid peer address format: {peer}")
                            peers_to_remove.add(peer)

                    # Remove identified peers
                    for peer in peers_to_remove:
                        await self.remove_peer(peer)
                    
                    # Log cleanup results
                    self.logger.info(f"Removed {len(peers_to_remove)} invalid connections")
                    self.logger.info(f"Remaining peers: {len(self.peers)}")
                    
                    # Update active peers list
                    self.connected_peers = {peer for peer in self.peers.keys() 
                                         if self.peer_states.get(peer) == 'connected'}
                    
                # Log final state
                self.log_peer_status()
                self.logger.info("=== Connection Cleanup Complete ===\n")
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error during connection cleanup: {str(e)}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(30)

    def is_self_connection(self, ip: str, port: int) -> bool:
        """
        Check if a given IP and port combination represents this node itself.
        
        Args:
            ip (str): IP address to check
            port (int): Port number to check
            
        Returns:
            bool: True if this is a self-connection, False otherwise
        """
        own_ips = {
            self.host, 
            self.external_ip, 
            'localhost', 
            '127.0.0.1',
            socket.gethostbyname(socket.gethostname())
        }
        
        # Clean up None values
        own_ips.discard(None)
        
        is_self = (ip in own_ips and port == self.port)
        if is_self:
            self.logger.debug(f"Detected self-connection attempt: {ip}:{port}")
        return is_self
    async def connect_to_peer(self, node: KademliaNode) -> bool:
        """Connect to peer with hybrid classical/quantum security and proper address tracking."""
        original_address = f"{node.ip}:{node.port}"
        websocket = None
        crypto = None
        actual_address = None
        
        try:
            self.logger.info(f"\n[CONNECT] {'='*20} Connecting to Peer {'='*20}")
            self.logger.info(f"[CONNECT] Target: {original_address}")

            # Initialize connection state
            async with self.peer_lock:
                if original_address in self.peer_address_map:
                    actual_address = self.peer_address_map[original_address]
                    if actual_address in self.peers:
                        self.logger.info(f"[CONNECT] Already connected to {actual_address}")
                        return True
                    
                if len(self.connected_peers) >= self.max_peers:
                    self.logger.warning(f"[CONNECT] Max peers ({self.max_peers}) reached")
                    return False

            # Step 1: Establish WebSocket connection
            try:
                websocket = await asyncio.wait_for(
                    websockets.connect(
                        f"ws://{node.ip}:{node.port}",
                        timeout=30,
                        ping_interval=None,
                        close_timeout=5
                    ),
                    timeout=30
                )
                
                # Get actual connection address
                remote_ip, remote_port = websocket.remote_address[:2]
                actual_address = f"{remote_ip}:{remote_port}"
                
                # Store address mappings
                async with self.peer_lock:
                    self.peer_address_map[original_address] = actual_address
                    self.peer_address_map[actual_address] = actual_address
                    self.peer_ports[actual_address] = node.port
                    
                self.logger.info(f"[CONNECT] ✓ WebSocket connection established")
                self.logger.debug(f"[CONNECT] Address mapping: {original_address} -> {actual_address}")
                
            except Exception as e:
                raise ConnectionError(f"WebSocket connection failed: {str(e)}")

            # Initialize CryptoProvider for quantum-resistant encryption
            crypto = CryptoProvider()
            
            # Register peer with actual address
            async with self.peer_lock:
                self.peers[actual_address] = websocket
                self.peer_states[actual_address] = "connecting"

            # Step 2: Hybrid Key Exchange
            rsa_public_key = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            quantum_public_key = base64.b64encode(crypto.pq_public_key).decode('utf-8')

            key_message = Message(
                type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                payload={
                    "rsa_key": rsa_public_key,
                    "quantum_key": quantum_public_key,
                    "node_id": self.node_id,
                    "role": "client",
                    "original_address": original_address,
                    "actual_address": actual_address
                }
            )
            
            await self.send_raw_message(actual_address, key_message)
            self.logger.info(f"[CONNECT] ✓ Sent hybrid key exchange")

            # Wait for peer's hybrid key exchange response
            response = await asyncio.wait_for(
                self.receive_message(actual_address),
                timeout=10.0
            )
            
            if not response or response.type != MessageType.PUBLIC_KEY_EXCHANGE.value:
                raise ValueError(f"Expected hybrid key exchange, got {response.type if response else 'no response'}")

            # Handle key exchange response using actual address
            if not await self.handle_public_key_exchange(response, actual_address):
                raise ValueError("Public key exchange failed")
            
            self.logger.info(f"[CONNECT] ✓ Hybrid key exchange completed")

            # Step 3: Wait for and handle challenge
            challenge_msg = await asyncio.wait_for(
                self.receive_message(actual_address),
                timeout=10.0
            )
            
            if not challenge_msg or challenge_msg.type != MessageType.CHALLENGE.value:
                raise ValueError(f"Expected challenge, got {challenge_msg.type if challenge_msg else 'no message'}")

            challenge_data = challenge_msg.payload.get('challenge')
            if not challenge_data:
                raise ValueError("Invalid challenge format")

            # Handle the challenge using actual address
            if not await self.handle_challenge(actual_address, challenge_data, challenge_msg.challenge_id):
                raise ValueError("Failed to handle challenge")
                
            self.logger.info(f"[CONNECT] ✓ Challenge handled successfully")

            # Wait for verification with timeout
            verification_timeout = 30.0
            verification_start = time.time()
            verified = False
            
            while time.time() - verification_start < verification_timeout:
                async with self.peer_lock:
                    state = self.peer_states.get(actual_address)
                    self.logger.debug(f"[CONNECT] Current peer state: {state}")
                    
                    if state == "verified":
                        verified = True
                        break
                    elif state == "challenge_response_sent":
                        await asyncio.sleep(0.1)
                    else:
                        raise ValueError(f"Unexpected state during verification: {state}")

            if not verified:
                raise ValueError("Verification timeout")

            # Step 4: Complete handshake with actual address
            if not await self.complete_connection(actual_address):
                raise ValueError("Failed to complete connection")
            
            # Store quantum crypto provider with actual address
            self.peer_crypto_providers[actual_address] = crypto
            
            self.logger.info(f"[CONNECT] ✓ Connection completed successfully")
            self.logger.debug(f"[CONNECT] Final peer state: {self.peer_states.get(actual_address)}")
            self.logger.debug(f"[CONNECT] Connected peers: {list(self.connected_peers)}")
            self.logger.info(f"[CONNECT] {'='*50}\n")
            return True

        except Exception as e:
            self.logger.error(f"[CONNECT] Connection failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Clean up with proper address
            if websocket and not websocket.closed:
                await websocket.close()
                
            if actual_address:
                if crypto:
                    self.peer_crypto_providers.pop(actual_address, None)
                await self.remove_peer(actual_address)
                
                # Clean up address mappings
                async with self.peer_lock:
                    self.peer_address_map.pop(original_address, None)
                    self.peer_address_map.pop(actual_address, None)
                    self.peer_ports.pop(actual_address, None)
                    
            return False

    async def verify_quantum_challenge_response(
        self, 
        response: Message,
        challenge: bytes,
        shared_secret: bytes,
        crypto: CryptoProvider
    ) -> bool:
        """Verify challenge response with quantum resistance."""
        try:
            if not response or response.type != MessageType.CHALLENGE_RESPONSE.value:
                return False
                
            encrypted_response = bytes.fromhex(response.payload['encrypted_response'])
            decrypted = crypto.pq_decrypt(encrypted_response)
            
            return decrypted == challenge + shared_secret
            
        except Exception as e:
            self.logger.error(f"Challenge verification failed: {str(e)}")
            return False




    async def validate_connection(self, node: KademliaNode, peer_address: str) -> bool:
        """Validate the connection attempt."""
        try:
            if self.is_self_connection(node.ip, node.port):
                self.logger.warning(f"[CONNECT] Prevented self-connection to {peer_address}")
                return False

            if peer_address in self.connected_peers:
                self.logger.info(f"[CONNECT] Already connected to {peer_address}")
                return True

            if len(self.connected_peers) >= self.max_peers:
                self.logger.warning(f"[CONNECT] Max peers ({self.max_peers}) reached")
                return False

            return True
        except Exception as e:
            self.logger.error(f"[CONNECT] Validation error: {str(e)}")
            return False

    async def establish_websocket(self, peer_address: str) -> Optional[websockets.WebSocketClientProtocol]:
        """Establish WebSocket connection with retry logic."""
        try:
            self.logger.info(f"[CONNECT] Establishing WebSocket connection to {peer_address}")
            websocket = await asyncio.wait_for(
                websockets.connect(
                    f"ws://{peer_address}",
                    timeout=30,
                    ping_interval=None,
                    max_size=2**23,
                    close_timeout=5
                ),
                timeout=30
            )
            self.logger.info(f"[CONNECT] ✓ WebSocket connection established")
            return websocket
        except Exception as e:
            self.logger.error(f"[CONNECT] WebSocket connection failed: {str(e)}")
            return None

    async def register_peer(self, peer_address: str, websocket) -> bool:
        """Register the peer in the node's peer tracking."""
        try:
            async with self.peer_lock:
                self.peers[peer_address] = websocket
                self.peer_states[peer_address] = "connecting"
                self.connected_peers.add(peer_address)
                self.logger.info(f"[CONNECT] ✓ Peer registered: {peer_address}")
                return True
        except Exception as e:
            self.logger.error(f"[CONNECT] Peer registration failed: {str(e)}")
            return False

    async def perform_security_handshake(self, peer_address: str) -> bool:
        """Perform security handshake with the peer."""
        try:
            if self.peer_states.get(peer_address) in ["verified", "quantum_initializing"]:
                return True

            exchange_success = await asyncio.wait_for(
                self.exchange_public_keys(peer_address), 
                timeout=60
            )
            if not exchange_success:
                raise ValueError("Public key exchange failed")

            challenge_id = await self.send_challenge(peer_address)
            if not challenge_id:
                if self.peer_states.get(peer_address) in ["verified", "quantum_initializing"]:
                    return True
                raise ValueError("Failed to send challenge")

            if not await self.wait_for_challenge_response(peer_address, challenge_id, timeout=60):
                raise ValueError("Challenge response verification failed")

            self.logger.info(f"[CONNECT] ✓ Security handshake completed")
            return True

        except Exception as e:
            self.logger.error(f"[CONNECT] Security handshake failed: {str(e)}")
            return False

    async def initialize_quantum_components(self, peer_address: str):
        """Initialize quantum components for the peer connection."""
        try:
            if not self.quantum_initialized:
                await self.setup_quantum_components()

            await self.establish_quantum_entanglement(peer_address)

        except Exception as e:
            self.logger.error(f"[QUANTUM] Initialization error: {str(e)}")
            self.logger.warning(f"[QUANTUM] Continuing without quantum entanglement")

    def start_peer_monitoring(self, websocket, peer_address: str):
        """Start monitoring tasks for the peer."""
        asyncio.create_task(self.handle_messages(websocket, peer_address))
        asyncio.create_task(self.keep_connection_alive(websocket, peer_address))
        if peer_address in self.quantum_sync.entangled_peers:
            asyncio.create_task(self.monitor_peer_quantum_state(peer_address))
            asyncio.create_task(self.send_quantum_heartbeats(peer_address))

    async def handle_connection_error(self, peer_address: str, websocket, error: Exception):
        """Handle connection errors and cleanup."""
        self.logger.error(f"[CONNECT] Connection failed to {peer_address}: {str(error)}")
        self.logger.error(traceback.format_exc())

        if websocket:
            try:
                await websocket.close()
            except Exception:
                pass

        await self.remove_peer(peer_address)
        self.logger.info(f"[CONNECT] {'='*50}\n")

    async def log_connection_success(self, peer_address: str, start_time: float):
        """Log successful connection details."""
        connection_time = time.time() - start_time
        self.logger.info(f"[CONNECT] ✓ Connected to {peer_address} in {connection_time:.2f}s")
        self.logger.info(f"[CONNECT] ✓ State: {self.peer_states.get(peer_address, 'unknown')}")
        if hasattr(self, 'quantum_sync'):
            self.logger.info(f"[CONNECT] ✓ Quantum peers: {len(self.quantum_sync.entangled_peers)}")
        self.logger.info(f"[CONNECT] {'='*50}\n")

    async def validate_connection(self, node: KademliaNode, peer_address: str) -> bool:
        """Validate the connection attempt."""
        try:
            if self.is_self_connection(node.ip, node.port):
                self.logger.warning(f"[CONNECT] Prevented self-connection to {peer_address}")
                return False

            if peer_address in self.connected_peers:
                self.logger.info(f"[CONNECT] Already connected to {peer_address}")
                return True

            if len(self.connected_peers) >= self.max_peers:
                self.logger.warning(f"[CONNECT] Max peers ({self.max_peers}) reached")
                return False

            return True
        except Exception as e:
            self.logger.error(f"[CONNECT] Validation error: {str(e)}")
            return False

    async def establish_websocket(self, peer_address: str) -> Optional[websockets.WebSocketClientProtocol]:
        """Establish WebSocket connection with retry logic."""
        try:
            self.logger.info(f"[CONNECT] Establishing WebSocket connection to {peer_address}")
            websocket = await asyncio.wait_for(
                websockets.connect(
                    f"ws://{peer_address}",
                    timeout=30,
                    ping_interval=None,
                    max_size=2**23,
                    close_timeout=5
                ),
                timeout=30
            )
            self.logger.info(f"[CONNECT] ✓ WebSocket connection established")
            return websocket
        except Exception as e:
            self.logger.error(f"[CONNECT] WebSocket connection failed: {str(e)}")
            return None

    async def register_peer(self, peer_address: str, websocket) -> bool:
        """Register the peer in the node's peer tracking."""
        try:
            async with self.peer_lock:
                self.peers[peer_address] = websocket
                self.peer_states[peer_address] = "connecting"
                self.connected_peers.add(peer_address)
                self.logger.info(f"[CONNECT] ✓ Peer registered: {peer_address}")
                return True
        except Exception as e:
            self.logger.error(f"[CONNECT] Peer registration failed: {str(e)}")
            return False

    async def perform_security_handshake(self, peer_address: str) -> bool:
        """Perform security handshake with the peer."""
        try:
            if self.peer_states.get(peer_address) in ["verified", "quantum_initializing"]:
                return True

            exchange_success = await asyncio.wait_for(
                self.exchange_public_keys(peer_address), 
                timeout=60
            )
            if not exchange_success:
                raise ValueError("Public key exchange failed")

            challenge_id = await self.send_challenge(peer_address)
            if not challenge_id:
                if self.peer_states.get(peer_address) in ["verified", "quantum_initializing"]:
                    return True
                raise ValueError("Failed to send challenge")

            if not await self.wait_for_challenge_response(peer_address, challenge_id, timeout=60):
                raise ValueError("Challenge response verification failed")

            self.logger.info(f"[CONNECT] ✓ Security handshake completed")
            return True

        except Exception as e:
            self.logger.error(f"[CONNECT] Security handshake failed: {str(e)}")
            return False

    async def initialize_quantum_components(self):
        """Initialize quantum components without requiring a peer."""
        try:
            self.logger.info("[QUANTUM] Initializing quantum components...")
            
            # Initialize quantum sync if not already done
            if not hasattr(self, 'quantum_sync') or self.quantum_sync is None:
                self.quantum_sync = QuantumEntangledSync(self.node_id)
                
            # Get initial state from blockchain
            initial_data = {
                'wallets': [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else [],
                'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions(limit=100)] if self.blockchain else [],
                'blocks': [block.to_dict() for block in self.blockchain.chain] if self.blockchain else [],
                'mempool': [tx.to_dict() for tx in self.blockchain.mempool] if hasattr(self.blockchain, 'mempool') else []
            }

            # Initialize quantum state
            await self.quantum_sync.initialize_quantum_state(initial_data)
            
            # Initialize quantum notifier
            if not hasattr(self, 'quantum_notifier'):
                self.quantum_notifier = QuantumStateNotifier(self)
            
            # Initialize quantum monitor
            if not hasattr(self, 'quantum_monitor'):
                self.quantum_monitor = QuantumNetworkMonitor(self)
                
            # Initialize quantum consensus
            if not hasattr(self, 'quantum_consensus'):
                self.quantum_consensus = QuantumConsensusManager(self)

            # Initialize quantum heartbeat tracking
            if not hasattr(self, 'quantum_heartbeats'):
                self.quantum_heartbeats = {}
            if not hasattr(self, 'last_quantum_heartbeat'):
                self.last_quantum_heartbeat = {}
                
            # Set initialization flags
            self.quantum_initialized = True
            
            # Start quantum monitoring
            asyncio.create_task(self.monitor_quantum_state())
            
            self.logger.info("[QUANTUM] ✓ Quantum components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error initializing quantum components: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise



    def start_peer_monitoring(self, websocket, peer_address: str):
        """Start monitoring tasks for the peer."""
        asyncio.create_task(self.handle_messages(websocket, peer_address))
        asyncio.create_task(self.keep_connection_alive(websocket, peer_address))
        if peer_address in self.quantum_sync.entangled_peers:
            asyncio.create_task(self.monitor_peer_quantum_state(peer_address))
            asyncio.create_task(self.send_quantum_heartbeats(peer_address))

    async def handle_connection_error(self, peer_address: str, websocket, error: Exception):
        """Handle connection errors and cleanup."""
        self.logger.error(f"[CONNECT] Connection failed to {peer_address}: {str(error)}")
        self.logger.error(traceback.format_exc())

        if websocket:
            try:
                await websocket.close()
            except Exception:
                pass

        await self.remove_peer(peer_address)
        self.logger.info(f"[CONNECT] {'='*50}\n")

    async def log_connection_success(self, peer_address: str, start_time: float):
        """Log successful connection details."""
        connection_time = time.time() - start_time
        self.logger.info(f"[CONNECT] ✓ Connected to {peer_address} in {connection_time:.2f}s")
        self.logger.info(f"[CONNECT] ✓ State: {self.peer_states.get(peer_address, 'unknown')}")
        if hasattr(self, 'quantum_sync'):
            self.logger.info(f"[CONNECT] ✓ Quantum peers: {len(self.quantum_sync.entangled_peers)}")
        self.logger.info(f"[CONNECT] {'='*50}\n")



    async def wait_for_public_key(self, peer: str, timeout: float = 30.0) -> bool:
        """Wait for peer's public key with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if peer in self.peer_public_keys:
                return True
            await asyncio.sleep(0.1)
        return False

    async def setup_quantum_entanglement(self, peer: str):
        """Set up quantum entanglement with a peer."""
        try:
            self.logger.info(f"[QUANTUM] Initializing quantum entanglement with {peer}")
            
            # Get current state
            current_state = {
                'wallets': [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else {},
                'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions()] if self.blockchain else {},
                'blocks': [block.to_dict() for block in self.blockchain.chain] if self.blockchain else {},
                'mempool': [tx.to_dict() for tx in self.blockchain.mempool] if self.blockchain else {}
            }

            # Create entanglement request
            request_message = Message(
                type=MessageType.QUANTUM_ENTANGLEMENT_REQUEST.value,
                payload={
                    'node_id': self.node_id,
                    'state_data': current_state,
                    'timestamp': time.time()
                }
            )

            # Send request and wait for response
            response = await self.send_and_wait_for_response(
                peer,
                request_message,
                timeout=30.0
            )

            if not response or response.type != MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value:
                raise ValueError("Failed to receive quantum entanglement response")

            # Establish entanglement
            peer_state = response.payload.get('state_data', {})
            peer_register = await self.quantum_sync.entangle_with_peer(peer, peer_state)

            if not peer_register:
                raise ValueError("Failed to create quantum register")

            # Start quantum monitoring for this peer
            asyncio.create_task(self.send_quantum_heartbeats(peer))
            asyncio.create_task(self.monitor_peer_quantum_state(peer))

            self.logger.info(f"[QUANTUM] ✓ Quantum entanglement established with {peer}")
            return True

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error in quantum setup: {str(e)}")
            raise


    async def finalize_connection(self, peer_address: str):
        self.logger.info(f"[FINALIZE] Finalizing connection with peer {peer_address}")
        async with self.peer_lock:
            try:
                # Set peer state to connected and add to active peers
                self.peer_states[peer_address] = "connected"
                logger.debug(f"Attempting to add peer {peer_address} to active peer list")
                self.connected_peers.add(peer_address)
                logger.debug(f"Peer list after adding: {self.connected_peers}")

                # Check if peer is already in self.peers, if not, establish WebSocket connection
                if peer_address not in self.peers:
                    self.logger.warning(f"[FINALIZE] Peer {peer_address} not found in self.peers. Adding it.")
                    websocket = await websockets.connect(f"ws://{peer_address}")
                    self.peers[peer_address] = websocket
            
                # Complete connection setup for the peer
                await self.complete_connection(peer_address)
                self.logger.info(f"[FINALIZE] Connection finalized with peer {peer_address}")
                self.log_peer_status()

                # Sync state with the new peer
                await self.sync_new_peer(peer_address)

                # Set up real-time message handling
                asyncio.create_task(self.handle_real_time_messages(peer_address))

                # Set up periodic sync check
                asyncio.create_task(self.periodic_sync_check(peer_address))

                # Verification logs after finalization
                self.logger.info(f"[FINALIZE] Verification after finalization:")
                self.logger.info(f"  Peer in self.peers: {peer_address in self.peers}")
                self.logger.info(f"  Peer in self.connected_peers: {peer_address in self.connected_peers}")
                self.logger.info(f"  Peer state: {self.peer_states.get(peer_address)}")

            except Exception as e:
                self.logger.error(f"Error finalizing connection with peer {peer_address}: {str(e)}")
                await self.remove_peer(peer_address)
    async def handle_real_time_messages(self, peer: str):
        """Handle real-time messages from a peer"""
        while peer in self.connected_peers:
            try:
                message = await self.receive_message(peer)
                if not message:
                    continue

                # Handle high-priority messages immediately
                if message.type in [MessageType.TRANSACTION.value, MessageType.BLOCK.value]:
                    asyncio.create_task(self.handle_message(message, peer))
                else:
                    # Queue other messages for normal processing
                    await self.message_queue.put((message, peer))

            except Exception as e:
                self.logger.error(f"Error in real-time message handling for {peer}: {str(e)}")
                await self.remove_peer(peer)
                break

    def log_peer_status(self):
        self.logger.info("Current peer status:")
        self.logger.info(f"  All peers: {list(self.peers.keys())}")
        self.logger.info(f"  Peer states: {self.peer_states}")
        self.logger.info(f"  Connected peers: {self.connected_peers}")
        active_peers = [peer for peer in self.connected_peers if self.peer_states.get(peer) == "connected"]
        self.logger.info(f"  Active peers: {active_peers}")
    async def handle_messages(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        """Handle incoming messages with enhanced debugging."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            role = "client" if peer_normalized in self.bootstrap_nodes else "server"
            
            self.logger.info(f"\n[HANDLER] {'='*20} Message Handler Started {'='*20}")
            self.logger.info(f"[HANDLER] Peer: {peer_normalized} (Role: {role})")
            self.logger.info(f"[HANDLER] Current peer state: {self.peer_states.get(peer_normalized, 'unknown')}")

            while True:
                try:
                    self.logger.debug(f"[HANDLER] Waiting for message from {peer_normalized}")
                    raw_message = await websocket.recv()
                    self.logger.debug(f"[HANDLER] Received raw message: {raw_message[:100]}...")
                    
                    message = Message.from_json(raw_message)
                    message_type = message.type.lower() if message.type else ""
                    
                    self.logger.info(f"[HANDLER] Processing {message_type} message from {peer_normalized}")
                    
                    if message_type == MessageType.CHALLENGE.value:
                        self.logger.info(f"[HANDLER] Processing incoming challenge from {peer_normalized}")
                        challenge_str = message.payload.get('challenge')
                        challenge_id = message.challenge_id
                        
                        self.logger.debug(f"[HANDLER] Challenge details:")
                        self.logger.debug(f"  - Challenge ID: {challenge_id}")
                        self.logger.debug(f"  - Challenge string: {challenge_str[:32]}...")
                        
                        if challenge_str and challenge_id:
                            try:
                                # Split challenge
                                if ':' in challenge_str:
                                    _, challenge = challenge_str.split(':', 1)
                                else:
                                    challenge = challenge_str
                                
                                self.logger.debug(f"[HANDLER] Generating signature for challenge")
                                # Generate signature
                                challenge_bytes = challenge.encode()
                                signature = self.private_key.sign(
                                    challenge_bytes,
                                    padding.PSS(
                                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                        salt_length=padding.PSS.MAX_LENGTH
                                    ),
                                    hashes.SHA256()
                                )
                                
                                self.logger.debug(f"[HANDLER] Signature generated successfully")
                                
                                # Create response
                                response = Message(
                                    type=MessageType.CHALLENGE_RESPONSE.value,
                                    payload={"signature": signature.hex()},
                                    challenge_id=challenge_id,
                                    sender=self.node_id
                                )
                                
                                self.logger.debug(f"[HANDLER] Sending challenge response")
                                # Send response immediately
                                await self.send_raw_message(peer_normalized, response)
                                self.logger.info(f"[HANDLER] ✓ Challenge response sent to {peer_normalized}")
                            except Exception as challenge_error:
                                self.logger.error(f"[HANDLER] Error processing challenge: {str(challenge_error)}")
                                self.logger.error(traceback.format_exc())
                        else:
                            self.logger.warning(f"[HANDLER] Invalid challenge received from {peer_normalized}")
                    
                    elif message_type == MessageType.CHALLENGE_RESPONSE.value:
                        self.logger.info(f"[HANDLER] Processing challenge response from {peer_normalized}")
                        if await self.verify_challenge_response(peer_normalized, message.challenge_id, message.payload):
                            self.logger.info(f"[HANDLER] ✓ Challenge response verified for {peer_normalized}")
                        else:
                            self.logger.error(f"[HANDLER] Challenge response verification failed for {peer_normalized}")
                            break
                    
                    else:
                        self.logger.debug(f"[HANDLER] Handling other message type: {message_type}")
                        await self.handle_message(message, peer_normalized)

                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning(f"[HANDLER] Connection closed with {peer_normalized}")
                    break
                except Exception as e:
                    self.logger.error(f"[HANDLER] Error in message loop: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    continue

        except Exception as e:
            self.logger.error(f"[HANDLER] Fatal error in message handler: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info(f"[HANDLER] Message handler stopping for {peer_normalized}")
            await self.remove_peer(peer_normalized)


        async def attempt_reconnection(self, peer: str):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Attempting to reconnect to {peer} (Attempt {attempt + 1}/{max_retries})")
                    ip, port = peer.split(':')
                    node = KademliaNode(id=self.generate_node_id(), ip=ip, port=int(port))
                    
                    if await self.connect_to_peer(node):
                        self.logger.info(f"Successfully reconnected to {peer}")
                        return
                except Exception as e:
                    self.logger.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                
                await asyncio.sleep(5 * (2 ** attempt))  # Exponential backoff
            
            self.logger.warning(f"Failed to reconnect to {peer} after {max_retries} attempts")
                
    async def process_message(self, message: Message, sender: str):
        try:
            logger.debug(f"Processing message from {sender}: {message.to_json()}")

            # Generate a unique hash for the incoming message
            message_hash = hashlib.sha256(message.to_json().encode()).hexdigest()
            logger.debug(f"Generated message hash: {message_hash}")

            # Check if the message has already been processed
            if message_hash in self.seen_messages:
                logger.info(f"Duplicate message from {sender}, ignoring.")
                return

            # Add the message hash to the set of seen_messages
            self.seen_messages.add(message_hash)
            logger.debug(f"Added message hash to seen_messages: {message_hash}")

            # Schedule cleanup of the message hash after processing
            asyncio.create_task(self.clean_seen_messages(message_hash))

            # Handle different message types with detailed logs
            if message.type == MessageType.BLOCK_PROPOSAL.value:
                logger.info(f"[BLOCK_PROPOSAL] Processing block proposal from {sender}: {message.payload}")
                await self.handle_block_proposal(message, sender)
            
            elif message.type == MessageType.NEW_WALLET.value:
                logger.info(f"[NEW_WALLET] Processing new wallet registration from {sender}: {message.payload}")
                await self.handle_new_wallet(message.payload)

            elif message.type == MessageType.PRIVATE_TRANSACTION.value:
                logger.info(f"[PRIVATE_TRANSACTION] Processing private transaction from {sender}: {message.payload}")
                await self.handle_private_transaction(message.payload)

            else:
                logger.warning(f"Unknown message type {message.type} received from {sender}")

        except Exception as e:
            logger.error(f"Error processing message from {sender}: {str(e)}")
            logger.error(traceback.format_exc())


    async def send_ping(self, peer):
        try:
            await self.send_message(peer, Message(type="ping", payload={}))
        except Exception as e:
            logger.error(f"Failed to send ping to {peer}: {e}")
            raise
    async def keep_connection_alive(self, websocket: websockets.WebSocketClientProtocol, peer: str):
        """Maintain connection with verified peer."""
        try:
            while peer in self.peers and not websocket.closed:
                try:
                    # Send ping and wait for pong
                    pong_waiter = await websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    self.logger.debug(f"✓ Keepalive successful for {peer}")
                    
                    # Update last activity time
                    self.peer_info[peer]['last_activity'] = time.time()
                    
                    await asyncio.sleep(20)  # Wait before next ping
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Keepalive timeout for {peer}")
                    break
                except Exception as e:
                    self.logger.error(f"Keepalive error for {peer}: {str(e)}")
                    break
                    
            # Connection is dead, clean up
            await self.remove_peer(peer)
            
        except Exception as e:
            self.logger.error(f"Error in keepalive for {peer}: {str(e)}")
            await self.remove_peer(peer)




    async def is_peer_reachable(self, peer):
        try:
            # Attempt to establish a new connection to the peer
            ip, port = peer.split(':')
            async with websockets.connect(f"ws://{ip}:{port}", timeout=5) as ws:
                await ws.ping()
            return True
        except:
            return False

    async def handle_disconnection(self, peer: str):
        """Handle peer disconnection with proper state cleanup."""
        try:
            self.logger.info(f"Handling disconnection for peer {peer}")
            
            async with self.peer_lock:  # Use lock to prevent race conditions
                # Check if peer still exists before removing
                if peer in self.peers:
                    await self.remove_peer(peer)
                    self.logger.info(f"Disconnected peer {peer} removed")
                else:
                    self.logger.debug(f"Peer {peer} already removed")

            # Only attempt reconnection if not already in progress
            if not self.peer_states.get(peer) == "reconnecting":
                await self.attempt_reconnection(peer)

        except Exception as e:
            self.logger.error(f"Error handling disconnection for {peer}: {str(e)}")




    async def keep_alive(self, websocket: websockets.WebSocketClientProtocol, peer: str):
        try:
            while self.is_running and websocket.open:
                try:
                    await websocket.ping()
                    await asyncio.sleep(self.heartbeat_interval)
                except asyncio.TimeoutError:
                    logger.warning(f"Ping timeout for peer {peer}")
                    break
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed with {peer} during keep-alive: {e.code} {e.reason}")
        finally:
            await self.handle_disconnection(peer)


    def print_pending_challenges(self):
        logger.debug("Current pending challenges:")
        for challenge_id, challenge in self.pending_challenges.items():
            logger.debug(f"  {challenge_id}: {challenge}")
    def create_challenge(self, peer):
        challenge_id = str(uuid.uuid4())
        challenge = f"{challenge_id}:{os.urandom(32).hex()}"
        self.pending_challenges[peer] = {
            "challenge": challenge,
            "timestamp": time.time()
        }
        return challenge_id, challenge

    def cleanup_expired_challenges(self):
        current_time = time.time()
        for peer in list(self.pending_challenges.keys()):
            challenges = self.pending_challenges[peer]
            for challenge_id in list(challenges.keys()):
                if current_time - challenges[challenge_id]["timestamp"] > CHALLENGE_TIMEOUT:
                    del challenges[challenge_id]
            if not challenges:
                del self.pending_challenges[peer]

    async def connection_monitor(self):
        while True:
            try:
                for peer in list(self.peers.keys()):
                    if not await self.is_connection_alive(peer):
                        if await self.is_peer_reachable(peer):
                            self.logger.info(f"Peer {peer} is reachable but connection seems dead. Attempting to reconnect.")
                            await self.handle_disconnection(peer)
                        else:
                            self.logger.warning(f"Peer {peer} is not reachable. Removing from peer list.")
                            await self.remove_peer(peer)
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in connection monitor: {str(e)}")
                await asyncio.sleep(30)



    async def is_connection_alive(self, peer: str) -> bool:
        if peer not in self.peers:
            return False
        try:
            pong_waiter = await self.peers[peer].ping()
            await asyncio.wait_for(pong_waiter, timeout=5)
            return True
        except Exception:
            return False


    async def process_message(self, message: Message, sender: str):
        try:
            logger.debug(f"Processing message from {sender}: {message.to_json()}")

            # Generate a unique hash for the incoming message
            message_hash = hashlib.sha256(message.to_json().encode()).hexdigest()
            logger.debug(f"Generated message hash: {message_hash}")

            # Check if the message has already been processed
            if message_hash in self.seen_messages:
                logger.info(f"Duplicate message from {sender}, ignoring.")
                return

            # Add the message hash to the set of seen_messages
            self.seen_messages.add(message_hash)
            logger.debug(f"Added message hash to seen_messages: {message_hash}")

            # Schedule cleanup of the message hash after processing
            asyncio.create_task(self.clean_seen_messages(message_hash))

            # Handle different message types
            if message.type == "keepalive":
                await self.handle_keepalive(message, sender)
            elif message.type == MessageType.CHALLENGE_RESPONSE.value:
                logger.debug(f"Received challenge response from {sender}")
                await self.handle_challenge_response(sender, message)
            else:
                # Handle other message types based on its content
                await self.handle_message(message, sender)

        except Exception as e:
            # Log the error details if message processing fails
            logger.error(f"Failed to process message from {sender}: {str(e)}")
            logger.error(f"Message content: {message.to_json()}")
            logger.error(f"Traceback: {traceback.format_exc()}")




    async def clean_seen_messages(self, message_hash: str):
        await asyncio.sleep(self.message_expiry)
        self.seen_messages.discard(message_hash)

    async def announce_to_network(self):
        """Announce presence to the network with enhanced logging."""
        self.logger.info("\n=== Announcing to Network ===")
        try:
            if not self.connected_peers:
                self.logger.warning("No peers connected. Cannot announce to network.")
                return False

            self.logger.info(f"Preparing announcement for {len(self.connected_peers)} peers")
            message = Message(
                MessageType.MAGNET_LINK.value,
                {"magnet_link": self.magnet_link.to_uri()}
            )

            # Log magnet link details
            self.logger.debug(f"Magnet Link: {self.magnet_link.to_uri()}")
            self.logger.debug(f"Node ID: {self.node_id}")

            # Broadcast with detailed logging
            announcement_success = await self.broadcast(message)
            
            if announcement_success:
                self.logger.info("✓ Successfully announced to network")
                # Log current network state
                self.logger.info(f"Connected Peers: {len(self.connected_peers)}")
                self.logger.info(f"Active Peers: {list(self.connected_peers)}")
            else:
                self.logger.warning("✗ Failed to announce to some peers")

            self.logger.info("=== Network Announcement Complete ===\n")
            return announcement_success

        except Exception as e:
            self.logger.error("Error announcing to network:")
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            self.logger.info("=== Network Announcement Failed ===\n")
            return False


    async def find_peer_by_magnet(self, magnet_link: MagnetLink) -> Optional[KademliaNode]:
        try:
            nodes = await self.find_node(magnet_link.info_hash)
            for node in nodes:
                if node.magnet_link and node.magnet_link.info_hash == magnet_link.info_hash:
                    return node
            return None
        except Exception as e:
            logger.error(f"Failed to find peer by magnet: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def store_file(self, file_hash: str, file_data: bytes):
        try:
            encoded_data = base64.b64encode(file_data).decode()
            await self.store(file_hash, encoded_data)
        except Exception as e:
            logger.error(f"Failed to store file: {str(e)}")
            logger.error(traceback.format_exc())

    async def retrieve_file(self, file_hash: str) -> Optional[bytes]:
        try:
            encoded_data = await self.get(file_hash)
            if encoded_data:
                return base64.b64decode(encoded_data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve file: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def submit_computation(self, computation_id: str, computation_data: dict):
        try:
            await self.store(computation_id, json.dumps(computation_data))
        except Exception as e:
            logger.error(f"Failed to submit computation: {str(e)}")
            logger.error(traceback.format_exc())

    async def retrieve_computation_result(self, computation_id: str) -> Optional[dict]:
        try:
            result_data = await self.get(computation_id)
            if result_data:
                return json.loads(result_data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve computation result: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    async def keepalive_loop(self, peer):
        while True:
            try:
                await self.send_message(peer, Message(type="keepalive", payload={"timestamp": time.time()}))
                await asyncio.sleep(15)  # Send keepalive every 15 seconds
            except Exception as e:
                logger.error(f"Keepalive failed for {peer}: {str(e)}")
                await self.handle_disconnection(peer)
                break
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle incoming connection with proper state transitions and quantum sync."""
        peer_ip, peer_port = websocket.remote_address[:2]
        peer_address = f"{peer_ip}:{peer_port}"
        peer_normalized = self.normalize_peer_address(peer_address)
        
        try:
            self.logger.info(f"\n[SERVER] {'='*20} New Connection {'='*20}")
            self.logger.info(f"[SERVER] Remote peer: {peer_normalized}")
            self.logger.debug(f"[SERVER] Original address: {peer_address}")

            # Initialize peer state
            async with self.peer_lock:
                # Clean up any existing connection
                if peer_normalized in self.peers:
                    self.logger.warning(f"[SERVER] Existing connection found for {peer_normalized}, cleaning up")
                    await self.remove_peer(peer_normalized)

                # Initialize new connection state
                self.peers[peer_normalized] = websocket
                self.peer_states[peer_normalized] = "server_init"
                self.peer_info[peer_normalized] = {
                    'connection_time': time.time(),
                    'last_activity': time.time(),
                    'handshake_complete': False,
                    'attempts': 0,
                    'capabilities': set(),
                    'original_address': peer_address,
                    'quantum_retry_count': 0
                }

            # Set timeouts for different phases
            timeouts = {
                'key_exchange': 15.0,
                'challenge': 15.0,
                'handshake': 15.0,
                'quantum_init': 30.0,
                'quantum_sync': 45.0
            }

            # Step 1: Public Key Exchange with retries
            for attempt in range(3):
                try:
                    # Send keepalive before waiting
                    if attempt > 0:
                        await self.send_keepalive(peer_normalized)
                    
                    # Wait for client's key with progressive timeout
                    message = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=timeouts['key_exchange'] * (1 + attempt * 0.5)
                    )
                    key_message = Message.from_json(message)
                    
                    if key_message.type != MessageType.PUBLIC_KEY_EXCHANGE.value:
                        raise ValueError(f"Expected public_key_exchange, got {key_message.type}")
                    
                    # Process client's key
                    if not await self.handle_public_key_exchange(key_message, peer_normalized):
                        raise ValueError("Public key exchange failed")
                        
                    self.logger.info(f"[SERVER] ✓ Received and verified client's public key")
                    
                    # Update state and send server's key
                    async with self.peer_lock:
                        self.peer_states[peer_normalized] = "key_exchanged"
                        self.peer_info[peer_normalized]['last_activity'] = time.time()
                    
                    public_key_pem = self.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ).decode()
                    
                    response = Message(
                        type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                        payload={
                            "public_key": public_key_pem,
                            "node_id": self.node_id,
                            "role": "server",
                            "timestamp": time.time()
                        }
                    )
                    await self.send_raw_message(peer_normalized, response)
                    self.logger.info(f"[SERVER] ✓ Sent server public key")
                    break
                    
                except Exception as e:
                    if attempt == 2:
                        raise ValueError(f"Public key exchange failed after 3 attempts: {str(e)}")
                    self.logger.warning(f"[SERVER] Key exchange attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

            # Step 2: Challenge-Response with extended timeout
            try:
                # Send keepalive before challenge
                await self.send_keepalive(peer_normalized)
                
                async with self.peer_lock:
                    current_state = self.peer_states.get(peer_normalized)
                    if current_state != "key_exchanged":
                        raise ValueError(f"Invalid state for challenge: {current_state}")

                challenge_id = await self.send_challenge(peer_normalized)
                if not challenge_id:
                    raise ValueError("Failed to generate challenge")

                self.logger.info(f"[SERVER] ✓ Challenge sent with ID: {challenge_id}")

                # Wait for response with extended timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeouts['challenge'])
                    response = Message.from_json(message)
                    
                    if response.type != MessageType.CHALLENGE_RESPONSE.value:
                        raise ValueError(f"Expected challenge_response, got {response.type}")

                    if not await self.verify_challenge_response(peer_normalized, challenge_id, response.payload):
                        raise ValueError("Challenge verification failed")
                        
                    self.logger.info(f"[SERVER] ✓ Challenge verified")
                    await self.send_keepalive(peer_normalized)

                except asyncio.TimeoutError:
                    raise ValueError("Challenge response timeout")

            except Exception as e:
                self.logger.error(f"[SERVER] Challenge sequence failed: {str(e)}")
                raise

            # Step 3: Complete Connection with Quantum Support
            try:
                await self.ensure_blockchain()
                handshake_data = {
                    "node_id": self.node_id,
                    "version": "1.0",
                    "blockchain_height": len(self.blockchain.chain) if self.blockchain else 0,
                    "timestamp": time.time(),
                    "capabilities": ["quantum", "dagknight", "zkp"],
                    "role": "server"
                }
                
                # Send handshake
                handshake_message = Message(
                    type=MessageType.HANDSHAKE.value,
                    payload=handshake_data
                )
                await self.send_raw_message(peer_normalized, handshake_message)
                self.logger.info(f"[SERVER] ✓ Sent handshake")

                # Wait for peer handshake
                message = await asyncio.wait_for(websocket.recv(), timeout=timeouts['handshake'])
                peer_handshake = Message.from_json(message)
                
                if peer_handshake.type != MessageType.HANDSHAKE.value:
                    raise ValueError(f"Expected handshake, got {peer_handshake.type}")
                
                # Finalize connection
                await self.handle_handshake(peer_normalized, peer_handshake.payload)
                
                # Update connection state
                async with self.peer_lock:
                    self.peer_info[peer_normalized]['handshake_complete'] = True
                    self.peer_states[peer_normalized] = "connected"
                    if peer_normalized not in self.connected_peers:
                        self.connected_peers.add(peer_normalized)

                # Start core handlers with keepalive
                message_handler = asyncio.create_task(
                    self.handle_messages(websocket, peer_normalized)
                )
                keepalive_handler = asyncio.create_task(
                    self.keep_connection_alive(websocket, peer_normalized)
                )

                # Initialize quantum with timeout and retry
                if hasattr(self, 'quantum_sync'):
                    try:
                        # Set quantum initialization state
                        async with self.peer_lock:
                            self.peer_states[peer_normalized] = "quantum_initializing"
                        
                        # Start quantum entanglement with timeout
                        quantum_task = asyncio.create_task(
                            self.establish_quantum_entanglement(peer_normalized)
                        )
                        
                        # Wait for quantum initialization with extended timeout
                        await asyncio.wait_for(
                            quantum_task,
                            timeout=timeouts['quantum_init']
                        )
                        
                        self.logger.info(f"[SERVER] ✓ Quantum initialization completed for {peer_normalized}")
                        
                    except asyncio.TimeoutError:
                        self.logger.warning(f"[SERVER] Quantum initialization timed out for {peer_normalized}")
                        # Continue without quantum - connection is still valid
                    except Exception as qe:
                        self.logger.error(f"[SERVER] Quantum initialization error: {str(qe)}")
                        # Continue without quantum
                    finally:
                        async with self.peer_lock:
                            self.peer_states[peer_normalized] = "connected"

                # Store tasks
                self.peer_tasks[peer_normalized] = {
                    'message_handler': message_handler,
                    'keepalive': keepalive_handler
                }

                self.logger.info(f"[SERVER] ✓ Connection established with {peer_normalized}")

            except Exception as e:
                self.logger.error(f"[SERVER] Connection completion failed: {str(e)}")
                raise

        except Exception as e:
            self.logger.error(f"[SERVER] Connection failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            await self.remove_peer(peer_normalized)

        finally:
            self.logger.info(f"[SERVER] {'='*50}\n")

    async def wait_for_message(self, websocket: websockets.WebSocketServerProtocol, 
                             expected_type: str, timeout: float = 10.0) -> Optional[Message]:
        """Wait for a specific type of message with improved handling."""
        try:
            start_time = time.time()
            processed_messages = set()

            while time.time() - start_time < timeout:
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message = Message.from_json(raw_message)
                    
                    if message.id in processed_messages:
                        continue
                    processed_messages.add(message.id)

                    # Process the message if it's what we're waiting for
                    if message.type.lower() == expected_type.lower():
                        self.logger.debug(f"Received expected message type: {expected_type}")
                        return message
                    else:
                        # Handle other message types that might come in
                        await self.handle_unexpected_message(message)
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error receiving message: {str(e)}")

            self.logger.warning(f"Timeout waiting for message type: {expected_type}")
            return None

        except Exception as e:
            self.logger.error(f"Error in wait_for_message: {str(e)}")
            return None

    async def handle_unexpected_message(self, message: Message):
        """Handle messages that arrive during wait_for_message."""
        try:
            if message.type == MessageType.HANDSHAKE.value:
                peer = message.sender
                if peer:
                    await self.handle_handshake(peer, message.payload)
            elif message.type == MessageType.CHALLENGE.value:
                peer = message.sender
                if peer:
                    challenge = message.payload.get('challenge')
                    await self.handle_challenge(peer, challenge, message.challenge_id)
        except Exception as e:
            self.logger.error(f"Error handling unexpected message: {str(e)}")



    async def verify_connection_state(self, peer: str):
        """Verify final connection state."""
        try:
            # Verify peer state
            if self.peer_states.get(peer) != "quantum_ready":
                raise ValueError(f"Unexpected peer state: {self.peer_states.get(peer)}")
                
            # Verify peer is connected
            if peer not in self.connected_peers:
                raise ValueError("Peer not in connected peers set")
                
            # Verify quantum entanglement
            if peer not in self.quantum_sync.entangled_peers:
                raise ValueError("Peer not quantum entangled")
                
            # Verify Bell pair
            if peer not in self.quantum_sync.bell_pairs:
                raise ValueError("No Bell pair for peer")
                
            self.logger.info(f"[VERIFY] ✓ Connection state verified for {peer}")
            return True
            
        except Exception as e:
            self.logger.error(f"[VERIFY] Connection state verification failed: {str(e)}")
            return False

    async def initialize_quantum_components(self):
        """Initialize quantum components for node."""
        try:
            self.logger.info("[QUANTUM] Initializing quantum components...")
            
            if not hasattr(self, 'quantum_sync') or self.quantum_sync is None:
                self.quantum_sync = QuantumEntangledSync(self.node_id)
                
            # Get initial state data
            initial_data = {
                'wallets': [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else [],
                'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions(limit=100)] if self.blockchain else [],
                'blocks': [block.to_dict() for block in self.blockchain.chain] if self.blockchain else [],
                'mempool': [tx.to_dict() for tx in self.blockchain.mempool] if hasattr(self.blockchain, 'mempool') else []
            }

            await self.quantum_sync.initialize_quantum_state(initial_data)
            self.quantum_initialized = True
            self.logger.info("[QUANTUM] ✓ Quantum components initialized")
            
            return True

        except Exception as e:
            self.logger.error(f"[QUANTUM] Initialization error: {str(e)}")
            return False


    async def establish_quantum_entanglement(self, peer: str) -> bool:
        """Establish quantum entanglement with peer."""
        try:
            self.logger.info(f"\n[QUANTUM] {'='*20} Establishing Entanglement {'='*20}")
            self.logger.info(f"[QUANTUM] Initiating quantum entanglement with {peer}")

            # Ensure quantum components are initialized
            if not self.quantum_initialized:
                await self.initialize_quantum_components()

            # Get current state for entanglement
            current_state = {
                'wallets': [w.to_dict() for w in self.blockchain.get_wallets()],
                'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions()],
                'blocks': [block.to_dict() for block in self.blockchain.chain],
                'mempool': [tx.to_dict() for tx in self.blockchain.mempool]
            }

            # Create and send entanglement request
            request_message = Message(
                type=MessageType.QUANTUM_ENTANGLEMENT_REQUEST.value,
                payload={
                    'node_id': self.node_id,
                    'state_data': current_state,
                    'timestamp': time.time()
                }
            )

            # Wait for response with timeout
            response = await self.send_and_wait_for_response(peer, request_message)
            
            if not response or response.type != MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value:
                self.logger.error(f"[QUANTUM] Failed to receive entanglement response from {peer}")
                return False

            # Create quantum entanglement
            peer_state = response.payload.get('state_data', {})
            peer_register = await self.quantum_sync.entangle_with_peer(peer, peer_state)
            
            if not peer_register:
                self.logger.error(f"[QUANTUM] Failed to create quantum register with {peer}")
                return False

            # Generate and store Bell pair
            bell_pair = self.quantum_sync._generate_bell_pair()
            self.quantum_sync.bell_pairs[peer] = bell_pair

            # Start quantum monitoring
            asyncio.create_task(self.monitor_peer_quantum_state(peer))
            asyncio.create_task(self.send_quantum_heartbeats(peer))

            self.logger.info(f"[QUANTUM] ✓ Quantum entanglement established with {peer}")
            self.logger.info(f"[QUANTUM] {'='*50}\n")
            return True

        except Exception as e:
            self.logger.error(f"[QUANTUM] Error establishing entanglement: {str(e)}")
            return False
    async def handle_successful_challenge(self, peer: str):
        """Handle successful challenge verification and initialize quantum components."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            
            # Update peer state
            async with self.peer_lock:
                self.peer_states[peer_normalized] = "verified"
                self.connected_peers.add(peer_normalized)
                
            # Initialize quantum components if needed
            if not self.quantum_initialized:
                await self.initialize_quantum_components()
                
            # Establish quantum entanglement
            if await self.establish_quantum_entanglement(peer_normalized):
                self.peer_states[peer_normalized] = "quantum_ready"
                self.logger.info(f"[QUANTUM] Node {peer_normalized} is quantum-ready")
            else:
                self.logger.warning(f"[QUANTUM] Failed to establish quantum entanglement with {peer_normalized}")
                # Continue even if quantum setup fails
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in post-challenge handling: {str(e)}")
            return False


    async def monitor_peer_tasks(self, peer: str, tasks: List[asyncio.Task]):
        """Monitor peer-specific tasks and restart them if they fail."""
        while peer in self.connected_peers:
            try:
                for task in tasks[:]:  # Copy list to avoid modification during iteration
                    if task.done():
                        try:
                            await task
                        except Exception as e:
                            self.logger.error(f"Task {task.get_name()} failed for peer {peer}: {str(e)}")
                            # Restart the failed task
                            if task.get_name().startswith('quantum_monitor'):
                                new_task = asyncio.create_task(self.monitor_peer_quantum_state(peer))
                            elif task.get_name().startswith('message_handler'):
                                new_task = asyncio.create_task(self.handle_messages(self.peers[peer], peer))
                            elif task.get_name().startswith('keepalive'):
                                new_task = asyncio.create_task(self.keep_connection_alive(peer))
                            elif task.get_name().startswith('quantum_heartbeat'):
                                new_task = asyncio.create_task(self.send_quantum_heartbeats(peer))
                            
                            tasks.remove(task)
                            tasks.append(new_task)
                            self.logger.info(f"Restarted failed task {task.get_name()} for peer {peer}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring tasks for peer {peer}: {str(e)}")
                await asyncio.sleep(5)
    async def get_initial_quantum_state(self):
        """Get initial state for quantum initialization"""
        try:
            return {
                'wallets': [w.to_dict() for w in self.blockchain.get_wallets()],
                'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions()],
                'blocks': [block.to_dict() for block in self.blockchain.chain],
                'mempool': [tx.to_dict() for tx in self.blockchain.mempool]
            }
        except Exception as e:
            self.logger.error(f"Error getting initial quantum state: {str(e)}")
            return {
                'wallets': {},
                'transactions': {},
                'blocks': {},
                'mempool': {}
            }
    async def maintain_quantum_states(self):
        """Periodically check and maintain quantum states"""
        while True:
            try:
                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                    fidelity = await self.quantum_sync.measure_sync_state(component)
                    
                    if fidelity < self.quantum_sync.decoherence_threshold:
                        self.logger.warning(f"Low fidelity detected for {component}: {fidelity}")
                        
                        # Attempt recovery
                        success = await self.recover_quantum_state(component)
                        if success:
                            self.logger.info(f"Successfully recovered {component} quantum state")
                        else:
                            self.logger.error(f"Failed to recover {component} quantum state")
                            
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in quantum state maintenance: {str(e)}")
                await asyncio.sleep(30)


    async def recover_quantum_state(self, component: str) -> bool:
        try:
            self.logger.info(f"Starting quantum state recovery for {component}")
            
            # Find highest fidelity peer
            best_peer = None
            best_fidelity = 0
            
            for peer_id in self.quantum_sync.entangled_peers:
                fidelity = await self.quantum_sync.measure_sync_state(component, peer_id)
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_peer = peer_id
                    
            if not best_peer:
                return False
                
            # Request state update from best peer
            state_data = await self.request_component_state(best_peer, component)
            if not state_data:
                return False
                
            # Update local quantum state
            await self.quantum_sync.update_component_state(component, state_data)
            
            # Verify recovery
            new_fidelity = await self.quantum_sync.measure_sync_state(component)
            return new_fidelity >= self.quantum_sync.decoherence_threshold
            
        except Exception as e:
            self.logger.error(f"Error in quantum state recovery: {str(e)}")
            return False


    async def validate_block_with_quantum_consensus(self, block: QuantumBlock) -> bool:
        try:
            # First check traditional validation
            if not self.blockchain.validate_block(block):
                return False
                
            # Verify quantum consensus before accepting block
            block_consensus = await self.quantum_consensus.check_quantum_consensus('blocks')
            if not block_consensus:
                self.logger.warning(f"Quantum consensus verification failed for block {block.hash}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in quantum block validation: {str(e)}")
            return False

    async def initialize_quantum_sync(self, initial_data: dict):
        """Initialize quantum synchronization with current blockchain state"""
        try:
            if not self.quantum_sync:
                self.quantum_sync = QuantumEntangledSync(self.node_id)
                
            await self.quantum_sync.initialize_quantum_state(initial_data)
            self.quantum_initialized = True
            self.logger.info("✓ Quantum state initialized with current blockchain state")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum sync: {str(e)}")
            raise
    async def establish_quantum_entanglement(self, peer: str) -> bool:
        """Establish quantum entanglement with a peer with enhanced reliability."""
        try:
            self.logger.info(f"\n[QUANTUM] {'='*20} Establishing Entanglement {'='*20}")
            self.logger.info(f"[QUANTUM] Initiating quantum entanglement with {peer}")
            
            # Send initial keepalive
            await self.send_keepalive(peer)

            # Initialize quantum components with timeout
            if not self.quantum_initialized:
                try:
                    await asyncio.wait_for(
                        self.initialize_quantum_components(),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    self.logger.error("[QUANTUM] Quantum initialization timed out")
                    return False
                except Exception as e:
                    self.logger.error(f"[QUANTUM] Quantum initialization failed: {str(e)}")
                    return False

            # Get current state with error handling
            try:
                current_state = {
                    'wallets': [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else [],
                    'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions(limit=100)] if self.blockchain else [],
                    'blocks': [block.to_dict() for block in self.blockchain.chain] if self.blockchain else [],
                    'mempool': [tx.to_dict() for tx in self.blockchain.mempool] if hasattr(self.blockchain, 'mempool') else []
                }
            except Exception as e:
                self.logger.error(f"[QUANTUM] Error getting current state: {str(e)}")
                return False0

            # Intermediate keepalive
            await self.send_keepalive(peer)

            # Create and send entanglement request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    request_message = Message(
                        type=MessageType.QUANTUM_ENTANGLEMENT_REQUEST.value,
                        payload={
                            'node_id': self.node_id,
                            'state_data': current_state,
                            'timestamp': time.time()
                        }
                    )

                    # Wait for response with timeout
                    response = await asyncio.wait_for(
                        self.send_and_wait_for_response(peer, request_message),
                        timeout=20.0
                    )

                    if not response or response.type != MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"[QUANTUM] Retrying entanglement request ({attempt + 1}/{max_retries})")
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        self.logger.error(f"[QUANTUM] Failed to receive entanglement response from {peer}")
                        return False
                    break
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"[QUANTUM] Request timeout, retrying ({attempt + 1}/{max_retries})")
                        continue
                    self.logger.error("[QUANTUM] Entanglement request timed out")
                    return False

            # Another keepalive before entanglement
            await self.send_keepalive(peer)

            # Create quantum entanglement with timeout
            try:
                peer_state = response.payload.get('state_data', {})
                peer_register = await asyncio.wait_for(
                    self.quantum_sync.entangle_with_peer(peer, peer_state),
                    timeout=30.0
                )
                
                if not peer_register:
                    self.logger.error(f"[QUANTUM] Failed to create quantum register with {peer}")
                    return False

            except asyncio.TimeoutError:
                self.logger.error("[QUANTUM] Quantum register creation timed out")
                return False
            except Exception as e:
                self.logger.error(f"[QUANTUM] Error creating quantum register: {str(e)}")
                return False

            # Generate and store Bell pair
            try:
                bell_pair = self.quantum_sync._generate_bell_pair()
                self.quantum_sync.bell_pairs[peer] = bell_pair

                # Start monitoring tasks with error handling
                for task_name, coro in [
                    ("quantum_monitor", self.monitor_peer_quantum_state(peer)),
                    ("quantum_heartbeat", self.send_quantum_heartbeats(peer))
                ]:
                    try:
                        task = asyncio.create_task(coro)
                        task.set_name(f"{task_name}_{peer}")
                        task.add_done_callback(
                            lambda t: self.logger.error(f"Task {t.get_name()} failed: {t.exception()}")
                            if t.exception() else None
                        )
                    except Exception as e:
                        self.logger.error(f"[QUANTUM] Failed to start {task_name}: {str(e)}")

                # Final keepalive
                await self.send_keepalive(peer)

                self.logger.info(f"[QUANTUM] ✓ Quantum entanglement established with {peer}")
                self.logger.info(f"[QUANTUM] {'='*50}\n")
                return True

            except Exception as e:
                self.logger.error(f"[QUANTUM] Error in final entanglement stage: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"[QUANTUM] Fatal error establishing entanglement: {str(e)}")
            self.logger.error(traceback.format_exc())
            try:
                # Attempt cleanup on failure
                self.quantum_sync.entangled_peers.pop(peer, None)
                self.quantum_sync.bell_pairs.pop(peer, None)
            except Exception:
                pass
            return False



    async def initialize_quantum_entanglement(self, peer: str) -> bool:
        """Initialize quantum entanglement after successful security handshake"""
        try:
            self.logger.info(f"[QUANTUM] Initializing quantum entanglement with {peer}")

            # Wait for any pending security operations to complete
            async with self.peer_lock:
                if self.peer_states.get(peer) != "connected":
                    self.logger.warning(f"Cannot initialize quantum entanglement - peer {peer} not fully connected")
                    return False

            # Get current node state for entanglement
            current_state = await self.get_node_state_data()
            
            # Create and send quantum entanglement request
            entangle_message = Message(
                type=MessageType.QUANTUM_ENTANGLEMENT_REQUEST.value,
                payload={
                    'node_id': self.node_id,
                    'state_data': current_state,
                    'timestamp': time.time()
                }
            )

            # Wait for entanglement response with timeout
            response = await self.send_and_wait_for_response(peer, entangle_message, timeout=30.0)
            
            if not response or response.type != MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value:
                self.logger.warning(f"Failed to receive quantum entanglement response from {peer}")
                return False

            # Initialize quantum sync if not already done
            if not hasattr(self, 'quantum_sync'):
                self.quantum_sync = QuantumEntangledSync(self.node_id)
                self.quantum_sync_initialized = False

            if not self.quantum_sync_initialized:
                await self.initialize_quantum_sync(current_state)

            # Establish quantum entanglement with peer
            peer_state = response.payload.get('state_data', {})
            peer_register = await self.quantum_sync.entangle_with_peer(peer, peer_state)
            
            if not peer_register:
                self.logger.warning(f"Failed to establish quantum entanglement with {peer}")
                return False

            self.logger.info(f"[QUANTUM] Successfully established quantum entanglement with {peer}")
            
            # Start quantum state monitoring for this peer
            asyncio.create_task(self.monitor_peer_quantum_state(peer))
            
            return True

        except Exception as e:
            self.logger.error(f"Error in quantum entanglement initialization with {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    async def finalize_connection_quantum(self, peer: str):
        """Complete connection setup with quantum synchronization"""
        try:
            self.logger.info(f"[FINALIZE] Finalizing quantum-enhanced connection with {peer}")
            
            # Update peer state
            async with self.peer_lock:
                self.peer_states[peer] = "entangled"
                self.connected_peers.add(peer)
            
            # Verify initial quantum sync
            await self.verify_initial_quantum_sync(peer)
            
            # Start quantum sync monitoring
            asyncio.create_task(self.monitor_quantum_sync())
            
            self.logger.info(f"[FINALIZE] Connection with quantum sync completed for {peer}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing quantum connection with {peer}: {str(e)}")
            raise

    async def verify_initial_quantum_sync(self, peer: str):
        """Verify initial quantum synchronization with peer"""
        try:
            components = ['wallets', 'transactions', 'blocks', 'mempool']
            sync_status = {}
            
            for component in components:
                fidelity = await self.quantum_sync.measure_sync_state(component)
                sync_status[component] = {
                    'fidelity': fidelity,
                    'synced': fidelity >= self.quantum_sync.decoherence_threshold
                }
                
                if not sync_status[component]['synced']:
                    self.logger.warning(
                        f"Initial quantum sync verification failed for {component} "
                        f"with peer {peer} (fidelity: {fidelity:.3f})"
                    )
            
            self.logger.info(f"Initial quantum sync verification results for {peer}:")
            for component, status in sync_status.items():
                self.logger.info(
                    f"  {component}: fidelity={status['fidelity']:.3f}, "
                    f"synced={status['synced']}"
                )
                
            return all(status['synced'] for status in sync_status.values())
            
        except Exception as e:
            self.logger.error(f"Error verifying initial quantum sync: {str(e)}")
            return False
    async def handle_quantum_decoherence(self, peer: str, components: List[str]):
        """Handle quantum decoherence with peer"""
        try:
            self.logger.info(f"[QUANTUM] Handling decoherence with peer {peer}")
            
            # Request resynchronization for decoherent components
            resync_message = Message(
                type=MessageType.QUANTUM_RESYNC_REQUEST.value,
                payload={
                    'components': components,
                    'node_id': self.node_id,
                    'timestamp': time.time()
                }
            )
            
            response = await self.send_and_wait_for_response(peer, resync_message)
            
            if response and response.type == MessageType.QUANTUM_RESYNC_RESPONSE.value:
                # Re-establish quantum entanglement for decoherent components
                for component in components:
                    state_data = response.payload.get(f'{component}_data')
                    if state_data:
                        await self.quantum_sync.update_component_state(component, state_data)
                        self.logger.info(f"Re-established quantum sync for {component} with {peer}")
                    else:
                        self.logger.warning(f"Missing state data for {component} from {peer}")
            
            else:
                self.logger.warning(f"Failed to resynchronize quantum state with {peer}")
                
        except Exception as e:
            self.logger.error(f"Error handling quantum decoherence: {str(e)}")
            raise
    async def handle_resync_request(self, message: Message, sender: str):
        """Handle quantum resynchronization request"""
        try:
            components = message.payload.get('components', [])
            response_data = {}
            
            # Gather current state data for requested components
            for component in components:
                if component == 'wallets':
                    response_data['wallets_data'] = [w.to_dict() for w in self.blockchain.get_wallets()]
                elif component == 'transactions':
                    response_data['transactions_data'] = [tx.to_dict() for tx in self.blockchain.get_recent_transactions()]
                elif component == 'blocks':
                    response_data['blocks_data'] = [block.to_dict() for block in self.blockchain.chain]
                elif component == 'mempool':
                    response_data['mempool_data'] = [tx.to_dict() for tx in self.blockchain.mempool]
            
            # Send response with current state data
            response = Message(
                type=MessageType.QUANTUM_RESYNC_RESPONSE.value,
                payload=response_data
            )
            
            await self.send_message(sender, response)
            
        except Exception as e:
            self.logger.error(f"Error handling resync request: {str(e)}")
            raise

    async def monitor_peer_quantum_state(self, peer: str):
        """Monitor quantum synchronization with a peer"""
        while peer in self.connected_peers:
            try:
                # Measure fidelity for all components
                fidelities = {}
                decoherent_components = []
                
                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                    fidelity = await self.quantum_sync.measure_sync_state(component)
                    fidelities[component] = fidelity
                    
                    if fidelity < self.quantum_sync.decoherence_threshold:
                        decoherent_components.append(component)

                # Handle any decoherent components
                if decoherent_components:
                    self.logger.warning(
                        f"[QUANTUM] Decoherence detected with {peer} for components: "
                        f"{decoherent_components}"
                    )
                    await self.request_quantum_resync(peer, decoherent_components)
                
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Error monitoring quantum state for {peer}: {str(e)}")
                await asyncio.sleep(5)

    async def request_quantum_resync(self, peer: str, components: List[str]):
        """Request resynchronization of quantum states"""
        try:
            self.logger.info(f"[QUANTUM] Requesting resync from {peer} for: {components}")
            
            resync_message = Message(
                type=MessageType.QUANTUM_RESYNC_REQUEST.value,
                payload={
                    'components': components,
                    'node_id': self.node_id,
                    'timestamp': time.time()
                }
            )

            response = await self.send_and_wait_for_response(peer, resync_message)
            
            if response and response.type == MessageType.QUANTUM_RESYNC_RESPONSE.value:
                # Update quantum states with received data
                for component in components:
                    state_data = response.payload.get(f'{component}_data')
                    if state_data:
                        await self.quantum_sync.update_component_state(
                            component,
                            state_data
                        )
                        self.logger.info(f"✓ Resynced quantum state for {component}")
                    else:
                        self.logger.warning(f"Missing state data for {component}")

            else:
                self.logger.warning(f"Failed to resync quantum states with {peer}")

        except Exception as e:
            self.logger.error(f"Error requesting quantum resync: {str(e)}")
            raise


    async def perform_handshake(self, peer: str, websocket: websockets.WebSocketServerProtocol) -> bool:
        """Perform connection sequence with correct ordering."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.info(f"\n[CONNECT] {'='*20} Starting Connection {'='*20}")
            self.logger.info(f"[CONNECT] Peer: {peer_normalized}")

            # Step 1: Public Key Exchange
            async with self.peer_lock:
                self.peer_states[peer_normalized] = "key_exchange"
                
            # Send our public key first
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
            
            key_message = Message(
                type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                payload={
                    "public_key": public_key_pem,
                    "node_id": self.node_id,
                    "role": "client"  # We're initiating
                }
            )
            await self.send_raw_message(peer_normalized, key_message)
            self.logger.info(f"[CONNECT] ✓ Sent public key")

            # Wait for peer's public key
            response = await self.receive_message(peer_normalized)
            if not response or response.type != MessageType.PUBLIC_KEY_EXCHANGE.value:
                raise ValueError("Invalid key exchange response")

            if not await self.handle_public_key_exchange(response, peer_normalized):
                raise ValueError("Public key exchange failed")
            self.logger.info(f"[CONNECT] ✓ Key exchange completed")

            # Step 2: Wait for Challenge
            challenge_response = await self.receive_message(peer_normalized)
            if not challenge_response or challenge_response.type != MessageType.CHALLENGE.value:
                raise ValueError("Expected challenge, got different message type")

            # Handle challenge
            challenge_str = challenge_response.payload.get('challenge')
            challenge_id = challenge_response.challenge_id
            if not challenge_str or not challenge_id:
                raise ValueError("Invalid challenge format")

            # Process challenge and send response
            if not await self.handle_challenge(peer_normalized, challenge_str, challenge_id):
                raise ValueError("Failed to process challenge")
            self.logger.info(f"[CONNECT] ✓ Challenge handled")

            # Step 3: Send Handshake only after challenge is complete
            async with self.peer_lock:
                if self.peer_states[peer_normalized] == "verified":
                    handshake_data = {
                        "node_id": self.node_id,
                        "version": "1.0",
                        "blockchain_height": len(self.blockchain.chain) if self.blockchain else 0,
                        "capabilities": ["quantum", "dagknight", "zkp"],
                        "timestamp": time.time(),
                        "quantum_ready": hasattr(self, 'quantum_sync')
                    }
                    
                    handshake_message = Message(
                        type=MessageType.HANDSHAKE.value,
                        payload=handshake_data,
                        sender=self.node_id
                    )
                    await self.send_raw_message(peer_normalized, handshake_message)
                    self.logger.info(f"[CONNECT] ✓ Sent handshake")

                    # Wait for peer's handshake
                    peer_handshake = await self.receive_message(peer_normalized)
                    if not peer_handshake or peer_handshake.type != MessageType.HANDSHAKE.value:
                        raise ValueError("Invalid handshake response")

                    await self.handle_handshake(peer_normalized, peer_handshake.payload)
                    self.logger.info(f"[CONNECT] ✓ Connection sequence completed")
                    return True
                else:
                    raise ValueError("Peer not verified after challenge")

        except Exception as e:
            self.logger.error(f"[CONNECT] Connection sequence failed: {str(e)}")
            await self.remove_peer(peer_normalized)
            return False

        finally:
            self.logger.info(f"[CONNECT] {'='*50}\n")



    async def queue_message_for_later(self, message, peer):
        # Implement a method to queue messages received during handshake
        # This could be a simple list or a more sophisticated queue
        if not hasattr(self, 'queued_messages'):
            self.queued_messages = {}
        if peer not in self.queued_messages:
            self.queued_messages[peer] = []
        self.queued_messages[peer].append(message)
        logger.debug(f"Queued message of type {message.type} from {peer} for later processing")


    async def exchange_node_info(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        try:
            # Send our node info
            node_info = {
                "node_id": self.node_id,
                "version": "1.0",  # Add a version number to your P2PNode
                "capabilities": ["kademlia", "dht", "zkp"]  # List of capabilities
            }
            await websocket.send(json.dumps({"type": "node_info", "data": node_info}))

            # Receive peer's node info
            response = await websocket.recv()
            logger.debug(f"Received challenge response from peer: {response}")

            peer_info = json.loads(response)
            if peer_info["type"] != "node_info":
                raise ValueError("Expected node_info message")

            # Process and store peer's info
            peer_node_id = peer_info["data"]["node_id"]
            peer_version = peer_info["data"]["version"]
            peer_capabilities = peer_info["data"]["capabilities"]

            logger.info(f"Peer {peer} info - ID: {peer_node_id}, Version: {peer_version}, Capabilities: {peer_capabilities}")

            # You might want to store this information for later use
            self.peer_info[peer] = peer_info["data"]

        except Exception as e:
            logger.error(f"Error exchanging node info with {peer}: {str(e)}")
            raise  # Re-raise the exception to be handled by the caller
    def get_peer_count(self):
        return len(self.peers)
    async def start_server(self):
        """Start the WebSocket server with proper message handling."""
        try:
            self.server = await websockets.serve(
                self.handle_websocket_connection,  # Use the connection handler
                self.host,
                self.port
            )
            self.logger.info(f"P2P node listening on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start server: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def handle_websocket_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new WebSocket connections."""
        try:
            self.logger.info(f"New WebSocket connection from {websocket.remote_address}")
            
            async for message in websocket:
                await self.handle_websocket_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed with {websocket.remote_address}")
        except Exception as e:
            self.logger.error(f"Error in WebSocket connection handler: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                # Create a temporary socket to test if port is available
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(1)
                test_socket.bind(('0.0.0.0', port))
                test_socket.close()
                self.logger.info(f"Found available port: {port}")
                return port
            except OSError:
                self.logger.debug(f"Port {port} is in use, trying next port")
                continue
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")
    async def find_and_connect_to_new_peers(self):
        """Find and connect to new peers with enhanced error handling."""
        try:
            self.logger.info("[FIND_PEERS] Attempting to find and connect to new peers")
            nodes = await self.find_node(self.node_id)
            
            if nodes is None:
                self.logger.warning("[FIND_PEERS] No nodes found in network")
                return
                
            if not isinstance(nodes, list):
                self.logger.warning(f"[FIND_PEERS] Invalid nodes response type: {type(nodes)}")
                return
                
            self.logger.info(f"[FIND_PEERS] Found {len(nodes)} potential new peers")
            
            connected_count = 0
            for node in nodes:
                try:
                    if not isinstance(node, KademliaNode):
                        self.logger.warning(f"Invalid node type: {type(node)}")
                        continue
                        
                    if node.id not in self.peers and len(self.peers) < self.target_peer_count:
                        self.logger.info(f"[FIND_PEERS] Attempting to connect to new peer: {node.id}")
                        success = await self.connect_to_peer(node)
                        if success:
                            connected_count += 1
                            self.logger.info(f"[FIND_PEERS] Successfully connected to new peer: {node.id}")
                        else:
                            self.logger.warning(f"[FIND_PEERS] Failed to connect to new peer: {node.id}")
                except Exception as e:
                    self.logger.error(f"[FIND_PEERS] Error connecting to peer {node.id}: {str(e)}")
                    continue

            self.logger.info(f"[FIND_PEERS] Connected to {connected_count} new peers")
            self.logger.info(f"[FIND_PEERS] Current peer count: {len(self.peers)}")
            
        except Exception as e:
            self.logger.error(f"[FIND_PEERS] Error finding new peers: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info("[FIND_PEERS] Finished search for new peers")

    async def run(self):
        """Run the P2P node with enhanced task management and monitoring."""
        try:
            # Initialize core services
            await self.start_server()
            await self.join_network()
            await self.announce_to_network()
            
            self.is_running = True
            self.logger.info("Starting periodic tasks...")

            # Define task configurations with name, coroutine function (not called), and interval
            task_configs = [
                ('Message Queue', self.process_message_queue, 1),
                ('Peer Discovery', self.periodic_peer_discovery, 300),
                ('Data Republish', self.periodic_data_republish, 3600),
                ('Heartbeat', self.send_heartbeats, 15),
                ('Logo Sync', self.periodic_logo_sync, 300),
                ('Connection Check', self.periodic_connection_check, 60),
                ('Peer Maintenance', self.maintain_peer_connections, 30),
                ('Connection Monitor', self.connection_monitor, 30),
                ('Peer Check', self.periodic_peer_check, 30),
                ('Sync Monitor', self.monitor_sync_status, 10),
                ('Resource Monitor', self.monitor_resources, 60)
            ]

            # Create monitored tasks
            running_tasks = []
            for task_name, task_func, interval in task_configs:
                # Create a wrapper task that uses run_monitored_task
                task = asyncio.create_task(
                    self.run_monitored_task(task_name, task_func, interval)
                )
                task.set_name(task_name)
                running_tasks.append(task)
                self.logger.info(f"Started monitored task: {task_name}")

            # Start status monitoring
            status_task = asyncio.create_task(self.update_node_status())
            status_task.set_name("Status Monitor")
            running_tasks.append(status_task)

            self.logger.info("P2P node is now running with monitored tasks")
            
            # Main monitoring loop
            while self.is_running:
                try:
                    # Check task status
                    for task in running_tasks[:]:  # Use slice to avoid modification during iteration
                        if task.done():
                            if exception := task.exception():
                                self.logger.error(f"Task {task.get_name()} failed with error: {exception}")
                                # Find the original config for the failed task
                                task_config = next(
                                    (tc for tc in task_configs if tc[0] == task.get_name()),
                                    None
                                )
                                if task_config:
                                    # Restart the failed task
                                    self.logger.info(f"Restarting failed task: {task_config[0]}")
                                    new_task = asyncio.create_task(
                                        self.run_monitored_task(
                                            task_config[0],
                                            task_config[1],
                                            task_config[2]
                                        )
                                    )
                                    new_task.set_name(task_config[0])
                                    running_tasks.remove(task)
                                    running_tasks.append(new_task)
                                    self.logger.info(f"Restarted task: {task_config[0]}")

                    # Log current task metrics
                    if hasattr(self, 'task_metrics'):
                        self.logger.info("\n=== Task Metrics ===")
                        for task_name, metrics in self.task_metrics.items():
                            avg_time = metrics['total_time'] / metrics['count'] if metrics['count'] > 0 else 0
                            self.logger.info(f"Task: {task_name}")
                            self.logger.info(f"  Count: {metrics['count']}")
                            self.logger.info(f"  Avg Time: {avg_time:.3f}s")
                            self.logger.info(f"  Max Time: {metrics['max_time']:.3f}s")
                            self.logger.info(f"  Min Time: {metrics['min_time']:.3f}s")

                    await asyncio.sleep(60)  # Check every minute

                except Exception as e:
                    self.logger.error(f"Error in task monitoring loop: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    await asyncio.sleep(5)  # Brief pause before continuing

        except Exception as e:
            self.logger.error(f"Fatal error in P2P node run method: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            # Cleanup
            self.logger.info("Shutting down P2P node...")
            self.is_running = False
            
            # Cancel all tasks
            for task in running_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if running_tasks:
                self.logger.info("Waiting for tasks to complete...")
                await asyncio.gather(*running_tasks, return_exceptions=True)
            
            self.logger.info("P2P node shutdown complete")

    async def update_node_status(self):
        """Periodic node status updates."""
        while self.is_running:
            try:
                status = {
                    'connected_peers': len(self.connected_peers),
                    'active_tasks': len([t for t in asyncio.all_tasks() if not t.done()]),
                    'blockchain_height': len(self.blockchain.chain) if self.blockchain else 0,
                    'mempool_size': len(self.mempool),
                    'sync_status': {k: v.is_syncing for k, v in self.sync_states.items()}
                }
                self.logger.info(f"\n=== Node Status ===\n{json.dumps(status, indent=2)}\n==================")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error updating node status: {str(e)}")
                await asyncio.sleep(5)
    async def run_monitored_task(self, name: str, task_func, interval: int):
        """Run a task with monitoring and error handling."""
        while self.is_running:
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(task_func):
                    await task_func()
                else:
                    await asyncio.to_thread(task_func)
                
                # Calculate task metrics
                execution_time = time.time() - start_time
                self.update_task_metrics(name, execution_time)
                
                # Wait for next interval, accounting for execution time
                wait_time = max(0, interval - execution_time)
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Error in task {name}: {str(e)}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # Brief pause before retry

    async def monitor_resources(self):
        """Monitor system resources used by the node."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        
        self.logger.info(f"Resource usage - CPU: {cpu_percent}%, Memory: {memory_info.rss / 1024 / 1024:.1f}MB")

    async def monitor_sync_status(self):
        """Monitor synchronization status and state across the network."""
        while self.is_running:  # Make sure we only run while the node is active
            try:
                # Log sync states for each component
                self.logger.info("\n=== Synchronization Status ===")
                for component_name, sync_state in self.sync_states.items():
                    self.logger.info(f"Component: {component_name}")
                    self.logger.info(f"  Is Syncing: {sync_state.is_syncing}")
                    self.logger.info(f"  Last Sync: {time.time() - sync_state.last_sync:.2f}s ago")
                    self.logger.info(f"  Current Hash: {sync_state.current_hash}")
                    self.logger.info(f"  Sync Progress: {sync_state.sync_progress}%")
                    self.logger.info(f"  Last Validated: {time.time() - sync_state.last_validated:.2f}s ago")

                # Log network state
                mempool_size = len(self.blockchain.mempool) if self.blockchain else 0
                chain_height = len(self.blockchain.chain) if self.blockchain else 0
                self.logger.info("\n=== Network State ===")
                self.logger.info(f"Connected Peers: {len(self.connected_peers)}")
                self.logger.info(f"Active Peers: {[peer for peer in self.connected_peers]}")
                self.logger.info(f"Chain Height: {chain_height}")
                self.logger.info(f"Mempool Size: {mempool_size}")

                # Check for out-of-sync components
                for component_name, sync_state in self.sync_states.items():
                    if time.time() - sync_state.last_sync > 300:  # 5 minutes
                        self.logger.warning(f"Component {component_name} hasn't been synced in 5 minutes")
                        # Trigger sync with a random peer if available
                        if self.connected_peers:
                            random_peer = random.choice(list(self.connected_peers))
                            self.logger.info(f"Initiating sync of {component_name} with {random_peer}")
                            await self.start_sync(random_peer, component_name)

                self.logger.info("================================\n")
                await asyncio.sleep(10)  # Wait 10 seconds before next check

            except Exception as e:
                self.logger.error(f"Error in sync status monitoring: {str(e)}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(10)  # Wait before retrying

    async def update_node_status(self):
        """Update and log overall node status."""
        status = {
            'connected_peers': len(self.connected_peers),
            'active_tasks': len([t for t in asyncio.all_tasks() if not t.done()]),
            'blockchain_height': len(self.blockchain.chain) if self.blockchain else 0,
            'mempool_size': len(self.mempool),
            'sync_status': {k: v.is_syncing for k, v in self.sync_states.items()}
        }
        self.logger.info(f"Node status: {status}")

    def update_task_metrics(self, task_name: str, execution_time: float):
        """Update metrics for task execution."""
        if not hasattr(self, 'task_metrics'):
            self.task_metrics = {}
        
        if task_name not in self.task_metrics:
            self.task_metrics[task_name] = {
                'count': 0,
                'total_time': 0,
                'max_time': 0,
                'min_time': float('inf')
            }
        
        metrics = self.task_metrics[task_name]
        metrics['count'] += 1
        metrics['total_time'] += execution_time
        metrics['max_time'] = max(metrics['max_time'], execution_time)
        metrics['min_time'] = min(metrics['min_time'], execution_time)

    async def shutdown(self, sig=None):
        """Gracefully shutdown the node."""
        if sig:
            self.logger.info(f"Received exit signal {sig.name}...")
        
        self.is_running = False
        
        # Cancel all tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        
        # Close connections
        for peer in list(self.peers.keys()):
            await self.remove_peer(peer)
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.logger.info("Node shutdown complete")

    async def cleanup_expired_states(self):
        """Clean up expired states and resources."""
        # Clean up seen messages
        current_time = time.time()
        self.seen_messages = {
            msg_hash: timestamp 
            for msg_hash, timestamp in self.seen_messages.items()
            if current_time - timestamp < self.message_expiry
        }
        
        # Clean up other expired states
        await self.cleanup_challenges()
    async def periodic_logo_sync(self):
        while True:
            try:
                await self.sync_logos()
                await asyncio.sleep(5)  # Sync logos every hour
            except Exception as e:
                logger.error(f"Failed during periodic logo sync: {str(e)}")
                logger.error(traceback.format_exc())

    async def process_message_queue(self):
        while True:
            try:
                message = await self.message_queue.get()
                await self.process_message(message, message.sender)
            except Exception as e:
                logger.error(f"Failed to process message queue: {str(e)}")
                logger.error(traceback.format_exc())

    async def periodic_peer_discovery(self):
        while True:
            try:
                await self.find_node(self.generate_random_id())
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Failed during periodic peer discovery: {str(e)}")
                logger.error(traceback.format_exc())

    async def periodic_data_republish(self):
        while True:
            try:
                for key, value in self.data_store.items():
                    await self.store(key, value)
                await asyncio.sleep(3600)  # Every hour
            except Exception as e:
                logger.error(f"Failed during periodic data republish: {str(e)}")
                logger.error(traceback.format_exc())

    async def send_heartbeats(self):
        """Send both regular and quantum heartbeats to all peers"""
        while self.is_running:
            try:
                peers = list(self.connected_peers)
                self.logger.info(f"\n[HEARTBEAT] {'='*20} Heartbeat Round {'='*20}")
                self.logger.info(f"[HEARTBEAT] Sending heartbeats to {len(peers)} peers")

                for peer in peers:
                    try:
                        # First send regular heartbeat
                        await self.send_message(  # Removed extra peer argument
                            peer, 
                            Message(
                                type=MessageType.HEARTBEAT.value,
                                payload={'timestamp': time.time()}
                            )
                        )
                        self.logger.debug(f"[HEARTBEAT] Regular heartbeat sent to {peer}")

                        # Then send quantum heartbeat if peer is quantum-entangled
                        if (hasattr(self, 'quantum_sync') and 
                            self.quantum_sync and 
                            peer in self.quantum_sync.entangled_peers):
                            
                            try:
                                self.logger.info(f"[QUANTUM] Sending quantum heartbeat to {peer}")
                                # Get current quantum states and fidelities with error handling
                                quantum_states = {}
                                fidelities = {}
                                
                                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                                    try:
                                        qubit = getattr(self.quantum_sync.register, component)
                                        if qubit is None:
                                            continue
                                            
                                        quantum_states[component] = qubit.value
                                        fidelity = await self.quantum_sync.measure_sync_state(component)
                                        fidelities[component] = fidelity
                                        self.logger.debug(f"[QUANTUM] {component} fidelity: {fidelity:.3f}")
                                    except Exception as e:
                                        self.logger.error(f"[QUANTUM] Error getting state for {component}: {str(e)}")
                                        continue

                                # Skip quantum heartbeat if no states were collected
                                if not quantum_states:
                                    self.logger.warning(f"[QUANTUM] No valid quantum states for peer {peer}")
                                    continue

                                # Get and validate Bell pair
                                bell_pair = self.quantum_sync.bell_pairs.get(peer)
                                if not bell_pair:
                                    self.logger.warning(f"[QUANTUM] No Bell pair found for peer {peer}")
                                    continue

                                # Get Bell pair ID
                                try:
                                    bell_pair_id = self.quantum_notifier._get_bell_pair_id(bell_pair)
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error getting Bell pair ID: {str(e)}")
                                    continue

                                # Generate secure nonce
                                try:
                                    nonce = os.urandom(16).hex()
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error generating nonce: {str(e)}")
                                    continue
                                    
                                # Create quantum heartbeat
                                heartbeat = QuantumHeartbeat(
                                    node_id=self.node_id,
                                    timestamp=time.time(),
                                    quantum_states=quantum_states,
                                    fidelities=fidelities,
                                    bell_pair_id=bell_pair_id,
                                    nonce=nonce
                                )

                                # Sign heartbeat with error handling
                                try:
                                    message_str = (
                                        f"{heartbeat.node_id}:{heartbeat.timestamp}:"
                                        f"{heartbeat.bell_pair_id}:{heartbeat.nonce}"
                                    ).encode()
                                    
                                    signature = self.private_key.sign(
                                        message_str,
                                        padding.PSS(
                                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                            salt_length=padding.PSS.MAX_LENGTH
                                        ),
                                        hashes.SHA256()
                                    )
                                    heartbeat.signature = base64.b64encode(signature).decode()
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error signing heartbeat: {str(e)}")
                                    continue

                                # Send quantum heartbeat with retry
                                try:
                                    await self.send_message(
                                        peer,
                                        Message(
                                            type=MessageType.QUANTUM_HEARTBEAT.value,
                                            payload=heartbeat.to_dict()
                                        )
                                    )
                                    self.logger.info(f"[QUANTUM] ✓ Quantum heartbeat sent to {peer}")
                                    self.logger.debug(f"[QUANTUM] Current fidelities: {fidelities}")
                                    
                                    # Update tracking only on successful send
                                    self.last_quantum_heartbeat[peer] = time.time()
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error sending quantum heartbeat: {str(e)}")
                                    continue

                            except Exception as qe:
                                self.logger.error(f"[QUANTUM] Error in quantum heartbeat for {peer}: {str(qe)}")
                                # Continue with next peer without removing - quantum errors shouldn't break connection
                                continue

                    except Exception as e:
                        self.logger.error(f"Error sending heartbeats to {peer}: {str(e)}")
                        # Only remove peer for non-quantum errors
                        await self.remove_peer(peer)
                        continue

                    # Brief pause between peers to avoid flooding
                    await asyncio.sleep(0.1)

                self.logger.info(f"[HEARTBEAT] {'='*50}\n")
                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)


    async def send_heartbeat(self, peer: str):
        heartbeat_message = json.dumps({"type": "heartbeat", "timestamp": time.time()})
        websocket = self.peers.get(peer)
        if websocket:
            await websocket.send(heartbeat_message)
            logger.debug(f"Sent heartbeat to {peer}")


    def generate_random_id(self) -> str:
        return format(random.getrandbits(160), '040x')

    async def request_node_state(self, peer_address: str) -> NodeState:
        try:
            message = Message(type=MessageType.STATE_REQUEST.value, payload={})
            response = await self.send_and_wait_for_response(peer_address, message)
            return NodeState(**response.payload)
        except Exception as e:
            logger.error(f"Failed to request node state from {peer_address}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def verify_network_consistency(self) -> Dict[str, bool]:
        try:
            local_state = self.blockchain.get_node_state()
            consistency_results = {}

            for peer_address in self.peers:
                try:
                    peer_state = await self.request_node_state(peer_address)
                    if peer_state:
                        is_consistent = self.compare_states(local_state, peer_state)
                        consistency_results[peer_address] = is_consistent
                except Exception as e:
                    logger.error(f"Failed to verify consistency with {peer_address}: {str(e)}")
                    consistency_results[peer_address] = False

            return consistency_results
        except Exception as e:
            logger.error(f"Failed to verify network consistency: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def compare_states(self, state1: NodeState, state2: NodeState) -> bool:
        try:
            return (
                state1.blockchain_length == state2.blockchain_length and
                state1.latest_block_hash == state2.latest_block_hash and
                state1.total_supply == state2.total_supply and
                state1.difficulty == state2.difficulty
            )
        except Exception as e:
            logger.error(f"Failed to compare states: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def generate_encryption_key(self) -> bytes:
        password = os.getenv('ENCRYPTION_PASSWORD', 'default_password').encode()
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    def encrypt_data(self, data: str) -> str:
        f = Fernet(self.encryption_key)
        return f.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()

    def calculate_data_hash(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    async def store(self, key: str, value: str, owner: str = None):
        try:
            encrypted_value = self.encrypt_data(value)
            data_hash = self.calculate_data_hash(value)
            nodes = await self.find_node(key)
            store_operations = [self.send_store(node, key, encrypted_value, data_hash, owner) for node in nodes]
            await asyncio.gather(*store_operations)
            if owner:
                self.access_control[key] = {owner}
        except Exception as e:
            logger.error(f"Failed to store value: {str(e)}")
            logger.error(traceback.format_exc())

    async def get(self, key: str, requester: str = None) -> Optional[str]:
        try:
            if key in self.data_store:
                if requester and key in self.access_control and requester not in self.access_control[key]:
                    logger.warning(f"Unauthorized access attempt by {requester} for key {key}")
                    return None
                encrypted_value, stored_hash = self.data_store[key]
                decrypted_value = self.decrypt_data(encrypted_value)
                if self.calculate_data_hash(decrypted_value) != stored_hash:
                    logger.error(f"Data integrity check failed for key {key}")
                    return None
                return decrypted_value
            nodes = await self.find_node(key)
            for node in nodes:
                value = await self.send_find_value(node, key, requester)
                if value:
                    await self.store(key, value, requester)
                    return value
            return None
        except Exception as e:
            logger.error(f"Failed to get value: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def handle_store(self, data: dict, sender: str):
        try:
            key, encrypted_value, data_hash, owner = data['key'], data['value'], data['hash'], data.get('owner')
            self.data_store[key] = (encrypted_value, data_hash)
            if owner:
                if key not in self.access_control:
                    self.access_control[key] = set()
                self.access_control[key].add(owner)
            if len(self.data_store) > self.max_data_store_size:
                self.data_store.popitem(last=False)
        except Exception as e:
            logger.error(f"Failed to handle store: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_find_value(self, data: dict, sender: str):
        try:
            key, requester = data['key'], data.get('requester')
            if key in self.data_store:
                if requester and key in self.access_control and requester not in self.access_control[key]:
                    logger.warning(f"Unauthorized access attempt by {requester} for key {key}")
                    await self.send_message(sender, Message(MessageType.FIND_VALUE.value, {"value": None}))
                else:
                    encrypted_value, _ = self.data_store[key]
                    await self.send_message(sender, Message(MessageType.FIND_VALUE.value, {"value": encrypted_value}))
            else:
                await self.handle_find_node({"node_id": key}, sender)
        except Exception as e:
            logger.error(f"Failed to handle find value: {str(e)}")
            logger.error(traceback.format_exc())

    async def send_store(self, node: KademliaNode, key: str, encrypted_value: str, data_hash: str, owner: str = None):
        try:
            await self.send_message(node.address, Message(MessageType.STORE.value, {
                "key": key, 
                "value": encrypted_value, 
                "hash": data_hash,
                "owner": owner
            }))
        except Exception as e:
            logger.error(f"Failed to send store: {str(e)}")
            logger.error(traceback.format_exc())

    async def send_find_value(self, node: KademliaNode, key: str, requester: str = None) -> Optional[str]:
        try:
            response = await self.send_and_wait_for_response(node.address, Message(MessageType.FIND_VALUE.value, {
                "key": key,
                "requester": requester
            }))
            return response.payload.get('value')
        except Exception as e:
            logger.error(f"Failed to send find value: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def store_proof(self, transaction_hash: str, proof: Tuple[Tuple, Tuple]):
        proof_hash = self.hash_proof(proof)
        await self.store(f"proof:{transaction_hash}", proof_hash)

    async def verify_stored_proof(self, transaction_hash: str, public_input: int, proof: Tuple[Tuple, Tuple]) -> bool:
        stored_proof_hash = await self.get(f"proof:{transaction_hash}")
        if stored_proof_hash != self.hash_proof(proof):
            return False
        return self.zkp_system.verify(public_input, proof)

    def hash_proof(self, proof: Tuple[Tuple, Tuple]) -> str:
        return hashlib.sha256(json.dumps(proof).encode()).hexdigest()

    async def generate_proof(self, secret, public_input):
        self.logger.debug(f"Starting ZKP generation with secret: {secret}, public input: {public_input}")
        start_time = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                proof = await asyncio.get_event_loop().run_in_executor(
                    executor, self.zk_system.prove, secret, public_input
                )
            elapsed_time = time.time() - start_time
            self.logger.debug(f"ZKP generation completed in {elapsed_time:.2f} seconds")
            return proof
        except Exception as e:
            self.logger.error(f"Error during ZKP generation: {str(e)}")
            raise


        except Exception as e:
            # Log any errors encountered during the process
            self.logger.error(f"Error during ZKP generation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    async def verify_proof(self, public_input: int, proof: Tuple[Tuple, Tuple]) -> bool:
        public_input_elem = self.zk_system.field.element(public_input)
        return await asyncio.to_thread(self.zk_system.verify, public_input_elem, proof)

    def deserialize_proof(self, serialized_proof):
        if isinstance(serialized_proof, list) and len(serialized_proof) == 2:
            return tuple(self.deserialize_proof(p) for p in serialized_proof)
        elif isinstance(serialized_proof, list):
            return [self.deserialize_proof(p) for p in serialized_proof]
        elif isinstance(serialized_proof, int):
            return self.zk_system.field.element(serialized_proof)
        else:
            return serialized_proof
    async def authenticate_peer(self, websocket):
        try:
            # Generate a random challenge
            challenge = os.urandom(32).hex()
            logger.debug(f"Generated challenge: {challenge}")

            # Send the challenge to the peer
            await websocket.send(json.dumps({"type": "challenge", "data": challenge}))
            logger.debug(f"Sent challenge to peer: {websocket.remote_address}")

            # Wait for the peer's response
            response = await websocket.recv()
            response_data = json.loads(response)
            logger.debug(f"Received response from peer: {response_data}")

            # Check if public key exchange comes first and handle it
            if response_data["type"] == "public_key_exchange":
                await self.handle_public_key_exchange(Message.from_json(response), websocket.remote_address[0])

                # Wait for the challenge response after key exchange
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.debug(f"Received challenge response from peer: {response_data}")

            # Ensure the response type is correct
            if response_data["type"] != "challenge_response":
                logger.warning(f"Invalid response type from peer: {response_data['type']}")
                return False

            # Extract public key and proof from the response
            public_key_str = response_data.get("public_key")
            proof = response_data.get("proof")

            if not public_key_str or not proof:
                logger.warning("Missing public key or proof in response")
                return False

            # Convert challenge to integer (as public input for ZKP system)
            public_input = int.from_bytes(bytes.fromhex(challenge), 'big')

            # Verify the zero-knowledge proof
            is_valid = await self.verify_proof(public_input, proof)

            if is_valid:
                logger.info(f"Successfully authenticated peer: {websocket.remote_address}")

                # Store the peer's public key
                try:
                    public_key = serialization.load_pem_public_key(
                        public_key_str.encode(),
                        backend=default_backend()
                    )
                    self.peer_public_keys[websocket.remote_address[0]] = public_key
                    logger.debug(f"Stored public key for peer: {websocket.remote_address}")
                except Exception as key_error:
                    logger.error(f"Failed to process peer's public key: {str(key_error)}")
                    return False

                return True
            else:
                logger.warning(f"Failed to verify proof from peer: {websocket.remote_address}")
                return False

        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to decode JSON from peer: {str(json_error)}")
        except websockets.exceptions.ConnectionClosed as conn_error:
            logger.error(f"WebSocket connection closed during authentication: {str(conn_error)}")
        except Exception as e:
            logger.error(f"Unexpected error during peer authentication: {str(e)}")
            logger.error(traceback.format_exc())
        
        return False
    def verify_key_pair_integrity(self):
        try:
            test_message = b"Test message for key pair verification"
            signature = self.private_key.sign(
                test_message,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            self.public_key.verify(
                signature,
                test_message,
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            logger.info("Public/private key pair integrity verified successfully")
            return True
        except Exception as e:
            logger.error(f"Public/private key pair integrity verification failed: {str(e)}")
            return False
    async def handle_challenge(self, peer: str, challenge: str, challenge_id: str) -> bool:
        """Handle incoming challenge with proper state management."""
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.info(f"[CHALLENGE] Processing challenge from {peer_normalized}")
            self.logger.debug(f"[CHALLENGE] ID: {challenge_id}")
            self.logger.debug(f"[CHALLENGE] Data: {challenge}")
            self.logger.debug(f"[CHALLENGE] Role: CLIENT")

            # Store challenge
            stored = await self.challenge_manager.store_challenge(
                peer_normalized,
                challenge_id, 
                challenge,
                ChallengeRole.CLIENT
            )
            
            if not stored:
                self.logger.error(f"Failed to store challenge for {peer_normalized}")
                return False

            # Generate signature
            signature = self.private_key.sign(
                challenge.encode(),
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            # Create and send response
            response = Message(
                type=MessageType.CHALLENGE_RESPONSE.value,
                payload={
                    'signature': base64.b64encode(signature).decode('utf-8')
                },
                challenge_id=challenge_id,
                sender=self.node_id
            )

            await self.send_raw_message(peer_normalized, response)
            self.logger.debug(f"[CHALLENGE] Response sent with signature: {base64.b64encode(signature).decode('utf-8')[:64]}...")

            # Update peer state to indicate challenge response sent
            async with self.peer_lock:
                self.peer_states[peer_normalized] = "challenge_response_sent"
                self.logger.debug(f"[CHALLENGE] Updated peer state to: {self.peer_states[peer_normalized]}")

            return True

        except Exception as e:
            self.logger.error(f"Error handling challenge: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False





    async def initiate_reconnection(self, peer: str):
        """Attempt to reconnect to a peer with proper cleanup first."""
        try:
            self.logger.info(f"[RECONNECT] Initiating reconnection to {peer}")
            
            # Ensure proper cleanup before reconnecting
            await self.remove_peer(peer)
            
            # Get original peer info
            peer_ip, peer_port = peer.split(':')
            
            # Create new node object
            node = KademliaNode(
                id=self.generate_node_id(),
                ip=peer_ip,
                port=int(peer_port),
                magnet_link=None
            )
            
            # Attempt reconnection with backoff
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Wait before retry with exponential backoff
                    if attempt > 0:
                        wait_time = 2 ** attempt
                        self.logger.info(f"[RECONNECT] Waiting {wait_time}s before attempt {attempt + 1}")
                        await asyncio.sleep(wait_time)
                    
                    # Attempt connection
                    if await self.connect_to_peer(node):
                        self.logger.info(f"[RECONNECT] ✓ Successfully reconnected to {peer}")
                        return True
                        
                except Exception as e:
                    self.logger.error(f"[RECONNECT] Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_attempts - 1:
                        raise
                    
            return False
            
        except Exception as e:
            self.logger.error(f"[RECONNECT] Failed to reconnect to {peer}: {str(e)}")
            return False
    def clear_peer_public_key(self, peer: str):
        normalized_peer = self.normalize_peer_address(peer)
        if normalized_peer in self.peer_public_keys:
            del self.peer_public_keys[normalized_peer]
            logger.debug(f"Cleared public key for peer {normalized_peer}")
        else:
            logger.debug(f"No public key found to clear for peer {normalized_peer}")
    def dump_peer_public_keys(self):
        logger.debug("Current state of peer_public_keys:")
        for peer, key in self.peer_public_keys.items():
            logger.debug(f"  {peer}: {type(key)}")
                    
        def verify_key_pair_integrity(self):
            try:
                test_message = b"Test message for key pair verification"
                signature = self.private_key.sign(
                    test_message,
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                self.public_key.verify(
                    signature,
                    test_message,
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                self.logger.info("Public/private key pair integrity verified successfully")
                return True
            except Exception as e:
                self.logger.error(f"Public/private key pair integrity verification failed: {str(e)}")
                return False
    async def verify_challenge_response(self, peer: str, incoming_challenge_id: str, response_payload: dict) -> bool:
        """Verify challenge response with guaranteed challenge matching."""
        try:
            # Initial logging
            self.logger.info(f"\n[VERIFY] {'='*20} Verifying Response {'='*20}")
            
            # Normalize addresses with proper port handling
            peer_ip, peer_port = peer.split(':')
            client_port = '50510'  # Original client port
            client_address = f"{peer_ip}:{client_port}"
            
            self.logger.debug(f"[VERIFY] Client address: {client_address}")
            self.logger.debug(f"[VERIFY] Original challenge ID: {incoming_challenge_id}")

            # Get latest challenge stored for the client address
            client_challenges = await self.challenge_manager.get_active_challenges(client_address)
            if not client_challenges:
                self.logger.error(f"[VERIFY] No challenges found for client {client_address}")
                return False
                
            # Get most recent challenge
            recent_challenge = None
            recent_id = None
            
            # First try to find the incoming challenge ID
            if incoming_challenge_id in client_challenges:
                recent_challenge = client_challenges[incoming_challenge_id]
                recent_id = incoming_challenge_id
                self.logger.debug(f"[VERIFY] Found exact matching challenge: {incoming_challenge_id}")
            else:
                # Get most recent challenge by timestamp
                sorted_challenges = sorted(
                    client_challenges.items(),
                    key=lambda x: x[1].timestamp if hasattr(x[1], 'timestamp') else 0,
                    reverse=True
                )
                if sorted_challenges:
                    recent_id, recent_challenge = sorted_challenges[0]
                    self.logger.debug(f"[VERIFY] Using most recent challenge: {recent_id}")

            if not recent_challenge:
                self.logger.error("[VERIFY] No valid challenge found")
                self.logger.debug(f"[VERIFY] Available challenges: {list(client_challenges.keys())}")
                return False

            # Log challenge details
            self.logger.debug(f"[VERIFY] Challenge roles:")
            self.logger.debug(f"  - Original role: {recent_challenge.role}")
            self.logger.debug(f"  - Challenge data: {recent_challenge.challenge}")

            # Get public key from client address
            peer_public_key = self.peer_public_keys.get(client_address)
            if not peer_public_key:
                self.logger.error(f"[VERIFY] No public key found for {client_address}")
                return False

            try:
                # Verify signature
                signature = base64.b64decode(response_payload['signature'])
                challenge_bytes = recent_challenge.challenge.encode()
                
                self.logger.debug(f"[VERIFY] Verification details:")
                self.logger.debug(f"  - Signature length: {len(signature)}")
                self.logger.debug(f"  - Challenge bytes length: {len(challenge_bytes)}")
                self.logger.debug(f"  - First 32 bytes of challenge: {recent_challenge.challenge[:32]}")

                peer_public_key.verify(
                    signature,
                    challenge_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )

                # Update states only after successful verification
                async with self.peer_lock:
                    # Update primary client address
                    self.peer_states[client_address] = "verified"
                    self.connected_peers.add(client_address)
                    
                    # Map current peer address to client address if different
                    if peer != client_address:
                        self.peer_address_map[peer] = client_address
                        self.peer_states[peer] = "verified"
                        self.connected_peers.add(peer)

                # Clean up the verified challenge
                await self.challenge_manager.remove_challenge(client_address, recent_id)
                
                self.logger.info(f"[VERIFY] ✓ Challenge verification successful")
                self.logger.info(f"[VERIFY] {'='*50}\n")
                return True

            except InvalidSignature:
                self.logger.error(f"[VERIFY] Invalid signature for challenge {recent_id}")
                self.logger.debug(f"[VERIFY] Challenge being verified: {recent_challenge.challenge}")
                return False

            except Exception as e:
                self.logger.error(f"[VERIFY] Verification error: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False

        except Exception as e:
            self.logger.error(f"[VERIFY] Fatal error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    def get_normalized_peer_address(self, peer: str) -> str:
        """Get normalized peer address from mapping."""
        try:
            # Return normalized address if it exists in mapping
            if hasattr(self, 'peer_address_map'):
                if peer in self.peer_address_map:
                    return self.peer_address_map[peer]
                    
            # Otherwise normalize the address directly
            return self.normalize_peer_address(peer)
            
        except Exception as e:
            self.logger.error(f"Error normalizing peer address: {str(e)}")
            return peer


    async def monitor_challenge_timeout(self, peer: str, challenge_id: str):
        """Monitor challenge response timeout."""
        try:
            await asyncio.sleep(30)  # 30 second timeout
            challenge = await self.challenge_manager.get_challenge(peer, challenge_id)
            if challenge and not challenge.verified:
                self.logger.warning(f"[CHALLENGE] Challenge timeout for {peer}")
                await self.challenge_manager.remove_challenge(peer, challenge_id)
                await self.remove_peer(peer)
        except Exception as e:
            self.logger.error(f"[CHALLENGE] Error monitoring timeout: {str(e)}")



    async def create_challenge(self) -> Tuple[str, str]:
        """Create a new challenge with base64 encoding."""
        challenge_id = str(uuid.uuid4())
        challenge_bytes = os.urandom(32)
        challenge = base64.b64encode(challenge_bytes).decode('utf-8')
        return challenge_id, challenge
    async def handle_challenge_response(self, peer: str, message: Message):
        """Handle challenge response with port mapping and enhanced verification."""
        try:
            # Get all possible peer addresses (including mapped ports)
            peer_ip, peer_port = peer.split(':')
            possible_peers = {peer}  # Start with original peer address
            
            # Add mapped addresses from address map
            normalized_peer = self.normalize_peer_address(peer)
            for orig, mapped in self.peer_address_map.items():
                if mapped == normalized_peer or orig == normalized_peer:
                    possible_peers.add(orig)
                    possible_peers.add(mapped)
            
            self.logger.info(f"\n[CHALLENGE] {'='*20} Processing Response {'='*20}")
            self.logger.info(f"[CHALLENGE] Peer: {normalized_peer}")
            self.logger.debug(f"[CHALLENGE] Possible addresses: {possible_peers}")
            self.logger.debug(f"[CHALLENGE] Response payload: {message.payload}")

            # Extract and validate challenge ID
            challenge_id = message.challenge_id
            if not challenge_id:
                raise ValueError(f"Missing challenge ID in response from {normalized_peer}")

            # Find challenge under any possible peer address
            challenge = None
            challenge_peer = None
            for possible_peer in possible_peers:
                if possible_peer in self.pending_challenges:
                    if challenge_id in self.pending_challenges[possible_peer]:
                        challenge = self.pending_challenges[possible_peer][challenge_id]
                        challenge_peer = possible_peer
                        self.logger.debug(f"[CHALLENGE] Found challenge under address: {possible_peer}")
                        break

            if not challenge:
                self.logger.warning(f"[CHALLENGE] No challenge found for any address")
                self.logger.debug(f"[CHALLENGE] Available peers: {list(self.pending_challenges.keys())}")
                self.logger.debug(f"[CHALLENGE] Challenge ID: {challenge_id}")
                return False

            # Validate signature format
            signature_b64 = message.payload.get('signature')
            if not signature_b64:
                self.logger.error("[CHALLENGE] Missing signature in response")
                return False

            try:
                signature = base64.b64decode(signature_b64)
                self.logger.debug(f"[CHALLENGE] Decoded signature length: {len(signature)}")
            except Exception as decode_error:
                self.logger.error(f"[CHALLENGE] Failed to decode signature: {str(decode_error)}")
                return False

            # Get public key from any possible peer address
            peer_public_key = None
            for possible_peer in possible_peers:
                if possible_peer in self.peer_public_keys:
                    peer_public_key = self.peer_public_keys[possible_peer]
                    self.logger.debug(f"[CHALLENGE] Found public key under address: {possible_peer}")
                    break

            if not peer_public_key:
                self.logger.error("[CHALLENGE] No public key found under any address")
                return False

            # Convert PEM string to RSAPublicKey if needed
            if isinstance(peer_public_key, str):
                try:
                    peer_public_key = serialization.load_pem_public_key(
                        peer_public_key.encode(),
                        backend=default_backend()
                    )
                    self.logger.debug("[CHALLENGE] Converted PEM string to RSAPublicKey")
                except Exception as key_error:
                    self.logger.error(f"[CHALLENGE] Failed to load public key: {str(key_error)}")
                    return False

            # Verify signature with detailed logging
            try:
                self.logger.debug("[CHALLENGE] Attempting signature verification")
                self.logger.debug(f"[CHALLENGE] Challenge length: {len(challenge)}")
                self.logger.debug(f"[CHALLENGE] Signature length: {len(signature)}")
                self.logger.debug(f"[CHALLENGE] Challenge data: {challenge[:32]}...")

                peer_public_key.verify(
                    signature,
                    challenge.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                self.logger.info(f"[CHALLENGE] ✓ Signature verified")

                # Update state for all related peer addresses
                async with self.peer_lock:
                    for possible_peer in possible_peers:
                        self.peer_states[possible_peer] = "verified"
                        if possible_peer not in self.connected_peers:
                            self.connected_peers.add(possible_peer)
                        self.logger.debug(f"[CHALLENGE] Updated state for {possible_peer}")

                # Clean up challenge data
                if challenge_peer in self.pending_challenges:
                    self.pending_challenges[challenge_peer].pop(challenge_id)
                    if not self.pending_challenges[challenge_peer]:
                        del self.pending_challenges[challenge_peer]
                self.logger.debug("[CHALLENGE] Challenge data cleaned up")

                # Complete the connection
                await self.complete_connection(normalized_peer)
                
                self.logger.info(f"[CHALLENGE] {'='*50}\n")
                return True

            except InvalidSignature as sig_error:
                self.logger.error(f"[CHALLENGE] ✗ Invalid signature: {str(sig_error)}")
                return False
                
            except Exception as verify_error:
                self.logger.error(f"[CHALLENGE] Verification error: {str(verify_error)}")
                self.logger.error(traceback.format_exc())
                return False

        except Exception as e:
            self.logger.error(f"[CHALLENGE] Fatal error handling response: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    def cleanup_challenges(self):
        """Safely cleanup old challenges"""
        try:
            current_time = time.time()
            for peer_id in list(self.challenges.keys()):
                peer_challenges = self.challenges[peer_id]
                for ch_id in list(peer_challenges.keys()):
                    challenge_data = peer_challenges[ch_id]
                    if isinstance(challenge_data, dict) and current_time - challenge_data.get('timestamp', 0) > 300:
                        del peer_challenges[ch_id]
                if not peer_challenges:
                    del self.challenges[peer_id]
        except Exception as e:
            self.logger.error(f"Error cleaning up challenges: {str(e)}")

    def serialize_proof(self, proof):
        if isinstance(proof, tuple) and len(proof) == 2:
            return [self.serialize_proof(p) for p in proof]
        elif isinstance(proof, list):
            return [self.serialize_proof(p) for p in proof]
        elif hasattr(proof, 'to_int'):
            return proof.to_int()
        else:
            return proof



    async def store_with_proof(self, key: str, value: str):
        proof = await self.generate_proof(int(hashlib.sha256(value.encode()).hexdigest(), 16), len(value))
        await self.store(key, json.dumps({"value": value, "proof": proof}))

    async def get_with_proof(self, key: str) -> Optional[str]:
        data = await self.get(key)
        if data:
            data = json.loads(data)
            if await self.verify_proof(len(data["value"]), data["proof"]):
                return data["value"]
        return None
    async def handle_new_block(self, block_data: dict, sender: str):
        try:
            new_block = QuantumBlock.from_dict(block_data)
            
            if self.blockchain.validate_block(new_block):
                # Add to blockchain
                success = await self.blockchain.add_block(new_block)
                if success:
                    # Immediately propagate to other peers
                    await self.broadcast(Message(
                        type=MessageType.BLOCK.value,
                        payload=block_data
                    ), exclude=sender)
                    
                    # Update sync state
                    self.sync_states[SyncComponent.BLOCKS].current_hash = await self.calculate_state_hash(SyncComponent.BLOCKS)
                    
        except Exception as e:
            self.logger.error(f"Error handling new block: {str(e)}")

    async def handle_transaction(self, transaction_data: dict, sender: str):
        try:
            if transaction_data.get('type') == 'multisig_transaction':
                # Handle multisig transaction
                tx_hash = await self.blockchain.add_multisig_transaction(
                    transaction_data['multisig_address'],
                    transaction_data['sender_public_keys'],
                    transaction_data['threshold'],
                    transaction_data['receiver'],
                    Decimal(transaction_data['amount']),
                    transaction_data['message'],
                    transaction_data['aggregate_proof']
                )
            else:
                # Handle regular transaction
                transaction = Transaction.from_dict(transaction_data)
                tx_hash = await self.blockchain.add_transaction(transaction)

            logger.info(f"Transaction added to blockchain: {tx_hash}")
            await self.broadcast(Message(MessageType.TRANSACTION.value, transaction_data), exclude=sender)
        except Exception as e:
            logger.error(f"Failed to handle transaction: {str(e)}")
            logger.error(traceback.format_exc())
def enhance_p2p_node(node: P2PNode) -> P2PNode:
    """Enhance P2P node with additional functionality"""
    try:
        logger.debug("Starting enhancement of P2P node.")

        # Add sync states
        node.sync_states = {
            SyncComponent.WALLETS: SyncStatus(),
            SyncComponent.TRANSACTIONS: SyncStatus(),
            SyncComponent.BLOCKS: SyncStatus(),
            SyncComponent.MEMPOOL: SyncStatus()
        }
        logger.debug("Added sync states to P2P node.")

        async def initialize_quantum_components(self):
            """Initialize quantum components without requiring a peer."""
            try:
                self.logger.info("[QUANTUM] Initializing quantum components...")
                
                if not hasattr(self, 'quantum_sync') or self.quantum_sync is None:
                    self.quantum_sync = QuantumEntangledSync(self.node_id)
                    logger.debug("Quantum sync initialized.")

                initial_data = {
                    'wallets': [w.to_dict() for w in self.blockchain.get_wallets()] if self.blockchain else [],
                    'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions(limit=100)] if self.blockchain else [],
                    'blocks': [block.to_dict() for block in self.blockchain.chain] if self.blockchain else [],
                    'mempool': [tx.to_dict() for tx in self.blockchain.mempool] if hasattr(self.blockchain, 'mempool') else []
                }

                await self.quantum_sync.initialize_quantum_state(initial_data)
                logger.debug("Quantum state initialized with blockchain data.")

                if not hasattr(self, 'quantum_notifier'):
                    self.quantum_notifier = QuantumStateNotifier(self)
                    logger.debug("Quantum notifier initialized.")
                
                if not hasattr(self, 'quantum_monitor'):
                    self.quantum_monitor = QuantumNetworkMonitor(self)
                    logger.debug("Quantum monitor initialized.")
                    
                if not hasattr(self, 'quantum_consensus'):
                    self.quantum_consensus = QuantumConsensusManager(self)
                    logger.debug("Quantum consensus manager initialized.")

                if not hasattr(self, 'quantum_heartbeats'):
                    self.quantum_heartbeats = {}
                if not hasattr(self, 'last_quantum_heartbeat'):
                    self.last_quantum_heartbeat = {}
                    
                self.quantum_initialized = True
                asyncio.create_task(self.monitor_quantum_state())
                
                self.logger.info("[QUANTUM] ✓ Quantum components initialized successfully")
                return True

            except Exception as e:
                self.logger.error(f"[QUANTUM] Error initializing quantum components: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise

        async def establish_quantum_entanglement(self, peer: str) -> bool:
            """Establish quantum entanglement with a peer."""
            try:
                self.logger.info(f"[QUANTUM] Establishing Entanglement with {peer}")
                
                if not self.quantum_initialized:
                    await self.initialize_quantum_components()

                current_state = {
                    'wallets': [w.to_dict() for w in self.blockchain.get_wallets()],
                    'transactions': [tx.to_dict() for tx in self.blockchain.get_recent_transactions()],
                    'blocks': [block.to_dict() for block in self.blockchain.chain],
                    'mempool': [tx.to_dict() for tx in self.blockchain.mempool]
                }

                request_message = Message(
                    type=MessageType.QUANTUM_ENTANGLEMENT_REQUEST.value,
                    payload={'node_id': self.node_id, 'state_data': current_state, 'timestamp': time.time()}
                )

                response = await self.send_and_wait_for_response(peer, request_message)

                if not response or response.type != MessageType.QUANTUM_ENTANGLEMENT_RESPONSE.value:
                    self.logger.error(f"[QUANTUM] Failed to receive entanglement response from {peer}")
                    return False

                peer_state = response.payload.get('state_data', {})
                peer_register = await self.quantum_sync.entangle_with_peer(peer, peer_state)
                
                if not peer_register:
                    self.logger.error(f"[QUANTUM] Failed to create quantum register with {peer}")
                    return False

                bell_pair = self.quantum_sync._generate_bell_pair()
                self.quantum_sync.bell_pairs[peer] = bell_pair

                asyncio.create_task(self.monitor_peer_quantum_state(peer))
                asyncio.create_task(self.send_quantum_heartbeats(peer))

                self.logger.info(f"[QUANTUM] ✓ Quantum entanglement established with {peer}")
                return True

            except Exception as e:
                self.logger.error(f"[QUANTUM] Error establishing entanglement: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False

        # Assigning enhanced methods to node
        node.initialize_quantum_components = types.MethodType(initialize_quantum_components, node)
        node.establish_quantum_entanglement = types.MethodType(establish_quantum_entanglement, node)
        logger.debug("Quantum methods assigned to P2P node.")

        # Add enhanced start
        original_start = node.start
        async def enhanced_start(self):
            await original_start()
            asyncio.create_task(node.process_sync_queue())
            node.logger.info("Enhanced sync system started.")
        
        node.start = types.MethodType(enhanced_start, node)
        logger.debug("Enhanced start method assigned.")

        # Return the fully enhanced node
        logger.debug("P2P node enhancement completed successfully.")
        return node

    except Exception as e:
        logger.error(f"Error enhancing P2P node: {str(e)}")
        logger.error(traceback.format_exc())
        return None




    # Define sync status handling method
    async def handle_sync_status(self, peer: str, data: dict):
        """Handle incoming sync status with improved error handling."""
        try:
            peer_sync_status = data.get('sync_status', {})
            peer_height = data.get('blockchain_height', 0)

            # Compare sync status for each component
            for component in [SyncComponent.WALLETS, SyncComponent.TRANSACTIONS, 
                             SyncComponent.BLOCKS, SyncComponent.MEMPOOL]:
                try:
                    our_hash = await self.calculate_state_hash(component)
                    peer_hash = peer_sync_status.get(str(component))

                    if peer_hash and peer_hash != our_hash:
                        if not self.sync_states[component].is_syncing:
                            self.logger.info(f"State mismatch detected for {component} with {peer}")
                            self.logger.debug(f"Our hash: {our_hash}")
                            self.logger.debug(f"Peer hash: {peer_hash}")
                            await self.start_sync(peer, component)
                except Exception as comp_error:
                    self.logger.error(f"Error checking sync for component {component}: {str(comp_error)}")

            # Handle blockchain height synchronization
            try:
                our_height = len(self.blockchain.chain) if self.blockchain else 0
                if peer_height > our_height:
                    if not self.sync_states[SyncComponent.BLOCKS].is_syncing:
                        self.logger.info(f"Chain height mismatch detected with {peer} "
                                       f"(our height: {our_height}, peer height: {peer_height})")
                        await self.start_sync(peer, SyncComponent.BLOCKS)
            except Exception as height_error:
                self.logger.error(f"Error handling blockchain height sync: {str(height_error)}")

        except Exception as e:
            self.logger.error(f"Error handling sync status from {peer}: {str(e)}")
            self.logger.debug(traceback.format_exc())
    async def start_sync(self, peer: str, component: SyncComponent):
        """Start synchronization process for a specific component."""
        try:
            if self.sync_states[component].is_syncing:
                self.logger.debug(f"Sync already in progress for {component}")
                return

            self.sync_states[component].is_syncing = True
            self.sync_states[component].last_sync = time.time()

            try:
                if component == SyncComponent.WALLETS:
                    await self.sync_wallets(peer)
                elif component == SyncComponent.TRANSACTIONS:
                    await self.sync_transactions(peer)
                elif component == SyncComponent.BLOCKS:
                    await self.sync_blocks(peer)
                elif component == SyncComponent.MEMPOOL:
                    await self.sync_mempool(peer)

                self.logger.info(f"Successfully completed sync of {component} with {peer}")
                
            except Exception as sync_error:
                self.logger.error(f"Error during sync of {component} with {peer}: {str(sync_error)}")
                self.logger.debug(traceback.format_exc())
                
            finally:
                self.sync_states[component].is_syncing = False
                self.sync_states[component].last_sync = time.time()

        except Exception as e:
            self.logger.error(f"Error in start_sync for {component} with {peer}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            self.sync_states[component].is_syncing = False

    async def enhanced_handle_message(self, message: Message, sender: str):
        """Enhanced message handler with sync and detailed logging."""
        try:
            # First handle sync-related messages
            if message.type == MessageType.SYNC_STATUS.value:
                self.logger.info(f"[SYNC] Received sync status from {sender}")
                await self.handle_sync_status(sender, message.payload)
                return

            # Handle different message types with detailed logging
            if message.type == MessageType.TRANSACTION.value:
                self.logger.info(f"[TRANSACTION] ===============================")
                self.logger.info(f"[TRANSACTION] Received from {sender}")
                if isinstance(message.payload, dict):
                    tx_type = message.payload.get('type', 'regular')
                    amount = message.payload.get('amount', 'unknown')
                    self.logger.info(f"[TRANSACTION] Type: {tx_type}")
                    self.logger.info(f"[TRANSACTION] Amount: {amount}")
                await self.handle_transaction(message.payload, sender)
                self.logger.info(f"[TRANSACTION] ===============================")

            elif message.type == MessageType.BLOCK.value:
                self.logger.info(f"[BLOCK] ===============================")
                self.logger.info(f"[BLOCK] Received from {sender}")
                await self.handle_block(message.payload, sender)
                self.logger.info(f"[BLOCK] ===============================")

            elif message.type == MessageType.GET_TRANSACTIONS.value:
                self.logger.info(f"[SYNC] Transactions requested by {sender}")
                await self.handle_get_transactions(sender)

            elif message.type == MessageType.GET_WALLETS.value:
                self.logger.info(f"[SYNC] Wallets requested by {sender}")
                await self.handle_get_wallets(sender)

            elif message.type == MessageType.GET_MEMPOOL.value:
                self.logger.info(f"[SYNC] Mempool requested by {sender}")
                await self.handle_get_mempool(sender)

            # Handle other message types through original handler
            else:
                await self.handle_message(message, sender)

            # After handling any message, check if sync is needed
            await self.check_sync_needs(sender)

        except Exception as e:
            self.logger.error(f"Error in enhanced message handler: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def enhanced_periodic_tasks(self):
        await original_periodic_tasks(self)
        while True:
            for component, state in self.sync_states.items():
                if time.time() - state.last_sync > 300:  # 5 minutes
                    self.logger.info(f"Component {component.value} hasn't been synced recently")
                    # Trigger sync with a random peer
                    if self.connected_peers:
                        peer = random.choice(list(self.connected_peers))
                        await self.start_sync(peer, component)
            await asyncio.sleep(60)


        return node
    
    
    async def broadcast_new_wallet(self, wallet_data: dict):
        """Broadcast new wallet creation to the network."""
        try:
            self.logger.info(f"[BROADCAST] Broadcasting new wallet: {wallet_data['address']}")
            message = Message(
                type=MessageType.NEW_WALLET.value,
                payload={
                    'wallet': wallet_data,
                    'timestamp': time.time()
                }
            )
            await self.broadcast(message)
            self.logger.debug(f"Successfully broadcast new wallet to {len(self.connected_peers)} peers")
        except Exception as e:
            self.logger.error(f"Error broadcasting new wallet: {str(e)}")

    async def broadcast_new_transaction(self, transaction: Transaction):
        """Broadcast new transaction to the network with enhanced logging."""
        try:
            self.logger.info(f"[BROADCAST] Broadcasting new transaction: {transaction.tx_hash}")
            message = Message(
                type=MessageType.TRANSACTION.value,
                payload=transaction.to_dict()
            )
            
            # Track transaction propagation
            self.transaction_tracker.add_transaction(transaction.tx_hash, self.node_id)
            
            # Broadcast to peers
            peers = list(self.connected_peers)
            self.logger.info(f"Broadcasting transaction to {len(peers)} peers")
            
            broadcast_tasks = []
            for peer in peers:
                task = asyncio.create_task(self.send_transaction_to_peer(peer, message))
                broadcast_tasks.append(task)
            
            # Wait for all broadcasts with timeout
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            successful = sum(1 for r in results if r and not isinstance(r, Exception))
            self.logger.info(f"Transaction broadcast completed: {successful}/{len(peers)} successful")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting transaction: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def send_transaction_to_peer(self, peer: str, message: Message) -> bool:
        """Send transaction to a specific peer with timeout and error handling."""
        try:
            await asyncio.wait_for(self.send_message(peer, message), timeout=5.0)
            self.logger.debug(f"Successfully sent transaction to peer {peer}")
            return True
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout sending transaction to peer {peer}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending transaction to peer {peer}: {str(e)}")
            return False

    async def broadcast_new_block(self, block: QuantumBlock):
        """Broadcast new mined block to the network with comprehensive logging."""
        try:
            self.logger.info(f"\n[BLOCK] ========= Broadcasting New Block =========")
            self.logger.info(f"[BLOCK] Block Hash: {block.hash}")
            self.logger.info(f"[BLOCK] Block Height: {len(self.blockchain.chain)}")
            self.logger.info(f"[BLOCK] Transaction Count: {len(block.transactions)}")
            
            message = Message(
                type=MessageType.BLOCK.value,
                payload=block.to_dict()
            )

            # Track active peers before broadcast
            active_peers = list(self.connected_peers)
            self.logger.info(f"[BLOCK] Broadcasting to {len(active_peers)} peers")

            # Broadcast to all peers concurrently
            broadcast_tasks = []
            for peer in active_peers:
                task = asyncio.create_task(self.send_block_to_peer(peer, message))
                broadcast_tasks.append(task)

            # Wait for all broadcasts with timeout
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            successful = sum(1 for r in results if r and not isinstance(r, Exception))

            self.logger.info(f"[BLOCK] Broadcast completed:")
            self.logger.info(f"[BLOCK] - Successful: {successful}")
            self.logger.info(f"[BLOCK] - Failed: {len(active_peers) - successful}")
            self.logger.info(f"[BLOCK] =======================================\n")

        except Exception as e:
            self.logger.error(f"Error broadcasting block: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def handle_new_wallet(self, data: dict):
        """Handle incoming new wallet with verification."""
        try:
            self.logger.info(f"[WALLET] Processing new wallet: {data['wallet']['address']}")
            
            # Verify wallet data
            wallet_data = data['wallet']
            if not self.verify_wallet_data(wallet_data):
                self.logger.warning(f"Invalid wallet data received")
                return

            # Add to blockchain
            await self.blockchain.add_wallet(wallet_data)
            
            # Update sync state
            self.sync_states[SyncComponent.WALLETS].current_hash = (
                await self.calculate_state_hash(SyncComponent.WALLETS)
            )
            
            self.logger.info(f"Successfully processed new wallet")
            
        except Exception as e:
            self.logger.error(f"Error handling new wallet: {str(e)}")

    async def handle_new_transaction(self, transaction_data: dict, sender: str):
        """Handle incoming new transaction with enhanced validation and propagation tracking."""
        try:
            self.logger.info(f"\n[TRANSACTION] ======= New Transaction =======")
            self.logger.info(f"[TRANSACTION] Received from: {sender}")
            
            # Create transaction object
            transaction = Transaction.from_dict(transaction_data)
            self.logger.info(f"[TRANSACTION] Hash: {transaction.tx_hash}")
            self.logger.info(f"[TRANSACTION] Amount: {transaction.amount}")

            # Check if we've seen this transaction before
            if transaction.tx_hash in self.transaction_tracker.transactions:
                self.logger.debug(f"Transaction {transaction.tx_hash} already known")
                return

            # Validate transaction
            if not await self.blockchain.validate_transaction(transaction):
                self.logger.warning(f"Invalid transaction received: {transaction.tx_hash}")
                return

            # Add to blockchain
            await self.blockchain.add_transaction(transaction)
            
            # Update tracking and sync state
            self.transaction_tracker.add_transaction(transaction.tx_hash, sender)
            self.sync_states[SyncComponent.TRANSACTIONS].current_hash = (
                await self.calculate_state_hash(SyncComponent.TRANSACTIONS)
            )

            # Propagate to other peers
            await self.propagate_transaction(transaction, sender)
            
            self.logger.info(f"[TRANSACTION] Successfully processed and propagated")
            self.logger.info(f"[TRANSACTION] =============================\n")

        except Exception as e:
            self.logger.error(f"Error handling transaction: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def handle_new_block(self, block_data: dict, sender: str):
        """Handle incoming new block with comprehensive validation and chain updates."""
        try:
            self.logger.info(f"\n[BLOCK] ========= Processing New Block =========")
            self.logger.info(f"[BLOCK] Received from: {sender}")
            
            # Deserialize and validate block
            new_block = QuantumBlock.from_dict(block_data)
            self.logger.info(f"[BLOCK] Hash: {new_block.hash}")
            self.logger.info(f"[BLOCK] Transactions: {len(new_block.transactions)}")
            
            if not self.blockchain.validate_block(new_block):
                self.logger.warning(f"Invalid block received: {new_block.hash}")
                return

            # Check if we already have this block
            if self.blockchain.has_block(new_block.hash):
                self.logger.debug(f"Block {new_block.hash} already in chain")
                return

            # Process block's transactions
            for tx in new_block.transactions:
                if not tx.tx_hash in self.transaction_tracker.transactions:
                    self.transaction_tracker.add_transaction(tx.tx_hash, sender)

            # Add block to chain
            success = await self.blockchain.add_block(new_block)
            if success:
                self.logger.info(f"[BLOCK] Successfully added to chain")
                
                # Update sync state
                self.sync_states[SyncComponent.BLOCKS].current_hash = (
                    await self.calculate_state_hash(SyncComponent.BLOCKS)
                )
                
                # Propagate to other peers
                await self.propagate_block(new_block, sender)
                
                # Remove included transactions from mempool
                await self.update_mempool_after_block(new_block)
            else:
                self.logger.warning(f"[BLOCK] Failed to add block to chain")

            self.logger.info(f"[BLOCK] ====================================\n")

        except Exception as e:
            self.logger.error(f"Error handling new block: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def propagate_transaction(self, transaction: Transaction, exclude_peer: str = None):
        """Propagate transaction to peers with retry logic."""
        peers = [p for p in self.connected_peers if p != exclude_peer]
        if not peers:
            return

        message = Message(
            type=MessageType.TRANSACTION.value,
            payload=transaction.to_dict()
        )

        for peer in peers:
            try:
                await asyncio.wait_for(
                    self.send_message(peer, message),
                    timeout=5.0
                )
                self.transaction_tracker.update_transaction(
                    transaction.tx_hash,
                    peer,
                    "propagated"
                )
            except Exception as e:
                self.logger.warning(f"Failed to propagate transaction to {peer}: {str(e)}")

    async def update_mempool_after_block(self, block: QuantumBlock):
        """Update mempool after new block is added."""
        try:
            # Remove transactions included in the block
            tx_hashes = {tx.tx_hash for tx in block.transactions}
            self.blockchain.mempool = [
                tx for tx in self.blockchain.mempool 
                if tx.tx_hash not in tx_hashes
            ]
            
            # Update mempool sync state
            self.sync_states[SyncComponent.MEMPOOL].current_hash = (
                await self.calculate_state_hash(SyncComponent.MEMPOOL)
            )
            
            self.logger.info(f"Updated mempool after block {block.hash}")
            self.logger.info(f"Remaining mempool size: {len(self.blockchain.mempool)}")
            
        except Exception as e:
            self.logger.error(f"Error updating mempool: {str(e)}")

    def verify_wallet_data(self, wallet_data: dict) -> bool:
        """Verify wallet data structure and format."""
        required_fields = {'address', 'public_key'}
        return all(field in wallet_data for field in required_fields)
    def _start_sync_queue_processor(self):
        """Start the sync queue processor"""
        try:
            asyncio.create_task(self.process_sync_queue())
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to start sync queue processor: {str(e)}")
            else:
                print(f"Failed to start sync queue processor: {str(e)}")
            raise

    async def process_sync_queue(self):
        """Process items in the sync queue"""
        while True:
            try:
                if not hasattr(self, 'sync_queue'):
                    self.logger.error("Sync queue not initialized")
                    await asyncio.sleep(1)
                    continue

                # Get sync operation from queue
                sync_op = await self.sync_queue.get()
                
                # Process the sync operation
                await self.perform_sync_operation(sync_op)
                
                # Mark task as done
                self.sync_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error processing sync queue: {str(e)}")
                else:
                    print(f"Error processing sync queue: {str(e)}")
                await asyncio.sleep(1)

    async def perform_sync_operation(self, sync_op: dict):
        """Perform a specific sync operation"""
        try:
            component = sync_op['component']
            peer = sync_op['peer']
            
            if component not in self.sync_states:
                self.logger.error(f"Invalid sync component: {component}")
                return
                
            # Check if we're already syncing this component
            if self.sync_states[component].is_syncing:
                self.logger.debug(f"Sync already in progress for {component}")
                return
            
            # Set sync state
            self.sync_states[component].is_syncing = True
            self.sync_states[component].last_sync = time.time()
            
            try:
                # Perform appropriate sync based on component
                if component == SyncComponent.BLOCKS:
                    await self.sync_blocks(peer)
                elif component == SyncComponent.WALLETS:
                    await self.sync_wallets(peer)
                elif component == SyncComponent.TRANSACTIONS:
                    await self.sync_transactions(peer)
                elif component == SyncComponent.MEMPOOL:
                    await self.sync_mempool(peer)
                
                # Update sync state
                self.sync_states[component].is_syncing = False
                self.sync_states[component].last_sync = time.time()
                self.sync_states[component].sync_progress = 100
                
            except Exception as sync_error:
                self.logger.error(f"Error syncing {component} with {peer}: {str(sync_error)}")
                # Update retry count and reset sync state
                self.sync_retry_count[component] = self.sync_retry_count.get(component, 0) + 1
                self.sync_states[component].is_syncing = False
                
        except Exception as e:
            self.logger.error(f"Error performing sync operation: {str(e)}")

    async def queue_sync_operation(self, peer: str, component: str):
        """Queue a sync operation"""
        try:
            if not hasattr(self, 'sync_queue'):
                self.sync_queue = asyncio.Queue()
                
            if not self.sync_states[component].is_syncing:
                await self.sync_queue.put({
                    'peer': peer,
                    'component': component,
                    'timestamp': time.time()
                })
                self.logger.debug(f"Queued sync operation for {component} with {peer}")
                
        except Exception as e:
            self.logger.error(f"Error queuing sync operation: {str(e)}")


async def create_enhanced_p2p_node(ip_address: str, p2p_port: int) -> P2PNode:
    """
    Create and initialize an enhanced P2P node.
    """
    try:
        logger.info(f"Initializing enhanced P2P node at {ip_address}:{p2p_port}")
        
        # Create base node
        node = P2PNode(blockchain=None, host=ip_address, port=p2p_port)
        
        # Enhance node with sync capabilities
        enhanced_node = enhance_p2p_node(node)
        
        # Start the enhanced node
        await enhanced_node.start()
        
        logger.info("Enhanced P2P node started successfully")
        return enhanced_node
        
    except Exception as e:
        logger.error(f"Error initializing enhanced P2P node: {str(e)}")
        logger.error(traceback.format_exc())
        return None

class QuantumNetworkMonitor:
    """Monitors quantum network health and performance"""
    
    def __init__(self, node: P2PNode):
        self.node = node
        self.logger = logging.getLogger("QuantumMonitor")
        self.metrics = defaultdict(list)
        self.alert_thresholds = {
            'fidelity': 0.7,
            'latency': 1.0,  # seconds
            'consensus_time': 5.0,  # seconds
            'peer_stability': 0.2     # Maximum acceptable peer count standard deviation ratio

        }
        self.monitoring_interval = 10  # seconds
    async def analyze_trends(self):
        """Analyze quantum network trends and identify patterns."""
        try:
            # Get recent metrics, minimum window size of 10
            if len(self.metrics['network_state']) < 10:
                return

            recent_metrics = self.metrics['network_state'][-10:]
            
            # Analyze each component's trends
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                try:
                    # Get fidelity trends
                    fidelities = [m['fidelities'].get(component, 0) for m in recent_metrics]
                    latencies = [m['latencies'].get(component, 0) for m in recent_metrics]
                    
                    # Calculate trend statistics
                    avg_fidelity = sum(fidelities) / len(fidelities)
                    avg_latency = sum(latencies) / len(latencies)
                    fidelity_trend = self.calculate_trend(fidelities)
                    latency_trend = self.calculate_trend(latencies)
                    
                    self.logger.debug(f"[QUANTUM] {component} trend analysis:")
                    self.logger.debug(f"  - Average fidelity: {avg_fidelity:.3f}")
                    self.logger.debug(f"  - Average latency: {avg_latency:.3f}s")
                    self.logger.debug(f"  - Fidelity trend: {fidelity_trend:+.3f}")
                    self.logger.debug(f"  - Latency trend: {latency_trend:+.3f}")
                    
                    # Check for concerning trends
                    if fidelity_trend < -0.05:  # Declining fidelity
                        await self.handle_declining_fidelity(component, avg_fidelity, fidelity_trend)
                    elif latency_trend > 0.1:  # Increasing latency
                        await self.handle_increasing_latency(component, avg_latency, latency_trend)
                        
                except Exception as comp_error:
                    self.logger.error(f"Error analyzing {component} trends: {str(comp_error)}")
                    continue

            # Analyze peer stability
            peer_counts = [m.get('peers', 0) for m in recent_metrics]
            if self.is_peer_count_unstable(peer_counts):
                await self.handle_unstable_peer_count(peer_counts)

        except Exception as e:
            self.logger.error(f"Error analyzing trends: {str(e)}")
            self.logger.error(traceback.format_exc())

    def calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in a series of values."""
        try:
            if not values:
                return 0.0
                
            n = len(values)
            if n < 2:
                return 0.0
                
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            # Calculate slope using least squares
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            return numerator / denominator if denominator != 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend: {str(e)}")
            return 0.0

    def is_peer_count_unstable(self, peer_counts: List[int]) -> bool:
        """Check if peer count is unstable."""
        if not peer_counts or len(peer_counts) < 2:
            return False
            
        # Calculate standard deviation
        mean = sum(peer_counts) / len(peer_counts)
        variance = sum((x - mean) ** 2 for x in peer_counts) / len(peer_counts)
        std_dev = variance ** 0.5
        
        # Consider unstable if std dev is more than 20% of mean
        return std_dev > (mean * 0.2)

    async def handle_declining_fidelity(self, component: str, avg_fidelity: float, trend: float):
        """Handle declining fidelity trend."""
        self.logger.warning(f"[QUANTUM] Declining fidelity detected for {component}")
        self.logger.warning(f"  - Average fidelity: {avg_fidelity:.3f}")
        self.logger.warning(f"  - Trend: {trend:+.3f}")
        
        # Try to recover quantum state
        try:
            # Find highest fidelity peer
            best_peer = await self.node.find_highest_fidelity_peer(component)
            if best_peer:
                self.logger.info(f"[QUANTUM] Attempting recovery using peer {best_peer}")
                await self.node.request_quantum_resync(best_peer, [component])
        except Exception as e:
            self.logger.error(f"Error in fidelity recovery: {str(e)}")

    async def handle_increasing_latency(self, component: str, avg_latency: float, trend: float):
        """Handle increasing latency trend."""
        self.logger.warning(f"[QUANTUM] Increasing latency detected for {component}")
        self.logger.warning(f"  - Average latency: {avg_latency:.3f}s")
        self.logger.warning(f"  - Trend: {trend:+.3f}")
        
        # Alert network about latency issues
        alert_message = Message(
            type=MessageType.QUANTUM_ALERT.value,
            payload={
                'type': 'high_latency',
                'component': component,
                'latency': avg_latency,
                'trend': trend,
                'timestamp': time.time()
            }
        )
        await self.node.broadcast(alert_message)

    async def handle_unstable_peer_count(self, peer_counts: List[int]):
        """Handle unstable peer count."""
        self.logger.warning("[QUANTUM] Unstable peer count detected")
        self.logger.warning(f"  - Recent counts: {peer_counts}")
        
        try:
            # Request peer list refresh from stable peers
            stable_peers = await self.node.get_stable_peers()
            if stable_peers:
                self.logger.info(f"[QUANTUM] Refreshing peer list using {len(stable_peers)} stable peers")
                for peer in stable_peers:
                    await self.node.request_peer_list(peer)
        except Exception as e:
            self.logger.error(f"Error handling unstable peer count: {str(e)}")

    async def generate_status_report(self) -> dict:
        """Generate comprehensive quantum network status report."""
        try:
            report = {
                'timestamp': time.time(),
                'network_size': len(self.node.quantum_sync.entangled_peers),
                'components': {},
                'alerts': [],
                'recommendations': []
            }

            # Analyze each component
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                metrics = await self.get_component_metrics(component)
                report['components'][component] = metrics

                # Generate recommendations based on metrics
                if metrics['fidelity'] < self.alert_thresholds['fidelity']:
                    report['recommendations'].append({
                        'component': component,
                        'action': 'increase_sync_frequency',
                        'reason': f"Low fidelity: {metrics['fidelity']:.3f}"
                    })

                if metrics['latency'] > self.alert_thresholds['latency']:
                    report['recommendations'].append({
                        'component': component,
                        'action': 'optimize_network',
                        'reason': f"High latency: {metrics['latency']:.3f}s"
                    })

            return report

        except Exception as e:
            self.logger.error(f"Error generating status report: {str(e)}")
            return {'error': str(e)}
    async def start_monitoring(self):
        """Start continuous network monitoring"""
        try:
            while True:
                await self.collect_metrics()
                await self.analyze_metrics()
                await self.generate_report()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {str(e)}")

    async def collect_metrics(self):
        """Collect quantum network metrics"""
        try:
            metrics = {
                'timestamp': time.time(),
                'peers': len(self.node.quantum_sync.entangled_peers),
                'fidelities': {},
                'latencies': {},
                'consensus_times': {}
            }

            # Measure fidelity for each component
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                start_time = time.time()
                fidelity = await self.node.quantum_sync.measure_sync_state(component)
                metrics['fidelities'][component] = fidelity
                metrics['latencies'][component] = time.time() - start_time

            self.metrics['network_state'].append(metrics)
            
            # Trim old metrics
            if len(self.metrics['network_state']) > 1000:
                self.metrics['network_state'] = self.metrics['network_state'][-1000:]

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")

    async def analyze_metrics(self):
        """Analyze collected metrics and detect issues"""
        try:
            latest_metrics = self.metrics['network_state'][-1]

            # Check fidelity thresholds
            for component, fidelity in latest_metrics['fidelities'].items():
                if fidelity < self.alert_thresholds['fidelity']:
                    await self.handle_low_fidelity_alert(component, fidelity)

            # Check latency thresholds
            for component, latency in latest_metrics['latencies'].items():
                if latency > self.alert_thresholds['latency']:
                    await self.handle_high_latency_alert(component, latency)

            # Analyze trends
            if len(self.metrics['network_state']) >= 10:
                await self.analyze_trends()

        except Exception as e:
            self.logger.error(f"Error analyzing metrics: {str(e)}")

    async def handle_low_fidelity_alert(self, component: str, fidelity: float):
        """Handle low fidelity alert"""
        self.logger.warning(
            f"Low quantum fidelity detected for {component}: {fidelity:.3f}"
        )
        
        # Trigger recovery
        await self.node.quantum_sync.trigger_recovery(component)
        
        # Notify network
        alert_message = Message(
            type=MessageType.QUANTUM_ALERT.value,
            payload={
                'type': 'low_fidelity',
                'component': component,
                'value': fidelity,
                'timestamp': time.time()
            }
        )
        await self.node.broadcast(alert_message)

    async def generate_report(self):
        """Generate quantum network health report"""
        try:
            report = {
                'timestamp': time.time(),
                'network_size': len(self.node.quantum_sync.entangled_peers),
                'components': {},
                'alerts': [],
                'recommendations': []
            }

            # Analyze each component
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                metrics = await self.get_component_metrics(component)
                report['components'][component] = metrics

                # Generate recommendations based on metrics
                if metrics['fidelity'] < 0.9:
                    report['recommendations'].append({
                        'component': component,
                        'action': 'increase_sync_frequency',
                        'reason': f"Low fidelity: {metrics['fidelity']:.3f}"
                    })

            # Save report
            report_file = f"quantum_network_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Generated quantum network report: {report_file}")
            return report

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None

    async def get_component_metrics(self, component: str) -> dict:
        """Get detailed metrics for a component with enhanced error handling."""
        try:
            metrics = {
                'fidelity': 0.0,
                'avg_latency': 0.0,
                'fidelity_trend': 0.0,
                'sync_failures': 0
            }
            
            try:
                fidelity = await self.node.quantum_sync.measure_sync_state(component)
                metrics['fidelity'] = fidelity if fidelity is not None else 0.0
            except Exception as e:
                self.logger.error(f"Error measuring fidelity for {component}: {str(e)}")
                
            # Get recent metrics with error handling
            try:
                recent_metrics = [
                    m for m in self.metrics['network_state'][-100:]
                    if component in m['fidelities']
                ]
                
                if recent_metrics:
                    # Calculate average latency
                    latencies = [m['latencies'].get(component, 0) for m in recent_metrics]
                    metrics['avg_latency'] = sum(latencies) / len(latencies) if latencies else 0
                    
                    # Calculate fidelity trend safely
                    fidelities = [m['fidelities'].get(component, 0) for m in recent_metrics]
                    if len(fidelities) > 1:
                        try:
                            coeffs = np.polyfit(range(len(fidelities)), fidelities, 1)
                            metrics['fidelity_trend'] = coeffs[0]
                        except Exception as e:
                            self.logger.error(f"Error calculating fidelity trend: {str(e)}")
                            metrics['fidelity_trend'] = 0.0
                    
                    # Count sync failures
                    metrics['sync_failures'] = sum(
                        1 for m in recent_metrics
                        if m['fidelities'].get(component, 0) < self.node.quantum_sync.decoherence_threshold
                    )
                    
            except Exception as e:
                self.logger.error(f"Error processing recent metrics: {str(e)}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting component metrics: {str(e)}")
            # Return default metrics instead of raising
            return {
                'fidelity': 0.0,
                'avg_latency': 0.0,
                'fidelity_trend': 0.0,
                'sync_failures': 0
            }
class QuantumP2PTestHandler:
    """Handler for P2P test messages with quantum features"""
    
    def __init__(self, node: 'P2PNode'):
        self.node = node
        self.logger = logging.getLogger("QuantumP2PTest")

    async def handle_test_message(self, websocket: websockets.WebSocketServerProtocol, data: dict) -> dict:
        """Handle P2P test messages with comprehensive error handling"""
        try:
            action = data.get('action')
            if not action:
                return {"status": "error", "message": "No action specified"}

            handler_map = {
                'test_consensus': self.handle_consensus_test,
                'test_peer_connection': self.handle_peer_connection_test,
                'test_quantum_entanglement': self.handle_quantum_entanglement_test,
                'test_transaction_propagation': self.handle_transaction_propagation_test,
                'get_network_metrics': self.handle_network_metrics
            }

            handler = handler_map.get(action)
            if not handler:
                return {"status": "error", "message": f"Unknown action: {action}"}

            return await handler(data)

        except Exception as e:
            self.logger.error(f"Error handling test message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    async def handle_consensus_test(self, data: dict) -> dict:
        """Handle DAGKnight consensus test"""
        try:
            params = data.get('params', {})
            
            # Initialize DAGKnight consensus
            self.node.dagknight = DAGKnightConsensus(
                min_k_cluster=params.get('min_k_cluster', 10),
                max_latency_ms=params.get('max_latency_ms', 1000)
            )

            # Create test blocks
            blocks = []
            for i in range(5):
                block = QuantumBlock(
                    index=len(self.node.blockchain.chain),
                    previous_hash=self.node.blockchain.get_latest_block().hash,
                    timestamp=int(time.time() * 1000),
                    transactions=[],
                    quantum_enabled=True
                )
                blocks.append(block)

            # Track consensus metrics
            metrics = {
                'consensus_time': 0,
                'final_agreement': 0,
                'quantum_verification': False,
                'k_clusters': [],
                'latencies': [],
                'confirmation_scores': []
            }

            start_time = time.time()

            # Process blocks through consensus
            for block in blocks:
                network_latency = random.uniform(100, 500)  # Simulated network latency
                confirmable = await self.node.dagknight.add_block(block, network_latency)
                
                if confirmable:
                    metrics['confirmation_scores'].append(1.0)
                    metrics['k_clusters'].append(
                        len(self.node.dagknight.k_clusters)
                    )
                    metrics['latencies'].append(network_latency)

            metrics['consensus_time'] = time.time() - start_time
            metrics['final_agreement'] = len([s for s in metrics['confirmation_scores'] if s >= 0.8]) / len(blocks)
            metrics['quantum_verification'] = all(block.quantum_enabled for block in blocks)

            return {
                "status": "success",
                "consensus_metrics": metrics,
                "block_hash": blocks[-1].hash if blocks else None
            }

        except Exception as e:
            self.logger.error(f"Consensus test failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    async def handle_peer_connection_test(self, data: dict) -> dict:
        """Handle peer connection test"""
        try:
            peer_address = data.get('peer_address')
            if not peer_address:
                return {"status": "error", "message": "No peer address provided"}

            # Attempt connection
            start_time = time.time()
            connected = await self.node.connect_to_peer(
                KademliaNode(
                    id=self.node.generate_node_id(),
                    ip=peer_address.split(':')[0],
                    port=int(peer_address.split(':')[1])
                )
            )
            connection_time = time.time() - start_time

            if connected:
                return {
                    "status": "success",
                    "connection_status": "connected",
                    "message_exchange": True,
                    "peer_info": {
                        "latency": connection_time,
                        "quantum_ready": peer_address in self.node.quantum_sync.entangled_peers
                    }
                }
            else:
                return {"status": "error", "message": "Connection failed"}

        except Exception as e:
            self.logger.error(f"Peer connection test failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_quantum_entanglement_test(self, data: dict) -> dict:
        """Handle quantum entanglement test"""
        try:
            peer_address = data.get('peer_address')
            if not peer_address:
                return {"status": "error", "message": "No peer address provided"}

            # Establish quantum entanglement
            success = await self.node.establish_quantum_entanglement(peer_address)
            if not success:
                return {"status": "error", "message": "Failed to establish quantum entanglement"}

            # Measure fidelities
            fidelities = {}
            for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                fidelity = await self.node.quantum_sync.measure_sync_state(component)
                fidelities[component] = fidelity

            return {
                "status": "success",
                "entanglement_status": "established",
                "fidelities": fidelities,
                "quantum_metrics": {
                    "sync_quality": sum(fidelities.values()) / len(fidelities),
                    "entanglement_time": time.time()
                }
            }

        except Exception as e:
            self.logger.error(f"Quantum entanglement test failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_transaction_propagation_test(self, data: dict) -> dict:
        """Handle transaction propagation test"""
        try:
            # Create test transaction
            tx = Transaction(
                sender=data.get('sender', 'test_sender'),
                receiver=data.get('receiver', 'test_receiver'),
                amount=Decimal(data.get('amount', '1.0')),
                quantum_enabled=data.get('quantum_enabled', False)
            )

            start_time = time.time()
            await self.node.broadcast_transaction(tx)
            propagation_time = time.time() - start_time

            # Calculate network coverage
            reached_peers = len([p for p in self.node.connected_peers 
                               if tx.tx_hash in self.node.transaction_tracker.transactions])
            coverage = reached_peers / len(self.node.connected_peers) if self.node.connected_peers else 0

            return {
                "status": "success",
                "tx_hash": tx.tx_hash,
                "propagation_metrics": {
                    "propagation_time": propagation_time,
                    "network_coverage": coverage,
                    "reached_peers": reached_peers
                }
            }

        except Exception as e:
            self.logger.error(f"Transaction propagation test failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def handle_network_metrics(self, data: dict) -> dict:
        """Handle network metrics request"""
        try:
            return {
                "status": "success",
                "network_metrics": {
                    "peer_metrics": {
                        "total_peers": len(self.node.peers),
                        "active_peers": len(self.node.connected_peers)
                    },
                    "quantum_metrics": {
                        "average_fidelity": await self.calculate_average_fidelity(),
                        "decoherence_events": self.get_decoherence_events()
                    },
                    "consensus_metrics": {
                        "network_status": await self.get_consensus_status()
                    },
                    "transaction_metrics": {
                        "confirmation_rate": self.calculate_confirmation_rate()
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting network metrics: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def calculate_average_fidelity(self) -> float:
        """Calculate average quantum fidelity across all components"""
        fidelities = []
        for component in ['wallets', 'transactions', 'blocks', 'mempool']:
            try:
                fidelity = await self.node.quantum_sync.measure_sync_state(component)
                fidelities.append(fidelity)
            except Exception as e:
                self.logger.error(f"Error measuring fidelity for {component}: {str(e)}")
        
        return sum(fidelities) / len(fidelities) if fidelities else 0.0

    def get_decoherence_events(self) -> List[dict]:
        """Get list of recent decoherence events"""
        return [
            {
                "component": event["component"],
                "timestamp": event["timestamp"],
                "fidelity": event["fidelity"]
            }
            for event in self.node.quantum_monitor.decoherence_events[-10:]
        ]

    async def get_consensus_status(self) -> dict:
        """Get current consensus status"""
        if hasattr(self.node, 'dagknight'):
            status = await self.node.dagknight.get_network_status()
            return {
                "consensus_level": status.get('final_agreement', 0),
                "k_clusters": len(status.get('k_clusters', [])),
                "average_latency": status.get('average_latency', 0)
            }
        return {"consensus_level": 0, "k_clusters": 0, "average_latency": 0}

    def calculate_confirmation_rate(self) -> float:
        """Calculate transaction confirmation rate"""
        if not self.node.transaction_tracker.transactions:
            return 1.0
        
        confirmed = sum(1 for tx in self.node.transaction_tracker.transactions.values()
                       if tx.status == "confirmed")
        return confirmed / len(self.node.transaction_tracker.transactions)
class DAGKnightConsensus:
    """Implementation of DAGKnight consensus protocol."""
    
    def __init__(self, min_k_cluster: int = 10, max_latency_ms: int = 1000):
        self.min_k_cluster = min_k_cluster
        self.max_latency_ms = max_latency_ms
        self.block_dag = {}  # DAG structure: block_hash -> {parents: [], children: [], timestamp: float}
        self.network_latencies = {}  # peer -> {timestamp: latency}
        self.k_clusters = {}  # cluster_id -> {blocks: [], latency: float}
        self.confirmed_blocks = set()
        self.logger = logging.getLogger("DAGKnight")

    async def add_block(self, block: QuantumBlock, network_latency: float) -> bool:
        """Add a new block to the DAG with network latency information."""
        try:
            block_hash = block.hash
            
            # Add block to DAG
            self.block_dag[block_hash] = {
                'parents': block.parent_hashes,
                'children': [],
                'timestamp': block.timestamp,
                'latency': network_latency
            }
            
            # Update parent-child relationships
            for parent_hash in block.parent_hashes:
                if parent_hash in self.block_dag:
                    self.block_dag[parent_hash]['children'].append(block_hash)
            
            # Update k-clusters
            await self.update_k_clusters(block_hash)
            
            # Check if block can be confirmed
            confirmable = await self.verify_block_confirmability(block_hash)
            
            if confirmable:
                self.confirmed_blocks.add(block_hash)
                self.logger.info(f"Block {block_hash[:8]} confirmed with latency {network_latency:.2f}ms")
            
            return confirmable
            
        except Exception as e:
            self.logger.error(f"Error adding block: {str(e)}")
            return False

    async def update_k_clusters(self, new_block_hash: str):
        """Update k-clusters after new block addition."""
        try:
            # Find all k-clusters containing the new block
            current_blocks = {new_block_hash}
            cluster_blocks = set()
            
            # Expand cluster by following parent and child links
            while len(current_blocks) < self.min_k_cluster:
                next_blocks = set()
                
                for block_hash in current_blocks:
                    block_data = self.block_dag[block_hash]
                    next_blocks.update(block_data['parents'])
                    next_blocks.update(block_data['children'])
                
                if not next_blocks - cluster_blocks:
                    break
                    
                cluster_blocks.update(current_blocks)
                current_blocks = next_blocks - cluster_blocks
            
            # Calculate cluster latency
            cluster_latency = max(
                self.block_dag[block_hash]['latency']
                for block_hash in cluster_blocks
            )
            
            # Store new k-cluster
            cluster_id = hashlib.sha256(
                ''.join(sorted(cluster_blocks)).encode()
            ).hexdigest()
            
            self.k_clusters[cluster_id] = {
                'blocks': cluster_blocks,
                'latency': cluster_latency
            }
            
            # Remove old clusters
            self._cleanup_old_clusters()
            
        except Exception as e:
            self.logger.error(f"Error updating k-clusters: {str(e)}")

    async def verify_block_confirmability(self, block_hash: str) -> bool:
        """Verify if a block can be confirmed based on k-cluster analysis."""
        try:
            # Get all k-clusters containing this block
            relevant_clusters = [
                cluster for cluster in self.k_clusters.values()
                if block_hash in cluster['blocks']
            ]
            
            if not relevant_clusters:
                return False
                
            # Sort clusters by latency
            sorted_clusters = sorted(
                relevant_clusters,
                key=lambda c: c['latency']
            )
            
            # Find median cluster latency
            median_cluster = sorted_clusters[len(sorted_clusters) // 2]
            median_latency = median_cluster['latency']
            
            # Check if block is confirmable
            block_age = time.time() - self.block_dag[block_hash]['timestamp']
            return block_age >= 2 * median_latency
            
        except Exception as e:
            self.logger.error(f"Error verifying block confirmability: {str(e)}")
            return False

    def _cleanup_old_clusters(self):
        """Remove outdated k-clusters."""
        current_time = time.time()
        old_clusters = [
            cluster_id
            for cluster_id, cluster in self.k_clusters.items()
            if all(
                current_time - self.block_dag[block_hash]['timestamp'] > self.max_latency_ms
                for block_hash in cluster['blocks']
            )
        ]
        
        for cluster_id in old_clusters:
            del self.k_clusters[cluster_id]

    async def get_network_status(self) -> dict:
        """Get current network status and metrics."""
        try:
            current_time = time.time()
            
            # Calculate network statistics
            confirmed_count = len(self.confirmed_blocks)
            pending_count = len(self.block_dag) - confirmed_count
            
            # Calculate average confirmation latency
            recent_latencies = [
                self.block_dag[block_hash]['latency']
                for block_hash in self.confirmed_blocks
                if current_time - self.block_dag[block_hash]['timestamp'] < 3600  # Last hour
            ]
            
            avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
            
            # Get k-cluster statistics
            cluster_stats = {
                'count': len(self.k_clusters),
                'avg_size': sum(len(c['blocks']) for c in self.k_clusters.values()) / len(self.k_clusters) if self.k_clusters else 0,
                'avg_latency': sum(c['latency'] for c in self.k_clusters.values()) / len(self.k_clusters) if self.k_clusters else 0
            }
            
            return {
                'confirmed_blocks': confirmed_count,
                'pending_blocks': pending_count,
                'average_latency': avg_latency,
                'k_clusters': cluster_stats,
                'timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting network status: {str(e)}")
            return {}

    async def analyze_network_security(self) -> dict:
        """Analyze network security based on k-cluster distribution."""
        try:
            if not self.k_clusters:
                return {'status': 'insufficient_data'}
                
            # Calculate latency distribution
            latencies = [cluster['latency'] for cluster in self.k_clusters.values()]
            latencies.sort()
            
            # Find 50th percentile latency
            median_latency = latencies[len(latencies) // 2]
            
            # Calculate network coverage at different latency thresholds
            coverage_analysis = {}
            for threshold in [0.5, 1.0, 1.5, 2.0]:
                max_latency = median_latency * threshold
                covered_blocks = set()
                
                for cluster in self.k_clusters.values():
                    if cluster['latency'] <= max_latency:
                        covered_blocks.update(cluster['blocks'])
                
                coverage_analysis[f'{threshold}x_median'] = {
                    'latency': max_latency,
                    'coverage': len(covered_blocks) / len(self.block_dag) if self.block_dag else 0
                }
            
            return {
                'status': 'secure' if coverage_analysis['1.0x_median']['coverage'] >= 0.5 else 'warning',
                'median_latency': median_latency,
                'coverage_analysis': coverage_analysis,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing network security: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    def extend_p2p_node():
        """Extend P2PNode with enhanced sync and consensus functionality"""
        
        # Add new attributes to P2PNode.__init__
        original_init = P2PNode.__init__
        
        def enhanced_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Initialize consensus handler
            self.consensus_handler = ConsensusMessageHandler(self)
            
            # Update message handlers to include consensus
            self.message_handlers = {
                'mining': self.handle_mining_message,
                'transaction': self.handle_transaction_message,
                'wallet': self.handle_wallet_message,
                'consensus': self.consensus_handler.handle_message
            }
            
            # Initialize DAGKnight consensus
            asyncio.create_task(self.initialize_dagknight())
            
        P2PNode.__init__ = enhanced_init

        # Add DAGKnight initialization method
        async def initialize_dagknight(self):
            """Initialize DAGKnight consensus system."""
            self.dagknight = DAGKnightConsensus(
                min_k_cluster=10,
                max_latency_ms=1000
            )
            self.logger.info("DAGKnight consensus system initialized")

        P2PNode.initialize_dagknight = initialize_dagknight

    # Add enhanced heartbeat methods
    async def send_enhanced_heartbeat(self, peer: str):
        """Send enhanced heartbeat with sync information"""
        try:
            heartbeat_data = {
                'timestamp': time.time(),
                'node_state': await self.get_node_state_data(),
                'sync_status': await self.get_sync_status(),
                'chain_height': len(self.blockchain.chain) if self.blockchain else 0,
                'mempool_size': len(self.blockchain.mempool) if self.blockchain else 0
            }

            message = Message(
                type=MessageType.HEARTBEAT.value,
                payload={
                    'heartbeat': heartbeat_data,
                    'sync_request': await self.check_sync_needs()
                }
            )
            await self.send_message(peer, message)
            self.logger.debug(f"Enhanced heartbeat sent to {peer}")

        except Exception as e:
            self.logger.error(f"Error sending enhanced heartbeat to {peer}: {str(e)}")

    P2PNode.send_enhanced_heartbeat = send_enhanced_heartbeat

    async def handle_enhanced_heartbeat(self, peer: str, data: dict):
        """Handle incoming enhanced heartbeat messages"""
        try:
            # Update peer's last seen time
            self.last_heartbeat_data[peer] = data
            
            # Check for sync needs
            if 'sync_request' in data:
                await self.handle_sync_request(peer, data['sync_request'])
            
            # Compare states and trigger sync if needed
            peer_state = data['node_state']
            await self.check_and_trigger_sync(peer, peer_state)
            
        except Exception as e:
            self.logger.error(f"Error handling enhanced heartbeat from {peer}: {str(e)}")

    P2PNode.handle_enhanced_heartbeat = handle_enhanced_heartbeat

    # Add sync management methods
    async def check_sync_needs(self, peer: str):
        """Check if sync is needed after handling a message."""
        try:
            current_state = {
                SyncComponent.WALLETS: await self.calculate_state_hash(SyncComponent.WALLETS),
                SyncComponent.TRANSACTIONS: await self.calculate_state_hash(SyncComponent.TRANSACTIONS),
                SyncComponent.BLOCKS: await self.calculate_state_hash(SyncComponent.BLOCKS),
                SyncComponent.MEMPOOL: await self.calculate_state_hash(SyncComponent.MEMPOOL)
            }

            # Request peer's state
            response = await self.send_and_wait_for_response(
                peer,
                Message(
                    type=MessageType.SYNC_STATUS.value,
                    payload={"state_hashes": current_state}
                )
            )

            if response and response.type == MessageType.SYNC_STATUS.value:
                peer_state = response.payload.get("state_hashes", {})
                
                # Compare states and trigger sync if needed
                for component in SyncComponent.__dict__.values():
                    if isinstance(component, str):  # Filter actual components
                        our_hash = current_state.get(component)
                        peer_hash = peer_state.get(component)
                        
                        if peer_hash and peer_hash != our_hash:
                            self.logger.info(f"[SYNC] State mismatch detected for {component}")
                            self.logger.info(f"[SYNC] Our hash: {our_hash}")
                            self.logger.info(f"[SYNC] Peer hash: {peer_hash}")
                            await self.start_sync(peer, component)

        except Exception as e:
            self.logger.error(f"Error checking sync needs: {str(e)}")
    
    async def check_and_trigger_sync(self, peer: str, peer_state: dict):
        """Compare states and trigger sync if needed"""
        try:
            # Check blockchain height
            local_height = len(self.blockchain.chain) if self.blockchain else 0
            peer_height = peer_state.get('blockchain_height', 0)
            
            if peer_height > local_height:
                await self.queue_sync_operation(peer, 'blocks')
            
            # Check wallet differences
            local_wallet_hash = await self.calculate_wallet_hash()
            peer_wallet_hash = peer_state.get('wallet_hash')
            
            if peer_wallet_hash and peer_wallet_hash != local_wallet_hash:
                await self.queue_sync_operation(peer, 'wallets')
            
            # Check transaction differences
            local_tx_hash = await self.calculate_transaction_hash()
            peer_tx_hash = peer_state.get('transaction_hash')
            
            if peer_tx_hash and peer_tx_hash != local_tx_hash:
                await self.queue_sync_operation(peer, 'transactions')
                
        except Exception as e:
            self.logger.error(f"Error checking and triggering sync with {peer}: {str(e)}")

    P2PNode.check_and_trigger_sync = check_and_trigger_sync

    async def queue_sync_operation(self, peer: str, component: str):
        """Queue a sync operation"""
        try:
            if not self.sync_states[component].in_progress:
                await self.sync_queue.put({
                    'peer': peer,
                    'component': component,
                    'timestamp': time.time()
                })
                self.sync_states[component].in_progress = True
                self.sync_states[component].last_attempt = time.time()
                
        except Exception as e:
            self.logger.error(f"Error queuing sync operation: {str(e)}")

    P2PNode.queue_sync_operation = queue_sync_operation

    # Add main sync processing loop
    async def process_sync_queue(self):
        """Process queued sync operations"""
        while True:
            try:
                if not self.sync_queue.empty():
                    sync_op = await self.sync_queue.get()
                    await self.perform_sync_operation(sync_op)
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error processing sync queue: {str(e)}")
                await asyncio.sleep(5)

    P2PNode.process_sync_queue = process_sync_queue

    async def perform_sync_operation(self, sync_op: dict):
        """Perform a specific sync operation"""
        peer = sync_op['peer']
        component = sync_op['component']
        
        try:
            self.logger.info(f"Starting sync operation: {component} with peer {peer}")
            
            if component == 'blocks':
                await self.sync_blocks_with_peer(peer)
            elif component == 'wallets':
                await self.sync_wallets_with_peer(peer)
            elif component == 'transactions':
                await self.sync_transactions_with_peer(peer)
                
            # Update sync state
            self.sync_states[component].status = "synced"
            self.sync_states[component].last_update = time.time()
            self.sync_states[component].in_progress = False
            self.sync_states[component].peers_synced.add(peer)
            
        except Exception as e:
            self.logger.error(f"Error performing sync operation {component} with {peer}: {str(e)}")
            self.sync_states[component].status = "failed"
            self.sync_states[component].in_progress = False
            self.sync_states[component].retry_count += 1

    P2PNode.perform_sync_operation = perform_sync_operation

    # Modify existing heartbeat method to use enhanced version
    async def send_heartbeats(self):
        """Send both regular and quantum heartbeats to all peers"""
        while self.is_running:
            try:
                peers = list(self.connected_peers)
                self.logger.info(f"\n[HEARTBEAT] {'='*20} Heartbeat Round {'='*20}")
                self.logger.info(f"[HEARTBEAT] Sending heartbeats to {len(peers)} peers")

                for peer in peers:
                    try:
                        # First send regular heartbeat
                        await self.send_message(  # Removed extra peer argument
                            peer, 
                            Message(
                                type=MessageType.HEARTBEAT.value,
                                payload={'timestamp': time.time()}
                            )
                        )
                        self.logger.debug(f"[HEARTBEAT] Regular heartbeat sent to {peer}")

                        # Then send quantum heartbeat if peer is quantum-entangled
                        if (hasattr(self, 'quantum_sync') and 
                            self.quantum_sync and 
                            peer in self.quantum_sync.entangled_peers):
                            
                            try:
                                self.logger.info(f"[QUANTUM] Sending quantum heartbeat to {peer}")
                                # Get current quantum states and fidelities with error handling
                                quantum_states = {}
                                fidelities = {}
                                
                                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                                    try:
                                        qubit = getattr(self.quantum_sync.register, component)
                                        if qubit is None:
                                            continue
                                            
                                        quantum_states[component] = qubit.value
                                        fidelity = await self.quantum_sync.measure_sync_state(component)
                                        fidelities[component] = fidelity
                                        self.logger.debug(f"[QUANTUM] {component} fidelity: {fidelity:.3f}")
                                    except Exception as e:
                                        self.logger.error(f"[QUANTUM] Error getting state for {component}: {str(e)}")
                                        continue

                                # Skip quantum heartbeat if no states were collected
                                if not quantum_states:
                                    self.logger.warning(f"[QUANTUM] No valid quantum states for peer {peer}")
                                    continue

                                # Get and validate Bell pair
                                bell_pair = self.quantum_sync.bell_pairs.get(peer)
                                if not bell_pair:
                                    self.logger.warning(f"[QUANTUM] No Bell pair found for peer {peer}")
                                    continue

                                # Get Bell pair ID
                                try:
                                    bell_pair_id = self.quantum_notifier._get_bell_pair_id(bell_pair)
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error getting Bell pair ID: {str(e)}")
                                    continue

                                # Generate secure nonce
                                try:
                                    nonce = os.urandom(16).hex()
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error generating nonce: {str(e)}")
                                    continue
                                    
                                # Create quantum heartbeat
                                heartbeat = QuantumHeartbeat(
                                    node_id=self.node_id,
                                    timestamp=time.time(),
                                    quantum_states=quantum_states,
                                    fidelities=fidelities,
                                    bell_pair_id=bell_pair_id,
                                    nonce=nonce
                                )

                                # Sign heartbeat with error handling
                                try:
                                    message_str = (
                                        f"{heartbeat.node_id}:{heartbeat.timestamp}:"
                                        f"{heartbeat.bell_pair_id}:{heartbeat.nonce}"
                                    ).encode()
                                    
                                    signature = self.private_key.sign(
                                        message_str,
                                        padding.PSS(
                                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                            salt_length=padding.PSS.MAX_LENGTH
                                        ),
                                        hashes.SHA256()
                                    )
                                    heartbeat.signature = base64.b64encode(signature).decode()
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error signing heartbeat: {str(e)}")
                                    continue

                                # Send quantum heartbeat with retry
                                try:
                                    await self.send_message(
                                        peer,
                                        Message(
                                            type=MessageType.QUANTUM_HEARTBEAT.value,
                                            payload=heartbeat.to_dict()
                                        )
                                    )
                                    self.logger.info(f"[QUANTUM] ✓ Quantum heartbeat sent to {peer}")
                                    self.logger.debug(f"[QUANTUM] Current fidelities: {fidelities}")
                                    
                                    # Update tracking only on successful send
                                    self.last_quantum_heartbeat[peer] = time.time()
                                except Exception as e:
                                    self.logger.error(f"[QUANTUM] Error sending quantum heartbeat: {str(e)}")
                                    continue

                            except Exception as qe:
                                self.logger.error(f"[QUANTUM] Error in quantum heartbeat for {peer}: {str(qe)}")
                                # Continue with next peer without removing - quantum errors shouldn't break connection
                                continue

                    except Exception as e:
                        self.logger.error(f"Error sending heartbeats to {peer}: {str(e)}")
                        # Only remove peer for non-quantum errors
                        await self.remove_peer(peer)
                        continue

                    # Brief pause between peers to avoid flooding
                    await asyncio.sleep(0.1)

                self.logger.info(f"[HEARTBEAT] {'='*50}\n")
                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(self.heartbeat_interval)


    P2PNode.send_heartbeats = send_heartbeats
    async def monitor_peer_connections(self):
        """Monitor peer connections and status."""
        while self.is_running:
            try:
                self.logger.info("\n=== Monitoring Peer Connections ===")
                current_peers = []
                
                # Get current peers safely
                try:
                    async with self.peer_lock:
                        current_peers = list(self.peers.keys())
                except Exception as e:
                    self.logger.error(f"Error accessing peers: {str(e)}")
                    continue

                # Process each peer
                for peer in current_peers:
                    try:
                        # Check peer connection
                        if not await self.is_peer_connected(peer):
                            self.logger.warning(f"Peer {peer} not responsive")
                            await self.handle_unresponsive_peer(peer)
                        else:
                            async with self.peer_lock:
                                if peer in self.peer_info:
                                    self.peer_info[peer]['last_activity'] = time.time()
                    except Exception as peer_error:
                        self.logger.error(f"Error processing peer {peer}: {str(peer_error)}")
                        continue

                # Log status
                try:
                    async with self.peer_lock:
                        connected_count = len(self.connected_peers)
                        total_peers = len(self.peers)
                        active_peers = list(self.connected_peers)
                        
                    self.logger.info(f"Connected peers: {connected_count}/{total_peers}")
                    self.logger.info(f"Active peers: {active_peers}")
                except Exception as status_error:
                    self.logger.error(f"Error logging peer status: {str(status_error)}")

                # Check peer count
                try:
                    if len(self.connected_peers) < self.target_peer_count:
                        self.logger.info(f"Need more peers. Current: {len(self.connected_peers)}, Target: {self.target_peer_count}")
                        await self.find_and_connect_to_new_peers()
                except Exception as peer_count_error:
                    self.logger.error(f"Error checking peer count: {str(peer_count_error)}")

                # Wait before next check
                await asyncio.sleep(15)

            except Exception as e:
                self.logger.error(f"Error in peer connection monitoring: {str(e)}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(15)

    async def handle_unresponsive_peer(self, peer: str):
        """Handle unresponsive peer with recovery attempts."""
        try:
            # Check quantum entanglement status
            was_entangled = False
            try:
                was_entangled = (hasattr(self, 'quantum_sync') and 
                               peer in self.quantum_sync.entangled_peers)
            except Exception as quantum_error:
                self.logger.error(f"Error checking quantum status: {str(quantum_error)}")

            # Remove peer
            try:
                await self.remove_peer(peer)
            except Exception as remove_error:
                self.logger.error(f"Error removing peer {peer}: {str(remove_error)}")

            # Attempt recovery if was quantum entangled
            if was_entangled:
                self.logger.warning(f"Lost quantum peer {peer}, attempting recovery")
                try:
                    await self.attempt_peer_recovery(peer)
                except Exception as recovery_error:
                    self.logger.error(f"Error in peer recovery: {str(recovery_error)}")

        except Exception as e:
            self.logger.error(f"Error handling unresponsive peer {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def attempt_peer_recovery(self, peer: str):
        """Attempt to recover connection to a lost peer."""
        try:
            # Parse peer address
            ip, port = peer.split(':')
            port = int(port)

            # Create new node instance
            node = KademliaNode(
                id=self.generate_node_id(),
                ip=ip,
                port=port
            )

            # Attempt connection
            if await self.connect_to_peer(node):
                self.logger.info(f"Successfully recovered peer {peer}")
            else:
                self.logger.warning(f"Failed to recover peer {peer}")

        except Exception as e:
            self.logger.error(f"Error attempting peer recovery: {str(e)}")
            self.logger.error(traceback.format_exc())


    # Start the sync processing when the node starts
    original_start = P2PNode.start
    async def handle_consensus_message(self, data: dict) -> dict:
        """
        Route consensus messages to the consensus handler
        """
        try:
            self.logger.info(f"[CONSENSUS] Processing consensus message: {data}")
            if not self.consensus_handler:
                return {
                    'status': 'error',
                    'message': 'Consensus handler not initialized'
                }

            response = await self.consensus_handler.handle_message(data)
            self.logger.debug(f"[CONSENSUS] Handler response: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling consensus message: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }

    P2PNode.handle_consensus_message = handle_consensus_message



        # Enhanced start method without passing `self` to original_start
    def enhance_p2p_node(node: P2PNode) -> P2PNode:
        """Enhance P2P node with additional functionality"""
        try:
            # Add systemd journal handling
            node.systemd_journal = systemd.journal.JournalHandler()
            node.logger.addHandler(node.systemd_journal)
            
            # Add NetworkOptimizer
            node.network_optimizer = NetworkOptimizer(node)
            
            # Add other enhancements from the original implementation...
            
            # Add sync states
            node.sync_states = {
                SyncComponent.WALLETS: SyncStatus(),
                SyncComponent.TRANSACTIONS: SyncStatus(),
                SyncComponent.BLOCKS: SyncStatus(),
                SyncComponent.MEMPOOL: SyncStatus()
            }
            
            # Add monitoring tasks list
            node._monitoring_tasks = []
            
            return node

        except Exception as e:
            logger.error(f"Error enhancing P2P node: {str(e)}")
            logger.error(traceback.format_exc())
            return None



class QuantumConsensusManager:
    """Manages quantum consensus across the P2P network"""
    
    def __init__(self, node: P2PNode, consensus_threshold: float = 0.8):
        self.node = node
        self.consensus_threshold = consensus_threshold
        self.logger = logging.getLogger("QuantumConsensus")
        self.consensus_states: Dict[str, Dict[str, float]] = {}
        self.last_consensus_check = defaultdict(float)
        self.consensus_cache: Dict[str, bool] = {}
        self.consensus_lock = asyncio.Lock()

    async def check_quantum_consensus(self, component: str) -> bool:
        """Check quantum consensus for a specific component"""
        try:
            async with self.consensus_lock:
                current_time = time.time()
                
                # Use cached result if recent
                if component in self.consensus_cache:
                    if current_time - self.last_consensus_check[component] < 5:
                        return self.consensus_cache[component]

                consensus_achieved = False
                peer_states = {}
                
                # Collect quantum states from all entangled peers
                for peer_id in self.node.quantum_sync.entangled_peers:
                    try:
                        fidelity = await self.node.quantum_sync.measure_sync_state(
                            component, peer_id
                        )
                        peer_states[peer_id] = fidelity
                    except Exception as e:
                        self.logger.error(f"Error measuring state for {peer_id}: {str(e)}")

                if peer_states:
                    # Calculate consensus metrics
                    avg_fidelity = np.mean(list(peer_states.values()))
                    min_fidelity = min(peer_states.values())
                    max_fidelity = max(peer_states.values())
                    
                    # Log consensus metrics
                    self.logger.info(f"Consensus metrics for {component}:")
                    self.logger.info(f"  Average fidelity: {avg_fidelity:.3f}")
                    self.logger.info(f"  Min fidelity: {min_fidelity:.3f}")
                    self.logger.info(f"  Max fidelity: {max_fidelity:.3f}")
                    
                    # Check if consensus is achieved
                    consensus_achieved = (
                        avg_fidelity >= self.consensus_threshold and
                        min_fidelity >= self.consensus_threshold * 0.9
                    )

                # Update cache
                self.consensus_cache[component] = consensus_achieved
                self.last_consensus_check[component] = current_time

                return consensus_achieved

        except Exception as e:
            self.logger.error(f"Error checking quantum consensus: {str(e)}")
            return False
            
class ConsensusMessageHandler:
    """Handler for DAGKnight consensus messages"""
    
    def __init__(self, node: 'P2PNode'):  # Note the quotes around P2PNode for forward reference
        self.node = node
        self.logger = logging.getLogger("ConsensusHandler")


    async def handle_message(self, message: dict) -> dict:
        """Handle incoming consensus messages"""
        try:
            action = message.get('action')
            if not action:
                return {'status': 'error', 'message': 'No action specified'}
            if action == 'initialize':
                return await self.handle_initialize(message)

            handler = getattr(self, f'handle_{action}', None)
            if not handler:
                return {'status': 'error', 'message': f'Unknown action: {action}'}

            return await handler(message)

        except Exception as e:
            self.logger.error(f"Error handling consensus message: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_initialize(self, message: dict) -> dict:
        """Initialize DAGKnight consensus"""
        try:
            params = message.get('params', {})
            min_k_cluster = params.get('min_k_cluster', 10)
            max_latency_ms = params.get('max_latency_ms', 1000)
            quantum_threshold = params.get('quantum_threshold', 0.85)

            # Initialize DAGKnight consensus
            self.node.dagknight = DAGKnightConsensus(
                min_k_cluster=min_k_cluster,
                max_latency_ms=max_latency_ms
            )

            self.logger.info(f"DAGKnight consensus initialized with parameters:")
            self.logger.info(f"  Min K-Cluster: {min_k_cluster}")
            self.logger.info(f"  Max Latency: {max_latency_ms}ms")
            self.logger.info(f"  Quantum Threshold: {quantum_threshold}")

            return {
                'status': 'success',
                'message': 'DAGKnight consensus initialized'
            }

        except Exception as e:
            self.logger.error(f"Failed to initialize consensus: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_submit_block(self, message: dict) -> dict:
        """Submit block to DAGKnight consensus"""
        try:
            block_data = message.get('block')
            if not block_data:
                return {'status': 'error', 'message': 'No block data provided'}

            # Create QuantumBlock from data
            block = QuantumBlock.from_dict(block_data)
            
            # Calculate network latency (simulated or real)
            network_latency = time.time() - block.timestamp

            # Add block to DAGKnight
            confirmation_status = await self.node.dagknight.add_block(
                block, 
                network_latency
            )

            return {
                'status': 'success',
                'confirmation_status': confirmation_status,
                'block_hash': block.hash
            }

        except Exception as e:
            self.logger.error(f"Failed to submit block: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    async def handle_get_block_metrics(self, message: dict) -> dict:
        """Get metrics for a specific block"""
        try:
            block_hash = message.get('block_hash')
            if not block_hash:
                return {'status': 'error', 'message': 'No block hash provided'}

            # Get block metrics from DAGKnight
            metrics = await self.node.dagknight.get_network_status()
            
            # Get specific block metrics
            block_metrics = {
                'confirmation_score': metrics.get('confirmation_scores', {}).get(block_hash, 0),
                'latency': metrics.get('latencies', {}).get(block_hash, 0),
                'security_level': self._determine_security_level(
                    metrics.get('confirmation_scores', {}).get(block_hash, 0)
                ),
                'k_clusters': len(metrics.get('k_clusters', [])),
                'quantum_metrics': {
                    'quantum_strength': metrics.get('quantum_strengths', {}).get(block_hash, 0),
                    'entanglement_count': metrics.get('entanglement_counts', {}).get(block_hash, 0)
                }
            }

            return {
                'status': 'success',
                'metrics': block_metrics
            }

        except Exception as e:
            self.logger.error(f"Failed to get block metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _determine_security_level(self, confirmation_score: float) -> str:
        """Determine security level based on confirmation score"""
        if confirmation_score >= 0.95:
            return "MAXIMUM"
        elif confirmation_score >= 0.85:
            return "VERY_HIGH"
        elif confirmation_score >= 0.75:
            return "HIGH"
        elif confirmation_score >= 0.60:
            return "MEDIUM"
        else:
            return "LOW"
import systemd.daemon
import systemd.journal
import os
import pwd
import grp
from typing import Optional
import socket
import fcntl
import struct

class LinuxQuantumNode(P2PNode):
    """Enhanced P2PNode with Linux-specific optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.systemd_journal = systemd.journal.JournalHandler()
        self.logger.addHandler(self.systemd_journal)
        self.socket_path = "/var/run/quantum_dagknight.sock"
        self.pid_file = "/var/run/quantum_dagknight.pid"
        
    async def start(self):
        """Enhanced start method with Linux system integration"""
        # Drop privileges if running as root
        if os.geteuid() == 0:
            self._drop_privileges('quantum_node')
            
        # Create Unix domain socket for local communication
        self.unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.unix_socket.bind(self.socket_path)
        self.unix_socket.listen(5)
        
        # Set up process monitoring
        self._write_pid_file()
        
        # Enable systemd socket activation
        self._setup_systemd_socket()
        
        # Set real-time scheduling priority
        self._set_rt_priority()
        
        # Start main node operations
        await super().start()
        
        # Notify systemd about successful startup
        systemd.daemon.notify('READY=1')
        
    def _drop_privileges(self, user: str):
        """Drop root privileges and switch to specified user"""
        try:
            pwd_entry = pwd.getpwnam(user)
            os.setgid(pwd_entry.pw_gid)
            os.setuid(pwd_entry.pw_uid)
            os.umask(0o022)
            self.logger.info(f"Dropped privileges to user {user}")
        except Exception as e:
            self.logger.error(f"Failed to drop privileges: {str(e)}")
            raise
            
    def _setup_systemd_socket(self):
        """Set up systemd socket activation"""
        try:
            # Check if running under systemd
            if os.environ.get('LISTEN_PID', None):
                pid = int(os.environ['LISTEN_PID'])
                if pid == os.getpid():
                    fds = int(os.environ['LISTEN_FDS'])
                    self.systemd_socket = socket.fromfd(3, socket.AF_INET, socket.SOCK_STREAM)
                    self.logger.info("Using systemd socket activation")
        except Exception as e:
            self.logger.error(f"Failed to setup systemd socket: {str(e)}")
            
    def _set_rt_priority(self):
        """Set real-time scheduling priority for quantum operations"""
        try:
            # Import here to avoid issues on non-Linux systems
            import sched_setscheduler
            
            # Set SCHED_FIFO with priority 50
            sched_setscheduler.sched_setscheduler(
                0, 
                sched_setscheduler.SCHED_FIFO, 
                sched_setscheduler.sched_param(50)
            )
            self.logger.info("Set real-time scheduling priority")
        except Exception as e:
            self.logger.warning(f"Failed to set RT priority: {str(e)}")
            
    def _write_pid_file(self):
        """Write PID file for process management"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            self.logger.error(f"Failed to write PID file: {str(e)}")
            
    async def get_network_interface_info(self, interface: str) -> dict:
        """Get network interface information using Linux SIOCGIFADDR"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            info = fcntl.ioctl(
                sock.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', interface[:15].encode('utf-8'))
            )
            ip = socket.inet_ntoa(info[20:24])
            return {'interface': interface, 'ip': ip}
        except Exception as e:
            self.logger.error(f"Failed to get interface info: {str(e)}")
            return {}
            
    def cleanup(self):
        """Enhanced cleanup with Linux-specific resources"""
        try:
            # Remove socket and PID files
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            if os.path.exists(self.pid_file):
                os.unlink(self.pid_file)
                
            # Close systemd journal handler
            self.systemd_journal.close()
            
            # Release RT scheduling
            try:
                import sched_setscheduler
                sched_setscheduler.sched_setscheduler(
                    0, 
                    sched_setscheduler.SCHED_OTHER, 
                    sched_setscheduler.sched_param(0)
                )
            except:
                pass
                
            super().cleanup()
        except Exception as e:
            self.logger.error(f"Error during Linux cleanup: {str(e)}")
import socket
import fcntl
import struct
import os
import subprocess
from typing import Dict, Optional

class NetworkOptimizer:
    def __init__(self, p2p_node: 'P2PNode'):
        self.node = p2p_node
        self.logger = logging.getLogger("NetworkOptimizer")
        self.metrics = {}
        self.last_optimization = None
        self.socket_opts = {
            socket.TCP_NODELAY: 1,
            socket.SO_KEEPALIVE: 1,
            socket.SO_REUSEADDR: 1,
            socket.SO_RCVBUF: 16777216,  # 16MB buffer
            socket.SO_SNDBUF: 16777216
        }

    async def optimize_network(self):
        """Apply network optimizations for quantum P2P communication."""
        try:
            # Configure network interface
            interface = self.get_default_interface()
            if interface:
                self.set_interface_optimizations(interface)
                self.logger.info(f"Applied interface optimizations to {interface}")

            # Set socket options
            if hasattr(self.node, 'server') and self.node.server:
                self.apply_socket_options(self.node.server)
                self.logger.info("Applied socket optimizations")

            # Configure TCP BBR if available
            self.enable_bbr()
            
            # Set process network priority
            self.set_process_priority()
            
            self.last_optimization = time.time()
            self.logger.info("Network optimizations applied successfully")
            
        except Exception as e:
            self.logger.error(f"Error applying network optimizations: {str(e)}")

            
    def get_default_interface(self) -> Optional[str]:
        """Get the default network interface."""
        try:
            route = subprocess.check_output(['ip', 'route', 'show', 'default']).decode()
            interface = route.split()[4]
            return interface
        except Exception:
            return None
            
    def set_interface_optimizations(self, interface: str):
        """Set network interface optimizations."""
        try:
            # Set interface txqueuelen
            os.system(f'ip link set {interface} txqueuelen 10000')
            
            # Enable jumbo frames if supported
            os.system(f'ip link set {interface} mtu 9000')
            
            # Disable TCP segmentation offload for better latency
            os.system(f'ethtool -K {interface} tso off gso off')
            
        except Exception as e:
            self.logger.error(f"Error setting interface optimizations: {str(e)}")
            
    def apply_socket_options(self, sock: socket.socket):
        """Apply optimized socket options."""
        for opt, value in self.socket_opts.items():
            try:
                sock.setsockopt(socket.SOL_SOCKET, opt, value)
            except Exception as e:
                self.logger.error(f"Error setting socket option {opt}: {str(e)}")
                
    def enable_bbr(self):
        """Enable TCP BBR congestion control if available."""
        try:
            with open('/proc/sys/net/ipv4/tcp_congestion_control', 'w') as f:
                f.write('bbr')
        except Exception:
            pass
            
    def set_process_priority(self):
        """Set process scheduling priority."""
        try:
            os.nice(-10)  # Set high priority
        except Exception:
            pass

    async def monitor_network_metrics(self):
        """Monitor network performance metrics."""
        while True:
            try:
                metrics = self.collect_network_metrics()
                await self.analyze_metrics(metrics)
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error monitoring network metrics: {str(e)}")
                await asyncio.sleep(60)
                
    def collect_network_metrics(self) -> Dict:
        """Collect network performance metrics."""
        metrics = {
            'latency': {},
            'bandwidth': {},
            'packet_loss': {},
            'connection_stats': {}
        }
        
        # Add metrics collection logic here
        return metrics
    async def analyze_metrics(self, metrics: Dict):
        """Analyze collected network metrics and trigger optimizations if needed."""
        try:
            # Calculate performance indicators
            latency_score = self.calculate_latency_score(metrics['latency'])
            bandwidth_score = self.calculate_bandwidth_score(metrics['bandwidth'])
            packet_loss_score = self.calculate_packet_loss_score(metrics['packet_loss'])
            
            # Overall network health score
            health_score = (latency_score + bandwidth_score + packet_loss_score) / 3
            
            self.logger.info(f"Network Health Score: {health_score:.2f}")
            self.logger.debug(f"Latency Score: {latency_score:.2f}")
            self.logger.debug(f"Bandwidth Score: {bandwidth_score:.2f}")
            self.logger.debug(f"Packet Loss Score: {packet_loss_score:.2f}")
            
            # Trigger optimizations if score is too low
            if health_score < 0.7:
                self.logger.warning(f"Poor network health detected (score: {health_score:.2f})")
                await self.optimize_network()
                
            return health_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing metrics: {str(e)}")
            return 0.0

    def calculate_latency_score(self, latency_metrics: Dict) -> float:
        """Calculate score based on latency metrics (0-1)."""
        if not latency_metrics:
            return 1.0
        avg_latency = sum(latency_metrics.values()) / len(latency_metrics)
        return max(0, min(1, 1 - (avg_latency / 1000)))  # Normalize to 0-1

    def calculate_bandwidth_score(self, bandwidth_metrics: Dict) -> float:
        """Calculate score based on bandwidth metrics (0-1)."""
        if not bandwidth_metrics:
            return 1.0
        avg_bandwidth = sum(bandwidth_metrics.values()) / len(bandwidth_metrics)
        return max(0, min(1, avg_bandwidth / 1000000))  # Normalize to 0-1 (1Gbps = 1.0)

    def calculate_packet_loss_score(self, packet_loss_metrics: Dict) -> float:
        """Calculate score based on packet loss metrics (0-1)."""
        if not packet_loss_metrics:
            return 1.0
        avg_loss = sum(packet_loss_metrics.values()) / len(packet_loss_metrics)
        return max(0, min(1, 1 - (avg_loss * 10)))  # Normalize to 0-1 (10% loss = 0.0)

    async def cleanup(self):
        """Cleanup network optimizations."""
        try:
            # Reset interface settings if necessary
            interface = self.get_default_interface()
            if interface:
                os.system(f'ip link set {interface} txqueuelen 1000')  # Reset to default
                os.system(f'ip link set {interface} mtu 1500')  # Reset to default MTU
            
            # Reset process priority
            os.nice(0)  # Reset to default priority
            
            self.logger.info("Network optimizations cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up network optimizations: {str(e)}")

    def get_default_interface(self) -> Optional[str]:
        """Get the default network interface."""
        try:
            route = subprocess.check_output(['ip', 'route', 'show', 'default']).decode()
            interface = route.split()[4]
            return interface
        except Exception:
            return None
class ConnectionHealthMonitor:
    """Monitors and diagnoses connection health issues."""
    
    def __init__(self, node: 'P2PNode'):
        self.node = node
        self.logger = logging.getLogger("ConnectionHealth")
        self.connection_states = {}
        self.timeout_intervals = {
            'challenge_response': 30,  # 30 seconds for challenge response
            'handshake': 60,          # 60 seconds for complete handshake
            'connection': 120         # 120 seconds for total connection
        }
        self.retry_attempts = {}
        self.MAX_RETRIES = 3

    async def monitor_connection_progress(self, peer: str):
        """Monitor the progress of a connection and handle timeouts."""
        try:
            start_time = time.time()
            connection_completed = False
            
            while time.time() - start_time < self.timeout_intervals['connection']:
                current_state = self.node.peer_states.get(peer)
                
                if not current_state:
                    self.logger.warning(f"No state found for peer {peer}")
                    break
                    
                # Update connection state tracking
                self.connection_states[peer] = {
                    'current_state': current_state,
                    'timestamp': time.time(),
                    'duration': time.time() - start_time
                }
                
                # Check for completion
                if current_state in ['connected', 'quantum_ready']:
                    connection_completed = True
                    self.logger.info(f"Connection successfully completed for {peer}")
                    break
                    
                # Check for timeout based on current state
                if await self.check_state_timeout(peer, current_state):
                    await self.handle_timeout(peer, current_state)
                    break
                    
                await asyncio.sleep(1)
            
            if not connection_completed:
                await self.handle_incomplete_connection(peer)
                
        except Exception as e:
            self.logger.error(f"Error monitoring connection for {peer}: {str(e)}")
            await self.node.remove_peer(peer)

    async def check_state_timeout(self, peer: str, state: str) -> bool:
        """Check if current state has timed out."""
        state_start = self.connection_states[peer]['timestamp']
        current_time = time.time()
        
        timeout_map = {
            'challenge_sent': self.timeout_intervals['challenge_response'],
            'handshake_sent': self.timeout_intervals['handshake'],
            'connecting': self.timeout_intervals['connection']
        }
        
        timeout = timeout_map.get(state, self.timeout_intervals['connection'])
        return (current_time - state_start) > timeout

    async def handle_timeout(self, peer: str, state: str):
        """Handle timeout for specific connection state."""
        try:
            self.logger.warning(f"Timeout in state {state} for peer {peer}")
            
            # Increment retry count
            self.retry_attempts[peer] = self.retry_attempts.get(peer, 0) + 1
            
            if self.retry_attempts[peer] >= self.MAX_RETRIES:
                self.logger.error(f"Max retries exceeded for {peer}")
                await self.node.remove_peer(peer)
                return
            
            # Handle specific states
            if state == 'challenge_sent':
                self.logger.info(f"Resending challenge to {peer}")
                challenge_id = await self.node.send_challenge(peer)
                if not challenge_id:
                    await self.node.remove_peer(peer)
                    
            elif state == 'handshake_sent':
                self.logger.info(f"Resending handshake to {peer}")
                await self.node.send_handshake(peer)
                
            else:
                self.logger.warning(f"Unhandled timeout state: {state}")
                await self.node.remove_peer(peer)
                
        except Exception as e:
            self.logger.error(f"Error handling timeout for {peer}: {str(e)}")
            await self.node.remove_peer(peer)

    async def handle_incomplete_connection(self, peer: str):
        """Handle connection that never completed."""
        try:
            state = self.connection_states.get(peer, {}).get('current_state', 'unknown')
            duration = self.connection_states.get(peer, {}).get('duration', 0)
            
            self.logger.warning(f"Incomplete connection for {peer}:")
            self.logger.warning(f"  Final State: {state}")
            self.logger.warning(f"  Duration: {duration:.1f}s")
            
            # Log connection diagnostics
            await self.log_connection_diagnostics(peer)
            
            # Clean up the failed connection
            await self.node.remove_peer(peer)
            
        except Exception as e:
            self.logger.error(f"Error handling incomplete connection for {peer}: {str(e)}")
            await self.node.remove_peer(peer)

    async def log_connection_diagnostics(self, peer: str):
        """Log diagnostic information for failed connection."""
        try:
            diagnostics = {
                'peer_address': peer,
                'connection_history': self.connection_states.get(peer, {}),
                'retry_attempts': self.retry_attempts.get(peer, 0),
                'peer_public_key': bool(self.node.peer_public_keys.get(peer)),
                'challenges': bool(self.node.challenges.get(peer)),
                'quantum_ready': peer in getattr(self.node, 'quantum_sync', {}).get('entangled_peers', set())
            }
            
            self.logger.info("Connection Diagnostics:")
            for key, value in diagnostics.items():
                self.logger.info(f"  {key}: {value}")
                
        except Exception as e:
            self.logger.error(f"Error logging diagnostics: {str(e)}")

    async def start_monitoring(self, peer: str):
        """Start monitoring a new connection."""
        asyncio.create_task(self.monitor_connection_progress(peer))

class SystemdJournalHandler:
    def __init__(self):
        self.journal_handler = journal.JournalHandler()
        self.journal_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        
    def setup_logging(self, logger_name='quantumdagknight'):
        # Get the root logger
        logger = logging.getLogger(logger_name)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Add systemd journal handler
        logger.addHandler(self.journal_handler)
        
        # Set logging level
        logger.setLevel(logging.INFO)
        
        return logger
        
    def log_startup(self, logger):
        logger.info("QuantumDAGKnight node starting up")
        
    def log_shutdown(self, logger):
        logger.info("QuantumDAGKnight node shutting down")

class LinuxQuantumNode(P2PNode):
    """Enhanced P2P Node with Linux-specific optimizations"""
    
    def __init__(self, blockchain=None, host='localhost', port=8000, security_level=10):
        """Initialize LinuxQuantumNode with required components"""
        try:
            # Initialize parent P2PNode first
            super().__init__(blockchain=blockchain, host=host, port=port, security_level=security_level)
            
            # Initialize sync queue
            self.sync_queue = asyncio.Queue()
            self.logger.info("Sync queue initialized")
            
            # Add systemd journal handler
            self.systemd_journal = systemd.journal.JournalHandler()
            self.logger.addHandler(self.systemd_journal)
            self.logger.info("Systemd journal handler initialized")
            
            # Initialize NetworkOptimizer
            self.network_optimizer = NetworkOptimizer(self)
            self.logger.info("Network optimizer initialized")
            
            # Initialize sync states for components
            self.sync_states = {
                SyncComponent.WALLETS: SyncStatus(),
                SyncComponent.TRANSACTIONS: SyncStatus(),
                SyncComponent.BLOCKS: SyncStatus(),
                SyncComponent.MEMPOOL: SyncStatus()
            }
            self.logger.info("Sync states initialized")
            
            # Socket and PID file paths
            self.socket_path = "/var/run/quantum_dagknight.sock"
            self.pid_file = "/var/run/quantum_dagknight.pid"
            
            # Initialize other required attributes
            self.pending_sync_operations = {}
            self.sync_locks = {}
            self.sync_retry_count = {}
            
            # Start sync queue processing
            self._start_sync_queue_processor()
            
            self.logger.info("LinuxQuantumNode initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing LinuxQuantumNode: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise


        
    async def initialize(self):
        """Initialize node with Linux-specific optimizations"""
        try:
            self.logger.info("\n=== Initializing QuantumDAGKnight Node ===")
            
            # Initialize base components
            await super().initialize()
            
            # Apply network optimizations
            self.logger.info("Applying network optimizations...")
            await self.network_optimizer.optimize_network()
            
            # Initialize quantum components with optimized network
            self.logger.info("Initializing quantum components...")
            await self.initialize_quantum_components()
            
            # Start monitoring tasks
            self.start_monitoring_tasks()
            
            self.logger.info("=== Node Initialization Complete ===\n")
            
        except Exception as e:
            self.logger.error(f"Node initialization failed: {str(e)}")
            raise
            
    def start_monitoring_tasks(self):
        """Start monitoring tasks for node health and performance"""
        self.monitoring_tasks = [
            asyncio.create_task(self.network_optimizer.monitor_network_metrics()),
            asyncio.create_task(self.monitor_quantum_state()),
            asyncio.create_task(self.monitor_system_resources()),
            
            asyncio.create_task(self.periodic_optimization_check())
        ]
        
        for task in self.monitoring_tasks:
            task.add_done_callback(self.handle_task_exception)
            
    def handle_task_exception(self, task):
        """Handle exceptions from monitoring tasks"""
        try:
            exc = task.exception()
            if exc:
                self.logger.error(f"Task {task.get_name()} failed: {str(exc)}")
                # Restart the failed task
                if not task.cancelled():
                    self.restart_monitoring_task(task)
        except asyncio.CancelledError:
            pass
            
    async def restart_monitoring_task(self, failed_task):
        """Restart a failed monitoring task"""
        task_name = failed_task.get_name()
        try:
            if task_name == 'network_metrics':
                new_task = asyncio.create_task(
                    self.network_optimizer.monitor_network_metrics()
                )
            elif task_name == 'quantum_state':
                new_task = asyncio.create_task(self.monitor_quantum_state())
            elif task_name == 'system_resources':
                new_task = asyncio.create_task(self.monitor_system_resources())
            elif task_name == 'optimization_check':
                new_task = asyncio.create_task(self.periodic_optimization_check())
                
            new_task.set_name(task_name)
            new_task.add_done_callback(self.handle_task_exception)
            self.monitoring_tasks.append(new_task)
            
        except Exception as e:
            self.logger.error(f"Failed to restart task {task_name}: {str(e)}")
            
    async def monitor_quantum_state(self):
        """Monitor quantum state with network optimization considerations"""
        while True:
            try:
                # Check quantum state fidelity
                fidelities = {}
                for component in ['wallets', 'transactions', 'blocks', 'mempool']:
                    fidelity = await self.quantum_sync.measure_sync_state(component)
                    fidelities[component] = fidelity
                    
                    # If fidelity is low, check network performance
                    if fidelity < self.quantum_sync.decoherence_threshold:
                        await self.handle_low_fidelity(component, fidelity)
                        
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in quantum state monitoring: {str(e)}")
                await asyncio.sleep(10)
                
    async def handle_low_fidelity(self, component: str, fidelity: float):
        """Handle low quantum fidelity by checking network conditions"""
        try:
            # Get network metrics
            metrics = await self.network_optimizer.collect_network_metrics()
            
            # Check if network issues might be causing decoherence
            if self.detect_network_issues(metrics):
                self.logger.warning(
                    f"Network issues detected affecting {component} fidelity: {fidelity}"
                )
                await self.optimize_for_component(component)
            else:
                # If network is fine, trigger quantum recovery
                await self.recover_quantum_state(component)
                
        except Exception as e:
            self.logger.error(f"Error handling low fidelity: {str(e)}")
            
    def detect_network_issues(self, metrics: Dict) -> bool:
        """Detect network issues that might affect quantum state"""
        try:
            # Check various network metrics
            high_latency = any(lat > 100 for lat in metrics['latency'].values())
            high_packet_loss = any(loss > 0.1 for loss in metrics['packet_loss'].values())
            low_bandwidth = any(bw < 1000000 for bw in metrics['bandwidth'].values())
            
            return high_latency or high_packet_loss or low_bandwidth
            
        except Exception as e:
            self.logger.error(f"Error detecting network issues: {str(e)}")
            return False
            
    async def optimize_for_component(self, component: str):
        """Apply specific optimizations for a quantum component"""
        try:
            if component in ['blocks', 'transactions']:
                # Optimize for higher throughput
                await self.network_optimizer.optimize_for_throughput()
            else:
                # Optimize for lower latency
                await self.network_optimizer.optimize_for_latency()
                
        except Exception as e:
            self.logger.error(f"Error optimizing for {component}: {str(e)}")
            
    async def monitor_system_resources(self):
        """Monitor system resources affecting node performance"""
        while True:
            try:
                resources = self.get_system_resources()
                if self.should_adjust_resources(resources):
                    await self.adjust_resource_usage(resources)
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {str(e)}")
                await asyncio.sleep(30)
                
    async def periodic_optimization_check(self):
        """Periodically check and adjust optimizations"""
        while True:
            try:
                metrics = await self.network_optimizer.collect_network_metrics()
                if self.needs_optimization(metrics):
                    await self.network_optimizer.optimize_network()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in optimization check: {str(e)}")
                await asyncio.sleep(300)

    def needs_optimization(self, metrics: Dict) -> bool:
        """Determine if network needs optimization based on metrics"""
        try:
            # Check if enough time has passed since last optimization
            if self.last_optimization and time.time() - self.last_optimization < 3600:
                return False
                
            # Check metric thresholds
            high_latency = any(lat > 100 for lat in metrics['latency'].values())
            high_packet_loss = any(loss > 0.1 for loss in metrics['packet_loss'].values())
            
            return high_latency or high_packet_loss
            
        except Exception as e:
            self.logger.error(f"Error checking optimization needs: {str(e)}")
            return False
    def initialize_linux_monitoring(self):
        """Initialize Linux-specific monitoring"""
        try:
            # Set up process monitoring
            self._write_pid_file()
            
            # Monitor system resources
            self.start_resource_monitoring()
            
            # Set up network interface monitoring
            self.setup_network_monitoring()
            
        except Exception as e:
            self.logger.error(f"Error initializing Linux monitoring: {str(e)}")
            raise
    
    def _write_pid_file(self):
        """Write PID file for process management"""
        try:
            pid = os.getpid()
            os.makedirs(os.path.dirname(self.pid_file), exist_ok=True)
            with open(self.pid_file, 'w') as f:
                f.write(str(pid))
            self.logger.info(f"PID file written: {self.pid_file}")
        except Exception as e:
            self.logger.error(f"Failed to write PID file: {str(e)}")
            
    def start_resource_monitoring(self):
        """Start monitoring system resources"""
        try:
            import psutil
            self.process = psutil.Process()
            self.logger.info("Resource monitoring initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize resource monitoring: {str(e)}")
            
    def setup_network_monitoring(self):
        """Set up network interface monitoring"""
        try:
            default_interface = self.network_optimizer.get_default_interface()
            if default_interface:
                self.monitored_interface = default_interface
                self.logger.info(f"Network monitoring set up for interface: {default_interface}")
        except Exception as e:
            self.logger.error(f"Failed to setup network monitoring: {str(e)}")
    
    async def start(self):
        """Enhanced start method with Linux-specific features"""
        try:
            # Drop privileges if running as root
            if os.geteuid() == 0:
                self._drop_privileges('quantum_node')
            
            # Apply network optimizations before starting
            await self.network_optimizer.optimize_network()
            
            # Create Unix domain socket
            self.create_unix_socket()
            
            # Notify systemd about startup
            systemd.daemon.notify('READY=1')
            
            # Start the node
            await super().start()
            
        except Exception as e:
            self.logger.error(f"Error starting LinuxQuantumNode: {str(e)}")
            raise
    
    def _drop_privileges(self, user: str):
        """Drop root privileges and switch to specified user"""
        try:
            pwd_entry = pwd.getpwnam(user)
            os.setgid(pwd_entry.pw_gid)
            os.setuid(pwd_entry.pw_uid)
            os.umask(0o022)
            self.logger.info(f"Dropped privileges to user {user}")
        except Exception as e:
            self.logger.error(f"Failed to drop privileges: {str(e)}")
            raise
            
    def create_unix_socket(self):
        """Create Unix domain socket for local communication"""
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
            self.unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.unix_socket.bind(self.socket_path)
            self.unix_socket.listen(5)
            self.logger.info(f"Unix domain socket created: {self.socket_path}")
        except Exception as e:
            self.logger.error(f"Failed to create Unix socket: {str(e)}")
            raise
    def _start_sync_queue_processor(self):
        """Start the sync queue processor"""
        try:
            asyncio.create_task(self.process_sync_queue())
            self.logger.info("Sync queue processor started")
        except Exception as e:
            self.logger.error(f"Failed to start sync queue processor: {str(e)}")
            raise

    async def process_sync_queue(self):
        """Process items in the sync queue"""
        while True:
            try:
                # Get sync operation from queue
                sync_op = await self.sync_queue.get()
                
                # Process the sync operation
                await self.perform_sync_operation(sync_op)
                
                # Mark task as done
                self.sync_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing sync queue: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before next attempt

    async def perform_sync_operation(self, sync_op: dict):
        """Perform a specific sync operation"""
        try:
            component = sync_op['component']
            peer = sync_op['peer']
            
            # Check if we're already syncing this component
            if self.sync_states[component].is_syncing:
                self.logger.debug(f"Sync already in progress for {component}")
                return
            
            # Set sync state
            self.sync_states[component].is_syncing = True
            self.sync_states[component].last_sync = time.time()
            
            try:
                # Perform appropriate sync based on component
                if component == SyncComponent.BLOCKS:
                    await self.sync_blocks_with_peer(peer)
                elif component == SyncComponent.WALLETS:
                    await self.sync_wallets_with_peer(peer)
                elif component == SyncComponent.TRANSACTIONS:
                    await self.sync_transactions_with_peer(peer)
                elif component == SyncComponent.MEMPOOL:
                    await self.sync_mempool_with_peer(peer)
                
                # Update sync state
                self.sync_states[component].is_syncing = False
                self.sync_states[component].last_sync = time.time()
                self.sync_states[component].sync_progress = 100
                
            except Exception as sync_error:
                self.logger.error(f"Error syncing {component} with {peer}: {str(sync_error)}")
                # Update retry count
                self.sync_retry_count[component] = self.sync_retry_count.get(component, 0) + 1
                # Reset sync state
                self.sync_states[component].is_syncing = False
                
        except Exception as e:
            self.logger.error(f"Error performing sync operation: {str(e)}")

    async def queue_sync_operation(self, peer: str, component: str):
        """Queue a sync operation"""
        try:
            if not self.sync_states[component].is_syncing:
                await self.sync_queue.put({
                    'peer': peer,
                    'component': component,
                    'timestamp': time.time()
                })
                self.logger.debug(f"Queued sync operation for {component} with {peer}")
                
        except Exception as e:
            self.logger.error(f"Error queuing sync operation: {str(e)}")

    async def cleanup(self):
        """Enhanced cleanup with sync queue handling"""
        try:
            # Wait for sync queue to be empty
            if hasattr(self, 'sync_queue'):
                try:
                    await self.sync_queue.join()
                except Exception as sync_error:
                    self.logger.error(f"Error waiting for sync queue: {str(sync_error)}")
            
            # Perform standard cleanup
            await super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during LinuxQuantumNode cleanup: {str(e)}")
            raise


# Usage example
async def main():
    blockchain = ...  # Initialize your blockchain object here
    node = P2PNode(host='localhost', port=8000, blockchain=blockchain)

    # Register the computation function
    node.computation_system.register_function(example_computation)
    
    # Start the node
    await node.run()
    
    # Submit a computation task
    task_id = await node.submit_computation("example_computation", 5, 3)
    await node.store("test_key", "test_value", "owner1")
    
    # Retrieve the data
    value = await node.get("test_key", "owner1")
    print(f"Retrieved value: {value}")
    
    # Attempt unauthorized access
    unauthorized_value = await node.get("test_key", "unauthorized_user")
    print(f"Unauthorized access result: {unauthorized_value}")

    # Wait for the result
    while True:
        task = await node.get_computation_result(task_id)
        if task and task.status == "completed":
            print(f"Computation result: {task.result}")
            break
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
