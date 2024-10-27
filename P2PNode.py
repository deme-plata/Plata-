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
    receiver: Optional[str] = None  # Added receiver field
    challenge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Add this line

    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        data = json.loads(json_str)
        # Ensure 'payload' is present in the data, otherwise provide a default value
        if 'payload' not in data:
            data['payload'] = {}
        return cls(
            type=data.get('type', ""),
            payload=data.get('payload', {}),
            timestamp=data.get('timestamp', time.time()),
            sender=data.get('sender', None),
            receiver=data.get('receiver', None),  # Handle receiver
            challenge_id=data.get('challenge_id', str(uuid.uuid4()))
        )

class P2PNode:
    def __init__(self, blockchain, host='localhost', port=8000, security_level=10, k: int = 20):
        load_dotenv()  # Load environment variables from .env file

        self.host = os.getenv('P2P_HOST', host)
        self.port = int(os.getenv('P2P_PORT', port))
        self.blockchain = blockchain
        self.security_level = int(os.getenv('SECURITY_LEVEL', security_level))
        self.max_peers = int(os.getenv('MAX_PEERS', '10'))
        self.field_size = calculate_field_size(self.security_level)
        self.zk_system = SecureHybridZKStark(self.security_level)
        self.peers: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.peer_lock = asyncio.Lock()  # Change to asyncio.Lock()
        self.peer_lock = Lock()
        self.message_queue = asyncio.Queue()
        self.last_broadcast: Dict[str, float] = {}
        self.mempool: List[Transaction] = []
        self.seen_messages: Set[str] = set()
        self.message_expiry = int(os.getenv('MESSAGE_EXPIRY', '300'))
        self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '30'))
        self.server = None
        self.last_challenge = {}
        self.pending_challenges = {}
        self.peer_states = {}
        self.peer_tasks = {}

        # Initialize ws_host and ws_port before using them
        ws_host = os.getenv('WS_HOST', 'localhost')
        ws_port = int(os.getenv('WS_PORT', '8080'))

        # WebSocket updates handler
        self.ws_updates = ExchangeWebSocketUpdates(ws_host, ws_port)

        # Kademlia-like DHT
        self.node_id = self.generate_node_id()
        self.k = k
        self.buckets: List[List[KademliaNode]] = [[] for _ in range(160)]  # 160 bits for SHA-1
        self.data_store: OrderedDict[str, str] = OrderedDict()
        self.max_data_store_size = int(os.getenv('MAX_DATA_STORE_SIZE', '1000'))

        # Magnet link
        self.magnet_link = self.generate_magnet_link()

        self.computation_system = DistributedComputationSystem(self)
        self.encryption_key = self.generate_encryption_key()
        self.access_control = {}  # Dictionary to store access rights
        self.zkp_system = SecureHybridZKStark(self.security_level)
        
        # Parse bootstrap and seed nodes
        self.bootstrap_nodes = [node for node in os.getenv('BOOTSTRAP_NODES', '').split(',') if node]
        self.seed_nodes = [node for node in os.getenv('SEED_NODES', '').split(',') if node]

        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

        self.peer_public_keys = {}
        self.token_logos = {}  # Store token logos

        # STUN and TURN server configuration
        self.stun_server = os.getenv('STUN_SERVER')
        self.turn_server = os.getenv('TURN_SERVER')
        self.turn_username = os.getenv('TURN_USERNAME')
        self.turn_password = os.getenv('TURN_PASSWORD')

        # Initialize external IP and port (to be set later)
        
        self.external_port = None
        self.external_ip = requests.get('https://api.ipify.org').text
        self.challenges = {}
        self.node_directory = NodeDirectory(self)  # Pass the current P2PNode instance
        self.logger = logging.getLogger('P2PNode')
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs; adjust as needed

        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - P2PNode - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        self.blockchain = blockchain
        self.heartbeat_interval = 15  # seconds
        self.reconnect_interval = 30  # seconds
        self.max_reconnect_attempts = 5

        self.is_running = False
        self.peer_locks = {}  # Add this line
        self.last_activity_time = time.time()  # Track last activity (ping/pong or message)
        self.pending_block_proposals: Dict[str, Any] = {}
        self.ws_server = None
        self.ws_clients = set()
        self.subscription_topics = {
            'new_wallet': set(),
            'private_transaction': set()
        }
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.peers = {}  # Dictionary to store active peer connections
        self.last_peer_check = time.time()
        self.connected_peers = set()  # Use a set to store unique connected peers


        logger.info(f"P2PNode initialized with host: {self.host}, port: {self.port}")
        logger.info(f"Bootstrap nodes: {self.bootstrap_nodes}")
        logger.info(f"Seed nodes: {self.seed_nodes}")
        

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

    async def complete_connection(self, peer_address: str):
        self.logger.info(f"[COMPLETE] Completing connection with peer {peer_address}")
        async with self.peer_lock:
            self.peer_states[peer_address] = "connected"
            self.connected_peers.add(peer_address)
            
        await self.sync_transactions(peer_address)
        await self.sync_wallets(peer_address)
        self.start_peer_tasks(peer_address)
        await self.send_handshake(peer_address)
        
        self.logger.info(f"[COMPLETE] Connection finalized with peer {peer_address}")
        self.log_peer_status()
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
    async def send_handshake(self, peer):
        try:
            await self.ensure_blockchain()
            handshake_data = {
                "node_id": self.node_id,
                "version": "1.0",
                "blockchain_height": len(self.blockchain.chain)
            }
            message = Message(type=MessageType.HANDSHAKE.value, payload=handshake_data)
            await self.send_message(peer, message)
            self.logger.debug(f"Handshake sent to peer {peer}")
        except Exception as e:
            self.logger.error(f"Error sending handshake to peer {peer}: {str(e)}")
            raise


        except Exception as e:
            self.logger.error(f"Error sending handshake to peer {peer}: {str(e)}")
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
        try:
            self.logger.debug(f"Starting public key exchange with peer {peer}")

            # Parse and normalize peer address
            try:
                peer_ip, peer_port = peer.split(':')
                peer_port = int(peer_port)
                peer_normalized = self.normalize_peer_address(peer_ip, peer_port)
                self.logger.debug(f"Normalized peer address: {peer_normalized}")
            except ValueError:
                self.logger.error(f"Invalid peer address format: {peer}")
                return False

            # Initialize state variables
            public_key_sent = False
            public_key_received = False
            challenge_sent = False
            challenge_received = False
            challenge_response_received = False

            our_challenge_id = None

            # Set a timeout for the entire exchange process
            exchange_timeout = 60  # 60 seconds total for the exchange
            start_time = time.time()

            while not (public_key_received and challenge_response_received):
                if time.time() - start_time > exchange_timeout:
                    self.logger.warning(f"Public key exchange timed out for peer {peer_normalized}")
                    return False

                # Step 1: Send our public key if not already sent
                if not public_key_sent:
                    await self.send_public_key(peer_normalized)
                    public_key_sent = True
                    self.logger.debug(f"Sent our public key to {peer_normalized}")

                # Step 2: Send our challenge if public key is received but challenge not sent
                if public_key_received and not challenge_sent:
                    our_challenge_id = await self.send_challenge(peer_normalized)
                    challenge_sent = True
                    self.logger.debug(f"Sent challenge with ID {our_challenge_id} to {peer_normalized}")

                # Step 3: Wait for the next message from the peer
                try:
                    message = await asyncio.wait_for(self.receive_message(peer_normalized), timeout=15)
                    if message is None:
                        self.logger.warning(f"No message received from {peer_normalized} within timeout period")
                        continue
                    self.logger.debug(f"Received message of type {message.type} from {peer_normalized}")

                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout while waiting for message from {peer_normalized}")
                    continue

                # Step 4: Handle incoming messages based on type
                if message.type == MessageType.PUBLIC_KEY_EXCHANGE.value and not public_key_received:
                    if await self.handle_public_key_exchange(message, peer_normalized):
                        public_key_received = True
                        self.logger.debug(f"Received and validated public key from {peer_normalized}")
                    else:
                        self.logger.error(f"Failed to handle public key from {peer_normalized}")
                        return False

                elif message.type == MessageType.CHALLENGE.value and not challenge_received:
                    await self.handle_challenge(peer_normalized, message.payload, message.challenge_id)
                    challenge_received = True
                    self.logger.debug(f"Received challenge from {peer_normalized}")

                elif message.type == MessageType.CHALLENGE_RESPONSE.value and not challenge_response_received:
                    if await self.verify_challenge_response(peer_normalized, our_challenge_id, message.payload):
                        challenge_response_received = True
                        self.logger.debug(f"Challenge response from {peer_normalized} verified")
                    else:
                        self.logger.error(f"Invalid challenge response from {peer_normalized}")
                        return False

                else:
                    self.logger.warning(f"Unexpected message type from {peer_normalized}: {message.type}")

            # Final success state
            self.logger.info(f"Successfully exchanged keys and completed challenge-response with {peer_normalized}")
            return True

        except Exception as e:
            self.logger.error(f"Error during key exchange with peer {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False


    async def send_public_key(self, peer: str):
        try:
            # Convert the public key to PEM format
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()

            # Log the full public key being sent for better tracking
            self.logger.debug(f"Preparing to send public key to {peer}: {public_key_pem[:100]}...")  # Log first 100 chars

            # Create the public key exchange message
            message = Message(
                type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                payload={"public_key": public_key_pem}
            )

            # Send the public key message
            if peer in self.peers:
                await self.send_raw_message(peer, message)
                self.logger.debug(f"Sent public key to peer {peer} via existing peer connection")
            else:
                # Use the WebSocket directly if peer is not already in the peers list
                websocket = self.peers.get(peer)
                if websocket:
                    await websocket.send(message.to_json())
                    self.logger.debug(f"Sent public key to peer {peer} via direct WebSocket connection")
                else:
                    # Log an error if no WebSocket connection is found
                    self.logger.error(f"No WebSocket connection found for peer {peer}")
                    return False

            # Log a successful public key send operation
            self.logger.info(f"Successfully sent public key to peer {peer}")

        except Exception as e:
            # Log any exceptions that occur during the process
            self.logger.error(f"Failed to send public key to {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

        # Helper function to send a challenge
    async def send_challenge(self, peer: str) -> str:
        try:
            peer_normalized = self.normalize_peer_address(peer)
            challenge_id = str(uuid.uuid4())
            challenge = os.urandom(32).hex()

            if peer_normalized not in self.challenges:
                self.challenges[peer_normalized] = {}
            self.challenges[peer_normalized][challenge_id] = challenge

            message = Message(
                type=MessageType.CHALLENGE.value,
                payload={'challenge': f"{challenge_id}:{challenge}"},
                sender=self.node_id,
                receiver=peer_normalized,
                challenge_id=challenge_id
            )

            await self.send_message(peer_normalized, message)
            logger.debug(f"Sent challenge with ID {challenge_id} to {peer_normalized}")
            return challenge_id

        except Exception as e:
            logger.error(f"Error sending challenge to {peer}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

        # Helper function to send a challenge
    async def send_challenge(self, peer: str) -> str:
        try:
            peer_ip, peer_port = peer.split(':')
            peer_port = int(peer_port)
        except ValueError:
            logger.error(f"Invalid peer address format: {peer}")
            return None

        normalized_peer = self.normalize_peer_address(peer_ip, peer_port)

        challenge_id = str(uuid.uuid4())
        challenge = os.urandom(32).hex()

        if normalized_peer not in self.challenges:
            self.challenges[normalized_peer] = {}
        self.challenges[normalized_peer][challenge_id] = challenge

        message = Message(
            type=MessageType.CHALLENGE.value,
            payload={'challenge': f"{challenge_id}:{challenge}"},
            sender=self.node_id,
            receiver=normalized_peer,
            challenge_id=challenge_id
        )

        await self.send_message(normalized_peer, message)
        logger.debug(f"Sent challenge with ID {challenge_id} to {normalized_peer}")
        return challenge_id



    def generate_node_id(self) -> str:
        return hashlib.sha1(f"{self.host}:{self.port}".encode()).hexdigest()

    def generate_magnet_link(self) -> MagnetLink:
        info_hash = self.node_id
        peer_id = base64.b64encode(f"{self.host}:{self.port}".encode()).decode()
        return MagnetLink(info_hash, [], peer_id)
    async def start(self):
        try:
            # Step 1: Start the WebSocket server
            self.server = await websockets.serve(self.handle_connection, self.host, self.port)
            logger.info(f"P2P node started on {self.host}:{self.port}")

            # Step 2: Join the network (if applicable)
            await self.join_network()

            # Step 3: Start periodic tasks for managing connections and heartbeats
            asyncio.create_task(self.periodic_tasks())         # Handle periodic operations like cleanups
            asyncio.create_task(self.periodic_peer_check())    # Regularly check the health of peers
            asyncio.create_task(self.send_heartbeats())        # Send heartbeats to peers for liveness checks
            asyncio.create_task(self.periodic_connection_check()) # Check connections periodically
            asyncio.create_task(self.periodic_sync())
            asyncio.create_task(self.update_active_peers())  # Add this line

            
            # Step 4: Start WebSocket updates handling
            await self.ws_updates.start()
            self.ws_updates.register_handler('subscribe_orderbook', self.handle_subscribe_orderbook)
            self.ws_updates.register_handler('subscribe_trades', self.handle_subscribe_trades)

        except Exception as e:
            # Log errors during node startup
            logger.error(f"Failed to start P2P node: {str(e)}")
            logger.error(traceback.format_exc())


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
    async def join_network(self):
        logger.info("Attempting to join the network...")
        connected = False

        for node in self.bootstrap_nodes + self.seed_nodes:
            if node:
                try:
                    ip, port = node.split(':')
                    node_id = self.generate_node_id()
                    kademlia_node = KademliaNode(node_id, ip, int(port))
                    
                    logger.info(f"Attempting to connect to node: {node}")
                    connection_result = await self.connect_to_peer(kademlia_node)
                    
                    if connection_result:
                        connected = True
                        logger.info(f"Successfully connected to node: {node}")
                        # Add the connected peer to the active list
                        self.connected_peers.add(f"{ip}:{port}")
                    else:
                        logger.warning(f"Failed to establish connection with node: {node}")

                except Exception as e:
                    logger.error(f"Failed to connect to node {node}. Error: {str(e)}")
                    logger.error(traceback.format_exc())

        if connected:
            logger.info(f"Successfully joined the network. Connected to {len(self.connected_peers)} peers.")
            await self.sync_with_connected_peers()
        else:
            logger.warning("Failed to connect to any nodes. Running as a standalone node.")

        self.log_connected_peers()

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
    async def request_peer_list(self, seed_node: str) -> List[str]:
        message = Message(MessageType.PEER_LIST_REQUEST.value, {})
        response = await self.send_and_wait_for_response(seed_node, message)
        return response.payload.get('peers', [])

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


    async def find_node(self, node_id: str, k: int = 5) -> Optional[List[Dict]]:
        try:
            closest_peers = self.select_closest_peers(node_id, k)
            nodes = []

            for peer in closest_peers:
                if isinstance(peer, dict):
                    peer_address = f"{peer['ip']}:{peer['port']}"
                else:
                    peer_address = peer
                response = await self.send_find_node(node_id, peer_address)
                if response and 'nodes' in response.payload:
                    nodes.extend(response.payload['nodes'])
                else:
                    self.logger.warning(f"No nodes found in response from peer {peer_address}")

            if not nodes:
                self.logger.warning("No nodes found during find_node operation.")
                return None

            self.logger.debug(f"Found nodes: {nodes}")
            return nodes

        except Exception as e:
            self.logger.error(f"Failed to find node: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

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
        try:
            logger.debug(f"Handling message of type {message.type} from {sender}")
            response = None

            # Handle BLOCK message
            if message.type == MessageType.BLOCK.value:
                logger.info(f"[BLOCK] Received BLOCK message from {sender}. Block details: {message.payload}")
                await self.handle_new_block(message.payload, sender)

            # Handle GET_WALLETS message
            elif message.type == MessageType.GET_WALLETS.value:
                logger.info(f"[WALLETS] Received GET_WALLETS request from {sender}.")
                await self.handle_get_wallets(sender)

            # Handle WALLETS message
            elif message.type == MessageType.WALLETS.value:
                logger.info(f"[WALLETS] Received WALLETS message from {sender}. Wallet details: {message.payload}")
                await self.handle_wallets(message.payload, sender)

            # Handle CHALLENGE message
            elif message.type == MessageType.CHALLENGE.value:
                logger.info(f"[CHALLENGE] Received CHALLENGE message from {sender}. Challenge data: {message.payload}")
                await self.handle_challenge(sender, message.payload.get('challenge') or message.payload.get('data'), message.challenge_id)

            # Handle PUBLIC_KEY_EXCHANGE message
            elif message.type == MessageType.PUBLIC_KEY_EXCHANGE.value:
                logger.info(f"[KEY_EXCHANGE] Received PUBLIC_KEY_EXCHANGE message from {sender}. Public key exchange data: {message.payload}")
                await self.handle_public_key_exchange(message, sender)

            # Handle CHALLENGE_RESPONSE message
            elif message.type == MessageType.CHALLENGE_RESPONSE.value:
                logger.info(f"[CHALLENGE_RESPONSE] Received CHALLENGE_RESPONSE message from {sender}. Response data: {message.payload}")
                await self.handle_challenge_response(sender, message.payload)

            # Handle FIND_NODE message
            elif message.type == MessageType.FIND_NODE.value:
                logger.info(f"[FIND_NODE] Received FIND_NODE request from {sender}. Request details: {message.payload}")
                response = await self.handle_find_node(message.payload, sender)

            # Handle FIND_VALUE message
            elif message.type == MessageType.FIND_VALUE.value:
                logger.info(f"[FIND_VALUE] Received FIND_VALUE request from {sender}. Value details: {message.payload}")
                response = await self.handle_find_value(message.payload, sender)

            # Handle STORE message
            elif message.type == MessageType.STORE.value:
                logger.info(f"[STORE] Received STORE message from {sender}. Store details: {message.payload}")
                await self.handle_store(message.payload, sender)

            # Handle TRANSACTION message
            elif message.type == MessageType.TRANSACTION.value:
                logger.info(f"[TRANSACTION] Received TRANSACTION message from {sender}. Transaction details: {message.payload}")
                await self.handle_transaction(message.payload, sender)

            # Handle BLOCK_PROPOSAL message
            elif message.type == MessageType.BLOCK_PROPOSAL.value:
                logger.info(f"[BLOCK_PROPOSAL] Received BLOCK_PROPOSAL message from {sender}. Proposal details: {message.payload}")
                await self.handle_block_proposal(message, sender)

            # Handle FULL_BLOCK_REQUEST message
            elif message.type == MessageType.FULL_BLOCK_REQUEST.value:
                logger.info(f"[FULL_BLOCK_REQUEST] Received FULL_BLOCK_REQUEST from {sender}. Request details: {message.payload}")
                await self.handle_full_block_request(message, sender)

            # Handle FULL_BLOCK_RESPONSE message
            elif message.type == MessageType.FULL_BLOCK_RESPONSE.value:
                logger.info(f"[FULL_BLOCK_RESPONSE] Received FULL_BLOCK_RESPONSE from {sender}. Block details: {message.payload}")
                await self.handle_full_block_response(message, sender)

            # Handle BLOCK_ACCEPTANCE message
            elif message.type == MessageType.BLOCK_ACCEPTANCE.value:
                logger.info(f"[BLOCK_ACCEPTANCE] Received BLOCK_ACCEPTANCE message from {sender}. Acceptance details: {message.payload}")
                await self.handle_block_acceptance(message, sender)

            # Handle STATE_REQUEST message
            elif message.type == MessageType.STATE_REQUEST.value:
                logger.info(f"[STATE_REQUEST] Received STATE_REQUEST from {sender}")
                state = self.blockchain.get_node_state()
                response = Message(type=MessageType.STATE_RESPONSE.value, payload=asdict(state))

            # Handle GET_ALL_DATA message
            elif message.type == MessageType.GET_ALL_DATA.value:
                logger.info(f"Received GET_ALL_DATA request from {sender}")
                all_data = await self.blockchain.get_all_data()
                response = Message(MessageType.ALL_DATA.value, all_data)

            # Handle LOGO_UPLOAD message
            elif message.type == MessageType.LOGO_UPLOAD.value:
                logger.info(f"[LOGO_UPLOAD] Received LOGO_UPLOAD message from {sender}. Logo details: {message.payload}")
                await self.handle_logo_upload(message.payload, sender)

            # Handle GET_TRANSACTIONS message
            elif message.type == MessageType.GET_TRANSACTIONS.value:
                logger.info(f"[GET_TRANSACTIONS] Received GET_TRANSACTIONS message from {sender}")
                await self.handle_get_transactions(sender)

            # Handle UNKNOWN message type
            else:
                logger.warning(f"Received unknown message type from {sender}: {message.type}")

            # If a response is generated, send it back to the sender
            if response:
                response.id = message.id  # Set the response ID to match the request ID
                await self.send_message(sender, response)

        except Exception as e:
            logger.error(f"Failed to handle message from {sender}: {str(e)}")
            logger.error(f"Message content: {message.to_json()}")
            logger.error(f"Traceback: {traceback.format_exc()}")




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
 



    async def handle_handshake(self, payload, sender):
        node_id = payload.get('node_id')
        version = payload.get('version')
        blockchain_height = payload.get('blockchain_height')
        logger.info(f"Received handshake from {sender}: Node ID: {node_id}, Version: {version}, Blockchain Height: {blockchain_height}")
        # You might want to store this information or use it to decide whether to sync
        if blockchain_height > len(self.blockchain.chain):
            await self.sync_blockchain(sender)

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
    async def handle_public_key_exchange(self, message: Message, peer: str):
        try:
            # Log receipt of the public key exchange message
            self.logger.debug(f"Received public key exchange message from {peer}: {message.payload}")
            peer_public_key_pem = message.payload.get("public_key")
            if not peer_public_key_pem:
                self.logger.error(f"Public key missing in the payload from {peer}")
                return False

            peer_normalized = self.normalize_peer_address(peer)
            self.logger.debug(f"Normalized peer address: {peer_normalized}")

            # Attempt to load the received public key
            self.logger.debug(f"Attempting to load public key for {peer}")
            try:
                peer_public_key = serialization.load_pem_public_key(
                    peer_public_key_pem.encode(),
                    backend=default_backend()
                )
                self.logger.debug(f"Successfully loaded public key for {peer}: {type(peer_public_key)}")
            except ValueError as e:
                self.logger.error(f"Failed to load public key for {peer}: {str(e)}")
                return False

            # Ensure the public key is an RSA public key
            if not isinstance(peer_public_key, rsa.RSAPublicKey):
                self.logger.error(f"Received key from {peer} is not an RSA public key")
                return False

            # Store the public key
            self.peer_public_keys[peer_normalized] = peer_public_key
            self.logger.info(f"Public key for {peer_normalized} successfully stored.")

            # Log the type of the stored public key for verification
            stored_key = self.peer_public_keys.get(peer_normalized)
            self.logger.debug(f"Stored public key type for {peer_normalized}: {type(stored_key)}")

            # Dump the current state of peer_public_keys for debugging purposes
            self.dump_peer_public_keys()

            # Send our public key in response to complete the exchange
            our_public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()

            response = Message(
                type=MessageType.PUBLIC_KEY_EXCHANGE.value,
                payload={"public_key": our_public_key_pem}
            )

            # Log the action of sending our public key and send the message
            self.logger.debug(f"Sending our public key to {peer}")
            await self.send_raw_message(peer, response)

            # Log the success of the public key exchange process
            self.logger.info(f"Public key exchange completed successfully with {peer_normalized}")
            return True

        except Exception as e:
            # Log detailed error information including traceback
            self.logger.error(f"Error during public key exchange with {peer}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False


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
        try:
            transaction = Transaction.from_dict(transaction_data)
            
            # Generate and verify ZKP
            secret = transaction.amount  # Or any other secret data
            public_input = int(transaction.hash(), 16)
            proof = self.zkp_system.prove(secret, public_input)
            
            if not self.zkp_system.verify(public_input, proof):
                logger.warning(f"Invalid ZKP for transaction from {sender}")
                return

            if await self.blockchain.add_transaction(transaction):
                await self.broadcast(Message(MessageType.TRANSACTION.value, transaction_data), exclude=sender)
        except Exception as e:
            logger.error(f"Failed to handle transaction: {str(e)}")
            logger.error(traceback.format_exc())



    async def handle_block(self, block_data: dict, sender: str):
        logger.info(f"Received new block from {sender}")
        try:
            new_block = QuantumBlock.from_dict(block_data)
            logger.debug(f"Deserialized block: {new_block.to_dict()}")
            
            if self.blockchain.validate_block(new_block):
                logger.info(f"Block {new_block.hash} is valid. Adding to blockchain.")
                await self.blockchain.add_block(new_block)
                logger.info(f"Block {new_block.hash} successfully added to blockchain")
                
                # Propagate the block to other peers
                await self.propagate_block(new_block)
            else:
                logger.warning(f"Received invalid block from {sender}: {new_block.hash}")
        except Exception as e:
            logger.error(f"Error processing block from {sender}: {str(e)}")
            logger.error(traceback.format_exc())



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
        self.logger.info(f"Broadcasting message of type {message.type}")
        self.logger.debug(f"Current connected peers: {self.connected_peers}")

        """Broadcast a message to all active peers with a retry mechanism."""
        retries = 0
        success = False

        while retries < max_retries and not success:
            # Log all connected peers
            self.logger.debug(f"All connected peers: {self.connected_peers}")

            # If no active peers, log and return early
            if not self.connected_peers:
                self.logger.warning("No active peers available for broadcasting.")
                self.log_peer_status()  # Log peer statuses when no active peers are found
                return
            
            # Log attempt details
            self.logger.info(f"Attempting to broadcast message of type {message.type} to {len(self.connected_peers)} active peers (Attempt {retries + 1})")

            # Initialize successful broadcast counter
            successful_broadcasts = 0

            # Loop through active peers and attempt to send messages
            for peer_address in self.connected_peers.copy():  # Use .copy() to avoid modifying set during iteration
                try:
                    # Send the message to the peer
                    await self.send_message(peer_address, message)
                    successful_broadcasts += 1
                    self.logger.info(f"Message of type {message.type} successfully sent to peer {peer_address}")
                except Exception as e:
                    # If sending the message fails, log the error and remove the peer
                    self.logger.error(f"Failed to send message to peer {peer_address}: {str(e)}")
                    self.connected_peers.remove(peer_address)
            
            # Log the success rate of the broadcast
            self.logger.info(f"Successfully broadcasted message to {successful_broadcasts} out of {len(self.connected_peers)} active peers (Attempt {retries + 1})")

            # Log current connected peers after broadcast
            self.log_connected_peers()

            # Check if all peers acknowledged the message
            if successful_broadcasts == len(self.connected_peers):
                success = True
                self.logger.info("Message broadcasted to all peers successfully.")
            else:
                self.logger.warning(f"Some peers failed to receive the message. {len(self.connected_peers) - successful_broadcasts} peers did not acknowledge.")

                # Retry mechanism if broadcast was unsuccessful
                if successful_broadcasts == 0:
                    retries += 1
                    self.logger.warning(f"No peers acknowledged the broadcast. Retrying broadcast in {retry_delay} seconds (Attempt {retries}/{max_retries})...")
                    await asyncio.sleep(retry_delay)

        if not success:
            self.logger.error(f"Broadcast failed after {max_retries} retries.")



    async def is_peer_connected(self, peer):
        try:
            websocket = self.peers.get(peer)
            if websocket and websocket.open:
                await asyncio.wait_for(websocket.ping(), timeout=5)
                return True
        except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
            pass
        return False
    def cleanup_challenges(self):
        current_time = time.time()
        for peer in list(self.challenges.keys()):
            for challenge_id in list(self.challenges[peer].keys()):
                if current_time - self.challenges[peer][challenge_id]['timestamp'] > 300:  # 5 minutes timeout
                    del self.challenges[peer][challenge_id]
            if not self.challenges[peer]:
                del self.challenges[peer]
    async def periodic_cleanup(self):
        while True:
            self.cleanup_challenges()
            await asyncio.sleep(60)  # Run every minute
    async def send_message(self, peer_normalized: str, message: Message):
        """Send an encrypted message to a peer."""
        try:
            # Check if peer is known
            if peer_normalized not in self.peers and peer_normalized not in [node.address for bucket in self.buckets for node in bucket]:
                self.logger.warning(f"Attempted to send message to unknown peer: {peer_normalized}")
                return

            # Check if the peer is still connected or establish a new connection
            if not await self.ensure_peer_connection(peer_normalized):
                self.logger.warning(f"Failed to establish connection with peer {peer_normalized}. Removing from peer list.")
                await self.remove_peer(peer_normalized)
                return

            # Get the public key of the recipient (peer)
            recipient_public_key = await self.get_peer_public_key(peer_normalized)
            if not recipient_public_key:
                self.logger.error(f"Missing public key for peer {peer_normalized}. Removing peer.")
                await self.remove_peer(peer_normalized)
                return

            # Encrypt the message before sending
            message_json = message.to_json()
            self.logger.debug(f"Message to encrypt: {message_json[:100]}...")  # Log first 100 characters for debugging
            encrypted_message = self.encrypt_message(message_json, recipient_public_key)
            self.logger.debug(f"Encrypted message length: {len(encrypted_message)}")

            # Send the encrypted message via WebSocket
            await self.peers[peer_normalized].send(encrypted_message)
            self.logger.info(f"Encrypted message of type {message.type} sent to peer {peer_normalized}")

        except websockets.exceptions.ConnectionClosed as closed_exception:
            self.logger.warning(f"Connection to peer {peer_normalized} closed: {str(closed_exception)}. Removing peer.")
            await self.remove_peer(peer_normalized)
        except Exception as e:
            # Handle unexpected errors, log the exception, and remove the peer
            self.logger.error(f"Failed to send message to peer {peer_normalized}: {str(e)}")
            self.logger.error(traceback.format_exc())
            await self.remove_peer(peer_normalized)

        # Log the current state of peers after the operation
        self.logger.debug(f"Current peers after send_message: {list(self.peers.keys())}")

    async def ensure_peer_connection(self, peer_normalized: str) -> bool:
        """Ensure that a connection exists with the peer, establishing one if necessary."""
        if peer_normalized in self.peers and await self.is_peer_connected(peer_normalized):
            return True

        try:
            # If the peer is not connected, try to establish a new connection
            websocket = await websockets.connect(f"ws://{peer_normalized}", timeout=10)
            self.peers[peer_normalized] = websocket
            
            # Perform handshake and key exchange
            if not await self.perform_handshake(peer_normalized):
                self.logger.warning(f"Handshake failed with peer {peer_normalized}")
                await self.remove_peer(peer_normalized)
                return False

            self.logger.info(f"Established new connection with peer {peer_normalized}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to establish connection with peer {peer_normalized}: {str(e)}")
            return False


    async def send_raw_message(self, peer: str, message: Message):
        if peer in self.peers:
            try:
                await self.peers[peer].send(message.to_json())
                logger.debug(f"Raw message sent to peer {peer}: {message.to_json()[:100]}...")  # Log first 100 chars
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"Failed to send raw message to peer {peer}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error sending raw message to peer {peer}: {str(e)}")
                logger.error(traceback.format_exc())

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

            
    async def wait_for_challenge_response(self, peer_address: str, challenge_id: str, timeout: float = 60.0) -> bool:
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                message = await self.receive_message(peer_address)
                if message and message.type == MessageType.CHALLENGE_RESPONSE.value:
                    if message.challenge_id == challenge_id:
                        return await self.verify_challenge_response(peer_address, challenge_id, message.payload)
                await asyncio.sleep(0.1)
            logger.warning(f"Timeout waiting for challenge response from {peer_address}. Elapsed time: {time.time() - start_time:.2f}s")
            return False
        except Exception as e:
            logger.error(f"Error waiting for challenge response from {peer_address}: {str(e)}")
            return False
    async def receive_message(self, peer: str, timeout: float = 300.0) -> Optional[Message]:
        peer_normalized = self.normalize_peer_address(peer)

        # Initialize the lock for this peer if not already present
        if peer_normalized not in self.peer_locks:
            self.peer_locks[peer_normalized] = Lock()

        # Acquire the peer-specific lock to avoid race conditions
        async with self.peer_locks[peer_normalized]:
            if peer_normalized in self.peers:
                try:
                    while True:
                        try:
                            # Attempt to receive the message with a timeout
                            raw_message = await asyncio.wait_for(self.peers[peer_normalized].recv(), timeout=timeout)

                            if not raw_message:
                                self.logger.warning(f"Received empty message from {peer_normalized}")
                                return None

                            try:
                                # Try to parse the message as JSON (assuming it might not be encrypted)
                                message = Message.from_json(raw_message)
                                if message.type in [MessageType.PUBLIC_KEY_EXCHANGE.value, MessageType.CHALLENGE.value, MessageType.CHALLENGE_RESPONSE.value]:
                                    # These message types are not encrypted, return the message directly
                                    return message
                            except json.JSONDecodeError:
                                # If it's not valid JSON, assume it's encrypted
                                pass

                            # For other message types, decrypt the message if we have the peer's public key
                            if peer_normalized in self.peer_public_keys:
                                try:
                                    decrypted_message = self.decrypt_message(raw_message)
                                    return Message.from_json(decrypted_message)
                                except ValueError as decrypt_error:
                                    self.logger.error(f"Failed to decrypt message from {peer_normalized}: {str(decrypt_error)}")
                                    return None
                            else:
                                self.logger.error(f"No public key found for peer {peer_normalized}")
                                return None

                        except asyncio.TimeoutError:
                            # Instead of disconnecting, send a keep-alive message
                            await self.send_keepalive(peer_normalized)
                            continue  # Continue waiting for a message

                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning(f"Connection closed with {peer_normalized}")
                    await self.remove_peer(peer_normalized)
                except Exception as e:
                    self.logger.error(f"Unexpected error receiving message from {peer_normalized}: {str(e)}")
                    await self.remove_peer(peer_normalized)

            return None

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



    async def handle_public_key_exchange(self, message: Message, peer: str):
        try:
            logger.debug(f"Received public key exchange message from {peer}: {message.payload}")
            peer_public_key_pem = message.payload["public_key"]
            peer_normalized = self.normalize_peer_address(peer)

            logger.debug(f"Attempting to load public key for {peer}")
            try:
                peer_public_key = serialization.load_pem_public_key(
                    peer_public_key_pem.encode(),
                    backend=default_backend()
                )
                logger.debug(f"Successfully loaded public key for {peer}: {type(peer_public_key)}")
            except ValueError as e:
                logger.error(f"Failed to load public key for {peer}: {str(e)}")
                return False

            if not isinstance(peer_public_key, rsa.RSAPublicKey):
                logger.error(f"Received key from {peer} is not an RSA public key")
                return False

            # Store the public key
            self.peer_public_keys[peer_normalized] = peer_public_key
            logger.info(f"Public key for {peer_normalized} successfully stored.")

            # Log the type of the stored public key
            stored_key = self.peer_public_keys.get(peer_normalized)
            logger.debug(f"Stored public key type for {peer_normalized}: {type(stored_key)}")

            # Dump the current state of peer_public_keys
            self.dump_peer_public_keys()

            # Mark the connection as ready for encrypted communication
            self.peer_states[peer_normalized] = "ready"

            logger.info(f"Public key exchange completed successfully with {peer_normalized}")
            return True

        except Exception as e:
            logger.error(f"Error in handle_public_key_exchange: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    def decrypt_message(self, encrypted_message: str) -> str:
        try:
            # Decode the base64-encoded encrypted message
            encrypted_bytes = base64.b64decode(encrypted_message)
            self.logger.debug(f"Decrypting message of length: {len(encrypted_bytes)} bytes")

            # Define the chunk size (256 bytes for RSA 2048-bit encryption)
            chunk_size = 256
            chunks = [encrypted_bytes[i:i + chunk_size] for i in range(0, len(encrypted_bytes), chunk_size)]
            self.logger.debug(f"Message split into {len(chunks)} chunks, each of size {chunk_size}")

            decrypted_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    # Decrypt each chunk using the private key
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
                except Exception as chunk_error:
                    self.logger.error(f"Error decrypting chunk {i + 1}: {str(chunk_error)}")
                    self.logger.error(f"Chunk {i + 1} content (base64): {base64.b64encode(chunk).decode()}")
                    raise

            # Combine decrypted chunks
            decrypted_message = b''.join(decrypted_chunks).decode('utf-8')

            # Log the successfully decrypted message (first 100 characters)
            self.logger.debug(f"Successfully decrypted message (first 100 chars): {decrypted_message[:100]}...")

            return decrypted_message

        except Exception as e:
            # Log the error and the full base64-encoded encrypted message for debugging
            self.logger.error(f"Decryption failed: {str(e)}")
            self.logger.error(f"Full encrypted message (base64): {encrypted_message}")
            raise ValueError(f"Decryption failed: {str(e)}")
    async def remove_peer(self, peer: str):
        normalized_peer = self.normalize_peer_address(peer)
        self.logger.info(f"[REMOVE] Removing peer: {normalized_peer}")

        async with self.peer_lock:
            if normalized_peer in self.peers:
                try:
                    await self.peers[normalized_peer].close()
                    self.logger.debug(f"[REMOVE] Closed WebSocket connection to peer {normalized_peer}")
                except Exception as e:
                    self.logger.error(f"[REMOVE] Error closing connection to peer {normalized_peer}: {str(e)}")
                
                del self.peers[normalized_peer]
            
            self.peer_states.pop(normalized_peer, None)
            self.peer_public_keys.pop(normalized_peer, None)
            self.pending_challenges.pop(normalized_peer, None)
            self.challenges.pop(normalized_peer, None)
            self.connected_peers.discard(normalized_peer)

        self.logger.info(f"[REMOVE] Peer removed: {normalized_peer}")
        self.log_peer_status()

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

    def is_connected(self):
        if len(self.peers) == 0:
            logger.debug("No peers connected.")
            return False

        # Ensure all peers are actively connected
        for peer_id, peer in self.peers.items():
            if not self.is_peer_active(peer):
                logger.debug(f"Peer {peer_id} is not active.")
                return False
        
        logger.debug("All peers are active and connected.")
        return True

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
    async def connect_to_peer(self, node: KademliaNode) -> bool:
        """
        Connect to a peer, perform key exchange, send a challenge, and update peer states.
        """
        peer_address = f"{node.ip}:{node.port}"
        self.logger.info(f"[CONNECT] Attempting to connect to peer: {peer_address}")

        try:
            # Step 1: Establish WebSocket connection with timeout
            websocket = await asyncio.wait_for(
                websockets.connect(f"ws://{peer_address}", timeout=30),
                timeout=30
            )
            self.logger.info(f"[CONNECT] WebSocket connection established with {peer_address}")

            # Step 2: Lock the peer list and update peer states
            async with self.peer_lock:
                self.peers[peer_address] = websocket
                self.peer_states[peer_address] = "connecting"
                self.connected_peers.add(peer_address)  # Add to connected_peers set
            self.logger.info(f"[CONNECT] Peer {peer_address} added to peers list and connected_peers")

            # Step 3: Perform public key exchange
            if await self.exchange_public_keys(peer_address):
                self.logger.info(f"[CONNECT] Public keys exchanged with {peer_address}")
                self.peer_states[peer_address] = "key_exchanged"

                # Step 4: Send a challenge to the peer
                our_challenge_id = await self.send_challenge(peer_address)
                self.logger.info(f"[CONNECT] Sent challenge with ID {our_challenge_id} to {peer_address}")
                self.peer_states[peer_address] = "challenge_sent"

                # Step 5: Wait for the challenge response
                if await self.wait_for_challenge_response(peer_address, our_challenge_id, timeout=60):
                    self.logger.info(f"[CONNECT] Challenge response verified for peer: {peer_address}")

                    # Step 6: Lock peer list, update peer states, and finalize connection
                    async with self.peer_lock:
                        self.peer_states[peer_address] = "connected"
                        self.logger.debug(f"Attempting to add peer {peer_address} to active peer list")
                        self.connected_peers.add(peer_address)
                        self.logger.debug(f"Peer list after adding: {self.connected_peers}")

                    self.logger.info(f"[CONNECT] Peer {peer_address} successfully connected and added to active list")

                    await self.finalize_connection(peer_address)

                    # Step 7: Start handling messages and keep-alive mechanisms
                    asyncio.create_task(self.handle_messages(websocket, peer_address))
                    asyncio.create_task(self.keep_connection_alive(websocket, peer_address))

                    # Step 8: Sync with connected peers after a successful connection
                    await self.sync_with_connected_peers()
                    self.logger.info(f"Successfully synced with connected peers after connecting to {peer_address}")

                    return True
                else:
                    self.logger.error(f"[CONNECT] Failed to receive valid challenge response from {peer_address}")
            else:
                self.logger.warning(f"[CONNECT] Failed to exchange keys with peer: {peer_address}")

        except Exception as e:
            self.logger.error(f"[CONNECT] Error connecting to peer {peer_address}: {str(e)}")
            self.logger.error(traceback.format_exc())

        # If the connection fails at any point, clean up and remove the peer
        await self.remove_peer(peer_address)
        return False


    async def finalize_connection(self, peer_address: str):
        self.logger.info(f"[FINALIZE] Finalizing connection with peer {peer_address}")
        async with self.peer_lock:
            self.peer_states[peer_address] = "connected"
            logger.debug(f"Attempting to add peer {peer_address} to active peer list")
            self.connected_peers.add(peer_address)
            logger.debug(f"Peer list after adding: {self.connected_peers}")

            if peer_address not in self.peers:
                self.logger.warning(f"[FINALIZE] Peer {peer_address} not found in self.peers. Adding it.")
                # You might need to establish a new WebSocket connection here if it's missing
                websocket = await websockets.connect(f"ws://{peer_address}")
                self.peers[peer_address] = websocket
        
        await self.complete_connection(peer_address)
        self.logger.info(f"[FINALIZE] Connection finalized with peer {peer_address}")
        self.log_peer_status()
        
        # Verify the peer was added correctly
        self.logger.info(f"[FINALIZE] Verification after finalization:")
        self.logger.info(f"  Peer in self.peers: {peer_address in self.peers}")
        self.logger.info(f"  Peer in self.connected_peers: {peer_address in self.connected_peers}")
        self.logger.info(f"  Peer state: {self.peer_states.get(peer_address)}")

    def log_peer_status(self):
        self.logger.info("Current peer status:")
        self.logger.info(f"  All peers: {list(self.peers.keys())}")
        self.logger.info(f"  Peer states: {self.peer_states}")
        self.logger.info(f"  Connected peers: {self.connected_peers}")
        active_peers = [peer for peer in self.connected_peers if self.peer_states.get(peer) == "connected"]
        self.logger.info(f"  Active peers: {active_peers}")
    async def handle_messages(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30)
                    try:
                        parsed_message = Message.from_json(message)
                        await self.process_message(parsed_message, peer)
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse message from {peer}")
                    except Exception as e:
                        self.logger.error(f"Error processing message from {peer}: {str(e)}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"No message received from {peer} in 30 seconds. Sending ping.")
                    try:
                        pong_waiter = await websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                        self.logger.debug(f"Received pong from {peer}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Pong timeout from {peer}")
                        raise websockets.exceptions.ConnectionClosed(1000, "Pong timeout")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed with {peer}")
        finally:
            await self.remove_peer(peer)
            await self.attempt_reconnection(peer)


            return False

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
            
    async def handle_messages(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        try:
            while True:
                try:
                    # Receive message with a 30-second timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=30)
                    
                    try:
                        # Parse the message
                        parsed_message = Message.from_json(message)
                        await self.process_message(parsed_message, peer)
                    except json.JSONDecodeError:
                        self.logger.error(f"Failed to parse message from {peer}")
                    except Exception as e:
                        self.logger.error(f"Error processing message from {peer}: {str(e)}")

                except asyncio.TimeoutError:
                    # No message received in 30 seconds, send a ping
                    self.logger.warning(f"No message received from {peer} in 30 seconds. Sending ping.")
                    
                    try:
                        pong_waiter = await websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                        self.logger.debug(f"Received pong from {peer}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Pong timeout from {peer}")
                        raise websockets.exceptions.ConnectionClosed(1000, "Pong timeout")
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed with {peer}")
        
        finally:
            await self.remove_peer(peer)
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

    async def keep_connection_alive(self, websocket, peer):
        missed_heartbeats = 0
        max_missed_heartbeats = 3
        while True:
            try:
                await asyncio.sleep(20)  # Send heartbeat every 20 seconds
                await websocket.send(json.dumps({"type": "heartbeat", "timestamp": time.time()}))
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10)
                    heartbeat_response = json.loads(response)
                    if heartbeat_response.get("type") == "heartbeat_ack":
                        missed_heartbeats = 0
                        self.last_activity_time = time.time()
                except asyncio.TimeoutError:
                    missed_heartbeats += 1
                    self.logger.warning(f"Missed heartbeat from {peer}, count: {missed_heartbeats}")
                    if missed_heartbeats >= max_missed_heartbeats:
                        self.logger.warning(f"Max missed heartbeats reached for {peer}, closing connection")
                        break
            except Exception as e:
                self.logger.error(f"Error in keep_connection_alive for {peer}: {str(e)}")
                break
        
        await self.handle_disconnection(peer)



    async def is_peer_reachable(self, peer):
        try:
            # Attempt to establish a new connection to the peer
            ip, port = peer.split(':')
            async with websockets.connect(f"ws://{ip}:{port}", timeout=5) as ws:
                await ws.ping()
            return True
        except:
            return False

    async def handle_disconnection(self, peer):
        self.logger.info(f"Handling disconnection for peer {peer}")
        await self.remove_peer(peer)
        
        max_retries = 5
        retry_delay = 5

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
            
            # Exponential backoff
            await asyncio.sleep(retry_delay * (2 ** attempt))
        
        self.logger.warning(f"Failed to reconnect to {peer} after {max_retries} attempts")



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
    async def handle_messages(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        while True:
            try:
                message = await websocket.recv()
                try:
                    parsed_message = Message.from_json(message)
                    await self.process_message(parsed_message, peer)
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse message from {peer}")
                except Exception as e:
                    self.logger.error(f"Error processing message from {peer}: {str(e)}")
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"Connection closed with {peer}, attempting to reconnect")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in handle_messages for {peer}: {str(e)}")
                await asyncio.sleep(1)  # Brief pause before continuing

        await self.handle_disconnection(peer)
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
        try:
            message = Message(MessageType.MAGNET_LINK.value, {"magnet_link": self.magnet_link.to_uri()})
            await self.broadcast(message)
        except Exception as e:
            logger.error(f"Failed to announce to network: {str(e)}")
            logger.error(traceback.format_exc())

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
        """
        Handles the connection lifecycle with a peer, including public key exchange,
        challenge-response handling, keep-alive mechanism, and reconnection attempts.
        """
        peer_ip, peer_port = websocket.remote_address[:2]
        peer_normalized = self.normalize_peer_address(peer_ip, peer_port)
        self.logger.info(f"New incoming connection from {peer_ip}:{peer_port}")

        try:
            # Add the peer to the peer list and set its state to "connecting"
            self.peers[peer_normalized] = websocket
            self.peer_states[peer_normalized] = "connecting"
            self.logger.debug(f"Added peer {peer_normalized} to the peer list")

            # Step 1: Exchange public keys
            if not await self.exchange_public_keys(peer_normalized):
                raise ValueError("Public key exchange failed")
            self.logger.debug(f"Public key exchange completed with {peer_normalized}")

            # Step 2: Send a challenge to the peer
            challenge_id = await self.send_challenge(peer_normalized)
            if not challenge_id:
                raise ValueError("Failed to send challenge")
            self.logger.debug(f"Sent challenge to {peer_normalized}")

            # Step 3: Wait for and handle challenge messages during the handshake
            handshake_completed = False
            while not handshake_completed:
                message = await self.receive_message(peer_normalized)
                if not message:
                    raise ValueError(f"No message received from {peer_normalized}")

                if message.type == MessageType.CHALLENGE.value:
                    await self.handle_challenge(peer_normalized, message.payload, message.challenge_id)
                    self.logger.debug(f"Handled challenge from {peer_normalized}")
                elif message.type == MessageType.CHALLENGE_RESPONSE.value:
                    if await self.verify_challenge_response(peer_normalized, challenge_id, message.payload):
                        handshake_completed = True
                        self.logger.info(f"Handshake completed successfully with {peer_normalized}")
                        self.peer_states[peer_normalized] = "connected"
                    else:
                        raise ValueError(f"Invalid challenge response from {peer_normalized}")
                else:
                    self.logger.warning(f"Unexpected message type during handshake: {message.type}")
                    await self.handle_message(message, peer_normalized)

            # Step 4: Start the keep-alive task
            keepalive_task = asyncio.create_task(self.keep_connection_alive(peer_normalized))

            # Step 5: Handle regular messages after handshake completion
            async for message in websocket:
                try:
                    parsed_message = Message.from_json(message)
                    if parsed_message.type == MessageType.GET_TRANSACTIONS.value:
                        await self.handle_get_transactions(peer_normalized)
                    else:
                        await self.handle_message(parsed_message, peer_normalized)
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON from {peer_normalized}")
                except Exception as e:
                    self.logger.error(f"Error handling message from {peer_normalized}: {str(e)}")

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"Connection closed with {peer_normalized}: {str(e)}")
            await asyncio.sleep(5)  # Wait before attempting to reconnect
            try:
                # Attempt reconnection
                websocket = await websockets.connect(f"ws://{peer_ip}:{peer_port}")
                self.logger.info(f"Reconnected to {peer_normalized}")
            except Exception as reconnect_error:
                self.logger.error(f"Failed to reconnect to {peer_normalized}: {reconnect_error}")
        except ValueError as e:
            self.logger.error(f"Error during handshake with {peer_normalized}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during connection with {peer_normalized}: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            # Cancel the keep-alive task and ensure peer is removed after disconnection
            if 'keepalive_task' in locals():
                keepalive_task.cancel()
            await self.handle_disconnection(peer_normalized)



    async def perform_handshake(self, websocket, peer):
        try:
            # Increase timeout from 60 to 120 seconds
            exchange_timeout = 120
            start_time = time.time()

            public_key_received = False
            challenge_response_received = False

            # Step 1: Send our public key to the peer
            await self.send_public_key(websocket, peer)
            self.handshake_states[peer] = HandshakeState.PUBLIC_KEY_SENT
            logger.debug(f"Public key sent to {peer}")

            # Step 2: Receive the peer's public key
            while not public_key_received and time.time() - start_time < exchange_timeout:
                peer_public_key = await self.receive_public_key(websocket, peer)
                if peer_public_key:
                    self.peer_public_keys[peer] = peer_public_key
                    self.handshake_states[peer] = HandshakeState.PUBLIC_KEY_RECEIVED
                    public_key_received = True
                    logger.debug(f"Public key received from {peer}")
                await asyncio.sleep(1)

            if not public_key_received:
                logger.warning(f"Public key exchange timed out for peer {peer}")
                return False

            # Step 3: Generate and send a challenge
            challenge = self.generate_challenge()
            await self.send_challenge(websocket, peer, challenge)
            self.handshake_states[peer] = HandshakeState.CHALLENGE_SENT
            logger.debug(f"Challenge sent to {peer}")

            # Step 4: Handle messages during handshake
            while not (public_key_received and challenge_response_received):
                if time.time() - start_time > exchange_timeout:
                    self.logger.warning(f"Public key exchange or challenge-response timed out for peer {peer}")
                    return False

                message = await self.receive_message(peer)
                if not message:
                    raise ValueError(f"No message received from {peer}")

                if message.type == MessageType.CHALLENGE.value:
                    await self.handle_challenge(peer, message.payload, message.challenge_id)
                    logger.debug(f"Handled challenge from {peer}")
                elif message.type == MessageType.CHALLENGE_RESPONSE.value:
                    await self.verify_challenge_response(peer, challenge, message.payload)
                    challenge_response_received = True
                    logger.debug(f"Challenge response received from {peer}")
                else:
                    logger.warning(f"Unexpected message type during handshake: {message.type}")
                    await self.queue_message_for_later(message, peer)

            # Mark handshake as completed
            self.handshake_states[peer] = HandshakeState.COMPLETED
            logger.info(f"Handshake completed successfully with {peer}")
            return True

        except asyncio.TimeoutError:
            logger.error(f"Handshake timed out with {peer}")
            return False
        except Exception as e:
            logger.error(f"Handshake failed with {peer}: {str(e)}")
            logger.error(traceback.format_exc())
            return False


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
        try:
            self.server = await websockets.serve(self.handle_connection, self.host, self.port)
            logger.info(f"P2P node listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            logger.error(traceback.format_exc())
    async def maintain_connections(self):
        while True:
            try:
                for peer in list(self.peers.keys()):
                    if not await self.is_connection_alive(peer):
                        self.logger.warning(f"Connection to {peer} lost, attempting to reconnect")
                        await self.handle_disconnection(peer)
                
                if len(self.peers) < self.target_peer_count:
                    await self.find_and_connect_to_new_peers()
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in maintain_connections: {str(e)}")
                await asyncio.sleep(60)

    async def find_and_connect_to_new_peers(self):
        self.logger.info("[FIND_PEERS] Starting search for new peers")
        try:
            self.logger.info("[FIND_PEERS] Attempting to find and connect to new peers")
            nodes = await self.find_node(self.node_id)
            self.logger.info(f"[FIND_PEERS] Found {len(nodes)} potential new peers")
            for node in nodes:
                self.logger.info(f"[FIND_PEERS] Considering peer: {node.id}")
                if node.id not in self.peers and len(self.peers) < self.target_peer_count:
                    self.logger.info(f"[FIND_PEERS] Attempting to connect to new peer: {node.id}")
                    success = await self.connect_to_peer(node)
                    if success:
                        self.logger.info(f"[FIND_PEERS] Successfully connected to new peer: {node.id}")
                    else:
                        self.logger.warning(f"[FIND_PEERS] Failed to connect to new peer: {node.id}")
                else:
                    self.logger.info(f"[FIND_PEERS] Skipping peer {node.id} (already connected or peer limit reached)")
            self.logger.info(f"[FIND_PEERS] Finished attempt to find new peers. Current peer count: {len(self.peers)}")
        except Exception as e:
            self.logger.error(f"[FIND_PEERS] Error finding new peers: {str(e)}")
            self.logger.error(traceback.format_exc())
        self.logger.info("[FIND_PEERS] Finished search for new peers")

    async def run(self):
        try:
            await self.start_server()
            await self.join_network()
            await self.announce_to_network()
            
            self.logger.info("Starting periodic tasks...")
            tasks = [
                asyncio.create_task(self.process_message_queue()),
                asyncio.create_task(self.periodic_peer_discovery()),
                asyncio.create_task(self.periodic_data_republish()),
                asyncio.create_task(self.send_heartbeats()),
                asyncio.create_task(self.periodic_logo_sync()),
                asyncio.create_task(self.periodic_connection_check()),
                asyncio.create_task(self.maintain_peer_connections()),
                asyncio.create_task(self.connection_monitor()),
                asyncio.create_task(self.maintain_connections()),
                asyncio.create_task(self.periodic_peer_check())
            ]
            
            for i, task in enumerate(tasks):
                self.logger.info(f"Started task {i+1}: {task.get_name()}")
            
            self.logger.info("P2P node is now running")
            
            # Keep the node running
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour, or use some other condition to keep the node running
                
        except Exception as e:
            self.logger.error(f"Error in P2P node run method: {str(e)}")
            self.logger.error(traceback.format_exc())
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
        while self.is_running:
            try:
                peers = list(self.peers.keys())
                for peer in peers:
                    try:
                        await self.send_heartbeat(peer)
                    except Exception as e:
                        logger.error(f"Error sending heartbeat to {peer}: {str(e)}")
                        await self.remove_peer(peer)
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in send_heartbeats: {str(e)}")

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
    async def handle_challenge(self, peer, challenge_message, challenge_id):
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.debug(f"Handling challenge for normalized peer {peer_normalized}")
            self.logger.debug(f"Challenge message: {challenge_message}")
            self.logger.debug(f"Challenge ID: {challenge_id}")

            # Extract challenge data
            if isinstance(challenge_message, dict):
                challenge_data = challenge_message.get('challenge', '')
            elif isinstance(challenge_message, str):
                challenge_data = challenge_message
            else:
                raise ValueError(f"Unexpected challenge message type: {type(challenge_message)}")

            # Extract just the challenge data (remove the challenge ID)
            challenge_data = challenge_data.split(':', 1)[1] if ':' in challenge_data else challenge_data

            self.logger.debug(f"Challenge data extracted: {challenge_data}")

            # Generate signature for the challenge using the private key
            signature = self.private_key.sign(
                challenge_data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            self.logger.debug(f"Generated signature for challenge: {signature.hex()}")

            # Store the challenge data for later verification
            if peer_normalized not in self.challenges:
                self.challenges[peer_normalized] = {}
            self.challenges[peer_normalized][challenge_id] = challenge_data
            self.logger.debug(f"Stored challenge data for peer {peer_normalized} with challenge ID {challenge_id}")

            # Create and send the challenge response message
            response_message = Message(
                type=MessageType.CHALLENGE_RESPONSE.value,
                payload={"signature": signature.hex()},
                challenge_id=challenge_id
            )
            self.logger.debug(f"Challenge response created for {peer_normalized}: {response_message.to_json()}")

            # Send the challenge response to the peer
            await self.send_raw_message(peer, response_message)
            self.logger.info(f"Challenge successfully handled and response sent to peer {peer_normalized}")

        except Exception as e:
            self.logger.error(f"Error handling challenge from peer {peer_normalized}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

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
    async def verify_challenge_response(self, peer: str, challenge_id: str, response_data: dict) -> bool:
        try:
            peer_normalized = self.normalize_peer_address(peer)
            self.logger.debug(f"Verifying challenge response from peer {peer_normalized} for challenge ID {challenge_id}")
            
            if peer_normalized not in self.challenges:
                self.logger.error(f"No pending challenges found for peer {peer_normalized}")
                return False

            challenge = self.challenges[peer_normalized].get(challenge_id)
            if not challenge:
                self.logger.error(f"Challenge ID {challenge_id} not found for peer {peer_normalized}")
                return False

            peer_public_key = self.peer_public_keys.get(peer_normalized)
            if not peer_public_key:
                self.logger.error(f"No public key found for peer {peer_normalized}")
                return False

            signature = bytes.fromhex(response_data['signature'])
            self.logger.debug(f"Challenge: {challenge}")
            self.logger.debug(f"Signature (hex): {signature.hex()}")
            self.logger.debug(f"Public key: {peer_public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()}")

            # Extract just the challenge data (remove the challenge ID)
            challenge_data = challenge.split(':', 1)[1] if ':' in challenge else challenge

            try:
                peer_public_key.verify(
                    signature,
                    challenge_data.encode('utf-8'),
                    padding.PSS(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                self.logger.info(f"Successfully verified challenge response from peer {peer_normalized}")
                del self.challenges[peer_normalized][challenge_id]
                if not self.challenges[peer_normalized]:
                    del self.challenges[peer_normalized]
                return True
            except InvalidSignature:
                self.logger.warning(f"Invalid signature from peer {peer_normalized}")
                self.logger.debug(f"Verification failed for challenge: {challenge_data}")
                self.logger.debug(f"Challenge encoded: {challenge_data.encode('utf-8').hex()}")
                return False

        except Exception as e:
            self.logger.error(f"Error verifying challenge response from peer {peer_normalized}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    async def handle_challenge_response(self, peer: str, message: Message):
        try:
            peer_normalized = self.normalize_peer_address(peer)
            logger.debug(f"Handling challenge response from peer {peer_normalized}")
            logger.debug(f"Response data: {message.payload}")
            
            # Extract the challenge ID from the message
            challenge_id = message.challenge_id
            if not challenge_id:
                raise ValueError(f"Missing challenge ID in response from peer {peer_normalized}")
            
            # Check if the challenge exists for this peer
            if peer_normalized not in self.pending_challenges or challenge_id not in self.pending_challenges[peer_normalized]:
                logger.warning(f"Challenge not found for peer {peer_normalized} with ID {challenge_id}. It may have been already processed.")
                return True  # Return True to indicate successful processing
            
            # Fetch the challenge data
            challenge = self.pending_challenges[peer_normalized][challenge_id]
            
            # Verify the signature with the challenge
            signature = bytes.fromhex(message.payload['signature'])
            peer_public_key = self.peer_public_keys.get(peer_normalized)
            
            if not peer_public_key:
                raise ValueError(f"No public key found for peer {peer_normalized}")
            
            # If the public key is stored as a PEM string, convert it back to an RSAPublicKey object
            if isinstance(peer_public_key, str):
                peer_public_key = serialization.load_pem_public_key(peer_public_key.encode(), backend=default_backend())
            
            # Verify the challenge response signature
            peer_public_key.verify(
                signature,
                challenge.encode(),
                padding.PSS(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            logger.info(f"Successfully verified challenge response from peer {peer_normalized}")
            
            # Remove the challenge after successful verification
            del self.pending_challenges[peer_normalized][challenge_id]
            if not self.pending_challenges[peer_normalized]:
                del self.pending_challenges[peer_normalized]
            
            return True
        
        except InvalidSignature:
            logger.warning(f"Invalid signature from peer {peer_normalized}")
            return False
        except Exception as e:
            logger.error(f"Error handling challenge response from peer {peer}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    async def cleanup_challenges(self):
        while True:
            try:
                current_time = time.time()
                for peer, challenges in list(self.pending_challenges.items()):
                    for challenge_id, challenge_data in list(challenges.items()):
                        if isinstance(challenge_data, dict) and 'timestamp' in challenge_data:
                            if current_time - challenge_data['timestamp'] > 300:  # 5 minutes timeout
                                del challenges[challenge_id]
                        elif isinstance(challenge_data, str):
                            # For backwards compatibility, assume old challenges are expired
                            del challenges[challenge_id]
                    if not challenges:
                        del self.pending_challenges[peer]
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Error in cleanup_challenges: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)
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
