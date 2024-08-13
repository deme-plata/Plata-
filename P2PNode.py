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
from common import QuantumBlock, Transaction, NodeState
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
from common import QuantumBlock, Transaction, NodeState
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

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
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

class NATTraversal:
    def __init__(self, stun_server: str = 'stun.l.google.com:19302', turn_server: str = None, turn_username: str = None, turn_password: str = None):
        self.stun_server = stun_server
        self.turn_server = turn_server
        self.turn_username = turn_username
        self.turn_password = turn_password
        self.upnp = miniupnpc.UPnP()

    async def get_external_ip_stun(self) -> Tuple[Optional[str], Optional[int]]:
        nat_type, external_ip, external_port = stun.get_ip_info(stun_host=self.stun_server)
        return external_ip, external_port

    async def get_external_ip_turn(self) -> Tuple[Optional[str], Optional[int]]:
        # No changes here
        if not self.turn_server:
            return None, None
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'https://{self.turn_server}/turn',
                auth=aiohttp.BasicAuth(self.turn_username, self.turn_password)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('ip'), data.get('port')
        return None, None

    def setup_upnp(self, internal_port: int, external_port: int) -> bool:
        # No changes here
        try:
            self.upnp.discoverdelay = 200
            self.upnp.discover()
            self.upnp.selectigd()
            
            # Try to create a new port mapping
            result = self.upnp.addportmapping(
                external_port,
                'TCP',
                self.upnp.lanaddr,
                internal_port,
                'P2PNode UPnP Mapping',
                ''
            )
            return result
        except Exception as e:
            logger.error(f"UPnP setup failed: {str(e)}")
            return False

    def remove_upnp_mapping(self, external_port: int) -> bool:
        # No changes here
        try:
            return self.upnp.deleteportmapping(external_port, 'TCP')
        except Exception as e:
            logger.error(f"UPnP mapping removal failed: {str(e)}")
            return False


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

@dataclass
class KademliaNode:
    id: str
    ip: str
    port: int
    magnet_link: Optional[MagnetLink] = None
    last_seen: float = field(default_factory=time.time)

    @property
    def address(self):
        return f"{self.ip}:{self.port}"

class MessageType(Enum):
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

@dataclass
class Message:
    type: str
    payload: dict
    timestamp: float = field(default_factory=time.time)
    sender: str = None

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

class P2PNode:
    def __init__(self, blockchain, k: int = 20):
        self.host = os.getenv('P2P_HOST', 'localhost')
        self.port = int(os.getenv('P2P_PORT', '8000'))
        self.blockchain = blockchain
        self.security_level = int(os.getenv('SECURITY_LEVEL', '10'))
        self.max_peers = int(os.getenv('MAX_PEERS', '10'))
        self.field_size = calculate_field_size(self.security_level)
        self.zk_system = SecureHybridZKStark(self.security_level)
        self.peers: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.peer_lock = Lock()
        self.message_queue = asyncio.Queue()
        self.last_broadcast: Dict[str, float] = {}
        self.mempool: List[Transaction] = []
        self.seen_messages: Set[str] = set()
        self.message_expiry = int(os.getenv('MESSAGE_EXPIRY', '300'))
        self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '30'))
        self.server = None

        # Kademlia-like DHT
        self.node_id = self.generate_node_id()
        self.k = k
        self.buckets: List[List[KademliaNode]] = [[] for _ in range(160)]  # 160 bits for SHA-1
        self.data_store: OrderedDict[str, str] = OrderedDict()
        self.max_data_store_size = int(os.getenv('MAX_DATA_STORE_SIZE', '1000'))

        # Magnet link
        self.magnet_link = self.generate_magnet_link()

        # NAT traversal
        self.stun_server = os.getenv('STUN_SERVER', 'stun.l.google.com:19302')
        self.external_ip = None
        self.external_port = None
        self.nat_traversal_obj = NATTraversal(
            stun_server=os.getenv('STUN_SERVER', 'stun.l.google.com:19302'),
            turn_server=os.getenv('TURN_SERVER'),
            turn_username=os.getenv('TURN_USERNAME'),
            turn_password=os.getenv('TURN_PASSWORD')
        )


        self.external_ip = None
        self.external_port = None
        self.computation_system = DistributedComputationSystem(self)
        self.encryption_key = self.generate_encryption_key()
        self.access_control = {}  # Dictionary to store access rights


    def generate_node_id(self) -> str:
        return hashlib.sha1(f"{self.host}:{self.port}".encode()).hexdigest()

    def generate_magnet_link(self) -> MagnetLink:
        info_hash = self.node_id
        peer_id = base64.b64encode(f"{self.host}:{self.port}".encode()).decode()
        return MagnetLink(info_hash, [], peer_id)

    async def start(self):
        try:
            await self.nat_traversal()
            self.server = await websockets.serve(self.handle_connection, self.host, self.port)
            logger.info(f"P2P node started on {self.host}:{self.port}")
            await self.join_network()
            asyncio.create_task(self.periodic_tasks())
        except Exception as e:
            logger.error(f"Failed to start P2P node: {str(e)}")
            logger.error(traceback.format_exc())
    async def nat_traversal(self):
        try:
            # Try STUN first
            self.external_ip, self.external_port = await self.nat_traversal_obj.get_external_ip_stun()
            
            if not self.external_ip or not self.external_port:
                # If STUN fails, try TURN
                self.external_ip, self.external_port = await self.nat_traversal_obj.get_external_ip_turn()
            
            if self.external_ip and self.external_port:
                logger.info(f"External IP:port - {self.external_ip}:{self.external_port}")
            else:
                logger.warning("Failed to obtain external IP and port through STUN or TURN")

            # Try UPnP as a last resort
            if not self.external_ip or not self.external_port:
                upnp_success = self.nat_traversal_obj.setup_upnp(self.port, self.port)
                if upnp_success:
                    self.external_ip = self.nat_traversal_obj.upnp.externalipaddress()
                    self.external_port = self.port
                    logger.info(f"UPnP successful. External IP:port - {self.external_ip}:{self.external_port}")
                else:
                    logger.warning("UPnP setup failed")

        except Exception as e:
            logger.error(f"NAT traversal failed: {str(e)}")
            logger.error(traceback.format_exc())



    async def join_network(self):
        try:
            bootstrap_nodes = os.getenv('BOOTSTRAP_NODES', '').split(',')
            for node in bootstrap_nodes:
                if node:
                    ip, port = node.split(':')
                    await self.connect_to_peer(KademliaNode(self.generate_node_id(), ip, int(port)))
        except Exception as e:
            logger.error(f"Failed to join network: {str(e)}")
            logger.error(traceback.format_exc())

    async def periodic_tasks(self):
        while True:
            try:
                await asyncio.gather(
                    self.refresh_buckets(),
                    self.republish_data(),
                    self.cleanup_data_store(),
                    self.send_heartbeats()
                )
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error during periodic tasks: {str(e)}")
                logger.error(traceback.format_exc())

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
                node.last_seen = time.time()
        except Exception as e:
            logger.error(f"Failed to add node to bucket: {str(e)}")
            logger.error(traceback.format_exc())

    async def find_node(self, node_id: str) -> List[KademliaNode]:
        try:
            closest_nodes = self.get_closest_nodes(node_id)
            already_contacted = set()
            to_contact = set(closest_nodes)

            while to_contact:
                concurrent_lookups = asyncio.gather(*[
                    self.send_find_node(node, node_id) for node in list(to_contact)[:self.k]
                ])
                results = await concurrent_lookups

                to_contact.difference_update(already_contacted)
                already_contacted.update(to_contact)

                for nodes in results:
                    for node in nodes:
                        if node.id not in already_contacted:
                            await self.add_node_to_bucket(node)
                            to_contact.add(node)

                closest_nodes = self.get_closest_nodes(node_id)
                if all(node in already_contacted for node in closest_nodes):
                    break

            return closest_nodes
        except Exception as e:
            logger.error(f"Failed to find node: {str(e)}")
            logger.error(traceback.format_exc())
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

    async def handle_message(self, message: Message, sender: str):
        try:
            if message.type == MessageType.FIND_NODE.value:
                await self.handle_find_node(message.payload, sender)
            elif message.type == MessageType.FIND_VALUE.value:
                await self.handle_find_value(message.payload, sender)
            elif message.type == MessageType.STORE.value:
                await self.handle_store(message.payload, sender)
            elif message.type == MessageType.MAGNET_LINK.value:
                await self.handle_magnet_link(message.payload, sender)
            elif message.type == MessageType.PEER_EXCHANGE.value:
                await self.handle_peer_exchange(message.payload, sender)
            elif message.type == MessageType.TRANSACTION.value:
                await self.handle_transaction(message.payload, sender)
            elif message.type == MessageType.BLOCK.value:
                await self.handle_block(message.payload, sender)
            elif message.type == MessageType.ZK_PROOF.value:
                await self.handle_zk_proof(message.payload, sender)
            elif message.type == MessageType.STATE_REQUEST.value:
                state = self.blockchain.get_node_state()
                response = Message(type=MessageType.STATE_RESPONSE.value, payload=asdict(state))
                await self.send_message(sender, response)
            elif message.type == MessageType.SUBMIT_COMPUTATION.value:
                await self.handle_submit_computation(message.payload, sender)
            elif message.type == MessageType.COMPUTATION_RESULT.value:
                await self.handle_computation_result(message.payload, sender)

            # ... handle other message types ...
        except Exception as e:
            logger.error(f"Failed to handle message: {str(e)}")
            logger.error(traceback.format_exc())
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
            if await self.blockchain.add_transaction(transaction):
                await self.broadcast(Message(MessageType.TRANSACTION.value, transaction_data), exclude=sender)
        except Exception as e:
            logger.error(f"Failed to handle transaction: {str(e)}")
            logger.error(traceback.format_exc())

    async def handle_block(self, block_data: dict, sender: str):
        try:
            block = QuantumBlock.from_dict(block_data)
            if self.blockchain.consensus.validate_block(block):
                self.blockchain.add_block(block)
                await self.broadcast(Message(MessageType.BLOCK.value, block_data), exclude=sender)
        except Exception as e:
            logger.error(f"Failed to handle block: {str(e)}")
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

    async def send_find_node(self, node: KademliaNode, node_id: str) -> List[KademliaNode]:
        try:
            response = await self.send_and_wait_for_response(node.address, Message(MessageType.FIND_NODE.value, {"node_id": node_id}))
            return [KademliaNode(**node_data) for node_data in response.payload.get('nodes', [])]
        except Exception as e:
            logger.error(f"Failed to send find node: {str(e)}")
            logger.error(traceback.format_exc())
            return []

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

    async def ping_node(self, node: KademliaNode) -> bool:
        try:
            await self.send_message(node.address, Message(MessageType.HEARTBEAT.value, {"timestamp": time.time()}))
            return True
        except Exception as e:
            logger.error(f"Failed to ping node: {str(e)}")
            return False

    async def broadcast(self, message: Message, exclude: str = None):
        try:
            for peer in self.peers:
                if peer != exclude:
                    await self.send_message(peer, message)
        except Exception as e:
            logger.error(f"Failed to broadcast message: {str(e)}")
            logger.error(traceback.format_exc())

    async def send_message(self, peer: str, message: Message):
        if peer in self.peers:
            try:
                await self.peers[peer].send(message.to_json())
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"Failed to send message to peer {peer}: {str(e)}")
                await self.remove_peer(peer)

    async def send_and_wait_for_response(self, peer: str, message: Message, timeout: float = 5.0) -> Optional[Message]:
        if peer in self.peers:
            try:
                await self.send_message(peer, message)
                return await asyncio.wait_for(self.receive_message(peer), timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for response from peer {peer}")
            except Exception as e:
                logger.error(f"Error communicating with peer {peer}: {str(e)}")
                await self.remove_peer(peer)
        return None

    async def receive_message(self, peer: str) -> Optional[Message]:
        if peer in self.peers:
            try:
                message = await self.peers[peer].recv()
                return Message.from_json(message)
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"Failed to receive message from peer {peer}: {str(e)}")
                await self.remove_peer(peer)
        return None

    async def remove_peer(self, peer: str):
        if peer in self.peers:
            await self.peers[peer].close()
            del self.peers[peer]
        for bucket in self.buckets:
            bucket[:] = [node for node in bucket if node.address != peer]

    async def connect_to_peer(self, node: KademliaNode):
        if node.address not in self.peers:
            try:
                websocket = await websockets.connect(f"ws://{node.address}")
                self.peers[node.address] = websocket
                await self.add_node_to_bucket(node)
                await self.send_message(node.address, Message(MessageType.MAGNET_LINK.value, {"magnet_link": self.magnet_link.to_uri()}))
                asyncio.create_task(self.handle_messages(websocket, node.address))
            except Exception as e:
                logger.error(f"Failed to connect to peer {node.address}: {str(e)}")

    async def handle_messages(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        try:
            async for message in websocket:
                await self.process_message(Message.from_json(message), peer)
        except websockets.exceptions.ConnectionClosed:
            await self.remove_peer(peer)

    async def process_message(self, message: Message, sender: str):
        try:
            message_hash = hashlib.sha256(message.to_json().encode()).hexdigest()
            if message_hash in self.seen_messages:
                return
            self.seen_messages.add(message_hash)
            asyncio.create_task(self.clean_seen_messages(message_hash))

            await self.handle_message(message, sender)
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            logger.error(traceback.format_exc())

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

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        peer = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.peers[peer] = websocket
        try:
            await self.handle_messages(websocket, peer)
        finally:
            await self.remove_peer(peer)

    async def start_server(self):
        try:
            self.server = await websockets.serve(self.handle_connection, self.host, self.port)
            logger.info(f"P2P node listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            logger.error(traceback.format_exc())

    async def run(self):
        try:
            await self.start_server()
            await self.join_network()
            await self.announce_to_network()
            await asyncio.gather(
                self.process_message_queue(),
                self.periodic_peer_discovery(),
                self.periodic_data_republish(),
                self.send_heartbeats()
            )
        except Exception as e:
            logger.error(f"Failed to run P2P node: {str(e)}")
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
        while True:
            try:
                heartbeat_message = Message(MessageType.HEARTBEAT.value, {"timestamp": time.time()})
                await self.broadcast(heartbeat_message)
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Failed to send heartbeats: {str(e)}")
                logger.error(traceback.format_exc())

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
