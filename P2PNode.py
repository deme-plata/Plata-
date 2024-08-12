import asyncio
import websockets
import json
import time
import hashlib
from typing import Dict, Set, List
from asyncio import Lock
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import traceback
from SecureHybridZKStark import SecureHybridZKStark
from STARK import calculate_field_size
from common import QuantumBlock, Transaction,NodeState
import time 
import logging
import random
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    type: str
    payload: dict
    timestamp: float = time.time()
    sender: str = None

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

class MessageType(Enum):
    TRANSACTION = "transaction"
    BLOCK = "block"
    PEER_DISCOVERY = "peer_discovery"
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

class P2PNode:
    def __init__(self, host: str, port: int, blockchain, security_level: int = 10, max_peers: int = 10):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.security_level = security_level
        self.field_size = calculate_field_size(security_level)
        self.zk_system = SecureHybridZKStark(security_level)
        self.max_peers = max_peers
        self.peers: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.peer_lock = Lock()
        self.message_queue = asyncio.Queue()
        self.last_broadcast: Dict[str, float] = {}
        self.mempool: List[Transaction] = []
        self.seen_messages: Set[str] = set()
        self.message_expiry = 300  # 5 minutes
        self.heartbeat_interval = 30  # 30 seconds
        self.exchange = None  # This can be set later if needed, depending on your architecture
        self.server = None
    async def start(self):
        try:
            # Attempt to start the WebSocket server with a timeout
            self.server = await asyncio.wait_for(
                websockets.serve(self.handle_connection, self.host, self.port),
                timeout=5
            )
            logger.info(f"[INFO] P2P node started successfully on {self.host}:{self.port}")

            # Start background tasks
            try:
                self.background_tasks = [
                    asyncio.create_task(self.process_message_queue()),
                    asyncio.create_task(self.periodic_peer_discovery()),
                    asyncio.create_task(self.periodic_chain_sync()),
                    asyncio.create_task(self.periodic_mempool_sync()),
                    asyncio.create_task(self.send_heartbeats())
                ]

                logger.info("[INFO] All background tasks started successfully")

            except Exception as task_error:
                logger.error(f"[ERROR] An error occurred while starting background tasks: {str(task_error)}")
                logger.error(traceback.format_exc())
                raise

            # Return the server to indicate successful startup
            return self.server

        except asyncio.TimeoutError:
            logger.error(f"[ERROR] Timeout while starting P2P node on {self.host}:{self.port}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Failed to start P2P node on {self.host}:{self.port}: {str(e)}")
            logger.error(traceback.format_exc())
            raise


    async def connect_to_peer(self, peer_address: str):
        logger.debug(f"Attempting to connect to peer: {peer_address}")
        try:
            websocket = await asyncio.wait_for(websockets.connect(f"ws://{peer_address}"), timeout=5)
            peer = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            self.peers[peer] = websocket
            asyncio.create_task(self.handle_messages(websocket, peer))
            logger.debug(f"Successfully connected to peer: {peer_address}")
        except asyncio.TimeoutError:
            logger.error(f"Connection attempt to {peer_address} timed out")
        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_address}: {str(e)}")



    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        peer = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        async with self.peer_lock:
            if len(self.peers) >= self.max_peers:
                await websocket.close(1008, "Max peers reached")
                return
            self.peers[peer] = websocket
        try:
            await self.handle_messages(websocket, peer)
        finally:
            async with self.peer_lock:
                self.peers.pop(peer, None)


    async def handle_messages(self, websocket: websockets.WebSocketServerProtocol, peer: str):
        try:
            async for message in websocket:
                await self.process_message(Message.from_json(message), peer)
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed with peer {peer}")

    async def process_message(self, message: Message, sender: str):
        message_hash = hashlib.sha256(message.to_json().encode()).hexdigest()
        if message_hash in self.seen_messages:
            return
        self.seen_messages.add(message_hash)
        asyncio.create_task(self.clean_seen_messages(message_hash))

        if message.type == MessageType.TRANSACTION.value:
            await self.handle_transaction(message.payload, sender)
        elif message.type == MessageType.BLOCK.value:
            await self.handle_block(message.payload, sender)
        elif message.type == MessageType.PLACE_ORDER.value:
            await self.handle_place_order(message.payload, sender)
        elif message.type == MessageType.CANCEL_ORDER.value:
            await self.handle_cancel_order(message.payload, sender)

        elif message.type == MessageType.PEER_DISCOVERY.value:
            await self.handle_peer_discovery(message.payload, sender)
        elif message.type == MessageType.CHAIN_REQUEST.value:
            await self.handle_chain_request(sender)
        elif message.type == MessageType.CHAIN_RESPONSE.value:
            await self.handle_chain_response(message.payload)
        elif message.type == MessageType.MEMPOOL_REQUEST.value:
            await self.handle_mempool_request(sender)
        elif message.type == MessageType.MEMPOOL_RESPONSE.value:
            await self.handle_mempool_response(message.payload)
        elif message.type == MessageType.HEARTBEAT.value:
            pass  # Just acknowledge the heartbeat
        elif message.type == MessageType.ZK_PROOF.value:
            await self.handle_zk_proof(message.payload, sender)
        else:
            print(f"Unknown message type: {message.type}")

    async def clean_seen_messages(self, message_hash: str):
        await asyncio.sleep(self.message_expiry)
        self.seen_messages.discard(message_hash)

    async def handle_transaction(self, transaction_data: dict, sender_peer: str):
        transaction = Transaction.from_dict(transaction_data)

        # Generate a ZK proof for the transaction
        secret = int(transaction.amount)  # Use transaction amount as secret
        public_input = int(transaction.hash(), 16)  # Use transaction hash as public input

        proof = self.zk_system.prove(secret, public_input)

        # Broadcast the transaction and its proof to all peers except the sender
        await self.broadcast({
            'type': 'zk_proof',
            'payload': {
                'transaction': transaction_data,
                'proof': proof
            }
        }, exclude=sender_peer)
    async def handle_block(self, block_data: dict, sender_peer: str):
        block = QuantumBlock.from_dict(block_data)

        # Verify the block using ZK proofs
        for tx in block.transactions:
            public_input = int(tx.hash(), 16)
            proof = block_data['proofs'][tx.hash()]
            if not self.zk_system.verify(public_input, proof):
                logger.error(f"Invalid ZK proof for transaction {tx.hash()} in block {block.hash}")
                return

        if self.blockchain.consensus.validate_block(block):
            # If valid, add to the blockchain
            self.blockchain.add_block(block)
            logger.info(f"Added new block {block.hash} to the chain")

            # Propagate to other peers, excluding the sender
            await self.broadcast(Message(MessageType.BLOCK.value, block_data), exclude=sender_peer)
        else:
            logger.warning(f"Received invalid block {block.hash}")


    async def handle_zk_proof(self, data: dict, sender_peer: str):
        transaction = Transaction.from_dict(data['transaction'])
        proof = data['proof']

        public_input = int(transaction.hash(), 16)

        if self.zk_system.verify(public_input, proof):
            # If the proof is valid, add the transaction to the blockchain
            if await self.blockchain.add_transaction(transaction):
                # If successfully added, propagate to other peers
                await self.broadcast({
                    'type': 'transaction',
                    'payload': data['transaction']
                }, exclude=sender_peer)
        else:
            print(f"Invalid ZK proof for transaction {transaction.hash()} from peer {sender_peer}")

    async def handle_peer_discovery(self, peer_data: dict, sender: str):
        new_peers = set(peer_data['peers']) - set(self.peers.keys())
        for peer in new_peers:
            asyncio.create_task(self.connect_to_peer(peer))

    async def handle_chain_request(self, sender: str):
        chain_data = [block.to_dict() for block in self.blockchain.chain]
        await self.send_message(sender, Message(MessageType.CHAIN_RESPONSE.value, {"chain": chain_data}))

    async def handle_chain_response(self, chain_data: dict):
        received_chain = [QuantumBlock.from_dict(block) for block in chain_data['chain']]
        if len(received_chain) > len(self.blockchain.chain):
            self.blockchain.chain = received_chain

    async def handle_mempool_request(self, sender: str):
        mempool_data = [tx.to_dict() for tx in self.mempool]
        await self.send_message(sender, Message(MessageType.MEMPOOL_RESPONSE.value, {"mempool": mempool_data}))

    async def handle_mempool_response(self, mempool_data: dict):
        received_mempool = [Transaction.from_dict(tx) for tx in mempool_data['mempool']]
        self.mempool = list(set(self.mempool + received_mempool))
    async def handle_place_order(self, data: dict, sender: str):
        order = data['order']
        zk_proof = data['zk_proof']
        
        # Verify the ZK proof
        public_input = self.zk_system.stark.hash(order['user_id'], order['pair'], int(order['amount'] * 10**18))
        if self.zk_system.verify(public_input, zk_proof):
            # If the proof is valid, add the order to the exchange
            await self.exchange.place_order(order)
        else:
            print(f"Invalid ZK proof for order from peer {sender}")
    async def handle_cancel_order(self, data: dict, sender: str):
        user_id = data['user_id']
        order_id = data['order_id']
        zk_proof = data['zk_proof']
        
        # Verify the ZK proof
        public_input = self.zk_system.stark.hash(user_id, order_id)
        if self.zk_system.verify(public_input, zk_proof):
            # If the proof is valid, cancel the order on the exchange
            await self.exchange.cancel_order(user_id, order_id)
        else:
            print(f"Invalid ZK proof for order cancellation from peer {sender}")


    async def broadcast(self, message: dict, exclude: str = None):
        for peer, websocket in self.peers.items():
            if peer != exclude:
                await websocket.send(json.dumps(message))

    async def process_message_queue(self):
        while True:
            message = await self.message_queue.get()
            current_time = time.time()
            if current_time - self.last_broadcast.get(message.type, 0) > 1:  # Rate limit to 1 message per second per type
                self.last_broadcast[message.type] = current_time
                peers_copy = {}
                async with self.peer_lock:
                    peers_copy = self.peers.copy()
                await asyncio.gather(*[self.send_message(peer, message) for peer in peers_copy])

    async def send_message(self, peer: str, message: Message):
        if peer in self.peers:
            try:
                await self.peers[peer].send(message.to_json())
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"Failed to send message to peer {peer}: {str(e)}")


    async def connect_to_peer(self, peer_address: str):
        logger.debug(f"Attempting to connect to peer: {peer_address}")

        if peer_address not in self.peers:
            try:
                websocket = await websockets.connect(f"ws://{peer_address}")
                peer = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
                self.peers[peer] = websocket
                asyncio.create_task(self.handle_messages(websocket, peer))
            except Exception as e:
                print(f"Failed to connect to peer {peer_address}: {str(e)}")

    async def propagate_transaction(self, transaction: Transaction):
        secret = int(transaction.amount)
        public_input = int(transaction.hash(), 16)
        proof = self.zk_system.prove(secret, public_input)
        
        message = {
            'type': 'zk_proof',
            'payload': {
                'transaction': transaction.to_dict(),
                'proof': proof
            }
        }
        await self.broadcast(message)

    async def propagate_block(self, block: QuantumBlock):
        proofs = {}
        for tx in block.transactions:
            secret = int(tx.amount)
            public_input = int(tx.hash(), 16)
            proofs[tx.hash()] = self.zk_system.prove(secret, public_input)
        
        block_data = block.to_dict()
        block_data['proofs'] = proofs
        
        message = {
            'type': 'block',
            'payload': block_data
        }
        await self.broadcast(message)

    async def periodic_peer_discovery(self):
        while True:
            try:
                async with self.peer_lock:
                    peer_list = list(self.peers.keys())
                await self.broadcast({'type': MessageType.PEER_DISCOVERY.value, 'payload': {'peers': peer_list}})
                await asyncio.sleep(0.1)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in periodic_peer_discovery: {str(e)}")
                logger.debug(traceback.format_exc())


    async def periodic_chain_sync(self):
        while True:
            async with self.peer_lock:
                if self.peers:
                    random_peer = random.choice(list(self.peers.keys()))
                    await self.send_message(random_peer, Message(MessageType.CHAIN_REQUEST.value, {}))
            await asyncio.sleep(600)  # Every 10 minutes

    async def periodic_mempool_sync(self):
        while True:
            async with self.peer_lock:
                if self.peers:
                    random_peer = random.choice(list(self.peers.keys()))
                    await self.send_message(random_peer, Message(MessageType.MEMPOOL_REQUEST.value, {}))
            await asyncio.sleep(60)  # Every 1 minute

    async def send_heartbeats(self):
        while True:
            await self.broadcast({'type': MessageType.HEARTBEAT.value, 'payload': {"timestamp": time.time()}})
            await asyncio.sleep(self.heartbeat_interval)
    async def request_node_state(self, peer_address: str) -> NodeState:
        # Send a request to the peer for its state
        message = Message(type=MessageType.STATE_REQUEST.value, payload={})
        response = await self.send_and_wait_for_response(peer_address, message)
        return NodeState(**response.payload)

    async def verify_network_consistency(self) -> Dict[str, bool]:
        local_state = self.blockchain.get_node_state()
        consistency_results = {}

        for peer_address in self.peers:
            try:
                peer_state = await self.request_node_state(peer_address)
                is_consistent = self.compare_states(local_state, peer_state)
                consistency_results[peer_address] = is_consistent
            except Exception as e:
                logger.error(f"Failed to verify consistency with {peer_address}: {str(e)}")
                consistency_results[peer_address] = False

        return consistency_results

    def compare_states(self, state1: NodeState, state2: NodeState) -> bool:
        # Compare the most critical aspects of the state
        return (
            state1.blockchain_length == state2.blockchain_length and
            state1.latest_block_hash == state2.latest_block_hash and
            state1.total_supply == state2.total_supply and
            state1.difficulty == state2.difficulty
        )

    async def handle_message(self, message: Message, sender: str):
        if message.type == MessageType.STATE_REQUEST.value:
            state = self.blockchain.get_node_state()
            response = Message(type=MessageType.STATE_RESPONSE.value, payload=asdict(state))
            await self.send_message(sender, response)
