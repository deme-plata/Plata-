# mongodb_manager.py

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from bson.objectid import ObjectId
import logging
from typing import Dict, Any, List
from collections import deque
import json
from fastapi import WebSocket
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne, ASCENDING, TEXT
from bson.objectid import ObjectId
import logging
from typing import Dict, Any, List
from collections import deque
import json
from fastapi import WebSocket
from pymongo.errors import OperationFailure
import time
from cachetools import TTLCache
from bson.json_util import dumps, loads
from typing import Dict, Any
import json
from bson.json_util import dumps, loads
from shared_logic import Transaction, QuantumBlock  # Add this line
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal


logger = logging.getLogger(__name__)

class DatabaseQueue:
    def __init__(self, max_size=1000):
        self.queue = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def add_operation(self, operation):
        async with self.lock:
            self.queue.append(operation)

    async def get_operations(self, batch_size=100):
        async with self.lock:
            operations = []
            while len(operations) < batch_size and self.queue:
                operations.append(self.queue.popleft())
            return operations

class MongoDBManager:
    def __init__(self, connection_string, database_name):
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        self.sync_lock = asyncio.Lock()
        self.queue = DatabaseQueue()
        self.subscribers: Dict[str, List[WebSocket]] = {}
        self.cache = TTLCache(maxsize=1000, ttl=300)  # Cache with 5-minute TTL

    async def process_queue(self):
        while True:
            operations = await self.queue.get_operations()
            if operations:
                for operation in operations:
                    await self._execute_operation(operation)
            else:
                await asyncio.sleep(0.1)  # Avoid busy-waiting

    async def _execute_operation(self, operation):
        collection = self.db[operation['collection']]
        method = getattr(collection, operation['method'])
        await method(**operation['params'])
        await self._notify_subscribers(operation['collection'], operation)

    async def insert_document(self, collection_name: str, document: Dict[str, Any]):
        await self.queue.add_operation({
            'collection': collection_name,
            'method': 'insert_one',
            'params': {'document': document}
        })
        return document.get('_id')

    async def find_document(self, collection_name: str, query: Dict[str, Any]):
        collection = self.db[collection_name]
        return await collection.find_one(query)

    async def update_document(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]):
        await self.queue.add_operation({
            'collection': collection_name,
            'method': 'update_one',
            'params': {'filter': query, 'update': {"$set": update}}
        })

    async def delete_document(self, collection_name: str, query: Dict[str, Any]):
        await self.queue.add_operation({
            'collection': collection_name,
            'method': 'delete_one',
            'params': {'filter': query}
        })

    async def find_many(self, collection_name: str, query: Dict[str, Any], limit: int = None):
        collection = self.db[collection_name]
        cursor = collection.find(query)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=None)

    async def sync_data(self, other_node_data: List[Dict[str, Any]], collection_name: str):
        async with self.sync_lock:
            local_data = await self.find_many(collection_name, {})
            local_data_dict = {str(doc['_id']): doc for doc in local_data}
            
            operations = []
            for doc in other_node_data:
                doc_id = str(doc['_id'])
                if doc_id not in local_data_dict:
                    operations.append(UpdateOne({'_id': ObjectId(doc_id)}, {'$set': doc}, upsert=True))
                elif doc != local_data_dict[doc_id]:
                    operations.append(UpdateOne({'_id': ObjectId(doc_id)}, {'$set': doc}))
            
            if operations:
                collection = self.db[collection_name]
                result = await collection.bulk_write(operations)
                logger.info(f"Sync completed. Inserted: {result.upserted_count}, Modified: {result.modified_count}")
                await self._notify_subscribers(collection_name, {'type': 'sync', 'data': other_node_data})
            else:
                logger.info("No changes needed during sync.")

    async def get_all_data(self, collection_name: str):
        return await self.find_many(collection_name, {})

    async def close_connection(self):
        self.client.close()

    async def subscribe(self, collection_name: str, websocket: WebSocket):
        if collection_name not in self.subscribers:
            self.subscribers[collection_name] = []
        self.subscribers[collection_name].append(websocket)

    async def unsubscribe(self, collection_name: str, websocket: WebSocket):
        if collection_name in self.subscribers:
            self.subscribers[collection_name].remove(websocket)

    async def _notify_subscribers(self, collection_name: str, data: Dict[str, Any]):
        if collection_name in self.subscribers:
            for websocket in self.subscribers[collection_name]:
                await websocket.send_json(data)
                
    async def create_time_series_collection(self, collection_name: str, time_field: str):
        try:
            # Drop the existing collection if it exists
            if collection_name in await self.db.list_collection_names():
                await self.db.drop_collection(collection_name)
                logger.info(f"Dropped existing collection '{collection_name}'.")
            
            # Create the new time series collection
            await self.db.create_collection(
                collection_name,
                timeseries={
                    "timeField": time_field,
                    "metaField": "metadata",
                    "granularity": "seconds"
                }
            )
            logger.info(f"Time-series collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create time-series collection: {str(e)}")

    async def insert_time_series_data(self, collection_name: str, data: Dict[str, Any]):
        await self.queue.add_operation({
            'collection': collection_name,
            'method': 'insert_one',
            'params': {'document': data}
        })

    async def get_time_series_data(self, collection_name: str, start_time: int, end_time: int):
        collection = self.db[collection_name]
        query = {
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        return await collection.find(query).to_list(None)

    async def create_text_index(self, collection_name: str, fields: List[str]):
        collection = self.db[collection_name]
        index_fields = [(field, TEXT) for field in fields]
        await collection.create_index(index_fields)
        logger.info(f"Text index created for collection '{collection_name}' on fields {fields}")

    async def full_text_search(self, collection_name: str, search_text: str):
        collection = self.db[collection_name]
        query = {"$text": {"$search": search_text}}
        return await collection.find(query).to_list(None)

    async def create_shard_key(self, collection_name: str, shard_key: Dict[str, Any]):
        try:
            await self.db.command({
                "shardCollection": f"{self.db.name}.{collection_name}",
                "key": shard_key
            })
            logger.info(f"Shard key created for collection '{collection_name}'")
        except OperationFailure as e:
            logger.error(f"Failed to create shard key: {str(e)}")

    async def get_cached_data(self, key: str):
        return self.cache.get(key)

    async def set_cached_data(self, key: str, value: Any):
        self.cache[key] = value

    async def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]):
        collection = self.db[collection_name]
        return await collection.aggregate(pipeline).to_list(None)

    async def create_backup(self):
        timestamp = int(time.time())
        backup_name = f"backup_{timestamp}"
        try:
            await self.client.admin.command("fsync", lock=True)
            # Implement your backup logic here (e.g., using mongodump)
            logger.info(f"Backup '{backup_name}' created successfully")
        finally:
            await self.client.admin.command("fsyncUnlock")

    async def restore_backup(self, backup_name: str):
        # Implement your restore logic here (e.g., using mongorestore)
        logger.info(f"Backup '{backup_name}' restored successfully")

    async def start_session(self):
        return await self.client.start_session()

    async def commit_transaction(self, session):
        await session.commit_transaction()

    async def abort_transaction(self, session):
        await session.abort_transaction()

    async def deploy_contract(self, contract_address: str, contract_code: str, creator: str, initial_state: Dict[str, Any]):
        contract_doc = {
            "address": contract_address,
            "code": contract_code,
            "creator": creator,
            "state": initial_state,
            "created_at": int(time.time())
        }
        await self.queue.add_operation({
            'collection': 'contracts',
            'method': 'insert_one',
            'params': {'document': contract_doc}
        })
        logger.info(f"Contract deployed: {contract_address}")

    async def get_contract(self, contract_address: str):
        return await self.find_document("contracts", {"address": contract_address})

    async def update_contract_state(self, contract_address: str, new_state: Dict[str, Any]):
        await self.queue.add_operation({
            'collection': 'contracts',
            'method': 'update_one',
            'params': {
                'filter': {"address": contract_address},
                'update': {"$set": {"state": new_state}}
            }
        })
        logger.info(f"Contract state updated: {contract_address}")

    async def get_all_contracts(self):
        return await self.find_many("contracts", {})

    async def create_user(self, user_id: str, permissions: List[str], roles: List[str]):
        user_doc = {
            "user_id": user_id,
            "permissions": permissions,
            "roles": roles,
            "created_at": int(time.time())
        }
        await self.queue.add_operation({
            'collection': 'users',
            'method': 'insert_one',
            'params': {'document': user_doc}
        })
        logger.info(f"User created: {user_id}")

    async def get_user(self, user_id: str):
        return await self.find_document("users", {"user_id": user_id})

    async def update_user_permissions(self, user_id: str, permissions: List[str]):
        await self.queue.add_operation({
            'collection': 'users',
            'method': 'update_one',
            'params': {
                'filter': {"user_id": user_id},
                'update': {"$set": {"permissions": permissions}}
            }
        })
        logger.info(f"User permissions updated: {user_id}")

    async def update_user_roles(self, user_id: str, roles: List[str]):
        await self.queue.add_operation({
            'collection': 'users',
            'method': 'update_one',
            'params': {
                'filter': {"user_id": user_id},
                'update': {"$set": {"roles": roles}}
            }
        })
        logger.info(f"User roles updated: {user_id}")

    async def get_all_users(self):
        return await self.find_many("users", {})

    async def create_token(self, token_address: str, token_name: str, token_symbol: str, total_supply: int, creator: str):
        token_doc = {
            "address": token_address,
            "name": token_name,
            "symbol": token_symbol,
            "total_supply": total_supply,
            "creator": creator,
            "created_at": int(time.time())
        }
        await self.queue.add_operation({
            'collection': 'tokens',
            'method': 'insert_one',
            'params': {'document': token_doc}
        })
        logger.info(f"Token created: {token_address}")

    async def get_token(self, token_address: str):
        return await self.find_document("tokens", {"address": token_address})

    async def update_token_supply(self, token_address: str, new_supply: int):
        await self.queue.add_operation({
            'collection': 'tokens',
            'method': 'update_one',
            'params': {
                'filter': {"address": token_address},
                'update': {"$set": {"total_supply": new_supply}}
            }
        })
        logger.info(f"Token supply updated: {token_address}")

    async def get_all_tokens(self):
        return await self.find_many("tokens", {})

    async def create_nft(self, nft_id: str, metadata: Dict[str, Any], owner: str):
        nft_doc = {
            "nft_id": nft_id,
            "metadata": metadata,
            "owner": owner,
            "created_at": int(time.time())
        }
        await self.queue.add_operation({
            'collection': 'nfts',
            'method': 'insert_one',
            'params': {'document': nft_doc}
        })
        logger.info(f"NFT created: {nft_id}")

    async def get_nft(self, nft_id: str):
        return await self.find_document("nfts", {"nft_id": nft_id})

    async def update_nft_owner(self, nft_id: str, new_owner: str):
        await self.queue.add_operation({
            'collection': 'nfts',
            'method': 'update_one',
            'params': {
                'filter': {"nft_id": nft_id},
                'update': {"$set": {"owner": new_owner}}
            }
        })
        logger.info(f"NFT owner updated: {nft_id}")

    async def get_all_nfts(self):
        return await self.find_many("nfts", {})

# Integration with QuantumDAGKnight system
class QuantumDAGKnightDB:
    def __init__(self, connection_string: str, database_name: str):
        self.mongo_manager = MongoDBManager(connection_string, database_name)
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        # Initialize collections
        self.blocks = self.db.blocks
        self.transactions = self.db.transactions
        self.wallets = self.db.wallets
        self.chain = []
        self.adjustment_interval = 10  # Adjust every 10 blocks
        self.target_block_time = 600  # Target 10 minutes (600 seconds) per block
        self.difficulty = 4  # Initial difficulty
        self.target = 2**(256 - self.difficulty)
        self.blockchain = None  # We'll set this later
    def set_blockchain(self, blockchain):
        self.blockchain = blockchain

    async def get_current_difficulty(self):
        return self.blockchain.difficulty


    def adjust_difficulty(self):
        if len(self.chain) >= self.adjustment_interval:
            start_block = self.chain[-self.adjustment_interval]
            end_block = self.chain[-1]
            
            total_time = end_block['timestamp'] - start_block['timestamp']
            
            if total_time <= 0:
                logger.error("Total time between blocks is zero or negative. Cannot adjust difficulty.")
                return
            
            avg_time = total_time / (self.adjustment_interval - 1)
            target_time = self.target_block_time
            
            logger.info(f"Start block timestamp: {start_block['timestamp']}")
            logger.info(f"End block timestamp: {end_block['timestamp']}")
            logger.info(f"Total time for last {self.adjustment_interval} blocks: {total_time:.2f} seconds")
            logger.info(f"Average time per block: {avg_time:.2f} seconds")
            logger.info(f"Target block time: {target_time:.2f} seconds")
            logger.info(f"Current difficulty: {self.difficulty}")

            # Calculate the adjustment factor
            adjustment_factor = target_time / avg_time
            logger.info(f"Adjustment factor: {adjustment_factor:.2f}")

            # Adjust difficulty based on the adjustment factor
            if adjustment_factor > 1:
                new_difficulty = min(int(self.difficulty * adjustment_factor), 256)
                logger.info(f"Increasing difficulty: {self.difficulty} -> {new_difficulty}")
            else:
                new_difficulty = max(int(self.difficulty / adjustment_factor), 1)
                logger.info(f"Decreasing difficulty: {self.difficulty} -> {new_difficulty}")

            # Update difficulty and target
            self.difficulty = new_difficulty
            self.target = 2**(256 - self.difficulty)
            logger.info(f"New difficulty: {self.difficulty}")
            logger.info(f"New target: {self.target:.2e}")
        else:
            logger.info(f"Not enough blocks to adjust difficulty. Current chain length: {len(self.chain)}")

    async def init_process_queue(self):
        await self.mongo_manager.process_queue()


    async def add_block(self, block):
        block_data = block.to_dict()
        block_id = await self.mongo_manager.insert_document("blocks", block_data)
        return block_id

    async def get_block(self, block_hash):
        return await self.mongo_manager.find_document("blocks", {"hash": block_hash})

    async def update_block_status(self, block_hash, status):
        await self.mongo_manager.update_document("blocks", {"hash": block_hash}, {"status": status})

    async def add_transaction(self, transaction):
        tx_data = transaction.to_dict()
        tx_id = await self.mongo_manager.insert_document("transactions", tx_data)
        return tx_id

    async def get_transaction(self, tx_hash):
        return await self.mongo_manager.find_document("transactions", {"hash": tx_hash})

    async def update_transaction_status(self, tx_hash, status):
        await self.mongo_manager.update_document("transactions", {"hash": tx_hash}, {"status": status})

    async def get_latest_blocks(self, limit=10):
        return await self.mongo_manager.find_many("blocks", {}, limit=limit)

    async def get_latest_transactions(self, limit=10):
        return await self.mongo_manager.find_many("transactions", {}, limit=limit)

    async def sync_with_peer(self, peer_data: Dict[str, Any]):
        try:
            # Sync blocks
            for block_data in peer_data.get('blocks', []):
                await self.add_block(QuantumBlock.from_dict(block_data))
            
            # Sync transactions
            for tx_data in peer_data.get('transactions', []):
                await self.add_transaction(Transaction.from_dict(tx_data))
            
            # Sync wallets
            for wallet_data in peer_data.get('wallets', []):
                await self.add_wallet(wallet_data)
            
            # Sync other data as needed
            # For example, you might want to sync token data, contracts, etc.
            
            logger.info("Successfully synced data with peer")
        except Exception as e:
            logger.error(f"Error syncing with peer: {str(e)}")
            logger.error(traceback.format_exc())

    async def get_all_data(self) -> Dict[str, Any]:
        try:
            # Fetch all blocks
            blocks = await self.get_latest_blocks(limit=0)  # 0 means no limit
            
            # Fetch all transactions
            transactions = await self.get_latest_transactions(limit=0)
            
            # Fetch all wallets
            wallets = await self.db.wallets.find().to_list(length=None)
            
            # Fetch other necessary data
            # For example, you might want to fetch token data, contracts, etc.
            
            # Prepare the data dictionary
            all_data = {
                'blocks': [block.to_dict() for block in blocks],
                'transactions': [tx.to_dict() for tx in transactions],
                'wallets': wallets,
                # Add other data as needed
            }
            
            # Convert to JSON-serializable format
            return json.loads(dumps(all_data))
        except Exception as e:
            logger.error(f"Error getting all data: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


    async def subscribe_to_updates(self, collection_name: str, websocket: WebSocket):
        await self.mongo_manager.subscribe(collection_name, websocket)

    async def unsubscribe_from_updates(self, collection_name: str, websocket: WebSocket):
        await self.mongo_manager.unsubscribe(collection_name, websocket)

    async def close(self):
        await self.mongo_manager.close_connection()
        
    async def init_collections(self):
        await self.mongo_manager.create_time_series_collection("block_stats", "timestamp")
        await self.mongo_manager.create_text_index("blocks", ["data"])
        await self.mongo_manager.create_text_index("transactions", ["data"])
        await self.mongo_manager.create_shard_key("blocks", {"timestamp": ASCENDING})
        await self.mongo_manager.create_shard_key("transactions", {"timestamp": ASCENDING})

    async def add_block_stats(self, block_hash: str, stats: Dict[str, Any]):
        stats["block_hash"] = block_hash
        stats["timestamp"] = int(time.time())
        await self.mongo_manager.insert_time_series_data("block_stats", stats)

    async def get_block_stats(self, start_time: int, end_time: int):
        return await self.mongo_manager.get_time_series_data("block_stats", start_time, end_time)

    async def search_blocks(self, query: str):
        return await self.mongo_manager.full_text_search("blocks", query)

    async def search_transactions(self, query: str):
        return await self.mongo_manager.full_text_search("transactions", query)

    async def get_cached_block(self, block_hash: str):
        cached_block = await self.mongo_manager.get_cached_data(f"block:{block_hash}")
        if cached_block:
            return loads(cached_block)
        block = await self.get_block(block_hash)
        if block:
            await self.mongo_manager.set_cached_data(f"block:{block_hash}", dumps(block))
        return block

    async def get_transaction_volume(self, start_time: int, end_time: int):
        pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_volume": {"$sum": "$amount"}
                }
            }
        ]
        result = await self.mongo_manager.aggregate("transactions", pipeline)
        return result[0]["total_volume"] if result else 0

    async def create_backup(self):
        await self.mongo_manager.create_backup()

    async def restore_backup(self, backup_name: str):
        await self.mongo_manager.restore_backup(backup_name)

    async def atomic_operation(self, operations):
        async with await self.mongo_manager.start_session() as session:
            async with session.start_transaction():
                try:
                    for op in operations:
                        await op(session)
                    await self.mongo_manager.commit_transaction(session)
                except Exception as e:
                    await self.mongo_manager.abort_transaction(session)
                    raise e
    async def deploy_contract(self, contract_address: str, contract_code: str, creator: str, initial_state: Dict[str, Any]):
        await self.mongo_manager.deploy_contract(contract_address, contract_code, creator, initial_state)

    async def get_contract(self, contract_address: str):
        return await self.mongo_manager.get_contract(contract_address)

    async def update_contract_state(self, contract_address: str, new_state: Dict[str, Any]):
        await self.mongo_manager.update_contract_state(contract_address, new_state)

    async def get_all_contracts(self):
        return await self.mongo_manager.get_all_contracts()

    async def create_user(self, user_id: str, permissions: List[str], roles: List[str]):
        await self.mongo_manager.create_user(user_id, permissions, roles)
    async def get_user(self, user_id: str):
        user_data = await self.db.users.find_one({"_id": user_id})
        if user_data:
            return user_data
        return None
    async def update_user(self, user_id: str, user_data: dict):
        await self.db.users.update_one(
            {"_id": user_id},
            {"$set": user_data},
            upsert=True
        )

    async def update_user_permissions(self, user_id: str, permissions: List[str]):
        await self.mongo_manager.update_user_permissions(user_id, permissions)

    async def update_user_roles(self, user_id: str, roles: List[str]):
        await self.mongo_manager.update_user_roles(user_id, roles)

    async def get_all_users(self):
        return await self.mongo_manager.get_all_users()

    async def create_token(self, token_address: str, token_name: str, token_symbol: str, total_supply: int, creator: str):
        await self.mongo_manager.create_token(token_address, token_name, token_symbol, total_supply, creator)

    async def get_token(self, token_address: str):
        return await self.mongo_manager.get_token(token_address)

    async def update_token_supply(self, token_address: str, new_supply: int):
        await self.mongo_manager.update_token_supply(token_address, new_supply)

    async def get_all_tokens(self):
        return await self.mongo_manager.get_all_tokens()

    async def create_nft(self, nft_id: str, metadata: Dict[str, Any], owner: str):
        await self.mongo_manager.create_nft(nft_id, metadata, owner)

    async def get_nft(self, nft_id: str):
        return await self.mongo_manager.get_nft(nft_id)

    async def update_nft_owner(self, nft_id: str, new_owner: str):
        await self.mongo_manager.update_nft_owner(nft_id, new_owner)

    async def get_all_nfts(self):
        return await self.mongo_manager.get_all_nfts()
    async def get_latest_block(self):
        return await self.db.blocks.find_one(sort=[('timestamp', -1)])

    async def get_pending_transactions(self):
        return await self.db.pending_transactions.find().to_list(length=None)

    async def get_current_difficulty(self):
        # Implement logic to retrieve and calculate current difficulty
        pass

    async def add_block(self, block):
        await self.db.blocks.insert_one(block)

    async def process_transactions(self, transactions):
        # Implement logic to process transactions
        pass

    async def reward_miner(self, wallet_address, reward):
        # Implement logic to reward the miner
        pass
    async def get_address_by_alias(self, alias: str) -> str:
        wallet = await self.db.wallets.find_one({"alias": alias})
        return wallet["address"] if wallet else None

    async def get_zk_system(self):
        # Implement logic to retrieve or create the ZK system
        # This might involve loading parameters from the database or creating a new instance
        pass

    async def add_transaction(self, transaction: Transaction) -> bool:
        # Check balance
        sender_balance = await self.get_balance(transaction.sender)
        if sender_balance < transaction.amount:
            return False

        # Add transaction to pending transactions
        await self.db.pending_transactions.insert_one(transaction.to_dict())

        # Update balances
        await self.db.wallets.update_one(
            {"address": transaction.sender},
            {"$inc": {"balance": -transaction.amount}}
        )
        await self.db.wallets.update_one(
            {"address": transaction.receiver},
            {"$inc": {"balance": transaction.amount}}
        )

        return True

    async def get_balance(self, address: str) -> Decimal:
        wallet = await self.db.wallets.find_one({"address": address})
        return Decimal(str(wallet["balance"])) if wallet else Decimal("0")
