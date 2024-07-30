import hashlib
import traceback
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature
from cachetools import LRUCache
from collections import defaultdict
from merkletools import MerkleTools  # Ensure you have the merkletools package installed
from enum import Enum, auto
import requests


class PersistentStorage:
    def __init__(self):
        self.storage = {}

    def load(self, key):
        return self.storage.get(key, {})

    def save(self, key, value):
        self.storage[key] = value


class ComplexDataTypes:
    def __init__(self):
        self.structs = {}
        self.enums = {}

    def define_struct(self, name, fields):
        self.structs[name] = fields

    def define_enum(self, name, values):
        self.enums[name] = values

    def create_struct_instance(self, struct_name, field_values):
        if struct_name not in self.structs:
            raise ValueError(f"Struct {struct_name} not defined")

        struct_fields = self.structs[struct_name]
        if set(field_values.keys()) != set(struct_fields):
            raise ValueError(f"Invalid fields for struct {struct_name}")

        return {field: field_values[field] for field in struct_fields}

    def get_enum_value(self, enum_name, value):
        if enum_name not in self.enums:
            raise ValueError(f"Enum {enum_name} not defined")

        if value not in self.enums[enum_name]:
            raise ValueError(f"Invalid value for enum {enum_name}")

        return self.enums[enum_name].index(value)


class Permission(Enum):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()


class Role(Enum):
    ADMIN = auto()
    USER = auto()
    AUDITOR = auto()


class AccessControlSystem:
    def __init__(self):
        self.user_roles = {}
        self.role_permissions = {
            Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            Role.USER: {Permission.READ, Permission.EXECUTE},
            Role.AUDITOR: {Permission.READ}
        }

    def assign_role(self, user_id, role):
        if not isinstance(role, Role):
            raise ValueError("Invalid role")
        self.user_roles[user_id] = role

    def check_permission(self, user_id, permission):
        if user_id not in self.user_roles:
            return False
        user_role = self.user_roles[user_id]
        return permission in self.role_permissions[user_role]

    def grant_permission(self, role, permission):
        if not isinstance(role, Role) or not isinstance(permission, Permission):
            raise ValueError("Invalid role or permission")
        self.role_permissions[role].add(permission)

    def revoke_permission(self, role, permission):
        if not isinstance(role, Role) or not isinstance(permission, Permission):
            raise ValueError("Invalid role or permission")
        self.role_permissions[role].discard(permission)


class PBFTConsensus:
    def __init__(self, nodes, node_id):
        self.nodes = nodes
        self.f = (len(nodes) - 1) // 3  # Maximum number of faulty nodes
        self.sequence_number = 0
        self.current_view = 0
        self.prepared = defaultdict(set)
        self.committed = defaultdict(set)
        self.node_id = node_id

    def propose(self, client_request):
        self.sequence_number += 1
        message = f"VIEW:{self.current_view},SEQ:{self.sequence_number},TYPE:PREPREPARE,DATA:{client_request}"
        self.broadcast(message)

    def prepare(self, node, message):
        self.prepared[message].add(node)
        if len(self.prepared[message]) >= 2 * self.f + 1:
            commit_message = f"VIEW:{self.current_view},SEQ:{self.sequence_number},TYPE:COMMIT,DATA:{message}"
            self.broadcast(commit_message)

    def commit(self, node, message):
        self.committed[message].add(node)
        if len(self.committed[message]) >= 2 * self.f + 1:
            self.execute_request(message)

    def execute_request(self, message):
        print(f"Executing request: {message}")

    def broadcast(self, message):
        for node in self.nodes:
            if node.node_id != self.node_id and random.random() > 0.1:  # Simulating 10% message loss
                node.receive(message)

    def receive(self, message):
        if "TYPE:PREPREPARE" in message:
            self.prepare(self.node_id, message)
        elif "TYPE:COMMIT" in message:
            self.commit(self.node_id, message)


class Node:
    def __init__(self, node_id, vm, host):
        self.node_id = node_id
        self.vm = vm
        self.host = host
        self.consensus = PBFTConsensus([self], node_id)

    def receive(self, message):
        self.consensus.receive(message)

    def propose_transaction(self, transaction):
        self.vm.propose_transaction(transaction)
        for node in self.vm.nodes:
            if node.node_id != self.node_id:
                requests.post(f'http://{node.host}/receive_message/', json={'node_id': node.node_id, 'message': transaction})


class SimpleVM:
    def __init__(self, gas_limit=10000, number_of_shards=10, nodes=None):
        self.global_scope = {}
        self.scope_stack = [self.global_scope]
        self.functions = {}
        self.transaction_queue = []
        self.users = {}
        self.roles = {}
        self.events = []
        self.gas_limit = gas_limit
        self.gas_used = 0
        self.contract_states = {}
        self.nonces = {}
        self.token_balances = {}
        self.nfts = {}
        self.persistent_storage = PersistentStorage()
        self.cache = LRUCache(maxsize=1000)
        self.number_of_shards = number_of_shards
        self.shard_locks = [threading.RLock() for _ in range(self.number_of_shards)]
        self.executor = ThreadPoolExecutor(max_workers=self.number_of_shards)
        self.load_state()  # Load state from persistent storage
        self.state_versions = defaultdict(dict)  # For conflict resolution
        self.merkle_trees = defaultdict(MerkleTools)  # For Merkle tree state management
        self.complex_data_types = ComplexDataTypes()  # For managing complex data types
        self.access_control = AccessControlSystem()  # For sophisticated permissions and access control
        self.nodes = nodes if nodes else []
        self.consensus = PBFTConsensus(self.nodes, self.nodes[0].node_id) if self.nodes else None

    def load_state(self):
        self.token_balances = self.persistent_storage.load('token_balances')
        self.nfts = self.persistent_storage.load('nfts')

    def save_state(self):
        self.persistent_storage.save('token_balances', self.token_balances)
        self.persistent_storage.save('nfts', self.nfts)

    def generate_contract_address(self, sender_address):
        nonce = self.nonces.get(sender_address, 0)
        raw_address = f"{sender_address}{nonce}".encode()
        contract_address = hashlib.sha256(raw_address).hexdigest()[:40]
        self.nonces[sender_address] = nonce + 1
        return "0x" + contract_address

    def verify_signature(self, signature, sender_public_key, transaction_data):
        try:
            public_key = serialization.load_pem_public_key(sender_public_key.encode())
            public_key.verify(
                signature,
                transaction_data.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            print(f"Error verifying signature: {e}")
            return False

    def validate_contract_transaction(self, transaction):
        if not self.verify_signature(transaction['signature'], transaction['sender_public_key'], str(transaction['data'])):
            print("Invalid transaction signature.")
            return False

        if transaction['contract_address'] not in self.global_scope:
            print(f"Contract at address {transaction['contract_address']} does not exist.")
            return False

        return True

    def apply_state_transition(self, transaction, execution_result):
        if 'new_state' in execution_result:
            self.global_scope[transaction['contract_address']]['storage'] = execution_result['new_state']
            print(f"Applied state transition for contract {transaction['contract_address']}.")
        else:
            print("Execution result did not contain a new state.")

    def get_shard_for_contract(self, contract_address):
        return hash(contract_address) % self.number_of_shards

    def update_contract_state(self, db_path, contract_address, new_state):
        self.contract_states[contract_address] = new_state

    def has_permission(self, user_id, permission):
        return self.access_control.check_permission(user_id, permission)

    def get_contract_state(self, address):
        if address in self.cache:
            return self.cache[address]
        state = self.contract_states.get(address, {}).copy()
        self.cache[address] = state
        return state

    def set_contract_state(self, address, state):
        self.contract_states[address] = state.copy()

    def add_user(self, user_id, permissions=None, roles=None):
        if user_id not in self.users:
            self.users[user_id] = {"permissions": set(), "roles": set()}
        self.users[user_id]["permissions"].update(permissions or [])
        self.users[user_id]["roles"].update(roles or [])
        if roles:
            for role in roles:
                self.access_control.assign_role(user_id, role)

    def assign_permission_to_user(self, user_id, permission):
        if user_id in self.users:
            self.users[user_id]["permissions"].add(permission)

    def revoke_permission_from_user(self, user_id, permission):
        if user_id in self.users:
            self.users[user_id]["permissions"].discard(permission)

    def assign_role_to_user(self, user_id, role):
        if user_id in self.users:
            self.users[user_id]["roles"].add(role)
            self.access_control.assign_role(user_id, role)
        else:
            print(f"User {user_id} not found.")

    def revoke_role_from_user(self, user_id, role):
        if user_id in self.users and role in self.users[user_id]["roles"]:
            self.users[user_id]["roles"].remove(role)
            self.access_control.user_roles.pop(user_id, None)
        else:
            print(f"Role {role} not found for user {user_id}.")

    def consume_gas(self, amount):
        self.gas_used += amount
        if self.gas_used > self.gas_limit:
            raise Exception("Out of gas")

    def call_contract_function(self, caller_id, target_address, function_name, *args):
        target_contract = self.global_scope.get(target_address)
        if not target_contract:
            raise Exception(f"Contract at address {target_address} not found.")

        target_function = target_contract['functions'].get(function_name)
        if not target_function:
            raise Exception(f"Function {function_name} not found in contract at address {target_address}.")

        result = target_function(*args)
        self.log_event(f"Function {function_name} called on contract {target_address} by {caller_id}.")
        return result

    def log_event(self, message):
        with threading.Lock():
            self.events.append(message)
        print(f"Event logged: {message}")

    def get_events(self):
        return self.events

    def deploy_contract(self, sender_address, code, permissions=None):
        contract_address = self.generate_contract_address(sender_address)
        parsed_permissions = self.parse_permissions(code)
        self.global_scope[contract_address] = {
            "code": code,
            "storage": {},
            "is_executing": False,
            "permissions": parsed_permissions,
            "version": 1  # Initial version of the contract
        }
        return contract_address

    def upgrade_contract(self, contract_address, new_code):
        if contract_address in self.global_scope:
            current_version = self.global_scope[contract_address].get("version", 1)
            self.global_scope[contract_address].update({
                "code": new_code,
                "version": current_version + 1
            })
            print(f"Contract {contract_address} upgraded to version {current_version + 1}.")
        else:
            print(f"Contract {contract_address} not found for upgrade.")

    def parse_permissions(self, code):
        permissions = {}
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("// Permissions for"):
                try:
                    parts = line.split(':')
                    if len(parts) != 2:
                        raise ValueError("Invalid permissions syntax")

                    func_name = parts[0].split(' ')[-1].strip()
                    permission_parts = parts[1].strip().strip('[]').split(',')

                    users = []
                    roles = []
                    for part in permission_parts:
                        part = part.strip()
                        if part.startswith("role:"):
                            role = part.split("role:")[1].strip()
                            if role:
                                roles.append(role)
                        else:
                            if part not in self.users:
                                raise ValueError(f"User ID '{part}' not recognized")
                            users.append(part)

                    if not users and not roles:
                        raise ValueError("No valid user IDs or roles specified in permissions")

                    permissions[func_name] = {"users": users, "roles": roles}
                except ValueError as e:
                    print(f"Error parsing permissions: {e}")
                    continue
        return permissions

    def add_transaction(self, user_id, transaction):
        if not self.has_permission(user_id, Permission.EXECUTE):
            raise Exception("Unauthorized: User does not have execute permission")
        self.transaction_queue.append(transaction)

    async def process_transactions(self):
        loop = asyncio.get_event_loop()

        async def execute_transaction(transaction):
            await loop.run_in_executor(self.executor, self.execute_contract,
                                       transaction["to"],
                                       transaction["data"],
                                       transaction.get("gas_limit", self.gas_limit),
                                       transaction["user_id"],
                                       transaction["sender_public_key"],
                                       transaction["signature"])

        tasks = [execute_transaction(transaction) for transaction in self.transaction_queue]
        await asyncio.gather(*tasks)
        self.transaction_queue.clear()

    def execute_contract(self, contract_address, input_data, gas_limit, user_id, sender_public_key, signature):
        shard_id = self.get_shard_for_contract(contract_address)
        with self.shard_locks[shard_id]:
            if contract_address not in self.contract_states:
                print(f"Contract {contract_address} not found.")
                return {"error": "Contract not found."}

            if not self.has_permission(user_id, Permission.EXECUTE):
                return {"error": "Unauthorized to execute contracts."}

            if not self.verify_signature(signature, sender_public_key, str(input_data)):
                return {"error": "Invalid signature."}

            execution_result = {'new_state': {'key': 'new_value'}}

            transaction = {
                'contract_address': contract_address,
                'sender_public_key': sender_public_key,
                'signature': signature,
                'data': input_data
            }

            initial_state = self.get_contract_state(contract_address)
            self.state_versions[shard_id][contract_address] = initial_state
            self.gas_used = 0
            self.gas_limit = gas_limit

            try:
                self.global_scope[contract_address]["is_executing"] = True
                self.consume_gas(100)

                for operation in input_data["code"]:
                    if self.gas_used > self.gas_limit:
                        raise Exception("Gas limit exceeded.")

                    gas_cost = self.get_operation_gas_cost(operation, input_data["code"][operation])
                    self.consume_gas(gas_cost)

                output_data = "Contract execution result"
                self.set_contract_state(contract_address, execution_result['new_state'])
                self.merkle_trees[shard_id].add_leaf(str(execution_result['new_state']), True)  # Update Merkle tree
                self.merkle_trees[shard_id].make_tree()  # Rebuild Merkle tree after adding new leaf
                print("Transaction validated and executed.")
                return {"output": "Contract execution result", "gas_used": self.gas_used}

            except Exception as e:
                # Revert to the initial state in case of an execution error
                self.set_contract_state(contract_address, initial_state)
                print("Transaction validation failed due to an error:", str(e))
                return {"error": str(e), "gas_used": self.gas_used}

            finally:
                self.global_scope[contract_address]["is_executing"] = False

    def get_operation_gas_cost(self, operation, input_data):
        # More granular gas metering based on operation type and complexity
        base_costs = {
            "STORE": 20,
            "LOAD": 10,
            "COMPUTE": 15,
            "ADD": 5,
            "SUBTRACT": 5,
            "MULTIPLY": 10,
            "DIVIDE": 10,
            "JUMP": 8,
            "JUMPI": 10,
            "CALL": 40,
            "RETURN": 2
        }
        size_factor = len(str(input_data)) * 0.01  # Example: cost increases with size of input data
        return base_costs.get(operation, 1) + size_factor

    def create_token(self, creator_address, token_name, total_supply):
        if token_name not in self.token_balances:
            self.token_balances[token_name] = {creator_address: total_supply}
            self.save_state()
            return True
        return False

    def transfer_token(self, from_address, to_address, token_name, amount):
        if (token_name in self.token_balances and
            from_address in self.token_balances[token_name] and
            self.token_balances[token_name][from_address] >= amount):

            self.token_balances[token_name][from_address] -= amount
            self.token_balances[token_name][to_address] = self.token_balances[token_name].get(to_address, 0) + amount
            self.save_state()
            return True
        return False

    def create_nft(self, creator_address, nft_id, metadata):
        if nft_id not in self.nfts:
            self.nfts[nft_id] = {
                'owner': creator_address,
                'metadata': metadata
            }
            self.save_state()
            return True
        return False

    def transfer_nft(self, from_address, to_address, nft_id):
        if nft_id in self.nfts and self.nfts[nft_id]['owner'] == from_address:
            self.nfts[nft_id]['owner'] = to_address
            self.save_state()
            return True
        return False

    def create_rental_agreement(self, landlord, tenant, property_id, rent_amount, duration):
        agreement_id = self.generate_unique_id()
        self.contracts[agreement_id] = {
            'landlord': landlord,
            'tenant': tenant,
            'property_id': property_id,
            'rent_amount': rent_amount,
            'duration': duration,
            'start_time': self.current_block_time
        }
        return agreement_id

    def collect_rent(self, agreement_id):
        agreement = self.contracts.get(agreement_id)
        if agreement and self.current_block_time >= agreement['start_time'] + agreement['duration']:
            self.transfer_token(agreement['tenant'], agreement['landlord'], 'RENT_TOKEN', agreement['rent_amount'])
            return True
        return False

    def generate_unique_id(self):
        return hashlib.sha256(str(time.time()).encode()).hexdigest()

    def propose_transaction(self, transaction):
        self.consensus.propose(transaction)
