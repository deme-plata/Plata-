
from dataclasses import dataclass
from typing import List, Dict
from decimal import Decimal
from vm import SimpleVM, Permission, Role, PBFTConsensus
from typing import List, Tuple
from typing import List, Tuple, Dict, Any
from typing import List, Tuple, Dict, Any, Optional
from shared_logic import Transaction, NodeState
import os
import time
import logging
import threading
from concurrent import futures
import uvicorn
import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, padding, hashes
from cryptography.hazmat.backends import default_backend
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile,assemble
from qiskit.providers.jobstatus import JobStatus
from qiskit.exceptions import QiskitError   
from nacl.public import PrivateKey, Box
from nacl.utils import random
from typing import List
import json
from qiskit.circuit.random import random_circuit
import asyncio
import aiohttp
import networkx as nx

import base64
import hashlib
import numpy as np
from mnemonic import Mnemonic
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP
from cryptography.exceptions import InvalidSignature
import traceback 
import random
from Crypto.PublicKey import RSA
from Crypto.PublicKey import ECC
from Crypto.Signature import DSS
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_OAEP
from typing import List
from qiskit.circuit.random import random_circuit
import aiohttp
from vm import SimpleVM
from pydantic import BaseModel, Field
from hashlib import sha256
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import hashlib
from tqdm import tqdm  # for progress tracking
from contextlib import asynccontextmanager
from contextlib import asynccontextmanager
import os
import logging
import threading
import uvicorn

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import statistics
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import time
import os
import base64
import re  
import string
from base64 import urlsafe_b64encode
from decimal import Decimal
import json
from typing import Dict
from vm import SimpleVM, Permission, Role,    PBFTConsensus
from vm import SimpleVM
import pytest
import httpx

import curses
from pydantic import BaseModel, Field, validator  # Ensure validator is imported
from pydantic import BaseModel, field_validator  # Use field_validator for Pydantic V2
from pydantic import BaseModel, Field, root_validator
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Tuple
from starlette.websockets import WebSocketState
from collections import defaultdict
import uuid
from enum import Enum, auto
from pydantic import BaseModel, validator, Field, model_validator
import tracemalloc
from contextlib import asynccontextmanager
from vm import SimpleVM, Permission, Role, PBFTConsensus
from decimal import Decimal, InvalidOperation
from typing import Any, Dict  # Ensure Any and Dict are imported
from Order import Order
from SecureHybridZKStark import SecureHybridZKStark


from EnhancedOrderBook import EnhancedOrderBook
from zk_vm import ZKVM
from shared_logic import QuantumBlock, Transaction, NodeState

import curses
import socket
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import os
from user_management import fake_users_db
from secure_qr_system import SecureQRSystem
import json
from helius_integration import HeliusAPI
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed

from blockcypher_integration import BlockCypherAPI
from helius_integration import HeliusAPI
from web3 import Web3
import aiohttp
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Optional
global blockchain, p2p_node
import aioredis
from mongodb_manager import QuantumDAGKnightDB
from multisig_zkp import MultisigZKP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import asyncio
from typing import Dict, Set
import weakref
import signal
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import asyncio
import logging
from typing import Callable, Dict, Set
from contextlib import asynccontextmanager
from sanic import Sanic, Blueprint
from sanic.response import json
from sanic.server.protocols.websocket_protocol import WebSocketProtocol
from sanic.request import Request
from sanic.exceptions import SanicException
from sanic.log import logger as sanic_logger
from async_timeout import timeout
import asyncio
from sanic.response import json
from sanic.exceptions import SanicException
from sanic import Blueprint
import traceback
import logging
from P2PNode import P2PNode

import copy
from quantum_signer import QuantumSigner
from shared_logic import Transaction, QuantumBlock
from CryptoProvider import CryptoProvider
import logging
from DAGConfirmationSystem import DAGConfirmationSystem
import hashlib
import time
from decimal import Decimal
from typing import Dict, List, Tuple
from enum import Enum, auto
from Wallet import Wallet
from pydantic import BaseModel
from enhanced_exchange import EnhancedExchange,LiquidityPoolManager,PriceOracle,MarginAccount,AdvancedOrderTypes
from DAGKnightMiner import DAGKnightMiner
from DAGKnightGasSystem import EnhancedDAGKnightGasSystem, EnhancedGasTransactionType
from DAGPruningSystem import DAGPruningSystem
logger = logging.getLogger(__name__)
class QuantumDAGRewardSystem:
    """
    Advanced reward system for QuantumDAGKnight that incorporates quantum state verification,
    node participation, entanglement strength, and gradual emissions.
    """
    def __init__(self, initial_supply: int = 100_000_000):
        # Basic reward parameters
        self.total_supply = Decimal(initial_supply)
        self.current_supply = Decimal('0')
        self.base_reward = Decimal('1000')
        self.quantum_state_manager = QuantumStateManager()
        self.last_adjustment = time.time()

        # Enhanced reward components
        self.quantum_bonus_max = Decimal('200')  # Max quantum verification bonus
        self.dag_bonus_max = Decimal('150')      # Max DAG participation bonus
        self.stake_bonus_max = Decimal('100')    # Max staking bonus
        self.entanglement_bonus_max = Decimal('50') # Max quantum entanglement bonus

        # Reward schedule parameters  
        self.reward_reduction_interval = 259200   # ~3 days vs Bitcoin's 4 years
        self.reward_reduction_rate = Decimal('0.99')  # 1% reduction vs Bitcoin's 50%
        self.min_reward = Decimal('0.01')        # Minimum base reward
        
        # Network health metrics
        self.network_metrics = {
            'node_count': 0,
            'total_stake': Decimal('0'),
            'avg_quantum_signature_strength': Decimal('0'),
            'total_confirmations': 0,
            'dag_depth': 0
        }

        # Reward distribution history
        self.reward_history = []

    def calculate_quantum_bonus(self, quantum_signature: str, entanglement_strength: float) -> Decimal:
        """Calculate bonus based on quantum signature strength and entanglement"""
        # Verify quantum signature strength
        quantum_bits = sum(1 for bit in quantum_signature if bit == '1')
        sig_strength = Decimal(quantum_bits) / Decimal(len(quantum_signature))
        
        # Calculate entanglement component
        entangle_factor = Decimal(min(entanglement_strength, 1.0))
        
        # Combine both factors for final bonus
        quantum_bonus = self.quantum_bonus_max * sig_strength * entangle_factor
        return min(quantum_bonus, self.quantum_bonus_max)

    def calculate_dag_bonus(self, confirmation_metrics: dict) -> Decimal:
        """Calculate bonus based on DAG participation and confirmation strength"""
        confirmation_score = Decimal(confirmation_metrics.get('confirmation_score', 0))
        dag_depth = Decimal(confirmation_metrics.get('dag_depth', 0))
        path_diversity = Decimal(confirmation_metrics.get('path_diversity', 0))
        
        # Weighted combination of DAG metrics
        dag_factor = (confirmation_score * Decimal('0.4') + 
                     (dag_depth / Decimal('100')) * Decimal('0.3') + 
                     path_diversity * Decimal('0.3'))
        
        return min(self.dag_bonus_max * dag_factor, self.dag_bonus_max)

    def calculate_stake_bonus(self, miner_stake: Decimal, total_stake: Decimal) -> Decimal:
        """Calculate bonus based on miner's stake in the network"""
        if total_stake == 0:
            return Decimal('0')
            
        stake_ratio = miner_stake / total_stake
        return min(self.stake_bonus_max * stake_ratio, self.stake_bonus_max)

    def calculate_block_reward(self, block_data: dict) -> Decimal:
        """
        Calculate total block reward with advanced quantum metrics, network participation,
        and dynamic scaling factors.
        
        Args:
            block_data (dict): Block information including quantum metrics and network state
            
        Returns:
            Decimal: Calculated reward amount
        """
        try:
            # 1. Calculate Base Reward with Time-Based Reduction
            blocks_since_start = block_data.get('height', 0)
            current_time = time.time()
            time_factor = Decimal(str(math.exp(-0.05 * (current_time - self.last_adjustment) / (365 * 24 * 3600))))
            
            # Apply smooth reduction curve
            periods_elapsed = blocks_since_start // self.reward_reduction_interval
            reduction_factor = self.reward_reduction_rate ** periods_elapsed
            time_adjusted_factor = (reduction_factor + time_factor) / Decimal('2')
            base_reward = self.base_reward * time_adjusted_factor
            base_reward = max(base_reward, self.min_reward)

            # 2. Enhanced Quantum Bonuses
            quantum_metrics = {
                'signature': block_data['quantum_signature'],
                'entanglement': block_data.get('entanglement_strength', 0),
                'coherence': block_data.get('quantum_coherence', 0.5),
                'interference_pattern': block_data.get('interference_pattern', 0.5)
            }
            
            quantum_bonus = self.calculate_quantum_bonus(
                quantum_metrics['signature'],
                quantum_metrics['entanglement']
            )
            
            # Add coherence and interference bonuses
            coherence_bonus = self.quantum_bonus_max * Decimal(str(quantum_metrics['coherence'])) * Decimal('0.2')
            interference_bonus = self.quantum_bonus_max * Decimal(str(quantum_metrics['interference_pattern'])) * Decimal('0.1')
            total_quantum_bonus = quantum_bonus + coherence_bonus + interference_bonus

            # 3. Enhanced DAG Network Participation
            confirmation_metrics = block_data.get('confirmation_metrics', {})
            dag_participation = {
                'confirmation_score': confirmation_metrics.get('confirmation_score', 0),
                'path_diversity': confirmation_metrics.get('path_diversity', 0),
                'dag_depth': confirmation_metrics.get('dag_depth', 0),
                'cross_shard_refs': confirmation_metrics.get('cross_shard_references', 0),
                'validation_participation': confirmation_metrics.get('validation_participation', 0)
            }
            
            # Calculate comprehensive DAG bonus
            dag_bonus = self.calculate_dag_bonus(dag_participation)
            
            # Add validation participation bonus
            validation_bonus = self.dag_bonus_max * Decimal(str(dag_participation['validation_participation'])) * Decimal('0.15')
            total_dag_bonus = dag_bonus + validation_bonus

            # 4. Enhanced Staking and Participation Rewards
            stake_metrics = {
                'miner_stake': block_data.get('miner_stake', 0),
                'stake_duration': block_data.get('stake_duration', 0),
                'total_network_stake': self.network_metrics['total_stake'],
                'participation_score': block_data.get('participation_score', 0)
            }
            
            # Calculate stake bonus with time-weighted component
            base_stake_bonus = self.calculate_stake_bonus(
                stake_metrics['miner_stake'],
                stake_metrics['total_network_stake']
            )
            
            # Add duration bonus (longer staking gets better rewards)
            duration_factor = min(stake_metrics['stake_duration'] / (180 * 24 * 3600), 1)  # Max bonus at 180 days
            time_weighted_bonus = base_stake_bonus * Decimal(str(duration_factor)) * Decimal('0.2')
            total_stake_bonus = base_stake_bonus + time_weighted_bonus

            # 5. Network Health and Performance Metrics
            network_health = self._calculate_network_multiplier()
            
            # Add performance metrics
            performance_metrics = {
                'transaction_throughput': block_data.get('tx_throughput', 0) / 1000,  # Per thousand
                'block_propagation': block_data.get('block_propagation_time', 1),  # In seconds
                'network_latency': block_data.get('network_latency', 100)  # In milliseconds
            }
            
            # Calculate performance multiplier
            performance_score = (
                Decimal(str(min(performance_metrics['transaction_throughput'], 1))) * Decimal('0.4') +
                Decimal(str(1 / max(performance_metrics['block_propagation'], 0.1))) * Decimal('0.3') +
                Decimal(str(100 / max(performance_metrics['network_latency'], 1))) * Decimal('0.3')
            )
            
            health_multiplier = (network_health + performance_score) / Decimal('2')

            # 6. Calculate Final Reward
            components = {
                'base_reward': base_reward,
                'quantum_bonus': total_quantum_bonus,
                'dag_bonus': total_dag_bonus,
                'stake_bonus': total_stake_bonus
            }
            
            # Apply dynamic weighting based on network state
            weights = self._calculate_dynamic_weights(block_data)
            total_reward = sum(
                amount * weights[component] 
                for component, amount in components.items()
            )
            
            # Apply health multiplier with bounds
            total_reward *= min(max(health_multiplier, Decimal('0.5')), Decimal('2.0'))

            # 7. Supply Management
            remaining_supply = self.total_supply - self.current_supply
            total_reward = min(total_reward, remaining_supply)

            # 8. Record Distribution and Metrics
            self._update_network_metrics(block_data)
            self._record_reward_distribution(total_reward, {
                **components,
                'health_multiplier': health_multiplier,
                'performance_score': performance_score,
                'time_factor': time_factor
            })

            logger.info(f"Calculated block reward: {total_reward} QDAG")
            return total_reward

        except Exception as e:
            logger.error(f"Error calculating block reward: {str(e)}")
            logger.error(traceback.format_exc())
            return self.min_reward
    def calculate_quantum_strength(self, quantum_signature: str) -> float:
        """Calculate quantum signature strength"""
        if not quantum_signature:
            return 0.0
        return sum(1 for bit in quantum_signature if bit == '1') / len(quantum_signature)

    def get_stake(self, address: str) -> Decimal:
        """Get stake amount for an address"""
        return self.stakes.get(address, Decimal('0'))

    def get_total_stake(self) -> Decimal:
        """Get total stake in the network"""
        return sum(self.stakes.values())

    def calculate_average_quantum_strength(self) -> Decimal:
        """Calculate average quantum signature strength across recent blocks"""
        if not self.chain:
            return Decimal('0')
        
        recent_blocks = self.chain[-100:]  # Last 100 blocks
        strengths = [self.calculate_quantum_strength(block.quantum_signature) 
                    for block in recent_blocks]
        
        return Decimal(str(sum(strengths) / len(strengths))) if strengths else Decimal('0')

    def _calculate_dynamic_weights(self, block_data: dict) -> Dict[str, Decimal]:
        """Calculate dynamic component weights based on network state"""
        try:
            # Default weights
            weights = {
                'base_reward': Decimal('0.4'),
                'quantum_bonus': Decimal('0.25'),
                'dag_bonus': Decimal('0.2'),
                'stake_bonus': Decimal('0.15')
            }

            # Adjust weights based on network needs
            network_needs = {
                'quantum_strength_needed': block_data.get('avg_quantum_strength', 1) < 0.7,
                'dag_participation_needed': block_data.get('dag_participation_rate', 1) < 0.6,
                'stake_ratio_needed': block_data.get('stake_ratio', 0) < 0.4
            }

            # Increase weights for needed components
            adjustment = Decimal('0.05')
            remaining = Decimal('1.0')

            for need, required in network_needs.items():
                if required:
                    if 'quantum' in need:
                        weights['quantum_bonus'] += adjustment
                    elif 'dag' in need:
                        weights['dag_bonus'] += adjustment
                    elif 'stake' in need:
                        weights['stake_bonus'] += adjustment
                    remaining -= adjustment

            # Normalize weights
            total_weight = sum(weights.values())
            return {k: (v / total_weight) for k, v in weights.items()}

        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {str(e)}")
            return {
                'base_reward': Decimal('0.4'),
                'quantum_bonus': Decimal('0.25'),
                'dag_bonus': Decimal('0.2'),
                'stake_bonus': Decimal('0.15')
            }

    def _calculate_network_multiplier(self) -> Decimal:
        """Calculate reward multiplier based on network health"""
        try:
            # Calculate health factors
            node_factor = min(self.network_metrics['node_count'] / 1000, 1)
            stake_factor = min(self.network_metrics['total_stake'] / (self.total_supply / 2), 1)
            quantum_factor = self.network_metrics['avg_quantum_signature_strength']
            
            # Weighted combination of health metrics
            health_score = (Decimal(node_factor) * Decimal('0.3') + 
                          Decimal(stake_factor) * Decimal('0.3') + 
                          Decimal(quantum_factor) * Decimal('0.4'))
            
            # Convert to multiplier (0.8 to 1.2 range)
            return Decimal('0.8') + (health_score * Decimal('0.4'))
        
        except Exception as e:
            logger.error(f"Error calculating network multiplier: {str(e)}")
            return Decimal('1.0')

    def _update_network_metrics(self, block_data: dict):
        """Update network health metrics"""
        self.network_metrics.update({
            'node_count': block_data.get('active_nodes', self.network_metrics['node_count']),
            'total_stake': block_data.get('total_stake', self.network_metrics['total_stake']),
            'avg_quantum_signature_strength': block_data.get('avg_quantum_strength', 
                self.network_metrics['avg_quantum_signature_strength']),
            'total_confirmations': self.network_metrics['total_confirmations'] + 1,
            'dag_depth': block_data.get('dag_depth', self.network_metrics['dag_depth'])
        })

    def _record_reward_distribution(self, total_reward: Decimal, components: dict):
        """Record reward distribution for analysis"""
        self.reward_history.append({
            'timestamp': time.time(),
            'total_reward': total_reward,
            'components': components,
            'network_metrics': self.network_metrics.copy()
        })

    def get_reward_statistics(self) -> dict:
        """Get reward distribution statistics"""
        if not self.reward_history:
            return {}
            
        recent_rewards = [r['total_reward'] for r in self.reward_history[-1000:]]
        return {
            'average_reward': sum(recent_rewards) / len(recent_rewards),
            'min_reward': min(recent_rewards),
            'max_reward': max(recent_rewards),
            'total_rewards_distributed': self.current_supply,
            'remaining_supply': self.total_supply - self.current_supply
        }

class Permission(Enum):
    MINT = auto()
    BURN = auto()
    TRANSFER = auto()
class Token:
    def __init__(self, address: str, name: str, symbol: str, creator: str):
        self.address = address
        self.name = name
        self.symbol = symbol
        self.creator = creator
        self.balances: Dict[str, Decimal] = {}
        self.authorized_minters: set = {creator}
        self.authorized_burners: set = {creator}
        self.total_supply = Decimal('0')

    def balance_of(self, user: str) -> Decimal:
        return self.balances.get(user, Decimal('0'))

    def mint(self, user: str, amount: Decimal):
        if user not in self.authorized_minters:
            raise ValueError("Unauthorized to mint tokens")
        self.balances[user] = self.balance_of(user) + amount
        self.total_supply += amount

    def burn(self, user: str, amount: Decimal):
        if user not in self.authorized_burners:
            raise ValueError("Unauthorized to burn tokens")
        if self.balance_of(user) < amount:
            raise ValueError("Insufficient balance to burn")
        self.balances[user] -= amount
        self.total_supply -= amount

    def transfer(self, sender: str, recipient: str, amount: Decimal):
        if self.balance_of(sender) < amount:
            raise ValueError("Insufficient balance to transfer")
        self.balances[sender] -= amount
        self.balances[recipient] = self.balance_of(recipient) + amount

    def authorize_minter(self, user: str):
        self.authorized_minters.add(user)

    def revoke_minter(self, user: str):
        if user != self.creator:
            self.authorized_minters.discard(user)

    def authorize_burner(self, user: str):
        self.authorized_burners.add(user)

    def revoke_burner(self, user: str):
        if user != self.creator:
            self.authorized_burners.discard(user)
            
class QuantumStateManager:
    def __init__(self):
        self.shards = {}

    def store_quantum_state(self, shard_id, quantum_state):
        self.shards[shard_id] = quantum_state

    def retrieve_quantum_state(self, shard_id):
        return self.shards.get(shard_id)


quantum_state_manager = QuantumStateManager()


class SecurityManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def create_token(self, node_id, expiration_time=1):
        payload = {
            'exp': datetime.utcnow() + timedelta(hours=expiration_time),
            'iat': datetime.utcnow(),
            'sub': node_id
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def sign_message(self, message):
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    def verify_signature(self, message, signature):
        try:
            self.public_key.verify(
                signature,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False

# Constants
MAX_SUPPLY = 21_000_000  # Maximum number of coins that can ever exist
INITIAL_REWARD = 1000    # Initial mining reward
HALVING_INTERVAL = 4 * 365 * 24 * 3600  # 4 years in seconds
TARGET_BLOCK_TIME = 600  # 10 minutes in seconds
class QuantumBlockchain:
    def __init__(self, consensus, secret_key, node_directory, vm, p2p_node=None):
        self.globalMetrics = {
            'totalTransactions': 0,
            'totalBlocks': 0,
        }
        self.initial_reward = 1000
        self.chain = []
        self.pending_transactions = []
        self.consensus = consensus
        if vm is None:  
            raise ValueError("VM cannot be None. Check SimpleVM initialization.")
        self.vm = vm
        self.secret_key = secret_key
        self.node_directory = node_directory
        self.halving_interval = 4 * 365 * 24 * 3600  # 4 years in seconds
        self.start_time = time.time()
        self.difficulty = 1
        self.target_block_time = 600  # 10 minutes in seconds
        self.adjustment_interval = 10
        self.max_supply = MAX_SUPPLY
        self.target = 2**(256 - self.difficulty)
        self.blocks_since_last_adjustment = 0
        self.security_manager = SecurityManager(secret_key)
        self.quantum_state_manager = QuantumStateManager()
        self.peers = []
        self.new_block_listeners = []
        self.new_transaction_listeners = []
        self.genesis_wallet_address = "genesis_wallet"
        self.balances = {}  # Initialize balances as a dictionary
        self.wallets = []
        self.transactions = []
        self.contracts = []
        self.tokens = {}
        self.liquidity_pool_manager = LiquidityPoolManager()
        self.zk_system = SecureHybridZKStark(security_level=2)  # Adjust security level as needed

        # Initialize the P2P node and lock
        if p2p_node is None:
            self.p2p_node = P2PNode(blockchain=self)
        else:
            self.p2p_node = p2p_node
            self.p2p_node.blockchain = self  # Set blockchain reference

        logger.info(f"QuantumBlockchain initialized with p2p_node: {self.p2p_node}")

        self.db = QuantumDAGKnightDB("mongodb://localhost:27017", "quantumdagknight_db")
        self.initialized = False



        self._p2p_node_lock = asyncio.Lock()  # Lock for accessing the p2p_node
        logger.info(f"QuantumBlockchain initialized with p2p_node: {self.p2p_node}")
        
        if self.p2p_node is None:
            logger.warning("P2P node is None in QuantumBlockchain initialization")
        else:
            logger.info(f"P2P node type: {type(self.p2p_node)}")
            logger.info(f"P2P node attributes: {vars(self.p2p_node)}")
        try:
            miner = DAGKnightMiner(difficulty=2, security_level=20)
            # Verify NetworkX functionality
            if miner.verify_networkx_installation():
                print("Miner initialized successfully with NetworkX")
            else:
                print("NetworkX verification failed")
        except Exception as e:
            print(f"Error initializing miner: {str(e)}")
        self.confirmation_system = DAGConfirmationSystem(
        quantum_threshold=0.85,  # Minimum quantum signature strength required
        min_confirmations=6,     # Minimum confirmations needed
        max_confirmations=100    # Maximum confirmations to track
        )
        
        # Add confirmation tracking to metrics
        self.globalMetrics.update({
            'averageConfirmationScore': 0.0,
            'highSecurityTransactions': 0,
            'totalConfirmedTransactions': 0
        })

        self._mempool = [] 
        self.gas_system = EnhancedDAGKnightGasSystem()
        self.network_state = {
            'avg_block_time': 30.0,
            'network_load': 0.0,
            'active_nodes': 0,
            'quantum_entangled_pairs': 0,
            'dag_depth': 0,
            'total_compute': 0.0
        }
        self.miner_rewards_pool = Decimal('0')
        self.gas_metrics = {}
        self.last_gas_update = time.time()
        self.gas_update_interval = 60  # Update every minute
        self.reward_system = QuantumDAGRewardSystem(initial_supply=MAX_SUPPLY)
        self.pruning_system = DAGPruningSystem(
            min_confirmations=6,
            quantum_threshold=0.85,
            prune_interval=1000,
            max_dag_size=10000,
            min_security_level="HIGH"
        )
        
        self.last_prune_time = time.time()
        self.pruning_metrics = {
            'total_nodes_pruned': 0,
            'pruning_operations': 0,
            'avg_prune_time': 0.0,
            'saved_memory': 0
        }
        
        # Add pruning lock
        self._pruning_lock = asyncio.Lock()

    async def update_gas_metrics(self):
        """Update gas system metrics based on current blockchain state"""
        try:
            current_time = time.time()
            if current_time - self.last_gas_update < self.gas_update_interval:
                return

            # Calculate network metrics
            self.network_state.update({
                'avg_block_time': self._calculate_avg_block_time(),
                'network_load': len(self.pending_transactions) / 1000,  # Assuming 1000 tx capacity
                'active_nodes': len(self.p2p_node.connected_peers) if self.p2p_node else 0,
                'quantum_entangled_pairs': self._count_quantum_entanglements(),
                'dag_depth': len(self.chain),
                'total_compute': self._calculate_total_compute()
            })

            # Update gas system with new metrics
            await self.gas_system.update_network_metrics(self.network_state)
            self.last_gas_update = current_time

        except Exception as e:
            logger.error(f"Error updating gas metrics: {str(e)}")
            logger.error(traceback.format_exc())

    async def estimate_transaction_gas(self, tx_data: dict) -> dict:
        """Estimate gas for a transaction with quantum-aware pricing"""
        try:
            # Determine transaction type
            tx_type = self._get_transaction_type(tx_data)
            
            # Check for quantum features
            quantum_enabled = tx_data.get('quantum_enabled', False)
            entanglement_count = tx_data.get('entanglement_count', 0)
            
            # Calculate data size
            data_size = len(str(tx_data).encode())
            
            # Get gas estimate
            total_gas, gas_price = await self.gas_system.calculate_gas(
                tx_type=tx_type,
                data_size=data_size,
                quantum_enabled=quantum_enabled,
                entanglement_count=entanglement_count
            )
            
            return {
                'gas_needed': total_gas,
                'gas_price': float(gas_price.total),
                'total_cost': float(Decimal(str(total_gas)) * gas_price.total),
                'components': {
                    'base_price': float(gas_price.base_price),
                    'security_premium': float(gas_price.security_premium),
                    'quantum_premium': float(gas_price.quantum_premium),
                    'entanglement_premium': float(gas_price.entanglement_premium),
                    'decoherence_discount': float(gas_price.decoherence_discount),
                    'congestion_premium': float(gas_price.congestion_premium)
                }
            }
        except Exception as e:
            logger.error(f"Error estimating transaction gas: {str(e)}")
            raise

    def _get_transaction_type(self, tx_data: dict) -> EnhancedGasTransactionType:
        """Determine transaction type with quantum awareness"""
        if tx_data.get('quantum_proof'):
            return EnhancedGasTransactionType.QUANTUM_PROOF
        elif tx_data.get('quantum_entangle'):
            return EnhancedGasTransactionType.QUANTUM_ENTANGLE
        elif tx_data.get('quantum_state'):
            return EnhancedGasTransactionType.QUANTUM_STATE
        elif tx_data.get('smart_contract'):
            return EnhancedGasTransactionType.SMART_CONTRACT
        elif tx_data.get('dag_reorg'):
            return EnhancedGasTransactionType.DAG_REORG
        elif tx_data.get('data'):
            return EnhancedGasTransactionType.DATA_STORAGE
        else:
            return EnhancedGasTransactionType.STANDARD

    async def add_transaction(self, transaction: Transaction) -> bool:
        """Add a transaction with enhanced gas handling"""
        try:
            # Update gas metrics before processing
            await self.update_gas_metrics()

            # Estimate gas
            gas_estimate = await self.estimate_transaction_gas(transaction.to_dict())
            
            # Check if sender has enough balance for gas
            sender_balance = self.get_balance(transaction.sender)
            gas_cost = Decimal(str(gas_estimate['total_cost']))
            
            if sender_balance < gas_cost + transaction.amount:
                logger.warning(f"Insufficient balance for transaction and gas: {transaction.sender}")
                return False
            
            # Add gas information to transaction
            transaction.gas_used = gas_estimate['gas_needed']
            transaction.gas_price = Decimal(str(gas_estimate['gas_price']))
            
            # Process quantum verification if needed
            if transaction.quantum_enabled:
                await self._process_quantum_verification(transaction)
            
            # Process transaction
            success = await super().add_transaction(transaction)
            
            if success:
                # Deduct gas cost and update rewards pool
                self.balances[transaction.sender] -= gas_cost
                self.miner_rewards_pool += gas_cost
                
                # Track gas metrics
                self._track_gas_metrics(transaction, gas_estimate)
                
            return success
            
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            return False

    async def _process_quantum_verification(self, transaction: Transaction):
        """Process quantum verification for transaction"""
        if transaction.quantum_enabled:
            try:
                # Verify quantum state if present
                if hasattr(transaction, 'quantum_state'):
                    success = await self.quantum_state_manager.verify_state(
                        transaction.quantum_state
                    )
                    if not success:
                        raise ValueError("Invalid quantum state")

                # Verify entanglement if present
                if hasattr(transaction, 'entanglement_proof'):
                    success = await self.quantum_state_manager.verify_entanglement(
                        transaction.entanglement_proof
                    )
                    if not success:
                        raise ValueError("Invalid entanglement proof")

            except Exception as e:
                logger.error(f"Quantum verification failed: {str(e)}")
                raise

    def _track_gas_metrics(self, transaction: Transaction, gas_estimate: dict):
        """Track gas usage metrics"""
        tx_type = self._get_transaction_type(transaction.to_dict())
        
        if tx_type not in self.gas_metrics:
            self.gas_metrics[tx_type] = {
                'count': 0,
                'total_gas': 0,
                'total_cost': Decimal('0'),
                'quantum_premiums': [],
                'entanglement_premiums': []
            }
        
        metrics = self.gas_metrics[tx_type]
        metrics['count'] += 1
        metrics['total_gas'] += gas_estimate['gas_needed']
        metrics['total_cost'] += Decimal(str(gas_estimate['total_cost']))
        
        if transaction.quantum_enabled:
            metrics['quantum_premiums'].append(
                gas_estimate['components']['quantum_premium']
            )
            
        if hasattr(transaction, 'entanglement_count'):
            metrics['entanglement_premiums'].append(
                gas_estimate['components']['entanglement_premium']
            )



    def get_gas_metrics(self) -> dict:
        """Get current gas usage metrics"""
        metrics = {
            'total_gas_used': sum(m['total_gas'] for m in self.gas_metrics.values()),
            'total_gas_cost': sum(m['total_cost'] for m in self.gas_metrics.values()),
            'transaction_types': {
                tx_type.value: {
                    'count': m['count'],
                    'avg_gas': m['total_gas'] / m['count'] if m['count'] > 0 else 0,
                    'avg_cost': float(m['total_cost'] / m['count']) if m['count'] > 0 else 0,
                    'quantum_premium_avg': np.mean(m['quantum_premiums']) if m['quantum_premiums'] else 0,
                    'entanglement_premium_avg': np.mean(m['entanglement_premiums']) if m['entanglement_premiums'] else 0
                }
                for tx_type, m in self.gas_metrics.items()
            },
            'network_state': self.network_state
        }
        return metrics

    async def create_block(self, data, transactions, miner_address):
        """Create a new block with enhanced gas handling"""
        try:
            logger.info("Creating a new block...")
            
            # Update gas metrics before block creation
            await self.update_gas_metrics()
            
            # Get previous hash
            previous_hash = self.chain[-1].hash if self.chain else "0"
            
            # Calculate base reward
            base_reward = self.get_block_reward()
            
            # Calculate gas rewards
            total_gas_reward = self.miner_rewards_pool
            
            # Generate quantum signature
            quantum_signature = self.generate_quantum_signature()
            
            # Process transactions with gas costs
            processed_transactions = []
            total_gas_used = 0
            
            for tx in transactions:
                try:
                    # Convert raw transaction to Transaction object if needed
                    if isinstance(tx, dict):
                        transaction = Transaction(**tx)
                    else:
                        transaction = tx
                    
                    # Get gas estimate for the transaction
                    gas_estimate = await self.estimate_transaction_gas(transaction.to_dict())
                    
                    # Add gas information to transaction
                    transaction.gas_used = gas_estimate['gas_needed']
                    transaction.gas_price = Decimal(str(gas_estimate['gas_price']))
                    
                    # Update total gas used
                    total_gas_used += transaction.gas_used
                    
                    processed_transactions.append(transaction)
                    
                except Exception as e:
                    logger.error(f"Error processing transaction in block creation: {str(e)}")
                    continue

            # Create new block with gas information
            new_block = QuantumBlock(
                previous_hash=previous_hash,
                data=data,
                quantum_signature=quantum_signature,
                reward=base_reward + total_gas_reward,
                transactions=processed_transactions
            )
            
            # Add gas metadata to block
            new_block.total_gas_used = total_gas_used
            new_block.gas_reward = total_gas_reward
            
            # Compute and set block hash
            new_block.hash = new_block.compute_hash()
            logger.debug(f"Initial block hash: {new_block.hash}")
            
            # Mine the block
            new_block.mine_block(self.difficulty)
            logger.info(f"Block mined. Hash: {new_block.hash}")
            
            # Validate block
            if self.consensus.validate_block(new_block):
                logger.info(f"Block validated. Adding to chain: {new_block.hash}")
                
                # Add block to chain
                self.chain.append(new_block)
                
                # Process transactions and update balances
                self.process_transactions(new_block.transactions)
                
                # Reward miner with both block reward and gas fees
                total_reward = base_reward + total_gas_reward
                self.reward_miner(miner_address, total_reward)
                
                # Reset gas rewards pool
                self.miner_rewards_pool = Decimal('0')
                
                # Update gas metrics after block creation
                await self.update_gas_metrics()
                
                return new_block
            else:
                logger.error("Block validation failed. Block will not be added.")
                return None
                
        except Exception as e:
            logger.error(f"Exception in create_block: {str(e)}")
            logger.error(traceback.format_exc())
            return None


    def get_wallets(self) -> List[Wallet]:
        """
        Get all registered wallets.
        
        Returns:
            List[Wallet]: List of all registered wallet objects
        """
        try:
            return list(self._wallets.values())
        except Exception as e:
            self.logger.error(f"Error retrieving wallets: {str(e)}")
            return []

    def add_wallet(self, wallet: Wallet) -> bool:
        """
        Add a new wallet to the blockchain.
        
        Args:
            wallet (Wallet): Wallet object to add
            
        Returns:
            bool: True if wallet was added successfully, False otherwise
        """
        try:
            if not wallet.address:
                self.logger.error("Cannot add wallet without address")
                return False

            if wallet.address in self._wallets:
                self.logger.warning(f"Wallet {wallet.address} already exists")
                return False

            # Validate wallet object
            if not wallet.public_key or not wallet.address.startswith('plata'):
                self.logger.error("Invalid wallet object")
                return False

            self._wallets[wallet.address] = wallet
            self.logger.info(f"Added new wallet: {wallet.address}")
            return True

        except Exception as e:
            self.logger.error(f"Error adding wallet: {str(e)}")
            return False

    def get_wallet(self, address: str) -> Optional[Wallet]:
        """
        Get a specific wallet by address.
        
        Args:
            address (str): The wallet address to look up
            
        Returns:
            Optional[Wallet]: The wallet if found, None otherwise
        """
        try:
            return self._wallets.get(address)
        except Exception as e:
            self.logger.error(f"Error retrieving wallet {address}: {str(e)}")
            return None

    async def get_recent_transactions(self, limit: int = 100) -> List[Transaction]:
        """
        Get recent transactions from the blockchain.
        
        Args:
            limit (int, optional): Maximum number of transactions to return. Defaults to 100.
            
        Returns:
            List[Transaction]: List of recent transactions
        """
        try:
            transactions = []
            # Get transactions from recent blocks
            for block in reversed(self.chain[-10:]):  # Look at last 10 blocks
                transactions.extend(block.transactions)
            # Add mempool transactions
            transactions.extend(self.mempool)
            return transactions[-limit:]  # Return most recent transactions up to limit
        except Exception as e:
            self.logger.error(f"Error getting recent transactions: {str(e)}")
            return []


    def remove_wallet(self, address: str) -> bool:
        """
        Remove a wallet from the blockchain.
        
        Args:
            address (str): Address of wallet to remove
            
        Returns:
            bool: True if wallet was removed, False otherwise
        """
        try:
            if address in self._wallets:
                del self._wallets[address]
                self.logger.info(f"Removed wallet: {address}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing wallet {address}: {str(e)}")
            return False

    def update_wallet(self, wallet: Wallet) -> bool:
        """
        Update an existing wallet.
        
        Args:
            wallet (Wallet): Updated wallet object
            
        Returns:
            bool: True if wallet was updated, False otherwise
        """
        try:
            if wallet.address in self._wallets:
                self._wallets[wallet.address] = wallet
                self.logger.info(f"Updated wallet: {wallet.address}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating wallet: {str(e)}")
            return False


    def get_recent_transactions(self, limit: int = 100) -> List[Transaction]:
        """
        Get recent transactions from the blockchain.
        
        Args:
            limit (int): Maximum number of transactions to return
            
        Returns:
            List[Transaction]: List of recent transactions
        """
        try:
            transactions = []
            # Get transactions from recent blocks
            for block in reversed(self.chain[-10:]):  # Look at last 10 blocks
                transactions.extend(block.transactions)
            # Add mempool transactions
            transactions.extend(self.mempool)
            return transactions[-limit:]  # Return most recent transactions up to limit
        except Exception as e:
            logger.error(f"Error getting recent transactions: {str(e)}")
            return []
    @property
    def mempool(self) -> List[Transaction]:
        """
        Get the current mempool.
        
        Returns:
            List[Transaction]: List of pending transactions
        """
        if not hasattr(self, '_mempool'):
            self._mempool = []
        return self._mempool

    @mempool.setter
    def mempool(self, value: List[Transaction]):
        """Set the mempool."""
        self._mempool = value

    def add_to_mempool(self, transaction: Transaction) -> bool:
        """
        Add a transaction to the mempool.
        
        Args:
            transaction (Transaction): Transaction to add
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            if transaction not in self._mempool:
                self._mempool.append(transaction)
                logger.info(f"Transaction {transaction.hash} added to mempool")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding transaction to mempool: {str(e)}")
            return False

    def remove_from_mempool(self, transaction: Transaction) -> bool:
        """
        Remove a transaction from the mempool.
        
        Args:
            transaction (Transaction): Transaction to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        try:
            if transaction in self._mempool:
                self._mempool.remove(transaction)
                logger.info(f"Transaction {transaction.hash} removed from mempool")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing transaction from mempool: {str(e)}")
            return False

    def clear_mempool(self):
        """Clear all transactions from the mempool."""
        self._mempool = []
        logger.info("Mempool cleared")

    def get_mempool_size(self) -> int:
        """
        Get the current size of the mempool.
        
        Returns:
            int: Number of transactions in the mempool
        """
        return len(self._mempool)

    async def initialize_async_components(self):
        await self.db.init_collections()
        self.initialized = True
        logging.info("QuantumBlockchain initialized successfully with database collections.")

    async def create_multisig_wallet(self, public_keys: List[str], threshold: int) -> str:
        multisig_zkp = app.state.multisig_zkp
        return multisig_zkp.create_multisig(public_keys, threshold)

    async def add_multisig_transaction(self, multisig_address: str, sender_public_keys: List[int], threshold: int, receiver: str, amount: Decimal, message: str, aggregate_proof: Tuple[List[int], List[int], List[Tuple[int, List[int]]]]):
        multisig_zkp = app.state.multisig_zkp
        
        if not multisig_zkp.verify_multisig(sender_public_keys, threshold, message, aggregate_proof):
            raise ValueError("Invalid multisig transaction proof")

        # Create and add the transaction to the blockchain
        tx = Transaction(
            sender=multisig_address,
            receiver=receiver,
            amount=amount,
            timestamp=int(time.time())
        )
        await self.add_transaction(tx)
        return tx.hash()

    async def create_wallet(self, user_id: str) -> Dict[str, Any]:
        if user_id in self.wallets:
            raise ValueError(f"Wallet already exists for user {user_id}")

        new_wallet = Wallet()
        self.wallets[user_id] = new_wallet

        wallet_info = {
            'user_id': user_id,
            'address': new_wallet.address,
            'public_key': new_wallet.public_key
        }

        if self.p2p_node:
            await self.p2p_node.broadcast_event('new_wallet', wallet_info)

        return wallet_info
    async def create_transaction(self, sender: str, receiver: str, amount: Decimal, 
                                 price: Decimal, buyer_id: str, seller_id: str,
                                 wallet: Any) -> Optional[Transaction]:
        """Create a transaction with enhanced security."""
        try:
            # Initialize transaction
            tx = Transaction(
                sender=sender,
                receiver=receiver,
                amount=amount,
                price=price,
                buyer_id=buyer_id,
                seller_id=seller_id,
                wallet=wallet
            )
            
            # Apply enhanced security (ZKP, encryption, etc.)
            if not await tx.apply_enhanced_security(self.crypto_provider):
                logger.error("Failed to apply enhanced security to transaction")
                return None

            # Verify enhanced security for the transaction
            if not await tx.verify_enhanced_security(self.crypto_provider):
                logger.error("Transaction failed security verification")
                return None

            # Add the transaction to pending transactions
            self.pending_transactions.append(tx)
            return tx

        except Exception as e:
            logger.error(f"Error creating transaction: {str(e)}")
            return None

    async def verify_transaction(self, tx: Transaction) -> bool:
        """Verify a transaction with enhanced security."""
        try:
            return await tx.verify_enhanced_security(self.crypto_provider)
        except Exception as e:
            logger.error(f"Transaction verification error: {str(e)}")
            return False

    async def process_block_transactions(self, block: 'QuantumBlock'):
        """Process transactions with enhanced security verification."""
        for tx in block.transactions:
            if not await self.verify_transaction(tx):
                raise ValueError(f"Invalid transaction in block: {tx.id}")
                
            await self.apply_transaction(tx)

    async def create_private_transaction(self, sender: str, receiver: str, amount: Decimal) -> Dict[str, Any]:
        if sender not in self.wallets or receiver not in self.wallets:
            raise ValueError("Sender or receiver wallet not found")

        sender_wallet = self.wallets[sender]
        receiver_wallet = self.wallets[receiver]

        # Create and sign the transaction
        tx = Transaction(sender_wallet.address, receiver_wallet.address, amount)
        tx.sign(sender_wallet.private_key)

        # Generate ZKP
        secret = int(amount * 10**18)  # Convert Decimal to integer
        public_input = int(tx.hash(), 16)
        zk_proof = self.zk_system.prove(secret, public_input)

        # Add ZKP to transaction
        tx.zk_proof = zk_proof

        # Add transaction to blockchain
        await self.add_transaction(tx)

        tx_info = {
            'tx_hash': tx.hash,
            'sender': sender,
            'receiver': receiver,
            'amount': str(amount),  # Convert Decimal to string for JSON serialization
            'zk_proof': self.serialize_proof(zk_proof)
        }

        if self.p2p_node:
            await self.p2p_node.broadcast_event('private_transaction', tx_info)

        return tx_info

    def serialize_proof(self, proof):
        # Implement this method to serialize the ZKP for network transmission
        # This will depend on the specific structure of your ZKP
        pass



            
    def get_wallets(self):
        """
        Returns a list of all wallets in the blockchain as dictionaries.
        """
        try:
            # Check if wallets exist in the system
            if not self.wallets:
                logger.info("No wallets found in the blockchain.")
                return []

            logger.debug(f"Fetched {len(self.wallets)} wallets from the blockchain.")

            # Return a list of dictionaries representing each wallet
            return [wallet.to_dict() for wallet in self.wallets]

        except Exception as e:
            logger.error(f"Error fetching wallets: {str(e)}")
            raise ValueError("Failed to retrieve wallets.")


    async def propagate_block_with_retry(self, block, retries=3):
        for attempt in range(retries):
            try:
                logger.debug(f"Attempting to propagate block: {block.hash}. Retries left: {retries - attempt}")

                # Retrieve the P2PNode from app.state
                

                # Proceed with block propagation
                await p2p_node.propagate_block(block)
                logger.info(f"Block {block.hash} successfully propagated.")
                return True

            except Exception as e:
                logger.error(f"Failed to propagate block: {block.hash}. Error: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

        logger.error(f"Max retries reached. Block {block.hash} could not be propagated.")
        return False


    async def get_p2p_node(self):
        """ Get the P2P node with locking to ensure no race conditions. """
        async with self._p2p_node_lock:
            logger.debug("Accessing p2p_node via get_p2p_node method")
            return self.p2p_node

    async def set_p2p_node(self, p2p_node):
        """ Set the P2P node with locking and detailed logging. """
        async with self._p2p_node_lock:
            self._p2p_node = p2p_node
            if self._p2p_node is not None:
                try:
                    logger.info(f"P2P node set successfully. Type: {type(self._p2p_node)}")
                    logger.info(f"P2P node attributes: {vars(self._p2p_node)}")
                    logger.info(f"P2P node methods: {dir(self._p2p_node)}")

                    # Log if P2P node is connected
                    if hasattr(self._p2p_node, 'is_connected'):
                        is_connected = await self._p2p_node.is_connected()
                        logger.info(f"P2P node connected: {is_connected}")
                    else:
                        logger.warning("P2P node does not have 'is_connected' method")

                    # Log number of peers
                    if hasattr(self._p2p_node, 'peers'):
                        logger.info(f"P2P node peers: {len(self._p2p_node.peers)}")
                    else:
                        logger.warning("P2P node does not have 'peers' attribute")

                except Exception as e:
                    logger.error(f"Error while setting P2P node: {str(e)}")
                    logger.error("Traceback for P2P node setting error:")
                    logger.error(traceback.format_exc())
            else:
                logger.error("Attempted to set P2P node, but it's None")
                logger.error("Traceback for setting None P2P node:")
                logger.error(traceback.format_exc())


    async def add_peer(self, peer):
        if peer not in self.peers:
            self.peers.append(peer)
            logger.info(f"Peer {peer} added to peers list")

    async def remove_peer(self, peer):
        if peer in self.peers:
            self.peers.remove(peer)
            logger.info(f"Peer {peer} removed from peers list")

    def get_latest_block_hash(self):
        if not self.chain:
            return "0"  # Return a default value for the first block (genesis block)
        return self.chain[-1].hash  # Return the hash of the latest block
    async def check_p2p_node_status(self):
        while True:
            logger.info(f"Checking P2P node status: {self.p2p_node}")
            if self.p2p_node is None:
                logger.warning("P2P node is None in QuantumBlockchain")
            else:
                logger.info(f"P2P node is connected: {self.p2p_node.is_connected()}")
                logger.info(f"P2P node peers: {self.p2p_node.peers}")
            await asyncio.sleep(60)  # Check every minute
    def get_pending_transactions(self):
        # Return the list of pending transactions
        return self.pending_transactions
    
    def add_transaction(self, transaction):
        # Add a new transaction to the list of pending transactions
        self.pending_transactions.append(transaction)
    def reward_miner(self, miner_address, reward):
        # Assuming balances is a dictionary storing the balance for each wallet address
        if miner_address in self.balances:
            self.balances[miner_address] += reward
        else:
            self.balances[miner_address] = reward

        logger.info(f"Rewarded miner {miner_address} with {reward} QuantumDAGKnight Coins.")

    def create_block(self, data, transactions, miner_address):
        try:
            logger.info("Creating a new block...")
            previous_hash = self.chain[-1].hash if self.chain else "0"
            reward = self.get_block_reward()
            quantum_signature = self.generate_quantum_signature()

            new_block = QuantumBlock(
                previous_hash=previous_hash,
                data=data,
                quantum_signature=quantum_signature,
                reward=reward,
                transactions=transactions
            )
            new_block.hash = new_block.compute_hash()  # Set initial hash

            logger.debug(f"Initial block hash: {new_block.hash}")

            new_block.mine_block(self.difficulty)
            logger.info(f"Block mined. Hash: {new_block.hash}")

            if self.consensus.validate_block(new_block):
                logger.info(f"Block validated. Adding to chain: {new_block.hash}")
                self.chain.append(new_block)
                self.process_transactions(new_block.transactions)
                self.reward_miner(miner_address, reward)
                return new_block
            else:
                logger.error("Block validation failed. Block will not be added.")
                return None
        except Exception as e:
            logger.error(f"Exception in create_block: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def get_transactions(self, wallet_address: str) -> List[Dict]:
        # Simulate an asynchronous operation
        await asyncio.sleep(0)
        # For now, let's return an empty list
        return []





    async def get_balances(self, address: str) -> Dict[str, Decimal]:
        balances = {}
        try:
            balances['PLATA'] = await self.get_plata_balance(address)
            
            btc_address = self.get_btc_address(address)
            if btc_address:
                balances['BTC'] = await self.get_btc_balance(btc_address)
            
            eth_address = self.get_eth_address(address)
            if eth_address:
                balances['ETH'] = await self.get_eth_balance(eth_address)
            
            ltc_address = self.get_ltc_address(address)
            if ltc_address:
                balances['LTC'] = await self.get_ltc_balance(ltc_address)
            
            doge_address = self.get_doge_address(address)
            if doge_address:
                balances['DOGE'] = await self.get_doge_balance(doge_address)
            
            sol_address = self.get_sol_address(address)
            if sol_address:
                balances['SOL'] = await self.get_sol_balance(sol_address)

            logger.info(f"Fetched balances for address {address}: {balances}")
        except Exception as e:
            logger.error(f"Error fetching balances for address {address}: {str(e)}")
            logger.error(traceback.format_exc())
        
        return balances

    async def get_plata_balance(self, address: str) -> Decimal:
        # Implement the logic to get PLATA balance
        balance = self.balances.get(address, 0)
        return Decimal(str(balance))

    async def get_btc_balance(self, address: str) -> Decimal:
        try:
            btc_info = self.blockcypher_api.get_address_info(address)
            return Decimal(btc_info['balance'] / 1e8)  # Convert satoshis to BTC
        except Exception as e:
            logger.error(f"Error fetching BTC balance for {address}: {str(e)}")
            return Decimal('0')

    def get_btc_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive BTC address from Plata address
        return None

    def get_eth_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive ETH address from Plata address
        return None

    def get_ltc_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive LTC address from Plata address
        return None

    def get_doge_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive DOGE address from Plata address
        return None

    def get_sol_address(self, plata_address: str) -> Optional[str]:
        # For now, return None. In the future, implement the logic to derive SOL address from Plata address
        return None

    # Add these methods if they're not already implemented
    async def get_eth_balance(self, address: str) -> Decimal:
        # Implement the logic to get ETH balance
        return Decimal('0')

    async def get_ltc_balance(self, address: str) -> Decimal:
        # Implement the logic to get LTC balance
        return Decimal('0')

    async def get_doge_balance(self, address: str) -> Decimal:
        # Implement the logic to get DOGE balance
        return Decimal('0')

    async def get_sol_balance(self, address: str) -> Decimal:
        # Implement the logic to get SOL balance
        return Decimal('0')

    async def get_transaction_history(self, limit=10):
        try:
            # Assuming the blockchain stores a list of blocks, and each block has transactions
            transactions = []
            for block in reversed(self.chain):  # Start from the most recent block
                for transaction in block.transactions:
                    # Assuming each transaction has a date, amount, and recipient field
                    tx_info = {
                        "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(transaction.timestamp)),
                        "amount": transaction.amount,
                        "recipient": transaction.receiver
                    }
                    transactions.append(tx_info)
                    if len(transactions) >= limit:
                        break
                if len(transactions) >= limit:
                    break

            return transactions

        except Exception as e:
            logger.error(f"Error fetching transaction history: {str(e)}")
            return []

    def get_node_state(self) -> NodeState:
        return NodeState(
            blockchain_length=len(self.chain),
            latest_block_hash=self.chain[-1].hash if self.chain else None,
            pending_transactions_count=len(self.pending_transactions),
            total_supply=self.get_total_supply(),
            difficulty=self.difficulty,
            mempool_size=len(self.mempool),
            connected_peers=len(self.p2p_node.peers),
            active_liquidity_pools=len(self.liquidity_pools),
            node_uptime=time.time() - self.start_time
        )



    async def wait_for_p2p_node(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            p2p_node = await self.get_p2p_node()
            if p2p_node is not None and await p2p_node.is_connected():
                return True
            await asyncio.sleep(1)
        return False




    async def initialize_p2p(self, host, port, retries=3, delay=5):
        """Initialize P2P node with retry logic for robust startup."""
        from P2PNode import P2PNode

        logger.debug(f"Initializing P2P node on {host}:{port}")
        self.p2p_node = P2PNode(host, port, self)
        
        for attempt in range(retries):
            try:
                # Try to start the P2P node with a timeout
                await asyncio.wait_for(self.p2p_node.start(), timeout=10)
                logger.debug("P2P node started successfully")
                return True
            except asyncio.TimeoutError:
                logger.error(f"Timeout while starting P2P node (attempt {attempt + 1}/{retries})")
            except Exception as e:
                logger.error(f"Unexpected error during P2P node initialization: {str(e)}")

            if attempt < retries - 1:
                logger.info(f"Retrying P2P node initialization in {delay} seconds...")
                await asyncio.sleep(delay)
        
        logger.error("Failed to initialize P2P node after maximum retries")
        raise RuntimeError("P2P node initialization failed")



    def get_blocks_since(self, last_known_block_index):
        """
        Returns the blocks added since the given block index.
        """
        return self.chain[last_known_block_index + 1:]
    async def mine_block(self, miner_address):
        """Modified mining function to include confirmation updates"""
        if self.p2p_node is None:
            logger.error("Cannot mine block: P2P node is not initialized")
            return None

        try:
            # Get pending transactions
            transactions = self.get_pending_transactions()[:10]
            
            # Get block reward
            reward = self.get_block_reward()
            
            # Mine the block using DAGKnight miner
            new_block = await self.miner.mine_block(
                previous_hash=self.get_latest_block_hash(),
                data=f"Block mined by {miner_address}",
                transactions=transactions,
                reward=reward,
                miner_address=miner_address
            )
            
            if new_block and self.miner.validate_block(new_block):
                # Add block to chain
                self.chain.append(new_block)
                
                # Update confirmations for the new block
                await self.update_confirmations(new_block)
                
                # Process transactions with confirmation tracking
                await self.process_transactions_with_confirmations(new_block)
                
                # Mint reward and propagate block
                await self.native_coin_contract.mint(miner_address, Decimal(new_block.reward))
                await self.propagate_block_to_peers(new_block)
                
                return new_block.reward
                
            return None

        except Exception as e:
            logger.error(f"Error during mining: {str(e)}")
            logger.error(traceback.format_exc())
            return None












    def batch_verify_transactions(self, transactions):
        proofs = []
        public_inputs = []

        for tx in transactions:
            # Since zk_system.hash is synchronous, do not await it
            public_input = self.zk_system.hash(tx.sender, tx.receiver, str(tx.amount))
            public_inputs.append(public_input)
            proofs.append(tx.zk_proof)

        combined_public_input = self.zk_system.hash(*public_inputs)
        return self.zk_system.verify(combined_public_input, proofs)





    async def import_token(self, address: str, user: str) -> Token:
        if address in self.tokens:
            return self.tokens[address]
        
        # Verify the token contract on-chain
        if not self.verify_token_contract(address):
            raise ValueError("Invalid token contract")
        
        # Fetch token details from the contract
        token_details = self.fetch_token_details(address)
        token = Token(address, token_details['name'], token_details['symbol'], token_details['creator'])
        self.tokens[address] = token
        return token

    def verify_token_contract(self, address: str) -> bool:
        if address in self.verified_contracts:
            return self.verified_contracts[address]

        # In a real implementation, you would:
        # 1. Fetch the contract bytecode from the blockchain
        # 2. Verify that the bytecode matches a known, audited token contract template
        # 3. Check if the contract implements standard token interfaces (e.g., ERC20)
        # 4. Verify that the contract has been deployed by a trusted source (optional)

        # For this example, we'll use a simple verification:
        contract_bytecode = self.vm.get_contract_bytecode(address)
        is_valid = self.validate_token_bytecode(contract_bytecode)
        self.verified_contracts[address] = is_valid
        return is_valid
    def validate_token_bytecode(self, bytecode: str) -> bool:
        # Adjust this method to correctly validate the bytecode
        return (len(bytecode) > 20 and 
                "transfer" in bytecode.lower() and 
                "balanceof" in bytecode.lower())


    def fetch_token_details(self, address: str) -> dict:
        # In a real implementation, you would call the token contract to get these details
        # For this example, we'll return dummy data
        return {
            'name': f"Token {address[:6]}",
            'symbol': f"TKN{address[:3]}",
            'creator': f"0x{hashlib.sha256(address.encode()).hexdigest()[:40]}"
        }

    async def mint_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if user not in token.authorized_minters:
            raise ValueError("Unauthorized to mint tokens")
        
        token.mint(user, amount)

    async def burn_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if user not in token.authorized_burners:
            raise ValueError("Unauthorized to burn tokens")
        
        token.burn(user, amount)

    async def transfer_token(self, address: str, sender: str, recipient: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        token.transfer(sender, recipient, amount)

    async def authorize_minter(self, address: str, authorizer: str, new_minter: str):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if authorizer != token.creator:
            raise ValueError("Only the token creator can authorize new minters")
        
        token.authorize_minter(new_minter)

    async def revoke_minter(self, address: str, authorizer: str, minter: str):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        if authorizer != token.creator:
            raise ValueError("Only the token creator can revoke minters")
        
        token.revoke_minter(minter)

    async def mint_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        # In a real implementation, you would check if the user has permission to mint
        token.mint(user, amount)

    async def burn_token(self, address: str, user: str, amount: Decimal):
        if address not in self.tokens:
            raise ValueError("Token not found")
        
        token = self.tokens[address]
        token.burn(user, amount)

    async def create_liquidity_pool(self, user: str, token_a: str, token_b: str, amount_a: Decimal, amount_b: Decimal) -> str:
        if token_a not in self.tokens or token_b not in self.tokens:
            raise ValueError("One or both tokens not found")

        pool_id = self.liquidity_pool_manager.create_pool(token_a, token_b, Decimal('0.003'))  # 0.3% fee
        
        # Transfer tokens to the pool
        self.tokens[token_a].burn(user, amount_a)
        self.tokens[token_b].burn(user, amount_b)
        
        # Add liquidity
        liquidity_minted = await self.liquidity_pool_manager.add_liquidity(user, token_a, token_b, amount_a, amount_b)
        
        return pool_id

    async def get_user_tokens(self, user: str) -> List[Token]:
        return [token for token in self.tokens.values() if token.balance_of(user) > 0]

    async def swap_tokens(self, user: str, amount_in: Decimal, token_in: str, token_out: str) -> Decimal:
        if token_in not in self.tokens or token_out not in self.tokens:
            raise ValueError("One or both tokens not found")

        # Check user balance
        if self.tokens[token_in].balance_of(user) < amount_in:
            raise ValueError("Insufficient balance")

        # Perform the swap
        amount_out = await self.liquidity_pool_manager.swap(amount_in, token_in, token_out)

        # Update token balances
        self.tokens[token_in].burn(user, amount_in)
        self.tokens[token_out].mint(user, amount_out)

        return amount_out
    async def propagate_block_to_peers(self, block):
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to propagate block")
                p2p_node = await self.get_p2p_node()
                logger.info(f"P2P node status: {p2p_node}")
                logger.info(f"P2P node type: {type(p2p_node)}")
                logger.info(f"P2P node attributes: {vars(p2p_node) if p2p_node else 'None'}")
                
                if p2p_node is None:
                    logger.error(f"P2P node is not initialized, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    logger.error("Traceback for P2P node being None:")
                    logger.error(traceback.format_exc())
                elif not await p2p_node.is_connected():
                    logger.warning(f"P2P node is not connected, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    logger.info(f"Current peers: {p2p_node.peers}")
                else:
                    await p2p_node.propagate_block(block)
                    logger.info(f"Block {block.hash} propagated successfully.")
                    return True

            except Exception as e:
                logger.error(f"Error propagating block: {str(e)}")
                logger.error(traceback.format_exc())
            
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        
        logger.error("Max retries reached. Block could not be propagated.")
        return False



    async def wait_for_peer_connections(self, timeout=30):
        start_time = time.time()
        while not self.p2p_node.is_connected():
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for peers to connect.")
                return False
            logger.debug("Waiting for peer connections...")
            await asyncio.sleep(1)  # Small wait before retrying
        logger.info("Peers connected successfully.")
        return True




    async def get_balance(self, user_id: str, currency: str) -> Decimal:
        balance = self.balances.get(address, 0)
        return balance
        
        
            
            
    def initialize_native_coin_contract(self):
        try:
            result = self.vm.get_existing_contract(NativeCoinContract)
            if result and len(result) == 2:
                self.native_coin_contract_address, self.native_coin_contract = result
            else:
                # Handle the case where the contract doesn't exist
                self.native_coin_contract_address = None
                self.native_coin_contract = None
        except Exception as e:
            logger.error(f"Error initializing NativeCoinContract: {str(e)}")
            self.native_coin_contract_address = None
            self.native_coin_contract = None








    def create_genesis_block(self):
        """Create the genesis block with required parameters."""
        genesis_block = QuantumBlock(
            previous_hash="0",
            data="Genesis Block",
            quantum_signature="00",
            reward=0,
            transactions=[],
            miner_address="genesis_address",  # Add genesis miner address
            nonce=0,                          # Add initial nonce
            parent_hashes=[],                 # Empty list for genesis block
            timestamp=time.time()             # Add timestamp
        )
        
        # Generate hash for the genesis block
        genesis_block.hash = genesis_block.compute_hash()
        
        # Add to blockchain
        self.chain.append(genesis_block)
        logger.info(f"Genesis block created with hash: {genesis_block.hash}")
        return genesis_block




    def on_new_block(self, callback):
        self.new_block_listeners.append(callback)

    def on_new_transaction(self, callback):
        self.new_transaction_listeners.append(callback)

    def get_recent_transactions(self, limit: int = 100) -> List[Transaction]:
        """
        Get recent transactions from the blockchain.
        
        Args:
            limit (int, optional): Maximum number of transactions to return. Defaults to 100.
            
        Returns:
            List[Transaction]: List of recent transactions
        """
        try:
            transactions = []
            # Get transactions from recent blocks
            for block in reversed(self.chain[-10:]):  # Look at last 10 blocks
                transactions.extend(block.transactions)
            # Add mempool transactions
            transactions.extend(self.mempool)
            return transactions[-limit:]  # Return most recent transactions up to limit
        except Exception as e:
            self.logger.error(f"Error getting recent transactions: {str(e)}")
            return []

    def get_block_by_hash(self, block_hash):
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def calculate_average_block_time(self):
        if len(self.chain) < 2:
            return 0
        block_times = [self.chain[i].timestamp - self.chain[i-1].timestamp for i in range(1, len(self.chain))]
        return sum(block_times) / len(block_times)

    def calculate_qhins(self):
        return sum(block.reward for block in self.chain)  # Example calculation

    def calculate_entanglement_strength(self):
        total_entanglement = sum(block.quantum_signature.count('1') for block in self.chain)
        return total_entanglement / len(self.chain) if self.chain else 0
    


    def get_total_supply(self):
        return sum(self.balances.values())
    def add_new_block(self, data, transactions, miner_address):
        previous_block = self.chain[-1]
        previous_hash = previous_block.hash
        reward = self.get_block_reward()
        total_supply = self.get_total_supply()

        if total_supply + Decimal(reward) > Decimal(MAX_SUPPLY):
            reward = Decimal(MAX_SUPPLY) - total_supply

        quantum_signature = self.generate_quantum_signature()  # Call the method here

        new_block = QuantumBlock(
            previous_hash=previous_hash,
            data=data,
            quantum_signature=quantum_signature,
            reward=reward,
            transactions=transactions,
            timestamp=time.time()
        )

        new_block.mine_block(self.difficulty)
        logger.info(f"Adding new block: {new_block.__dict__}")

        if self.consensus.validate_block(new_block):
            self.chain.append(new_block)
            self.process_transactions(transactions)
            self.native_coin_contract.mint(miner_address, Decimal(reward))
            return reward
        else:
            logger.error("Block validation failed. Block not added.")
            raise ValueError("Invalid block")



    def update_total_supply(self, reward):
        # Example implementation
        pass

    def process_transactions(self, transactions):
        for tx in transactions:
            if isinstance(tx, dict):
                sender = tx['sender']
                receiver = tx['receiver']
                amount = tx['amount']
            else:
                sender = tx.sender
                receiver = tx.receiver
                amount = tx.amount
            
            if self.balances.get(sender, 0) >= amount:
                self.balances[sender] = self.balances.get(sender, 0) - amount
                self.balances[receiver] = self.balances.get(receiver, 0) + amount
            else:
                logger.warning(f"Insufficient balance for transaction: {tx}")

    def get_block_reward(self):
        return self.current_reward()

    def current_reward(self):
        elapsed_time = time.time() - self.start_time
        halvings = int(elapsed_time // self.halving_interval)
        reward = self.initial_reward / (2 ** halvings)
        return reward
        
    def validate_block(self, block):
        """
        Validate block including hash difficulty, ZK proof, and confirmation requirements
        with enhanced security checks and detailed logging
        """
        try:
            logger.debug(f"Starting comprehensive block validation for block: {block.hash}")
            validation_results = {
                'hash_valid': False,
                'structure_valid': False,
                'zk_proof_valid': False,
                'quantum_signature_valid': False,
                'confirmation_valid': False,
                'parent_security_valid': False
            }

            # 1. Check hash difficulty target
            hash_int = int(block.hash, 16)
            logger.debug(f"Validating block with hash: {block.hash}, integer value: {hash_int}, against target: {self.target}")
            
            validation_results['hash_valid'] = hash_int < self.target
            if not validation_results['hash_valid']:
                logger.debug(f"Block validation failed: Hash {block.hash} exceeds target {self.target}.")
                return False

            # 2. Validate enhanced block structure
            required_attrs = [
                'data', 'nonce', 'zk_proof', 'hash', 'previous_hash',
                'quantum_signature', 'parent_hashes', 'miner_address'
            ]
            validation_results['structure_valid'] = all(hasattr(block, attr) for attr in required_attrs)
            
            if not validation_results['structure_valid']:
                missing_attrs = [attr for attr in required_attrs if not hasattr(block, attr)]
                logger.error(f"Block validation failed: Missing required attributes: {missing_attrs}")
                return False

            # 3. Verify block hash integrity
            computed_hash = block.compute_hash()
            if computed_hash != block.hash:
                logger.error(f"Block validation failed: Hash mismatch")
                return False

            # 4. Verify ZK proof with enhanced security
            try:
                # Compute public input with additional context
                block_data = {
                    'data': block.data,
                    'nonce': block.nonce,
                    'timestamp': block.timestamp,
                    'miner': block.miner_address,
                    'parent_hashes': block.parent_hashes
                }
                block_bytes = str(block_data).encode() + str(block.nonce).encode()
                secret = int.from_bytes(hashlib.sha256(block_bytes).digest(), 'big') % self.zkp_system.field.modulus
                public_input = secret

                # Verify ZK proof with enhanced checks
                validation_results['zk_proof_valid'] = self.zkp_system.verify(public_input, block.zk_proof)
                
                if not validation_results['zk_proof_valid']:
                    logger.error(f"Block validation failed: Invalid ZK proof")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during ZK proof verification: {e}")
                return False

            # 5. Validate quantum signature
            try:
                validation_results['quantum_signature_valid'] = (
                    self.confirmation_system.evaluate_quantum_signature(block.quantum_signature) 
                    >= self.confirmation_system.quantum_threshold
                )
                
                if not validation_results['quantum_signature_valid']:
                    logger.error(f"Block validation failed: Insufficient quantum signature strength")
                    return False
                    
            except Exception as e:
                logger.error(f"Error validating quantum signature: {e}")
                return False

            # 6. Validate parent blocks' security levels
            try:
                parent_security_scores = []
                for parent_hash in block.parent_hashes:
                    security_info = self.confirmation_system.get_transaction_security(
                        parent_hash,
                        block.hash
                    )
                    parent_security_scores.append(security_info['confirmation_score'])

                # Require minimum average parent security score
                avg_parent_security = sum(parent_security_scores) / len(parent_security_scores) if parent_security_scores else 0
                validation_results['parent_security_valid'] = avg_parent_security >= 0.6  # Minimum security threshold
                
                if not validation_results['parent_security_valid']:
                    logger.error(f"Block validation failed: Insufficient parent block security (score: {avg_parent_security})")
                    return False

            except Exception as e:
                logger.error(f"Error validating parent block security: {e}")
                return False

            # 7. Validate confirmation requirements
            try:
                # Check if block maintains DAG properties
                validation_results['confirmation_valid'] = self.confirmation_system.validate_dag_structure(
                    block.hash,
                    block.parent_hashes
                )
                
                if not validation_results['confirmation_valid']:
                    logger.error("Block validation failed: Invalid DAG structure")
                    return False

            except Exception as e:
                logger.error(f"Error validating confirmation requirements: {e}")
                return False

            # Log successful validation with details
            logger.info(f"Block {block.hash} validated successfully:")
            logger.debug(f"Validation results: {validation_results}")
            
            # Calculate and log overall security score
            security_score = sum(
                validation_results[key] * weight for key, weight in [
                    ('hash_valid', 0.2),
                    ('structure_valid', 0.1),
                    ('zk_proof_valid', 0.2),
                    ('quantum_signature_valid', 0.2),
                    ('confirmation_valid', 0.15),
                    ('parent_security_valid', 0.15)
                ]
            )
            
            logger.info(f"Block security score: {security_score:.4f}")
            
            return True

        except Exception as e:
            logger.error(f"Block validation error: {str(e)}")
            logger.error(traceback.format_exc())
            return False





    def is_valid_hash(self, block_hash):
        target = 2 ** (256 - self.blockchain.difficulty)
        is_valid = int(block_hash, 16) < target
        logger.debug(f"Validating hash: {block_hash}")
        logger.debug(f"Target: {target}")
        logger.debug(f"Is valid hash: {is_valid}")
        return is_valid



    def validate_transaction(self, tx):
        # Implement transaction validation logic
        # This is a placeholder and should be replaced with your actual transaction validation logic
        return True


    async def add_block(self, block):
        """
        Add a new block to the chain with enhanced confirmation tracking, security checks,
        and pruning functionality.
        """
        try:
            # Use pruning lock to prevent concurrent pruning operations
            async with self._pruning_lock:
                logger.info(f"Starting block addition process for block: {block.hash}")
                current_time = time.time()

                # 1. Enhanced timestamp validation
                timestamp_window = {
                    'future': 300,  # 5 minutes into future
                    'past': 7200    # 2 hours into past
                }
                
                if block.timestamp > current_time + timestamp_window['future']:
                    logger.warning(
                        f"Block timestamp too far in future. "
                        f"Current time: {current_time}, Block time: {block.timestamp}"
                    )
                    return False
                    
                if block.timestamp < current_time - timestamp_window['past']:
                    logger.warning(
                        f"Block timestamp too far in past. "
                        f"Current time: {current_time}, Block time: {block.timestamp}"
                    )
                    return False

                # 2. Enhanced block validation
                validation_result = await self.validate_block_with_confirmations(block)
                if not validation_result['is_valid']:
                    logger.warning(
                        f"Block validation failed: {validation_result['reason']}\n"
                        f"Details: {validation_result['details']}"
                    )
                    return False

                # 3. Process block in database transaction
                async with self.db.transaction() as txn:
                    try:
                        # Add block to chain
                        self.chain.append(block)
                        self.blocks_since_last_adjustment += 1

                        # Update confirmation system
                        await self.update_confirmations_for_block(block)

                        # Adjust difficulty if needed
                        if self.blocks_since_last_adjustment >= self.adjustment_interval:
                            await self.adjust_difficulty()
                            self.blocks_since_last_adjustment = 0

                        # Check if pruning is needed
                        chain_size = len(self.chain)
                        time_since_prune = current_time - self.last_prune_time
                        
                        if (chain_size > self.pruning_system.max_dag_size and 
                            time_since_prune >= self.pruning_system.prune_interval):
                            # Execute pruning operation
                            logger.info("Starting scheduled DAG pruning...")
                            pruned_dag, pruning_stats = await self.pruning_system.prune_dag(
                                self.dag,
                                self.confirmation_system
                            )
                            
                            if pruned_dag is not None:
                                self.dag = pruned_dag
                                # Update pruning metrics
                                nodes_pruned = chain_size - len(self.chain)
                                self._update_pruning_metrics(nodes_pruned, time.time() - current_time)
                                self.last_prune_time = time.time()
                                
                                # Broadcast pruning event
                                await self._broadcast_pruning_event(pruning_stats)
                                
                                logger.info(f"Pruned {nodes_pruned} nodes from DAG")

                        # Process transactions with enhanced security
                        await self.process_block_transactions_with_confirmations(block)

                        # Save block data
                        block_data = await self.prepare_block_data_for_storage(block)
                        block_id = await self.db.add_block(block_data, txn)

                        # Update block status with confirmation info
                        confirmation_status = await self.get_block_confirmation_status(block)
                        await self.db.update_block_status(
                            block.hash,
                            confirmation_status,
                            txn
                        )

                        # Add enhanced block stats
                        stats = await self.generate_enhanced_block_stats(block)
                        await self.db.add_block_stats(block.hash, stats, txn)

                        logger.info(f"Block {block.hash} data saved to database with ID: {block_id}")

                    except Exception as e:
                        logger.error(f"Database transaction failed: {str(e)}")
                        await txn.rollback()
                        raise

                # 4. Update network state
                try:
                    # Propagate block with enhanced metadata
                    propagation_task = asyncio.create_task(
                        self.propagate_block_with_enhanced_data(block)
                    )

                    # Update global metrics
                    await self.update_global_metrics_with_confirmations(block)

                    # Notify listeners with enhanced block data
                    await self.notify_listeners_with_confirmation_data(block)

                    # Validate chain state after all operations
                    if not await self.validate_chain_after_pruning():
                        logger.error("Chain validation failed after block addition and pruning")
                        raise ValueError("Chain validation failed")

                except Exception as e:
                    logger.error(f"Network state update failed: {str(e)}")
                    # Continue processing as block is already saved

                # 5. Return success with enhanced status
                logger.info(
                    f"Block {block.hash} added successfully with "
                    f"confirmation status: {confirmation_status}"
                )
                return True

        except Exception as e:
            logger.error(f"Error while adding block: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    async def _check_and_prune(self):
        """Check if pruning is needed and execute if necessary"""
        try:
            current_time = time.time()
            time_since_prune = current_time - self.last_prune_time
            
            # Check pruning conditions
            if (time_since_prune >= self.pruning_system.prune_interval and 
                len(self.chain) > self.pruning_system.max_dag_size):
                
                logger.info("Starting DAG pruning operation...")
                await self._execute_pruning()
                
        except Exception as e:
            logger.error(f"Error checking pruning conditions: {str(e)}")

    async def _execute_pruning(self):
        """Execute pruning operation with proper error handling"""
        try:
            start_time = time.time()
            initial_size = len(self.chain)
            
            # Execute pruning
            pruned_dag, pruning_stats = await self.pruning_system.prune_dag(
                self.dag,
                self.confirmation_system
            )
            
            if pruned_dag is not None:
                # Update DAG and chain state
                self.dag = pruned_dag
                
                # Update metrics
                end_time = time.time()
                prune_duration = end_time - start_time
                nodes_pruned = initial_size - len(self.chain)
                
                # Update pruning metrics
                self._update_pruning_metrics(nodes_pruned, prune_duration)
                
                # Log pruning results
                logger.info(
                    f"Pruning completed: removed {nodes_pruned} nodes in {prune_duration:.2f}s. "
                    f"New chain size: {len(self.chain)}"
                )
                
                # Update last prune time
                self.last_prune_time = end_time
                
                # Broadcast pruning event to peers
                await self._broadcast_pruning_event(pruning_stats)
                
                # Validate chain after pruning
                if not await self.validate_chain_after_pruning():
                    logger.error("Chain validation failed after pruning")
                    raise ValueError("Chain validation failed after pruning")
                
        except Exception as e:
            logger.error(f"Error during pruning execution: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _update_pruning_metrics(self, nodes_pruned: int, duration: float):
        """Update pruning metrics"""
        try:
            self.pruning_metrics['total_nodes_pruned'] += nodes_pruned
            self.pruning_metrics['pruning_operations'] += 1
            
            # Update average prune time
            prev_avg = self.pruning_metrics['avg_prune_time']
            n = self.pruning_metrics['pruning_operations']
            self.pruning_metrics['avg_prune_time'] = (
                (prev_avg * (n-1) + duration) / n
            )
            
            # Estimate saved memory
            avg_node_size = 1024  # Estimated average node size in bytes
            self.pruning_metrics['saved_memory'] += nodes_pruned * avg_node_size
            
        except Exception as e:
            logger.error(f"Error updating pruning metrics: {str(e)}")

    async def _broadcast_pruning_event(self, pruning_stats: Dict):
        """Broadcast pruning event to peers"""
        try:
            if self.p2p_node:
                pruning_event = {
                    'event_type': 'pruning',
                    'timestamp': time.time(),
                    'stats': pruning_stats,
                    'new_dag_size': len(self.chain)
                }
                
                await self.p2p_node.broadcast_event('pruning_update', pruning_event)
                logger.debug("Broadcast pruning event to peers")
                
        except Exception as e:
            logger.error(f"Error broadcasting pruning event: {str(e)}")

    async def receive_pruning_event(self, event_data: Dict):
        """Handle pruning events from peers"""
        try:
            peer_chain_size = event_data['new_dag_size']
            if peer_chain_size < len(self.chain):
                # Peer has pruned more aggressively
                logger.info("Peer has more aggressive pruning, checking our pruning status")
                await self._check_and_prune()
                
        except Exception as e:
            logger.error(f"Error handling pruning event: {str(e)}")

    async def get_pruning_metrics(self) -> Dict:
        """Get current pruning metrics and status"""
        try:
            metrics = self.pruning_metrics.copy()
            metrics.update({
                'last_prune_time': self.last_prune_time,
                'current_chain_size': len(self.chain),
                'time_since_prune': time.time() - self.last_prune_time,
                'pruning_enabled': True,
                'next_prune_in': max(0, self.pruning_system.prune_interval - 
                                   (time.time() - self.last_prune_time))
            })
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting pruning metrics: {str(e)}")
            return {}

    async def validate_chain_after_pruning(self) -> bool:
        """Validate chain integrity after pruning"""
        try:
            # 1. Verify DAG properties
            if not nx.is_directed_acyclic_graph(self.dag):
                logger.error("Chain validation failed: Not a valid DAG after pruning")
                return False

            # 2. Check confirmation paths for recent blocks
            recent_blocks = self.chain[-100:]  # Check last 100 blocks
            for block in recent_blocks:
                security_info = self.confirmation_system.get_transaction_security(
                    block.hash,
                    self.get_latest_block_hash()
                )
                
                if not (security_info['confirmations'] >= self.pruning_system.min_confirmations and
                        security_info['security_level'] in ['HIGH', 'VERY_HIGH', 'MAXIMUM']):
                    logger.error(f"Chain validation failed: Invalid confirmations for block {block.hash}")
                    return False

            # 3. Verify quantum signatures
            for block in self.chain:
                if block.quantum_signature:
                    quantum_score = self.confirmation_system.evaluate_quantum_signature(
                        block.quantum_signature
                    )
                    if quantum_score < self.pruning_system.quantum_threshold:
                        logger.error(f"Chain validation failed: Weak quantum signature for block {block.hash}")
                        return False

            logger.info("Chain validation after pruning: SUCCESS")
            return True

        except Exception as e:
            logger.error(f"Error validating chain after pruning: {str(e)}")
            return False



    async def process_block_transactions_with_confirmations(self, block):
        """Process block transactions with enhanced confirmation tracking"""
        try:
            for tx in block.transactions:
                # Get existing confirmation status
                existing_status = await self.get_transaction_confirmation_status(tx.hash)
                
                # Process transaction
                await self.vm.process_transaction(tx)
                
                # Update transaction status with confirmation data
                security_info = self.confirmation_system.get_transaction_security(
                    tx.hash,
                    block.hash
                )
                
                new_status = {
                    'status': 'confirmed',
                    'confirmation_score': security_info['confirmation_score'],
                    'security_level': security_info['security_level'],
                    'confirmations': security_info['num_confirmations']
                }
                
                await self.db.update_transaction_status(tx.hash, new_status)
                logger.info(
                    f"Processed transaction {tx.hash} with "
                    f"security level: {security_info['security_level']}"
                )
                
        except Exception as e:
            logger.error(f"Error processing block transactions: {str(e)}")
            raise

    async def generate_enhanced_block_stats(self, block):
        """Generate enhanced block statistics including confirmation data"""
        try:
            # Get confirmation metrics
            confirmation_metrics = await self.get_block_confirmation_metrics(block)
            
            # Calculate basic stats
            basic_stats = {
                "transactions_count": len(block.transactions),
                "total_amount": sum(tx.amount for tx in block.transactions),
                "difficulty": self.difficulty,
                "block_time": block.timestamp - self.chain[-2].timestamp if len(self.chain) > 1 else 0
            }
            
            # Add enhanced stats
            enhanced_stats = {
                **basic_stats,
                "confirmation_score": confirmation_metrics['avg_confirmation_score'],
                "security_level": confirmation_metrics['security_level'],
                "quantum_signature_strength": confirmation_metrics['quantum_strength'],
                "parent_blocks": len(block.parent_hashes),
                "dag_depth": confirmation_metrics['dag_depth'],
                "path_diversity": confirmation_metrics['path_diversity']
            }
            
            return enhanced_stats
            
        except Exception as e:
            logger.error(f"Error generating enhanced block stats: {str(e)}")
            return basic_stats

    async def update_global_metrics_with_confirmations(self, block):
        """Update global blockchain metrics with confirmation data"""
        try:
            confirmation_metrics = await self.get_block_confirmation_metrics(block)
            
            self.globalMetrics.update({
                'totalBlocks': len(self.chain),
                'totalTransactions': self.globalMetrics['totalTransactions'] + len(block.transactions),
                'averageConfirmationScore': (
                    self.globalMetrics.get('averageConfirmationScore', 0) * 0.95 +
                    confirmation_metrics['avg_confirmation_score'] * 0.05
                ),
                'securityDistribution': {
                    level: self.globalMetrics.get('securityDistribution', {}).get(level, 0) + 1
                    for level in confirmation_metrics['security_distribution']
                }
            })
            
            logger.info(f"Updated global metrics with new block data")
            
        except Exception as e:
            logger.error(f"Error updating global metrics: {str(e)}")

    async def notify_listeners_with_confirmation_data(self, block):
        """Notify listeners with enhanced block data including confirmations"""
        enhanced_block_data = {
            'block': block,
            'confirmation_metrics': await self.get_block_confirmation_metrics(block),
            'security_info': self.confirmation_system.get_transaction_security(
                block.hash,
                self.chain[-1].hash
            )
        }
        
        for listener in self.new_block_listeners:
            try:
                await listener(enhanced_block_data)
            except Exception as e:
                logger.error(f"Error notifying listener: {str(e)}")

    async def update_confirmations(self, block):
        """Update the confirmation system with a new block"""
        try:
            # Add block to confirmation system
            self.confirmation_system.add_block_confirmation(
                block_hash=block.hash,
                parent_hashes=block.parent_hashes,
                transactions=block.transactions,
                quantum_signature=block.quantum_signature
            )

            # Update confirmation metrics for all transactions in the block
            for tx in block.transactions:
                security_info = self.confirmation_system.get_transaction_security(
                    tx.hash,
                    block.hash
                )
                
                # Update global metrics
                if security_info['security_level'] in ['MAXIMUM', 'VERY_HIGH']:
                    self.globalMetrics['highSecurityTransactions'] += 1
                
                if security_info['is_final']:
                    self.globalMetrics['totalConfirmedTransactions'] += 1

            # Update average confirmation score
            all_txs = self.get_recent_transactions(1000)  # Get last 1000 transactions
            if all_txs:
                total_score = sum(
                    self.confirmation_system.calculate_confirmation_score(
                        tx.hash, 
                        block.hash
                    ) for tx in all_txs
                )
                self.globalMetrics['averageConfirmationScore'] = total_score / len(all_txs)

        except Exception as e:
            logger.error(f"Error updating confirmations: {str(e)}")
            raise

    async def process_transactions_with_confirmations(self, block):
        """Process transactions with confirmation tracking"""
        try:
            for tx in block.transactions:
                # Verify the transaction
                if not await self.verify_transaction(tx):
                    logger.warning(f"Invalid transaction in block: {tx.hash}")
                    continue

                # Get security info for the transaction
                security_info = self.confirmation_system.get_transaction_security(
                    tx.hash,
                    block.hash
                )

                # Update transaction status based on security level
                await self.update_transaction_status(tx, security_info)

                # Apply the transaction if it meets minimum security requirements
                if security_info['security_level'] not in ['UNSAFE', 'LOW']:
                    await self.apply_transaction(tx)

        except Exception as e:
            logger.error(f"Error processing transactions with confirmations: {str(e)}")
            raise

    async def update_transaction_status(self, tx, security_info):
        """Update transaction status based on security level"""
        try:
            status_mapping = {
                'MAXIMUM': 'final',
                'VERY_HIGH': 'confirmed',
                'HIGH': 'highly_secure',
                'MEDIUM_HIGH': 'secure',
                'MEDIUM': 'partially_confirmed',
                'MEDIUM_LOW': 'low_confirmation',
                'LOW': 'insufficient',
                'UNSAFE': 'unsafe'
            }
            
            status = status_mapping[security_info['security_level']]
            
            # Update status in database
            await self.db.update_transaction_status(tx.hash, status)
            
            # Update confirmation count
            tx.confirmations = security_info['num_confirmations']
            
            # Log status change
            logger.info(
                f"Transaction {tx.hash} status updated to {status} "
                f"with {security_info['num_confirmations']} confirmations "
                f"and security score {security_info['confirmation_score']:.4f}"
            )

        except Exception as e:
            logger.error(f"Error updating transaction status: {str(e)}")
            raise

    async def get_transaction_security_info(self, tx_hash: str) -> Dict[str, Any]:
        """Get detailed security information for a transaction"""
        try:
            latest_block = self.chain[-1]
            security_info = self.confirmation_system.get_transaction_security(
                tx_hash,
                latest_block.hash
            )
            
            # Add additional context
            security_info.update({
                'block_height': len(self.chain),
                'timestamp': time.time(),
                'network_difficulty': self.difficulty,
                'confirmation_threshold': self.confirmation_system.min_confirmations
            })
            
            return security_info

        except Exception as e:
            logger.error(f"Error getting transaction security info: {str(e)}")
            return {
                'error': str(e),
                'security_level': 'ERROR',
                'confirmation_score': 0.0
            }


    async def get_block(self, block_hash):
        return await self.db.get_block(block_hash)
    async def get_block_stats(self, start_time: int, end_time: int):
        return await self.db.get_block_stats(start_time, end_time)

    async def search_blocks(self, query: str):
        return await self.db.search_blocks(query)

    async def search_transactions(self, query: str):
        return await self.db.search_transactions(query)

    async def get_cached_block(self, block_hash: str):
        return await self.db.get_cached_block(block_hash)

    async def get_transaction_volume(self, start_time: int, end_time: int):
        return await self.db.get_transaction_volume(start_time, end_time)

    async def create_backup(self):
        await self.db.create_backup()

    async def restore_backup(self, backup_name: str):
        await self.db.restore_backup(backup_name)

    async def atomic_block_addition(self, block):
        async def add_block_op(session):
            # Add block to the database
            await self.db.add_block(block)

        async def update_balances_op(session):
            # Update balances based on block transactions
            for tx in block.transactions:
                await self.db.update_balance(tx.sender, -tx.amount, session)
                await self.db.update_balance(tx.receiver, tx.amount, session)

        await self.db.atomic_operation([add_block_op, update_balances_op])


    async def deploy_contract(self, sender, contract_class, *args):
        contract_address = await self.vm.deploy_contract(sender, contract_class, *args)
        return contract_address

    async def execute_contract(self, sender, contract_address, function_name, *args, **kwargs):
        return await self.vm.execute_contract(contract_address, function_name, *args, **kwargs)

    async def create_token(self, creator_address, token_name, token_symbol, total_supply):
        return await self.vm.create_token(creator_address, token_name, token_symbol, total_supply)

    async def create_nft(self, creator_address, nft_id, metadata):
        return await self.vm.create_nft(creator_address, nft_id, metadata)

    async def transfer_nft(self, from_address, to_address, nft_id):
        return await self.vm.transfer_nft(from_address, to_address, nft_id)

    async def get_balance(self, address):
        return await self.vm.get_balance(address)

    async def transfer(self, from_address, to_address, amount):
        return await self.vm.transfer(from_address, to_address, amount)

    async def sync_with_peer(self, peer_node):
        peer_data = await peer_node.get_all_data()
        await self.db.sync_with_peer(peer_data)
        await self.vm.sync_state_with_db()

    async def get_all_data(self):
        return await self.db.get_all_data()


    def adjust_difficulty(self):
        if len(self.chain) >= self.adjustment_interval:
            start_block = self.chain[-self.adjustment_interval]
            end_block = self.chain[-1]
            
            total_time = end_block.timestamp - start_block.timestamp
            
            if total_time <= 0:
                logger.error("Total time between blocks is zero or negative. Cannot adjust difficulty.")
                return
            
            avg_time = total_time / (self.adjustment_interval - 1)
            target_time = self.target_block_time
            
            logger.info(f"Start block timestamp: {start_block.timestamp}")
            logger.info(f"End block timestamp: {end_block.timestamp}")
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

    async def get_balance(self, user_id: str, currency: str) -> Decimal:
        balance = self.balances.get(address, 0)
        logger.info(f"Balance for {address}: {balance}")
        return balanceasync 
    async def add_transaction(self, transaction: Transaction):
        """Add a transaction with enhanced gas handling and verification"""
        try:
            logger.debug(f"Processing transaction from {transaction.sender} to {transaction.receiver} for amount {transaction.amount}")

            # Step 1: Update gas metrics
            await self.update_gas_metrics()
            
            # Step 2: Calculate gas costs
            gas_estimate = await self.estimate_transaction_gas(transaction.to_dict())
            total_cost = Decimal(str(gas_estimate['total_cost']))
            
            # Step 3: Verify the transaction using ZKP
            if not transaction.verify_transaction(self.zk_system):
                raise ValueError("Invalid transaction or ZKP verification failed")
            
            # Step 4: Check total required balance (amount + gas)
            sender_balance = self.balances.get(transaction.sender, Decimal('0'))
            required_balance = transaction.amount + total_cost
            
            if sender_balance < required_balance:
                logger.debug(f"Insufficient balance for transaction and gas. Required: {required_balance}, Available: {sender_balance}")
                return False
                
            # Step 5: Process quantum verification if enabled
            if transaction.quantum_enabled:
                await self._process_quantum_verification(transaction)
            
            # Step 6: Verify transaction signature
            wallet = Wallet()
            message = f"{transaction.sender}{transaction.receiver}{transaction.amount}"
            if not wallet.verify_signature(message, transaction.signature, transaction.public_key):
                logger.debug("Transaction signature verification failed")
                return False
                
            # Step 7: Add gas information to transaction
            transaction.gas_used = gas_estimate['gas_needed']
            transaction.gas_price = Decimal(str(gas_estimate['gas_price']))
            
            # Step 8: Deduct gas cost and add to rewards pool
            self.balances[transaction.sender] -= total_cost
            self.miner_rewards_pool += total_cost
            
            # Step 9: Add transaction to pending pool
            self.pending_transactions.append(transaction.to_dict())
            logger.debug(f"Transaction added. Pending transactions count: {len(self.pending_transactions)}")
            
            # Step 10: Track gas metrics
            self._track_gas_metrics(transaction, gas_estimate)
            
            # Step 11: Propagate to P2P network
            await self.p2p_node.propagate_transaction(transaction)
            
            # Step 12: Notify listeners
            for listener in self.new_transaction_listeners:
                listener(transaction)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            logger.error(traceback.format_exc())
            return False






    async def propagate_transaction_to_all_peers(self, transaction_data):
        nodes = self.node_directory.discover_nodes()
        for node in nodes:
            async with grpc.aio.insecure_channel(f'{node["ip_address"]}:{node["port"]}') as channel:
                stub = DAGKnightStub(channel)
                transaction_request = FullStateRequest(
                    # Populate with necessary fields from transaction_data
                )
                await stub.FullStateSync(transaction_request)

    async def propagate_block_to_all_peers(self, block_data):
        nodes = self.node_directory.discover_nodes()
        tasks = [self.propagate_block(f"http://{node['ip_address']}:{node['port']}/receive_block", block_data) for node in nodes]
        results = await asyncio.gather(*tasks)
        successful_propagations = sum(results)
        logger.info(f"Successfully propagated block to {successful_propagations}/{len(nodes)} peers")
    async def wait_for_peer_connections(self, timeout=30):
        start_time = time.time()
        while not self.is_connected():
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for peers to connect.")
                return False
            logger.debug("Waiting for peer connections...")
            await asyncio.sleep(1)  # Small wait before retrying
        logger.info("Peers connected successfully.")
        return True
    async def propagate_block(self, block):
        try:
            logger.info(f"Propagating block with hash: {block.hash}")
            message = Message(
                type=MessageType.BLOCK.value,
                payload=block.to_dict()
            )
            logger.debug(f"Created block message: {message.to_json()}")
            if self.p2p_node:
                logger.debug(f"P2P node before get_active_peers: {self.p2p_node}")
                active_peers = await self.p2p_node.get_active_peers()
                logger.debug(f"Active P2P node peers before propagation: {active_peers}")
                await self.p2p_node.broadcast(message)
            else:
                logger.error("P2P node is not initialized")
        except Exception as e:
            logger.error(f"Error propagating block {block.hash}: {str(e)}")
            logger.error(traceback.format_exc())


    async def sync_state(self, directory_ip, directory_port):
        async with grpc.aio.insecure_channel(f'{directory_ip}:{directory_port}') as channel:
            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
            request = dagknight_pb2.FullStateRequest()
            response = await stub.FullStateSync(request)
            
            self.chain = [QuantumBlock(
                previous_hash=blk.previous_hash,
                data=blk.data,
                quantum_signature=blk.quantum_signature,
                reward=blk.reward,
                transactions=[tx for tx in blk.transactions]
            ) for blk in response.chain]
            self.balances = {k: v for k, v in response.balances.items()}
            self.stakes = {k: v for k, v in response.stakes.items()}

    def stake_coins(self, address, amount):
        if self.balances.get(address, 0) >= amount:
            self.balances[address] -= amount
            self.stakes[address] = self.stakes.get(address, 0) + amount
            return True
        return False

    def unstake_coins(self, address, amount):
        if self.stakes.get(address, 0) >= amount:
            self.stakes[address] -= amount
            self.balances[address] = self.balances.get(address, 0) + amount
            return True
        return False

    def get_staked_balance(self, address):
        return self.stakes.get(address, 0)


    def full_state_sync(self, request, context):
        return dagknight_pb2.FullStateResponse(
            chain=[dagknight_pb2.Block(
                previous_hash=block.previous_hash,
                data=block.data,
                quantum_signature=block.quantum_signature,
                reward=block.reward,
                transactions=[dagknight_pb2.Transaction(sender=tx['sender'], receiver=tx['receiver'], amount=tx['amount']) for tx in block.transactions]
            ) for block in self.chain],
            balances=self.balances,
            stakes=self.stakes
        )

    def update_balances(self, new_block):
        for tx in new_block.transactions:
            transaction = Transaction.from_dict(tx)
            self.balances[transaction.sender] = self.balances.get(transaction.sender, 0) - transaction.amount
            self.balances[transaction.receiver] = self.balances.get(transaction.receiver, 0) + transaction.amount
            logger.info(f"Updated balance for sender {transaction.sender}: {self.balances[transaction.sender]}")
            logger.info(f"Updated balance for receiver {transaction.receiver}: {self.balances[transaction.receiver]}")

            # Log total supply after each transaction
            total_supply = self.get_total_supply()
            logger.info(f"Total supply after transaction: {total_supply}")
    def generate_quantum_signature(self):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                num_qubits = 8
                qr = QuantumRegister(num_qubits)
                cr = ClassicalRegister(num_qubits)
                qc = QuantumCircuit(qr, cr)

                for i in range(num_qubits):
                    qc.h(qr[i])
                    qc.measure(qr[i], cr[i])

                simulator = AerSimulator()
                transpiled_circuit = transpile(qc, simulator)
                job = simulator.run(transpiled_circuit, shots=1)
                result = job.result()

                if result.status != 'COMPLETED':
                    logger.error(f"Quantum job failed: {result.status}")
                    continue

                counts = result.get_counts()
                signature = list(counts.keys())[0]
                logger.info(f"Generated quantum signature: {signature}")

                if self.validate_quantum_signature(signature):
                    return signature

                logger.warning(f"Generated signature {signature} failed validation. Retrying...")
            except Exception as e:
                logger.error(f"Error in generate_quantum_signature: {str(e)}")

        # If we can't generate a valid signature after max attempts, return a default one
        return "00000000"


    def validate_quantum_signature(self, quantum_signature):
        logger.debug(f"Validating quantum signature: {quantum_signature}")
        try:
            num_qubits = len(quantum_signature)
            qc = QuantumCircuit(num_qubits, num_qubits)

            for i, bit in enumerate(quantum_signature):
                if bit == '1':
                    qc.x(i)
            
            qc.measure(range(num_qubits), range(num_qubits))

            simulator = AerSimulator()
            compiled_circuit = transpile(qc, simulator)
            job = simulator.run(compiled_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()

            max_key = max(counts, key=counts.get)
            max_value = counts[max_key] / 1024

            logger.info(f"Max measurement key: {max_key}, Max measurement probability: {max_value}")

            is_valid = max_value > 0.7 and max_key == quantum_signature

            if is_valid:
                logger.info(f"Quantum signature {quantum_signature} validated successfully with probability {max_value}")
            else:
                logger.error(f"Quantum signature {quantum_signature} is invalid with probability {max_value}")

            return is_valid
        except QiskitError as e:
            logger.error(f"Qiskit error during validation: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            return False


    def hamming_distance(self, s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def update_total_supply(self, reward):
        total_supply = self.get_total_supply()
        new_total_supply = total_supply + reward
        logger.info(f"Total supply updated from {total_supply:.2e} to {new_total_supply:.2e}")
    def get_recent_transactions(self, limit: int = 100) -> List[Transaction]:
        """
        Get recent transactions from the blockchain.
        
        Args:
            limit (int, optional): Maximum number of transactions to return. Defaults to 100.
            
        Returns:
            List[Transaction]: List of recent transactions
        """
        try:
            transactions = []
            # Get transactions from recent blocks
            for block in reversed(self.chain[-10:]):  # Look at last 10 blocks
                transactions.extend(block.transactions)
            # Add mempool transactions
            transactions.extend(self.mempool)
            return transactions[-limit:]  # Return most recent transactions up to limit
        except Exception as e:
            self.logger.error(f"Error getting recent transactions: {str(e)}")
            return []

    def get_block_by_hash(self, block_hash):
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def calculate_average_block_time(self):
        if len(self.chain) < 2:
            return 0
        block_times = [self.chain[i].timestamp - self.chain[i-1].timestamp for i in range(1, len(self.chain))]
        return sum(block_times) / len(block_times)

    def calculate_qhins(self):
        return sum(block.reward for block in self.chain)

    def calculate_entanglement_strength(self):
        total_entanglement = sum(block.quantum_signature.count('1') for block in self.chain)
        return total_entanglement / len(self.chain) if self.chain else 0
        
    def add_wallet(self, wallet):
        self.wallets.append(wallet)
        self.balances[wallet.address] = 0


    async def get_transaction(self, tx_hash):
        return await self.db.get_transaction(tx_hash)

    async def get_latest_blocks(self, limit=10):
        return await self.db.get_latest_blocks(limit)

    async def get_latest_transactions(self, limit=10):
        return await self.db.get_latest_transactions(limit)
    async def sync_with_peer(self, peer_node):
        peer_data = await peer_node.get_all_data()
        await self.db.sync_with_peer(peer_data)

    async def get_all_data(self):
        return await self.db.get_all_data()


    def add_contract(self, contract):
        self.contracts.append(contract)

    def search_wallets(self, query):
        return [wallet for wallet in self.wallets if query in wallet.address.lower()]

    def search_transactions(self, query):
        return [tx for tx in self.transactions if query in tx.hash.lower() or query in tx.sender.lower() or query in tx.receiver.lower()]

    def search_contracts(self, query):
        return [contract for contract in self.contracts if query in contract.address.lower() or query in contract.creator.lower()]

    async def get_balance(self, address, currency):
        # Assuming `balances` is a dictionary of dictionaries, where each key is an address,
        # and the value is another dictionary with currencies as keys and amounts as values.
        address_balances = self.balances.get(address, {})
        balance = address_balances.get(currency, 0)
        return balance
    def get_latest_block(self):
        """Return the latest block in the blockchain."""
        if not self.chain:
            raise ValueError("Blockchain is empty.")
        return self.chain[-1]  # Assuming the latest block is at the end of the chain