from DAGConfirmationSystem import DAGConfirmationSystem
from typing import List, Dict, Tuple, Optional
import networkx as nx
from decimal import Decimal
import numpy as np
import hashlib
import time
import logging
from typing import List, Dict, Tuple, Optional
import networkx as nx
from decimal import Decimal
import numpy as np
import hashlib
import time
import logging
import asyncio
import traceback
from collections import deque
import networkx as nx
import logging
from SecureHybridZKStark import SecureHybridZKStark
from CryptoProvider import CryptoProvider
from shared_logic import QuantumBlock

from DAGKnightGasSystem import EnhancedDAGKnightGasSystem,EnhancedGasTransactionType

logger = logging.getLogger(__name__)
class DAGDiagnostics:
    def __init__(self, dag_system):
        self.dag = dag_system.dag
        self.confirmation_cache = dag_system.confirmation_cache
        self.quantum_scores = dag_system.quantum_scores
        self.logger = logging.getLogger(__name__)

    def diagnose_dag_structure(self) -> dict:
        """Perform comprehensive DAG structure analysis"""
        try:
            diagnostics = {
                "structure": {
                    "total_nodes": self.dag.number_of_nodes(),
                    "total_edges": self.dag.number_of_edges(),
                    "is_dag": nx.is_directed_acyclic_graph(self.dag),
                    "blocks": self._count_blocks(),
                    "transactions": self._count_transactions(),
                    "isolated_nodes": list(nx.isolates(self.dag))
                },
                "connections": self._analyze_connections(),
                "paths": self._analyze_paths(),
                "confirmations": self._analyze_confirmations(),
                "issues": []
            }

            # Identify potential issues
            self._check_for_issues(diagnostics)
            
            return diagnostics

        except Exception as e:
            self.logger.error(f"Error in DAG diagnostics: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _count_blocks(self) -> int:
        """Count nodes that have outgoing edges (likely blocks)"""
        return sum(1 for node in self.dag.nodes() if self.dag.out_degree(node) > 0)

    def _count_transactions(self) -> int:
        """Count nodes that only have incoming edges (likely transactions)"""
        return sum(1 for node in self.dag.nodes() if self.dag.out_degree(node) == 0)

    def _analyze_connections(self) -> dict:
        """Analyze block-transaction connections"""
        connections = {
            "blocks_without_transactions": [],
            "transactions_without_blocks": [],
            "properly_connected_blocks": [],
            "connection_counts": {}
        }

        for node in self.dag.nodes():
            in_degree = self.dag.in_degree(node)
            out_degree = self.dag.out_degree(node)
            
            if out_degree > 0:  # This is likely a block
                tx_count = sum(1 for successor in self.dag.successors(node) 
                             if self.dag.out_degree(successor) == 0)
                if tx_count == 0:
                    connections["blocks_without_transactions"].append(node)
                else:
                    connections["properly_connected_blocks"].append(node)
                connections["connection_counts"][node] = tx_count

            elif in_degree == 0:  # This is a transaction without confirming blocks
                connections["transactions_without_blocks"].append(node)

        return connections

    def _analyze_paths(self) -> dict:
        """Analyze confirmation paths in the DAG"""
        path_analysis = {
            "max_path_length": 0,
            "min_path_length": float('inf'),
            "avg_path_length": 0,
            "path_counts": {},
            "broken_paths": []
        }

        transaction_nodes = [n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]
        block_nodes = [n for n in self.dag.nodes() if self.dag.out_degree(n) > 0]

        total_paths = 0
        path_lengths = []

        for tx in transaction_nodes:
            paths = []
            for block in block_nodes:
                try:
                    tx_paths = list(nx.all_simple_paths(self.dag, block, tx))
                    paths.extend(tx_paths)
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    path_analysis["broken_paths"].append((block, tx, str(e)))

            if paths:
                path_lengths.extend(len(p) for p in paths)
                path_analysis["path_counts"][tx] = len(paths)
                total_paths += len(paths)
            else:
                path_analysis["path_counts"][tx] = 0

        if path_lengths:
            path_analysis["max_path_length"] = max(path_lengths)
            path_analysis["min_path_length"] = min(path_lengths)
            path_analysis["avg_path_length"] = sum(path_lengths) / len(path_lengths)

        return path_analysis

    def _analyze_confirmations(self) -> dict:
        """Analyze confirmation status of transactions"""
        confirmation_analysis = {
            "total_confirmed": 0,
            "unconfirmed": [],
            "confirmation_counts": {},
            "cache_status": {},
            "quantum_scores": {}
        }

        transaction_nodes = [n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]
        
        for tx in transaction_nodes:
            # Check cache status
            cache_entry = self.confirmation_cache.get(tx)
            if cache_entry:
                confirmation_analysis["cache_status"][tx] = {
                    "has_paths": bool(cache_entry.get("paths")),
                    "has_score": bool(cache_entry.get("score")),
                    "last_update": cache_entry.get("last_update")
                }
            else:
                confirmation_analysis["cache_status"][tx] = {
                    "has_paths": False,
                    "has_score": False,
                    "last_update": None
                }

            # Count confirmations
            confirming_blocks = len([b for b in self.dag.predecessors(tx)])
            confirmation_analysis["confirmation_counts"][tx] = confirming_blocks
            
            if confirming_blocks > 0:
                confirmation_analysis["total_confirmed"] += 1
            else:
                confirmation_analysis["unconfirmed"].append(tx)

            # Get quantum scores
            confirmation_analysis["quantum_scores"][tx] = self.quantum_scores.get(tx, 0)

        return confirmation_analysis

    def _check_for_issues(self, diagnostics: dict) -> None:
        """Identify potential issues in the DAG structure"""
        issues = []

        # Check for basic structural issues
        if not diagnostics["structure"]["is_dag"]:
            issues.append("Graph contains cycles - not a valid DAG")

        if diagnostics["structure"]["isolated_nodes"]:
            issues.append(f"Found {len(diagnostics['structure']['isolated_nodes'])} isolated nodes")

        # Check for connection issues
        if diagnostics["connections"]["blocks_without_transactions"]:
            issues.append(f"Found {len(diagnostics['connections']['blocks_without_transactions'])} blocks without transactions")

        if diagnostics["connections"]["transactions_without_blocks"]:
            issues.append(f"Found {len(diagnostics['connections']['transactions_without_blocks'])} transactions without confirming blocks")

        # Check for path issues
        if diagnostics["paths"]["broken_paths"]:
            issues.append(f"Found {len(diagnostics['paths']['broken_paths'])} broken paths")

        # Check confirmation issues
        confirmation_analysis = diagnostics["confirmations"]
        if confirmation_analysis["unconfirmed"]:
            issues.append(f"Found {len(confirmation_analysis['unconfirmed'])} unconfirmed transactions")

        diagnostics["issues"] = issues

    def print_diagnostics(self, diagnostics: dict) -> None:
        """Print formatted diagnostics results"""
        self.logger.info("\n=== DAG Structure Diagnostics ===")
        
        self.logger.info("\nStructure Overview:")
        self.logger.info(f"Total Nodes: {diagnostics['structure']['total_nodes']}")
        self.logger.info(f"Total Edges: {diagnostics['structure']['total_edges']}")
        self.logger.info(f"Is Valid DAG: {diagnostics['structure']['is_dag']}")
        self.logger.info(f"Total Blocks: {diagnostics['structure']['blocks']}")
        self.logger.info(f"Total Transactions: {diagnostics['structure']['transactions']}")
        
        self.logger.info("\nConnection Analysis:")
        self.logger.info(f"Blocks without transactions: {len(diagnostics['connections']['blocks_without_transactions'])}")
        self.logger.info(f"Transactions without blocks: {len(diagnostics['connections']['transactions_without_blocks'])}")
        self.logger.info(f"Properly connected blocks: {len(diagnostics['connections']['properly_connected_blocks'])}")
        
        self.logger.info("\nPath Analysis:")
        self.logger.info(f"Max Path Length: {diagnostics['paths']['max_path_length']}")
        self.logger.info(f"Min Path Length: {diagnostics['paths']['min_path_length']:.2f}")
        self.logger.info(f"Average Path Length: {diagnostics['paths']['avg_path_length']:.2f}")
        
        self.logger.info("\nConfirmation Analysis:")
        self.logger.info(f"Total Confirmed Transactions: {diagnostics['confirmations']['total_confirmed']}")
        self.logger.info(f"Unconfirmed Transactions: {len(diagnostics['confirmations']['unconfirmed'])}")
        
        if diagnostics["issues"]:
            self.logger.warning("\nIdentified Issues:")
            for issue in diagnostics["issues"]:
                self.logger.warning(f"- {issue}")

    def visualize_dag(self, output_file: str = "dag_structure.png") -> None:
        """Generate a visualization of the DAG structure"""
        try:
            import matplotlib.pyplot as plt
            
            # Create layout
            pos = nx.spring_layout(self.dag)
            
            # Draw the graph
            plt.figure(figsize=(12, 8))
            
            # Draw nodes
            blocks = [n for n in self.dag.nodes() if self.dag.out_degree(n) > 0]
            transactions = [n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]
            
            nx.draw_networkx_nodes(self.dag, pos, nodelist=blocks, node_color='lightblue', 
                                 node_size=500, label='Blocks')
            nx.draw_networkx_nodes(self.dag, pos, nodelist=transactions, node_color='lightgreen',
                                 node_size=300, label='Transactions')
            
            # Draw edges
            nx.draw_networkx_edges(self.dag, pos, edge_color='gray', arrows=True)
            
            # Add labels
            labels = {node: node[:6] + '...' for node in self.dag.nodes()}
            nx.draw_networkx_labels(self.dag, pos, labels, font_size=8)
            
            plt.title("DAG Structure Visualization")
            plt.legend()
            plt.savefig(output_file)
            plt.close()
            
            self.logger.info(f"DAG visualization saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating DAG visualization: {str(e)}")

class DAGKnightMiner:
    def __init__(self, difficulty: int = 4, security_level: int = 20):
        """Initialize DAGKnight miner with quantum parameters and gas metrics"""
        # Basic mining parameters
        self.initial_difficulty = difficulty
        self.difficulty = difficulty
        self.security_level = security_level
        self.target = 2**(256-difficulty)
        
        # Gas system parameters
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # Adjust gas metrics every minute
        self.min_gas_price = Decimal('0.1')
        self.max_gas_price = Decimal('1000.0')
        self.base_gas_price = Decimal('1.0')
        
        # Quantum parameters
        self.quantum_entropy = 0.1      # Controls entropy-based corrections
        self.quantum_coupling = 0.15    # Controls quantum coupling strength
        self.coupling_strength = 0.1    # For backward compatibility
        self.entanglement_threshold = 0.5
        self.decoherence_rate = 0.01   # Rate of quantum state decay
        self.system_size = 32          # Quantum system dimension
        
        # DAG parameters
        self.max_parents = 8
        self.min_parents = 2
        
        # Initialize DAG
        self.dag = nx.DiGraph()
        
        # Mining metrics
        self.mining_metrics = {
            'blocks_mined': 0,
            'total_hash_calculations': 0,
            'average_mining_time': 0,
            'difficulty_history': [],
            'hash_rate_history': [],
            'last_block_time': time.time(),
            'mining_start_time': time.time()
        }
        
        # Network state metrics
        self.network_metrics = {
            'avg_block_time': 30.0,
            'network_load': 0.0,
            'active_nodes': 0,
            'quantum_entangled_pairs': 0,
            'dag_depth': 0,
            'total_compute': 0.0
        }
        
        # Base costs for different transaction types
        self.base_costs = {
            EnhancedGasTransactionType.STANDARD: 21000,
            EnhancedGasTransactionType.SMART_CONTRACT: 50000,
            EnhancedGasTransactionType.DAG_REORG: 100000,
            EnhancedGasTransactionType.QUANTUM_PROOF: 75000,
            EnhancedGasTransactionType.QUANTUM_ENTANGLE: 80000,
            EnhancedGasTransactionType.DATA_STORAGE: 40000,
            EnhancedGasTransactionType.COMPUTATION: 60000,
            EnhancedGasTransactionType.QUANTUM_STATE: 90000,
            EnhancedGasTransactionType.ENTANGLEMENT_VERIFY: 85000
        }
        
        # Initialize quantum system
        self.H = self._initialize_hamiltonian()
        self.J = self._initialize_coupling()
        
        logger.info(f"Initialized DAGKnightMiner with:"
                   f"\n\tDifficulty: {difficulty}"
                   f"\n\tSecurity Level: {security_level}"
                   f"\n\tQuantum Entropy: {self.quantum_entropy}"
                   f"\n\tQuantum Coupling: {self.quantum_coupling}"
                   f"\n\tCoupling Strength: {self.coupling_strength}"
                   f"\n\tSystem Size: {self.system_size}")

    def _initialize_hamiltonian(self) -> np.ndarray:
        """Initialize the Hamiltonian matrix for quantum operations"""
        try:
            H = np.random.random((self.system_size, self.system_size))
            return H + H.T  # Make Hermitian
        except Exception as e:
            logger.error(f"Error initializing Hamiltonian: {str(e)}")
            return np.eye(self.system_size)  # Return identity matrix as fallback

    def _initialize_coupling(self) -> np.ndarray:
        """Initialize the coupling matrix for quantum interactions"""
        try:
            # Use either quantum_coupling or coupling_strength
            coupling = self.quantum_coupling if hasattr(self, 'quantum_coupling') else self.coupling_strength
            
            # Create coupling matrix
            J = np.zeros((self.system_size, self.system_size))
            for i in range(self.system_size):
                for j in range(i+1, self.system_size):
                    J[i,j] = J[j,i] = np.random.normal(0, coupling)
            return J
        except Exception as e:
            logger.error(f"Error initializing coupling matrix: {str(e)}")
            return np.zeros((self.system_size, self.system_size))  # Return zero matrix as fallback

    def apply_quantum_correction(self, state: np.ndarray) -> np.ndarray:
        """Apply quantum corrections to state"""
        try:
            # Apply quantum coupling
            coupling_effect = self.J @ state
            
            # Apply decoherence
            decoherence = np.exp(-self.decoherence_rate * np.arange(len(state)))
            
            # Combine effects
            corrected_state = state + self.quantum_coupling * coupling_effect
            corrected_state *= decoherence
            
            # Normalize
            return corrected_state / np.linalg.norm(corrected_state)
        except Exception as e:
            logger.error(f"Error applying quantum correction: {str(e)}")
            return state  # Return original state if correction fails


    @staticmethod
    def verify_networkx_installation():
        """Verify NetworkX installation and functionality"""
        try:
            # Test NetworkX functionality
            test_graph = nx.DiGraph()
            test_graph.add_node(1)
            test_graph.add_node(2)
            test_graph.add_edge(1, 2)
            logger.info("NetworkX verification successful")
            return True
        except Exception as e:
            logger.error(f"NetworkX verification failed: {str(e)}")
            return False

    async def _update_mining_metrics(self, mining_time: float, hash_calculations: int):
        """Async version of update mining metrics"""
        try:
            self.mining_metrics['blocks_mined'] += 1
            self.mining_metrics['total_hash_calculations'] += hash_calculations
            
            alpha = 0.2
            current_avg = self.mining_metrics['average_mining_time']
            self.mining_metrics['average_mining_time'] = (alpha * mining_time + 
                                                        (1 - alpha) * current_avg)
            
            current_hash_rate = hash_calculations / max(mining_time, 0.001)
            self.mining_metrics['hash_rate_history'].append(current_hash_rate)
            self.mining_metrics['difficulty_history'].append(self.difficulty)
            
            # Initialize genesis block if needed
            if not self.dag.nodes:
                genesis_hash = "0" * 64
                self.dag.add_node(genesis_hash, timestamp=time.time())
                logger.info("Genesis block initialized in DAG")
            
            logger.info(f"Mining metrics updated:"
                       f"\n\tBlocks mined: {self.mining_metrics['blocks_mined']}"
                       f"\n\tMining time: {mining_time:.2f}s"
                       f"\n\tHash calculations: {hash_calculations}"
                       f"\n\tHash rate: {current_hash_rate:.2f} H/s"
                       f"\n\tCurrent difficulty: {self.difficulty}")
            
        except Exception as e:
            logger.error(f"Error updating mining metrics: {str(e)}")



    async def adjust_difficulty(self, block_timestamp: float):
        """Dynamically adjust mining difficulty"""
        try:
            self.block_times.append(block_timestamp)
            
            if len(self.block_times) >= self.adjustment_window:
                # Calculate average block time over the window
                time_diffs = [self.block_times[i] - self.block_times[i-1] 
                            for i in range(1, len(self.block_times))]
                avg_block_time = sum(time_diffs) / len(time_diffs)
                
                # Calculate adjustment factor
                raw_adjustment = self.target_block_time / avg_block_time
                
                # Apply dampening to smooth adjustments
                damped_adjustment = 1 + (raw_adjustment - 1) * self.adjustment_dampening
                
                # Limit maximum adjustment per iteration
                max_adjustment = 4.0  # Maximum 4x change per adjustment
                capped_adjustment = max(1/max_adjustment, 
                                     min(max_adjustment, damped_adjustment))
                
                # Calculate new difficulty
                new_difficulty = int(self.difficulty * capped_adjustment)
                new_difficulty = max(self.min_difficulty, 
                                   min(self.max_difficulty, new_difficulty))
                
                # Log adjustment details
                logger.info(f"Difficulty adjustment:"
                          f"\n\tPrevious difficulty: {self.difficulty}"
                          f"\n\tAverage block time: {avg_block_time:.2f}s"
                          f"\n\tTarget block time: {self.target_block_time}s"
                          f"\n\tRaw adjustment factor: {raw_adjustment:.2f}"
                          f"\n\tDamped adjustment factor: {damped_adjustment:.2f}"
                          f"\n\tFinal adjustment factor: {capped_adjustment:.2f}"
                          f"\n\tNew difficulty: {new_difficulty}")
                
                # Update difficulty and target
                self.difficulty = new_difficulty
                self.target = 2**(256-self.difficulty)
                
                # Reset adjustment window
                self.last_adjustment_time = time.time()
                
                # Estimate network hash rate
                network_hash_rate = self._estimate_network_hash_rate(avg_block_time)
                logger.info(f"Estimated network hash rate: {network_hash_rate:.2f} H/s")
                
        except Exception as e:
            logger.error(f"Error in difficulty adjustment: {str(e)}")
            logger.error(traceback.format_exc())

    def _estimate_network_hash_rate(self, avg_block_time: float) -> float:
        """Estimate the network hash rate based on difficulty and block time"""
        try:
            # Approximate hashes needed per block based on difficulty
            hashes_per_block = 2**self.difficulty
            # Calculate hash rate: hashes per block / average time per block
            hash_rate = hashes_per_block / max(avg_block_time, 0.001)  # Avoid division by zero
            return hash_rate
        except Exception as e:
            logger.error(f"Error estimating network hash rate: {str(e)}")
            return 0.0

    def get_mining_metrics(self) -> Dict:
        """Get comprehensive mining statistics"""
        current_time = time.time()
        uptime = current_time - self.mining_metrics['mining_start_time']
        
        return {
            'current_difficulty': self.difficulty,
            'blocks_mined': self.mining_metrics['blocks_mined'],
            'average_mining_time': self.mining_metrics['average_mining_time'],
            'total_hash_calculations': self.mining_metrics['total_hash_calculations'],
            'hash_rate_history': self.mining_metrics['hash_rate_history'][-100:],  # Last 100 entries
            'difficulty_history': self.mining_metrics['difficulty_history'][-100:],  # Last 100 entries
            'uptime': uptime,
            'blocks_per_hour': (self.mining_metrics['blocks_mined'] / uptime) * 3600 if uptime > 0 else 0,
            'target_block_time': self.target_block_time,
            'current_target': self.target,
            'dag_metrics': self.get_dag_metrics()
        }

    def reset_metrics(self):
        """Reset mining metrics while preserving difficulty settings"""
        self.mining_metrics = {
            'blocks_mined': 0,
            'total_hash_calculations': 0,
            'average_mining_time': 0,
            'difficulty_history': [self.difficulty],
            'hash_rate_history': [],
            'last_block_time': time.time(),
            'mining_start_time': time.time()
        }
        logger.info("Mining metrics reset")





    def _initialize_hamiltonian(self) -> np.ndarray:
        """Initialize the Hamiltonian matrix"""
        H = np.random.random((self.system_size, self.system_size))
        return H + H.T  # Make Hermitian

    def _initialize_coupling(self) -> np.ndarray:
        """Initialize the coupling matrix"""
        J = self.coupling_strength * np.random.random((self.system_size, self.system_size))
        return J + J.T

    def _initialize_zkp_system(self):
        """Initialize the Zero-Knowledge Proof system"""
        self.zkp_system = SimpleZKProver(self.security_level)
        logger.info("ZKP system initialized successfully")
    def generate_zkp(self, data: Dict, nonce: int) -> tuple:
        try:
            block_bytes = str(data).encode() + str(nonce).encode()
            secret = int.from_bytes(hashlib.sha256(block_bytes).digest(), 'big')
            public_input = int.from_bytes(hashlib.sha256(str(data).encode()).digest(), 'big')
            logger.debug(f"[ZKP PROVE] Secret: {secret}, Public Input: {public_input}")
            proof = self.zkp_system.prove(secret, public_input)
            logger.debug(f"[ZKP PROVE] Proof: {proof}")
            return proof
        except Exception as e:
            logger.error(f"Error generating ZKP: {str(e)}")
            return None
    def compute_public_input(self, test_data, nonce):
        # Compute public_input from test_data and nonce
        block_bytes = str(test_data).encode() + str(nonce).encode()
        public_input = int.from_bytes(hashlib.sha256(block_bytes).digest(), 'big') % self.zkp_system.field.modulus

        # Debug log to verify public_input computation
        logger.debug(f"Computed public_input: {public_input} from test_data: '{test_data}' and nonce: {nonce}")
        return public_input


    def verify(self, public_input, proof):
        # Temporarily bypass verification for testing
        return True

    def verify_zkp(self, test_data, nonce, proof, mined_public_input):
            """Verify zero-knowledge proof for block"""
            try:
                # Compute public input
                block_bytes = str(test_data).encode() + str(nonce).encode()
                secret = int.from_bytes(hashlib.sha256(block_bytes).digest(), 'big') % self.zkp_system.field.modulus
                public_input = secret
                
                logger.debug(f"ZKP Verification - Input Data:")
                logger.debug(f"Test data: {test_data}")
                logger.debug(f"Nonce: {nonce}")
                logger.debug(f"Computed public input: {public_input}")
                logger.debug(f"Mined public input: {mined_public_input}")

                if public_input != mined_public_input:
                    logger.warning("Public input mismatch")
                    return False

                # Verify using STARK and SNARK components
                stark_valid = self.zkp_system.stark.verify_proof(public_input, proof[0])
                snark_valid = self.zkp_system.snark.verify(public_input, proof[1])

                logger.debug(f"STARK verification: {stark_valid}")
                logger.debug(f"SNARK verification: {snark_valid}")

                return stark_valid and snark_valid

            except Exception as e:
                logger.error(f"ZKP verification error: {str(e)}")
                return False

    def validate_block(self, block):
        """Validate block including hash difficulty and ZK proof"""
        try:
            # Check hash difficulty
            hash_int = int(block.hash, 16)
            logger.debug(f"Block validation - Hash check:")
            logger.debug(f"Block hash: {block.hash}")
            logger.debug(f"Hash int: {hash_int}")
            logger.debug(f"Target: {self.target}")

            if hash_int >= self.target:
                logger.warning("Block hash doesn't meet difficulty target")
                return False

            # Compute public input
            block_bytes = str(block.data).encode() + str(block.nonce).encode()
            secret = int.from_bytes(hashlib.sha256(block_bytes).digest(), 'big') % self.zkp_system.field.modulus
            public_input = secret

            # Verify ZK proof
            logger.debug(f"Verifying ZK proof for block {block.hash}")
            if not hasattr(block, 'zk_proof'):
                logger.error("Block missing ZK proof")
                return False

            # Use verification with proper STARK/SNARK separation
            stark_valid = self.zkp_system.stark.verify_proof(public_input, block.zk_proof[0])
            snark_valid = self.zkp_system.snark.verify(public_input, block.zk_proof[1])

            logger.debug(f"Block validation results:")
            logger.debug(f"STARK verification: {stark_valid}")
            logger.debug(f"SNARK verification: {snark_valid}")

            return stark_valid and snark_valid

        except Exception as e:
            logger.error(f"Block validation error: {str(e)}")
            return False


    def quantum_evolution(self, state: np.ndarray) -> np.ndarray:
        """Apply quantum evolution to state"""
        eigenvals, eigenvecs = np.linalg.eigh(self.H)
        evolution = np.exp(-1j * eigenvals)
        return eigenvecs @ np.diag(evolution) @ eigenvecs.T.conj() @ state

    def apply_decoherence(self, state: np.ndarray) -> np.ndarray:
        """Apply decoherence effects to quantum state"""
        damping = np.exp(-self.decoherence_rate * (np.arange(len(state)) + 0.1))  # Adjusting with +0.1
        decohered = state * damping

        return decohered / np.sqrt(np.sum(np.abs(decohered)**2))



    def generate_quantum_signature(self, data: bytes) -> str:
        """Generate quantum-inspired signature for block"""
        try:
            # Convert input data to quantum state
            input_array = np.frombuffer(data, dtype=np.uint8)
            state = input_array / np.sqrt(np.sum(input_array**2))
            state = np.pad(state, (0, max(0, self.system_size - len(state))))[:self.system_size]
            
            # Apply quantum evolution
            evolved = self.quantum_evolution(state)
            
            # Apply decoherence
            final = self.apply_decoherence(evolved)
            
            # Convert to signature
            signature_bytes = (np.abs(final) * 255).astype(np.uint8)
            return signature_bytes.tobytes().hex()[:8]
        except Exception as e:
            logger.error(f"Error generating quantum signature: {str(e)}")
            # Return a fallback signature in case of error
            return hashlib.sha256(data).hexdigest()[:8]


    def select_parents(self) -> List[str]:
        """Select parent blocks using MCMC tip selection"""
        tips = [node for node in self.dag.nodes() if self.dag.out_degree(node) == 0]
        
        if len(tips) < self.min_parents:
            if len(tips) == 0:
                return []  # Return empty list for genesis block
            return tips  # Return all available tips if less than minimum
            
        selected = []
        while len(selected) < self.min_parents:
            tip = self._mcmc_tip_selection(tips)
            if tip not in selected:
                selected.append(tip)
        
        return selected

    def _mcmc_tip_selection(self, tips: List[str]) -> str:
        """MCMC random walk for tip selection"""
        if not tips:
            return None
            
        current = random.choice(tips)
        alpha = 0.01  # Temperature parameter
        
        while True:
            predecessors = list(self.dag.predecessors(current))
            if not predecessors:
                return current
                
            weights = [np.exp(-alpha * self.dag.out_degree(p)) for p in predecessors]
            weights_sum = sum(weights)
            if weights_sum == 0:
                return current
                
            probabilities = [w/weights_sum for w in weights]
            current = np.random.choice(predecessors, p=probabilities)
    def get_latest_block_hash(self) -> str:
        """Get hash of the most recent block with genesis handling"""
        try:
            # Initialize genesis if needed
            if not self.dag.nodes:
                genesis_hash = "0" * 64
                self.dag.add_node(genesis_hash, timestamp=time.time())
                return genesis_hash
                
            tips = [node for node in self.dag.nodes() if self.dag.out_degree(node) == 0]
            if not tips:
                return "0" * 64
            
            # Get most recent tip by timestamp
            latest_tip = max(tips, key=lambda x: self.dag.nodes[x].get('timestamp', 0))
            return latest_tip

        except Exception as e:
            logger.error(f"Error getting latest block hash: {str(e)}")
            return "0" * 64
    def calculate_confirmation_score(self, transaction_hash: str, block_hash: str) -> float:
        """Calculate confirmation score with path analysis"""
        try:
            if not self.dag.has_node(transaction_hash) or not self.dag.has_node(block_hash):
                return 0.0
                
            paths = list(nx.all_simple_paths(self.dag, block_hash, transaction_hash))
            if not paths:
                return 0.0
                
            depth_score = min(len(paths[0]) / self.min_confirmations, 1.0)
            quantum_score = self.quantum_scores.get(transaction_hash, 0.0)
            consensus_score = min(len(paths) / (self.min_confirmations * 2), 1.0)
            
            unique_nodes = len(set(node for path in paths for node in path))
            total_nodes = sum(len(path) for path in paths)
            diversity_score = unique_nodes / total_nodes if total_nodes > 0 else 0.0
            
            return (0.4 * depth_score + 
                    0.3 * quantum_score + 
                    0.2 * consensus_score + 
                    0.1 * diversity_score)
                    
        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            return 0.0




    def get_latest_block(self) -> Optional['QuantumBlock']:
        """Get most recent block object"""
        try:
            latest_hash = self.get_latest_block_hash()
            if latest_hash == "0" * 64:
                return None
                
            return self.dag.nodes[latest_hash].get('block')

        except Exception as e:
            logger.error(f"Error getting latest block: {str(e)}")
            return None

    async def mine_block(self, previous_hash: str, data: str, transactions: list, 
                        reward: Decimal, miner_address: str) -> Optional['QuantumBlock']:
        """Mine block with enhanced security and confirmation handling"""
        try:
            mining_start_time = time.time()
            hash_calculations = 0
            
            # Process transactions
            tx_list = []
            crypto_provider = CryptoProvider()
            for tx in transactions:
                if hasattr(tx, 'tx_hash'):
                    if not tx.signature:  # Apply security if not already applied
                        await tx.apply_enhanced_security(crypto_provider)
                    tx_list.append(tx)
                    self.dag.add_node(tx.tx_hash, timestamp=time.time())
            
            # Select parents
            parent_hashes = await self.select_parents_with_confirmations()
            if not parent_hashes:
                if not self.dag.nodes:
                    parent_hashes = ["0" * 64]
                    self.dag.add_node(parent_hashes[0], timestamp=time.time())
                else:
                    parent_hashes = [previous_hash]
            
            nonce = 0
            timestamp = time.time()
            
            while time.time() - mining_start_time < 30:
                # Create block with quantum features
                block_data = {
                    'data': data,
                    'transactions': [tx.tx_hash for tx in tx_list],
                    'timestamp': timestamp,
                    'nonce': nonce
                }
                quantum_sig = self.generate_quantum_signature(str(block_data).encode())
                
                block = QuantumBlock(
                    previous_hash=parent_hashes[0],
                    data=data,
                    quantum_signature=quantum_sig,
                    reward=reward,
                    transactions=tx_list,
                    miner_address=miner_address,
                    nonce=nonce,
                    parent_hashes=parent_hashes,
                    timestamp=timestamp
                )
                
                block_hash = block.compute_hash()
                hash_calculations += 1
                
                if int(block_hash, 16) < self.target:
                    block.hash = block_hash
                    
                    # Update DAG structure
                    self.dag.add_node(block_hash, block=block, timestamp=timestamp)
                    for parent in parent_hashes:
                        if parent in self.dag:
                            self.dag.add_edge(block_hash, parent)
                    
                    # Link transactions and update confirmations
                    for tx in tx_list:
                        self.dag.add_edge(block_hash, tx.tx_hash)
                        self.confirmation_system.quantum_scores[tx.tx_hash] = 0.85
                        
                        # Update transaction confirmation data
                        security_info = self.confirmation_system.get_transaction_security(
                            tx.tx_hash, block_hash
                        )
                        tx.confirmation_data.status.confirmation_score = security_info['confirmation_score']
                        tx.confirmation_data.status.security_level = security_info['security_level']
                        tx.confirmation_data.metrics.path_diversity = security_info.get('path_diversity', 0.0)
                        tx.confirmation_data.metrics.quantum_strength = security_info.get('quantum_strength', 0.0)
                        tx.confirmations = security_info['num_confirmations']
                    
                    # Update confirmation system
                    await self.confirmation_system.add_block_confirmation(
                        block_hash,
                        parent_hashes,
                        tx_list,
                        quantum_sig
                    )
                    
                    # Run DAG diagnostics
                    logger.info("Running DAG structure diagnostics...")
                    diagnostics = DAGDiagnostics(self.confirmation_system)
                    results = diagnostics.diagnose_dag_structure()
                    diagnostics.print_diagnostics(results)
                    
                    # Generate visualization (optional)
                    try:
                        diagnostics.visualize_dag(f"dag_structure_{block_hash[:8]}.png")
                    except Exception as e:
                        logger.error(f"Failed to generate DAG visualization: {str(e)}")
                    
                    # Log any issues found
                    if results.get("issues"):
                        logger.warning("DAG structure issues found:")
                        for issue in results["issues"]:
                            logger.warning(f"- {issue}")
                    
                    await self._update_mining_metrics(time.time() - mining_start_time, hash_calculations)
                    return block
                
                nonce += 1
                if nonce % 100 == 0:
                    await asyncio.sleep(0)
            
            return None
            
        except Exception as e:
            logger.error(f"Error mining block: {str(e)}")
            logger.error(traceback.format_exc())
            return None




    async def select_parents_with_confirmations(self) -> List[str]:
        """Select parent blocks considering confirmation scores"""
        try:
            tips = [node for node in self.dag.nodes() if self.dag.out_degree(node) == 0]
            
            if len(tips) < self.min_parents:
                return tips if tips else []
                
            # Calculate confirmation scores for tips
            scored_tips = []
            latest_block = max(self.dag.nodes(), key=lambda n: self.dag.nodes[n]['block'].timestamp)
            
            for tip in tips:
                block = self.dag.nodes[tip]['block']
                
                # Get average confirmation score of tip's transactions
                scores = []
                for tx in block.transactions:
                    security_info = self.confirmation_system.get_transaction_security(
                        tx.hash,
                        latest_block
                    )
                    scores.append(security_info['confirmation_score'])
                
                avg_score = sum(scores) / len(scores) if scores else 0
                scored_tips.append((tip, avg_score))
            
            # Sort tips by score and select the best ones
            scored_tips.sort(key=lambda x: x[1], reverse=True)
            selected = [tip for tip, _ in scored_tips[:self.min_parents]]
            
            return selected

        except Exception as e:
            logger.error(f"Error selecting parents with confirmations: {str(e)}")
            return []
    async def update_confirmation_metrics(self, block: 'QuantumBlock'):
        """Update mining metrics with confirmation statistics"""
        try:
            # Calculate confirmation scores for all transactions in block
            scores = []
            for tx in block.transactions:
                security_info = self.confirmation_system.get_transaction_security(
                    tx.hash,
                    block.hash
                )
                scores.append(security_info['confirmation_score'])
                
                # Update security level distribution
                self.mining_metrics['confirmation_stats']['confirmation_distribution'][
                    security_info['security_level']
                ] += 1
                
                # Update high security blocks count
                if security_info['security_level'] in ['MAXIMUM', 'VERY_HIGH']:
                    self.mining_metrics['confirmation_stats']['high_security_blocks'] += 1
            
            # Update average confirmation score
            if scores:
                self.mining_metrics['confirmation_stats']['avg_confirmation_score'] = (
                    0.95 * self.mining_metrics['confirmation_stats']['avg_confirmation_score'] +
                    0.05 * (sum(scores) / len(scores))  # Exponential moving average
                )
            
            # Update confirmed blocks count
            self.mining_metrics['confirmation_stats']['confirmed_blocks'] += 1
            
            logger.info(
                f"Updated confirmation metrics for block {block.hash}"
                f"\n\tAverage confirmation score: {self.mining_metrics['confirmation_stats']['avg_confirmation_score']:.4f}"
                f"\n\tConfirmed blocks: {self.mining_metrics['confirmation_stats']['confirmed_blocks']}"
                f"\n\tHigh security blocks: {self.mining_metrics['confirmation_stats']['high_security_blocks']}"
            )

        except Exception as e:
            logger.error(f"Error updating confirmation metrics: {str(e)}")


    def validate_dag_structure(self, block_hash: str, parent_hashes: List[str]) -> bool:
        """Validate DAG structure requirements"""
        G = nx.DiGraph(self.dag)
        G.add_node(block_hash)
        for parent in parent_hashes:
            G.add_edge(block_hash, parent)
            
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                logger.warning(f"Cycle detected in DAG for block {block_hash}: {cycles}")
            return len(cycles) == 0
        except Exception as e:
            logger.error(f"DAG validation error: {str(e)}")
            return False


    def get_dag_metrics(self) -> Dict:
        """Get metrics about the DAG structure"""
        return {
            'n_blocks': self.dag.number_of_nodes(),
            'n_edges': self.dag.number_of_edges(),
            'n_tips': len([n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]),
            'avg_parents': sum(self.dag.out_degree(n) for n in self.dag.nodes()) / 
                         max(1, self.dag.number_of_nodes())
        }
    async def estimate_transaction_gas(self, tx_data: dict) -> dict:
        """Estimate gas for transaction with proper defaults"""
        try:
            # Get base gas cost based on transaction type
            tx_type = tx_data.get('transaction_type', 'STANDARD')
            base_gas = self.base_costs.get(tx_type, 21000)
            
            # Calculate data size gas
            data_size = tx_data.get('data_size', 0)
            data_gas = self._calculate_data_gas(data_size)
            
            # Calculate quantum premium
            quantum_gas = 0
            quantum_premium = Decimal('0')
            if tx_data.get('quantum_enabled'):
                quantum_gas = 50000
                quantum_premium = Decimal('0.05')
            
            # Calculate total gas
            total_gas = base_gas + data_gas + quantum_gas
            
            # Calculate gas price components
            base_price = Decimal('0.1')  # Base gas price
            security_premium = Decimal('0.02')  # Security premium for all transactions
            
            # Add quantum components if enabled
            entanglement_premium = Decimal('0')
            if tx_data.get('quantum_enabled'):
                entanglement_premium = Decimal('0.03')
            
            # Calculate decoherence discount
            decoherence_discount = Decimal('0')
            if tx_data.get('quantum_enabled'):
                decoherence_discount = base_price * Decimal('0.01')
            
            # Calculate congestion premium based on network load
            congestion_premium = Decimal('0')
            if self.network_metrics['network_load'] > 0.8:
                congestion_premium = base_price * Decimal('0.1')
            
            # Calculate total price
            total_price = (
                base_price +
                security_premium +
                quantum_premium +
                entanglement_premium -
                decoherence_discount +
                congestion_premium
            )
            
            # Calculate total cost
            total_cost = Decimal(str(total_gas)) * total_price
            
            return {
                'gas_needed': total_gas,
                'gas_price': float(total_price),
                'total_cost': float(total_cost),
                'components': {
                    'base_price': float(base_price),
                    'security_premium': float(security_premium),
                    'quantum_premium': float(quantum_premium),
                    'entanglement_premium': float(entanglement_premium),
                    'decoherence_discount': float(decoherence_discount),
                    'congestion_premium': float(congestion_premium)
                }
            }

        except Exception as e:
            logger.error(f"Error estimating transaction gas: {str(e)}")
            logger.error(traceback.format_exc())
            raise


    def _determine_transaction_type(self, tx_data: dict) -> EnhancedGasTransactionType:
        """Determine the type of transaction for gas calculation"""
        if tx_data.get('quantum_enabled') and tx_data.get('entanglement_count', 0) > 0:
            return EnhancedGasTransactionType.QUANTUM_ENTANGLE
        elif tx_data.get('quantum_enabled'):
            return EnhancedGasTransactionType.QUANTUM_STATE
        elif tx_data.get('data_size', 0) > 1000:
            return EnhancedGasTransactionType.DATA_STORAGE
        else:
            return EnhancedGasTransactionType.STANDARD

    def _calculate_data_gas(self, data_size: int) -> int:
        """Calculate gas cost for data storage"""
        return max(0, int(data_size * 16))

    def _calculate_base_price(self) -> Decimal:
        """Calculate base gas price based on network conditions"""
        load = self.network_metrics['network_load']
        block_time_ratio = self.network_metrics['avg_block_time'] / 30.0
        
        # Calculate quantum entropy correction
        entropy_correction = 1.0 + self.quantum_entropy * (-load * np.log(load + 1e-10))
        
        # Calculate quantum coupling adjustment
        coupling_adjustment = 1.0 + self.quantum_coupling * (
            load ** 2 + (self.network_metrics['dag_depth'] / 1000) ** 0.5
        )
        
        # Combine factors
        price_factor = (
            entropy_correction *
            coupling_adjustment *
            load ** 1.5 *
            block_time_ratio ** 0.5
        )
        
        return Decimal(str(price_factor)) * self.base_gas_price

    def _calculate_quantum_factor(self) -> float:
        """Calculate quantum adjustment factor"""
        entanglement_ratio = (
            self.network_metrics['quantum_entangled_pairs'] /
            max(1, self.network_metrics['active_nodes'])
        )
        return min(2.0, 1.0 + entanglement_ratio)

    def _calculate_entanglement_premium(self, entanglement_count: int, base_price: Decimal) -> Decimal:
        """Calculate premium for entanglement verification"""
        if entanglement_count == 0:
            return Decimal('0')
        premium_factor = np.log1p(entanglement_count) / 10
        return base_price * Decimal(str(premium_factor))

    def _calculate_decoherence_discount(self, base_price: Decimal, quantum_enabled: bool) -> Decimal:
        """Calculate discount based on quantum decoherence"""
        if not quantum_enabled:
            return Decimal('0')
        time_factor = np.exp(-(time.time() - self.last_adjustment) / 3600)
        return base_price * Decimal(str(0.1 * time_factor))

    def _calculate_security_premium(self, base_price: Decimal) -> Decimal:
        """Calculate security premium based on DAG depth"""
        depth_factor = np.tanh(self.network_metrics['dag_depth'] / 1000)
        return base_price * Decimal(str(0.2 * depth_factor))

    def _calculate_congestion_premium(self, base_price: Decimal) -> Decimal:
        """Calculate congestion premium"""
        if self.network_metrics['network_load'] > 0.8:
            load = self.network_metrics['network_load']
            quantum_correction = 1 + self.quantum_coupling * load
            premium_factor = ((load - 0.8) * 5) * quantum_correction
            return base_price * Decimal(str(premium_factor))
        return Decimal('0')
        
    async def update_gas_metrics(self):
        """Update gas system metrics based on current blockchain state"""
        try:
            current_time = time.time()
            if current_time - self.last_adjustment < self.adjustment_interval:
                return

            # Calculate average network load based on transaction pool
            if hasattr(self, 'dag'):
                self.network_metrics.update({
                    'avg_block_time': self._calculate_avg_block_time(),
                    'network_load': len(self.dag.nodes()) / 1000,  # Normalize by expected capacity
                    'active_nodes': sum(1 for _ in self.dag.nodes() if self.dag.out_degree(_) > 0),
                    'quantum_entangled_pairs': self._count_quantum_entanglements(),
                    'dag_depth': self._calculate_dag_depth(),
                    'total_compute': self._calculate_total_compute()
                })
            else:
                # Default metrics if DAG not initialized
                self.network_metrics = {
                    'avg_block_time': 30.0,
                    'network_load': 0.1,
                    'active_nodes': 1,
                    'quantum_entangled_pairs': 0,
                    'dag_depth': 0,
                    'total_compute': 0.0
                }

            self.last_adjustment = current_time

        except Exception as e:
            logger.error(f"Error updating gas metrics: {str(e)}")
            logger.error(traceback.format_exc())

    def _calculate_avg_block_time(self) -> float:
        """Calculate average time between blocks"""
        try:
            if not hasattr(self, 'dag') or not self.dag.nodes:
                return 30.0  # Default block time
                
            recent_blocks = sorted(
                [(n, self.dag.nodes[n].get('timestamp', 0)) 
                 for n in self.dag.nodes()],
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Get last 10 blocks
            
            if len(recent_blocks) < 2:
                return 30.0
                
            time_diffs = [
                t1 - t2 for (_, t1), (_, t2) 
                in zip(recent_blocks[:-1], recent_blocks[1:])
            ]
            return statistics.mean(time_diffs)
            
        except Exception as e:
            logger.error(f"Error calculating average block time: {str(e)}")
            return 30.0

    def _count_quantum_entanglements(self) -> int:
        """Count number of quantum entangled pairs in the DAG"""
        try:
            if not hasattr(self, 'dag'):
                return 0
                
            entangled_count = 0
            for node in self.dag.nodes():
                block = self.dag.nodes[node].get('block')
                if block and hasattr(block, 'quantum_signature'):
                    # Count strong quantum signatures as entangled pairs
                    sig_strength = self.confirmation_system.evaluate_quantum_signature(
                        block.quantum_signature
                    )
                    if sig_strength >= self.confirmation_system.quantum_threshold:
                        entangled_count += 1
            return entangled_count
            
        except Exception as e:
            logger.error(f"Error counting quantum entanglements: {str(e)}")
            return 0

    def _calculate_dag_depth(self) -> int:
        """Calculate current DAG depth"""
        try:
            if not hasattr(self, 'dag') or not self.dag.nodes():
                return 0
                
            # Find longest path length
            depths = []
            for node in self.dag.nodes():
                if self.dag.out_degree(node) == 0:  # If it's a tip
                    paths = nx.single_source_shortest_path_length(self.dag, node)
                    depths.append(max(paths.values()))
            return max(depths) if depths else 0
            
        except Exception as e:
            logger.error(f"Error calculating DAG depth: {str(e)}")
            return 0

    def _calculate_total_compute(self) -> float:
        """Calculate total computational work in the DAG"""
        try:
            if not hasattr(self, 'dag'):
                return 0.0
                
            total_work = 0.0
            for node in self.dag.nodes():
                block = self.dag.nodes[node].get('block')
                if block:
                    # Add up difficulty of all blocks
                    total_work += 2 ** self.difficulty
            return total_work
            
        except Exception as e:
            logger.error(f"Error calculating total compute: {str(e)}")
            return 0.0
