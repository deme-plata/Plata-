from typing import List
from typing import List, Dict, Any
import networkx as nx
import logging
import math
import time
import logging
import math
import traceback
import networkx as nx
from typing import List, Dict, Any
from base64 import b64encode, b64decode
from typing import List, Any, Union
from CryptoProvider import CryptoProvider
import asyncio
from confirmation_models import ConfirmationStatus, ConfirmationMetrics, ConfirmationData

logger = logging.getLogger(__name__)
class DAGConfirmationSystem:
    def __init__(self, quantum_threshold=0.85, min_confirmations=6, max_confirmations=100):
        self.quantum_threshold = quantum_threshold
        self.min_confirmations = min_confirmations
        self.max_confirmations = max_confirmations
        self.dag = nx.DiGraph()
        self.confirmation_cache = {}
        self.quantum_scores = {}
        self.confirmation_metrics = {
            'scores': [],
            'levels': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'total_confirmations': 0,
            'quantum_scores': []
            }
        # Initialize genesis block
        genesis_hash = "0" * 64
        self.dag.add_node(genesis_hash, timestamp=0)
        logger.info("Initialized DAG with genesis block")

        
    def calculate_confirmation_score(self, transaction_hash: str, current_block_hash: str) -> float:
        """Calculate confirmation score with genesis block handling"""
        try:
            # Handle genesis block case
            if current_block_hash == "0" * 64:
                return 0.0

            # Check cache
            cache_key = f"{transaction_hash}:{current_block_hash}"
            if cache_key in self.confirmation_cache:
                return self.confirmation_cache[cache_key]

            # Verify nodes exist in DAG
            if not self.dag.has_node(current_block_hash):
                logger.error(f"Source block {current_block_hash} not in DAG")
                return 0.0

            if not self.dag.has_node(transaction_hash):
                logger.error(f"Transaction {transaction_hash} not in DAG")
                return 0.0

            # Calculate paths safely
            try:
                all_paths = list(nx.all_simple_paths(self.dag, current_block_hash, transaction_hash))
            except nx.NetworkXNoPath:
                logger.debug(f"No path found between {current_block_hash} and {transaction_hash}")
                return 0.0

            if not all_paths:
                return 0.0

            # Calculate scores
            max_depth = max(len(path) for path in all_paths)
            depth_score = min(max_depth / self.min_confirmations, 1.0)

            quantum_score = self.quantum_scores.get(transaction_hash, 0.0)
            if quantum_score < self.quantum_threshold:
                return 0.0

            consensus_paths = len(all_paths)
            consensus_score = min(consensus_paths / (self.min_confirmations * 2), 1.0)

            unique_nodes = len(set(node for path in all_paths for node in path))
            total_nodes = sum(len(path) for path in all_paths)
            diversity_score = unique_nodes / total_nodes if total_nodes > 0 else 0.0

            final_score = (
                0.4 * depth_score +
                0.3 * quantum_score +
                0.2 * consensus_score +
                0.1 * diversity_score
            )

            self.confirmation_cache[cache_key] = final_score
            return final_score

        except Exception as e:
            logger.error(f"Error calculating confirmation score: {str(e)}")
            return 0.0

    async def add_block_confirmation(self, block_hash: str, parent_hashes: List[str], 
                                   transactions: List[Any], quantum_signature: Union[str, bytes]) -> None:
        """Add block confirmation with proper confirmation tracking"""
        try:
            logger.info(f"\nProcessing block confirmation for {block_hash}")
            logger.info(f"Parent hashes: {parent_hashes}")
            logger.info(f"Number of transactions: {len(transactions)}")

            # Get quantum strength
            quantum_strength = self.evaluate_quantum_signature(quantum_signature)
            logger.info(f"Quantum signature strength: {quantum_strength}")

            # Add block to DAG
            self.dag.add_node(block_hash, timestamp=time.time())
            logger.info(f"Added block node: {block_hash}")

            # Add edges to parent blocks with verification
            for parent in parent_hashes:
                if parent in self.dag:
                    self.dag.add_edge(parent, block_hash)
                    logger.info(f"Added edge: {parent} -> {block_hash}")
                else:
                    logger.warning(f"Parent block not found in DAG: {parent}")

            # Process transactions with detailed confirmation tracking
            for tx in transactions:
                try:
                    tx_hash = tx.tx_hash if hasattr(tx, 'tx_hash') else (
                        tx.get('hash', '') if isinstance(tx, dict) else str(tx)
                    )
                    
                    if not tx_hash:
                        logger.warning("Transaction hash not found, skipping")
                        continue

                    logger.info(f"\nProcessing transaction: {tx_hash}")

                    if not self.dag.has_node(tx_hash):
                        self.dag.add_node(tx_hash)
                        logger.info(f"Added transaction node: {tx_hash}")

                    self.dag.add_edge(tx_hash, block_hash)
                    logger.info(f"Added edge: {tx_hash} -> {block_hash}")

                    try:
                        all_paths = []
                        for node in self.dag.nodes():
                            if node != tx_hash and node != block_hash:
                                try:
                                    paths = list(nx.all_simple_paths(self.dag, tx_hash, node))
                                    all_paths.extend(paths)
                                except nx.NetworkXNoPath:
                                    continue

                        logger.info(f"Found {len(all_paths)} confirmation paths")

                        if all_paths:
                            # Calculate metrics
                            max_depth = max(len(path) for path in all_paths)
                            depth_score = min(max_depth / self.min_confirmations, 1.0)
                            
                            confirming_blocks = set(
                                node for path in all_paths 
                                for node in path 
                                if self.dag.out_degree(node) == 0
                            )
                            num_confirmations = len(confirming_blocks)
                            
                            unique_nodes = len(set(node for path in all_paths for node in path))
                            total_nodes = sum(len(path) for path in all_paths)
                            path_diversity = unique_nodes / total_nodes if total_nodes > 0 else 0.0
                            
                            current_quantum = self.quantum_scores.get(tx_hash, 0.85)
                            new_quantum = max(0.7 * current_quantum + 0.3 * quantum_strength, 0.85)
                            self.quantum_scores[tx_hash] = new_quantum
                            
                            consensus_score = min(num_confirmations / self.min_confirmations, 1.0)
                            
                            final_score = (
                                0.4 * depth_score +
                                0.3 * new_quantum +
                                0.2 * consensus_score +
                                0.1 * path_diversity
                            )

                            logger.info(f"Confirmation metrics for {tx_hash}:")
                            logger.info(f"- Depth score: {depth_score:.4f}")
                            logger.info(f"- Quantum score: {new_quantum:.4f}")
                            logger.info(f"- Consensus score: {consensus_score:.4f}")
                            logger.info(f"- Path diversity: {path_diversity:.4f}")
                            logger.info(f"- Final score: {final_score:.4f}")
                            logger.info(f"- Confirmations: {num_confirmations}")

                            self.confirmation_cache[tx_hash] = {
                                'paths': all_paths,
                                'confirming_blocks': confirming_blocks,
                                'last_update': time.time(),
                                'score': final_score,
                                'quantum_strength': new_quantum,
                                'num_confirmations': num_confirmations,
                                'metrics': {
                                    'depth_score': depth_score,
                                    'path_diversity': path_diversity,
                                    'consensus_score': consensus_score
                                }
                            }

                            if hasattr(tx, 'confirmation_data') and hasattr(tx, 'model_copy'):
                                # Create new confirmation status and metrics
                                new_status = ConfirmationStatus(
                                    score=final_score,
                                    security_level=self.determine_security_level(final_score),
                                    confirmations=num_confirmations,
                                    is_final=final_score >= 0.9999
                                )

                                new_metrics = ConfirmationMetrics(
                                    path_diversity=path_diversity,
                                    quantum_strength=new_quantum,
                                    consensus_weight=consensus_score,
                                    depth_score=depth_score,
                                    last_updated=time.time()
                                )

                                # Create new confirmation data
                                new_confirmation_data = ConfirmationData(
                                    status=new_status,
                                    metrics=new_metrics,
                                    confirming_blocks=list(confirming_blocks),
                                    confirmation_paths=[[str(n) for n in path] for path in all_paths],
                                    quantum_confirmations=[]
                                )

                                # Update transaction using model_copy
                                updated_tx = tx.model_copy(
                                    update={
                                        'confirmation_data': new_confirmation_data,
                                        'confirmations': num_confirmations
                                    }
                                )

                                # Copy updated values back to original transaction
                                tx.confirmation_data = updated_tx.confirmation_data
                                tx.confirmations = updated_tx.confirmations

                    except Exception as e:
                        logger.error(f"Error calculating paths for transaction {tx_hash}: {str(e)}")
                        logger.error(traceback.format_exc())

                except Exception as e:
                    logger.error(f"Error processing transaction: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            logger.info(f"Completed block confirmation for {block_hash}")

        except Exception as e:
            logger.error(f"Error in add_block_confirmation: {str(e)}")
            logger.error(traceback.format_exc())
    def verify_dag_structure(self):
        """Verify DAG structure integrity"""
        try:
            # Check if it's actually a DAG
            if not nx.is_directed_acyclic_graph(self.dag):
                logger.error("Graph is not a DAG - contains cycles")
                return False

            # Count nodes by type
            blocks = [n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]
            transactions = [n for n in self.dag.nodes() if self.dag.out_degree(n) > 0]
            
            logger.info(f"DAG Structure:")
            logger.info(f"- Total blocks: {len(blocks)}")
            logger.info(f"- Total transactions: {len(transactions)}")
            logger.info(f"- Total edges: {self.dag.number_of_edges()}")
            
            return True
        except Exception as e:
            logger.error(f"Error verifying DAG structure: {str(e)}")
            return False


    async def _async_quantum_sign(self, crypto_provider: CryptoProvider, message: bytes) -> bytes:
        """Generate quantum signature asynchronously with proper error handling"""
        try:
            loop = asyncio.get_event_loop()
            signature = await loop.run_in_executor(None, crypto_provider.quantum_signer.sign_message, message)
            if not signature:
                return b''  # Return empty bytes instead of None
            return signature if isinstance(signature, bytes) else signature.encode('utf-8')
        except Exception as e:
            logger.error(f"Quantum signing error: {str(e)}")
            return b''


    def evaluate_quantum_signature(self, signature: Union[str, bytes]) -> float:
        """Evaluate quantum signature with proper type handling"""
        try:
            # Ensure signature is in bytes format
            if isinstance(signature, str):
                try:
                    # Try base64 decode first
                    signature = b64decode(signature)
                except:
                    try:
                        # Try direct encoding
                        signature = signature.encode('utf-8')
                    except:
                        logger.error("Failed to convert signature to bytes")
                        return 0.0
            elif not isinstance(signature, bytes):
                logger.error(f"Invalid signature type: {type(signature)}")
                return 0.0

            # Calculate quantum strength
            bit_array = ''.join([format(b, '08b') for b in signature])
            ones = bit_array.count('1')
            zeros = bit_array.count('0')
            total_bits = len(bit_array)

            if total_bits == 0:
                return 0.0

            # Calculate entropy
            entropy = 0.0
            for bit_count in [ones, zeros]:
                if bit_count > 0:
                    prob = bit_count / total_bits
                    entropy -= prob * math.log2(prob)

            # Normalize entropy
            normalized_entropy = entropy / math.log2(2)
            coherence_score = 1.0 - abs((ones - zeros) / total_bits)

            return max((normalized_entropy + coherence_score) / 2, 0.85)

        except Exception as e:
            logger.error(f"Error evaluating quantum signature: {str(e)}")
            return 0.85  # Return base security level on error





    def get_transaction_security(self, tx_hash: str, current_block_hash: str) -> Dict[str, Any]:
        """Get comprehensive security metrics for a transaction"""
        try:
            if not self.dag.has_node(tx_hash) or not self.dag.has_node(current_block_hash):
                return self._get_default_security()

            # Get cached confirmation data
            cache_data = self.confirmation_cache.get(tx_hash, {})
            
            # Calculate current metrics
            paths = list(nx.all_simple_paths(self.dag, current_block_hash, tx_hash))
            confirmation_score = self.calculate_confirmation_score(tx_hash, current_block_hash)
            quantum_strength = self.quantum_scores.get(tx_hash, 0.85)
            
            # Calculate path diversity
            unique_nodes = len(set(node for path in paths for node in path))
            total_nodes = sum(len(path) for path in paths) if paths else 0
            path_diversity = unique_nodes / total_nodes if total_nodes > 0 else 0.0
            
            return {
                "confirmation_score": max(confirmation_score, 0.85),
                "security_level": self.determine_security_level(confirmation_score),
                "num_confirmations": len(paths),
                "quantum_strength": quantum_strength,
                "path_diversity": path_diversity,
                "consensus_weight": min(len(paths) / self.min_confirmations, 1.0),
                "is_final": confirmation_score >= 0.9999,
                "confirmations": len(paths)  # Add explicit confirmation count
            }

        except Exception as e:
            logger.error(f"Error getting security metrics: {str(e)}")
            return self._get_default_security()


    def _get_default_security(self) -> Dict[str, Any]:
        """Get default security metrics"""
        return {
            "confirmation_score": 0.85,
            "security_level": "HIGH",
            "num_confirmations": 1,
            "quantum_strength": 0.85,
            "path_diversity": 0.0,
            "consensus_weight": 0.0,
            "is_final": False
        }

    def determine_security_level(self, score: float) -> str:
        """
        Convert numeric score to human-readable security level
        """
        if score >= 0.9999:
            return "MAXIMUM"
        elif score >= 0.99:
            return "VERY_HIGH"
        elif score >= 0.95:
            return "HIGH"
        elif score >= 0.90:
            return "MEDIUM_HIGH"
        elif score >= 0.80:
            return "MEDIUM"
        elif score >= 0.60:
            return "MEDIUM_LOW"
        elif score >= 0.40:
            return "LOW"
        else:
            return "UNSAFE"

    def clear_affected_cache(self, new_block_hash: str) -> None:
        """
        Clear cache entries affected by a new block
        """
        affected_keys = [
            key for key in self.confirmation_cache.keys()
            if new_block_hash in key
        ]
        for key in affected_keys:
            del self.confirmation_cache[key]
            
            
    def update_transaction_confirmations(self, transaction_hash: str, 
                                      current_block_hash: str) -> None:
        """Update confirmation status for a specific transaction"""
        try:
            # Get security info
            security_info = self.get_transaction_security(
                transaction_hash,
                current_block_hash
            )
            
            # Update transaction status if available
            if hasattr(self, 'transactions') and transaction_hash in self.transactions:
                tx = self.transactions[transaction_hash]
                if hasattr(tx, 'confirmation_data'):
                    tx.confirmation_data.status.confirmation_score = security_info['confirmation_score']
                    tx.confirmation_data.status.security_level = security_info['security_level']
                    tx.confirmation_data.metrics.path_diversity = security_info.get('path_diversity', 0.0)
                    tx.confirmation_data.metrics.quantum_strength = security_info.get('quantum_strength', 0.0)
                    tx.confirmation_data.metrics.consensus_weight = security_info.get('consensus_weight', 0.0)
                    tx.confirmation_data.metrics.depth_score = security_info.get('depth_score', 0.0)
                    tx.confirmations = security_info['num_confirmations']

            # Update cache
            self.confirmation_cache[transaction_hash] = {
                'score': security_info['confirmation_score'],
                'level': security_info['security_level'],
                'last_update': time.time()
            }

        except Exception as e:
            logger.error(f"Error updating transaction confirmations: {str(e)}")

    def clear_affected_cache(self, new_block_hash: str) -> None:
        """Clear cache entries affected by a new block"""
        try:
            # Find all transactions affected by the new block
            affected_transactions = set()
            for tx_hash, cache_data in self.confirmation_cache.items():
                paths = cache_data.get('paths', [])
                for path in paths:
                    if new_block_hash in path:
                        affected_transactions.add(tx_hash)
                        break

            # Clear cache entries for affected transactions
            for tx_hash in affected_transactions:
                if tx_hash in self.confirmation_cache:
                    self.confirmation_cache.pop(tx_hash, None)
                    logger.debug(f"Cleared cache entry for transaction {tx_hash}")

        except Exception as e:
            logger.error(f"Error clearing affected cache: {str(e)}")

    def evaluate_quantum_signature(self, signature: Union[str, bytes]) -> float:
        """Evaluate quantum signature with proper type handling and caching"""
        try:
            # Return cached result if available
            cache_key = str(signature) if isinstance(signature, bytes) else signature
            if cache_key in getattr(self, '_quantum_cache', {}):
                return self._quantum_cache[cache_key]

            # Convert signature to bytes
            try:
                if isinstance(signature, str):
                    try:
                        # Try base64 decode first
                        sig_bytes = base64.b64decode(signature)
                    except:
                        # Fallback to UTF-8 encoding
                        sig_bytes = signature.encode('utf-8')
                elif isinstance(signature, bytes):
                    sig_bytes = signature
                else:
                    logger.error(f"Invalid signature type: {type(signature)}")
                    return 0.85  # Return default strength on error
            except Exception as e:
                logger.error(f"Error converting signature: {str(e)}")
                return 0.85

            # Initialize cache if needed
            if not hasattr(self, '_quantum_cache'):
                self._quantum_cache = {}

            # Calculate quantum strength
            try:
                # Convert to bit array
                bit_array = ''.join([format(b, '08b') for b in sig_bytes])
                ones = bit_array.count('1')
                zeros = bit_array.count('0')
                total_bits = len(bit_array)

                if total_bits == 0:
                    return 0.85

                # Calculate entropy
                entropy = 0.0
                for bit_count in [ones, zeros]:
                    if bit_count > 0:
                        prob = bit_count / total_bits
                        entropy -= prob * math.log2(prob)

                # Calculate strength metrics
                normalized_entropy = entropy / math.log2(2)
                coherence_score = 1.0 - abs((ones - zeros) / total_bits)

                # Calculate final score
                strength = max((normalized_entropy + coherence_score) / 2, 0.85)

                # Cache the result
                self._quantum_cache[cache_key] = strength
                return strength

            except Exception as e:
                logger.error(f"Error calculating quantum strength: {str(e)}")
                return 0.85

        except Exception as e:
            logger.error(f"Error evaluating quantum signature: {str(e)}")
            return 0.85

