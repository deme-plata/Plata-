import networkx as nx
import numpy as np
import logging
import time
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import traceback

logger = logging.getLogger(__name__)

class DAGPruningSystem:
    """
    Enhanced DAG pruning system for QuantumDAGKnight with quantum-aware pruning strategies.
    Implements sophisticated pruning mechanisms while preserving quantum signatures and
    confirmation security.
    """
    def __init__(self, min_confirmations: int = 6, 
                 quantum_threshold: float = 0.85,
                 prune_interval: int = 1000,
                 max_dag_size: int = 10000,
                 min_security_level: str = "HIGH"):
        
        # Core parameters
        self.min_confirmations = min_confirmations
        self.quantum_threshold = quantum_threshold
        self.prune_interval = prune_interval
        self.max_dag_size = max_dag_size
        self.min_security_level = min_security_level
        
        # Track pruning metrics
        self.pruning_stats = {
            'nodes_pruned': 0,
            'last_prune_time': 0,
            'pruning_history': [],
            'preserved_nodes': set(),
            'quantum_scores': {},
            'confirmation_cache': {}
        }
        
        # Security levels mapping
        self.security_levels = {
            "MAXIMUM": 5,
            "VERY_HIGH": 4,
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
            "UNSAFE": 0
        }
        
        logger.info(f"Initialized DAG pruning system with min_confirmations={min_confirmations}, "
                   f"quantum_threshold={quantum_threshold}")

    async def prune_dag(self, dag: nx.DiGraph, confirmation_system) -> Tuple[nx.DiGraph, Dict]:
        """
        Prune the DAG while preserving quantum entanglement and confirmation security.
        
        Args:
            dag: Current DAG
            confirmation_system: DAG confirmation system instance
        
        Returns:
            Tuple[nx.DiGraph, Dict]: Pruned DAG and pruning statistics
        """
        try:
            start_time = time.time()
            initial_size = dag.number_of_nodes()
            
            if initial_size < self.max_dag_size:
                logger.debug("DAG size below threshold, skipping pruning")
                return dag, self._get_pruning_stats(0, start_time)

            logger.info(f"Starting DAG pruning. Initial size: {initial_size} nodes")

            # 1. Identify critical nodes (those with high quantum signatures or security)
            critical_nodes = await self._identify_critical_nodes(dag, confirmation_system)
            logger.debug(f"Identified {len(critical_nodes)} critical nodes")

            # 2. Calculate confirmation paths and scores
            path_scores = self._analyze_confirmation_paths(dag, confirmation_system)
            logger.debug("Analyzed confirmation paths")

            # 3. Identify prunable nodes
            prunable_nodes = await self._identify_prunable_nodes(
                dag, 
                critical_nodes,
                path_scores, 
                confirmation_system
            )
            logger.debug(f"Identified {len(prunable_nodes)} prunable nodes")

            # 4. Perform selective pruning
            pruned_dag = self._perform_selective_pruning(dag, prunable_nodes)
            nodes_pruned = initial_size - pruned_dag.number_of_nodes()

            # 5. Verify DAG integrity after pruning
            if not self._verify_dag_integrity(pruned_dag, confirmation_system):
                logger.error("DAG integrity check failed after pruning")
                return dag, self._get_pruning_stats(0, start_time)

            # 6. Update pruning statistics
            pruning_stats = self._get_pruning_stats(nodes_pruned, start_time)
            
            logger.info(
                f"Pruning complete. Removed {nodes_pruned} nodes. "
                f"New size: {pruned_dag.number_of_nodes()} nodes"
            )
            
            return pruned_dag, pruning_stats

        except Exception as e:
            logger.error(f"Error during DAG pruning: {str(e)}")
            logger.error(traceback.format_exc())
            return dag, self._get_pruning_stats(0, start_time)

    async def _identify_critical_nodes(self, dag: nx.DiGraph, 
                                     confirmation_system) -> Set[str]:
        """Identify nodes that are critical for DAG security and quantum state."""
        try:
            critical_nodes = set()

            # 1. Add nodes with high quantum signatures
            for node in dag.nodes():
                quantum_score = confirmation_system.quantum_scores.get(node, 0.0)
                if quantum_score >= self.quantum_threshold:
                    critical_nodes.add(node)

            # 2. Add nodes on critical confirmation paths
            tips = {n for n in dag.nodes() if dag.out_degree(n) == 0}
            
            for tip in tips:
                security_info = confirmation_system.get_transaction_security(
                    tip,
                    list(dag.predecessors(tip))[0] if dag.predecessors(tip) else None
                )
                
                if self._is_security_level_sufficient(security_info['security_level']):
                    paths = nx.all_simple_paths(dag, tip, 
                                              list(dag.predecessors(tip))[0] 
                                              if dag.predecessors(tip) else None)
                    
                    for path in paths:
                        critical_nodes.update(path)

            # 3. Add nodes with high confirmation scores
            for node in dag.nodes():
                cache_entry = confirmation_system.confirmation_cache.get(node, {})
                score = cache_entry.get('score', 0.0)
                
                if score >= 0.95:
                    critical_nodes.add(node)

            return critical_nodes

        except Exception as e:
            logger.error(f"Error identifying critical nodes: {str(e)}")
            return set()

    def _analyze_confirmation_paths(self, dag: nx.DiGraph, 
                                  confirmation_system) -> Dict[str, float]:
        """Analyze and score confirmation paths."""
        path_scores = {}
        
        try:
            # Calculate path importance scores
            for node in dag.nodes():
                # Get confirmation paths
                paths = []
                predecessors = list(dag.predecessors(node))
                
                if predecessors:
                    source = predecessors[0]
                    try:
                        paths = list(nx.all_simple_paths(dag, source, node))
                    except nx.NetworkXNoPath:
                        continue

                # Score based on multiple factors
                if paths:
                    # Path diversity score
                    unique_nodes = len(set(n for path in paths for n in path))
                    total_nodes = sum(len(path) for path in paths)
                    diversity_score = unique_nodes / total_nodes if total_nodes > 0 else 0

                    # Quantum strength score
                    quantum_scores = [
                        confirmation_system.quantum_scores.get(n, 0.0)
                        for path in paths
                        for n in path
                    ]
                    quantum_score = np.mean(quantum_scores) if quantum_scores else 0

                    # Confirmation score
                    confirmation_score = min(len(paths) / self.min_confirmations, 1.0)

                    # Combined score with weights
                    path_scores[node] = (
                        0.4 * diversity_score +
                        0.4 * quantum_score +
                        0.2 * confirmation_score
                    )
                else:
                    path_scores[node] = 0.0

            return path_scores

        except Exception as e:
            logger.error(f"Error analyzing confirmation paths: {str(e)}")
            return {}

    async def _identify_prunable_nodes(self, dag: nx.DiGraph, 
                                     critical_nodes: Set[str],
                                     path_scores: Dict[str, float],
                                     confirmation_system) -> Set[str]:
        """Identify nodes that can be safely pruned."""
        try:
            prunable_nodes = set()
            
            for node in dag.nodes():
                if node in critical_nodes:
                    continue

                # Check node age and confirmation status
                security_info = confirmation_system.get_transaction_security(
                    node,
                    list(dag.predecessors(node))[0] if dag.predecessors(node) else None
                )

                # Node is prunable if:
                # 1. It has sufficient confirmations
                # 2. It's not critical for quantum state
                # 3. It has low path importance score
                # 4. It's not needed for recent transactions
                if (security_info['num_confirmations'] >= self.min_confirmations and
                    node not in critical_nodes and
                    path_scores.get(node, 0.0) < 0.5 and
                    not self._is_recent_transaction(node, dag)):
                    
                    prunable_nodes.add(node)

            return prunable_nodes

        except Exception as e:
            logger.error(f"Error identifying prunable nodes: {str(e)}")
            return set()

    def _perform_selective_pruning(self, dag: nx.DiGraph, 
                                 prunable_nodes: Set[str]) -> nx.DiGraph:
        """Perform selective pruning while maintaining DAG connectivity."""
        try:
            # Create a copy of the DAG for pruning
            pruned_dag = dag.copy()
            
            # Sort prunable nodes by their impact on DAG structure
            sorted_prunable = sorted(
                prunable_nodes,
                key=lambda n: pruned_dag.degree(n),
                reverse=True  # Start with high-degree nodes
            )
            
            # Prune nodes while maintaining DAG properties
            for node in sorted_prunable:
                # Check if removing the node would disconnect the DAG
                if self._is_safe_to_remove(pruned_dag, node):
                    pruned_dag.remove_node(node)
                
                # Stop if we've reached target size
                if pruned_dag.number_of_nodes() <= self.max_dag_size:
                    break

            return pruned_dag

        except Exception as e:
            logger.error(f"Error during selective pruning: {str(e)}")
            return dag

    def _verify_dag_integrity(self, dag: nx.DiGraph, 
                            confirmation_system) -> bool:
        """Verify DAG integrity after pruning."""
        try:
            # 1. Check if it's still a DAG
            if not nx.is_directed_acyclic_graph(dag):
                logger.error("Pruned graph is not a DAG")
                return False

            # 2. Check connectivity
            if not nx.is_weakly_connected(dag):
                logger.error("Pruned DAG is not connected")
                return False

            # 3. Verify quantum state preservation
            quantum_scores = [
                confirmation_system.quantum_scores.get(node, 0.0)
                for node in dag.nodes()
            ]
            if min(quantum_scores) < self.quantum_threshold:
                logger.error("Quantum state compromised after pruning")
                return False

            # 4. Verify confirmation paths
            tips = {n for n in dag.nodes() if dag.out_degree(n) == 0}
            for tip in tips:
                security_info = confirmation_system.get_transaction_security(
                    tip,
                    list(dag.predecessors(tip))[0] if dag.predecessors(tip) else None
                )
                if security_info['num_confirmations'] < self.min_confirmations:
                    logger.error(f"Insufficient confirmations for tip {tip}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error verifying DAG integrity: {str(e)}")
            return False

    def _is_security_level_sufficient(self, level: str) -> bool:
        """Check if security level meets minimum requirement."""
        return self.security_levels.get(level, 0) >= self.security_levels.get(self.min_security_level, 0)

    def _is_recent_transaction(self, node: str, dag: nx.DiGraph) -> bool:
        """Check if node represents a recent transaction."""
        try:
            node_data = dag.nodes[node]
            if 'timestamp' in node_data:
                current_time = time.time()
                # Consider transactions from last hour as recent
                return (current_time - node_data['timestamp']) < 3600
            return False
        except:
            return False

    def _is_safe_to_remove(self, dag: nx.DiGraph, node: str) -> bool:
        """Check if it's safe to remove a node."""
        try:
            # Get node's neighbors
            predecessors = list(dag.predecessors(node))
            successors = list(dag.successors(node))

            # Check if removal would break important paths
            if predecessors and successors:
                # Temporarily remove node
                dag.remove_node(node)
                
                # Check if paths still exist between predecessors and successors
                for pred in predecessors:
                    for succ in successors:
                        if not nx.has_path(dag, pred, succ):
                            # Restore node and return False
                            dag.add_node(node)
                            for p in predecessors:
                                dag.add_edge(p, node)
                            for s in successors:
                                dag.add_edge(node, s)
                            return False
                
                # Restore node as we're just checking
                dag.add_node(node)
                for p in predecessors:
                    dag.add_edge(p, node)
                for s in successors:
                    dag.add_edge(node, s)
                
                return True
            
            return True  # Safe to remove if node has no connections

        except Exception as e:
            logger.error(f"Error checking node removal safety: {str(e)}")
            return False

    def _get_pruning_stats(self, nodes_pruned: int, start_time: float) -> Dict:
        """Get pruning statistics."""
        stats = {
            'nodes_pruned': nodes_pruned,
            'pruning_time': time.time() - start_time,
            'last_prune_time': time.time()
        }
        self.pruning_stats['pruning_history'].append(stats)
        return stats