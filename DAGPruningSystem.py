import networkx as nx
import numpy as np
import logging
import time
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import traceback
from unittest.mock import Mock

logger = logging.getLogger(__name__)

class DAGPruningSystem:
    def __init__(self, min_confirmations: int = 6, 
                 quantum_threshold: float = 0.85,
                 prune_interval: int = 1000,
                 max_dag_size: int = 10000,
                 min_security_level: str = "HIGH"):
        self.min_confirmations = min_confirmations
        self.quantum_threshold = quantum_threshold
        self.prune_interval = prune_interval
        self.max_dag_size = max_dag_size
        self.min_security_level = min_security_level
        self.security_levels = {
            "MAXIMUM": 5, "VERY_HIGH": 4, "HIGH": 3,
            "MEDIUM": 2, "LOW": 1, "UNSAFE": 0
        }
        self.pruning_stats = {
            'nodes_pruned': 0,
            'last_prune_time': 0,
            'conflicts_resolved': 0
        }


    def _safely_remove_node(self, dag: nx.DiGraph, node: str) -> None:
        """Safely remove a node while preserving valid paths between its neighbors."""
        try:
            # Never remove genesis node
            if node == "0" * 64:
                return
                
            # Get timestamps before removal
            pred_times = {}
            succ_times = {}
            
            for pred in dag.predecessors(node):
                timestamp = (dag.nodes[pred].get('timestamp') or 
                           getattr(dag.nodes[pred].get('block', None), 'timestamp', 0))
                pred_times[pred] = timestamp
                
            for succ in dag.successors(node):
                timestamp = (dag.nodes[succ].get('timestamp') or 
                           getattr(dag.nodes[succ].get('block', None), 'timestamp', 0))
                succ_times[succ] = timestamp
            
            # Only remove if node has both predecessors and successors or neither
            has_preds = bool(pred_times)
            has_succs = bool(succ_times)
            if (has_preds and has_succs) or (not has_preds and not has_succs):
                # Add edges between valid pairs before removing
                for pred, pred_time in pred_times.items():
                    for succ, succ_time in succ_times.items():
                        if pred_time < succ_time:
                            dag.add_edge(pred, succ)
                
                # Remove the node
                dag.remove_node(node)
                
        except Exception as e:
            logger.error(f"Error in safely removing node {node}: {str(e)}")

    async def _handle_transaction_conflicts(self, dag: nx.DiGraph, node_scores: Dict[str, float]) -> Tuple[nx.DiGraph, int]:
        """
        Handle transaction conflicts during pruning.
        
        Args:
            dag: The DAG to process
            node_scores: Dictionary of node scores
            
        Returns:
            Tuple[nx.DiGraph, int]: (pruned DAG, number of conflicts resolved)
        """
        try:
            working_dag = dag.copy()
            conflicts_resolved = 0
            
            # Map transactions to their containing blocks
            tx_locations = defaultdict(list)
            for node in working_dag.nodes():
                block = working_dag.nodes[node].get('block')
                if block and hasattr(block, 'transactions'):
                    # Handle both single transactions and lists
                    txs = block.transactions if isinstance(block.transactions, list) else [block.transactions]
                    for tx in txs:
                        # Handle both direct hash and transaction object
                        tx_hash = tx.hash if hasattr(tx, 'hash') else str(tx)
                        tx_locations[tx_hash].append(node)
            
            # Track blocks to remove
            blocks_to_remove = set()
            
            # Process each transaction and its locations
            for tx_hash, blocks in tx_locations.items():
                if len(blocks) > 1:  # If transaction appears in multiple blocks
                    conflicts_resolved += 1
                    
                    # Sort blocks by score and timestamp - prioritize higher scores and more recent blocks
                    sorted_blocks = sorted(
                        blocks,
                        key=lambda b: (
                            node_scores.get(b, 0),
                            working_dag.nodes[b].get('timestamp', 0)
                        ),
                        reverse=True
                    )
                    
                    # Keep highest scoring block, mark others for removal
                    blocks_to_remove.update(sorted_blocks[1:])
            
            # Process removals if any conflicts were found
            if blocks_to_remove:
                pruned_dag = working_dag.copy()
                edges_to_add = []
                
                # First collect all edges we need to add
                for block in blocks_to_remove:
                    if block in pruned_dag:
                        # Get predecessor and successor information before removal
                        predecessors = list(pruned_dag.predecessors(block))
                        successors = list(pruned_dag.successors(block))
                        
                        # Store timestamps with nodes
                        pred_times = {p: pruned_dag.nodes[p]['timestamp'] for p in predecessors}
                        succ_times = {s: pruned_dag.nodes[s]['timestamp'] for s in successors}
                        
                        # Collect valid edges to add
                        for pred, pred_time in pred_times.items():
                            for succ, succ_time in succ_times.items():
                                if pred_time < succ_time:
                                    edges_to_add.append((pred, succ))
                        
                        # Remove the conflicting block
                        pruned_dag.remove_node(block)
                
                # Add collected edges to maintain connectivity
                for edge in edges_to_add:
                    if (edge[0] in pruned_dag and edge[1] in pruned_dag and 
                        not pruned_dag.has_edge(edge[0], edge[1])):
                        pruned_dag.add_edge(edge[0], edge[1])
                
                # Ensure the result is still a DAG
                if pruned_dag.number_of_nodes() > 1:
                    if not nx.is_directed_acyclic_graph(pruned_dag):
                        # Remove edges that create cycles
                        while not nx.is_directed_acyclic_graph(pruned_dag):
                            try:
                                cycle = nx.find_cycle(pruned_dag)
                                if cycle:
                                    pruned_dag.remove_edge(*cycle[0])
                            except nx.NetworkXNoCycle:
                                break
                    
                    # Ensure connectivity
                    if not nx.is_weakly_connected(pruned_dag):
                        largest = max(nx.weakly_connected_components(pruned_dag), key=len)
                        pruned_dag = pruned_dag.subgraph(largest).copy()
                
                return pruned_dag, conflicts_resolved
            
            return working_dag, conflicts_resolved

        except Exception as e:
            logger.error(f"Error handling transaction conflicts: {str(e)}")
            logger.error(traceback.format_exc())
            return dag, 0


    async def _verify_timestamp_ordering(self, dag: nx.DiGraph) -> bool:
        """Verify proper timestamp ordering in the DAG"""
        try:
            node_times = nx.get_node_attributes(dag, 'timestamp')
            for edge in dag.edges():
                source_time = node_times[edge[0]]
                target_time = node_times[edge[1]]
                if source_time >= target_time:
                    logger.error(f"Timestamp violation: {edge[0]}({source_time}) -> {edge[1]}({target_time})")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error verifying timestamp ordering: {str(e)}")
            return False
    async def _verify_edge_directions(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Verify and fix edge directions to maintain proper temporal ordering"""
        corrected_dag = dag.copy()
        edges_to_reverse = []
        
        for edge in corrected_dag.edges():
            source_time = corrected_dag.nodes[edge[0]].get('timestamp', 0)
            target_time = corrected_dag.nodes[edge[1]].get('timestamp', 0)
            
            if source_time > target_time:
                edges_to_reverse.append(edge)
        
        for old_edge in edges_to_reverse:
            corrected_dag.remove_edge(old_edge[0], old_edge[1])
            corrected_dag.add_edge(old_edge[1], old_edge[0])
        
        return corrected_dag
        
            
    async def _handle_timestamp_violations(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Handle timestamp violations while preserving connectivity."""
        try:
            # Create new DAG to build corrected version
            corrected_dag = dag.copy()
            
            # Get all nodes in topological order to handle dependencies
            try:
                ordered_nodes = list(nx.topological_sort(dag))
            except nx.NetworkXUnfeasible:
                # If cycle exists, sort by timestamp
                ordered_nodes = sorted(
                    dag.nodes(),
                    key=lambda n: dag.nodes[n].get('timestamp', 0)
                )
            
            current_time = time.time()
            node_times = {}
            edges_to_adjust = []
            
            # First pass: normalize timestamps
            for node in ordered_nodes:
                timestamp = dag.nodes[node].get('timestamp', 0)
                
                # Fix future timestamps
                if timestamp > current_time:
                    timestamp = current_time
                    corrected_dag.nodes[node]['timestamp'] = timestamp
                    if 'block' in corrected_dag.nodes[node]:
                        corrected_dag.nodes[node]['block'].timestamp = timestamp
                
                # Fix very old timestamps
                elif timestamp < current_time - (86400 * 30):  # Older than 30 days
                    timestamp = current_time - 86400  # Set to 1 day ago
                    corrected_dag.nodes[node]['timestamp'] = timestamp
                    if 'block' in corrected_dag.nodes[node]:
                        corrected_dag.nodes[node]['block'].timestamp = timestamp
                
                node_times[node] = timestamp
            
            # Second pass: Fix timestamp ordering violations
            for i in range(len(ordered_nodes)-1):
                current = ordered_nodes[i]
                for next_node in ordered_nodes[i+1:]:
                    if corrected_dag.has_edge(current, next_node):
                        curr_time = node_times[current]
                        next_time = node_times[next_node]
                        
                        if curr_time >= next_time:
                            # Adjust timestamp to maintain ordering
                            new_time = curr_time + 1
                            node_times[next_node] = new_time
                            corrected_dag.nodes[next_node]['timestamp'] = new_time
                            if 'block' in corrected_dag.nodes[next_node]:
                                corrected_dag.nodes[next_node]['block'].timestamp = new_time
            
            # Third pass: Recreate edges while maintaining timestamp ordering
            edges_to_add = []
            edges_to_remove = []
            
            for edge in corrected_dag.edges():
                source_time = node_times[edge[0]]
                target_time = node_times[edge[1]]
                
                if source_time >= target_time:
                    edges_to_remove.append(edge)
                    # Find valid alternative path
                    for node in ordered_nodes:
                        if (node != edge[0] and node != edge[1] and
                            source_time < node_times[node] < target_time):
                            edges_to_add.extend([(edge[0], node), (node, edge[1])])
                            break
            
            # Apply edge changes
            corrected_dag.remove_edges_from(edges_to_remove)
            corrected_dag.add_edges_from(edges_to_add)
            
            # Verify and fix connectivity
            components = list(nx.weakly_connected_components(corrected_dag))
            if len(components) > 1:
                # Connect components while respecting timestamps
                sorted_components = sorted(
                    components,
                    key=lambda c: min(node_times[n] for n in c)
                )
                
                for i in range(len(sorted_components)-1):
                    curr_comp = sorted_components[i]
                    next_comp = sorted_components[i+1]
                    
                    # Find best nodes to connect components
                    curr_node = max(curr_comp, key=lambda n: node_times[n])
                    next_node = min(next_comp, key=lambda n: node_times[n])
                    
                    if node_times[curr_node] < node_times[next_node]:
                        corrected_dag.add_edge(curr_node, next_node)
            
            return corrected_dag
            
        except Exception as e:
            logger.error(f"Error handling timestamp violations: {str(e)}")
            logger.error(traceback.format_exc())
            return dag

    async def prune_dag(self, dag: nx.DiGraph, confirmation_system) -> Tuple[nx.DiGraph, Dict]:
        """Prune the DAG while preserving essential properties."""
        try:
            start_time = time.time()
            initial_size = dag.number_of_nodes()
            working_dag = dag.copy()
            current_time = time.time()
            conflicts_resolved = 0
            genesis_hash = "0" * 64

            if initial_size <= 1:
                return working_dag, self._get_pruning_stats(0, start_time, 0, True)

            # Step 1: Initial size check and aggressive pruning if needed
            if initial_size > self.max_dag_size * 1.5:  # If severely oversized
                # Keep only most recent and highest scored nodes
                all_nodes = sorted(
                    working_dag.nodes(),
                    key=lambda n: (
                        confirmation_system.quantum_scores.get(n, 0.0),
                        working_dag.nodes[n].get('timestamp', 0)
                    ),
                    reverse=True
                )[:self.max_dag_size]
                working_dag = working_dag.subgraph(all_nodes).copy()

            # Step 2: Resolve transaction conflicts
            tx_conflicts = defaultdict(list)
            for node in working_dag.nodes():
                block = working_dag.nodes[node].get('block')
                if block and hasattr(block, 'transactions'):
                    txs = block.transactions if isinstance(block.transactions, list) else [block.transactions]
                    for tx in txs:
                        tx_hash = tx.hash if hasattr(tx, 'hash') else str(tx)
                        tx_conflicts[tx_hash].append(node)

            # Apply conflict resolution
            blocks_to_remove = set()
            for tx_hash, containing_blocks in tx_conflicts.items():
                if len(containing_blocks) > 1:
                    sorted_blocks = sorted(
                        containing_blocks,
                        key=lambda b: (
                            confirmation_system.quantum_scores.get(b, 0),
                            working_dag.nodes[b].get('timestamp', 0)
                        ),
                        reverse=True
                    )
                    blocks_to_remove.update(sorted_blocks[1:])
                    conflicts_resolved += 1

            for block in blocks_to_remove:
                if block in working_dag:
                    preds = list(working_dag.predecessors(block))
                    succs = list(working_dag.successors(block))
                    for pred in preds:
                        for succ in succs:
                            if (working_dag.nodes[pred]['timestamp'] < 
                                working_dag.nodes[succ]['timestamp']):
                                working_dag.add_edge(pred, succ)
                    working_dag.remove_node(block)

            # Step 3: Identify critical components
            genesis_present = genesis_hash in working_dag
            min_required_blocks = set()
            recent_blocks = set()
            secure_blocks = set()
            critical_paths = set()

            # Track critical nodes with scoring
            node_scores = {}
            tx_blocks = {}
            for node in working_dag.nodes():
                timestamp = working_dag.nodes[node].get('timestamp', 0)
                if current_time - timestamp < 3600:
                    recent_blocks.add(node)
                    # Add predecessors of recent blocks to preserve paths
                    critical_paths.update(nx.ancestors(working_dag, node))
                try:
                    security_info = await confirmation_system.get_transaction_security(node, None)
                    if security_info and security_info['security_level'] in ['MAXIMUM', 'VERY_HIGH', 'HIGH']:
                        secure_blocks.add(node)
                        # Add predecessors of secure blocks to preserve paths
                        critical_paths.update(nx.ancestors(working_dag, node))
                except Exception as e:
                    logger.debug(f"Error checking security for node {node}: {str(e)}")

                # Calculate node score
                quantum_score = confirmation_system.quantum_scores.get(node, 0.0)
                time_factor = np.exp(-(current_time - timestamp) / 3600)

                if node == genesis_hash:
                    node_scores[node] = float('inf')
                elif node in recent_blocks or node in secure_blocks:
                    node_scores[node] = 1.0
                else:
                    node_scores[node] = quantum_score * 0.4 + time_factor * 0.6

                # Track transaction confirmations
                if 'transaction' in working_dag.nodes[node]:
                    tx_blocks[node] = set(nx.ancestors(working_dag, node))
                    conf_blocks = sorted(
                        list(tx_blocks[node]),
                        key=lambda n: working_dag.nodes[n].get('timestamp', 0),
                        reverse=True
                    )
                    min_required_blocks.update(conf_blocks[:self.min_confirmations])

            # Step 4: Strict size enforcement while preserving paths
            nodes_to_keep = {genesis_hash} if genesis_hash in working_dag else set()
            nodes_to_keep.update(recent_blocks)
            nodes_to_keep.update(secure_blocks)
            nodes_to_keep.update(min_required_blocks)
            nodes_to_keep.update(critical_paths)

            # If over size limit, keep highest scoring nodes while preserving critical paths
            if len(nodes_to_keep) > self.max_dag_size:
                scored_nodes = sorted(
                    nodes_to_keep,
                    key=lambda n: node_scores.get(n, 0.0),
                    reverse=True
                )
                essential_nodes = {n for n in nodes_to_keep if n in recent_blocks or n in secure_blocks}
                remaining_slots = self.max_dag_size - len(essential_nodes)
                
                # Ensure paths to critical blocks are preserved
                path_nodes = set()
                for critical in essential_nodes:
                    paths = nx.single_source_shortest_path(working_dag, critical)
                    path_nodes.update(*paths.values())
                
                nodes_to_keep = essential_nodes.union(
                    set(n for n in scored_nodes if n in path_nodes)[:remaining_slots]
                )
            else:
                # Fill remaining slots with highest scoring nodes
                remaining_slots = self.max_dag_size - len(nodes_to_keep)
                if remaining_slots > 0:
                    other_nodes = sorted(
                        [n for n in working_dag.nodes() if n not in nodes_to_keep],
                        key=lambda n: node_scores.get(n, 0.0),
                        reverse=True
                    )[:remaining_slots]
                    nodes_to_keep.update(other_nodes)

            # Create pruned DAG with only kept nodes and their predecessors
            pruned_dag = working_dag.subgraph(nodes_to_keep).copy()

            # Step 5: Ensure connectivity
            if pruned_dag.number_of_nodes() > 1:
                components = list(nx.weakly_connected_components(pruned_dag))
                if len(components) > 1:
                    # Keep largest component
                    largest_comp = max(components, key=len)
                    pruned_dag = pruned_dag.subgraph(largest_comp).copy()

                    # If still too large, trim while preserving critical paths
                    if pruned_dag.number_of_nodes() > self.max_dag_size:
                        essential_nodes = {n for n in pruned_dag.nodes() 
                                        if n in recent_blocks or n in secure_blocks}
                        path_nodes = set()
                        for critical in essential_nodes:
                            try:
                                paths = nx.single_source_shortest_path(pruned_dag, critical)
                                path_nodes.update(*paths.values())
                            except nx.NetworkXError:
                                continue
                        
                        nodes = sorted(
                            pruned_dag.nodes(),
                            key=lambda n: (n in path_nodes, node_scores.get(n, 0.0)),
                            reverse=True
                        )[:self.max_dag_size]
                        pruned_dag = pruned_dag.subgraph(nodes).copy()

            nodes_pruned = initial_size - pruned_dag.number_of_nodes()
            return pruned_dag, self._get_pruning_stats(
                nodes_pruned=nodes_pruned,
                start_time=start_time,
                conflicts_resolved=conflicts_resolved,
                preserve_ordering=True
            )

        except Exception as e:
            logger.error(f"Error during DAG pruning: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Emergency fallback with strict size enforcement
            try:
                # Keep only essential nodes up to size limit
                nodes_to_keep = set([genesis_hash] if genesis_present else [])
                nodes_to_keep.update(list(recent_blocks)[:self.max_dag_size//2])
                remaining_slots = self.max_dag_size - len(nodes_to_keep)
                
                if remaining_slots > 0:
                    # Fill remaining slots with highest quantum score nodes
                    other_nodes = sorted(
                        [n for n in working_dag.nodes() if n not in nodes_to_keep],
                        key=lambda n: confirmation_system.quantum_scores.get(n, 0.0),
                        reverse=True
                    )[:remaining_slots]
                    nodes_to_keep.update(other_nodes)
                
                pruned_dag = working_dag.subgraph(nodes_to_keep).copy()
                
                # Ensure final size limit
                if pruned_dag.number_of_nodes() > self.max_dag_size:
                    nodes = list(pruned_dag.nodes())[:self.max_dag_size]
                    pruned_dag = pruned_dag.subgraph(nodes).copy()
                    
                return pruned_dag, self._get_pruning_stats(
                    nodes_pruned=initial_size - pruned_dag.number_of_nodes(),
                    start_time=start_time,
                    conflicts_resolved=conflicts_resolved,
                    preserve_ordering=True
                )
            except Exception as e2:
                logger.error(f"Fallback failed: {str(e2)}")
                # Last resort: return truncated DAG
                nodes = list(working_dag.nodes())[:self.max_dag_size]
                final_dag = working_dag.subgraph(nodes).copy()
                return final_dag, self._get_pruning_stats(
                    nodes_pruned=0,
                    start_time=start_time,
                    conflicts_resolved=conflicts_resolved,
                    preserve_ordering=True
                )

    def _is_critical_node(self, node: str, dag: nx.DiGraph, 
                         critical_blocks: Set[str], confirmation_system) -> bool:
        """Determine if a node is critical and should not be pruned."""
        try:
            # Genesis is always critical
            if node == "0" * 64:
                return True
                
            # Check if part of critical blocks
            if node in critical_blocks:
                return True
                
            # Check quantum score
            if confirmation_system.quantum_scores.get(node, 0.0) >= self.quantum_threshold:
                return True
                
            # Check if needed for recent transaction confirmations
            timestamp = dag.nodes[node].get('timestamp', 0)
            if time.time() - timestamp < 3600:  # Within last hour
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking critical node: {str(e)}")
            return True  # Conservative approach


    def _can_safely_remove(self, dag: nx.DiGraph, node: str) -> bool:
        """Check if a node can be safely removed without disconnecting the DAG."""
        if dag.number_of_nodes() <= 1:
            return False
            
        # If node has no predecessors or successors, it can be safely removed
        if dag.in_degree(node) == 0 or dag.out_degree(node) == 0:
            return True
            
        # Check if removing would disconnect the graph
        test_dag = dag.copy()
        test_dag.remove_node(node)
        
        # If the resulting graph is still connected, removal is safe
        return nx.is_weakly_connected(test_dag)


    def _fallback_prune(self, dag: nx.DiGraph, confirmation_system, start_time: float) -> Tuple[nx.DiGraph, Dict]:
        """Simple fallback pruning strategy."""
        try:
            if dag.number_of_nodes() <= 1:
                return dag.copy(), self._get_pruning_stats(0, start_time)
                
            # Take largest component
            largest_comp = max(nx.weakly_connected_components(dag), key=len)
            pruned_dag = dag.subgraph(largest_comp).copy()
            
            # If still too large, take highest scoring nodes
            if pruned_dag.number_of_nodes() > self.max_dag_size:
                nodes = sorted(
                    pruned_dag.nodes(),
                    key=lambda x: confirmation_system.quantum_scores.get(x, 0.0),
                    reverse=True
                )[:self.max_dag_size]
                pruned_dag = pruned_dag.subgraph(nodes).copy()
                
            return pruned_dag, self._get_pruning_stats(0, start_time)
            
        except Exception as e:
            logger.error(f"Error in fallback pruning: {str(e)}")
            return dag.copy(), self._get_pruning_stats(0, start_time)


    def _is_critical_for_connectivity(self, dag: nx.DiGraph, node: str) -> bool:
        """Check if a node is critical for DAG connectivity."""
        if dag.number_of_nodes() <= 1:
            return True
            
        predecessors = list(dag.predecessors(node))
        successors = list(dag.successors(node))
        
        if not predecessors or not successors:
            return False
            
        # Check if removing would disconnect paths
        test_dag = dag.copy()
        test_dag.remove_node(node)
        
        for pred in predecessors:
            for succ in successors:
                if not nx.has_path(test_dag, pred, succ):
                    return True
                    
        return False



    def _can_safely_remove(self, dag: nx.DiGraph, node: str) -> bool:
        """Check if a node can be safely removed without breaking critical paths."""
        predecessors = list(dag.predecessors(node))
        successors = list(dag.successors(node))
        
        if not predecessors or not successors:
            return True
            
        # Check if alternate paths exist
        test_dag = dag.copy()
        test_dag.remove_node(node)
        
        for pred in predecessors:
            for succ in successors:
                if not nx.has_path(test_dag, pred, succ):
                    return False
        return True

    def _calculate_node_score(self, node: str, dag: nx.DiGraph, 
                            confirmation_system) -> float:
        """Calculate a node's importance score."""
        quantum_score = confirmation_system.quantum_scores.get(node, 0.0)
        time_score = 1.0 - (time.time() - dag.nodes[node]['timestamp']) / 86400
        
        # Calculate connectivity score
        total_nodes = dag.number_of_nodes()
        connectivity = (dag.in_degree(node) + dag.out_degree(node)) / (2 * max(1, total_nodes))
        
        return (quantum_score * 0.6) + (time_score * 0.3) + (connectivity * 0.1)

    def _score_component(self, component: nx.DiGraph, confirmation_system) -> float:
        """Score a component based on quantum coherence and other factors."""
        quantum_scores = [
            confirmation_system.quantum_scores.get(node, 0.0)
            for node in component.nodes()
        ]
        avg_quantum = np.mean(quantum_scores) if quantum_scores else 0.0
        
        # Get latest timestamp in component
        timestamps = [
            component.nodes[n]['timestamp'] 
            for n in component.nodes()
        ]
        latest_time = max(timestamps)
        time_score = 1.0 - (time.time() - latest_time) / 86400
        
        # Size score with diminishing returns
        size_score = np.log1p(component.number_of_nodes())
        
        return (avg_quantum * 0.5) + (time_score * 0.3) + (size_score * 0.2)
    def _get_pruning_stats(self, nodes_pruned: int, start_time: float, conflicts_resolved: int = 0, preserve_ordering: bool = True) -> dict:
        """
        Get pruning statistics.
        
        Args:
            nodes_pruned (int): Number of nodes pruned
            start_time (float): Start time of pruning operation
            conflicts_resolved (int): Number of conflicts resolved
            preserve_ordering (bool): Whether temporal ordering was preserved
            
        Returns:
            dict: Statistics about the pruning operation
        """
        stats = {
            'nodes_pruned': nodes_pruned,
            'pruning_time': time.time() - start_time,
            'last_prune_time': time.time(),
            'conflicts_resolved': conflicts_resolved,
            'ordering_preserved': preserve_ordering
        }
        self.pruning_stats.update(stats)
        return stats






    async def _calculate_all_node_scores(
        self, dag: nx.DiGraph, confirmation_system
    ) -> Dict[str, float]:
        """Calculate importance scores for all nodes."""
        node_scores = {}
        for node in dag.nodes():
            score = await self._calculate_node_score(node, dag, confirmation_system)
            node_scores[node] = score
        return node_scores
    async def _identify_minimum_critical_set(
        self, dag: nx.DiGraph, node_scores: Dict[str, float], confirmation_system
    ) -> Set[str]:
        """Identify minimum set of critical nodes that must be kept."""
        try:
            critical_set = set()
            
            # Add nodes with high quantum scores
            for node, score in node_scores.items():
                if (confirmation_system.quantum_scores.get(node, 0.0) >= 
                    self.quantum_threshold):
                    critical_set.add(node)
            
            # Add recent nodes
            current_time = time.time()
            for node in dag.nodes():
                if dag.nodes[node]['timestamp'] > current_time - 3600:  # Last hour
                    critical_set.add(node)
            
            # Add nodes needed for security
            for node in dag.nodes():
                security_info = await confirmation_system.get_transaction_security(node, None)
                if security_info and security_info['security_level'] in ['HIGH', 'MAXIMUM']:
                    critical_set.add(node)
            
            # Add necessary connecting nodes
            if len(critical_set) >= 2:
                connecting_nodes = await self._find_minimum_connecting_nodes(
                    dag, critical_set)
                critical_set.update(connecting_nodes)
            
            return critical_set

        except Exception as e:
            logger.error(f"Error identifying minimum critical set: {str(e)}")
            return set()

    async def _find_minimum_connecting_nodes(
        self, dag: nx.DiGraph, critical_nodes: Set[str]
    ) -> Set[str]:
        """Find minimum set of nodes needed to connect critical nodes."""
        try:
            connecting_nodes = set()
            
            # Sort critical nodes by timestamp
            sorted_critical = sorted(
                critical_nodes,
                key=lambda x: dag.nodes[x]['timestamp']
            )
            
            # Find paths between consecutive critical nodes
            for i in range(len(sorted_critical)-1):
                source = sorted_critical[i]
                target = sorted_critical[i+1]
                
                if nx.has_path(dag, source, target):
                    # Find shortest path
                    path = nx.shortest_path(dag, source, target)
                    connecting_nodes.update(path[1:-1])  # Exclude source and target
            
            return connecting_nodes

        except Exception as e:
            logger.error(f"Error finding connecting nodes: {str(e)}")
            return set()

    async def _ensure_dag_validity(
        self, dag: nx.DiGraph, confirmation_system
    ) -> nx.DiGraph:
        """Ensure DAG maintains required properties."""
        try:
            # Ensure temporal ordering
            node_times = nx.get_node_attributes(dag, 'timestamp')
            edges_to_remove = []
            for edge in dag.edges():
                if node_times[edge[0]] >= node_times[edge[1]]:
                    edges_to_remove.append(edge)
            dag.remove_edges_from(edges_to_remove)
            
            # Ensure connectivity
            components = list(nx.weakly_connected_components(dag))
            if len(components) > 1:
                # Keep largest/most important component
                main_component = max(
                    components,
                    key=lambda c: sum(confirmation_system.quantum_scores.get(n, 0)
                                    for n in c)
                )
                dag = dag.subgraph(main_component).copy()
            
            # Ensure acyclicity
            while not nx.is_directed_acyclic_graph(dag):
                cycles = list(nx.simple_cycles(dag))
                if not cycles:
                    break
                # Remove edge from least important node in cycle
                cycle = cycles[0]
                min_node = min(
                    cycle,
                    key=lambda n: confirmation_system.quantum_scores.get(n, 0)
                )
                cycle_idx = cycle.index(min_node)
                next_node = cycle[(cycle_idx + 1) % len(cycle)]
                dag.remove_edge(min_node, next_node)
            
            return dag

        except Exception as e:
            logger.error(f"Error ensuring DAG validity: {str(e)}")
            return dag

    async def _handle_disconnected_components(
        self, dag: nx.DiGraph, confirmation_system
    ) -> nx.DiGraph:
        """Handle disconnected components by selecting or merging them."""
        try:
            components = list(nx.weakly_connected_components(dag))
            if len(components) <= 1:
                return dag

            logger.info(f"Found {len(components)} disconnected components")
            
            # Score each component
            component_scores = []
            for comp in components:
                subgraph = dag.subgraph(comp)
                score = await self._calculate_component_score(subgraph, confirmation_system)
                component_scores.append((comp, score))
            
            # Sort components by score
            component_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create new DAG with merged components
            merged_dag = nx.DiGraph()
            
            # Start with the highest scored component
            primary_comp = component_scores[0][0]
            primary_subgraph = dag.subgraph(primary_comp)
            merged_dag.add_nodes_from(primary_subgraph.nodes(data=True))
            merged_dag.add_edges_from(primary_subgraph.edges())
            
            # Try to merge other components
            for comp, score in component_scores[1:]:
                if score > 0.5:  # Only merge significant components
                    subgraph = dag.subgraph(comp)
                    
                    # Find potential connection points
                    primary_times = {
                        n: merged_dag.nodes[n]['timestamp'] 
                        for n in merged_dag.nodes()
                    }
                    comp_times = {
                        n: subgraph.nodes[n]['timestamp'] 
                        for n in subgraph.nodes()
                    }
                    
                    # Find nodes with compatible timestamps
                    for comp_node, comp_time in comp_times.items():
                        # Find nodes in primary component with earlier timestamps
                        valid_targets = [
                            n for n, t in primary_times.items()
                            if t < comp_time
                        ]
                        
                        if valid_targets:
                            # Add component nodes
                            merged_dag.add_nodes_from(subgraph.nodes(data=True))
                            merged_dag.add_edges_from(subgraph.edges())
                            
                            # Add connection to primary component
                            best_target = max(
                                valid_targets,
                                key=lambda x: primary_times[x]
                            )
                            merged_dag.add_edge(best_target, comp_node)
                            break
            
            # Verify the merged DAG
            if not nx.is_directed_acyclic_graph(merged_dag):
                logger.error("Merged DAG contains cycles")
                # Fall back to highest scored component
                return dag.subgraph(component_scores[0][0]).copy()
                
            if not nx.is_weakly_connected(merged_dag):
                logger.error("Merged DAG is not connected")
                # Fall back to highest scored component
                return dag.subgraph(component_scores[0][0]).copy()
                
            return merged_dag

        except Exception as e:
            logger.error(f"Error handling disconnected components: {str(e)}")
            logger.error(traceback.format_exc())
            # Return the largest component as fallback
            largest_comp = max(nx.weakly_connected_components(dag), key=len)
            return dag.subgraph(largest_comp).copy()


    async def _calculate_component_score(
        self, component: nx.DiGraph, confirmation_system
    ) -> float:
        """Calculate importance score for a component."""
        try:
            score_factors = []
            
            # 1. Size factor
            size_factor = min(component.number_of_nodes() / 10, 1.0)
            score_factors.append(size_factor)
            
            # 2. Quantum signature strength
            quantum_scores = [
                confirmation_system.quantum_scores.get(node, 0.0)
                for node in component.nodes()
            ]
            quantum_factor = sum(quantum_scores) / len(quantum_scores)
            score_factors.append(quantum_factor)
            
            # 3. Recency factor
            timestamps = [
                component.nodes[n]['timestamp']
                for n in component.nodes()
            ]
            latest_time = max(timestamps)
            time_factor = np.exp(-(time.time() - latest_time) / 3600)
            score_factors.append(time_factor)
            
            # 4. Confirmation factor
            confirmations = []
            for node in component.nodes():
                security_info = await confirmation_system.get_transaction_security(node, None)
                if security_info:
                    conf_score = security_info.get('confirmation_score', 0.0)
                    confirmations.append(conf_score)
            
            if confirmations:
                conf_factor = sum(confirmations) / len(confirmations)
                score_factors.append(conf_factor)
            
            # Combine factors with weights
            weights = [0.3, 0.3, 0.2, 0.2]
            final_score = sum(f * w for f, w in zip(score_factors, weights))
            
            return final_score

        except Exception as e:
            logger.error(f"Error calculating component score: {str(e)}")
            return 0.0

                
    async def _fix_timestamp_ordering(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Fix timestamp ordering in the DAG to ensure temporal consistency"""
        try:
            # Create a new DAG for the fixed version
            fixed_dag = nx.DiGraph()
            current_time = time.time()
            
            # Get all nodes sorted by their intended order
            # Use edge relationships to determine order
            try:
                ordered_nodes = list(nx.topological_sort(dag))
            except nx.NetworkXUnfeasible:
                # If there's a cycle, fall back to simple sorting
                ordered_nodes = sorted(dag.nodes())

            # First pass: normalize timestamps
            node_times = {}
            base_time = current_time - (len(ordered_nodes) * 3600)  # 1 hour gaps
            
            for i, node in enumerate(ordered_nodes):
                # Calculate new timestamp with proper spacing
                new_time = base_time + (i * 3600)  # 1 hour between blocks
                
                # Add node with corrected timestamp
                fixed_dag.add_node(
                    node,
                    timestamp=new_time,
                    block=dag.nodes[node]['block']
                )
                
                # Update block timestamp
                fixed_dag.nodes[node]['block'].timestamp = new_time
                node_times[node] = new_time

            # Second pass: add edges while ensuring timestamp ordering
            for node in ordered_nodes:
                successors = list(dag.successors(node))
                for succ in successors:
                    if node_times[node] < node_times[succ]:
                        fixed_dag.add_edge(node, succ)

            # Verify the resulting DAG
            for edge in fixed_dag.edges():
                source_time = fixed_dag.nodes[edge[0]]['timestamp']
                target_time = fixed_dag.nodes[edge[1]]['timestamp']
                assert source_time < target_time, f"Timestamp violation: {edge[0]} -> {edge[1]}"

            return fixed_dag

        except Exception as e:
            logger.error(f"Error fixing timestamp ordering: {str(e)}")
            logger.error(traceback.format_exc())
            return dag


    async def _fix_timestamp_violations(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Fix timestamp violations in the DAG."""
        try:
            # Get all nodes and their timestamps
            node_times = nx.get_node_attributes(dag, 'timestamp')
            current_time = time.time()
            
            # First pass: fix obviously invalid timestamps
            for node in dag.nodes():
                timestamp = node_times[node]
                
                # Fix future timestamps
                if timestamp > current_time:
                    logger.warning(f"Found future timestamp in node {node}")
                    new_timestamp = current_time
                    dag.nodes[node]['timestamp'] = new_timestamp
                    dag.nodes[node]['block'].timestamp = new_timestamp
                    node_times[node] = new_timestamp
                
                # Fix very old timestamps
                elif timestamp < current_time - (86400 * 30):  # Older than 30 days
                    logger.warning(f"Found very old timestamp in node {node}")
                    new_timestamp = current_time - 86400  # Set to 1 day ago
                    dag.nodes[node]['timestamp'] = new_timestamp
                    dag.nodes[node]['block'].timestamp = new_timestamp
                    node_times[node] = new_timestamp

            # Second pass: fix timestamp ordering violations
            topo_sorted = list(nx.topological_sort(dag))
            fixed_dag = nx.DiGraph()
            
            # Add nodes with corrected timestamps
            for node in topo_sorted:
                fixed_dag.add_node(
                    node,
                    **dag.nodes[node]
                )
            
            # Add edges while ensuring timestamp ordering
            for node in topo_sorted:
                successors = list(dag.successors(node))
                for succ in successors:
                    # Only add edge if timestamps are properly ordered
                    if node_times[node] < node_times[succ]:
                        fixed_dag.add_edge(node, succ)
                    else:
                        # Fix timestamp violation
                        new_timestamp = node_times[node] + 1
                        fixed_dag.nodes[succ]['timestamp'] = new_timestamp
                        fixed_dag.nodes[succ]['block'].timestamp = new_timestamp
                        node_times[succ] = new_timestamp
                        fixed_dag.add_edge(node, succ)

            return fixed_dag

        except Exception as e:
            logger.error(f"Error fixing timestamp violations: {str(e)}")
            logger.error(traceback.format_exc())
            return dag

    async def _perform_safe_pruning(self, dag: nx.DiGraph, prunable_nodes: List[str],
                                  confirmation_system) -> nx.DiGraph:
        """Perform pruning while maintaining DAG properties."""
        try:
            pruned_dag = dag.copy()
            current_size = pruned_dag.number_of_nodes()
            genesis_hash = "0" * 64
            
            # Sort prunable nodes by importance (least important first)
            sorted_nodes = sorted(
                prunable_nodes,
                key=lambda n: (
                    confirmation_system.quantum_scores.get(n, 0.0),
                    pruned_dag.nodes[n].get('timestamp', 0)
                )
            )
            
            for node in sorted_nodes:
                if current_size <= self.max_dag_size:
                    break
                    
                # Skip genesis node
                if node == genesis_hash:
                    continue
                    
                # Only remove if it won't create isolated nodes
                test_dag = pruned_dag.copy()
                test_dag.remove_node(node)
                
                if (nx.is_directed_acyclic_graph(test_dag) and 
                    (test_dag.number_of_nodes() <= 1 or nx.is_weakly_connected(test_dag))):
                    
                    # Ensure genesis block stays connected if it exists
                    if (genesis_hash not in test_dag or 
                        (genesis_hash in test_dag and test_dag.out_degree(genesis_hash) > 0)):
                        self._safely_remove_node(pruned_dag, node)
                        current_size -= 1

            return pruned_dag

        except Exception as e:
            logger.error(f"Error performing safe pruning: {str(e)}")
            return dag

    async def _verify_dag_integrity(self, dag: nx.DiGraph, confirmation_system) -> bool:
        """Verify DAG integrity after pruning."""
        try:
            # 1. Check if it's still a DAG
            if not nx.is_directed_acyclic_graph(dag):
                logger.error("Pruned graph is not a DAG")
                return False

            # 2. Check connectivity
            if dag.number_of_nodes() > 1 and not nx.is_weakly_connected(dag):
                logger.error("Pruned DAG is not connected")
                return False

            # 3. Check genesis connectivity if present
            genesis_hash = "0" * 64
            if (genesis_hash in dag and 
                dag.number_of_nodes() > 1 and 
                dag.out_degree(genesis_hash) == 0):
                logger.error("Genesis block is isolated")
                return False

            # 4. Verify quantum state preservation
            quantum_scores = [
                confirmation_system.quantum_scores.get(node, 0.0)
                for node in dag.nodes()
            ]
            if quantum_scores and min(quantum_scores) < self.quantum_threshold:
                logger.error("Quantum state compromised after pruning")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying DAG integrity: {str(e)}")
            return False

    async def _verify_pruned_dag(self, dag: nx.DiGraph, confirmation_system) -> bool:
        """Verify the integrity of the pruned DAG."""
        try:
            # 1. Verify basic DAG properties
            if not nx.is_directed_acyclic_graph(dag):
                logger.error("Pruned graph is not a DAG")
                return False

            if not nx.is_weakly_connected(dag):
                logger.error("Pruned DAG is not connected")
                return False

            # 2. Verify timestamp ordering
            for edge in dag.edges():
                source_time = dag.nodes[edge[0]]['timestamp']
                target_time = dag.nodes[edge[1]]['timestamp']
                if source_time >= target_time:
                    logger.error(f"Timestamp violation: {edge[0]}({source_time}) -> {edge[1]}({target_time})")
                    return False

            # 3. Verify quantum state preservation
            quantum_scores = [
                confirmation_system.quantum_scores.get(node, 0.0)
                for node in dag.nodes()
            ]
            if min(quantum_scores) < self.quantum_threshold:
                logger.error("Quantum state compromised after pruning")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying pruned DAG: {str(e)}")
            return False




    async def _identify_critical_nodes(self, dag: nx.DiGraph, confirmation_system) -> Set[str]:
        """Identify nodes that cannot be pruned."""
        try:
            critical_nodes = set()
            current_time = time.time()

            for node in dag.nodes():
                # Check if node is recent (last hour)
                if dag.nodes[node]['timestamp'] > current_time - 3600:
                    critical_nodes.add(node)
                    continue

                # Check quantum score
                quantum_score = confirmation_system.quantum_scores.get(node, 0.0)
                if quantum_score >= self.quantum_threshold:
                    critical_nodes.add(node)
                    continue

                # Check security level
                security_info = await confirmation_system.get_transaction_security(node, None)
                if security_info and self._is_security_critical(security_info):
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
                                     confirmation_system) -> Set[str]:
        """Identify nodes that can potentially be pruned."""
        try:
            prunable_nodes = set()
            
            for node in dag.nodes():
                if node in critical_nodes:
                    continue

                security_info = await confirmation_system.get_transaction_security(node, None)
                if security_info and security_info['num_confirmations'] >= self.min_confirmations:
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

    async def _is_safe_to_remove(self, dag: nx.DiGraph, node: str, 
                                confirmation_system) -> bool:
        """Check if it's safe to remove a node."""
        try:
            # Get node's neighbors
            predecessors = list(dag.predecessors(node))
            successors = list(dag.successors(node))

            if not predecessors or not successors:
                return True

            # Create a copy of the DAG and remove the node
            test_dag = dag.copy()
            test_dag.remove_node(node)

            # Check if removal maintains connectivity
            for pred in predecessors:
                for succ in successors:
                    if not nx.has_path(test_dag, pred, succ):
                        return False

            return True

        except Exception as e:
            logger.error(f"Error checking node removal safety: {str(e)}")
            return False


    def _is_security_critical(self, security_info: Dict) -> bool:
        """Check if a node has critical security level."""
        try:
            level = security_info.get('security_level', 'UNSAFE')
            return self.security_levels.get(level, 0) >= self.security_levels.get(self.min_security_level, 0)
        except Exception as e:
            logger.error(f"Error checking security level: {str(e)}")
            return False

   

    def _sort_prunable_nodes(self, nodes: Set[str], confirmation_system) -> List[str]:
        """Sort prunable nodes by importance (least important first)."""
        try:
            node_scores = []
            for node in nodes:
                score = confirmation_system.quantum_scores.get(node, 0.0)
                node_scores.append((node, score))
            
            return [node for node, _ in sorted(node_scores, key=lambda x: x[1])]
            
        except Exception as e:
            logger.error(f"Error sorting prunable nodes: {str(e)}")
            return list(nodes)
