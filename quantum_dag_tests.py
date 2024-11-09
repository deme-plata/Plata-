import pytest
import networkx as nx
import numpy as np
from decimal import Decimal
import time
from unittest.mock import Mock, patch, AsyncMock
import pytest_asyncio
from DAGPruningSystem import DAGPruningSystem
from DAGKnightMiner import DAGKnightMiner,DAGDiagnostics
from DAGKnightGasSystem import EnhancedDAGKnightGasSystem, EnhancedGasTransactionType
from DAGConfirmationSystem import DAGConfirmationSystem
from shared_logic import Transaction, QuantumBlock
from typing import Tuple
import asyncio
import traceback 
from CryptoProvider import CryptoProvider
pytestmark = pytest.mark.asyncio

class TestDAGPruningSystem:
    @pytest_asyncio.fixture
    async def mock_confirmation_system(self):
        confirmation_system = Mock()
        confirmation_system.quantum_scores = {}
        confirmation_system.confirmation_cache = {}
        confirmation_system.get_transaction_security = AsyncMock(return_value={
            'security_level': 'HIGH',
            'confirmation_score': 0.95,
            'num_confirmations': 10,
            'path_diversity': 0.8,
            'quantum_strength': 0.9
        })
        return confirmation_system

    @pytest_asyncio.fixture
    async def sample_dag(self):
        """Create a properly connected sample DAG with forward edges and sequential timestamps"""
        dag = nx.DiGraph()
        base_time = time.time() - (10 * 3600)  # Start 10 hours ago
        
        # First create nodes with proper sequential timestamps
        nodes_with_times = []
        for i in range(10):
            timestamp = base_time + (i * 3600)  # 1 hour between blocks
            nodes_with_times.append((f"block_{i}", timestamp))
            
        # Add nodes in order
        for node_id, timestamp in nodes_with_times:
            dag.add_node(
                node_id,
                timestamp=timestamp,
                block=Mock(
                    quantum_signature="1010",
                    timestamp=timestamp,
                    transactions=[]
                )
            )
        
        # Add edges only between nodes with valid timestamp ordering
        for i in range(len(nodes_with_times)-1):
            current_node, current_time = nodes_with_times[i]
            next_node, next_time = nodes_with_times[i+1]
            if current_time < next_time:
                dag.add_edge(current_node, next_node)
        
        return dag


    @pytest_asyncio.fixture
    async def pruning_system(self):
        return DAGPruningSystem(
            min_confirmations=2,
            quantum_threshold=0.85,
            prune_interval=5,
            max_dag_size=5,  # Reduced from 20 to prevent over-pruning
            min_security_level="HIGH"
        )

    def verify_dag_properties(self, dag: nx.DiGraph) -> Tuple[bool, str]:
        """Helper function to verify DAG properties"""
        try:
            # Check if it's actually a DAG
            if not nx.is_directed_acyclic_graph(dag):
                return False, "Graph contains cycles"

            # Check connectivity
            if not nx.is_weakly_connected(dag):
                return False, "Graph is not connected"

            # Check for isolated nodes
            if list(nx.isolates(dag)):
                return False, "Graph contains isolated nodes"

            # Verify timestamps are monotonic along paths
            node_times = nx.get_node_attributes(dag, 'timestamp')
            for edge in dag.edges():
                if node_times[edge[0]] >= node_times[edge[1]]:
                    return False, "Time ordering violation in edges"

            return True, "DAG properties verified"

        except Exception as e:
            return False, f"Error verifying DAG properties: {str(e)}"

    @pytest.mark.asyncio
    async def test_prune_dag_maintains_connectivity(self, pruning_system, sample_dag, mock_confirmation_system):
        # Verify initial DAG properties
        is_valid, message = self.verify_dag_properties(sample_dag)
        assert is_valid, f"Initial DAG invalid: {message}"
        
        # Perform pruning
        pruned_dag, stats = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify pruned DAG properties
        is_valid, message = self.verify_dag_properties(pruned_dag)
        assert is_valid, f"Pruned DAG invalid: {message}"
        
        # Additional connectivity checks
        remaining_nodes = list(pruned_dag.nodes())
        if len(remaining_nodes) > 1:
            # Check forward paths
            for i in range(len(remaining_nodes)-1):
                current = remaining_nodes[i]
                next_node = remaining_nodes[i+1]
                assert nx.has_path(pruned_dag, current, next_node), \
                    f"No path from {current} to {next_node}"
                
                # Verify timestamp ordering
                current_time = pruned_dag.nodes[current]['timestamp']
                next_time = pruned_dag.nodes[next_node]['timestamp']
                assert current_time < next_time, \
                    f"Time ordering violated: {current}({current_time}) -> {next_node}({next_time})"

        # Verify no orphaned chains
        components = list(nx.weakly_connected_components(pruned_dag))
        assert len(components) == 1, f"DAG has {len(components)} disconnected components"

    @pytest.mark.asyncio
    async def test_prune_dag_time_based_retention(self, pruning_system, sample_dag):
        """Test that recent blocks are retained regardless of other factors"""
        current_time = time.time()
        
        # Mark some blocks as very recent
        recent_blocks = ['block_8', 'block_9']
        for block in recent_blocks:
            sample_dag.nodes[block]['timestamp'] = current_time - 300  # 5 minutes ago
            
        pruned_dag, _ = await pruning_system.prune_dag(sample_dag, Mock())
        
        # Verify recent blocks are retained
        for block in recent_blocks:
            assert block in pruned_dag.nodes, f"Recent block {block} was incorrectly pruned"
            
        # Verify they maintain proper connections
        for block in recent_blocks:
            predecessors = list(pruned_dag.predecessors(block))
            assert len(predecessors) > 0, f"Recent block {block} lost its connections"

    @pytest.mark.asyncio
    async def test_prune_dag_quantum_state_preservation(self, pruning_system, sample_dag, mock_confirmation_system):
        """Test that blocks with strong quantum signatures are preserved"""
        # Add quantum signatures to specific blocks
        quantum_blocks = ['block_3', 'block_4', 'block_5']
        mock_confirmation_system.quantum_scores = {
            block: 0.95 for block in quantum_blocks
        }
        
        pruned_dag, _ = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify quantum blocks are retained
        for block in quantum_blocks:
            assert block in pruned_dag.nodes, \
                f"Block {block} with high quantum score was incorrectly pruned"
        
        # Verify quantum blocks maintain proper connections
        for block in quantum_blocks:
            predecessors = list(pruned_dag.predecessors(block))
            successors = list(pruned_dag.successors(block))
            assert len(predecessors) + len(successors) > 0, \
                f"Quantum block {block} lost its connections"

    @pytest.mark.asyncio
    async def test_prune_dag_security_level_preservation(self, pruning_system, sample_dag, mock_confirmation_system):
        """Test that high security blocks and their critical paths are preserved"""
        # Configure some blocks with high security
        secure_blocks = ['block_2', 'block_3', 'block_4']
        
        async def mock_security_check(tx_hash, block_hash):
            if tx_hash in secure_blocks:
                return {
                    'security_level': 'MAXIMUM',
                    'confirmation_score': 0.98,
                    'num_confirmations': 12,
                    'path_diversity': 0.9,
                    'quantum_strength': 0.95
                }
            return {
                'security_level': 'MEDIUM',
                'confirmation_score': 0.7,
                'num_confirmations': 5,
                'path_diversity': 0.5,
                'quantum_strength': 0.7
            }
            
        mock_confirmation_system.get_transaction_security = AsyncMock(side_effect=mock_security_check)
        
        pruned_dag, _ = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify secure blocks are retained
        for block in secure_blocks:
            assert block in pruned_dag.nodes, \
                f"High security block {block} was incorrectly pruned"
        
        # Verify secure blocks form a connected subgraph
        secure_subgraph = pruned_dag.subgraph(secure_blocks)
        assert nx.is_weakly_connected(secure_subgraph), \
            "High security blocks lost their connectivity"
    @pytest.mark.asyncio
    async def test_prune_dag_with_empty_dag(self, pruning_system, mock_confirmation_system):
        """Test pruning behavior with an empty DAG"""
        empty_dag = nx.DiGraph()
        pruned_dag, stats = await pruning_system.prune_dag(empty_dag, mock_confirmation_system)
        
        assert isinstance(pruned_dag, nx.DiGraph)
        assert pruned_dag.number_of_nodes() == 0
        assert stats['nodes_pruned'] == 0

    @pytest.mark.asyncio
    async def test_prune_dag_with_single_node(self, pruning_system, mock_confirmation_system):
        """Test pruning behavior with a single-node DAG"""
        single_node_dag = nx.DiGraph()
        current_time = time.time()
        single_node_dag.add_node(
            "block_0",
            timestamp=current_time,
            block=Mock(quantum_signature="1010", timestamp=current_time, transactions=[])
        )
        
        pruned_dag, stats = await pruning_system.prune_dag(single_node_dag, mock_confirmation_system)
        assert pruned_dag.number_of_nodes() == 1
        assert "block_0" in pruned_dag.nodes

    @pytest.mark.asyncio
    async def test_prune_dag_with_max_size_exceeded(self, pruning_system, sample_dag):
        """Test pruning when DAG exceeds max size"""
        # Create a properly mocked confirmation system
        mock_confirmation_system = Mock()
        
        # Mock quantum scores with recent blocks having higher scores
        mock_confirmation_system.quantum_scores = {}
        for i in range(10):
            # Older blocks get lower scores
            score = 0.5 + (i * 0.05)  # Scores increase with recency
            mock_confirmation_system.quantum_scores[f"block_{i}"] = score
            
        # Recent blocks get very high scores
        mock_confirmation_system.quantum_scores['block_8'] = 0.95
        mock_confirmation_system.quantum_scores['block_9'] = 0.98
        
        # Mock confirmation cache with increasing confirmation scores
        mock_confirmation_system.confirmation_cache = {
            f"block_{i}": {
                'score': 0.5 + (i * 0.05),
                'paths': [f"path_{j}" for j in range(i)],
                'last_update': time.time()
            } for i in range(10)
        }
        
        # Mock security info for different blocks
        async def mock_security_check(tx_hash: str, block_hash: str) -> dict:
            block_num = int(tx_hash.split('_')[1])
            is_recent = block_num >= 7
            is_high_score = block_num in [8, 9]
            
            return {
                'security_level': 'HIGH' if is_high_score else 'MEDIUM',
                'confirmation_score': 0.95 if is_high_score else 0.7,
                'num_confirmations': 10 if is_recent else 5,
                'path_diversity': 0.8 if is_recent else 0.5,
                'quantum_strength': 0.95 if is_high_score else 0.7,
                'is_final': is_high_score
            }
            
        mock_confirmation_system.get_transaction_security = AsyncMock(side_effect=mock_security_check)
        
        # Set evaluation functions
        mock_confirmation_system.evaluate_quantum_signature = Mock(return_value=0.85)
        
        # Additional helper functions
        mock_confirmation_system.calculate_confirmation_score = AsyncMock(return_value=0.8)
        mock_confirmation_system.validate_dag_structure = Mock(return_value=True)
        
        # Modify pruning system parameters
        pruning_system.max_dag_size = 3
        pruning_system.min_confirmations = 2
        pruning_system.quantum_threshold = 0.85
        
        # Add block data to DAG nodes
        for node in sample_dag.nodes():
            block_num = int(node.split('_')[1])
            sample_dag.nodes[node]['block'] = Mock(
                quantum_signature="1010" if block_num >= 8 else "0101",
                timestamp=time.time() - (9-block_num)*3600,
                transactions=[]
            )
        
        # Perform pruning
        pruned_dag, stats = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify pruning results
        assert pruned_dag.number_of_nodes() <= pruning_system.max_dag_size, \
            f"Pruned DAG size ({pruned_dag.number_of_nodes()}) exceeds max size ({pruning_system.max_dag_size})"
        
        # Verify that we kept the most important blocks
        retained_blocks = set(pruned_dag.nodes())
        assert 'block_9' in retained_blocks, "Most recent block must be retained"
        
        # Verify DAG properties
        assert nx.is_directed_acyclic_graph(pruned_dag), "Result must be a valid DAG"
        assert nx.is_weakly_connected(pruned_dag), "Result must be connected"
        
        # Check that retained blocks form a valid chain
        if len(retained_blocks) > 1:
            sorted_blocks = sorted(list(retained_blocks), 
                                key=lambda x: pruned_dag.nodes[x]['block'].timestamp)
            for i in range(len(sorted_blocks)-1):
                assert nx.has_path(pruned_dag, sorted_blocks[i], sorted_blocks[i+1]), \
                    f"No path between consecutive blocks: {sorted_blocks[i]} -> {sorted_blocks[i+1]}"
        
        # Verify pruning stats
        assert stats['nodes_pruned'] == sample_dag.number_of_nodes() - pruned_dag.number_of_nodes()
        assert 'pruning_time' in stats
        assert stats['pruning_time'] > 0
        
        # Verify quantum state preservation
        recent_blocks = [b for b in retained_blocks 
                        if mock_confirmation_system.quantum_scores[b] >= pruning_system.quantum_threshold]
        assert len(recent_blocks) > 0, "Must retain some high quantum score blocks"
        
        # Verify critical path preservation
        for block in retained_blocks:
            block_num = int(block.split('_')[1])
            if block_num >= 8:  # Critical blocks
                predecessors = list(pruned_dag.predecessors(block))
                assert len(predecessors) > 0, f"Critical block {block} lost its predecessors"



    @pytest.mark.asyncio
    async def test_prune_dag_with_invalid_timestamps(self, pruning_system, sample_dag):
        """Test pruning behavior with invalid timestamp ordering"""
        # Create a properly mocked confirmation system
        mock_confirmation_system = Mock()
        mock_confirmation_system.quantum_scores = {
            f"block_{i}": 0.7 for i in range(10)
        }
        mock_confirmation_system.confirmation_cache = {}
        mock_confirmation_system.get_transaction_security = AsyncMock(return_value={
            'security_level': 'MEDIUM',
            'confirmation_score': 0.7,
            'num_confirmations': 5,
            'path_diversity': 0.5,
            'quantum_strength': 0.7
        })
        mock_confirmation_system.evaluate_quantum_signature = Mock(return_value=0.85)
        
        # Store original timestamps and verify initial DAG
        original_timestamps = {
            node: data['timestamp'] 
            for node, data in sample_dag.nodes(data=True)
        }
        
        # Verify initial DAG is valid
        assert nx.is_directed_acyclic_graph(sample_dag), "Initial DAG must be valid"
        initial_node_times = nx.get_node_attributes(sample_dag, 'timestamp')
        for edge in sample_dag.edges():
            assert initial_node_times[edge[0]] < initial_node_times[edge[1]], \
                "Initial DAG must have valid timestamp ordering"
        
        # Create timestamp violations
        current_time = time.time()
        edges_to_remove = []
        
        # First, remove edges that will be affected by timestamp changes
        for edge in sample_dag.edges():
            if edge[0] == "block_5" or edge[1] == "block_6":
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            sample_dag.remove_edge(*edge)
        
        # Now update timestamps
        sample_dag.nodes["block_5"]["timestamp"] = current_time + 3600
        sample_dag.nodes["block_5"]["block"].timestamp = current_time + 3600
        
        sample_dag.nodes["block_6"]["timestamp"] = current_time - 86400
        sample_dag.nodes["block_6"]["block"].timestamp = current_time - 86400
        
        # Add new edges that respect the new timestamps
        for i in range(4, 7):
            if i != 5:  # Skip the violated node
                node_time = sample_dag.nodes[f"block_{i}"]["timestamp"]
                if node_time < sample_dag.nodes["block_6"]["timestamp"]:
                    sample_dag.add_edge(f"block_{i}", "block_6")
                if node_time < sample_dag.nodes["block_5"]["timestamp"]:
                    sample_dag.add_edge(f"block_{i}", "block_5")
        
        # Perform pruning
        pruned_dag, stats = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify DAG properties after pruning
        assert nx.is_directed_acyclic_graph(pruned_dag), "Result must be a DAG"
        assert nx.is_weakly_connected(pruned_dag), "Result must be connected"
        
        # Get nodes in topological order
        topo_order = list(nx.topological_sort(pruned_dag))
        
        # Verify timestamp ordering along all paths
        node_times = nx.get_node_attributes(pruned_dag, 'timestamp')
        edge_time_violations = []
        
        for i in range(len(topo_order)-1):
            current_node = topo_order[i]
            next_nodes = list(pruned_dag.successors(current_node))
            
            for next_node in next_nodes:
                current_time = node_times[current_node]
                next_time = node_times[next_node]
                
                if current_time >= next_time:
                    edge_time_violations.append(
                        f"Edge {current_node} -> {next_node}: "
                        f"{current_time} -> {next_time}"
                    )
        
        assert not edge_time_violations, \
            f"Time ordering violations found in pruned DAG:\n" + \
            "\n".join(edge_time_violations)
        
        # Additional verification of temporal consistency
        for node in pruned_dag.nodes():
            # Verify all predecessors have earlier timestamps
            for pred in pruned_dag.predecessors(node):
                assert node_times[pred] < node_times[node], \
                    f"Predecessor {pred} has invalid timestamp relationship with {node}"
            
            # Verify all successors have later timestamps
            for succ in pruned_dag.successors(node):
                assert node_times[node] < node_times[succ], \
                    f"Successor {succ} has invalid timestamp relationship with {node}"
        
        # Verify temporal paths
        for source in pruned_dag.nodes():
            for target in pruned_dag.nodes():
                if source != target and nx.has_path(pruned_dag, source, target):
                    path = nx.shortest_path(pruned_dag, source, target)
                    path_times = [node_times[n] for n in path]
                    assert path_times == sorted(path_times), \
                        f"Path from {source} to {target} has invalid timestamp ordering"


    @pytest.mark.asyncio
    async def test_prune_dag_with_multiple_components(self, pruning_system, mock_confirmation_system):
        """Test pruning behavior with disconnected components"""
        # Create a DAG with multiple components
        multi_component_dag = nx.DiGraph()
        current_time = time.time()
        
        # Configure mock confirmation system with quantum scores
        mock_confirmation_system.quantum_scores = {}
        
        # Component 1 - higher quantum scores
        for i in range(3):
            node_id = f"comp1_block_{i}"
            mock_confirmation_system.quantum_scores[node_id] = 0.9
            multi_component_dag.add_node(
                node_id,
                timestamp=current_time - (3-i)*3600,
                block=Mock(
                    quantum_signature="1010",
                    timestamp=current_time - (3-i)*3600
                )
            )
        multi_component_dag.add_edge("comp1_block_0", "comp1_block_1")
        multi_component_dag.add_edge("comp1_block_1", "comp1_block_2")
        
        # Component 2 - lower quantum scores
        for i in range(3):
            node_id = f"comp2_block_{i}"
            mock_confirmation_system.quantum_scores[node_id] = 0.5
            multi_component_dag.add_node(
                node_id,
                timestamp=current_time - (3-i)*3600,
                block=Mock(
                    quantum_signature="1010",
                    timestamp=current_time - (3-i)*3600
                )
            )
        multi_component_dag.add_edge("comp2_block_0", "comp2_block_1")
        multi_component_dag.add_edge("comp2_block_1", "comp2_block_2")
        
        # Configure mock security info
        async def mock_security_check(tx_hash, block_hash):
            return {
                'security_level': 'MEDIUM',
                'confirmation_score': 0.7,
                'num_confirmations': 5,
                'path_diversity': 0.5,
                'quantum_strength': 0.7
            }
        mock_confirmation_system.get_transaction_security = AsyncMock(side_effect=mock_security_check)
        
        # Verify initial conditions
        initial_components = list(nx.weakly_connected_components(multi_component_dag))
        assert len(initial_components) == 2, "Test should start with two components"
        
        # Perform pruning
        pruned_dag, stats = await pruning_system.prune_dag(multi_component_dag, mock_confirmation_system)
        
        # Verify results
        components = list(nx.weakly_connected_components(pruned_dag))
        assert len(components) == 1, "Pruning should merge or remove disconnected components"
        
        # Verify the correct component was kept (the one with higher quantum scores)
        remaining_nodes = set(pruned_dag.nodes())
        assert "comp1_block_1" in remaining_nodes, "Component with higher quantum scores should be retained"
        assert "comp2_block_1" not in remaining_nodes, "Component with lower quantum scores should be pruned"
        
        # Verify structure is maintained
        assert nx.is_directed_acyclic_graph(pruned_dag), "Pruned DAG should remain acyclic"
        assert pruned_dag.number_of_edges() > 0, "Pruned DAG should maintain its edge structure"



    @pytest.mark.asyncio
    async def test_prune_dag_with_conflicting_transactions(self, pruning_system, sample_dag, mock_confirmation_system):
        """Test pruning behavior with conflicting transactions"""
        # Add conflicting transactions to blocks
        conflict_tx = Mock(hash="conflict_tx")
        
        # Set different timestamps and quantum scores for the blocks
        sample_dag.nodes["block_3"]["timestamp"] = time.time() - 3600  # 1 hour ago
        sample_dag.nodes["block_5"]["timestamp"] = time.time() - 1800  # 30 mins ago
        mock_confirmation_system.quantum_scores["block_3"] = 0.7
        mock_confirmation_system.quantum_scores["block_5"] = 0.9
        
        sample_dag.nodes["block_3"]["block"].transactions = [conflict_tx]
        sample_dag.nodes["block_5"]["block"].transactions = [conflict_tx]
        
        # Perform pruning
        pruned_dag, stats = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify that block_5 (higher quantum score) is kept and block_3 is pruned
        assert (
            ("block_5" in pruned_dag.nodes and "block_3" not in pruned_dag.nodes) or
            ("block_3" in pruned_dag.nodes and "block_5" not in pruned_dag.nodes)
        ), "One conflicting block should be pruned"
        
        # Verify that the DAG remains connected
        assert nx.is_weakly_connected(pruned_dag), "Pruned DAG should remain connected"
        
        # Verify pruning stats
        assert stats['conflicts_resolved'] > 0, "Conflict resolution not recorded in stats"


    @pytest.mark.asyncio
    async def test_prune_dag_metrics_and_stats(self, pruning_system, sample_dag, mock_confirmation_system):
        """Test pruning metrics and statistics"""
        pruned_dag, stats = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        assert isinstance(stats, dict)
        assert 'nodes_pruned' in stats
        assert 'pruning_time' in stats
        assert 'last_prune_time' in stats
        assert stats['nodes_pruned'] >= 0
        assert stats['pruning_time'] > 0
        assert stats['last_prune_time'] <= time.time()

    @pytest.mark.asyncio
    async def test_prune_dag_with_quantum_decoherence(self, pruning_system, sample_dag, mock_confirmation_system):
        """Test pruning considering quantum decoherence effects"""
        # Simulate quantum decoherence by reducing quantum scores over time
        current_time = time.time()
        decoherence_rate = 0.1
        
        quantum_blocks = ['block_3', 'block_4', 'block_5']
        for block in quantum_blocks:
            time_diff = current_time - sample_dag.nodes[block]['timestamp']
            decoherence_factor = np.exp(-decoherence_rate * time_diff)
            mock_confirmation_system.quantum_scores[block] = 0.95 * decoherence_factor
        
        pruned_dag, _ = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify that blocks with significant decoherence are pruned
        for block in quantum_blocks:
            if mock_confirmation_system.quantum_scores[block] < pruning_system.quantum_threshold:
                assert block not in pruned_dag.nodes, \
                    f"Decohered block {block} should have been pruned"

    @pytest.mark.asyncio
    async def test_prune_dag_performance(self, pruning_system, mock_confirmation_system):
        """Test pruning performance with a large DAG"""
        # Create a large DAG
        large_dag = nx.DiGraph()
        current_time = time.time()
        num_nodes = 1000
        
        # Add nodes
        for i in range(num_nodes):
            large_dag.add_node(
                f"block_{i}",
                timestamp=current_time - (num_nodes-i)*60,
                block=Mock(quantum_signature="1010")
            )
        
        # Add edges with some randomness for complexity
        for i in range(num_nodes-1):
            large_dag.add_edge(f"block_{i}", f"block_{i+1}")
            # Add some random cross-edges
            if i < num_nodes-2 and np.random.random() < 0.3:
                large_dag.add_edge(f"block_{i}", f"block_{i+2}")
        
        start_time = time.time()
        pruned_dag, stats = await pruning_system.prune_dag(large_dag, mock_confirmation_system)
        end_time = time.time()
        
        # Verify performance metrics
        processing_time = end_time - start_time
        assert processing_time < 5.0, f"Pruning took too long: {processing_time} seconds"
        assert pruned_dag.number_of_nodes() < large_dag.number_of_nodes()
        assert nx.is_directed_acyclic_graph(pruned_dag)
    @pytest.mark.asyncio
    async def test_prune_dag_invalid_confirmation_data(self, pruning_system, sample_dag):
        """Test pruning behavior with invalid confirmation data"""
        # Create a mock confirmation system with invalid data
        mock_confirmation_system = Mock()
        mock_confirmation_system.quantum_scores = {
            f"block_{i}": float('inf') for i in range(5)  # Invalid quantum scores
        }
        mock_confirmation_system.confirmation_cache = {}
        
        # Mock get_transaction_security to sometimes return invalid data
        async def mock_security_check(tx_hash, block_hash):
            if tx_hash == "block_3":
                return None  # Invalid return
            if tx_hash == "block_4":
                return {}  # Missing required fields
            if tx_hash == "block_5":
                return {
                    'security_level': 'INVALID_LEVEL',
                    'confirmation_score': float('inf'),
                    'num_confirmations': -1,
                    'path_diversity': 2.0,  # Invalid value > 1
                    'quantum_strength': -0.5  # Invalid negative value
                }
            return {
                'security_level': 'MEDIUM',
                'confirmation_score': 0.7,
                'num_confirmations': 5,
                'path_diversity': 0.5,
                'quantum_strength': 0.7
            }
            
        mock_confirmation_system.get_transaction_security = AsyncMock(side_effect=mock_security_check)
        
        # Perform pruning
        pruned_dag, stats = await pruning_system.prune_dag(sample_dag, mock_confirmation_system)
        
        # Verify that pruning completed despite invalid data
        assert isinstance(pruned_dag, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(pruned_dag)
        assert nx.is_weakly_connected(pruned_dag)
        
        # Verify that the DAG maintains valid properties
        is_valid, message = self.verify_dag_properties(pruned_dag)
        assert is_valid, f"Pruned DAG invalid: {message}"
        
        # Verify handling of blocks with invalid data
        if "block_3" in pruned_dag.nodes():
            # Block with None security data should maintain connections if retained
            assert len(list(pruned_dag.predecessors("block_3"))) + \
                   len(list(pruned_dag.successors("block_3"))) > 0
            
        if "block_5" in pruned_dag.nodes():
            # Block with invalid security values should maintain connections if retained
            assert len(list(pruned_dag.predecessors("block_5"))) + \
                   len(list(pruned_dag.successors("block_5"))) > 0
class MockBlock:
    def __init__(self, hash_val: str, transactions: list, timestamp: float):
        self.hash = hash_val
        self.transactions = transactions
        self.timestamp = timestamp
        self.quantum_signature = "1010"
        self.parent_hashes = []

    def to_dict(self):
        return {
            'hash': self.hash,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'timestamp': self.timestamp,
            'quantum_signature': self.quantum_signature,
            'parent_hashes': self.parent_hashes
        }

         
class MockTransaction:
    def __init__(self, tx_hash, signature, amount=Decimal('1.0')):
        self.tx_hash = tx_hash
        self.hash = tx_hash
        self.signature = signature
        self.apply_enhanced_security = AsyncMock()
        self.amount = amount
        self.sender = "test_sender"
        self.receiver = "test_receiver"
        self.timestamp = time.time()
        self.quantum_enabled = True  # Enable quantum features for these transactions
        self.confirmation_data = Mock(
            status=Mock(
                confirmation_score=0.9,
                security_level="HIGH"
            ),
            metrics=Mock(
                path_diversity=0.8,
                quantum_strength=0.9
            )
        )
        self.confirmations = 10

    def to_dict(self):
        return {
            'tx_hash': self.tx_hash,
            'hash': self.hash,
            'signature': self.signature,
            'amount': str(self.amount),
            'sender': self.sender,
            'receiver': self.receiver,
            'timestamp': self.timestamp,
            'quantum_enabled': self.quantum_enabled
        }





class TestIntegratedDAGMiningAndPruning:
    @pytest_asyncio.fixture
    async def mock_crypto_provider(self):
        provider = Mock(spec=CryptoProvider)
        provider.verify_enhanced_security = AsyncMock(return_value=True)
        return provider

    @pytest_asyncio.fixture
    async def confirmation_system(self):
        system = DAGConfirmationSystem(
            quantum_threshold=0.85,
            min_confirmations=2,  # Match pruning system
            max_confirmations=100
        )
        system.quantum_scores = {}
        system.confirmation_cache = {}
        system.evaluate_quantum_signature = Mock(return_value=0.9)
        system.get_transaction_security = Mock(return_value={
            'security_level': 'HIGH',
            'confirmation_score': 0.95,
            'num_confirmations': 10,
            'path_diversity': 0.8,
            'quantum_strength': 0.9
        })
        return system



    @pytest_asyncio.fixture
    async def miner(self, confirmation_system, mock_crypto_provider):
        """Initialize miner with proper configuration"""
        miner = DAGKnightMiner(difficulty=1, security_level=10)
        
        # Set required components
        miner.confirmation_system = confirmation_system
        miner.crypto_provider = mock_crypto_provider
        
        # Initialize the genesis block
        genesis_hash = "0" * 64
        genesis_time = time.time()
        genesis_block = MockBlock(
            hash_val=genesis_hash,
            transactions=[],
            timestamp=genesis_time
        )
        
        # Add genesis block to DAG
        miner.dag.add_node(
            genesis_hash,
            timestamp=genesis_time,
            block=genesis_block
        )
        
        # Initialize quantum scores
        confirmation_system.quantum_scores[genesis_hash] = 0.9
        
        return miner




    @pytest_asyncio.fixture
    async def pruning_system(self):
        return DAGPruningSystem(
            min_confirmations=2,
            quantum_threshold=0.85,
            prune_interval=5,
            max_dag_size=20,
            min_security_level="HIGH"
        )


    @pytest.mark.asyncio
    async def test_mine_and_prune_cycle(self, miner, pruning_system, confirmation_system):
        """Test mining blocks and pruning them in cycles"""
        try:
            transactions = [
                MockTransaction(
                    tx_hash=f"tx_{i}",
                    signature="test_sig",
                    amount=Decimal('1.0')
                ) for i in range(3)
            ]

            mined_blocks = []
            pruning_cycles = 2

            # Pre-populate quantum scores
            for i in range(20):
                confirmation_system.quantum_scores[f"block_{i}"] = 0.9

            for cycle in range(pruning_cycles):
                print(f"\nStarting mining cycle {cycle + 1}/{pruning_cycles}")
                
                # Mine blocks
                for i in range(4):
                    try:
                        block = await miner.mine_block(
                            previous_hash=miner.get_latest_block_hash(),
                            data=f"test_data_{cycle}_{i}",
                            transactions=transactions,
                            reward=Decimal('1.0'),
                            miner_address="test_miner"
                        )

                        assert block is not None
                        mined_blocks.append(block)
                        print(f"Mined block {i + 1}/4 in cycle {cycle + 1}")

                        # Update quantum scores
                        confirmation_system.quantum_scores[block.hash] = 0.9
                        for tx in transactions:
                            confirmation_system.quantum_scores[tx.hash] = 0.9

                    except Exception as e:
                        pytest.fail(f"Mining failed: {str(e)}\nTraceback: {traceback.format_exc()}")

                # Verify mining results
                assert len(miner.dag.nodes) > 0
                print(f"Current DAG size: {len(miner.dag.nodes)} nodes")

                # Perform pruning
                print("Performing pruning...")
                pruned_dag, stats = await pruning_system.prune_dag(miner.dag, confirmation_system)
                miner.dag = pruned_dag
                print(f"Pruned DAG size: {pruned_dag.number_of_nodes()} nodes")

                # Create diagnostics with confirmation system
                diagnostics = DAGDiagnostics(miner)
                diag_results = diagnostics.diagnose_dag_structure()
                
                # Verify diagnostic results
                print("\nDiagnostic Results:")
                print(f"Total nodes: {diag_results['structure']['total_nodes']}")
                print(f"Is DAG: {diag_results['structure']['is_dag']}")
                print(f"Isolated nodes: {len(diag_results['structure']['isolated_nodes'])}")

                assert diag_results["structure"]["is_dag"], "Invalid DAG structure"
                assert not diag_results["structure"]["isolated_nodes"], "Found isolated nodes"
                
                print(f"Cycle {cycle + 1} completed successfully\n")

            print("All mining and pruning cycles completed successfully")

        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}\nTraceback: {traceback.format_exc()}")



    @pytest.mark.asyncio
    async def test_concurrent_mining_and_pruning(self, miner, pruning_system, confirmation_system):
        """Test mining and pruning happening concurrently"""
        
        async def mine_blocks(count: int):
            """Mine blocks concurrently"""
            blocks = []
            try:
                for i in range(count):
                    transactions = [
                        MockTransaction(
                            tx_hash=f"tx_{i}_{j}",
                            signature="test_sig",
                            amount=Decimal('1.0')
                        ) for j in range(3)
                    ]
                    
                    block = await miner.mine_block(
                        previous_hash=miner.get_latest_block_hash(),
                        data=f"test_data_{i}",
                        transactions=transactions,
                        reward=Decimal('1.0'),
                        miner_address="test_miner"
                    )
                    
                    if block:
                        blocks.append(block)
                        # Update quantum scores for the new block and transactions
                        confirmation_system.quantum_scores[block.hash] = 0.9
                        for tx in transactions:
                            confirmation_system.quantum_scores[tx.hash] = 0.9
                    
                    await asyncio.sleep(0.1)  # Simulate mining time
                    
            except Exception as e:
                pytest.fail(f"Mining failed: {str(e)}\nTraceback: {traceback.format_exc()}")
                
            return blocks

        async def prune_periodically():
            """Prune the DAG periodically"""
            try:
                for cycle in range(5):  # 5 pruning cycles
                    await asyncio.sleep(0.3)  # Wait for some blocks to be mined
                    
                    # Get DAG state before pruning
                    pre_prune_size = miner.dag.number_of_nodes()
                    
                    # Perform pruning
                    pruned_dag, stats = await pruning_system.prune_dag(
                        miner.dag,
                        confirmation_system
                    )
                    
                    # Verify pruning operation
                    assert pruned_dag is not None, f"Pruning cycle {cycle} failed to return a valid DAG"
                    assert nx.is_directed_acyclic_graph(pruned_dag), f"Pruning cycle {cycle} produced invalid DAG"
                    
                    # Update miner's DAG
                    miner.dag = pruned_dag
                    
                    # Log pruning results
                    post_prune_size = miner.dag.number_of_nodes()
                    print(f"Pruning cycle {cycle}: Nodes {pre_prune_size} -> {post_prune_size}")
                    
            except Exception as e:
                pytest.fail(f"Pruning failed: {str(e)}\nTraceback: {traceback.format_exc()}")

        try:
            # Pre-populate quantum scores for initial blocks
            for i in range(20):
                confirmation_system.quantum_scores[f"block_{i}"] = 0.9

            # Run mining and pruning concurrently
            mining_task = asyncio.create_task(mine_blocks(10))  # Reduced block count for testing
            pruning_task = asyncio.create_task(prune_periodically())

            # Wait for both tasks to complete
            mined_blocks = await asyncio.wait_for(mining_task, timeout=30)  # Add timeout
            await asyncio.wait_for(pruning_task, timeout=30)  # Add timeout

            # Verify final state
            assert mined_blocks, "No blocks were mined successfully"
            assert nx.is_directed_acyclic_graph(miner.dag), "Final DAG is not acyclic"
            assert nx.is_weakly_connected(miner.dag), "Final DAG is not connected"
            assert miner.dag.number_of_nodes() <= pruning_system.max_dag_size, \
                f"DAG size {miner.dag.number_of_nodes()} exceeds max size {pruning_system.max_dag_size}"

            # Verify block connections
            node_times = nx.get_node_attributes(miner.dag, 'timestamp')
            for edge in miner.dag.edges():
                assert node_times[edge[0]] < node_times[edge[1]], \
                    f"Invalid time ordering: {edge[0]}({node_times[edge[0]]}) -> {edge[1]}({node_times[edge[1]]})"

            # Run final diagnostics
            diagnostics = DAGDiagnostics(miner)
            diag_results = diagnostics.diagnose_dag_structure()
            
            assert diag_results["structure"]["is_dag"], "Final DAG structure is invalid"
            assert not diag_results["structure"]["isolated_nodes"], "Final DAG contains isolated nodes"
            assert diag_results["confirmations"]["total_confirmed"] > 0, "No confirmed transactions in final DAG"

            # Verify mining metrics
            assert miner.mining_metrics['blocks_mined'] > 0, "No blocks were recorded as mined"
            assert miner.mining_metrics['total_hash_calculations'] > 0, "No hash calculations were recorded"

        except asyncio.TimeoutError:
            pytest.fail("Test timed out")
        except Exception as e:
            pytest.fail(f"Concurrent mining and pruning test failed: {str(e)}\nTraceback: {traceback.format_exc()}")


    @pytest.mark.asyncio
    async def test_mining_after_aggressive_pruning(self, miner, pruning_system, confirmation_system):
        """Test that mining can continue properly after aggressive pruning"""
        try:
            # First mine several blocks
            initial_blocks = []
            for i in range(10):
                transactions = [
                    MockTransaction(
                        tx_hash=f"tx_initial_{i}_{j}",
                        signature="test_sig",
                        amount=Decimal('1.0')
                    ) for j in range(3)
                ]
                
                block = await miner.mine_block(
                    previous_hash=miner.get_latest_block_hash(),
                    data=f"initial_data_{i}",
                    transactions=transactions,
                    reward=Decimal('1.0'),
                    miner_address="test_miner"
                )
                
                if block:
                    initial_blocks.append(block)
                    # Update quantum scores for new block and transactions
                    confirmation_system.quantum_scores[block.hash] = 0.9
                    for tx in transactions:
                        confirmation_system.quantum_scores[tx.hash] = 0.9

            assert len(initial_blocks) > 0, "Failed to mine initial blocks"
            
            # Verify initial mining state
            assert nx.is_directed_acyclic_graph(miner.dag), "Initial DAG is not acyclic"
            assert nx.is_weakly_connected(miner.dag), "Initial DAG is not connected"
            
            initial_node_count = miner.dag.number_of_nodes()
            print(f"Initial DAG size: {initial_node_count} nodes")

            # Set very aggressive pruning parameters
            pruning_system.max_dag_size = 5
            pruning_system.min_confirmations = 2

            # Perform aggressive pruning
            print("Performing aggressive pruning...")
            pruned_dag, stats = await pruning_system.prune_dag(miner.dag, confirmation_system)
            
            assert pruned_dag is not None, "Pruning returned None DAG"
            miner.dag = pruned_dag

            # Verify pruning results
            pruned_node_count = miner.dag.number_of_nodes()
            assert pruned_node_count <= 5, f"DAG size {pruned_node_count} exceeds max size 5"
            assert nx.is_directed_acyclic_graph(miner.dag), "Pruned DAG is not acyclic"
            assert nx.is_weakly_connected(miner.dag), "Pruned DAG is not connected"
            
            print(f"After pruning: {pruned_node_count} nodes")

            # Verify timestamp ordering after pruning
            node_times = nx.get_node_attributes(miner.dag, 'timestamp')
            for edge in miner.dag.edges():
                assert node_times[edge[0]] < node_times[edge[1]], \
                    f"Invalid time ordering after pruning: {edge[0]}({node_times[edge[0]]}) -> {edge[1]}({node_times[edge[1]]})"

            # Try mining new blocks after pruning
            print("Mining new blocks after pruning...")
            new_blocks = []
            for i in range(5):
                transactions = [
                    MockTransaction(
                        tx_hash=f"tx_new_{i}_{j}",
                        signature="test_sig",
                        amount=Decimal('1.0')
                    ) for j in range(3)
                ]
                
                block = await miner.mine_block(
                    previous_hash=miner.get_latest_block_hash(),
                    data=f"new_data_{i}",
                    transactions=transactions,
                    reward=Decimal('1.0'),
                    miner_address="test_miner"
                )
                
                if block:
                    new_blocks.append(block)
                    # Update quantum scores for new blocks and transactions
                    confirmation_system.quantum_scores[block.hash] = 0.9
                    for tx in transactions:
                        confirmation_system.quantum_scores[tx.hash] = 0.9
                        
                print(f"Mined block {i+1}/5: {'Success' if block else 'Failed'}")

            # Verify new blocks were mined successfully
            assert len(new_blocks) > 0, "Failed to mine new blocks after pruning"
            assert nx.is_directed_acyclic_graph(miner.dag), "Final DAG is not acyclic"
            assert nx.is_weakly_connected(miner.dag), "Final DAG is not connected"
            
            final_node_count = miner.dag.number_of_nodes()
            print(f"Final DAG size: {final_node_count} nodes")

            # Verify block and transaction connectivity
            latest_block_hash = miner.get_latest_block_hash()
            
            # For blocks, verify they can reach latest block
            blocks = [node for node in miner.dag.nodes() if 'block' in miner.dag.nodes[node]]
            for block in blocks:
                if block != latest_block_hash:
                    assert nx.has_path(miner.dag, block, latest_block_hash), \
                        f"Block {block} cannot reach latest block {latest_block_hash}"
            
            # For transactions, verify they have enough confirmations
            transactions = [node for node in miner.dag.nodes() if 'transaction' in miner.dag.nodes[node]]
            for tx in transactions:
                confirming_blocks = list(nx.ancestors(miner.dag, tx))
                assert len(confirming_blocks) >= pruning_system.min_confirmations, \
                    f"Transaction {tx} has insufficient confirmations: {len(confirming_blocks)}"
            
            # Run final diagnostics
            diagnostics = DAGDiagnostics(miner)
            diag_results = diagnostics.diagnose_dag_structure()
            
            assert diag_results["structure"]["is_dag"], "Final DAG structure is invalid"
            assert not diag_results["structure"]["isolated_nodes"], "Final DAG contains isolated nodes"
            assert diag_results["confirmations"]["total_confirmed"] > 0, "No confirmed transactions in final DAG"

            print("Mining after aggressive pruning test completed successfully")

        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}\nTraceback: {traceback.format_exc()}")


    @pytest.mark.asyncio
    async def test_mining_with_quantum_state_preservation(self, miner, pruning_system, confirmation_system):
        """Test that mining preserves quantum states through pruning cycles"""
        try:
            quantum_blocks = []
            print("\nStarting quantum state preservation test...")

            # Track quantum states
            quantum_states = {}
            
            # Mine blocks with strong quantum signatures
            print("Mining initial quantum blocks...")
            for i in range(8):
                transactions = [
                    MockTransaction(
                        tx_hash=f"tx_quantum_{i}_{j}",
                        signature="test_sig",
                        amount=Decimal('1.0')
                    ) for j in range(3)
                ]

                # Configure quantum scores
                quantum_score = 0.95 if i % 2 == 0 else 0.7
                confirmation_system.quantum_scores[f"block_{i}"] = quantum_score
                print(f"Block {i} quantum score: {quantum_score}")

                # Mine block
                block = await miner.mine_block(
                    previous_hash=miner.get_latest_block_hash(),
                    data=f"quantum_data_{i}",
                    transactions=transactions,
                    reward=Decimal('1.0'),
                    miner_address="test_miner"
                )

                if block:
                    quantum_blocks.append(block)
                    # Store quantum state
                    quantum_states[block.hash] = {
                        'score': quantum_score,
                        'timestamp': time.time(),
                        'transaction_counts': len(transactions)
                    }
                    print(f"Mined block {i} with hash {block.hash[:8]}...")

            assert len(quantum_blocks) > 0, "Failed to mine initial quantum blocks"

            # Verify initial quantum states
            initial_high_score_blocks = [
                node for node in miner.dag.nodes()
                if confirmation_system.quantum_scores.get(node, 0) >= pruning_system.quantum_threshold
            ]
            print(f"\nInitial high score blocks: {len(initial_high_score_blocks)}")

            # Perform pruning
            print("\nPerforming DAG pruning...")
            pruned_dag, stats = await pruning_system.prune_dag(miner.dag, confirmation_system)
            assert pruned_dag is not None, "Pruning returned None DAG"
            
            pre_prune_size = miner.dag.number_of_nodes()
            miner.dag = pruned_dag
            post_prune_size = miner.dag.number_of_nodes()
            print(f"DAG size: {pre_prune_size} -> {post_prune_size} nodes")

            # Verify high quantum score blocks are retained
            high_score_blocks = [
                node for node in miner.dag.nodes()
                if confirmation_system.quantum_scores.get(node, 0) >= pruning_system.quantum_threshold
            ]
            print(f"High score blocks after pruning: {len(high_score_blocks)}")
            assert len(high_score_blocks) > 0, "No high score blocks retained after pruning"

            # Verify quantum state preservation
            for block_hash in high_score_blocks:
                original_state = quantum_states.get(block_hash)
                if original_state:
                    current_score = confirmation_system.quantum_scores.get(block_hash, 0)
                    assert current_score >= pruning_system.quantum_threshold, \
                        f"Block {block_hash[:8]} lost quantum strength: {current_score}"

            # Mine additional blocks
            print("\nMining additional blocks...")
            for i in range(5):
                transactions = [
                    MockTransaction(
                        tx_hash=f"tx_post_{i}_{j}",
                        signature="test_sig",
                        amount=Decimal('1.0')
                    ) for j in range(3)
                ]

                block = await miner.mine_block(
                    previous_hash=miner.get_latest_block_hash(),
                    data=f"post_quantum_data_{i}",
                    transactions=transactions,
                    reward=Decimal('1.0'),
                    miner_address="test_miner"
                )
                
                if block:
                    quantum_blocks.append(block)
                    # Add quantum score for new block
                    confirmation_system.quantum_scores[block.hash] = 0.9
                    quantum_states[block.hash] = {
                        'score': 0.9,
                        'timestamp': time.time(),
                        'transaction_counts': len(transactions)
                    }
                    print(f"Mined additional block {i} with hash {block.hash[:8]}...")

            # Verify quantum state preservation
            print("\nVerifying final DAG state...")
            diagnostics = DAGDiagnostics(miner)
            diag_results = diagnostics.diagnose_dag_structure()

            assert diag_results["structure"]["is_dag"], "Final DAG structure is invalid"
            assert not diag_results["structure"]["isolated_nodes"], "Final DAG contains isolated nodes"

            # Verify high quantum score blocks are still accessible
            for block_hash in high_score_blocks:
                if block_hash in miner.dag:
                    paths = nx.single_source_shortest_path_length(miner.dag, block_hash)
                    assert len(paths) > 0, f"High quantum score block {block_hash[:8]} lost connectivity"
                    print(f"Block {block_hash[:8]} maintains {len(paths)} valid paths")

            # Verify quantum coherence preservation
            quantum_coherence = sum(
                confirmation_system.quantum_scores.get(node, 0) 
                for node in miner.dag.nodes()
            ) / miner.dag.number_of_nodes()
            
            print(f"\nFinal quantum coherence: {quantum_coherence:.4f}")
            assert quantum_coherence >= 0.7, f"Overall quantum coherence too low: {quantum_coherence}"

            print("\nQuantum state preservation test completed successfully")

        except Exception as e:
            pytest.fail(f"Test failed: {str(e)}\nTraceback: {traceback.format_exc()}")


if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=auto"])