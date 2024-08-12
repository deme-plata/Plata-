import asyncio
from decimal import Decimal
from datetime import datetime
from typing import List, Dict
import psutil
import numpy as np
import networkx as nx
import scipy.sparse as sparse
from scipy.optimize import minimize
import time
import logging
import random
import traceback
from enhanced_exchange import EnhancedExchangeWithZKStarks
from quantumdagknight import PriceOracle
from .shared_logic import get_quantum_blockchain, get_p2p_node, get_enhanced_exchange
from quantumdagknight import NodeDirectory, Wallet  # Importing NodeDirectory
from vm import SimpleVM, PBFTConsensus
from SecureHybridZKStark import SecureHybridZKStark
QuantumBlockchain = get_quantum_blockchain()
P2PNode = get_p2p_node()
EnhancedExchangeWithZKStarks = get_enhanced_exchange()

logger = logging.getLogger(__name__)
class BlockchainInterface:
    def __init__(self):
        self.node_directory = NodeDirectory()
        self.consensus = PBFTConsensus(nodes=[], node_id="node_1")
        self.secret_key = "your_secret_key_here"
        self.vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
        self.blockchain = QuantumBlockchain(self.consensus, self.secret_key, self.node_directory, self.vm)
        self.wallet = Wallet(public_key="your_public_key_here", address="your_address_here")  # Provide the required fields
        self.zk_system = SecureHybridZKStark(security_level=2)
        self.mining_active = False
        self.p2p_node = P2PNode("localhost", 8000, self.blockchain)  # Initialize P2P node
        self.exchange = EnhancedExchangeWithZKStarks(self.blockchain, self.vm, PriceOracle(), self.node_directory, 2)


    async def start_p2p_node(self):
        await self.p2p_node.start()

    async def get_node_stats(self) -> Dict:
        return {
            "node_id": self.consensus.node_id,
            "connected_peers": len(self.p2p_node.peers),
            "block_height": len(self.blockchain.chain),
            "last_block_time": self.blockchain.chain[-1].timestamp if self.blockchain.chain else "N/A"
        }

    async def get_wallet_balance(self, currency):
        return await self.blockchain.get_balance(self.wallet.address, currency)


    async def get_transaction_history(self) -> List[Dict]:
        transactions = await self.blockchain.get_transactions(self.wallet.address)
        return [
            {
                "date": tx.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": tx.amount,
                "recipient": tx.receiver
            }
            for tx in transactions[-10:]  # Get last 10 transactions
        ]

    async def send_transaction(self, recipient: str, amount: Decimal) -> str:
        try:
            tx = Transaction(
                sender=self.wallet.address,
                receiver=recipient,
                amount=amount,
                private_key=self.wallet.private_key,
                public_key=self.wallet.get_public_key()
            )
            tx.sign_transaction(self.zk_system)
            success = await self.blockchain.add_transaction(tx)
            if success:
                await self.p2p_node.propagate_transaction(tx)
                return f"Transaction of {amount} QDAGK sent to {recipient}"
            else:
                return "Transaction failed to be added to the blockchain"
        except Exception as e:
            return f"Error sending transaction: {str(e)}"

    async def start_mining(self):
        if not self.mining_active:
            self.mining_active = True
            asyncio.create_task(self._mine_blocks())

    async def stop_mining(self):
        self.mining_active = False

    async def _mine_blocks(self):
        while self.mining_active:
            try:
                result = await self.mining_algorithm(iterations=2)
                if result["success"]:
                    new_block = await self.blockchain.create_new_block(self.wallet.address)
                    if new_block:
                        await self.p2p_node.propagate_block(new_block)
                        logger.info(f"New block mined and propagated: {new_block.hash}")
                    else:
                        logger.error("Failed to create a new block")
                else:
                    logger.warning("Mining algorithm did not produce a valid result")
            except Exception as e:
                logger.error(f"Error during mining: {str(e)}")
            await asyncio.sleep(10)  # Adjust based on your desired block time

    async def mining_algorithm(self, iterations=5):
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage at start: {memory_info.rss / (1024 * 1024):.2f} MB")

            logger.info("Initializing Quantum Annealing Simulation")
            num_qubits = 10
            graph = nx.grid_graph(dim=[2, 5])
            logger.info(f"Initialized graph with {len(graph.nodes)} nodes and {len(graph.edges())} edges")

            def quantum_annealing_simulation(params):
                hamiltonian = sparse.csr_matrix((2**num_qubits, 2**num_qubits), dtype=complex)
                
                grid_dims = list(graph.nodes())[-1]
                rows, cols = grid_dims[0] + 1, grid_dims[1] + 1
                
                for edge in graph.edges():
                    i, j = edge
                    sigma_z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
                    
                    i_int = i[0] * cols + i[1]
                    j_int = j[0] * cols + j[1]
                    
                    term_i = sparse.kron(sparse.eye(2**i_int, dtype=complex), sigma_z)
                    term_i = sparse.kron(term_i, sparse.eye(2**(num_qubits-i_int-1), dtype=complex))
                    hamiltonian += term_i
                    
                    term_j = sparse.kron(sparse.eye(2**j_int, dtype=complex), sigma_z)
                    term_j = sparse.kron(term_j, sparse.eye(2**(num_qubits-j_int-1), dtype=complex))
                    hamiltonian += term_j

                problem_hamiltonian = sparse.diags(np.random.randn(2**num_qubits), dtype=complex)
                hamiltonian += params[0] * problem_hamiltonian

                initial_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
                evolution = sparse.linalg.expm(-1j * hamiltonian.tocsc() * params[1])
                final_state = evolution @ initial_state

                return -np.abs(final_state[0])**2

            cumulative_counts = {}
            for iteration in range(iterations):
                logger.info(f"Starting iteration {iteration + 1}/{iterations}")
                start_time_simulation = time.time()
                random_params = [random.uniform(0.1, 2.0), random.uniform(0.1, 2.0)]
                
                result = minimize(quantum_annealing_simulation, random_params, method='Nelder-Mead')
                end_time_simulation = time.time()

                simulation_duration = end_time_simulation - start_time_simulation
                logger.info(f"Simulation completed in {simulation_duration:.2f} seconds")

                mass_distribution = np.random.rand(2, 5)
                gravity_factor = np.sum(mass_distribution) / 10

                black_hole_position = np.unravel_index(np.argmax(mass_distribution), mass_distribution.shape)
                black_hole_strength = mass_distribution[black_hole_position]

                entanglement_matrix = np.abs(np.outer(result.x, result.x))
                
                hawking_radiation = np.random.exponential(scale=black_hole_strength, size=10)

                final_state = np.ones(2**num_qubits, dtype=complex) / np.sqrt(2**num_qubits)
                counts = {format(i, f'0{num_qubits}b'): abs(val)**2 for i, val in enumerate(final_state) if abs(val)**2 > 1e-6}

                for state, prob in counts.items():
                    if state in cumulative_counts:
                        cumulative_counts[state] += prob
                    else:
                        cumulative_counts[state] = prob

            memory_info = process.memory_info()
            logger.info(f"Memory usage after simulation: {memory_info.rss / (1024 * 1024):.2f} MB")

            qhins = np.trace(entanglement_matrix)
            hashrate = 1 / (simulation_duration * iterations)

            logger.info(f"QHINs: {qhins:.6f}")
            logger.info(f"Hashrate: {hashrate:.6f} hashes/second")

            return {
                "success": True,
                "counts": cumulative_counts,
                "energy": result.fun,
                "entanglement_matrix": entanglement_matrix,
                "qhins": qhins,
                "hashrate": hashrate
            }
        except Exception as e:
            logger.error(f"Error in mining_algorithm: {str(e)}")
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error in mining_algorithm: {str(e)}"}

    async def get_mining_stats(self) -> Dict:
        return {
            "status": "ACTIVE" if self.mining_active else "INACTIVE",
            "hash_rate": self.blockchain.calculate_hash_rate(),
            "blocks_mined": len(self.blockchain.chain)
        }

    async def get_network_stats(self) -> Dict:
        consistency_results = await self.p2p_node.verify_network_consistency()
        return {
            "connected_peers": len(self.p2p_node.peers),
            "network_consistency": consistency_results,
            "mempool_size": len(self.p2p_node.mempool)
        }
    async def get_orders(self):
        return await self.exchange.get_orders()

    async def place_limit_order(self, user: str, order_type: str, pair: str, amount: Decimal, price: Decimal):
        return await self.exchange.place_limit_order(user, order_type, pair, amount, price)

    async def cancel_order(self, user: str, order_id: str):
        return await self.exchange.cancel_order(user, order_id)

    async def add_liquidity(self, user: str, pool_id: str, amount_a: Decimal, amount_b: Decimal):
        return await self.exchange.add_liquidity(user, pool_id, amount_a, amount_b)

    async def get_liquidity_pools(self):
        return self.exchange.liquidity_pools

    async def get_tradable_assets(self):
        return await self.exchange.get_tradable_assets()
    async def send_transaction_ui(self, recipient: str, amount: Decimal) -> str:
        return await self.send_transaction(recipient, amount)

    async def view_transaction_history(self):
        return await self.get_transaction_history()

    async def toggle_mining(self):
        if self.mining_active:
            await self.stop_mining()
        else:
            await self.start_mining()

    async def get_wallet_balance(self, currency):
        return await self.blockchain.get_balance(self.wallet.address, currency)


# Create a global instance of the BlockchainInterface
blockchain_interface = BlockchainInterface()

# These functions will be imported and used in your UI code

async def get_node_stats():
    return await blockchain_interface.get_node_stats()

async def get_wallet_balance(self, currency):
    return await self.blockchain.get_balance(self.wallet.address, currency)

async def get_transaction_history():
    return await blockchain_interface.get_transaction_history()

async def send_transaction(recipient: str, amount: float):
    return await blockchain_interface.send_transaction(recipient, Decimal(str(amount)))

async def start_mining():
    await blockchain_interface.start_mining()

async def stop_mining():
    await blockchain_interface.stop_mining()

async def get_mining_stats():
    return await blockchain_interface.get_mining_stats()

async def get_network_stats():
    return await blockchain_interface.get_network_stats()

async def start_p2p_node():
    await blockchain_interface.start_p2p_node()
    
async def get_orders():
    return await blockchain_interface.get_orders()

async def place_limit_order(user: str, order_type: str, pair: str, amount: float, price: float):
    return await blockchain_interface.place_limit_order(user, order_type, pair, Decimal(str(amount)), Decimal(str(price)))

async def cancel_order(user: str, order_id: str):
    return await blockchain_interface.cancel_order(user, order_id)

async def add_liquidity(user: str, pool_id: str, amount_a: float, amount_b: float):
    return await blockchain_interface.add_liquidity(user, pool_id, Decimal(str(amount_a)), Decimal(str(amount_b)))

async def get_liquidity_pools():
    return await blockchain_interface.get_liquidity_pools()

async def get_tradable_assets():
    return await blockchain_interface.get_tradable_assets()
