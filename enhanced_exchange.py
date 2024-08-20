import uuid
from pydantic import BaseModel
from decimal import Decimal
from typing import Dict, Tuple, List
from EnhancedOrderBook import EnhancedOrderBook
from qiskit import QuantumCircuit, transpile
import aiohttp
import time
import grpc
import dagknight_pb2_grpc
import dagknight_pb2
import logging
from qiskit_aer import Aer, AerSimulator
from Order import Order
from Order import Order
from typing import Dict, List, Tuple, Any  # Make sure Any is imported
import asyncio
from Order import Order
from FiniteField import FiniteField, FieldElement
from finite_field_factory import FiniteFieldFactory
import traceback
from zero_x_swap import ZeroXSwapAPI
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import os
from EnhancedOrderBook import EnhancedOrderBook
from blockcypher_integration import BlockCypherAPI


logger = logging.getLogger(__name__)

class LiquidityPool:
    def __init__(self, token_a: str, token_b: str, fee_percent: Decimal):
        self.token_a = token_a
        self.token_b = token_b
        self.reserve_a = Decimal('0')
        self.reserve_b = Decimal('0')
        self.fee_percent = fee_percent
        self.total_liquidity = Decimal('0')
        self.liquidity_shares: Dict[str, Decimal] = {}
        self.address = str(uuid.uuid4())

    async def add_liquidity(self, user: str, amount_a: Decimal, amount_b: Decimal) -> Decimal:
        if self.total_liquidity == 0:
            liquidity_minted = (amount_a * amount_b).sqrt()
            self.reserve_a = amount_a
            self.reserve_b = amount_b
        else:
            liquidity_minted = min(
                amount_a * self.total_liquidity / self.reserve_a,
                amount_b * self.total_liquidity / self.reserve_b
            )
            self.reserve_a += amount_a
            self.reserve_b += amount_b

        self.total_liquidity += liquidity_minted
        self.liquidity_shares[user] = self.liquidity_shares.get(user, Decimal('0')) + liquidity_minted
        return liquidity_minted

    async def remove_liquidity(self, user: str, liquidity: Decimal) -> Tuple[Decimal, Decimal]:
        if liquidity > self.liquidity_shares.get(user, Decimal('0')):
            raise ValueError("Insufficient liquidity tokens")

        amount_a = liquidity * self.reserve_a / self.total_liquidity
        amount_b = liquidity * self.reserve_b / self.total_liquidity

        self.reserve_a -= amount_a
        self.reserve_b -= amount_b
        self.total_liquidity -= liquidity
        self.liquidity_shares[user] -= liquidity

        return amount_a, amount_b

    async def swap(self, amount_in: Decimal, token_in: str) -> Decimal:
        if token_in not in [self.token_a, self.token_b]:
            raise ValueError("Invalid input token")

        fee = amount_in * self.fee_percent
        amount_in_with_fee = amount_in - fee

        if token_in == self.token_a:
            amount_out = (amount_in_with_fee * self.reserve_b) / (self.reserve_a + amount_in_with_fee)
            self.reserve_a += amount_in
            self.reserve_b -= amount_out
        else:
            amount_out = (amount_in_with_fee * self.reserve_a) / (self.reserve_b + amount_in_with_fee)
            self.reserve_b += amount_in
            self.reserve_a -= amount_out

        return amount_out

    def get_reserves(self) -> Tuple[Decimal, Decimal]:
        return self.reserve_a, self.reserve_b

    def get_user_share(self, user: str) -> Decimal:
        return self.liquidity_shares.get(user, Decimal('0'))

    def get_pool_share(self, shares: Decimal) -> Tuple[Decimal, Decimal]:
        if shares > self.total_liquidity:
            raise ValueError("Insufficient shares in pool")

        amount_a = shares * self.reserve_a / self.total_liquidity
        amount_b = shares * self.reserve_b / self.total_liquidity
        return amount_a, amount_b

    def get_price(self) -> Decimal:
        return self.reserve_b / self.reserve_a if self.reserve_a > 0 else Decimal('0')

class PriceOracle:
    def __init__(self):
        self.prices: Dict[str, Decimal] = {}
        self.last_update: float = 0
        self.update_interval: float = 60  # Update prices every 60 seconds

    async def get_price(self, token: str) -> Decimal:
        if time.time() - self.last_update > self.update_interval:
            await self.update_prices()
        return self.prices.get(token, Decimal('0'))

    async def update_prices(self):
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,polkadot&vs_currencies=usd') as response:
                data = await response.json()
                self.prices = {
                    'BTC': Decimal(str(data['bitcoin']['usd'])),
                    'ETH': Decimal(str(data['ethereum']['usd'])),
                    'DOT': Decimal(str(data['polkadot']['usd'])),
                    'PLATA': Decimal('1.0')  # Assuming 1 PLATA = 1 USD for simplicity
                }
        self.last_update = time.time()

class MarginAccount:
    def __init__(self, user: str):
        self.user = user
        self.positions: Dict[str, Dict] = {}

    async def open_position(self, pair: str, side: str, amount: Decimal, leverage: int, price_oracle: PriceOracle):
        if pair in self.positions:
            raise ValueError("Position already exists")
        
        price = await price_oracle.get_price(pair)
        margin_required = amount / leverage
        
        self.positions[pair] = {
            "side": side,
            "amount": amount,
            "leverage": leverage,
            "entry_price": price,
            "margin": margin_required
        }

    async def close_position(self, pair: str, price_oracle: PriceOracle) -> Decimal:
        if pair not in self.positions:
            raise ValueError("Position does not exist")
        
        position = self.positions[pair]
        current_price = await price_oracle.get_price(pair)
        
        if position["side"] == "long":
            pnl = (current_price - position["entry_price"]) * position["amount"]
        else:
            pnl = (position["entry_price"] - current_price) * position["amount"]

        del self.positions[pair]
        return pnl + position["margin"]
class EnhancedLiquidityPool:
    def __init__(self, token_a: str, token_b: str, fee_percent: Decimal):
        self.token_a = token_a
        self.token_b = token_b
        self.balance_a = Decimal('0')
        self.balance_b = Decimal('0')
        self.total_shares = Decimal('0')
        self.shares: Dict[str, Decimal] = {}
        self.fee_percent = fee_percent
        self.MIN_LIQUIDITY = Decimal('0.001')  # Minimum liquidity to prevent attacks
        self.address = str(uuid.uuid4())  # Adding address attribute

    async def add_liquidity(self, user: str, amount_a: Decimal, amount_b: Decimal) -> Decimal:
        if self.total_shares == Decimal('0'):
            # First liquidity provision
            liquidity_minted = (amount_a * amount_b).sqrt() - self.MIN_LIQUIDITY
            self.total_shares = liquidity_minted + self.MIN_LIQUIDITY
            self.shares[user] = liquidity_minted
        else:
            # Subsequent liquidity provisions
            share_a = amount_a * self.total_shares / self.balance_a
            share_b = amount_b * self.total_shares / self.balance_b
            liquidity_minted = min(share_a, share_b)
            self.total_shares += liquidity_minted
            self.shares[user] = self.shares.get(user, Decimal('0')) + liquidity_minted

        self.balance_a += amount_a
        self.balance_b += amount_b

        return liquidity_minted

    async def remove_liquidity(self, user: str, shares: Decimal) -> Tuple[Decimal, Decimal]:
        if shares > self.shares.get(user, Decimal('0')):
            raise ValueError("Insufficient shares")

        amount_a = shares * self.balance_a / self.total_shares
        amount_b = shares * self.balance_b / self.total_shares

        self.balance_a -= amount_a
        self.balance_b -= amount_b
        self.total_shares -= shares
        self.shares[user] -= shares

        return amount_a, amount_b

    async def swap(self, amount_in: Decimal, token_in: str) -> Decimal:
        if token_in not in [self.token_a, self.token_b]:
            raise ValueError("Invalid input token")

        reserve_in = self.balance_a if token_in == self.token_a else self.balance_b
        reserve_out = self.balance_b if token_in == self.token_a else self.balance_a

        amount_in_with_fee = amount_in * (Decimal('1') - self.fee_percent)
        amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)

        if token_in == self.token_a:
            self.balance_a += amount_in
            self.balance_b -= amount_out
        else:
            self.balance_b += amount_in
            self.balance_a -= amount_out

        return amount_out

    def get_reserves(self) -> Tuple[Decimal, Decimal]:
        return self.balance_a, self.balance_b

    def get_user_share(self, user: str) -> Decimal:
        return self.shares.get(user, Decimal('0'))

    def get_pool_share(self, shares: Decimal) -> Tuple[Decimal, Decimal]:
        if shares > self.total_shares:
            raise ValueError("Insufficient shares in pool")
        
        amount_a = shares * self.balance_a / self.total_shares
        amount_b = shares * self.balance_b / self.total_shares
        return amount_a, amount_b

    def get_price(self) -> Decimal:
        return self.balance_b / self.balance_a if self.balance_a > 0 else Decimal('0')



class LiquidityPoolManager:
    def __init__(self):
        self.pools: Dict[str, EnhancedLiquidityPool] = {}

    def create_pool(self, token_a: str, token_b: str, fee_percent: Decimal) -> str:
        pool_id = f"{token_a}_{token_b}"
        if pool_id in self.pools:
            raise ValueError("Pool already exists")
        self.pools[pool_id] = EnhancedLiquidityPool(token_a, token_b, fee_percent)
        return pool_id

    def get_pool(self, token_a: str, token_b: str) -> EnhancedLiquidityPool:
        pool_id = f"{token_a}_{token_b}"
        if pool_id not in self.pools:
            raise ValueError("Pool does not exist")
        return self.pools[pool_id]

    async def add_liquidity(self, user: str, token_a: str, token_b: str, amount_a: Decimal, amount_b: Decimal) -> Decimal:
        pool = self.get_pool(token_a, token_b)
        return await pool.add_liquidity(user, amount_a, amount_b)

    async def remove_liquidity(self, user: str, token_a: str, token_b: str, shares: Decimal) -> Tuple[Decimal, Decimal]:
        pool = self.get_pool(token_a, token_b)
        return await pool.remove_liquidity(user, shares)

    async def swap(self, amount_in: Decimal, token_in: str, token_out: str) -> Decimal:
        pool = self.get_pool(token_in, token_out)
        return await pool.swap(amount_in, token_in)

class LendingPool:
    def __init__(self, currency: str):
        self.currency = currency
        self.total_supplied = Decimal('0')
        self.total_borrowed = Decimal('0')
        self.supplies: Dict[str, Decimal] = {}
        self.borrows: Dict[str, Dict] = {}
        self.interest_rate = Decimal('0.05')  # 5% APR
        self.collateral_ratio = Decimal('1.5')
        self.address = str(uuid.uuid4())  # Corrected to be within the constructor

    async def lend(self, user: str, amount: Decimal):
        self.supplies[user] = self.supplies.get(user, Decimal('0')) + amount
        self.total_supplied += amount

    async def borrow(self, user: str, amount: Decimal, collateral_currency: str, collateral_amount: Decimal, price_oracle: PriceOracle):
        if amount > self.total_supplied - self.total_borrowed:
            raise ValueError("Insufficient liquidity in the pool")

        collateral_price = await price_oracle.get_price(f"{collateral_currency}_{self.currency}")
        collateral_value = collateral_amount * collateral_price

        if collateral_value < amount * self.collateral_ratio:
            raise ValueError("Insufficient collateral")

        self.borrows[user] = {
            "amount": amount,
            "collateral_currency": collateral_currency,
            "collateral_amount": collateral_amount
        }
        self.total_borrowed += amount

    async def repay(self, user: str, amount: Decimal) -> Decimal:
        if user not in self.borrows:
            raise ValueError("No outstanding loan")

        loan = self.borrows[user]
        if amount > loan["amount"]:
            amount = loan["amount"]

        loan["amount"] -= amount
        self.total_borrowed -= amount

        if loan["amount"] == 0:
            collateral_released = loan["collateral_amount"]
            del self.borrows[user]
        else:
            collateral_released = loan["collateral_amount"] * (amount / loan["amount"])
            loan["collateral_amount"] -= collateral_released

        return collateral_released

    def get_collateral_currency(self, user: str) -> str:
        return self.borrows[user]["collateral_currency"]

class QuantumInspiredFeatures:
    @staticmethod
    async def quantum_hedging(exchange: 'EnhancedExchange', user: str, pair: str, amount: Decimal):
        quantum_circuit = QuantumCircuit(2, 2)
        quantum_circuit.h(0)  # Apply Hadamard gate
        quantum_circuit.cx(0, 1)  # Apply CNOT gate
        quantum_circuit.measure([0, 1], [0, 1])

        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(quantum_circuit, backend)
        job = backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()

        if counts.get('00', 0) + counts.get('11', 0) > counts.get('01', 0) + counts.get('10', 0):
            await exchange.open_margin_position(user, pair, "long", amount, 2)
        else:
            await exchange.open_margin_position(user, pair, "short", amount, 2)

        return "Quantum-inspired hedge position opened"

    @staticmethod
    async def entanglement_based_arbitrage(exchange: 'EnhancedExchange', user: str, pairs: List[str]):
        n = len(pairs)
        quantum_circuit = QuantumCircuit(n, n)
        for i in range(n):
            quantum_circuit.h(i)
        for i in range(n-1):
            quantum_circuit.cx(i, i+1)
        quantum_circuit.measure_all()

        backend = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(quantum_circuit, backend)
        job = backend.run(transpiled_circuit, shots=1000)
        result = job.result()
        counts = result.get_counts()

        most_common_state = max(counts, key=counts.get)
        arbitrage_paths = [pairs[i] for i, bit in enumerate(most_common_state) if bit == '1']

        for pair in arbitrage_paths:
            await exchange.place_market_order(user, "buy", pair, Decimal('0.1'))
            await exchange.place_market_order(user, "sell", pair, Decimal('0.1'))

        return f"Quantum-inspired arbitrage executed on pairs: {', '.join(arbitrage_paths)}"

class AdvancedOrderTypes:
    @staticmethod
    async def place_stop_loss_order(exchange: 'EnhancedExchange', user: str, pair: str, amount: Decimal, stop_price: Decimal):
        current_price = await exchange.price_oracle.get_price(pair)
        if current_price <= stop_price:
            return await exchange.place_market_order(user, "sell", pair, amount)
        else:
            exchange.stop_loss_orders.append({
                "user": user,
                "pair": pair,
                "amount": amount,
                "stop_price": stop_price
            })
            return "Stop loss order placed"

    @staticmethod
    async def place_take_profit_order(exchange: 'EnhancedExchange', user: str, pair: str, amount: Decimal, take_profit_price: Decimal):
        current_price = await exchange.price_oracle.get_price(pair)
        if current_price >= take_profit_price:
            return await exchange.place_market_order(user, "sell", pair, amount)
        else:
            exchange.take_profit_orders.append({
                "user": user,
                "pair": pair,
                "amount": amount,
                "take_profit_price": take_profit_price
            })
            return "Take profit order placed"

    @staticmethod
    async def place_trailing_stop_order(exchange: 'EnhancedExchange', user: str, pair: str, amount: Decimal, trail_percent: Decimal):
        current_price = await exchange.price_oracle.get_price(pair)
        trail_amount = current_price * trail_percent
        exchange.trailing_stop_orders.append({
            "user": user,
            "pair": pair,
            "amount": amount,
            "trail_amount": trail_amount,
            "highest_price": current_price
        })
        return "Trailing stop order placed"

class EnhancedExchange:
    from user_management import fake_users_db

    def __init__(self, blockchain, vm, price_oracle, node_directory):
        self.blockchain = blockchain
        self.vm = vm
        self.price_oracle = price_oracle
        self.order_book = EnhancedOrderBook()
        self.liquidity_pools: Dict[str, EnhancedLiquidityPool] = {}
        self.fee_percent = Decimal('0.001')  # 0.1% fee
        self.margin_accounts: Dict[str, MarginAccount] = {}
        self.lending_pools: Dict[str, LendingPool] = {}
        self.stop_loss_orders = []
        self.take_profit_orders = []
        self.trailing_stop_orders = []
        self.advanced_order_types = AdvancedOrderTypes()
        self.quantum_features = QuantumInspiredFeatures()
        self.node_directory = node_directory
        self.margin_positions = {}
        self.p2p_node = blockchain.p2p_node  # Use the same P2PNode as the blockchain
        load_dotenv()

        # Load and decrypt the API keys
        with open("encryption_key.key", "rb") as key_file:
            encryption_key = key_file.read()
        f = Fernet(encryption_key)
        decrypted_zero_x_api_key = f.decrypt(os.getenv("ZEROX_API_KEY").encode()).decode()
        decrypted_blockcypher_api_key = f.decrypt(os.getenv("BLOCKCYPHER_API_KEY").encode()).decode()

        # Initialize the ZeroXSwapAPI and BlockCypherAPI with the decrypted keys
        self.zero_x_api = ZeroXSwapAPI(decrypted_zero_x_api_key)
        self.blockcypher_api = BlockCypherAPI(decrypted_blockcypher_api_key)
    def get_blockchain_info(self):
        return self.blockcypher_api.get_blockchain_info()

    def generate_btc_address(self):
        return self.blockcypher_api.generate_address()

    def create_btc_transaction(self, inputs, outputs):
        tx = self.blockcypher_api.create_transaction(inputs, outputs)
        tx_signed = self.sign_transaction(tx)
        return self.blockcypher_api.send_transaction(tx_signed)

    def sign_transaction(self, tx):
        # Implement your signing logic here
        pass

    def setup_transaction_webhook(self, callback_url, address):
        return self.blockcypher_api.setup_webhook(event_type="unconfirmed-tx", url=callback_url, address=address)

    def get_user_wallet(self, user_id):
        from user_management import get_user_wallet

        return get_user_wallet(user_id)



    async def sync_state(self):
        try:
            remote_orders = []
            nodes = self.node_directory.discover_nodes()
            print(f"[SYNC] Discovered nodes: {nodes}")
            for node in nodes:
                if node['is_active']:
                    try:
                        async with grpc.aio.insecure_channel(f"{node['ip_address']}:{node['port']}") as channel:
                            stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                            request = dagknight_pb2.GetOrdersRequest()
                            response = await stub.GetOrders(request)
                            remote_orders.extend([self._convert_proto_to_order(order) for order in response.orders])
                    except Exception as e:
                        print(f"[SYNC] Failed to get orders from node {node['node_id']}: {str(e)}")

            print(f"[SYNC] Remote orders: {remote_orders}")
            local_orders = self.order_book.get_orders()
            print(f"[SYNC] Local orders before sync: {local_orders}")

            for remote_order in remote_orders:
                if not any(local_order.id == remote_order.id for local_order in local_orders):
                    self.order_book.add_order(remote_order)
                    print(f"[SYNC] Added new order: {remote_order}")

            local_order_ids = {order.id for order in local_orders}
            remote_order_ids = {order.id for order in remote_orders}
            orders_to_cancel = local_order_ids - remote_order_ids
            for order_id in orders_to_cancel:
                self.order_book.cancel_order(order_id)
                print(f"[SYNC] Cancelled order: {order_id}")

            print(f"[SYNC] After sync, order book contains {len(self.order_book.get_orders())} orders")
        except Exception as e:
            print(f"[SYNC] Error in sync_state: {str(e)}")
            raise

    def _convert_proto_to_order(self, proto_order):
        return Order(
            user_id=proto_order.user_id,
            type=proto_order.type,
            order_type=proto_order.order_type,
            pair=proto_order.pair,
            amount=Decimal(proto_order.amount),
            price=Decimal(proto_order.price),
            base_currency=proto_order.base_currency,
            quote_currency=proto_order.quote_currency,
            from_currency=proto_order.from_currency,
            to_currency=proto_order.to_currency
        )


    async def wait_for_orders(node, expected_count, max_retries=15, delay=2):
        for _ in range(max_retries):
            orders = node['exchange'].order_book.get_orders()
            print(f"Checking orders on node {node['node_id']}, attempt {_ + 1}: {orders}")
            if len(orders) == expected_count:
                return
            await asyncio.sleep(delay)
        raise AssertionError(f"Expected {expected_count} orders, but got {len(orders)}")


    async def _sync_with_single_node(self, remote_state):
        try:
            local_orders = self.order_book.get_orders()
            remote_orders = self._orders_from_remote_state(remote_state)

            # Add orders from remote that are not in local
            for remote_order in remote_orders:
                if remote_order not in local_orders:
                    self.order_book.add_order(remote_order)

            # Remove orders from local that are not in remote
            local_order_ids = {order.id for order in local_orders}
            remote_order_ids = {order.id for order in remote_orders}
            orders_to_cancel = local_order_ids - remote_order_ids
            for order_id in orders_to_cancel:
                self.order_book.cancel_order(order_id)

        except Exception as e:
            logger.error(f"Error in _sync_with_single_node: {str(e)}")

    def _orders_from_remote_state(self, remote_state):
        # Convert the remote state to a list of Order objects
        # This method will depend on how the remote state is structured
        # For example:
        return [Order(**order_data) for order_data in remote_state.orders]

    def get_orders(self):
        try:
            orders = self.order_book.get_orders()
            logger.info(f"Retrieved orders: {orders}")
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []

    def get_order(self, order_id: str):
        try:
            order = self.order_book.get_order(order_id)
            logger.info(f"Retrieved order: {order}")
            return order
        except Exception as e:
            logger.error(f"Error getting order: {str(e)}")
            return None

    async def cancel_order_with_retry(self, user_id: str, order_id: str, max_retries=3):
        for attempt in range(max_retries):
            result = await self.cancel_order(user_id, order_id)
            if result["status"] == "success":
                return result
            logger.warning(f"Cancel order attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(0.1)
        logger.error(f"Failed to cancel order after {max_retries} attempts")
        return result
    async def place_order(self, order_data):
        try:
            order = Order(**order_data)
            self.order_book.add_order(order)
            print(f"Order placed: {order}")
            await self.propagate_order(order)
            return {"status": "success", "order_id": str(order.id)}
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def cancel_order(self, user_id: str, order_id: str) -> Dict[str, str]:
        try:
            order = self.order_book.get_order(order_id)
            if order and order.user_id == user_id:
                if self.order_book.cancel_order(order_id):
                    print(f"Order cancelled: {order_id}")
                    return {"status": "success"}
            print(f"Order not found or unauthorized: {order_id}")
            return {"status": "failure", "message": "Order not found or unauthorized"}
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return {"status": "error", "message": str(e)}


    async def swap(self, user: str, pair: str, amount_in: Decimal, min_amount_out: Decimal) -> Decimal:
        if pair not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")
        
        pool = self.liquidity_pools[pair]
        amount_out = await pool.swap(amount_in, min_amount_out)
        
        base_currency, quote_currency = pair.split('_')
        await self.blockchain.transfer(user, pool.address, amount_in, base_currency)
        await self.blockchain.transfer(pool.address, user, amount_out, quote_currency)
        
        return amount_out

    async def add_liquidity(self, user_id: str, pool_id: str, amount_a: Decimal, amount_b: Decimal):
        if pool_id not in self.liquidity_pools:
            token_a, token_b = pool_id.split('_')
            self.liquidity_pools[pool_id] = EnhancedLiquidityPool(token_a, token_b, self.fee_percent)
        pool = self.liquidity_pools[pool_id]
        liquidity_minted = await pool.add_liquidity(user_id, amount_a, amount_b)
        await self.blockchain.transfer(user_id, pool.address, amount_a, pool.token_a)
        await self.blockchain.transfer(user_id, pool.address, amount_b, pool.token_b)
        return liquidity_minted

    async def remove_liquidity(self, user_id: str, pool_id: str, amount: Decimal):
        if pool_id not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")
        pool = self.liquidity_pools[pool_id]
        amount_a, amount_b = await pool.remove_liquidity(user_id, amount)
        await self.blockchain.transfer(pool.address, user_id, amount_a, pool.token_a)
        await self.blockchain.transfer(pool.address, user_id, amount_b, pool.token_b)
        return amount_a, amount_b

    async def lend(self, user: str, currency: str, amount: Decimal):
        if currency not in self.lending_pools:
            self.lending_pools[currency] = LendingPool(currency)
        
        pool = self.lending_pools[currency]
        await pool.lend(user, amount)
        await self.blockchain.transfer(user, pool.address, amount, currency)

    async def open_margin_position(self, user: str, pair: str, side: str, amount: Decimal, leverage: int):
        if user not in self.margin_accounts:
            self.margin_accounts[user] = MarginAccount(user)
        
        account = self.margin_accounts[user]
        await account.open_position(pair, side, amount, leverage, self.price_oracle)
        position_key = (user, pair)
        self.margin_positions[position_key] = {
            "position_type": side,
            "entry_price": await self.price_oracle.get_price(pair),
            "amount": amount,
            "leverage": leverage
        }

    async def wait_for_orders(self, node, expected_count, max_retries=5, delay=1):
        for _ in range(max_retries):
            orders = node['exchange'].order_book.get_orders()
            if len(orders) == expected_count:
                return
            await asyncio.sleep(delay)
        raise AssertionError(f"Expected {expected_count} orders, but got {len(orders)}")

    async def place_limit_order(self, user: str, order_type: str, pair: str, amount: Decimal, price: Decimal) -> str:
        order = Order(
            user_id=user,
            type='limit',
            order_type=order_type,
            pair=pair,
            amount=amount,
            price=price,
            from_currency=pair.split('_')[1],
            to_currency=pair.split('_')[0]
        )
        
        if order_type == "buy":
            required_balance = amount * price
            balance = await self.blockchain.get_balance(user, pair.split('_')[1])
        else:
            required_balance = amount
            balance = await self.blockchain.get_balance(user, pair.split('_')[0])

        if balance < required_balance:
            raise ValueError("Insufficient balance")

        self.order_book.add_order(order)
        await self.match_orders(pair)
        return order.id
    async def create_liquidity_pool(self, user: str, token_a: str, token_b: str, amount_a: Decimal, amount_b: Decimal):
        pair = f"{token_a}_{token_b}"
        if pair in self.liquidity_pools:
            raise ValueError("Liquidity pool already exists")

        self.liquidity_pools[pair] = EnhancedLiquidityPool(token_a=token_a, token_b=token_b, fee_percent=Decimal('0.003'))
        print(f"Liquidity pool created for {pair}")
        await self.add_liquidity(user, pair, amount_a, amount_b)

        return f"Liquidity pool created for {pair}"

    async def swap(self, user: str, pair: str, amount_in: Decimal, min_amount_out: Decimal) -> Decimal:
        if pair not in self.liquidity_pools:
            print(f"Liquidity pool {pair} does not exist")
            raise ValueError("Liquidity pool does not exist")
        
        pool = self.liquidity_pools[pair]
        amount_out = await pool.swap(amount_in, min_amount_out)
        
        base_currency, quote_currency = pair.split('_')
        await self.blockchain.transfer(user, pool.address, amount_in, base_currency)
        await self.blockchain.transfer(pool.address, user, amount_out, quote_currency)
        
        return amount_out


    async def place_stop_loss_order(self, user: str, pair: str, amount: Decimal, stop_price: Decimal):
        return await self.advanced_order_types.place_stop_loss_order(self, user, pair, amount, stop_price)

    async def execute_quantum_hedging(self, user: str, pair: str, amount: Decimal):
        return await self.quantum_features.quantum_hedging(self, user, pair, amount)

    async def place_take_profit_order(self, user: str, pair: str, amount: Decimal, take_profit_price: Decimal):
        return await self.advanced_order_types.place_take_profit_order(self, user, pair, amount, take_profit_price)

    async def place_trailing_stop_order(self, user: str, pair: str, amount: Decimal, trail_percent: Decimal):
        return await self.advanced_order_types.place_trailing_stop_order(self, user, pair, amount, trail_percent)

    async def execute_entanglement_arbitrage(self, user: str, pairs: List[str]):
        return await self.quantum_features.entanglement_based_arbitrage(self, user, pairs)

    async def get_pool_info(self, pair: str):
        if pair not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")

        pool = self.liquidity_pools[pair]
        return {
            "reserve_a": pool.balance_a,
            "reserve_b": pool.balance_b,
            "total_supply": pool.total_shares,
            "fee_percent": self.fee_percent
        }

    async def get_user_liquidity_balance(self, user: str, pair: str):
        if pair not in self.liquidity_pools:
            raise ValueError("Liquidity pool does not exist")

        pool = self.liquidity_pools[pair]
        return pool.get_user_share(user)

    async def get_exchange_stats(self):
        total_liquidity = sum(pool.balance_a + pool.balance_b for pool in self.liquidity_pools.values())
        total_lending = sum(pool.total_supplied for pool in self.lending_pools.values())
        total_borrowing = sum(pool.total_borrowed for pool in self.lending_pools.values())

        return {
            "total_liquidity": total_liquidity,
            "total_lending": total_lending,
            "total_borrowing": total_borrowing,
            "active_pairs": list(self.liquidity_pools.keys()),
            "lending_currencies": list(self.lending_pools.keys())
        }

    async def flash_loan(self, user: str, currency: str, amount: Decimal):
        if currency not in self.lending_pools:
            raise ValueError("Lending pool does not exist")

        pool = self.lending_pools[currency]
        if amount > pool.total_supplied - pool.total_borrowed:
            raise ValueError("Insufficient liquidity for flash loan")

        await self.blockchain.transfer(pool.address, user, amount, currency)

        # Assuming get_balance returns Decimal directly
        user_balance = await self.blockchain.get_balance(user, currency)
        return "Flash loan executed successfully"

    async def close_margin_position(self, user: str, pair: str) -> Decimal:
        position_key = (user, pair)
        if position_key not in self.margin_positions:
            raise ValueError("No open margin position for user and pair")

        position = self.margin_positions[position_key]
        exit_price = await self.price_oracle.get_price(pair)

        if position["position_type"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["amount"] * position["leverage"]
        elif position["position_type"] == "short":
            pnl = (position["entry_price"] - exit_price) * position["amount"] * position["leverage"]
        else:
            raise ValueError("Invalid position type")

        del self.margin_positions[position_key]

        return pnl

    async def place_market_order(self, user: str, order_type: str, pair: str, amount: Decimal):
        order_id = str(uuid.uuid4())
        # Assign a dummy positive price for validation purposes
        price = Decimal('1') if order_type == 'buy' else Decimal('1')
        order = Order(
            order_id=order_id,
            user_id=user,
            type='market',
            order_type=order_type,
            pair=pair,
            base_currency=pair.split('_')[0],
            quote_currency=pair.split('_')[1],
            amount=amount,
            price=price,
            from_currency=pair.split('_')[1],
            to_currency=pair.split('_')[0]
        )
        self.order_book.add_order(order)
        return order_id

    async def repay(self, user: str, currency: str, amount: Decimal):
        if currency not in self.lending_pools:
            raise ValueError("Lending pool does not exist")

        pool = self.lending_pools[currency]
        collateral_released = await pool.repay(user, amount)
        await self.blockchain.transfer(user, pool.address, amount, currency)
        await self.blockchain.transfer(pool.address, user, collateral_released, pool.get_collateral_currency(user))

    async def propagate_order(self, order):
        nodes = self.node_directory.discover_nodes()
        propagation_tasks = []

        for node in nodes:
            if node['is_active']:
                task = asyncio.create_task(self._propagate_order_to_node(node, order))
                propagation_tasks.append(task)

        # Wait for all propagation tasks to complete
        results = await asyncio.gather(*propagation_tasks, return_exceptions=True)

        # Process results
        successful_propagations = sum(1 for result in results if not isinstance(result, Exception))
        logger.info(f"Order propagated to {successful_propagations}/{len(nodes)} active nodes")

    async def _propagate_order_to_node(self, node, order):
        try:
            async with grpc.aio.insecure_channel(f"{node['ip_address']}:{node['port']}") as channel:
                stub = dagknight_pb2_grpc.DAGKnightStub(channel)
                request = dagknight_pb2.PropagateOrderRequest(
                    user_id=order.user_id,
                    type=order.type,
                    order_type=order.order_type,
                    pair=order.pair,
                    amount=str(order.amount),
                    price=str(order.price),
                    base_currency=order.base_currency,
                    quote_currency=order.quote_currency,
                    from_currency=order.from_currency,
                    to_currency=order.to_currency
                )
                response = await stub.PropagateOrder(request)
                if response.success:
                    logger.info(f"Order successfully propagated to node {node['node_id']}")
                else:
                    logger.warning(f"Failed to propagate order to node {node['node_id']}: {response.message}")
        except grpc.RpcError as rpc_error:
            logger.error(f"RPC error when propagating order to node {node['node_id']}: {rpc_error.code()}: {rpc_error.details()}")
        except Exception as e:
            logger.error(f"Unexpected error when propagating order to node {node['node_id']}: {str(e)}")

    async def get_orders(self) -> List[Order]:
        return self.order_book.get_orders()
    async def get_tradable_assets(self) -> List[str]:
        """
        Returns a list of tradable assets on the exchange.
        This includes all tokens that have active liquidity pools and supported trading pairs.
        """
        try:
            tradable_assets = set()

            # Add assets from active liquidity pools
            for pool_id, pool in self.liquidity_pools.items():
                if '_' in pool_id:
                    tokens = pool_id.split('_')
                    if len(tokens) == 2:
                        token_a, token_b = tokens
                        if token_a:
                            tradable_assets.add(token_a)
                        if token_b:
                            tradable_assets.add(token_b)
                    else:
                        logger.warning(f"Skipping invalid pool_id: '{pool_id}'")
                else:
                    logger.warning(f"Skipping invalid pool_id: '{pool_id}'")

            # Add assets from supported lending pools
            for currency in self.lending_pools.keys():
                if currency:
                    tradable_assets.add(currency)

            # Return the list of tradable assets
            return list(tradable_assets)

        except Exception as e:
            logger.error(f"Error in get_tradable_assets: {e}")
            logger.error(traceback.format_exc())
            return []

    async def get_order_book(self, pair):
        return self.order_book.get(pair, {"bids": [], "asks": []})

    async def get_current_price(self, pair):
        # Retrieve the latest price from the price oracle
        return self.price_oracle.get_price(pair)

    async def place_limit_order(self, user, order_type, pair, amount, price):
        order_id = f"{user}_{pair}_{order_type}_{amount}_{price}"
        # Logic to place the order in the order book
        if order_type == "buy":
            self.order_book[pair]["bids"].append((price, amount))
        elif order_type == "sell":
            self.order_book[pair]["asks"].append((price, amount))
        return order_id

    async def get_swap_estimate(self, from_token, to_token, amount):
        # Estimate the output based on current reserves and prices
        rate = self.price_oracle.get_price(f"{from_token}/{to_token}")
        return amount * rate
    async def swap_tokens(self, user_id, sell_token, buy_token, sell_amount):
        """
        Swap tokens using Plata exchange, 0x API, or BlockCypher.
        """
        try:
            # Validate inputs
            if not all([user_id, sell_token, buy_token, sell_amount]):
                raise ValueError("Invalid input parameters provided for token swap.")

            user_wallet = self.get_user_wallet(user_id)
            if not user_wallet or 'address' not in user_wallet:
                raise ValueError(f"Wallet not found for user_id: {user_id}")
            
            taker_address = user_wallet['address']

            # Fetch a firm quote from 0x API
            logging.info(f"Fetching quote for {sell_token} to {buy_token}...")
            quote = self.zero_x_api.get_quote(sell_token, buy_token, sell_amount, taker_address)
            if not quote:
                raise ValueError("Failed to fetch quote from 0x API.")

            # If necessary, set token allowance
            logging.info(f"Setting token allowance for {sell_token}...")
            allowance_tx = self.zero_x_api.set_token_allowance(sell_token, quote.get('allowanceTarget', ''), sell_amount, user_wallet)
            if allowance_tx:
                logging.info(f"Allowance transaction hash: {allowance_tx}")

            # Sign the Permit2 message
            logging.info("Signing Permit2 message...")
            signature = self.zero_x_api.sign_permit2_message(quote.get('permit2', {}).get('eip712', {}), user_wallet)
            if not signature:
                raise ValueError("Failed to sign Permit2 message.")

            # Submit the transaction
            logging.info("Submitting transaction to 0x API...")
            tx_hash = self.zero_x_api.submit_transaction(quote, signature, user_wallet)
            if not tx_hash:
                raise ValueError("Failed to submit transaction.")

            logging.info(f"Transaction successful with hash: {tx_hash}")
            return {
                'tx_hash': tx_hash,
                'amount_out': Decimal(quote.get('buyAmount', '0')) / Decimal(10 ** 18),  # Assuming 18 decimals for tokens
                'status': 'success'
            }

        except KeyError as e:
            logging.error(f"KeyError during token swap: {e}")
            return {'status': 'error', 'message': f"Missing data: {str(e)}"}
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            return {'status': 'error', 'message': str(e)}
        except Exception as e:
            logging.error(f"Unexpected error during token swap: {e}")
            return {'status': 'error', 'message': "An unexpected error occurred"}


from decimal import Decimal
from typing import Dict, List, Tuple, Any
from blockcypher_integration import BlockCypherAPI
from zero_x_swap import ZeroXSwapAPI
from wallet_registration import WalletRegistration
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from SecureHybridZKStark import SecureHybridZKStark, calculate_security_parameters
import os
import asyncio


class EnhancedExchangeWithCryptoAndZKP(EnhancedExchange):
    def __init__(self, blockchain, vm, price_oracle, node_directory):
        super().__init__(blockchain, vm, price_oracle, node_directory)
        load_dotenv()

        # Load and decrypt the API keys
        with open("encryption_key.key", "rb") as key_file:
            encryption_key = key_file.read()
        f = Fernet(encryption_key)
        decrypted_zero_x_api_key = f.decrypt(os.getenv("ZEROX_API_KEY").encode()).decode()
        decrypted_blockcypher_api_key = f.decrypt(os.getenv("BLOCKCYPHER_API_KEY").encode()).decode()
        decrypted_alchemy_key = f.decrypt(os.getenv("ALCHEMY_KEY").encode()).decode()

        # Initialize the APIs
        self.zero_x_api = ZeroXSwapAPI(decrypted_zero_x_api_key)
        self.blockcypher_api = BlockCypherAPI(decrypted_blockcypher_api_key)
        self.wallet_registration = WalletRegistration(decrypted_blockcypher_api_key, decrypted_alchemy_key)

        # Initialize ZKP components
        security_params = calculate_security_parameters()
        self.zkp_system = SecureHybridZKStark(security_params['field_size'])

    async def register_user_wallets(self, user_id: str):
        """Register all necessary wallets for a user."""
        return self.wallet_registration.register_all_wallets(user_id)

    async def get_btc_address(self, user_id: str):
        """Get or generate a Bitcoin address for a user."""
        user_wallets = self.fake_users_db.get(user_id, {}).get('wallets', {})
        if 'bitcoin' not in user_wallets:
            btc_wallet = self.blockcypher_api.generate_address()
            user_wallets['bitcoin'] = btc_wallet
            self.fake_users_db[user_id]['wallets'] = user_wallets
        return user_wallets['bitcoin']['address']

    async def get_btc_balance(self, address: str):
        """Get the Bitcoin balance for a given address."""
        address_info = self.blockcypher_api.get_address_info(address)
        balance_satoshis = address_info.get('balance', 0)
        return Decimal(balance_satoshis) / Decimal(100000000)  # Convert satoshis to BTC

    async def send_btc(self, from_user_id: str, to_address: str, amount: Decimal):
        """Send Bitcoin from a user's address to another address."""
        from_address = await self.get_btc_address(from_user_id)
        amount_satoshis = int(amount * Decimal(100000000))  # Convert BTC to satoshis
        inputs = [{"address": from_address}]
        outputs = [{"address": to_address, "value": amount_satoshis}]
        
        unsigned_tx = self.blockcypher_api.create_transaction(inputs, outputs)
        # Note: In a real-world scenario, you would need to sign this transaction
        # with the user's private key. For demonstration, we're skipping this step.
        signed_tx = unsigned_tx  # Placeholder for signed transaction
        
        sent_tx = self.blockcypher_api.send_transaction(signed_tx)
        return sent_tx.get('tx', {}).get('hash')

    async def setup_btc_webhook(self, user_id: str, callback_url: str):
        """Set up a webhook for Bitcoin transactions for a user's address."""
        address = await self.get_btc_address(user_id)
        return self.blockcypher_api.setup_webhook("tx-confirmation", callback_url, address)

    async def swap_tokens(self, user_id: str, sell_token: str, buy_token: str, sell_amount: Decimal):
        """Swap tokens using 0x API with ZKP."""
        user_wallet = self.get_user_wallet(user_id)
        taker_address = user_wallet['ethereum']['address']

        quote = self.zero_x_api.get_quote(sell_token, buy_token, str(sell_amount), taker_address)

        # Generate ZKP for the swap
        secret = int(sell_amount * 10**18)  # Convert to integer
        public_input = int.from_bytes(sell_token.encode(), 'big')  # Use sell_token as public input
        zkp = self.zkp_system.prove(secret, public_input)

        # Set token allowance if necessary
        allowance_tx = self.zero_x_api.set_token_allowance(
            sell_token, 
            quote.get('allowanceTarget', ''), 
            sell_amount, 
            user_wallet['ethereum']
        )
        if allowance_tx:
            await self.wait_for_transaction(allowance_tx)

        # Sign the Permit2 message
        signature = self.zero_x_api.sign_permit2_message(
            quote.get('permit2', {}).get('eip712', {}), 
            user_wallet['ethereum']
        )

        # Submit the swap transaction
        tx_hash = self.zero_x_api.submit_transaction(quote, signature, user_wallet['ethereum'])
        if not tx_hash:
            raise ValueError("Failed to submit swap transaction.")

        # Wait for the transaction to be confirmed
        tx_receipt = await self.wait_for_transaction(tx_hash)

        # Verify the ZKP
        is_valid = self.zkp_system.verify(public_input, zkp)
        if not is_valid:
            raise ValueError("ZKP verification failed for the swap.")

        return tx_receipt, zkp

    async def wait_for_transaction(self, tx_hash: str, max_attempts: int = 50, delay: int = 5):
        """Wait for a transaction to be confirmed."""
        for _ in range(max_attempts):
            tx_receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            if tx_receipt and tx_receipt['status'] == 1:
                return tx_receipt
            await asyncio.sleep(delay)
        raise TimeoutError(f"Transaction {tx_hash} was not confirmed after {max_attempts * delay} seconds")

    async def get_token_balance(self, user_id: str, token_address: str, chain: str = 'ethereum'):
        """Get the balance of a specific token for a user."""
        user_wallet = self.get_user_wallet(user_id)
        address = user_wallet[chain]['address']
        
        # Use Alchemy or a similar provider to get the token balance
        balance = self.w3.eth.contract(address=token_address, abi=ERC20_ABI).functions.balanceOf(address).call()
        decimals = self.w3.eth.contract(address=token_address, abi=ERC20_ABI).functions.decimals().call()
        
        return Decimal(balance) / Decimal(10 ** decimals)

    async def private_transfer(self, sender_id: str, receiver_id: str, amount: Decimal, token: str):
        """Perform a private transfer using ZKP."""
        sender_balance = await self.get_token_balance(sender_id, token)
        if sender_balance < amount:
            raise ValueError("Insufficient balance for transfer.")

        secret = int(amount * 10**18)  # Convert to integer
        public_input = int.from_bytes(token.encode(), 'big')  # Use token as public input
        zkp = self.zkp_system.prove(secret, public_input)

        # Perform the transfer (in a real scenario, this would update balances in a confidential manner)
        # For demonstration, we're just updating the balances directly
        await self._update_balance(sender_id, token, -amount)
        await self._update_balance(receiver_id, token, amount)

        # Verify the ZKP
        is_valid = self.zkp_system.verify(public_input, zkp)
        if not is_valid:
            raise ValueError("ZKP verification failed for the transfer.")

        return zkp

    async def _update_balance(self, user_id: str, token: str, amount: Decimal):
        """Update the balance of a user (for demonstration purposes)."""
        if user_id not in self.fake_users_db:
            self.fake_users_db[user_id] = {'balances': {}}
        if token not in self.fake_users_db[user_id]['balances']:
            self.fake_users_db[user_id]['balances'][token] = Decimal('0')
        self.fake_users_db[user_id]['balances'][token] += amount

        
from decimal import Decimal
from typing import Dict, List, Tuple, Any
from SecureHybridZKStark import SecureHybridZKStark
from P2PNode import P2PNode, Message, MessageType
class EnhancedExchangeWithZKStarks(EnhancedExchange):
    def __init__(self, blockchain, vm, price_oracle, node_directory, desired_security_level, host=None, port=None):
        super().__init__(blockchain, vm, price_oracle, node_directory)
        self.p2p_node = P2PNode(host, port, self.blockchain, security_level=desired_security_level)
        self.host = host
        self.port = port

        # Initialize the FiniteField instance using a factory or method
        self.finite_field = FiniteFieldFactory.get_instance()
        
        # Ensure the desired security level is an integer
        if not isinstance(desired_security_level, int):
            raise TypeError("desired_security_level must be an integer")
        
        # Set the security level to the provided integer
        self.security_level = desired_security_level
        
        # Initialize the SecureHybridZKStark system with the same finite field
        self.zk_system = SecureHybridZKStark(security_level=self.security_level, field=self.finite_field)
        
        # Initialize private balances as a dictionary, mapping user IDs to their respective balance dictionaries
        self.private_balances: Dict[str, Dict[str, int]] = {}
     

    async def start_p2p(self):
        from P2PNode import P2PNode  # Move the import here

        await self.p2p_node.start()


    async def private_deposit(self, user: str, currency: str, amount: Decimal):
        amount_int = int(amount * 10**18)
        if user not in self.private_balances:
            self.private_balances[user] = {}
        if currency not in self.private_balances[user]:
            self.private_balances[user][currency] = 0
        self.private_balances[user][currency] += amount_int

        # Ensure we're using the correct field for hashing
        public_input = self.zk_system.field.element(self.zk_system.stark.hash(user, currency, amount_int))
        
        # Use the same field for both STARK and SNARK
        stark_proof = self.zk_system.stark.prove(self.zk_system.field.element(amount_int), public_input)
        snark_proof = self.zk_system.snark.prove(amount_int, public_input.value)

        return stark_proof, snark_proof
                
    async def private_transfer(self, sender: str, receiver: str, currency: str, amount: Decimal) -> Tuple[Tuple, Tuple]:
        amount_int = int(amount * 10**18)
        if self.private_balances.get(sender, {}).get(currency, 0) < amount_int:
            raise ValueError("Insufficient private balance")

        self.private_balances[sender][currency] -= amount_int
        if receiver not in self.private_balances:
            self.private_balances[receiver] = {}
        if currency not in self.private_balances[receiver]:
            self.private_balances[receiver][currency] = 0
        self.private_balances[receiver][currency] += amount_int

        public_input = self.zk_system.stark.hash(sender, receiver, currency, amount_int)
        proof = self.zk_system.prove(amount_int, public_input)

        # In a real implementation, you would publish this proof to the blockchain
        return proof

    async def verify_private_transaction(self, proof: Tuple[Tuple, Tuple], public_input: int) -> bool:
        return self.zk_system.verify(public_input, proof)

    async def get_private_balance(self, user: str, currency: str) -> Decimal:
        balance_int = self.private_balances.get(user, {}).get(currency, 0)
        return Decimal(balance_int) / Decimal(10**18)

    # Overriding necessary methods to include privacy features

    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, str]:
        result = await super().place_order(order_data)
        if result['status'] == 'success':
            proof = self.zk_system.prove(int(order_data['amount'] * 10**18), self.zk_system.stark.hash(order_data['user_id'], order_data['pair'], int(order_data['amount'] * 10**18)))
            result['zk_proof'] = proof
            
            # Propagate the order with ZK proof
            await self.p2p_node.broadcast({
                'type': MessageType.PLACE_ORDER.value,
                'payload': {
                    'order': order_data,
                    'zk_proof': proof
                }
            })
        return result

    async def cancel_order(self, user_id: str, order_id: str) -> Dict[str, str]:
        result = await super().cancel_order(user_id, order_id)
        if result['status'] == 'success':
            proof = self.zk_system.prove(int(order_id, 16), self.zk_system.stark.hash(user_id, order_id))
            result['zk_proof'] = proof
            
            # Propagate the cancellation with ZK proof
            await self.p2p_node.broadcast({
                'type': MessageType.CANCEL_ORDER.value,
                'payload': {
                    'user_id': user_id,
                    'order_id': order_id,
                    'zk_proof': proof
                }
            })
        return result


    async def add_liquidity(self, user_id: str, pool_id: str, amount_a: Decimal, amount_b: Decimal):
        result = await super().add_liquidity(user_id, pool_id, amount_a, amount_b)
        # Generate a zero-knowledge proof of liquidity addition
        proof = self.zk_system.prove(int((amount_a + amount_b) * 10**18), self.zk_system.stark.hash(user_id, pool_id, int(amount_a * 10**18), int(amount_b * 10**18)))
        return result, proof

    async def remove_liquidity(self, user_id: str, pool_id: str, amount: Decimal):
        result = await super().remove_liquidity(user_id, pool_id, amount)
        # Generate a zero-knowledge proof of liquidity removal
        proof = self.zk_system.prove(int(amount * 10**18), self.zk_system.stark.hash(user_id, pool_id, int(amount * 10**18)))
        return result, proof

    async def swap(self, user: str, pair: str, amount_in: Decimal, min_amount_out: Decimal) -> Tuple[Decimal, Tuple[Tuple, Tuple]]:
        amount_out = await super().swap(user, pair, amount_in, min_amount_out)
        # Generate a zero-knowledge proof of swap
        proof = self.zk_system.prove(int(amount_in * 10**18), self.zk_system.stark.hash(user, pair, int(amount_in * 10**18), int(amount_out * 10**18)))
        return amount_out, proof

    async def open_margin_position(self, user: str, pair: str, side: str, amount: Decimal, leverage: int):
        result = await super().open_margin_position(user, pair, side, amount, leverage)
        # Generate a zero-knowledge proof of margin position opening
        proof = self.zk_system.prove(int(amount * 10**18), self.zk_system.stark.hash(user, pair, side, int(amount * 10**18), leverage))
        return result, proof

    async def close_margin_position(self, user: str, pair: str) -> Tuple[Decimal, Tuple[Tuple, Tuple]]:
        pnl = await super().close_margin_position(user, pair)
        # Generate a zero-knowledge proof of margin position closing
        proof = self.zk_system.prove(int(pnl * 10**18), self.zk_system.stark.hash(user, pair, int(pnl * 10**18)))
        return pnl, proof
    async def get_tradable_assets(self) -> List[str]:
        """
        Returns a list of tradable assets on the exchange.
        This includes all tokens that have active liquidity pools and supported trading pairs.
        """
        try:
            tradable_assets = set()

            # Add assets from active liquidity pools
            for pool_id, pool in self.liquidity_pools.items():
                if '_' in pool_id:
                    tokens = pool_id.split('_')
                    if len(tokens) == 2:
                        token_a, token_b = tokens
                        if token_a:
                            tradable_assets.add(token_a)
                        if token_b:
                            tradable_assets.add(token_b)
                    else:
                        logger.warning(f"Skipping invalid pool_id: '{pool_id}'")
                else:
                    logger.warning(f"Skipping invalid pool_id: '{pool_id}'")

            # Add assets from supported lending pools
            for currency in self.lending_pools.keys():
                if currency:
                    tradable_assets.add(currency)

            # Return the list of tradable assets
            return list(tradable_assets)

        except Exception as e:
            logger.error(f"Error in get_tradable_assets: {e}")
            logger.error(traceback.format_exc())
            return []
            
    async def get_orders(self):
        return self.order_book.get_orders()

    async def place_limit_order(self, user: str, order_type: str, pair: str, amount: Decimal, price: Decimal):
        # Implementation details depend on your specific order book logic
        order = Order(user_id=user, type='limit', order_type=order_type, pair=pair, amount=amount, price=price)
        self.order_book.add_order(order)
        return order.id

    async def cancel_order(self, user: str, order_id: str):
        # Implementation details depend on your specific order book logic
        success = self.order_book.cancel_order(order_id)
        return {"status": "success" if success else "failure"}

    async def add_liquidity(self, user: str, pool_id: str, amount_a: Decimal, amount_b: Decimal):
        # Implementation details depend on your liquidity pool logic
        if pool_id not in self.liquidity_pools:
            self.liquidity_pools[pool_id] = LiquidityPool(pool_id.split('_')[0], pool_id.split('_')[1])
        pool = self.liquidity_pools[pool_id]
        return await pool.add_liquidity(user, amount_a, amount_b)

    async def get_tradable_assets(self):
        # Return a list of all tradable assets
        assets = set()
        for pool in self.liquidity_pools.values():
            assets.add(pool.token_a)
            assets.add(pool.token_b)
        return list(assets)

    # Other methods from EnhancedExchange remain unchanged       

class SimpleStateSynchronization:
    @staticmethod
    def compute_state_hash(orders):
        order_data = [order.to_json_serializable() for order in orders]
        state_json = json.dumps(order_data, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()

    @staticmethod
    async def sync_states(local_exchange, remote_exchange):
        local_orders = local_exchange.order_book.get_orders()
        remote_orders = remote_exchange.order_book.get_orders()

        local_hash = SimpleStateSynchronization.compute_state_hash(local_orders)
        remote_hash = SimpleStateSynchronization.compute_state_hash(remote_orders)

        if local_hash != remote_hash:
            # Synchronize the states
            for order in remote_orders:
                if order not in local_orders:
                    local_exchange.order_book.add_order(order)

            local_order_ids = {order.id for order in local_orders}
            remote_order_ids = {order.id for order in remote_orders}
            orders_to_cancel = local_order_ids - remote_order_ids

            for order_id in orders_to_cancel:
                local_exchange.order_book.cancel_order(order_id)

        return local_hash == remote_hash

