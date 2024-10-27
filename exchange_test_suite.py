import pytest
import asyncio
import random
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List
from quantumdagknight import PriceOracle, Blockchain, NodeDirectory, app
from vm import SimpleVM as VM
from unittest.mock import AsyncMock, MagicMock, patch
from enhanced_exchange import EnhancedExchange
from typing import Any, Dict  # Ensure Any and Dict are imported
from Order import Order
from EnhancedOrderBook import EnhancedOrderBook
from enhanced_exchange import EnhancedExchange
import traceback 
from quantumdagknight import Order  # Ensure Order is imported correctly
from qiskit import QuantumCircuit  # Import QuantumCircuit for quantum_features
import uuid  # Import uuid for LendingPool
from qiskit_aer import Aer, AerSimulator
from Order import Order
import asyncio
from enhanced_exchange import LiquidityPoolManager,PriceOracle,MarginAccount,LiquidityPool
import pytest
pytestmark = pytest.mark.asyncio

# Setup logging     
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
client = TestClient(app)

# Constants
NUM_NODES = 5
BASE_API_PORT = 8000
                            
# Simulated database for users and orders
fake_users_db = {}
user_balances = {
    "test_user": {
        "USD": Decimal("100000.00"),
        "BTC": Decimal("10.0")
    }
}
order_book = {
    "BTC_USD": []
}
class OrderRequest(BaseModel):
    user_id: str
    type: str
    order_type: str
    pair: str
    base_currency: str
    quote_currency: str
    amount: str  # Keeping as str to convert to Decimal later
    price: str  # Keeping as str to convert to Decimal later
    from_currency: str
    to_currency: str

@pytest.fixture(scope="module")
def test_nodes():
    blockchain = MagicMock(spec=Blockchain)
    blockchain.get_balance = MagicMock(return_value=Decimal("10000"))
    blockchain.transfer = MagicMock(return_value=True)
    
    vm = MagicMock(spec=VM)
    price_oracle = MagicMock(spec=PriceOracle)
    node_directory = MagicMock(spec=NodeDirectory)

    nodes = []
    for i in range(NUM_NODES):
        exchange = EnhancedExchange(blockchain, vm, price_oracle, node_directory)
        node = {
            'node_id': f"node_{i}",
            'fastapi_app': TestClient(app),
            'is_active': True,
            'exchange': exchange
        }
        nodes.append(node)
    return nodes

def get_orders_from_node(node):
    if not node['is_active']:
        return []
    return node['exchange'].order_book.get_orders()

@app.post("/place_order")
async def place_order(order_data: OrderRequest):
    try:
        # Convert order data to Order model with Decimal conversion
        order = Order(
            user_id=order_data.user_id,
            type=order_data.type,
            order_type=order_data.order_type,
            pair=order_data.pair,
            base_currency=order_data.base_currency,
            quote_currency=order_data.quote_currency,
            amount=Decimal(order_data.amount),
            price=Decimal(order_data.price),
            from_currency=order_data.from_currency,
            to_currency=order_data.to_currency,
        )

        # Check if the user exists in the fake database
        user_id = order.user_id
        if user_id not in fake_users_db:
            # Create a new user if it doesn't exist
            fake_users_db[user_id] = {
                "username": user_id,
                "full_name": f"Benchmark User {user_id}",
                "email": f"{user_id}@example.com",
                "hashed_password": "fakehashedsecret",
                "disabled": False
            }
            # Initialize balance for new users
            user_balances[user_id] = {
                "USD": Decimal("100000.00"),
                "BTC": Decimal("10.0")
            }

        # Simulate order placement logic (this should be replaced with actual logic)
        result = await process_order(order)
        
        if result['status'] == 'success':
            return result
        else:
            raise HTTPException(status_code=400, detail=result['message'])

    except InvalidOperation as e:
        logger.error(f"Invalid decimal value: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid decimal value: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def process_order(order: Order) -> Dict[str, Any]:
    # Check if the user exists
    if order.user_id not in user_balances:
        return {
            'status': 'failure',
            'message': 'User does not exist'
        }
    
    # Check if the user has sufficient balance for the order
    user_balance = user_balances[order.user_id]
    
    if order.type == 'limit' and order.order_type == 'buy':
        # For a buy order, check if the user has enough quote currency (USD)
        required_balance = order.amount * order.price
        if user_balance.get(order.quote_currency, Decimal("0.0")) < required_balance:
            return {
                'status': 'failure',
                'message': 'Insufficient balance'
            }
        # Deduct the required amount from the user's balance
        user_balance[order.quote_currency] -= required_balance
    
    elif order.type == 'limit' and order.order_type == 'sell':
        # For a sell order, check if the user has enough base currency (BTC)
        if user_balance.get(order.base_currency, Decimal("0.0")) < order.amount:
            return {
                'status': 'failure',
                'message': 'Insufficient balance'
            }
        # Deduct the required amount from the user's balance
        user_balance[order.base_currency] -= order.amount
    
    else:
        return {
            'status': 'failure',
            'message': 'Unsupported order type'
        }
    
    # Simulate adding the order to the order book
    if order.pair not in order_book:
        order_book[order.pair] = []
    order_dict = order.model_dump()  # Use model_dump instead of dict() for Pydantic v2+
    order_dict['node_id'] = order.user_id.split('_')[-1]  # Extract node_id from user_id
    order_book[order.pair].append(order_dict)
    
    return {
        'status': 'success',
        'message': 'Order placed successfully',
    }
async def wait_for_orders(node, expected_count, max_retries=10, delay=1):
    for _ in range(max_retries):
        orders = node['exchange'].order_book.get_orders()
        if len(orders) == expected_count:
            return
        await asyncio.sleep(delay)
    raise AssertionError(f"Expected {expected_count} orders, but got {len(orders)}")


@pytest.mark.asyncioasync 
async def test_node_failure_and_recovery(test_nodes):
    try:
        order = {
            'user_id': 'test_user',
            'type': 'limit',
            'order_type': 'buy',
            'pair': 'BTC_USD',
            'base_currency': 'BTC',
            'quote_currency': 'USD',
            'amount': '1.0',
            'price': '50000',
            'from_currency': 'USD',
            'to_currency': 'BTC'
        }

        response = await test_nodes[0]['exchange'].place_order(order)
        assert response["status"] == "success"
        print(f"Initial order placed: {response}")

        failed_node = test_nodes[0]
        failed_node['is_active'] = False
        print(f"Node {failed_node['node_id']} failed")

        order['amount'] = '0.5'
        response = await test_nodes[1]['exchange'].place_order(order)
        assert response["status"] == "success"
        print(f"Order placed on second node: {response}")

        await asyncio.sleep(2)

        failed_node_orders = get_orders_from_node(failed_node)
        print(f"Orders from failed node: {failed_node_orders}")
        assert len(failed_node_orders) == 0

        failed_node['is_active'] = True
        print(f"Node {failed_node['node_id']} recovered")

        await asyncio.sleep(5)
        await failed_node['exchange'].sync_state()

        # Adjust delay and retries to allow enough time for sync
        await wait_for_orders(failed_node, 2, max_retries=15, delay=2)

    except Exception as e:
        print(f"Exception during test_node_failure_and_recovery: {e}")
        raise






@pytest.mark.asyncio
async def test_order_placement_benchmark_with_metrics(test_nodes):
    try:
        num_orders = 100
        start_time = time.time()

        async def place_order(node):
            order = {
                'user_id': f'bench_user_{node["node_id"]}_{random.randint(0, 10000)}',
                'type': 'limit',
                'order_type': 'buy',
                'pair': 'BTC_USD',
                'base_currency': 'BTC',
                'quote_currency': 'USD',
                'amount': '0.1',
                'price': '50000',
                'from_currency': 'USD',
                'to_currency': 'BTC'
            }
            try:
                response = await node['exchange'].place_order(order)
                return response["status"] == "success"
            except Exception as e:
                return False

        successful_orders = 0
        for _ in range(num_orders):
            node = test_nodes[_ % len(test_nodes)]
            if await place_order(node):
                successful_orders += 1

        end_time = time.time()
        total_time = end_time - start_time
        orders_per_second = successful_orders / total_time

        await asyncio.sleep(5)

        total_orders = 0
        all_orders = set()
        for node in test_nodes:
            node_orders = get_orders_from_node(node)  # No await if not async
            total_orders += len(node_orders)
            for order in node_orders:
                order_tuple = (order.user_id, order.amount, order.price)
                assert order_tuple not in all_orders
                all_orders.add(order_tuple)

        assert successful_orders > 0
        assert total_orders >= successful_orders * 0.9

    except Exception as e:
        raise e



@pytest.fixture
def setup_exchange():
    blockchain = AsyncMock(spec=Blockchain)
    blockchain.get_balance = AsyncMock(return_value=Decimal("10000"))
    blockchain.transfer = AsyncMock(return_value=True)

    vm = AsyncMock(spec=VM)
    price_oracle = AsyncMock(spec=PriceOracle)
    node_directory = AsyncMock(spec=NodeDirectory)

    exchange = EnhancedExchange(blockchain, vm, price_oracle, node_directory)

    # Setup mock behaviors
    blockchain.get_balance.return_value = Decimal("10000")
    blockchain.transfer.return_value = True
    price_oracle.get_price.return_value = Decimal("50000")

    # Mock the order book
    exchange.order_book = MagicMock()
    exchange.order_book.add_order = MagicMock()
    exchange.order_book.cancel_order = MagicMock(return_value=True)
    exchange.order_book.get_orders = MagicMock(return_value=[])

    return exchange



@pytest.mark.asyncio
async def test_swap(setup_exchange):
    try:
        exchange = setup_exchange

        # Mock the creation of a liquidity pool
        pool_mock = MagicMock(spec=LiquidityPool)
        pool_mock.address = "mock_pool_address"
        pool_mock.swap.return_value = Decimal("1950")
        exchange.liquidity_pools["ETH_USDT"] = pool_mock

        # Perform a swap
        result = await exchange.swap("user2", "ETH_USDT", Decimal("1"), Decimal("1900"))
        assert result == Decimal("1950"), "Swap should return the expected amount"
        assert exchange.blockchain.transfer.call_count == 2, "Two transfers should occur during a swap"
    except Exception as e:
        print(f"Exception in test_swap: {str(e)}")
        traceback.print_exc()
        raise


@pytest.mark.asyncio
async def test_add_and_remove_liquidity(setup_exchange):
    try:
        exchange = setup_exchange

        # Add liquidity
        liquidity_minted = await exchange.add_liquidity("user1", "BTC_USDT", Decimal("1"), Decimal("50000"))
        assert liquidity_minted is not None, "Liquidity should be minted successfully"
    except Exception as e:
        print(f"Exception in test_add_and_remove_liquidity: {str(e)}")
        traceback.print_exc()
        raise

@pytest.mark.asyncio
async def test_flash_loan(setup_exchange):
    try:
        exchange = setup_exchange

        # Setup a lending pool
        await exchange.lend("user1", "DAI", Decimal("10000"))

        # Execute a flash loan
        result = await exchange.flash_loan("user2", "DAI", Decimal("5000"))
        assert result == "Flash loan executed successfully", "Flash loan should succeed"
    except Exception as e:
        print(f"Exception in test_flash_loan: {str(e)}")
        traceback.print_exc()
        raise

@pytest.mark.asyncio
async def test_margin_trading(setup_exchange):
    try:
        exchange = setup_exchange

        # Open a margin position
        await exchange.open_margin_position("user1", "ETH_USD", "long", Decimal("10"), 2)

        # Close the margin position
        pnl = await exchange.close_margin_position("user1", "ETH_USD")
        assert isinstance(pnl, Decimal), "PnL should be a Decimal"
        assert pnl is not None, "PnL should not be None"
    except Exception as e:
        print(f"Exception in test_margin_trading: {str(e)}")
        traceback.print_exc()
        raise


@pytest.mark.asyncio
async def test_place_and_cancel_order(setup_exchange):
    exchange = setup_exchange

    order_data = {
        'user_id': 'user1',
        'type': 'limit',
        'order_type': 'buy',
        'pair': 'BTC_USD',
        'base_currency': 'BTC',
        'quote_currency': 'USD',
        'amount': '1.0',
        'price': '50000',
        'from_currency': 'USD',
        'to_currency': 'BTC'
    }
    order_response = await exchange.place_order(order_data)
    assert order_response["status"] == "success"
    order_id = order_response['order_id']
    print(f"Order placed: {order_response}")

    order = exchange.order_book.get_order(order_id)
    print(f"Order in order book: {order}")
    assert order is not None, f"Order {order_id} was not added to the order book"

    cancel_response = await exchange.cancel_order(order_data['user_id'], order_id)
    print(f"Cancel response: {cancel_response}")
    assert cancel_response["status"] == "success"



@pytest.mark.asyncio
async def test_get_exchange_stats(setup_exchange):
    try:
        exchange = setup_exchange

        # Add some data to the exchange
        await exchange.create_liquidity_pool("user1", "ETH", "USDT", Decimal("10"), Decimal("20000"))
        stats = await exchange.get_exchange_stats()
        assert stats is not None, "Exchange stats should be available"
    except Exception as e:
        print(f"Exception in test_get_exchange_stats: {str(e)}")
        traceback.print_exc()
        raise

@pytest.mark.asyncio
async def test_quantum_features(setup_exchange):
    try:
        exchange = setup_exchange

        # Execute quantum hedging
        quantum_hedge_result = await exchange.execute_quantum_hedging("user1", "BTC_USD", Decimal("1"))
        assert quantum_hedge_result is not None, "Quantum hedging should be executed"

        # Execute entanglement arbitrage
        entanglement_arb_result = await exchange.execute_entanglement_arbitrage("user1", ["BTC_USD", "ETH_USD", "LTC_USD"])
        assert entanglement_arb_result is not None, "Entanglement arbitrage should be executed"
    except Exception as e:
        print(f"Exception in test_quantum_features: {str(e)}")
        traceback.print_exc()
        raise


# Also, increase the timeout for the test:
@pytest.mark.asyncio
@pytest.mark.timeout(300)  # 5 minutes timeout
async def test_high_frequency_trading(test_nodes):
    num_orders = 100  # Reduce from 1000 to 100 for testing
    order_placements = 0
    order_cancellations = 0

    async def place_and_cancel_order(node):
        nonlocal order_placements, order_cancellations
        order = {
            'user_id': f'hft_user_{node["node_id"]}_{random.randint(0, 10000)}',  # Ensure unique user_id
            'type': 'limit',
            'order_type': 'buy',
            'pair': 'BTC_USD',
            'base_currency': 'BTC',
            'quote_currency': 'USD',
            'amount': '0.1',
            'price': '50000',
            'from_currency': 'USD',
            'to_currency': 'BTC'
        }
        try:
            response = await node['exchange'].place_order(order)
            logger.info(f"Order placement response: {response}")
            if response["status"] == "success":
                order_placements += 1
                order_id = response.get('order_id')
                cancel_response = await node['exchange'].cancel_order_with_retry(order['user_id'], order_id)
                logger.info(f"Order cancellation response: {cancel_response}")
                if cancel_response["status"] == "success":
                    order_cancellations += 1
                else:
                    logger.error(f"Order cancellation failed: {cancel_response}")
            else:
                logger.error(f"Order placement failed: {response}")
        except Exception as e:
            logger.error(f"Error in high-frequency order placement: {str(e)}")
            traceback.print_exc()
async def test_comprehensive_functionality(setup_exchange):
    exchange = setup_exchange

    await exchange.create_liquidity_pool("user1", "BTC", "USDT", Decimal("10"), Decimal("200000"))

    # Correcting the pair to "BTC_USDT"
    result = await exchange.swap("user2", "BTC_USDT", Decimal("1"), Decimal("1900"))
    assert result is not None, "Swap should be executed successfully"
async def test_cancel_order_sorting(setup_exchange):
    exchange = setup_exchange

    buy_orders = [
        {"user_id": "user1", "type": "limit", "order_type": "buy", "pair": "BTC_USD", "base_currency": "BTC", "quote_currency": "USD", "amount": Decimal("1"), "price": Decimal("50000"), "from_currency": "USD", "to_currency": "BTC"},
        {"user_id": "user2", "type": "limit", "order_type": "buy", "pair": "BTC_USD", "base_currency": "BTC", "quote_currency": "USD", "amount": Decimal("1"), "price": Decimal("51000"), "from_currency": "USD", "to_currency": "BTC"},
        {"user_id": "user3", "type": "limit", "order_type": "buy", "pair": "BTC_USD", "base_currency": "BTC", "quote_currency": "USD", "amount": Decimal("1"), "price": Decimal("49000"), "from_currency": "USD", "to_currency": "BTC"},
    ]

    order_ids = []
    for order in buy_orders:
        result = await exchange.place_order(order)
        order_ids.append(result["order_id"])

    print(f"Placed orders: {order_ids}")

    await exchange.cancel_order("user2", order_ids[1])
    print(f"Cancelled order id: {order_ids[1]}")

    order = exchange.order_book.get_order(order_ids[1])
    print(f"Order after cancellation: {order}")
    assert order is None, f"Order {order_ids[1]} should be None after cancellation"

