import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import curses
from decimal import Decimal
from curses_dashboard.blockchain_interface import start_p2p_node, get_node_stats, get_wallet_balance, get_mining_stats
from curses_dashboard.ExchangeDashboardUI import ExchangeDashboardUI
from curses_dashboard.dashboard import DashboardUI
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import curses
from decimal import Decimal
from curses_dashboard.blockchain_interface import start_p2p_node, get_node_stats, get_wallet_balance, get_mining_stats
from curses_dashboard.ExchangeDashboardUI import ExchangeDashboardUI
from curses_dashboard.dashboard import DashboardUI
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import curses
from curses_dashboard.ExchangeDashboardUI import ExchangeDashboardUI
from curses_dashboard.dashboard import DashboardUI
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from curses_dashboard.ExchangeDashboardUI import ExchangeDashboardUI
from curses_dashboard.dashboard import DashboardUI

class MockCurses:
    def __init__(self):
        self.COLORS = 8
        self.COLOR_PAIRS = 64
        self.A_NORMAL = 0
        self.A_BOLD = 1
        self.A_REVERSE = 2
        self.screen = Mock()

    def initscr(self):
        return self.screen

    def newwin(self, *args):
        return Mock()

    def endwin(self):
        pass

    def noecho(self):
        pass

    def cbreak(self):
        pass

    def start_color(self):
        pass

    def use_default_colors(self):
        pass

    def init_pair(self, *args):
        pass

    def color_pair(self, n):
        return n

    def curs_set(self, visibility):
        pass

mock_curses = MockCurses()


class TestExchangeDashboardUI(unittest.TestCase):
    def setUp(self):
        self.mock_stdscr = mock_curses.screen
        self.mock_stdscr.getmaxyx.return_value = (50, 150)  # Simulate a 50x150 terminal

        patcher = patch('curses_dashboard.ExchangeDashboardUI.curses', mock_curses)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Mock blockchain_interface
        self.mock_blockchain_interface = Mock()
        self.mock_blockchain_interface.exchange = AsyncMock()
        self.mock_blockchain_interface.get_node_stats = AsyncMock()
        self.mock_blockchain_interface.get_network_stats = AsyncMock()

        with patch('curses_dashboard.ExchangeDashboardUI.blockchain_interface', self.mock_blockchain_interface):
            self.ui = ExchangeDashboardUI(self.mock_stdscr)




    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_order_book(self, mock_sleep):
        mock_orders = [
            Mock(order_type='buy', amount=1.0, price=100.0),
            Mock(order_type='sell', amount=2.0, price=101.0),
        ]
        self.mock_blockchain_interface.exchange.get_orders.return_value = mock_orders

        # Run the update_order_book method for one iteration
        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_order_book()

        self.ui.order_book_win.addstr.assert_any_call(2, 2, "Buy  1.0000 @ 100.00", self.ui.BUY_COLOR)
        self.ui.order_book_win.addstr.assert_any_call(2, 40, "Sell 2.0000 @ 101.00", self.ui.SELL_COLOR)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_trading_pairs(self, mock_sleep):
        self.mock_blockchain_interface.exchange.get_tradable_assets.return_value = ['BTC', 'ETH']
        self.mock_blockchain_interface.exchange.price_oracle.get_price.side_effect = [50000, 3000]

        # Run the update_trading_pairs method for one iteration
        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_trading_pairs()

        self.ui.trading_pairs_win.addstr.assert_any_call(2, 2, "BTC: $50000.00", self.ui.NORMAL_COLOR)
        self.ui.trading_pairs_win.addstr.assert_any_call(3, 2, "ETH: $3000.00", self.ui.NORMAL_COLOR)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_liquidity_pools(self, mock_sleep):
        mock_pool = Mock()
        mock_pool.get_reserves.return_value = (1000, 5000)
        self.ui.blockchain_interface.exchange.liquidity_pools = {'BTC_ETH': mock_pool}

        # Run the update_liquidity_pools method for one iteration
        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_liquidity_pools()

        self.ui.liquidity_pools_win.addstr.assert_called_with(2, 2, "BTC_ETH: 1000.00 / 5000.00", self.ui.NORMAL_COLOR)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_network_status(self, mock_sleep):
        self.mock_blockchain_interface.get_node_stats.return_value = {
            'connected_peers': 5,
            'block_height': 1000,
            'last_block_time': '2023-06-01 12:00:00'
        }
        self.mock_blockchain_interface.get_network_stats.return_value = {
            'mempool_size': 100,
            'network_consistency': {'node1': True, 'node2': True}
        }

        # Run the update_network_status method for one iteration
        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_network_status()

        self.ui.network_status_win.addstr.assert_any_call(2, 2, "Connected Peers: 5", self.ui.NORMAL_COLOR)
        self.ui.network_status_win.addstr.assert_any_call(3, 2, "Block Height: 1000", self.ui.NORMAL_COLOR)
        self.ui.network_status_win.addstr.assert_any_call(6, 2, "Network Consistency: 100.00%", self.ui.HIGHLIGHT_COLOR)

    @patch('curses_dashboard.ExchangeDashboardUI.show_popup')
    async def test_place_trade_ui(self, mock_show_popup):
        self.ui.stdscr.getstr.side_effect = [b'BTC_USD', b'buy', b'1.0', b'50000']
        self.mock_blockchain_interface.exchange.place_limit_order.return_value = 'order123'

        await self.ui.place_trade_ui()

        self.mock_blockchain_interface.exchange.place_limit_order.assert_called_with(
            "current_user", "buy", "BTC_USD", Decimal('1.0'), Decimal('50000')
        )
        mock_show_popup.assert_called_with("Trade Result", "Order placed with ID: order123")

    @patch('curses_dashboard.ExchangeDashboardUI.show_popup')
    async def test_cancel_order_ui(self, mock_show_popup):
        self.ui.stdscr.getstr.return_value = b'order123'
        self.mock_blockchain_interface.exchange.cancel_order.return_value = {'status': 'success'}

        await self.ui.cancel_order_ui()

        self.mock_blockchain_interface.exchange.cancel_order.assert_called_with("current_user", "order123")
        mock_show_popup.assert_called_with("Cancel Result", "Order cancellation: success")

    @patch('curses_dashboard.ExchangeDashboardUI.show_popup')
    async def test_add_liquidity_ui(self, mock_show_popup):
        self.ui.stdscr.getstr.side_effect = [b'BTC_ETH', b'1.0', b'10.0']
        self.mock_blockchain_interface.exchange.add_liquidity.return_value = 'LP_TOKEN_123'

        await self.ui.add_liquidity_ui()

        self.mock_blockchain_interface.exchange.add_liquidity.assert_called_with(
            "current_user", "BTC_ETH", Decimal('1.0'), Decimal('10.0')
        )
        mock_show_popup.assert_called_with("Liquidity Result", "Liquidity added: LP_TOKEN_123")

    @patch('curses_dashboard.ExchangeDashboardUI.place_trade_ui')
    @patch('curses_dashboard.ExchangeDashboardUI.cancel_order_ui')
    @patch('curses_dashboard.ExchangeDashboardUI.add_liquidity_ui')
    async def test_handle_input(self, mock_add_liquidity, mock_cancel_order, mock_place_trade):
        # Simulate user pressing 't', 'c', 'l', and 'q'
        self.ui.stdscr.getch.side_effect = [ord('t'), ord('c'), ord('l'), ord('q')]

        await self.ui.handle_input()

        mock_place_trade.assert_called_once()
        mock_cancel_order.assert_called_once()
        mock_add_liquidity.assert_called_once()
class TestDashboardUI(unittest.TestCase):
    def setUp(self):
        self.mock_stdscr = mock_curses.screen
        self.mock_stdscr.getmaxyx.return_value = (50, 150)  # Simulate a 50x150 terminal

        patcher = patch('curses_dashboard.ExchangeDashboardUI.curses', mock_curses)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Mock blockchain_interface
        self.mock_blockchain_interface = Mock()
        self.mock_blockchain_interface.exchange = AsyncMock()
        self.mock_blockchain_interface.get_node_stats = AsyncMock()
        self.mock_blockchain_interface.get_network_stats = AsyncMock()

        with patch('curses_dashboard.ExchangeDashboardUI.blockchain_interface', self.mock_blockchain_interface):
            self.ui = ExchangeDashboardUI(self.mock_stdscr)



    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_node_info(self, mock_sleep):
        self.mock_blockchain_interface.get_node_stats.return_value = {
            'node_id': 'node1',
            'connected_peers': 5,
            'block_height': 1000,
            'last_block_time': '2023-06-01 12:00:00'
        }

        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_node_info()

        self.ui.node_win.addstr.assert_any_call(1, 2, "Node ID: node1", self.ui.NORMAL_COLOR)
        self.ui.node_win.addstr.assert_any_call(2, 2, "Connected Peers: 5", self.ui.NORMAL_COLOR)
        self.ui.node_win.addstr.assert_any_call(3, 2, "Block Height: 1000", self.ui.HIGHLIGHT_COLOR)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_wallet_info(self, mock_sleep):
        self.mock_blockchain_interface.get_wallet_balance.return_value = Decimal('100.50')

        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_wallet_info()

        self.ui.wallet_win.addstr.assert_any_call(1, 2, "Balance: 100.50 QDAGK", self.ui.HIGHLIGHT_COLOR | curses.A_BOLD)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_mining_info(self, mock_sleep):
        self.mock_blockchain_interface.get_mining_stats.return_value = {
            'status': 'ACTIVE',
            'hash_rate': 100.5,
            'blocks_mined': 50
        }

        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_mining_info()

        self.ui.mining_win.addstr.assert_any_call(1, 2, "Status: ACTIVE", self.ui.HIGHLIGHT_COLOR | curses.A_BOLD)
        self.ui.mining_win.addstr.assert_any_call(2, 2, "Hash Rate: 100.5 H/s", self.ui.NORMAL_COLOR)
        self.ui.mining_win.addstr.assert_any_call(3, 2, "Blocks Mined: 50", self.ui.NORMAL_COLOR)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_update_transactions(self, mock_sleep):
        self.mock_blockchain_interface.get_transaction_history.return_value = [
            {'date': '2023-06-01', 'amount': 10, 'recipient': 'address1'},
            {'date': '2023-06-02', 'amount': 20, 'recipient': 'address2'}
        ]

        mock_sleep.side_effect = [None, asyncio.CancelledError]
        with self.assertRaises(asyncio.CancelledError):
            await self.ui.update_transactions()

        self.ui.transactions_win.addstr.assert_any_call(1, 2, "2023-06-01 - 10 QDAGK to address1", self.ui.NORMAL_COLOR)
        self.ui.transactions_win.addstr.assert_any_call(2, 2, "2023-06-02 - 20 QDAGK to address2", self.ui.NORMAL_COLOR)

    @patch('curses_dashboard.dashboard.DashboardUI.show_popup')
    async def test_send_transaction_ui(self, mock_show_popup):
        self.ui.stdscr.getstr.side_effect = [b'recipient_address', b'10.5']
        self.mock_blockchain_interface.send_transaction.return_value = "Transaction sent successfully"

        await self.ui.send_transaction_ui()

        self.mock_blockchain_interface.send_transaction.assert_called_with('recipient_address', 10.5)
        mock_show_popup.assert_called_with("Transaction Result", "Transaction sent successfully")

    @patch('curses_dashboard.dashboard.DashboardUI.show_popup')
    async def test_toggle_mining(self, mock_show_popup):
        self.mock_blockchain_interface.get_mining_stats.return_value = {'status': 'INACTIVE'}
        
        await self.ui.toggle_mining()

        self.mock_blockchain_interface.start_mining.assert_called_once()
        mock_show_popup.assert_called_with("Mining", "Mining started")

        self.mock_blockchain_interface.get_mining_stats.return_value = {'status': 'ACTIVE'}
        
        await self.ui.toggle_mining()

        self.mock_blockchain_interface.stop_mining.assert_called_once()
        mock_show_popup.assert_called_with("Mining", "Mining stopped")

    @patch('curses_dashboard.dashboard.DashboardUI.send_transaction_ui')
    @patch('curses_dashboard.dashboard.DashboardUI.view_transaction_history')
    @patch('curses_dashboard.dashboard.DashboardUI.toggle_mining')
    async def test_handle_input(self, mock_toggle_mining, mock_view_history, mock_send_transaction):
        # Simulate user pressing 't', 'h', 'm', and 'q'
        self.ui.stdscr.getch.side_effect = [ord('t'), ord('h'), ord('m'), ord('q')]

        await self.ui.handle_input()

        mock_send_transaction.assert_called_once()
        mock_view_history.assert_called_once()
        mock_toggle_mining.assert_called_once()

    async def test_draw_network_stats(self):
        self.mock_blockchain_interface.get_network_stats.return_value = {
            'connected_peers': 10,
            'mempool_size': 50,
            'network_consistency': {'node1': True, 'node2': True, 'node3': False}
        }

        await self.ui.draw_network_stats(self.mock_stdscr)

        self.mock_stdscr.addstr.assert_any_call(20, 1, "Network Information", curses.A_BOLD)
        self.mock_stdscr.addstr.assert_any_call(21, 1, "Connected Peers: 10")
        self.mock_stdscr.addstr.assert_any_call(22, 1, "Mempool Size: 50")
        self.mock_stdscr.addstr.assert_any_call(23, 1, "Network Consistency: 66.67%")

if __name__ == '__main__':
    unittest.main()
