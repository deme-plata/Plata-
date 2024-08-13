import curses
import asyncio
from decimal import Decimal
from typing import Dict, List
from .blockchain_interface import blockchain_interface
class ExchangeDashboardUI:
    def __init__(self, stdscr, blockchain, exchange):
        self.stdscr = stdscr
        self.blockchain = blockchain
        self.exchange = exchange
        self.height, self.width = self.stdscr.getmaxyx()
        self.menu_active = False  # Initialize menu_active
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            curses.init_pair(i + 1, i, -1)
        self.setup_colors()
        self.setup_windows()
        self.current_view = "main"
        self.main_menu_selection = 0
        self.trading_pairs = ["BTC/USD", "ETH/USD", "QDAGK/USD", "BTC/ETH", "ETH/QDAGK"]
        self.selected_pair = self.trading_pairs[0]
        self.price_history: Dict[str, List[float]] = {pair: [] for pair in self.trading_pairs}


    def setup_colors(self):
        self.TITLE_COLOR = curses.color_pair(14) | curses.A_BOLD
        self.NORMAL_COLOR = curses.color_pair(7)
        self.HIGHLIGHT_COLOR = curses.color_pair(10) | curses.A_BOLD
        self.WARNING_COLOR = curses.color_pair(9) | curses.A_BOLD
        self.BUY_COLOR = curses.color_pair(2)
        self.SELL_COLOR = curses.color_pair(1)
        self.MENU_COLOR = curses.color_pair(5)

    def setup_windows(self):
        self.main_win = curses.newwin(self.height, self.width, 0, 0)
        self.order_book_win = curses.newwin(self.height - 3, self.width // 3, 0, 0)
        self.trading_pairs_win = curses.newwin(self.height // 2 - 2, self.width // 3, 0, self.width // 3)
        self.price_action_win = curses.newwin(self.height // 2 - 1, self.width // 3, self.height // 2 - 2, self.width // 3)
        self.network_status_win = curses.newwin(self.height - 3, self.width // 3, 0, 2 * self.width // 3)
        self.menu_win = curses.newwin(3, self.width, self.height - 3, 0)

    def draw_main_menu(self):
        self.menu_win.clear()
        self.menu_win.box()
        menu_items = ["Order Book", "Trading Pairs", "Price Action", "Swap", "Network Status", "Exit"]
        item_width = self.width // len(menu_items)
        for i, item in enumerate(menu_items):
            if i == self.main_menu_selection:
                self.menu_win.attron(self.HIGHLIGHT_COLOR)
                self.menu_win.addstr(1, i * item_width + 2, item.center(item_width - 4))
                self.menu_win.attroff(self.HIGHLIGHT_COLOR)
            else:
                self.menu_win.addstr(1, i * item_width + 2, item.center(item_width - 4), self.MENU_COLOR)
        self.menu_win.refresh()

    def draw_borders(self):
        for win in [self.order_book_win, self.trading_pairs_win, self.price_action_win, self.network_status_win]:
            win.box()

    def draw_title(self, win, title, y, x):
        win.addstr(y, x, f"╔═ {title} ═╗", self.TITLE_COLOR)

    async def update_order_book(self):
        while True:
            self.order_book_win.clear()
            self.draw_borders()
            self.draw_title(self.order_book_win, f"Order Book: {self.selected_pair}", 0, 2)

            # Fetch the order book through the EnhancedOrderBook instance within EnhancedExchange
            order_book = {
                'bids': self.exchange.order_book.buy_orders.get(self.selected_pair, {}),
                'asks': self.exchange.order_book.sell_orders.get(self.selected_pair, {}),
            }

            buy_orders = list(order_book['bids'].items())[:10]  # Get top 10 buy orders
            sell_orders = list(order_book['asks'].items())[:10]  # Get top 10 sell orders

            for i, (price, orders) in enumerate(buy_orders, start=2):
                amount = sum(order.amount for order in orders)
                self.order_book_win.addstr(i, 2, f"Buy  {amount:.4f} @ {price:.2f}", self.BUY_COLOR)

            for i, (price, orders) in enumerate(sell_orders, start=2):
                amount = sum(order.amount for order in orders)
                self.order_book_win.addstr(i, 40, f"Sell {amount:.4f} @ {price:.2f}", self.SELL_COLOR)

            self.order_book_win.refresh()
            await asyncio.sleep(1)



    async def update_trading_pairs(self):
        while True:
            self.trading_pairs_win.clear()
            self.draw_borders()
            self.draw_title(self.trading_pairs_win, "Trading Pairs", 0, 2)

            for i, pair in enumerate(self.trading_pairs, start=2):
                price = await self.exchange.get_current_price(pair)
                color = self.HIGHLIGHT_COLOR if pair == self.selected_pair else self.NORMAL_COLOR
                self.trading_pairs_win.addstr(i, 2, f"{pair}: ${price:.2f}", color)

            self.trading_pairs_win.refresh()
            await asyncio.sleep(1)

    async def update_price_action(self):
        while True:
            self.price_action_win.clear()
            self.draw_borders()
            self.draw_title(self.price_action_win, f"Price Action: {self.selected_pair}", 0, 2)

            price = await self.exchange.get_current_price(self.selected_pair)
            self.price_history[self.selected_pair].append(price)
            if len(self.price_history[self.selected_pair]) > 30:
                self.price_history[self.selected_pair] = self.price_history[self.selected_pair][-30:]

            self.draw_price_graph(self.price_history[self.selected_pair])

            self.price_action_win.refresh()
            await asyncio.sleep(1)


    def draw_price_graph(self, price_history):
        if not price_history:
            return

        height, width = self.price_action_win.getmaxyx()
        graph_height = height - 4
        graph_width = width - 4

        min_price = min(price_history)
        max_price = max(price_history)
        price_range = max_price - min_price

        for i, price in enumerate(price_history):
            x = int(i * graph_width / len(price_history)) + 2
            y = int((price - min_price) * graph_height / price_range) if price_range else 0
            y = height - y - 3
            self.price_action_win.addch(y, x, '█', self.HIGHLIGHT_COLOR)
    def get_current_price(self, pair):
        base_price = 50000 if "BTC" in pair else 3000 if "ETH" in pair else 1
        return base_price * (1 + random.uniform(-0.01, 0.01))



    async def update_liquidity_pools(self):
        while True:
            self.liquidity_pools_win.clear()
            self.draw_borders()
            self.draw_title(self.liquidity_pools_win, "Liquidity Pools", 0, 2)

            pools = await blockchain_interface.exchange.get_liquidity_pools()
            for i, (pair, pool) in enumerate(list(pools.items())[:10], start=2):
                reserves = pool.get_reserves()
                self.liquidity_pools_win.addstr(i, 2, f"{pair}: {reserves[0]:.2f} / {reserves[1]:.2f}", self.NORMAL_COLOR)

            self.liquidity_pools_win.refresh()
            await asyncio.sleep(5)

    async def update_network_status(self):
        while True:
            self.network_status_win.clear()
            self.draw_borders()
            self.draw_title(self.network_status_win, "Network Status", 0, 2)

            node_stats = await blockchain_interface.get_node_stats()
            network_stats = await blockchain_interface.get_network_stats()

            self.network_status_win.addstr(2, 2, f"Connected Peers: {node_stats['connected_peers']}", self.NORMAL_COLOR)
            self.network_status_win.addstr(3, 2, f"Block Height: {node_stats['block_height']}", self.NORMAL_COLOR)
            self.network_status_win.addstr(4, 2, f"Last Block Time: {node_stats['last_block_time']}", self.NORMAL_COLOR)
            self.network_status_win.addstr(5, 2, f"Mempool Size: {network_stats['mempool_size']}", self.NORMAL_COLOR)

            consistency = sum(network_stats['network_consistency'].values()) / len(network_stats['network_consistency'])
            color = self.HIGHLIGHT_COLOR if consistency > 0.9 else self.WARNING_COLOR
            self.network_status_win.addstr(6, 2, f"Network Consistency: {consistency:.2%}", color)

            self.network_status_win.refresh()
            await asyncio.sleep(5)
    async def handle_input(self):
        self.stdscr.nodelay(True)
        while True:
            try:
                key = self.stdscr.getch()
                if key != -1:
                    if key == ord('m'):  # 'm' to toggle menu visibility
                        self.menu_active = not self.menu_active
                    elif self.menu_active:
                        if key == curses.KEY_LEFT:
                            self.main_menu_selection = (self.main_menu_selection - 1) % 6
                        elif key == curses.KEY_RIGHT:
                            self.main_menu_selection = (self.main_menu_selection + 1) % 6
                        elif key in [curses.KEY_ENTER, 10, 13]:
                            if self.main_menu_selection == 0:
                                await self.place_trade_ui()
                            elif self.main_menu_selection == 1:
                                await self.select_trading_pair()
                            elif self.main_menu_selection == 2:
                                # Price action is always visible now
                                pass
                            elif self.main_menu_selection == 3:
                                await self.swap_ui()
                            elif self.main_menu_selection == 4:
                                self.menu_active = False  # Close the menu
                            elif self.main_menu_selection == 5:
                                return  # Exit the Exchange UI and return to the main dashboard
                    elif key == ord('p'):  # 'p' to change selected pair when menu is closed
                        await self.select_trading_pair()
                    self.draw_main_menu()
            except Exception as e:
                self.show_popup("Error", str(e))
            await asyncio.sleep(0.1)

    async def select_trading_pair(self):
        current_index = self.trading_pairs.index(self.selected_pair)
        self.selected_pair = self.trading_pairs[(current_index + 1) % len(self.trading_pairs)]

    async def select_trading_pair(self):
        current_index = self.trading_pairs.index(self.selected_pair)
        self.selected_pair = self.trading_pairs[(current_index + 1) % len(self.trading_pairs)]

    async def place_trade_ui(self):
        # Implement place trade UI
        self.show_popup("Place Trade", "Trade UI not implemented yet")

    async def swap_ui(self):
        # Implement swap UI
        self.show_popup("Swap Tokens", "Swap UI not implemented yet")


    async def place_trade_ui(self):
        popup = curses.newwin(14, 60, (self.height - 14) // 2, (self.width - 60) // 2)
        popup.box()
        popup.addstr(0, 2, " Place Trade ", self.TITLE_COLOR)
        popup.addstr(2, 2, f"Pair: {self.selected_pair}", self.NORMAL_COLOR)
        popup.addstr(4, 2, "Type (buy/sell): ", self.NORMAL_COLOR)
        popup.addstr(6, 2, "Amount: ", self.NORMAL_COLOR)
        popup.addstr(8, 2, "Price: ", self.NORMAL_COLOR)
        popup.refresh()

        curses.echo()
        order_type = popup.getstr(4, 18, 4).decode('utf-8').lower()
        amount = Decimal(popup.getstr(6, 10, 10).decode('utf-8'))
        price = Decimal(popup.getstr(8, 9, 10).decode('utf-8'))
        curses.noecho()

        try:
            result = await self.exchange.place_limit_order(
                "current_user",  # Replace with actual user authentication
                order_type,
                self.selected_pair,
                amount,
                price
            )
            self.show_popup("Trade Result", f"Order placed with ID: {result}")
        except Exception as e:
            self.show_popup("Error", f"Failed to place order: {str(e)}")


    async def view_trading_history(self):
        history = await blockchain_interface.exchange.get_trading_history("current_user")
        popup = curses.newwin(self.height - 4, self.width - 4, 2, 2)
        popup.box()
        popup.addstr(0, 2, " Trading History ", self.TITLE_COLOR)
        for i, trade in enumerate(history[:self.height - 8], start=2):
            popup.addstr(i, 2, f"{trade['date']} - {trade['type']} {trade['amount']} {trade['pair']} @ {trade['price']}", self.NORMAL_COLOR)
        popup.addstr(self.height - 6, 2, "Press any key to return...", self.NORMAL_COLOR)
        popup.refresh()
        popup.getch()

    async def add_liquidity_ui(self):
        popup = curses.newwin(16, 60, (self.height - 16) // 2, (self.width - 60) // 2)
        popup.box()
        popup.addstr(0, 2, " Add Liquidity ", self.TITLE_COLOR)
        popup.addstr(2, 2, "Pool ID: ", self.NORMAL_COLOR)
        popup.addstr(4, 2, "Token A: ", self.NORMAL_COLOR)
        popup.addstr(6, 2, "Amount A: ", self.NORMAL_COLOR)
        popup.addstr(8, 2, "Token B: ", self.NORMAL_COLOR)
        popup.addstr(10, 2, "Amount B: ", self.NORMAL_COLOR)
        popup.refresh()

        curses.echo()
        pool_id = popup.getstr(2, 11, 20).decode('utf-8')
        token_a = popup.getstr(4, 11, 10).decode('utf-8')
        amount_a = Decimal(popup.getstr(6, 11, 10).decode('utf-8'))
        token_b = popup.getstr(8, 11, 10).decode('utf-8')
        amount_b = Decimal(popup.getstr(10, 11, 10).decode('utf-8'))
        curses.noecho()

        result = await blockchain_interface.exchange.add_liquidity("current_user", pool_id, token_a, amount_a, token_b, amount_b)
        self.show_popup("Liquidity Result", f"Liquidity added: {result}")
    async def swap_ui(self):
        popup = curses.newwin(14, 60, (self.height - 14) // 2, (self.width - 60) // 2)
        popup.box()
        popup.addstr(0, 2, " Swap Tokens ", self.TITLE_COLOR)
        popup.addstr(2, 2, "From Token: ", self.NORMAL_COLOR)
        popup.addstr(4, 2, "Amount: ", self.NORMAL_COLOR)
        popup.addstr(6, 2, "To Token: ", self.NORMAL_COLOR)
        popup.refresh()

        curses.echo()
        from_token = popup.getstr(2, 13, 10).decode('utf-8')
        amount = Decimal(popup.getstr(4, 10, 10).decode('utf-8'))
        to_token = popup.getstr(6, 11, 10).decode('utf-8')
        curses.noecho()

        try:
            estimated_output = await self.exchange.get_swap_estimate(from_token, to_token, amount)
            
            popup.addstr(8, 2, f"Estimated output: {estimated_output:.6f} {to_token}", self.HIGHLIGHT_COLOR)
            popup.addstr(10, 2, "Confirm swap? (y/n): ", self.NORMAL_COLOR)
            popup.refresh()

            confirm = popup.getch()
            if confirm == ord('y'):
                result = await self.exchange.swap("current_user", from_token, to_token, amount)
                self.show_popup("Swap Result", f"Swap executed: {result}")
            else:
                self.show_popup("Swap Cancelled", "Swap operation was cancelled.")
        except Exception as e:
            self.show_popup("Error", f"Failed to execute swap: {str(e)}")


    def show_popup(self, title, message):
        h, w = 8, 60
        y, x = (self.height - h) // 2, (self.width - w) // 2
        popup = curses.newwin(h, w, y, x)
        popup.box()
        popup.addstr(0, 2, f" {title} ", self.TITLE_COLOR)
        popup.addstr(2, 2, message, self.NORMAL_COLOR)
        popup.addstr(h-2, 2, "Press any key to close", self.NORMAL_COLOR)
        popup.refresh()
        popup.getch()


    async def run(self):
        tasks = [
            self.update_order_book(),
            self.update_trading_pairs(),
            self.update_liquidity_pools(),
            self.update_network_status(),
            self.handle_input()
        ]
        await asyncio.gather(*tasks)
async def main(stdscr):
    blockchain = blockchain_interface.blockchain
    exchange = blockchain_interface.exchange
    ui = ExchangeDashboardUI(stdscr, blockchain, exchange)
    await ui.run()

if __name__ == "__main__":
    curses.wrapper(lambda stdscr: asyncio.run(main(stdscr)))
