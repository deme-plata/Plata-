import curses
import asyncio
import logging

from .shared_logic import get_quantum_blockchain, get_p2p_node, get_enhanced_exchange
from .blockchain_interface import BlockchainInterface
from quantumdagknight import get_mining_stats, get_transaction_history

# Configure logging to output to a file
logging.basicConfig(
    filename='dashboard_ui.log',  # Log file name
    filemode='w',  # Overwrite the log file each run
    level=logging.DEBUG,  # Log all levels DEBUG and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

blockchain_interface = BlockchainInterface()
class DashboardUI:
    def __init__(self, stdscr, blockchain, exchange):
        self.stdscr = stdscr
        self.blockchain = blockchain
        self.exchange = exchange
        self.menu_items = [
            ("Send Transaction", self.send_transaction_ui),
            ("View Transaction History", self.view_transaction_history),
            ("Toggle Mining", self.toggle_mining),
            ("Exit", self.exit_ui)
        ]
        self.current_selection = 0

        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            curses.init_pair(i + 1, i, -1)
        self.setup_colors()
        self.setup_windows()
    def setup_colors(self):
        self.TITLE_COLOR = curses.color_pair(14)
        self.NORMAL_COLOR = curses.color_pair(7)
        self.HIGHLIGHT_COLOR = curses.color_pair(10)
        self.WARNING_COLOR = curses.color_pair(9)

    def setup_windows(self):
        self.height, self.width = self.stdscr.getmaxyx()
        self.node_win = curses.newwin(6, self.width // 2, 1, 0)
        self.wallet_win = curses.newwin(6, self.width // 2, 1, self.width // 2)
        self.mining_win = curses.newwin(6, self.width // 2, 7, 0)
        self.menu_win = curses.newwin(6, self.width // 2, 7, self.width // 2)
        self.transactions_win = curses.newwin(self.height - 13, self.width, 13, 0)

    def draw_menu(self):
        self.menu_win.clear()
        self.menu_win.box()
        self.draw_title(self.menu_win, "Menu", 0, 2)
        for idx, (item_text, _) in enumerate(self.menu_items):
            if idx == self.current_selection:
                self.menu_win.attron(self.HIGHLIGHT_COLOR)
                self.menu_win.addstr(idx + 1, 2, f"> {item_text}")
                self.menu_win.attroff(self.HIGHLIGHT_COLOR)
            else:
                self.menu_win.addstr(idx + 1, 2, f"  {item_text}")
        self.menu_win.refresh()


    def draw_borders(self):
        for win in [self.node_win, self.wallet_win, self.mining_win, self.transactions_win]:
            win.box()

    def draw_title(self, win, title, y, x):
        win.addstr(y, x, f"╔═ {title} ═╗", self.TITLE_COLOR | curses.A_BOLD)

    def draw_menu(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        for idx, (menu_text, _) in enumerate(self.menu_items):
            x = w // 2 - len(menu_text) // 2
            y = h // 2 - len(self.menu_items) // 2 + idx
            if idx == self.current_selection:
                self.stdscr.attron(curses.color_pair(1))
                self.stdscr.addstr(y, x, menu_text)
                self.stdscr.attroff(curses.color_pair(1))
            else:
                self.stdscr.addstr(y, x, menu_text)

        self.stdscr.refresh()

    async def update_node_info(self):
        node_stats = await blockchain_interface.get_node_stats()
        self.node_win.clear()
        self.draw_borders()
        self.draw_title(self.node_win, "Node Information", 0, 2)
        self.node_win.addstr(1, 2, f"Node ID: {node_stats['node_id']}", self.NORMAL_COLOR)
        self.node_win.addstr(2, 2, f"Connected Peers: {node_stats['connected_peers']}", self.NORMAL_COLOR)
        self.node_win.addstr(3, 2, f"Block Height: {node_stats['block_height']}", self.HIGHLIGHT_COLOR)
        self.node_win.addstr(4, 2, f"Last Block Time: {node_stats['last_block_time']}", self.NORMAL_COLOR)
        self.node_win.refresh()
        await asyncio.sleep(5)

    async def update_wallet_info(self):
        balance = await blockchain_interface.get_wallet_balance("USD")  # Example currency

        self.wallet_win.clear()
        self.draw_borders()
        self.draw_title(self.wallet_win, "Wallet", 0, 2)
        self.wallet_win.addstr(1, 2, f"Balance: {balance:.2f} QDAGK", self.HIGHLIGHT_COLOR | curses.A_BOLD)
        self.wallet_win.addstr(3, 2, "Press 'T' to send a transaction", self.NORMAL_COLOR)
        self.wallet_win.addstr(4, 2, "Press 'H' to view transaction history", self.NORMAL_COLOR)
        self.wallet_win.refresh()
        await asyncio.sleep(5)

    async def update_mining_info(self):
        mining_stats = await get_mining_stats()
        mining_active = mining_stats.get('current_hashrate', None) is not None

        self.mining_win.clear()
        self.draw_borders()
        self.draw_title(self.mining_win, "Mining", 0, 2)

        status = "ACTIVE" if mining_active else "INACTIVE"
        color = self.HIGHLIGHT_COLOR if mining_active else self.WARNING_COLOR

        self.mining_win.addstr(1, 2, f"Status: {status}", color | curses.A_BOLD)
        self.mining_win.addstr(2, 2, f"Hash Rate: {mining_stats.get('current_hashrate', 'N/A')} H/s", self.NORMAL_COLOR)
        self.mining_win.addstr(3, 2, f"Blocks Mined: {mining_stats.get('total_blocks_mined', 0)}", self.NORMAL_COLOR)
        self.mining_win.addstr(4, 2, "Press 'M' to start/stop mining", self.NORMAL_COLOR)

        self.mining_win.refresh()
        await asyncio.sleep(1)

    async def update_transactions(self):
        try:
            transactions = await self.blockchain.get_transaction_history()

            self.transactions_win.clear()
            self.draw_borders()
            self.draw_title(self.transactions_win, "Recent Transactions", 0, 2)

            for i, tx in enumerate(transactions[:self.height - 16], start=1):
                transaction_line = f"{tx['date']} - {tx['amount']} QDAGK to {tx['recipient'][:20]}..."
                self.transactions_win.addstr(i, 2, transaction_line, self.NORMAL_COLOR)

            self.transactions_win.refresh()

        except Exception as e:
            logger.error(f"Error updating transactions: {str(e)}")
            self.transactions_win.addstr(1, 2, "Failed to load transactions.", self.WARNING_COLOR)
            self.transactions_win.refresh()

        await asyncio.sleep(10)

    def show_popup(self, title, message):
        h, w = 10, 40
        y, x = (self.height - h) // 2, (self.width - w) // 2
        popup = curses.newwin(h, w, y, x)
        popup.box()
        popup.addstr(0, 2, f" {title} ", self.TITLE_COLOR | curses.A_BOLD)
        popup.addstr(2, 2, message, self.NORMAL_COLOR)
        popup.addstr(h-2, 2, "Press any key to close", self.NORMAL_COLOR)
        popup.refresh()
        popup.getch()
    async def handle_input(self):
        self.stdscr.nodelay(True)  # Set non-blocking mode
        while True:
            try:
                key = self.stdscr.getch()
                if key != -1:
                    if key == ord('q'):
                        return
                    elif key == curses.KEY_UP:
                        self.current_selection = (self.current_selection - 1) % len(self.menu_items)
                    elif key == curses.KEY_DOWN:
                        self.current_selection = (self.current_selection + 1) % len(self.menu_items)
                    elif key in [curses.KEY_ENTER, ord('\n')]:
                        _, action = self.menu_items[self.current_selection]
                        await action()  # Execute the selected action
                    self.draw_menu()  # Redraw the menu after handling input
            except Exception as e:
                logging.error(f"Error in handle_input: {str(e)}")
            await asyncio.sleep(0.05)  # Small delay to prevent CPU overload



    async def async_getch(self):
        return await asyncio.get_event_loop().run_in_executor(None, self.stdscr.getch)


    async def send_transaction_ui(self):
        popup = curses.newwin(10, 50, (self.height - 10) // 2, (self.width - 50) // 2)
        popup.box()
        popup.addstr(0, 2, " Send Transaction ", self.TITLE_COLOR | curses.A_BOLD)
        popup.addstr(2, 2, "Recipient: ", self.NORMAL_COLOR)
        popup.addstr(4, 2, "Amount: ", self.NORMAL_COLOR)

        # Initialize button positions
        button_y = 7
        ok_button_x = 15
        cancel_button_x = 30

        # Display the buttons
        popup.addstr(button_y, ok_button_x, "[ OK ]", self.HIGHLIGHT_COLOR)
        popup.addstr(button_y, cancel_button_x, "[ Cancel ]", self.NORMAL_COLOR)
        popup.refresh()

        curses.echo()
        recipient = popup.getstr(2, 12, 30).decode('utf-8')
        amount = float(popup.getstr(4, 10, 10).decode('utf-8'))
        curses.noecho()

        selected_button = 0  # 0 for OK, 1 for Cancel

        while True:
            key = popup.getch()
            if key == curses.KEY_LEFT or key == curses.KEY_RIGHT:
                selected_button = 1 - selected_button  # Toggle between 0 and 1
            elif key == curses.KEY_ENTER or key in [10, 13]:
                break  # Confirm the selected button

            # Update button highlighting
            popup.addstr(button_y, ok_button_x, "[ OK ]", self.HIGHLIGHT_COLOR if selected_button == 0 else self.NORMAL_COLOR)
            popup.addstr(button_y, cancel_button_x, "[ Cancel ]", self.HIGHLIGHT_COLOR if selected_button == 1 else self.NORMAL_COLOR)
            popup.refresh()

        # Handle OK/Cancel actions
        if selected_button == 0:  # OK button was pressed
            result = await blockchain_interface.send_transaction(recipient, amount)
            self.show_popup("Transaction Result", result)
        else:  # Cancel button was pressed
            self.show_popup("Transaction Canceled", "The transaction was not sent.")


    async def view_transaction_history(self):
        history = await self.blockchain.get_transaction_history()
        popup = curses.newwin(self.height - 4, self.width - 4, 2, 2)
        popup.box()
        popup.addstr(0, 2, " Transaction History ", self.TITLE_COLOR | curses.A_BOLD)
        for i, tx in enumerate(history, start=1):
            popup.addstr(i, 2, f"{tx['date']} - {tx['amount']} QDAGK to {tx['recipient']}", self.NORMAL_COLOR)
        popup.addstr(self.height - 6, 2, "Press any key to return...", self.NORMAL_COLOR)
        popup.refresh()
        popup.getch()


    async def toggle_mining(self):
        if await get_mining_stats()['status'] == 'INACTIVE':
            await blockchain_interface.start_mining()
            self.show_popup("Mining", "Mining started")
        else:
            await blockchain_interface.stop_mining()
            self.show_popup("Mining", "Mining stopped")

    def exit_ui(self):
        curses.endwin()
        raise SystemExit

    async def refresh_dashboard(self):
        while True:
            await self.update_node_info()
            await self.update_wallet_info()
            await self.update_mining_info()
            await self.update_transactions()
            await asyncio.sleep(5)  # Adjust the refresh rate as needed
    async def run(self):
        try:
            self.draw_menu()
            await asyncio.gather(
                self.update_node_info(),
                self.update_wallet_info(),
                self.update_mining_info(),
                self.update_transactions(),
                self.handle_input()
            )
        except Exception as e:
            logging.error(f"Error in run method: {str(e)}")
        finally:
            curses.endwin()


def main(stdscr):
    blockchain = None  # Initialize blockchain as needed
    exchange = None  # Initialize exchange as needed
    ui = DashboardUI(stdscr, blockchain, exchange)
    ui.run()

if __name__ == "__main__":
    curses.wrapper(main)
