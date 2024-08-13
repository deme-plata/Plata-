import curses
import asyncio
import logging
from .ExchangeDashboardUI import ExchangeDashboardUI  # Import the exchange UI
import random 
from .shared_logic import get_quantum_blockchain, get_p2p_node, get_enhanced_exchange
from .blockchain_interface import BlockchainInterface
from quantumdagknight import get_mining_stats, get_transaction_history
import os
import json
import secrets
import hashlib
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
import aiohttp
import psutil
import socket
from .TokenManagementUI import *
# Configure logging to output to a file
logging.basicConfig(
    filename='app.log',  # Log file name
    filemode='w',  # Overwrite the log file each run
    level=logging.DEBUG,  # Log all levels DEBUG and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger().handlers[0].flush = lambda: None
logging.basicConfig(level=logging.ERROR)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)  # Only show critical errors in the terminal
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
logger = logging.getLogger(__name__)

blockchain_interface = BlockchainInterface()
async def run_uvicorn_server():
    config = uvicorn.Config("quantumdagknight:app", host="0.0.0.0", port=50503, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

class DashboardUI:
    def __init__(self, stdscr, blockchain, exchange, p2p_node):
        self.stdscr = stdscr
        self.blockchain = blockchain
        self.exchange = exchange
        self.menu_items = [
            ("Send Transaction", self.send_transaction_ui),
            ("View Transaction History", self.view_transaction_history),
            ("Toggle Mining", self.toggle_mining),
            ("Launch Exchange UI", self.launch_exchange_ui),
            ("Quantum Price Predictor", self.quantum_price_predictor),  # New killer feature
            ("Dashboard", self.show_dashboard),  # New menu item to return to dashboard
            ("Token Management", self.launch_token_management_ui),  # New menu item

            ("Create Wallet", self.create_wallet),  # Add this line
            ("Start Mining", self.start_mining),  # Add this line
            ("WebSocket IPs Status", self.websocket_ips_status_ui),  # New menu item for WebSocket IPs status
            ("Restart Servers", self.restart_servers),  # Add this line

            ("Exit", self.exit_ui)
        ]
        self.current_selection = 0
        self.current_view = "menu"
        self.wallet = None  # Store the created wallet here
        self.uvicorn_task = None
        self.websocket_task = None
        self.p2p_node = p2p_node  # Assuming p2p_node is passed when creating DashboardUI

        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            curses.init_pair(i + 1, i, -1)
        self.setup_colors()
        self.setup_windows()
    async def uvicorn_status_ui(self):
        status = "Running" if await self.check_uvicorn_status() else "Stopped"
        self.show_popup("Uvicorn Status", f"Uvicorn server is currently {status}.")

    async def check_uvicorn_status(self) -> bool:
        # Check if Uvicorn is running by looking for the process name in the system process list
        for proc in psutil.process_iter(['pid', 'name']):
            if "uvicorn" in proc.info['name'].lower():
                return True
        return False

    async def websocket_ips_status_ui(self):
        ips = await self.get_websocket_ips()
        message = "\n".join([f"WebSocket IP: {ip}" for ip in ips])
        self.show_popup("WebSocket IPs Status", message)


    async def get_websocket_ips(self) -> list:
        # Logic to retrieve WebSocket IPs
        websocket_ips = []
        
        # Example to find active WebSocket connections on port 8080 (adjust the port as necessary)
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' and conn.laddr.port == 50504:  # Replace 8080 with your WebSocket port
                websocket_ips.append(conn.laddr.ip)
        
        # If no WebSocket IPs found, return a default message
        if not websocket_ips:
            websocket_ips.append("No active WebSocket connections found.")
        
        return websocket_ips


    def setup_colors(self):
        self.TITLE_COLOR = curses.color_pair(14)
        self.NORMAL_COLOR = curses.color_pair(7)
        self.HIGHLIGHT_COLOR = curses.color_pair(2) | curses.A_BOLD
        self.WARNING_COLOR = curses.color_pair(9)
        self.HIGHLIGHT_COLOR = curses.color_pair(0) | curses.A_REVERSE  # Black background with reverse attribute

    def setup_windows(self):
        self.height, self.width = self.stdscr.getmaxyx()
        self.node_win = curses.newwin(6, self.width // 2, 1, 0)
        self.wallet_win = curses.newwin(6, self.width // 2, 1, self.width // 2)
        self.mining_win = curses.newwin(6, self.width // 2, 7, 0)
        self.menu_win = curses.newwin(6, self.width // 2, 7, self.width // 2)
        self.transactions_win = curses.newwin(self.height - 13, self.width, 13, 0)
    def draw_menu(self):
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        for idx, (menu_text, _) in enumerate(self.menu_items):
            x = w // 2 - len(menu_text) // 2
            y = h // 2 - len(self.menu_items) // 2 + idx
            if idx == self.current_selection:
                self.stdscr.attron(self.HIGHLIGHT_COLOR)
                self.stdscr.addstr(y, x, menu_text)
                self.stdscr.attroff(self.HIGHLIGHT_COLOR)
            else:
                self.stdscr.addstr(y, x, menu_text)

        self.stdscr.refresh()




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
        
        if self.wallet and "address" in self.wallet:
            self.wallet_win.addstr(2, 2, f"Address: {self.wallet['address']}", self.NORMAL_COLOR)

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
            
            
    async def start_mining(self):
        if not hasattr(self, 'wallet') or not self.wallet:
            self.show_popup("Error", "No wallet created. Please create a wallet first.")
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:8000/mine_block", json={
                    "node_id": "your_node_id",
                    "wallet_address": self.wallet["wallet_address"],  # Use the created wallet address
                    "node_ip": "127.0.0.1",
                    "node_port": 8000,
                    "wallet": self.wallet  # Pass the entire wallet
                })

                if response.status_code == 200:
                    result = response.json()
                    self.show_popup("Mining Started", f"Mining started successfully. {result.get('message', '')}")
                else:
                    self.show_popup("Error", f"Failed to start mining. Status code: {response.status_code}")

        except Exception as e:
            self.show_popup("Error", f"Exception occurred while starting mining: {str(e)}")
    async def handle_input(self):
        self.stdscr.nodelay(True)
        while True:
            try:
                key = self.stdscr.getch()
                if key != -1:
                    if self.current_view == "menu":
                        if key == ord('q'):
                            return
                        elif key == curses.KEY_UP:
                            self.current_selection = (self.current_selection - 1) % len(self.menu_items)
                        elif key == curses.KEY_DOWN:
                            self.current_selection = (self.current_selection + 1) % len(self.menu_items)
                        elif key in [curses.KEY_ENTER, ord('\n')]:
                            _, action = self.menu_items[self.current_selection]
                            await action()
                        self.draw_menu()
                    elif self.current_view == "dashboard":
                        if key == ord('b'):
                            self.current_view = "menu"
                            self.draw_menu()
                        elif key == ord('r'):
                            await self.refresh_dashboard()
                await asyncio.sleep(0.05)
            except Exception as e:
                logging.error(f"Error in handle_input: {str(e)}")
                await asyncio.sleep(0.1)



    async def refresh_dashboard(self):
        while self.current_view == "dashboard":
            await self.update_node_info()
            await self.update_wallet_info()
            await self.update_mining_info()
            await self.update_transactions()
            self.stdscr.addstr(self.height - 1, 0, "Press 'b' to go back to menu, 'r' to refresh", self.NORMAL_COLOR)
            self.stdscr.refresh()
            await asyncio.sleep(1)
    async def create_wallet(self):
        try:
            # Define the pincode for the wallet registration (ensure this is unique or handle it accordingly)
            pincode = "123456"  # Replace with dynamic input if needed

            # Prepare the user data for registration
            user_data = {
                "pincode": pincode
            }

            # Send a POST request to the /register endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:50503/register", json=user_data) as response:
                    if response.status == 200:
                        response_data = await response.json()

                        # Extract the wallet details from the response
                        wallet = response_data.get("wallet", {})
                        wallet_address = wallet.get("address", "N/A")
                        
                        # Store the wallet details securely
                        self.wallet = wallet

                        # Display the wallet address to the user
                        self.show_popup("Wallet Created", f"Address: {wallet_address}")
                    else:
                        # Handle the error if the registration fails
                        error_message = await response.text()
                        self.show_popup("Error", f"Failed to create wallet: {error_message}")

        except Exception as e:
            self.show_popup("Error", f"Exception occurred: {str(e)}")


    async def create_wallet_ui(self):
        try:
            # Simulate user registration and wallet creation
            user_pincode = "user_pincode_example"  # This would normally come from user input
            user = {"pincode": user_pincode}
            
            response = await self.register_user(user)
            self.wallet = response["wallet"]
            self.show_popup("Wallet Created", f"Wallet Address: {self.wallet['address']}\nAlias: {response['alias']}")
        except Exception as e:
            self.show_popup("Error", f"Failed to create wallet: {str(e)}")

    async def register_user(self, user):
        # Simulate the API call to register a new user and create a wallet
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_URL}/register", json=user) as response:
                return await response.json()

    async def start_mining_ui(self):
        if not self.wallet:
            self.show_popup("Error", "Please create a wallet first!")
            return

        try:
            # Start the mining process with the created wallet
            await self.mine_block(self.wallet)
            self.show_popup("Mining Started", f"Mining started with wallet {self.wallet['address']}")
        except Exception as e:
            self.show_popup("Error", f"Failed to start mining: {str(e)}")

    async def mine_block(self, wallet):
        # Simulate the API call to start mining
        async with aiohttp.ClientSession() as session:
            data = {
                "node_id": "node_id_example",
                "wallet_address": wallet["address"],
                "node_ip": "127.0.0.1",
                "node_port": 50502,
                "wallet": wallet
            }
            async with session.post(f"{API_URL}/mine_block", json=data) as response:
                return await response.json()


    async def launch_exchange_ui(self):
        # Initialize and run the EnhancedExchange UI
        exchange_ui = ExchangeDashboardUI(self.stdscr, self.blockchain, self.exchange)
        await exchange_ui.run()
    async def show_dashboard(self):
        self.current_view = "dashboard"
        while self.current_view == "dashboard":
            await self.refresh_dashboard()
            self.stdscr.addstr(self.height - 1, 0, "Press 'b' to go back to menu, 'r' to refresh", self.NORMAL_COLOR)
            self.stdscr.refresh()
            await asyncio.sleep(1)


    async def quantum_price_predictor(self):
        h, w = 20, 60
        y, x = (self.height - h) // 2, (self.width - w) // 2
        popup = curses.newwin(h, w, y, x)
        popup.box()
        popup.addstr(0, 2, " Quantum Price Predictor ", self.TITLE_COLOR | curses.A_BOLD)

        assets = ["BTC", "ETH", "QDAGK", "DOT", "ADA"]
        predictions = {}

        for i, asset in enumerate(assets):
            # Simulate quantum computation for price prediction
            await asyncio.sleep(0.5)  # Simulate quantum processing time
            prediction = random.uniform(0.8, 1.2)  # Random price multiplier
            current_price = await self.exchange.get_price(asset)
            predicted_price = current_price * prediction
            predictions[asset] = predicted_price

            popup.addstr(i+2, 2, f"{asset}: Current ${current_price:.2f} → Predicted ${predicted_price:.2f}", self.NORMAL_COLOR)
            
            # Add a quantum-inspired confidence meter
            confidence = int(prediction * 10) % 10
            popup.addstr(i+2, 40, "Confidence: [" + "█" * confidence + " " * (10-confidence) + "]", self.HIGHLIGHT_COLOR)

        popup.addstr(h-3, 2, "Based on quantum superposition analysis", self.WARNING_COLOR)
        popup.addstr(h-2, 2, "Press any key to close", self.NORMAL_COLOR)
        popup.refresh()
        popup.getch()

    async def launch_exchange_ui(self):
        exchange_ui = ExchangeDashboardUI(self.stdscr, self.blockchain, self.exchange)
        await exchange_ui.run()
        self.current_view = "menu"  # Set the view back to the main menu
        self.draw_menu()  # Redraw the main menu

    async def async_getch(self):
        return await asyncio.get_event_loop().run_in_executor(None, self.stdscr.getch)
    async def launch_token_management_ui(self):
        from TokenManagementUI import TokenManagementUI

        # Import the TokenManagementUI class
        from TokenManagementUI import TokenManagementUI
        
        # Initialize and run the TokenManagementUI
        token_management_ui = TokenManagementUI(self.stdscr, self.blockchain)
        await token_management_ui.run()

    async def send_transaction_ui(self):
        h, w = 14, 50
        y, x = (self.height - h) // 2, (self.width - w) // 2
        popup = curses.newwin(h, w, y, x)
        popup.keypad(True)  # Enable keypad input for the popup window
        popup.box()
        popup.addstr(0, 2, " Send Transaction ", self.TITLE_COLOR | curses.A_BOLD)

        recipient = ""
        amount_str = ""
        current_field = 0  # 0 for recipient, 1 for amount
        buttons = ["OK", "Cancel"]
        current_button = 0

        while True:
            popup.clear()
            popup.box()
            popup.addstr(0, 2, " Send Transaction ", self.TITLE_COLOR | curses.A_BOLD)
            popup.addstr(2, 2, "Recipient: ", self.NORMAL_COLOR)
            popup.addstr(2, 12, recipient)
            popup.addstr(4, 2, "Amount: ", self.NORMAL_COLOR)
            popup.addstr(4, 10, amount_str)

            for i, button in enumerate(buttons):
                button_x = 10 + (i * 15)
                if i == current_button and current_field == 2:
                    popup.attron(self.HIGHLIGHT_COLOR)
                popup.addstr(h - 3, button_x, f"[{button}]")
                popup.attroff(self.HIGHLIGHT_COLOR)

            popup.refresh()

            if current_field == 0:
                popup.move(2, 12 + len(recipient))
            elif current_field == 1:
                popup.move(4, 10 + len(amount_str))

            key = popup.getch()

            if key == ord('\n'):  # Enter key
                if current_field < 2:
                    current_field += 1
                elif buttons[current_button] == "OK":
                    if not recipient or not amount_str:
                        popup.addstr(h - 5, 2, "Error: Recipient and amount are required", self.WARNING_COLOR)
                        popup.refresh()
                        popup.getch()
                    else:
                        try:
                            amount = float(amount_str)
                            result = await blockchain_interface.send_transaction(recipient, amount)
                            self.show_popup("Transaction Result", result)
                            break
                        except ValueError:
                            popup.addstr(h - 5, 2, "Error: Invalid amount", self.WARNING_COLOR)
                            popup.refresh()
                            popup.getch()
                else:  # Cancel button
                    break
            elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                if current_field == 0 and recipient:
                    recipient = recipient[:-1]
                elif current_field == 1 and amount_str:
                    amount_str = amount_str[:-1]
                elif current_field == 2:  # In button selection, backspace closes the popup
                    break
            elif key == curses.KEY_LEFT and current_field == 2:
                current_button = (current_button - 1) % len(buttons)
            elif key == curses.KEY_RIGHT and current_field == 2:
                current_button = (current_button + 1) % len(buttons)
            elif key == curses.KEY_UP:
                current_field = max(0, current_field - 1)
            elif key == curses.KEY_DOWN:
                current_field = min(2, current_field + 1)
            elif 32 <= key <= 126:  # Printable characters
                if current_field == 0:
                    recipient += chr(key)
                elif current_field == 1:
                    amount_str += chr(key)

        popup.clear()
        popup.refresh()






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
        try:
            mining_stats = await get_mining_stats()  # Await the coroutine to get the actual result
            if mining_stats['status'] == 'INACTIVE':
                await blockchain_interface.start_mining()
                self.show_popup("Mining", "Mining started")
            else:
                await blockchain_interface.stop_mining()
                self.show_popup("Mining", "Mining stopped")
        except Exception as e:
            logging.error(f"Error toggling mining: {str(e)}")
            self.show_popup("Error", str(e))

    async def restart_servers(self):
        try:
            # Cancel existing server tasks if they are running
            if self.websocket_task:
                self.websocket_task.cancel()
                await self.websocket_task

            if self.uvicorn_task:
                self.uvicorn_task.cancel()
                await self.uvicorn_task

            # Restart the servers
            self.uvicorn_task = asyncio.create_task(run_uvicorn_server())
            self.websocket_task = asyncio.create_task(self.p2p_node.start())

            self.show_popup("Success", "Servers restarted successfully.")
        except Exception as e:
            self.show_popup("Error", f"Failed to restart servers: {str(e)}")


    def exit_ui(self):
        curses.endwin()
        raise SystemExit

    async def refresh_dashboard(self):
        while self.current_view == "dashboard":
            await self.update_node_info()
            await self.update_wallet_info()
            await self.update_mining_info()
            await self.update_transactions()
            self.stdscr.addstr(self.height - 1, 0, "Press 'b' to go back to menu, 'r' to refresh", self.NORMAL_COLOR)
            self.stdscr.refresh()
            await asyncio.sleep(1)

    async def run(self):
        # Start the Uvicorn server and WebSocket server initially
        self.uvicorn_task = asyncio.create_task(run_uvicorn_server())
        self.websocket_task = asyncio.create_task(self.p2p_node.start())

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
