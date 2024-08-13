import curses
import asyncio
from decimal import Decimal

class TokenManagementUI:
    def __init__(self, stdscr, blockchain_interface):
        self.stdscr = stdscr
        self.blockchain = blockchain_interface
        self.current_menu = "main"
        self.current_selection = 0
        self.token_name = ""
        self.token_supply = ""
        self.pool_token_a = ""
        self.pool_token_b = ""
        self.pool_amount_a = ""
        self.pool_amount_b = ""
        self.setup_colors()

    def setup_colors(self):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    def draw_menu(self):
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        # Draw title
        title = "Token Management Dashboard"
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(0, (width - len(title)) // 2, title)
        self.stdscr.attroff(curses.color_pair(1))

        if self.current_menu == "main":
            menu_items = [
                "Create New Token",
                "Create Liquidity Pool",
                "View Tokens",
                "View Liquidity Pools",
                "Exit"
            ]
        elif self.current_menu == "create_token":
            menu_items = [
                f"Token Name: {self.token_name}",
                f"Token Supply: {self.token_supply}",
                "Create Token",
                "Back to Main Menu"
            ]
        elif self.current_menu == "create_pool":
            menu_items = [
                f"Token A: {self.pool_token_a}",
                f"Token B: {self.pool_token_b}",
                f"Amount A: {self.pool_amount_a}",
                f"Amount B: {self.pool_amount_b}",
                "Create Pool",
                "Back to Main Menu"
            ]

        for idx, item in enumerate(menu_items):
            x = width // 2 - len(item) // 2
            y = height // 2 - len(menu_items) // 2 + idx
            if idx == self.current_selection:
                self.stdscr.attron(curses.color_pair(2))
                self.stdscr.addstr(y, x, item)
                self.stdscr.attroff(curses.color_pair(2))
            else:
                self.stdscr.addstr(y, x, item)

        self.stdscr.refresh()

    async def handle_input(self):
        while True:
            key = self.stdscr.getch()
            if key == curses.KEY_UP and self.current_selection > 0:
                self.current_selection -= 1
            elif key == curses.KEY_DOWN and self.current_selection < len(self.get_menu_items()) - 1:
                self.current_selection += 1
            elif key == ord('\n'):  # Enter key
                await self.handle_selection()
            self.draw_menu()
            await asyncio.sleep(0.1)

    def get_menu_items(self):
        if self.current_menu == "main":
            return ["Create New Token", "Create Liquidity Pool", "View Tokens", "View Liquidity Pools", "Exit"]
        elif self.current_menu == "create_token":
            return [f"Token Name: {self.token_name}", f"Token Supply: {self.token_supply}", "Create Token", "Back to Main Menu"]
        elif self.current_menu == "create_pool":
            return [f"Token A: {self.pool_token_a}", f"Token B: {self.pool_token_b}", 
                    f"Amount A: {self.pool_amount_a}", f"Amount B: {self.pool_amount_b}", 
                    "Create Pool", "Back to Main Menu"]

    async def handle_selection(self):
        if self.current_menu == "main":
            if self.current_selection == 0:
                self.current_menu = "create_token"
                self.current_selection = 0
            elif self.current_selection == 1:
                self.current_menu = "create_pool"
                self.current_selection = 0
            elif self.current_selection == 2:
                await self.view_tokens()
            elif self.current_selection == 3:
                await self.view_liquidity_pools()
            elif self.current_selection == 4:
                raise SystemExit
        elif self.current_menu == "create_token":
            if self.current_selection == 0:
                self.token_name = await self.get_user_input("Enter token name: ")
            elif self.current_selection == 1:
                self.token_supply = await self.get_user_input("Enter token supply: ")
            elif self.current_selection == 2:
                await self.create_token()
            elif self.current_selection == 3:
                self.current_menu = "main"
                self.current_selection = 0
        elif self.current_menu == "create_pool":
            if self.current_selection == 0:
                self.pool_token_a = await self.get_user_input("Enter Token A: ")
            elif self.current_selection == 1:
                self.pool_token_b = await self.get_user_input("Enter Token B: ")
            elif self.current_selection == 2:
                self.pool_amount_a = await self.get_user_input("Enter Amount A: ")
            elif self.current_selection == 3:
                self.pool_amount_b = await self.get_user_input("Enter Amount B: ")
            elif self.current_selection == 4:
                await self.create_liquidity_pool()
            elif self.current_selection == 5:
                self.current_menu = "main"
                self.current_selection = 0

    async def get_user_input(self, prompt):
        curses.echo()
        self.stdscr.addstr(curses.LINES - 1, 0, prompt)
        self.stdscr.refresh()
        input_win = curses.newwin(1, curses.COLS - len(prompt), curses.LINES - 1, len(prompt))
        input_win.refresh()
        curses.curs_set(1)
        user_input = input_win.getstr().decode('utf-8')
        curses.noecho()
        curses.curs_set(0)
        return user_input

    async def create_token(self):
        try:
            result = await self.blockchain.create_zk_token(self.blockchain.wallet.address, self.token_name, int(self.token_supply))
            if result:
                self.show_message(f"Token '{self.token_name}' created successfully!")
            else:
                self.show_message("Failed to create token.")
        except Exception as e:
            self.show_message(f"Error creating token: {str(e)}")
        self.token_name = ""
        self.token_supply = ""
        self.current_menu = "main"
        self.current_selection = 0

    async def create_liquidity_pool(self):
        try:
            pool_id = f"{self.pool_token_a}_{self.pool_token_b}"
            result = await self.blockchain.add_liquidity(
                self.blockchain.wallet.address,
                pool_id,
                Decimal(self.pool_amount_a),
                Decimal(self.pool_amount_b)
            )
            if result:
                self.show_message(f"Liquidity pool for {self.pool_token_a}/{self.pool_token_b} created successfully!")
            else:
                self.show_message("Failed to create liquidity pool.")
        except Exception as e:
            self.show_message(f"Error creating liquidity pool: {str(e)}")
        self.pool_token_a = ""
        self.pool_token_b = ""
        self.pool_amount_a = ""
        self.pool_amount_b = ""
        self.current_menu = "main"
        self.current_selection = 0

    async def view_tokens(self):
        tokens = await self.blockchain.get_tradable_assets()
        self.show_scrollable_window("Available Tokens", tokens)

    async def view_liquidity_pools(self):
        pools = await self.blockchain.get_liquidity_pools()
        pool_info = [f"{pool_id}: {pool.balance_a} {pool.token_a} / {pool.balance_b} {pool.token_b}" for pool_id, pool in pools.items()]
        self.show_scrollable_window("Liquidity Pools", pool_info)

    def show_scrollable_window(self, title, items):
        height, width = self.stdscr.getmaxyx()
        win = curses.newwin(height - 4, width - 4, 2, 2)
        win.box()
        win.addstr(0, 2, title, curses.A_BOLD)
        
        max_lines = height - 6
        current_line = 0
        
        while True:
            win.clear()
            win.box()
            win.addstr(0, 2, title, curses.A_BOLD)
            
            for i in range(current_line, min(current_line + max_lines, len(items))):
                win.addstr(i - current_line + 1, 2, items[i])
            
            win.refresh()
            key = win.getch()
            
            if key == ord('q'):
                break
            elif key == curses.KEY_UP and current_line > 0:
                current_line -= 1
            elif key == curses.KEY_DOWN and current_line < len(items) - max_lines:
                current_line += 1

    def show_message(self, message):
        height, width = self.stdscr.getmaxyx()
        win = curses.newwin(5, width - 10, height // 2 - 2, 5)
        win.box()
        win.addstr(2, 2, message, curses.color_pair(3))
        win.refresh()
        win.getch()

    async def run(self):
        while True:
            self.draw_menu()
            await self.handle_input()

def main(stdscr):
    ui = TokenManagementUI(stdscr, blockchain_interface)
    asyncio.run(ui.run())

if __name__ == "__main__":
    curses.wrapper(main)