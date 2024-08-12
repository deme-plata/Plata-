import curses
from your_blockchain_module import get_wallet_balance, get_transaction_history, send_transaction

def draw_wallet_info(stdscr, wallet_info):
    stdscr.addstr(7, 1, "Wallet", curses.A_BOLD)
    stdscr.addstr(8, 1, f"Balance: {wallet_info['balance']} QuantumDAGKnight Coins")
    stdscr.addstr(9, 1, "Press 't' to send a transaction")
    stdscr.addstr(10, 1, "Press 'h' to view transaction history")

def handle_wallet_input(stdscr):
    key = stdscr.getch()
    if key == ord('t'):
        send_transaction_ui(stdscr)
    elif key == ord('h'):
        view_transaction_history(stdscr)

def send_transaction_ui(stdscr):
    curses.echo()
    stdscr.addstr(15, 1, "Enter recipient address: ")
    recipient = stdscr.getstr(15, 26, 64).decode('utf-8')
    stdscr.addstr(16, 1, "Enter amount to send: ")
    amount = float(stdscr.getstr(16, 23, 10).decode('utf-8'))
    curses.noecho()
    
    # Call your send_transaction function here
    result = send_transaction(recipient, amount)
    
    stdscr.addstr(18, 1, f"Transaction result: {result}")
    stdscr.addstr(20, 1, "Press any key to continue...")
    stdscr.getch()

def view_transaction_history(stdscr):
    history = get_transaction_history()
    stdscr.clear()
    stdscr.addstr(1, 1, "Transaction History", curses.A_BOLD)
    for i, tx in enumerate(history, start=2):
        stdscr.addstr(i, 1, f"{tx['date']} - {tx['amount']} to {tx['recipient']}")
    stdscr.addstr(len(history) + 3, 1, "Press any key to return...")
    stdscr.getch()

