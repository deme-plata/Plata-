import curses
from your_blockchain_module import start_mining, stop_mining, get_mining_stats

mining_active = False

def draw_mining_status(stdscr, mining_stats):
    global mining_active
    stdscr.addstr(11, 1, "Mining", curses.A_BOLD)
    status = "Active" if mining_active else "Inactive"
    stdscr.addstr(12, 1, f"Status: {status}")
    stdscr.addstr(13, 1, f"Hash Rate: {mining_stats['hash_rate']} H/s")
    stdscr.addstr(14, 1, f"Blocks Mined: {mining_stats['blocks_mined']}")
    stdscr.addstr(15, 1, "Press 'm' to start/stop mining")

def handle_mining_input(stdscr):
    global mining_active
    key = stdscr.getch()
    if key == ord('m'):
        if mining_active:
            stop_mining()
            mining_active = False
        else:
            start_mining()
            mining_active = True

async def get_real_time_mining_stats():
    while True:
        mining_stats = await get_mining_stats()
        yield mining_stats
        await asyncio.sleep(1)  # Update every second