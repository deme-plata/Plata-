import asyncio
from your_blockchain_module import get_node_stats  # You'll need to implement this function

async def get_real_time_node_info():
    while True:
        node_stats = await get_node_stats()
        yield node_stats
        await asyncio.sleep(5)  # Update every 5 seconds

def draw_node_info(stdscr, node_stats):
    stdscr.addstr(1, 1, "Node Information", curses.A_BOLD)
    stdscr.addstr(2, 1, f"Node ID: {node_stats['node_id']}")
    stdscr.addstr(3, 1, f"Connected Peers: {node_stats['connected_peers']}")
    stdscr.addstr(4, 1, f"Block Height: {node_stats['block_height']}")
    stdscr.addstr(5, 1, f"Last Block Time: {node_stats['last_block_time']}")

