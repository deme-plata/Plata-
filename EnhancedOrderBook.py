from decimal import Decimal
from typing import Dict, List
from Order import Order
from typing import Dict, List, Tuple
from collections import defaultdict
from typing import List, Optional  # Add Optional here
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
from Order import Order
from collections import defaultdict

class EnhancedOrderBook:
    def __init__(self):
        self.bids: Dict[str, Dict[Decimal, List[Order]]] = defaultdict(lambda: defaultdict(list))
        self.asks: Dict[str, Dict[Decimal, List[Order]]] = defaultdict(lambda: defaultdict(list))
        self.orders: Dict[str, Order] = {}  # Dictionary to store orders with order_id as key

    def add_order(self, order: Order):
        pair = f"{order.from_currency}_{order.to_currency}"
        if order.order_type == 'buy':
            self.bids[pair][order.price].append(order)
        else:
            self.asks[pair][order.price].append(order)
        self.orders[order.id] = order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            order = self.orders.pop(order_id)
            pair = f"{order.from_currency}_{order.to_currency}"
            order_list = self.bids if order.order_type == 'buy' else self.asks
            order_list[pair][order.price].remove(order)
            if not order_list[pair][order.price]:
                del order_list[pair][order.price]
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_orders(self) -> List[Order]:
        return list(self.orders.values())

    def remove_order(self, order: Order):
        pair = f"{order.from_currency}_{order.to_currency}"
        if order.order_type == "buy":
            self.bids[pair][order.price].remove(order)
            if not self.bids[pair][order.price]:
                del self.bids[pair][order.price]
        else:
            self.asks[pair][order.price].remove(order)
            if not self.asks[pair][order.price]:
                del self.asks[pair][order.price]

    def get_best_bid(self, pair: str) -> Tuple[Optional[Decimal], Optional[Order]]:
        if pair in self.bids and self.bids[pair]:
            best_price = max(self.bids[pair].keys())
            return best_price, self.bids[pair][best_price][0]
        return None, None

    def get_best_ask(self, pair: str) -> Tuple[Optional[Decimal], Optional[Order]]:
        if pair in self.asks and self.asks[pair]:
            best_price = min(self.asks[pair].keys())
            return best_price, self.asks[pair][best_price][0]
        return None, None
    @property
    def buy_orders(self):
        return self.bids
