from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from Order import Order

class EnhancedOrderBook:
    def __init__(self):
        self._bids: Dict[str, Dict[Decimal, List[Order]]] = defaultdict(lambda: defaultdict(list))
        self._asks: Dict[str, Dict[Decimal, List[Order]]] = defaultdict(lambda: defaultdict(list))
        self._orders: Dict[str, Order] = {}  # Dictionary to store orders with order_id as key

    @property
    def bids(self):
        return self._bids

    @property
    def asks(self):
        return self._asks

    @property
    def orders(self):
        return self._orders

    def add_order(self, order: Order):
        pair = f"{order.from_currency}_{order.to_currency}"
        if order.order_type == 'buy':
            self._bids[pair][order.price].append(order)
        else:
            self._asks[pair][order.price].append(order)
        self._orders[order.id] = order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            order = self._orders.pop(order_id)
            pair = f"{order.from_currency}_{order.to_currency}"
            order_list = self._bids if order.order_type == 'buy' else self._asks
            order_list[pair][order.price].remove(order)
            if not order_list[pair][order.price]:
                del order_list[pair][order.price]
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_orders(self) -> List[Order]:
        return list(self._orders.values())

    def remove_order(self, order: Order):
        pair = f"{order.from_currency}_{order.to_currency}"
        if order.order_type == "buy":
            self._bids[pair][order.price].remove(order)
            if not self._bids[pair][order.price]:
                del self._bids[pair][order.price]
        else:
            self._asks[pair][order.price].remove(order)
            if not self._asks[pair][order.price]:
                del self._asks[pair][order.price]

    def get_best_bid(self, pair: str) -> Tuple[Optional[Decimal], Optional[Order]]:
        if pair in self._bids and self._bids[pair]:
            best_price = max(self._bids[pair].keys())
            return best_price, self._bids[pair][best_price][0]
        return None, None

    def get_best_ask(self, pair: str) -> Tuple[Optional[Decimal], Optional[Order]]:
        if pair in self._asks and self._asks[pair]:
            best_price = min(self._asks[pair].keys())
            return best_price, self._asks[pair][best_price][0]
        return None, None
    @property
    def buy_orders(self):
        return self._bids

    @property
    def sell_orders(self):
        return self._asks

    @property
    def orders(self):
        return self._orders
