from decimal import Decimal
from typing import Dict, List
from pydantic import BaseModel, validator, Field, model_validator
from pydantic import BaseModel, field_validator  # Use field_validator for Pydantic V2
import uuid

class Order(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Ensure each order has a unique id

    user_id: str
    type: str
    order_type: str
    pair: str
    base_currency: str
    quote_currency: str
    amount: Decimal
    price: Decimal
    from_currency: str
    to_currency: str
    def to_json_serializable(self):
        return {
            "user_id": self.user_id,
            "type": self.type,
            "order_type": self.order_type,
            "pair": self.pair,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "amount": str(self.amount),
            "price": str(self.price),
            "from_currency": self.from_currency,
            "to_currency": self.to_currency,
        }


    @field_validator('order_type')
    def validate_order_type(cls, value):
        if value not in ['buy', 'sell']:
            raise ValueError('order_type must be either "buy" or "sell"')
        return value

    @field_validator('amount', 'price')
    def validate_positive(cls, value, info):
        if value <= 0:
            raise ValueError(f'{info.field_name} must be positive')
        return value

    def __eq__(self, other):
        if not isinstance(other, Order):
            return False
        return (self.id == other.id and
                self.user_id == other.user_id and
                self.type == other.type and
                self.order_type == other.order_type and
                self.pair == other.pair and
                self.amount == other.amount and
                self.price == other.price and
                self.base_currency == other.base_currency and
                self.quote_currency == other.quote_currency and
                self.from_currency == other.from_currency and
                self.to_currency == other.to_currency)


    def __hash__(self):
        return hash((self.id, self.user_id, self.order_type, self.from_currency, self.to_currency, self.amount, self.price, self.status, self.created_at))
