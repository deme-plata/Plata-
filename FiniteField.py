import sympy as sp

class FieldElement:
    def __init__(self, value, field):
        if isinstance(value, FieldElement):
            if value.field != field:
                raise ValueError("FieldElement must be from the same field")
            self.value = value.value
        else:
            self.value = value % field.modulus
        self.field = field

    def __mod__(self, other):
        if isinstance(other, int):
            return FieldElement(self.value % other, self.field)
        elif isinstance(other, FieldElement):
            if self.field != other.field:
                raise TypeError("Cannot mod elements from different fields")
            return FieldElement(self.value % other.value, self.field)
        return NotImplemented

    def __pow__(self, exponent):
        result = FieldElement(1, self.field)
        base = self
        if isinstance(exponent, FieldElement):
            exponent = exponent.value
        exponent = exponent % (self.field.modulus - 1)  # Ensure exponent is within field order
        while exponent > 0:
            if exponent & 1:
                result *= base
            exponent >>= 1
            base *= base
        return result

    def __add__(self, other):
        if isinstance(other, int):
            other = FieldElement(other, self.field)
        if self.field != other.field:
            raise TypeError("Cannot add elements from different fields")
        return FieldElement((self.value + other.value) % self.field.modulus, self.field)

    def __sub__(self, other):
        if isinstance(other, int):
            other = FieldElement(other, self.field)
        if self.field != other.field:
            raise TypeError("Cannot subtract elements from different fields")
        return FieldElement((self.value - other.value) % self.field.modulus, self.field)

    def __mul__(self, other):
        if isinstance(other, int):
            other = FieldElement(other, self.field)
        if self.field != other.field:
            raise TypeError("Cannot multiply elements from different fields")
        result_value = (self.value * other.value) % self.field.modulus
        return FieldElement(result_value, self.field)

    def __truediv__(self, other):
        if isinstance(other, int):
            other = FieldElement(other, self.field)
        if self.field != other.field:
            raise TypeError("Cannot divide elements from different fields")
        return self * other.inverse()
    def inverse(self):
        if self.value == 0:
            raise ZeroDivisionError("Cannot invert zero in a finite field")
        
        # Check if the GCD of the value and the modulus is 1
        gcd = sp.gcd(self.value, self.field.modulus)
        if gcd != 1:
            raise ValueError(f"Inverse does not exist since gcd({self.value}, {self.field.modulus}) = {gcd}")
        
        # Using sympy to compute the inverse
        inv_value = sp.mod_inverse(self.value, self.field.modulus)
        
        # Verifying the inverse
        product = (self.value * inv_value) % self.field.modulus
        print(f"[DEBUG] self.value: {self.value}, inv_value: {inv_value}, product: {product}")
        
        assert product == 1, "Inverse calculation failed"
        return FieldElement(inv_value, self.field)


    def __neg__(self):
        return FieldElement(-self.value % self.field.modulus, self.field)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        elif isinstance(other, FieldElement):
            return self.value == other.value and self.field == other.field
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, FieldElement):
            if self.field != other.field:
                raise TypeError("Cannot compare elements from different fields")
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        return NotImplemented

    def __repr__(self):
        return f"FieldElement_{self.field.modulus}({self.value})"
    
    def to_int(self):
        """Convert the FieldElement to an integer."""
        return self.value

    def __int__(self):
        """Override the int() function to work with FieldElement."""
        return self.to_int()


class FiniteField:
    def __init__(self, modulus: int, security_level: int):
        if not isinstance(modulus, int) or modulus <= 1:
            raise ValueError("Modulus must be a positive integer greater than 1")
        self.modulus = modulus

        if not isinstance(security_level, int):
            raise TypeError("security_level must be an integer")
        self._security_level = security_level

    def get_security_level(self) -> int:
        """
        Returns the security level as an integer.
        """
        return self._security_level


    def element(self, value):
        if isinstance(value, FieldElement):
            if value.field != self:
                raise ValueError("FieldElement must belong to the same finite field")
            return value
        return FieldElement(value, self)

    def add(self, a, b):
        return self.element((self.element(a).value + self.element(b).value) % self.modulus)

    def sub(self, a, b):
        return self.element((self.element(a).value - self.element(b).value) % self.modulus)

    def mul(self, a, b):
        return self.element((self.element(a).value * self.element(b).value) % self.modulus)

    def div(self, a, b):
        return self.mul(a, self.inv(b))

    def exp(self, base, exponent):
        base = self.element(base)
        if isinstance(exponent, FieldElement):
            exponent = exponent.value
        return self.element(pow(base.value, exponent, self.modulus))

    def inv(self, a):
        return self.element(pow(self.element(a).value, self.modulus - 2, self.modulus))

    def __repr__(self):
        return f"FiniteField(modulus={self.modulus})"
