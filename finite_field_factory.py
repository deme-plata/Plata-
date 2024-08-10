# finite_field_factory.py
from FiniteField import FiniteField
# Assuming FiniteFieldFactory is supposed to create FiniteField instances
class FiniteFieldFactory:
    @staticmethod
    def get_instance(modulus: int = 340282366920938463463374607431768211455, security_level: int = 128) -> 'FiniteField':
        """
        Creates an instance of the FiniteField class using the provided modulus and security level.
        Default modulus and security level values are provided, but can be overridden.

        :param modulus: The modulus for the finite field.
        :param security_level: The security level associated with the finite field.
        :return: An instance of the FiniteField class.
        """
        return FiniteField(modulus, security_level)

