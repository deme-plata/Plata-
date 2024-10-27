import random
from typing import List, Tuple, Generator
from hashlib import sha256
import numpy as np
from STARK import STARK, calculate_field_size
from FiniteField import FiniteField, FieldElement
from finite_field_factory import FiniteFieldFactory


import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # You can change this level to control logging output

class PolynomialCommitment:
    def __init__(self, coefficients: List[int], field: FiniteField):
        self.coefficients = []
        for coeff in coefficients:
            if isinstance(coeff, FieldElement) and coeff.field != field:
                raise ValueError("FieldElement must belong to the same finite field")
            self.coefficients.append(field.element(coeff))  # Ensure all coefficients are FieldElements
        self.field = field

    def evaluate(self, x: int) -> int:
        result = self.field.element(0)
        x_element = self.field.element(x)
        for i, coeff in enumerate(self.coefficients):
            result = self.field.add(result, self.field.mul(coeff, self.field.exp(x_element, i)))
        
        # Use logger instead of print
        logger.debug(f"[POLY EVAL] x: {x}, result: {result.value}")
        return result.value
class AdvancedCircularSTARK:
    def __init__(self, finite_field):
        self.field = finite_field  # finite_field should be a FiniteField instance
        self.fri = FRI(self.field)

    def hash(self, *args: int) -> int:
        hash_value = int(sha256(''.join(map(str, args)).encode()).hexdigest(), 16) % self.field.modulus
        # Replace print with logger
        logger.debug(f"Hashing args {args}: Hash={hash_value}")
        return hash_value
        
    def generate_proof(self, secret: int, public_input: int) -> Tuple[List[int], List[int], List[Tuple[int, List[int]]]]:
        # Generate polynomial coefficients including the secret and random coefficients
        coefficients = [self.field.element(secret)] + [self.field.element(random.randint(0, self.field.modulus - 1)) for _ in range(3)]
        polynomial = PolynomialCommitment(np.array(coefficients), self.field)
        # Replace print with logger
        logger.debug(f"Generated polynomial coefficients: {coefficients}")

        # Commit the polynomial using FRI
        commitment = self.fri.commit(polynomial)
        # Replace print with logger
        logger.debug(f"Commitment: {commitment}")

        # Determine the number of queries and generate the domain
        num_queries = 10
        domain = list(self.fri.generate_domain(min(self.field.modulus, 1000000)))

        # Generate challenge points
        commitment_element = self.field.element(commitment[0])
        challenge_points = [domain[self.hash(commitment_element.value, public_input, i) % len(domain)] for i in range(num_queries)]

        # Generate FRI proofs for the challenge points
        fri_proofs = [self.fri.query(polynomial, x.value) for x in challenge_points]

        # Generate the final challenge and the response
        challenge = self.hash(commitment_element.value, public_input, *[cp.value for cp in challenge_points])
        response = self.field.add(self.field.element(secret), self.field.element(challenge)).value

        # Replace print with logger
        logger.debug(f"Generated proof: Commitment={commitment}, Response={response}, FRI Proofs={fri_proofs}")
        return (commitment, [response], fri_proofs)

    def verify_proof(self, public_input: int, proof: Tuple[List[int], List[int], List[Tuple[int, List[int]]]]) -> bool:
        commitment, [response], fri_proofs = proof
        
        commitment_element = self.field.element(commitment[0])

        challenge_points = [
            self.hash(commitment_element.value, public_input, i) % self.fri.domain_size
            for i in range(len(fri_proofs))
        ]

        all_fri_proofs_valid = all(
            self.fri.verify(commitment, x, y, proof)
            for x, (y, proof) in zip(challenge_points, fri_proofs)
        )

        challenge = self.hash(commitment_element.value, public_input, *challenge_points)
        computed_public_input = self.field.sub(self.field.element(response), self.field.element(challenge))

        return all_fri_proofs_valid and computed_public_input.value == public_input

import numpy as np

import traceback
from typing import Tuple
import sympy as sp
from FiniteField import FiniteField, FieldElement
import traceback
from typing import Tuple
import sympy as sp
from FiniteField import FiniteField, FieldElement
class ZKSnark:
    def __init__(self, field_size, security_level):
        try:
            self.field = FiniteField(field_size, security_level)
            self.setup_params = self.setup()
            # Replace print with logger
            logger.info("[INIT] ZKSnark initialized successfully")
        except Exception as e:
            # Replace print with logger
            logger.error(f"[INIT ERROR] Failed to initialize ZKSnark: {str(e)}")
            logger.debug(traceback.format_exc())

    def setup(self):
        try:
            self.toxic_waste = self.field.element(12345)
            g1 = self.field.element(2)
            g2 = self.field.element(5)
            # Replace print with logger
            logger.info("[SETUP] Setup completed with g1 and g2 initialized")
            return g1, g2
        except Exception as e:
            # Replace print with logger
            logger.error(f"[SETUP ERROR] Failed during setup: {str(e)}")
            logger.debug(traceback.format_exc())

    def prove(self, secret: int, public_input: int) -> Tuple[FieldElement, FieldElement, FieldElement]:
        try:
            g1, g2 = self.setup_params
            r = self.field.element(secret + public_input)
            secret_element = self.field.element(secret)
            public_input_element = self.field.element(public_input)

            # Intermediate step debug outputs
            logger.debug(f"[PROVE DEBUG] r: {r}, secret_element: {secret_element}, public_input_element: {public_input_element}")

            A = self.mod_exp(g1, secret_element)
            B = self.mod_exp(g2, r)
            C = self.mod_add(
                self.mod_mul(secret_element, public_input_element),
                self.mod_mul(r, self.toxic_waste)
            )

            # Debugging information
            logger.debug(f"[SNARK PROVE] secret: {secret}, public_input: {public_input}")
            logger.debug(f"[SNARK PROVE] A: {A}, B: {B}, C: {C}, r: {r}, g1: {g1}, g2: {g2}")

            return A, B, C
        except Exception as e:
            # Replace print with logger
            logger.error(f"[PROVE ERROR] Failed during proof generation: {str(e)}")
            logger.debug(traceback.format_exc())

    def verify(self, public_input: int, proof: Tuple[FieldElement, FieldElement, FieldElement]) -> bool:
        try:
            A, B, C = proof
            g1, g2 = self.setup_params
            public_input_element = self.field.element(public_input)

            # Field operations
            exp_g1_C = self.mod_exp(g1, C)
            A_exp_g1_pub = self.mod_mul(A, self.mod_exp(g1, public_input_element))
            exp_g2_C = self.mod_exp(g2, C)
            B_exp_g2_toxic = self.mod_mul(B, self.mod_exp(g2, self.toxic_waste))

            # SymPy operations for comparison
            modulus = self.field.modulus
            sympy_exp_g1_C = pow(int(g1), int(C), modulus)
            sympy_A_exp_g1_pub = (int(A) * pow(int(g1), int(public_input_element), modulus)) % modulus
            sympy_exp_g2_C = pow(int(g2), int(C), modulus)
            sympy_B_exp_g2_toxic = (int(B) * pow(int(g2), int(self.toxic_waste), modulus)) % modulus

            logger.debug(f"[SYMPY DEBUG] sympy_exp_g1_C: {sympy_exp_g1_C}, sympy_A_exp_g1_pub: {sympy_A_exp_g1_pub}")
            logger.debug(f"[SYMPY DEBUG] sympy_exp_g2_C: {sympy_exp_g2_C}, sympy_B_exp_g2_toxic: {sympy_B_exp_g2_toxic}")

            logger.debug(f"[FIELD DEBUG] exp_g1_C: {exp_g1_C}, A_exp_g1_pub: {A_exp_g1_pub}")
            logger.debug(f"[FIELD DEBUG] exp_g2_C: {exp_g2_C}, B_exp_g2_toxic: {B_exp_g2_toxic}")

            # Compare the results
            check1 = int(exp_g1_C) == sympy_exp_g1_C and int(A_exp_g1_pub) == sympy_A_exp_g1_pub
            check2 = int(exp_g2_C) == sympy_exp_g2_C and int(B_exp_g2_toxic) == sympy_B_exp_g2_toxic

            if not check1:
                logger.warning(f"[VERIFY FAILURE] Check 1 failed: exp_g1_C ({exp_g1_C}) != sympy_exp_g1_C ({sympy_exp_g1_C}) or A_exp_g1_pub ({A_exp_g1_pub}) != sympy_A_exp_g1_pub ({sympy_A_exp_g1_pub})")
            if not check2:
                logger.warning(f"[VERIFY FAILURE] Check 2 failed: exp_g2_C ({exp_g2_C}) != sympy_exp_g2_C ({sympy_exp_g2_C}) or B_exp_g2_toxic ({B_exp_g2_toxic}) != sympy_B_exp_g2_toxic ({sympy_B_exp_g2_toxic})")

            return check1 and check2
        except Exception as e:
            logger.error(f"[VERIFY ERROR] Failed during verification: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def mod_exp(self, base, exponent):
        try:
            result = self.field.exp(base, exponent)
            assert result is not None, "[MOD EXP] Result is None"
            return result
        except Exception as e:
            logger.error(f"[MOD EXP ERROR] Failed during modular exponentiation: {str(e)}")
            logger.debug(traceback.format_exc())

    def mod_mul(self, a, b):
        try:
            result = self.field.mul(a, b)
            assert result is not None, "[MOD MUL] Result is None"
            return result
        except Exception as e:
            logger.error(f"[MOD MUL ERROR] Failed during modular multiplication: {str(e)}")
            logger.debug(traceback.format_exc())

    def mod_add(self, a, b):
        try:
            result = self.field.add(a, b)
            assert result is not None, "[MOD ADD] Result is None"
            return result
        except Exception as e:
            logger.error(f"[MOD ADD ERROR] Failed during modular addition: {str(e)}")
            logger.debug(traceback.format_exc())

        
class HybridZKStark:
    def __init__(self, field_size: int, security_level: int):
        self.field = FiniteField(field_size, security_level)
        self.stark = STARK(security_level, self.field)
        self.snark = ZKSnark(field_size, security_level)

    def prove(self, secret: int, public_input: int) -> Tuple[Tuple, Tuple]:
        secret_elem = self.field.element(secret)
        public_input_elem = self.field.element(public_input)
        stark_proof = self.stark.prove(secret_elem, public_input_elem)
        snark_proof = self.snark.prove(secret, public_input)
        return stark_proof, snark_proof

    def verify(self, public_input: int, proof: Tuple[Tuple, Tuple]) -> bool:
        public_input_elem = self.field.element(public_input)
        stark_proof, snark_proof = proof
        stark_valid = self.stark.verify_proof(public_input_elem, stark_proof)
        snark_valid = self.snark.verify(public_input, snark_proof)
        return stark_valid and snark_valid


    def _ensure_field_element(self, value):
        """
        Ensure the value is a FieldElement belonging to this instance's FiniteField.
        If it is not, convert it; if it is, verify it belongs to the correct field.
        """
        if isinstance(value, FieldElement):
            if value.field != self.field:
                raise ValueError("FieldElement must belong to the same finite field.")
            return value
        return self.field.element(value)




def calculate_security_parameters(desired_security_level: int = 128):
    field_size = 2**desired_security_level - 59

    num_queries = 2 * desired_security_level
    expansion_factor = 4
    merkle_depth = desired_security_level
    snark_curve_size = 2 * desired_security_level

    return {
        "field_size": field_size,
        "num_queries": num_queries,
        "expansion_factor": expansion_factor,
        "merkle_depth": merkle_depth,
        "snark_curve_size": snark_curve_size
    }


class PolynomialCommitment:
    def __init__(self, coefficients: List[int], field: FiniteField):
        self.coefficients = coefficients
        self.field = field

    def evaluate(self, x: int) -> int:
        """
        Evaluate the polynomial using Horner's method for numerical stability.
        """
        result = 0
        for coeff in reversed(self.coefficients):
            result = self.field.mul(result, x)
            result = self.field.add(result, coeff)
        
        # Replace print with logger
        logger.debug(f"Evaluating polynomial at x={x}: Result={result}")
        return result
class SecureHybridZKStark:
    def __init__(self, security_level, field=None):
        if not isinstance(security_level, int):
            raise TypeError("security_level must be an integer")

        self.security_level = security_level
        self.field_size = calculate_field_size(security_level)
        self.field = field if field else FiniteField(self.field_size, security_level)

        # Use the same field for both STARK and SNARK
        self.stark = STARK(security_level, self.field)
        self.snark = ZKSnark(self.field.modulus, security_level)

    def prove(self, secret: int, public_input: int) -> Tuple[Tuple, Tuple]:
        # Replace print with logger
        logger.info(f"[PROVE] Starting proof generation with secret: {secret}, public_input: {public_input}")
        secret_elem = self.field.element(secret)
        public_input_elem = self.field.element(public_input)
        
        logger.info("[PROVE] Generating STARK proof...")
        stark_proof = self.stark.prove(secret_elem.value, public_input_elem.value)
        logger.debug(f"[PROVE] STARK proof generated: {stark_proof}")
        
        logger.info("[PROVE] Generating SNARK proof...")
        snark_proof = self.snark.prove(secret_elem.value, public_input_elem.value)
        logger.debug(f"[PROVE] SNARK proof generated: {snark_proof}")
        
        return stark_proof, snark_proof

    def verify(self, public_input: int, proof: Tuple[Tuple, Tuple]) -> bool:
        stark_proof, snark_proof = proof
        stark_valid = self.stark.verify_proof(public_input, stark_proof)
        snark_valid = self.snark.verify(public_input, snark_proof)
        # Replace print with logger
        logger.info(f"[VERIFY] STARK verification result: {stark_valid}")
        logger.info(f"[VERIFY] SNARK verification result: {snark_valid}")
        return stark_valid and snark_valid

    def log_failure_details(self, stark_valid: bool, snark_valid: bool, stark_proof: Tuple, snark_proof: Tuple):
        """
        Logs detailed error information when verification fails.
        """
        logger.error(f"[ERROR] Verification failed: STARK valid: {stark_valid}, SNARK valid: {snark_valid}")
        logger.debug(f"[ERROR] STARK Proof: {stark_proof}")
        logger.debug(f"[ERROR] SNARK Proof: {snark_proof}")

        # Additional detailed logging for internal state, useful for debugging.
        logger.debug(f"[DEBUG] STARK proof details: {self.extract_stark_details(stark_proof)}")
        logger.debug(f"[DEBUG] SNARK proof details: {self.extract_snark_details(snark_proof)}")

    def extract_stark_details(self, stark_proof: Tuple) -> str:
        """
        Extracts and returns detailed information about the STARK proof for logging purposes.
        """
        polynomial_evaluations, fri_commitment, merkle_proofs = stark_proof

        polynomial_details = f"Polynomial Evaluations: {polynomial_evaluations}"
        fri_details = f"FRI Commitment (Merkle Root): {fri_commitment}"

        merkle_details = "Merkle Proofs:\n"
        for i, proof in enumerate(merkle_proofs):
            merkle_details += f"  Proof {i+1}: {proof}\n"

        stark_details = (
            f"STARK Proof Details:\n"
            f"{polynomial_details}\n"
            f"{fri_details}\n"
            f"{merkle_details}"
        )

        return stark_details

    def extract_snark_details(self, snark_proof: Tuple) -> str:
        """
        Extracts and returns detailed information about the SNARK proof for logging purposes.
        """
        group_element_a, group_element_b, group_element_c = snark_proof

        a_details = f"Group Element A: {group_element_a}"
        b_details = f"Group Element B: {group_element_b}"
        c_details = f"Group Element C: {group_element_c}"

        snark_details = (
            f"SNARK Proof Details:\n"
            f"{a_details}\n"
            f"{b_details}\n"
            f"{c_details}"
        )

        return snark_details

    def hash(self, *args) -> FieldElement:
        """Hash the input arguments and return a FieldElement."""
        concatenated_input = ''.join(map(str, args))
        hash_value = int(sha256(concatenated_input.encode()).hexdigest(), 16)
        return self.field.element(hash_value % self.field.modulus)
def calculate_field_size(security_level: int) -> int:
    if not isinstance(security_level, int):
        raise TypeError("security_level must be an integer")
    if security_level <= 0:
        raise ValueError("security_level must be a positive integer")

    # For this example, let's assume you want the size of the field to be the smallest prime number 
    # greater than or equal to 2^security_level. This is a common approach to map security levels 
    # to finite field sizes.

    def next_prime(n: int) -> int:
        # A simple function to find the next prime number greater than or equal to n
        def is_prime(num: int) -> bool:
            if num <= 1:
                return False
            if num == 2:
                return True
            if num % 2 == 0:
                return False
            p = 3
            while p * p <= num:
                if num % p == 0:
                    return False
                p += 2
            return True

        if n <= 2:
            return 2
        prime_candidate = n if n % 2 != 0 else n + 1
        while not is_prime(prime_candidate):
            prime_candidate += 2
        return prime_candidate

    # Map the security level to the smallest prime greater than or equal to 2^security_level
    field_size = next_prime(2**security_level)
    
    # Replace print with logger
    logger.info(f"The field size for security level {security_level} is {field_size}.")
    return field_size

# Example usage
