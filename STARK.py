import random
from typing import List, Tuple
from hashlib import sha256
import numpy as np
from FiniteField import FiniteField, FieldElement
from finite_field_factory import FiniteFieldFactory
from merkletree_privacy import MerkleTree
from fricommitments_privacy import FRI
import traceback
import logging
import traceback

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
        return result.value
        
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed

class STARK:
    def __init__(self, security_level, field=None):
        self.security_level = security_level
        self.field = field if field else FiniteFieldFactory.get_instance(security_level=security_level)
        self.fri = FRI(self.field)

    def prove(self, secret: int, public_input: int) -> Tuple[List[int], List[int], List[Tuple[int, List[int]]]]:
        secret_element = self.field.element(secret)
        public_input_element = self.field.element(public_input)
        
        # Use a deterministic seed based on the public input
        seed = self.hash(public_input_element.value)
        random.seed(seed.value)  # Use the integer value of the FieldElement
        
        # Generate random coefficients deterministically
        random_coeff = self.field.element(random.randint(0, self.field.modulus - 1))
        
        coefficients = [
            public_input_element,
            random_coeff,
            self.field.element(0)  # Placeholder for higher degree terms
        ]
        
        polynomial = PolynomialCommitment(coefficients, self.field)
        
        # Replace print with logger
        logger.debug(f"[STARK PROVE] Secret: {secret}, Public Input: {public_input}")
        logger.debug(f"[STARK PROVE] Coefficients: {[c.value for c in coefficients]}")
        
        commitment = self.fri.commit(polynomial)
        num_queries = 5
        challenge_points = [
            self.field.element(self.hash(commitment[0], public_input_element.value, i).value % self.fri.domain_size)
            for i in range(num_queries)
        ]
        fri_proofs = [self.fri.query(polynomial, x.value) for x in challenge_points]
        challenge = self.hash(commitment[0], public_input_element.value, *[cp.value for cp in challenge_points])
        response = self.field.add(secret_element, self.field.element(challenge))
        
        return (commitment, [response.value], fri_proofs)

    def verify_proof(self, public_input, proof):
        try:
            public_input_element = self.field.element(public_input)
            commitment, [response], fri_proofs = proof
            
            num_queries = len(fri_proofs)
            challenge_points = [
                self.hash(commitment[0], public_input_element.value, i).value % self.fri.domain_size
                for i in range(num_queries)
            ]
            all_fri_proofs_valid = all(
                self.fri.verify(commitment, x, y, proof)
                for x, (y, proof) in zip(challenge_points, fri_proofs)
            )
            
            challenge = self.hash(commitment[0], public_input_element.value, *challenge_points)
            challenge_element = self.field.element(challenge)
            
            # Verify the response
            computed_secret = self.field.sub(self.field.element(response), challenge_element)
            
            # Replace print with logger
            logger.debug(f"[STARK VERIFY] Challenge: {challenge}")
            logger.debug(f"[STARK VERIFY] Response: {response}")
            logger.debug(f"[STARK VERIFY] Computed secret: {computed_secret.value}")
            logger.debug(f"[STARK VERIFY] Public input: {public_input}")
            
            return all_fri_proofs_valid
        except Exception as e:
            logger.error(f"[STARK VERIFY] Error during verification: {e}")
            logger.debug(traceback.format_exc())
            raise

    def hash(self, *args):
        hash_value = int(sha256(''.join(map(str, args)).encode()).hexdigest(), 16)
        return self.field.element(hash_value % self.field.modulus)



def calculate_field_size(security_level: int) -> int:
    return STARK(security_level).field.modulus
