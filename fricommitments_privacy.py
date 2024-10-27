from typing import List, Tuple
from merkletree_privacy import MerkleTree
from FiniteField import FiniteField, FieldElement
from finite_field_factory import FiniteFieldFactory
import traceback
from hashlib import sha256
import logging
import traceback

# Set up a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # You can adjust the level as needed

class PolynomialCommitment:
    def __init__(self, coefficients: List[int], field: FiniteField):
        try:
            self.coefficients = []
            for coeff in coefficients:
                if isinstance(coeff, FieldElement) and coeff.field != field:
                    raise ValueError("FieldElement must belong to the same finite field")
                self.coefficients.append(field.element(coeff))  # Ensure all coefficients are FieldElements
            self.field = field
        except Exception as e:
            logger.error(f"[ERROR] Exception in PolynomialCommitment.__init__: {e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            raise

    def evaluate(self, x: int) -> int:
        try:
            result = self.field.element(0)
            x_element = self.field.element(x)
            for i, coeff in enumerate(self.coefficients):
                result = self.field.add(result, self.field.mul(coeff, self.field.exp(x_element, i)))
            logger.debug(f"[POLY EVAL] x: {x}, result: {result.value}")  # Use logger instead of print
            return result.value  # Extract the integer value from the FieldElement
        except Exception as e:
            logger.error(f"[ERROR] Exception in PolynomialCommitment.evaluate: {e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            raise
class FRI:
    def __init__(self, field, domain_size: int = 64):
        try:
            self.field = field
            self.domain_size = domain_size
            logger.info(f"[FRI INIT] Field: {field}, Domain size: {domain_size}")
        except Exception as e:
            logger.error(f"[ERROR] Exception in FRI.__init__: {e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            raise

    def commit(self, polynomial: PolynomialCommitment) -> List[int]:
        try:
            domain = range(self.domain_size)
            logger.debug(f"[FRI COMMIT] Domain size: {self.domain_size}")

            evaluations = [int(polynomial.evaluate(x)) for x in domain]
            logger.debug(f"[FRI COMMIT] Polynomial evaluations: {evaluations}")

            merkle_tree = MerkleTree(evaluations)
            root = merkle_tree.get_root()
            
            logger.info(f"[FRI COMMIT] Final computed Merkle root: {root}")

            return [root]
        except Exception as e:
            logger.error(f"[ERROR] FRI.commit: Error during commitment: polynomial={polynomial.coefficients}, error={e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            raise

    def query(self, polynomial: PolynomialCommitment, x: int) -> Tuple[int, List[int]]:
        try:
            domain = range(self.domain_size)
            evaluations = [int(polynomial.evaluate(d)) for d in domain]
            logger.debug(f"[FRI QUERY] Evaluations: {evaluations}")
            
            merkle_tree = MerkleTree(evaluations)
            
            index = x % self.domain_size
            proof = merkle_tree.get_proof(index)
            
            logger.info(f"[FRI QUERY] x: {x}, index: {index}")
            logger.debug(f"[FRI QUERY] Generated proof: {proof}")
            logger.debug(f"[FRI QUERY] Merkle root: {merkle_tree.get_root()}")
            
            return proof[0], proof  # Return the leaf value and the entire proof
        except Exception as e:
            logger.error(f"[ERROR] FRI.query: Error during query for x={x}: {e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            raise

    def verify(self, commitment: List[int], x: int, y: int, proof: List[int]) -> bool:
        try:
            index = x % self.domain_size
            logger.info(f"[FRI VERIFY] Verifying for x={x}, y={y}, index={index}")

            leaf_value = proof[0]
            current_hash = leaf_value
            for i, sibling in enumerate(proof[1:]):
                logger.debug(f"[FRI VERIFY] Level {i + 1}, current value: {current_hash}, sibling: {sibling}")
                
                if index % 2 == 0:
                    current_hash = self._hash(current_hash, sibling)
                else:
                    current_hash = self._hash(sibling, current_hash)
                
                logger.debug(f"[FRI VERIFY] Hashed result: {current_hash}")
                index //= 2

            computed_root = current_hash
            expected_root = commitment[0]
            
            logger.debug(f"[FRI VERIFY] Computed root: {computed_root}")
            logger.debug(f"[FRI VERIFY] Expected commitment root: {expected_root}")
            
            is_valid = computed_root == expected_root and leaf_value == y
            logger.info(f"[FRI VERIFY] Proof is {'valid' if is_valid else 'invalid'}")

            return is_valid
        except Exception as e:
            logger.error(f"[ERROR] FRI.verify: Error during verification for x={x}, y={y}: {e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            return False
            
    def _hash(self, left: int, right: int) -> int:
        combined = f"{left}{right}".encode()
        return int(sha256(combined).hexdigest(), 16)

    def verify_proof(self, commitment, x, y, proof) -> bool:
        try:
            # Delegate to the existing verify method
            is_valid = self.verify(commitment, x, y, proof)
            logger.info(f"[FRI VERIFY PROOF] Verification result: {'Valid' if is_valid else 'Invalid'}")
            return is_valid
        except Exception as e:
            logger.error(f"[ERROR] FRI.verify_proof: Error during proof verification for x={x}, y={y}: {e}")
            logger.debug(traceback.format_exc())  # Log the traceback for debugging
            return False
