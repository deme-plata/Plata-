import random
from hashlib import sha256
from typing import List, Tuple

class CircularSTARK:
    def __init__(self, field_size: int):
        self.field_size = field_size

    def hash(self, *args: int) -> int:
        return int(sha256(''.join(map(str, args)).encode()).hexdigest(), 16) % self.field_size

    def generate_proof(self, secret: int, public_input: int) -> Tuple[List[int], List[int]]:
        # This is a simplified version of STARK proof generation
        # In a real implementation, this would involve polynomial commitments,
        # Merkle trees, and FRI (Fast Reed-Solomon Interactive Oracle Proofs)
        r = random.randint(0, self.field_size - 1)
        t = (secret * r) % self.field_size
        commitment = self.hash(t)
        challenge = self.hash(commitment, public_input)
        response = (r + challenge * secret) % self.field_size
        return ([commitment], [response])

    def verify_proof(self, public_input: int, proof: Tuple[List[int], List[int]]) -> bool:
        # This is a simplified version of STARK proof verification
        [commitment] = proof[0]
        [response] = proof[1]
        challenge = self.hash(commitment, public_input)
        t = (response * public_input) % self.field_size
        return commitment == self.hash(t)