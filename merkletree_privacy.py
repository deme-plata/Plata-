import random
from hashlib import sha256
from typing import List
import traceback

# Set a consistent seed for reproducibility
random.seed(42)

class MerkleTree:
    def __init__(self, leaves: List[int]):
        try:
            if not leaves:
                raise ValueError("Leaves cannot be an empty list")

            # Introduce a consistent random number to each leaf
            random.seed(42)  # Consistent seed
            self.leaves = [leaf + random.randint(0, 1000) for leaf in leaves]

            # Ensure all leaves are integers
            assert all(isinstance(leaf, int) for leaf in self.leaves), "[MERKLE TREE] All leaves must be integers"

            # If the number of leaves is odd, duplicate the last leaf to make it even
            if len(self.leaves) % 2 != 0:
                print(f"[MERKLE TREE] Number of leaves is odd, duplicating last leaf: {self.leaves[-1]}")
                self.leaves.append(self.leaves[-1])

            self.tree = self._build_tree()
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree.__init__: {e}")
            traceback.print_exc()
            raise

    def _build_tree(self) -> List[List[int]]:
        try:
            tree = [self.leaves]  # Start with the leaves as the first level
            print(f"[MERKLE TREE] Initial leaves: {self.leaves}")

            while len(tree[-1]) > 1:
                level = []
                for i in range(0, len(tree[-1]), 2):
                    left = tree[-1][i]
                    right = tree[-1][i + 1]
                    hashed_value = self._hash(left, right)
                    level.append(hashed_value)
                    print(f"[MERKLE TREE] Hashing {left} and {right} -> {hashed_value}")
                
                # Assert the new level is half the size of the previous level
                assert len(level) == len(tree[-1]) // 2, "[MERKLE TREE] Incorrect level size after hashing"
                
                # Cross-verify the root at each level
                if len(level) == 1:
                    expected_root = level[0]
                    actual_root = self._compute_root_from_leaves(tree[0])
                    assert expected_root == actual_root, f"[MERKLE TREE] Root mismatch: expected {expected_root}, got {actual_root}"
                
                tree.append(level)
                print(f"[MERKLE TREE] Completed level: {level}")

            print(f"[MERKLE TREE] Final Merkle tree levels: {tree}")
            return tree
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree._build_tree: {e}")
            traceback.print_exc()
            raise

    def _hash(self, left: int, right: int) -> int:
        """
        Hashes two integers together using SHA-256.
        """
        try:
            # Ensure that both inputs are integers
            assert isinstance(left, int), f"[MERKLE TREE HASH] Left input is not an integer: {left}"
            assert isinstance(right, int), f"[MERKLE TREE HASH] Right input is not an integer: {right}"

            # Combine the integers as strings and encode them for hashing
            combined = f"{left}{right}".encode()

            # Generate the SHA-256 hash and convert the hexadecimal result to an integer
            hashed_hex = sha256(combined).hexdigest()
            hashed_int = int(hashed_hex, 16)

            # Print debug information
            print(f"[MERKLE TREE HASH] Hashing ({left}, {right}) -> {hashed_int} (Hex: {hashed_hex})")

            return hashed_int
        except AssertionError as ae:
            print(f"[ASSERTION ERROR] {ae}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree._hash: {e}")
            traceback.print_exc()
            raise

    def _compute_root_from_leaves(self, leaves: List[int]) -> int:
        """
        Computes the Merkle root from the initial leaves.
        This method simulates the construction of the Merkle tree to verify the consistency of the computed root.
        """
        try:
            current_level = leaves
            while len(current_level) > 1:
                next_level = []
                for i in range(0, len(current_level), 2):
                    left = current_level[i]
                    right = current_level[i + 1]
                    next_level.append(self._hash(left, right))
                current_level = next_level
            return current_level[0]
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree._compute_root_from_leaves: {e}")
            traceback.print_exc()
            raise

    def get_root(self) -> int:
        """
        Retrieves the root of the Merkle Tree.
        """
        try:
            root = self.tree[-1][0]
            assert isinstance(root, int), "[MERKLE TREE] Root must be an integer"
            print(f"[MERKLE TREE] Final computed root: {root}")
            return root
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree.get_root: {e}")
            traceback.print_exc()
            raise
    def get_proof(self, index: int) -> List[int]:
        try:
            proof = [self.leaves[index]]  # Include the leaf value
            current_index = index
            for level in self.tree[:-1]:  # Exclude the root level
                sibling_index = current_index ^ 1  # Toggle the last bit to find the sibling index
                if sibling_index < len(level):
                    sibling = level[sibling_index]
                    proof.append(sibling)
                current_index //= 2
            return proof
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree.get_proof: {e}")
            traceback.print_exc()
            raise




    def verify_proof(self, index: int, value: int, proof: List[int]) -> bool:
        """
        Verifies a proof for a given leaf value and its index.
        Recomputes the root using the provided proof and compares it with the actual root.
        """
        try:
            computed_hash = value
            for i, sibling in enumerate(proof):
                assert isinstance(sibling, int), "[MERKLE TREE VERIFY] Each proof element must be an integer"
                if (index >> i) & 1:
                    computed_hash = self._hash(sibling, computed_hash)
                else:
                    computed_hash = self._hash(computed_hash, sibling)
                print(f"[MERKLE TREE VERIFY] Level {i+1}, computed_hash: {computed_hash}, sibling: {sibling}")

            # Compare the recomputed root with the actual root
            assert isinstance(computed_hash, int), "[MERKLE TREE VERIFY] Computed hash must be an integer"
            root = self.get_root()
            is_valid = computed_hash == root
            print(f"[MERKLE TREE VERIFY] Computed root: {computed_hash}, Expected root: {root}, Valid: {is_valid}")
            return is_valid
        except Exception as e:
            print(f"[ERROR] Exception in MerkleTree.verify_proof: {e}")
            traceback.print_exc()
            raise
    def print_tree(self):
        print("[MERKLE TREE] Tree structure:")
        for i, level in enumerate(self.tree):
            print(f"Level {i}: {level}")