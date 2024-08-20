import qrcode
import json
from typing import Dict
from SecureHybridZKStark import SecureHybridZKStark
from FiniteField import FiniteField, FieldElement  # Import FieldElement here
class SecureQRSystem:
    def __init__(self, security_level: int = 20):
        self.zk_system = SecureHybridZKStark(security_level)
        self.field = self.zk_system.field

    def generate_qr_code(self, wallet_address: str, coin_type: str) -> str:
        # Generate a random secret
        secret = self.field.random_element()

        # Create a public input by hashing the wallet address and coin type
        public_input = self.zk_system.hash(wallet_address, coin_type)

        # Generate ZKP
        stark_proof, snark_proof = self.zk_system.prove(secret.value, public_input.value)

        # Convert FieldElement objects to integers for serialization
        qr_data = {
            "address": wallet_address,
            "coin_type": coin_type,
            "public_input": public_input.value,  # Use the integer value of the FieldElement
            "stark_proof": self._serialize_proof(stark_proof),
            "snark_proof": self._serialize_proof(snark_proof)
        }

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(qr_data))
        qr.make(fit=True)

        # Create an image from the QR code
        img = qr.make_image(fill_color="black", back_color="white")

        # Save the image and return the filename
        filename = f"{coin_type}_{wallet_address[:8]}.png"
        img.save(filename)
        return filename

    def verify_qr_code(self, qr_data: Dict) -> bool:
        public_input = qr_data["public_input"]
        stark_proof = self._deserialize_proof(qr_data["stark_proof"])
        snark_proof = self._deserialize_proof(qr_data["snark_proof"])

        # Verify the ZKP
        return self.zk_system.verify(public_input, (stark_proof, snark_proof))

    def _serialize_proof(self, proof):
        """Helper method to convert proof elements to serializable format."""
        return tuple([elem.value if isinstance(elem, FieldElement) else elem for elem in proof])

    def _deserialize_proof(self, proof):
        """Helper method to convert serialized proof elements back to FieldElements."""
        return tuple([self.field.element(elem) if isinstance(elem, int) else elem for elem in proof])

# Usage example:
# qr_system = SecureQRSystem()
# qr_file = qr_system.generate_qr_code("0x1234567890abcdef", "ETH")
# print(f"QR code saved as {qr_file}")