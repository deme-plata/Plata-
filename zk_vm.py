import hashlib
from SecureHybridZKStark import SecureHybridZKStark
from vm import SimpleVM  # Import your existing VM

class ZKVM:
    def __init__(self, security_level=2):
        self.vm = SimpleVM()
        self.zk_system = SecureHybridZKStark(security_level=security_level)
        self.zk_proofs = {}

    def execute_contract_with_zk(self, contract_address, input_data, gas_limit, user_id, sender_public_key, signature):
        execution_result = self.vm.execute_contract(contract_address, input_data, gas_limit, user_id, sender_public_key, signature)
        
        if 'error' not in execution_result:
            # Generate ZK proof for the execution
            public_input = self.zk_system.stark.hash(contract_address, str(input_data), str(execution_result['output']))
            secret = self.vm.gas_used  # Use gas_used as the secret
            zk_proof = self.zk_system.prove(secret, public_input)
            self.zk_proofs[contract_address] = zk_proof
            
            execution_result['zk_proof'] = zk_proof
        
        return execution_result

    def verify_zk_proof(self, contract_address, input_data, output_data):
        if contract_address not in self.zk_proofs:
            return False
        
        zk_proof = self.zk_proofs[contract_address]
        public_input = self.zk_system.stark.hash(contract_address, str(input_data), str(output_data))
        return self.zk_system.verify(public_input, zk_proof)

    def create_zk_token(self, creator_address, token_name, total_supply):
        result = self.vm.create_token(creator_address, token_name, total_supply)
        if result:
            public_input = self.zk_system.stark.hash(creator_address, token_name, total_supply)
            zk_proof = self.zk_system.prove(total_supply, public_input)
            self.zk_proofs[token_name] = zk_proof
        return result

    async def zk_transfer_token(self, from_address, to_address, token_address, amount):
        contract = self.vm.get_contract(token_address)
        transfer_result = self.vm.transfer_token(from_address, to_address, token_address, amount)
        if transfer_result:
            public_input = self.zk_system.stark.hash(from_address, to_address, amount)
            secret = int.from_bytes(hashlib.sha256(f"{from_address}{to_address}{amount}".encode()).digest(), 'big')
            transfer_proof = self.zk_system.prove(secret, public_input)
            return transfer_proof
        else:
            raise ValueError("Invalid transfer")

    def verify_zk_token_transfer(self, from_address, to_address, token_name, amount):
        public_input = self.zk_system.stark.hash(from_address, to_address, token_name, amount)
        proof_key = f"{token_name}_{from_address}_{to_address}"
        if proof_key not in self.zk_proofs:
            return False
        
        zk_proof = self.zk_proofs[proof_key]
        return self.zk_system.verify(public_input, zk_proof)

    # Delegate other methods to the underlying VM
    def __getattr__(self, name):
        return getattr(self.vm, name)