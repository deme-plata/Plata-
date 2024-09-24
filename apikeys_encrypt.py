import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

def generate_key():
    """Generate a new encryption key"""
    return Fernet.generate_key()

def encrypt_data(key, data):
    """Encrypt data with the provided key"""
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data

def decrypt_data(key, encrypted_data):
    """Decrypt data with the provided key"""
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data).decode()
    return decrypted_data

# Load the current .env file
load_dotenv()

# Generate or load the encryption key (this should be stored securely)
encryption_key_path = "encryption_key.key"
if not os.path.exists(encryption_key_path):
    encryption_key = generate_key()
    with open(encryption_key_path, "wb") as key_file:
        key_file.write(encryption_key)
else:
    with open(encryption_key_path, "rb") as key_file:
        encryption_key = key_file.read()

# Encrypt the sensitive keys
alchemy_key_encrypted = encrypt_data(encryption_key, os.getenv("ALCHEMY_KEY"))
blockcypher_key_encrypted = encrypt_data(encryption_key, os.getenv("BLOCKCYPHER_API_KEY"))
zerox_key_encrypted = encrypt_data(encryption_key, os.getenv("ZEROX_API_KEY"))

# Write the encrypted values back to the .env file
with open(".env", "w") as env_file:
    env_file.write(f"P2P_HOST={os.getenv('P2P_HOST')}\n")
    env_file.write(f"P2P_PORT={os.getenv('P2P_PORT')}\n")
    env_file.write(f"SECURITY_LEVEL={os.getenv('SECURITY_LEVEL')}\n")
    env_file.write(f"MAX_PEERS={os.getenv('MAX_PEERS')}\n")
    env_file.write(f"MESSAGE_EXPIRY={os.getenv('MESSAGE_EXPIRY')}\n")
    env_file.write(f"HEARTBEAT_INTERVAL={os.getenv('HEARTBEAT_INTERVAL')}\n")
    env_file.write(f"BOOTSTRAP_NODES={os.getenv('BOOTSTRAP_NODES')}\n")
    env_file.write(f"STUN_SERVER={os.getenv('STUN_SERVER')}\n")
    env_file.write(f"TURN_SERVER={os.getenv('TURN_SERVER')}\n")
    env_file.write(f"TURN_USERNAME={os.getenv('TURN_USERNAME')}\n")
    env_file.write(f"TURN_PASSWORD={os.getenv('TURN_PASSWORD')}\n")
    env_file.write(f"ALCHEMY_KEY={alchemy_key_encrypted.decode()}\n")
    env_file.write(f"BLOCKCYPHER_API_KEY={blockcypher_key_encrypted.decode()}\n")
    env_file.write(f"ZEROX_API_KEY={zerox_key_encrypted.decode()}\n")

print("API keys have been encrypted and stored in the .env file.")
