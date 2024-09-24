import requests
from contracts import simple_storage, simple_token

def deploy_contract(vm_host, sender_address, contract_code, constructor_args=None):
    url = f'http://{vm_host}/deploy_contract'
    payload = {
        'sender_address': sender_address,
        'contract_code': contract_code,
        'constructor_args': constructor_args or []
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to deploy contract: {response.text}")
        return None

if __name__ == "__main__":
    vm_host = '161.35.219.10:50503'
    sender_address = '0x123456'

    print("Deploying SimpleStorage contract...")
    storage_response = deploy_contract(vm_host, sender_address, simple_storage)
    print(storage_response)

    print("Deploying SimpleToken contract with 1 billion initial supply...")
    token_response = deploy_contract(vm_host, sender_address, simple_token, [1000000000])
    print(token_response)
