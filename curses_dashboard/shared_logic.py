import importlib

def get_quantum_blockchain(config=None):
    QuantumBlockchain = getattr(importlib.import_module("quantumdagknight"), "QuantumBlockchain")
    if config is None:
        return QuantumBlockchain
    
    # Initialize QuantumBlockchain with config
    NodeDirectory = getattr(importlib.import_module("quantumdagknight"), "NodeDirectory")
    PBFTConsensus = getattr(importlib.import_module("vm"), "PBFTConsensus")
    SimpleVM = getattr(importlib.import_module("vm"), "SimpleVM")
    
    node_directory = NodeDirectory()
    consensus = PBFTConsensus(nodes=[], node_id=config['node_name'])
    secret_key = config.get('secret_key', 'your_default_secret_key')
    vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
    
    blockchain = QuantumBlockchain(
        consensus=consensus,
        secret_key=secret_key,
        node_directory=node_directory,
        vm=vm
    )
    
    blockchain.node_name = config['node_name']
    blockchain.port = config['port']
    
    return blockchain

def get_p2p_node(config=None):
    P2PNode = getattr(importlib.import_module("quantumdagknight"), "P2PNode")
    if config is None:
        return P2PNode
    
    # Initialize P2PNode with config
    blockchain = get_quantum_blockchain(config)
    
    p2p_node = P2PNode(
        host=config.get('host', 'localhost'),
        port=config['port'],
        blockchain=blockchain,
        security_level=config.get('security_level', 10)
    )
    
    return p2p_node

def get_enhanced_exchange(config=None):
    EnhancedExchangeWithZKStarks = getattr(importlib.import_module("enhanced_exchange"), "EnhancedExchangeWithZKStarks")
    if config is None:
        return EnhancedExchangeWithZKStarks
    
    # Initialize EnhancedExchangeWithZKStarks with config
    PriceOracle = getattr(importlib.import_module("quantumdagknight"), "PriceOracle")
    NodeDirectory = getattr(importlib.import_module("quantumdagknight"), "NodeDirectory")
    SimpleVM = getattr(importlib.import_module("vm"), "SimpleVM")
    
    blockchain = get_quantum_blockchain(config)
    vm = SimpleVM(gas_limit=10000, number_of_shards=10, nodes=[])
    price_oracle = PriceOracle()
    node_directory = NodeDirectory()
    
    exchange = EnhancedExchangeWithZKStarks(
        blockchain=blockchain,
        vm=vm,
        price_oracle=price_oracle,
        node_directory=node_directory,
        desired_security_level=config.get('security_level', 2),
        host=config.get('host', 'localhost'),
        port=config['port']
    )
    
    return exchange