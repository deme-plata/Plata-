import importlib

def get_quantum_blockchain():
    QuantumBlockchain = getattr(importlib.import_module("quantumdagknight"), "QuantumBlockchain")
    return QuantumBlockchain

def get_p2p_node():
    P2PNode = getattr(importlib.import_module("quantumdagknight"), "P2PNode")
    return P2PNode

def get_enhanced_exchange():
    EnhancedExchangeWithZKStarks = getattr(importlib.import_module("enhanced_exchange"), "EnhancedExchangeWithZKStarks")
    return EnhancedExchangeWithZKStarks
