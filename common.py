from pydantic import BaseModel, Field
from typing import Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import base64
from decimal import Decimal
from typing import Dict, List, Tuple,Any
from SecureHybridZKStark import SecureHybridZKStark
import time 
import json 
from hashlib import sha256
import logging
from dataclasses import dataclass, field
import hashlib
import threading
from shared_logic import QuantumBlock, Transaction, NodeState, NodeDirectory

logger = logging.getLogger(__name__)


class NodeSetup:
    def __init__(self):
        self.node_directory = None

    def initialize_node_directory(self):
        # Import P2PNode here to avoid circular import
        from P2PNode import P2PNode

        p2p_node_instance = P2PNode(
            blockchain=None,
            host='localhost',
            port=50510
        )

        self.node_directory = NodeDirectory(p2p_node=p2p_node_instance)
        return self.node_directory

# Ensure NodeSetup is only used after everything is initialized
node_setup = NodeSetup()
node_directory = node_setup.initialize_node_directory()
