from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Optional, Tuple
import numpy as np
import logging
import time
from decimal import Decimal

# Constants from quantum cosmos implementation
G = 6.67430e-11
c = 3.0e8
hbar = 1.0545718e-34
k_B = 1.380649e-23
l_p = np.sqrt(G * hbar / c**3)  # Planck length
t_p = l_p / c  # Planck time 
m_p = np.sqrt(hbar * c / G)  # Planck mass

class ScaleHierarchy:
    """Manages scale hierarchies and enforces proper UV/IR cutoffs"""
    def __init__(self):
        self.l_p = np.sqrt(G * hbar / c**3)
        self.t_p = self.l_p / c
        self.m_p = np.sqrt(hbar * c / G)
        
        # Scale hierarchy parameters
        self.UV_cutoff = self.l_p
        self.IR_cutoff = 1e3 * self.l_p
        self.energy_cutoff = self.m_p * c**2
        
    def enforce_length_scale(self, length):
        """Enforce proper length scale hierarchy"""
        return np.clip(length, self.UV_cutoff, self.IR_cutoff)
    
    def enforce_energy_scale(self, energy):
        """Enforce proper energy scale hierarchy"""
        return np.clip(energy, 0, self.energy_cutoff)

class EnhancedGasTransactionType(Enum):
    """Enhanced transaction types with quantum effects"""
    STANDARD = "standard"
    SMART_CONTRACT = "smart_contract" 
    DAG_REORG = "dag_reorg"
    QUANTUM_PROOF = "quantum_proof"
    QUANTUM_ENTANGLE = "quantum_entangle"
    DATA_STORAGE = "data_storage"
    COMPUTATION = "computation"
    QUANTUM_STATE = "quantum_state"
    ENTANGLEMENT_VERIFY = "entanglement_verify"

@dataclass
class QuantumGasParameters:
    """Enhanced gas parameters with quantum corrections"""
    base_cost: int
    security_multiplier: float
    quantum_factor: float = 1.0
    dag_depth: int = 0
    entanglement_count: int = 0
    data_size: int = 0
    decoherence_rate: float = 0.1
    entanglement_strength: float = 0.5

@dataclass
class EnhancedGasPrice:
    """Enhanced gas price with quantum components"""
    base_price: Decimal
    security_premium: Decimal 
    quantum_premium: Decimal
    entanglement_premium: Decimal
    decoherence_discount: Decimal
    congestion_premium: Decimal
    total: Decimal
class EnhancedGasMetrics:
    def __init__(self):
        self.total_gas_used = 0
        self.total_gas_cost = Decimal('0')
        self.gas_prices = []
        self.quantum_premiums = []
        self.entanglement_premiums = []
        self.transaction_types = {}

    def track_transaction_gas(self, tx_type: str, gas_used: int, gas_price: EnhancedGasPrice):
        # Track gas usage
        self.total_gas_used += gas_used
        total_cost = Decimal(str(gas_used)) * gas_price.total
        self.total_gas_cost += total_cost
        
        # Track gas price
        self.gas_prices.append(gas_price.total)
        
        # Track premiums if they exist
        if gas_price.quantum_premium > 0:
            self.quantum_premiums.append(gas_price.quantum_premium)
        if gas_price.entanglement_premium > 0:
            self.entanglement_premiums.append(gas_price.entanglement_premium)
            
        # Track by transaction type
        if tx_type not in self.transaction_types:
            self.transaction_types[tx_type] = {
                'count': 0,
                'total_gas': 0,
                'total_cost': Decimal('0'),
                'gas_prices': []
            }
        
        tx_stats = self.transaction_types[tx_type]
        tx_stats['count'] += 1
        tx_stats['total_gas'] += gas_used
        tx_stats['total_cost'] += total_cost
        tx_stats['gas_prices'].append(float(gas_price.total))

    def get_metrics(self) -> dict:
        metrics = {
            'total_gas_used': self.total_gas_used,
            'total_gas_cost': float(self.total_gas_cost),
            'average_gas_price': float(statistics.mean(self.gas_prices)) if self.gas_prices else 0,
            'gas_price_range': {
                'min': float(min(self.gas_prices)) if self.gas_prices else 0,
                'max': float(max(self.gas_prices)) if self.gas_prices else 0
            },
            'gas_price_volatility': float(statistics.stdev(self.gas_prices)) if len(self.gas_prices) > 1 else 0,
            'quantum_metrics': {
                'avg_premium': float(statistics.mean(self.quantum_premiums)) if self.quantum_premiums else 0,
                'premium_count': len(self.quantum_premiums)
            },
            'entanglement_metrics': {
                'avg_premium': float(statistics.mean(self.entanglement_premiums)) if self.entanglement_premiums else 0,
                'premium_count': len(self.entanglement_premiums)
            },
            'transaction_types': {}
        }
        
        # Add per-type metrics
        for tx_type, stats in self.transaction_types.items():
            metrics['transaction_types'][tx_type] = {
                'count': stats['count'],
                'avg_gas': stats['total_gas'] / stats['count'] if stats['count'] > 0 else 0,
                'avg_cost': float(stats['total_cost'] / stats['count']) if stats['count'] > 0 else 0,
                'avg_gas_price': statistics.mean(stats['gas_prices']) if stats['gas_prices'] else 0
            }
            
        return metrics
class EnhancedDAGKnightGasSystem:
    """Enhanced gas system with quantum mechanics integration"""
    
    def __init__(self):
        self.scales = ScaleHierarchy()
        self.base_costs = {
            EnhancedGasTransactionType.STANDARD: 21000,
            EnhancedGasTransactionType.SMART_CONTRACT: 50000,
            EnhancedGasTransactionType.DAG_REORG: 100000,
            EnhancedGasTransactionType.QUANTUM_PROOF: 75000,
            EnhancedGasTransactionType.QUANTUM_ENTANGLE: 80000,
            EnhancedGasTransactionType.DATA_STORAGE: 40000,
            EnhancedGasTransactionType.COMPUTATION: 60000,
            EnhancedGasTransactionType.QUANTUM_STATE: 90000,
            EnhancedGasTransactionType.ENTANGLEMENT_VERIFY: 85000
        }

        # Network state
        self.network_metrics = {
            'avg_block_time': 30.0,
            'network_load': 0.0,
            'active_nodes': 0,
            'quantum_entangled_pairs': 0,
            'dag_depth': 0,
            'total_compute': 0.0
        }
        
        # Quantum parameters
        self.quantum_entropy = 0.1
        self.quantum_coupling = 0.15
        self.entanglement_threshold = 0.5
        
        # Gas price limits
        self.min_gas_price = Decimal('0.1')
        self.max_gas_price = Decimal('1000.0')
        self.base_gas_price = Decimal('1.0')
        
        # Timing
        self.last_adjustment = time.time()
        self.adjustment_interval = 300  # 5 minutes
        self.metrics = EnhancedGasMetrics()
        self.confirmation_thresholds = {
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.0
        }
        
        # Track confirmation metrics
        self.confirmation_metrics = {
            'scores': [],
            'levels': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'total_confirmations': 0,
            'quantum_scores': []
        }

    async def calculate_confirmation_score(self, tx_type: str, quantum_enabled: bool, 
                                        entanglement_count: int) -> dict:
        """Calculate comprehensive confirmation score"""
        try:
            # Base confirmation score based on transaction type
            base_scores = {
                'STANDARD': 0.5,
                'SMART_CONTRACT': 0.6,
                'DAG_REORG': 0.7,
                'QUANTUM_PROOF': 0.8,
                'QUANTUM_ENTANGLE': 0.85,
                'DATA_STORAGE': 0.6,
                'COMPUTATION': 0.65,
                'QUANTUM_STATE': 0.9,
                'ENTANGLEMENT_VERIFY': 0.85
            }
            
            base_score = base_scores.get(tx_type, 0.5)
            
            # Quantum enhancement
            quantum_score = 0.0
            if quantum_enabled:
                # Calculate quantum score based on entanglement and network state
                quantum_factor = (
                    self.quantum_coupling * 
                    np.exp(-self.network_metrics['network_load']) *
                    (1 + np.tanh(self.network_metrics['quantum_entangled_pairs'] / 100))
                )
                quantum_score = min(0.3, quantum_factor)
                
                # Add entanglement bonus
                if entanglement_count > 0:
                    entanglement_bonus = min(0.2, 0.05 * entanglement_count)
                    quantum_score += entanglement_bonus
            
            # Network security contribution
            network_score = min(0.2, 0.1 * np.tanh(self.network_metrics['dag_depth'] / 100))
            
            # Calculate final score
            total_score = min(1.0, base_score + quantum_score + network_score)
            
            # Determine security level
            if total_score >= self.confirmation_thresholds['HIGH']:
                security_level = 'HIGH'
            elif total_score >= self.confirmation_thresholds['MEDIUM']:
                security_level = 'MEDIUM'
            else:
                security_level = 'LOW'
                
            # Update metrics
            self.confirmation_metrics['scores'].append(total_score)
            self.confirmation_metrics['levels'][security_level] += 1
            self.confirmation_metrics['total_confirmations'] += 1
            if quantum_enabled:
                self.confirmation_metrics['quantum_scores'].append(quantum_score)
            
            return {
                'score': total_score,
                'security_level': security_level,
                'components': {
                    'base_score': base_score,
                    'quantum_score': quantum_score,
                    'network_score': network_score
                }
            }
            
        except Exception as e:
            logging.error(f"Error calculating confirmation score: {str(e)}")
            raise


    async def calculate_gas(self, tx_type: EnhancedGasTransactionType, 
                           data_size: int, quantum_enabled: bool = False,
                           entanglement_count: int = 0) -> Tuple[int, EnhancedGasPrice]:
        """Calculate gas with quantum effects"""
        try:
            # Base gas calculation
            base_gas = self.base_costs[tx_type]
            data_gas = self._calculate_data_gas(data_size)
            
            # Get current gas price
            gas_price = await self.get_current_gas_price(
                quantum_enabled=quantum_enabled,
                entanglement_count=entanglement_count
            )
            
            # Calculate total gas with quantum effects if enabled
            quantum_gas = await self._calculate_quantum_premium(
                quantum_enabled,
                entanglement_count
            )
            
            total_gas = int(base_gas + data_gas + quantum_gas)
            
            # Track metrics
            self.metrics.track_transaction_gas(
                tx_type.value, 
                total_gas, 
                gas_price
            )
            
            return total_gas, gas_price
            
        except Exception as e:
            logging.error(f"Error in calculate_gas: {str(e)}")
            raise



    def get_confirmation_metrics(self) -> dict:
        """Get current confirmation metrics"""
        try:
            return {
                'average_score': statistics.mean(self.confirmation_metrics['scores']) 
                                if self.confirmation_metrics['scores'] else 0.0,
                'max_score': max(self.confirmation_metrics['scores']) 
                            if self.confirmation_metrics['scores'] else 0.0,
                'security_levels': self.confirmation_metrics['levels'],
                'total_confirmations': self.confirmation_metrics['total_confirmations'],
                'quantum_stats': {
                    'average_score': statistics.mean(self.confirmation_metrics['quantum_scores'])
                                    if self.confirmation_metrics['quantum_scores'] else 0.0,
                    'count': len(self.confirmation_metrics['quantum_scores'])
                }
            }
        except Exception as e:
            logging.error(f"Error getting confirmation metrics: {str(e)}")
            return {}

    def _calculate_data_gas(self, data_size: int) -> int:
        """Calculate gas for data with quantum storage effects"""
        # Scale data size to Planck units
        scaled_size = self.scales.enforce_length_scale(float(data_size))
        return max(0, int(scaled_size * 16))

    async def _calculate_quantum_premium(self, quantum_enabled: bool,
                                      entanglement_count: int) -> int:
        """Calculate quantum premium with entanglement effects"""
        if not quantum_enabled:
            return 0

        # Calculate quantum factor with network effects
        quantum_factor = (
            1.0 +
            self.quantum_entropy * np.exp(-self.network_metrics['network_load']) +
            self.quantum_coupling * np.sin(self.network_metrics['dag_depth'] / 100 * np.pi)
        )

        # Add entanglement contribution
        if entanglement_count > 0:
            entanglement_factor = min(entanglement_count / 10, 2.0)
            quantum_factor *= entanglement_factor

        return int(50000 * quantum_factor)

    def _calculate_entanglement_gas(self, entanglement_count: int) -> int:
        """Calculate gas cost for entanglement verification"""
        # Base cost per entangled pair
        base_cost = 10000
        
        # Scale with number of entangled pairs
        if entanglement_count == 0:
            return 0
        
        # Non-linear scaling for multiple entanglements
        return int(base_cost * (1 + np.log(entanglement_count)))

    async def get_current_gas_price(self, quantum_enabled: bool = False,
                                  entanglement_count: int = 0) -> EnhancedGasPrice:
        """Calculate current gas price with quantum effects"""
        try:
            # Calculate base price with network load
            base_price = self._calculate_base_price()

            # Calculate quantum premium
            quantum_premium = Decimal('0')
            if quantum_enabled:
                quantum_factor = self._calculate_quantum_factor()
                quantum_premium = base_price * Decimal(str(quantum_factor))

            # Calculate entanglement premium
            entanglement_premium = self._calculate_entanglement_premium(
                entanglement_count, base_price
            )

            # Calculate decoherence discount
            decoherence_discount = self._calculate_decoherence_discount(
                base_price, quantum_enabled
            )

            # Calculate congestion premium
            congestion_premium = self._calculate_congestion_premium(base_price)

            # Security premium based on DAG depth
            security_premium = self._calculate_security_premium(base_price)

            # Calculate total price
            total_price = (
                base_price +
                security_premium +
                quantum_premium +
                entanglement_premium -
                decoherence_discount +
                congestion_premium
            )

            # Ensure price is within bounds
            total_price = min(max(total_price, self.min_gas_price),
                            self.max_gas_price)

            return EnhancedGasPrice(
                base_price=base_price,
                security_premium=security_premium,
                quantum_premium=quantum_premium,
                entanglement_premium=entanglement_premium,
                decoherence_discount=decoherence_discount,
                congestion_premium=congestion_premium,
                total=total_price
            )

        except Exception as e:
            logging.error(f"Error calculating gas price: {str(e)}")
            raise


    def _calculate_base_price(self) -> Decimal:
        """Calculate base gas price with quantum corrections"""
        # Get network metrics
        load = self.network_metrics['network_load']
        block_time_ratio = self.network_metrics['avg_block_time'] / 30.0

        # Calculate quantum entropy correction
        entropy_correction = 1.0 + self.quantum_entropy * (
            -load * np.log(load + 1e-10)
        )

        # Calculate quantum coupling adjustment
        coupling_adjustment = 1.0 + self.quantum_coupling * (
            load ** 2 + (self.network_metrics['dag_depth'] / 1000) ** 0.5
        )

        # Combine factors
        price_factor = (
            entropy_correction *
            coupling_adjustment *
            load ** 1.5 *
            block_time_ratio ** 0.5
        )

        return Decimal(str(price_factor)) * self.base_gas_price

    def _calculate_quantum_factor(self) -> float:
        """Calculate quantum adjustment factor"""
        entanglement_ratio = (
            self.network_metrics['quantum_entangled_pairs'] /
            max(1, self.network_metrics['active_nodes'])
        )
        return min(2.0, 1.0 + entanglement_ratio)

    def _calculate_entanglement_premium(self,
                                      entanglement_count: int,
                                      base_price: Decimal) -> Decimal:
        """Calculate premium for entanglement verification"""
        if entanglement_count == 0:
            return Decimal('0')
        
        # Non-linear scaling with number of entanglements
        premium_factor = np.log1p(entanglement_count) / 10
        return base_price * Decimal(str(premium_factor))

    def _calculate_decoherence_discount(self,
                                      base_price: Decimal,
                                      quantum_enabled: bool) -> Decimal:
        """Calculate discount based on quantum decoherence"""
        if not quantum_enabled:
            return Decimal('0')
        
        # Time-based decoherence factor
        current_time = time.time()
        time_factor = np.exp(-(current_time - self.last_adjustment) / 3600)
        
        return base_price * Decimal(str(0.1 * time_factor))

    def _calculate_security_premium(self, base_price: Decimal) -> Decimal:
        """Calculate security premium based on DAG depth"""
        depth_factor = np.tanh(self.network_metrics['dag_depth'] / 1000)
        return base_price * Decimal(str(0.2 * depth_factor))

    def _calculate_congestion_premium(self, base_price: Decimal) -> Decimal:
        """Calculate congestion premium with quantum effects"""
        if self.network_metrics['network_load'] > 0.8:
            # High congestion premium with quantum correction
            load = self.network_metrics['network_load']
            quantum_correction = 1 + self.quantum_coupling * load
            premium_factor = ((load - 0.8) * 5) * quantum_correction
            return base_price * Decimal(str(premium_factor))
        return Decimal('0')

    async def update_network_metrics(self, metrics: dict):
        """Update network metrics and adjust gas parameters"""
        self.network_metrics = metrics
        
        # Check if adjustment is needed
        current_time = time.time()
        if current_time - self.last_adjustment > self.adjustment_interval:
            await self.adjust_gas_parameters()
            self.last_adjustment = current_time

    async def adjust_gas_parameters(self):
        """Adjust gas parameters based on network state"""
        try:
            # Calculate average network load
            avg_load = self.network_metrics['network_load']
            
            # Adjust quantum parameters
            if avg_load > 0.7:  # Target load threshold
                self.quantum_entropy *= 1.1
                self.quantum_coupling *= 1.1
            else:
                self.quantum_entropy *= 0.9
                self.quantum_coupling *= 0.9
            
            # Keep parameters within bounds
            self.quantum_entropy = np.clip(self.quantum_entropy, 0.05, 0.2)
            self.quantum_coupling = np.clip(self.quantum_coupling, 0.1, 0.3)
            
            # Adjust base gas price based on block time
            block_time_ratio = self.network_metrics['avg_block_time'] / 30.0
            self.base_gas_price *= Decimal(str(block_time_ratio))
            
            # Keep base price within bounds
            self.base_gas_price = min(
                max(self.base_gas_price, self.min_gas_price),
                self.max_gas_price
            )
            
        except Exception as e:
            logging.error(f"Error adjusting gas parameters: {str(e)}")
            raise