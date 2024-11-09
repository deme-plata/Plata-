from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
import time

class ConfirmationStatus(BaseModel):
    """Status of transaction confirmation"""
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
    
    score: float = Field(default=0.0, alias='confirmation_score')  # Alias for compatibility
    security_level: str = Field(default='LOW')
    confirmations: int = Field(default=0)
    is_final: bool = Field(default=False)

    @property
    def confirmation_score(self) -> float:
        """Compatibility property for old code"""
        return self.score

class ConfirmationMetrics(BaseModel): 
    """Metrics for transaction confirmation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    path_diversity: float = Field(default=0.0)
    quantum_strength: float = Field(default=0.0)
    consensus_weight: float = Field(default=0.0)
    depth_score: float = Field(default=0.0)
    last_updated: float = Field(default_factory=time.time)

class ConfirmationData(BaseModel):
    """Complete confirmation data structure"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    status: ConfirmationStatus = Field(default_factory=ConfirmationStatus)
    metrics: ConfirmationMetrics = Field(default_factory=ConfirmationMetrics)
    confirming_blocks: List[str] = Field(default_factory=list)
    confirmation_paths: List[List[str]] = Field(default_factory=list)
    quantum_confirmations: List[str] = Field(default_factory=list)