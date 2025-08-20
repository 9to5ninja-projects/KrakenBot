"""
Position Manager module for KrakenBot.
Handles position sizing, capital allocation, and trade management.
"""
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

import config
from arbitrage import ArbitrageOpportunity

class Position:
    """Represents a trading position."""
    
    def __init__(self, opportunity: ArbitrageOpportunity, position_size: float, entry_time: datetime = None):
        """
        Initialize a position.
        
        Args:
            opportunity: The arbitrage opportunity
            position_size: The position size in base currency
            entry_time: Entry time (defaults to now)
        """
        self.opportunity = opportunity
        self.position_size = position_size
        self.entry_time = entry_time or datetime.now()
        self.exit_time = None
        self.status = "open"  # open, closed, or failed
        self.profit = 0
        self.profit_pct = 0
        self.id = f"pos_{int(time.time())}_{hash(str(opportunity))}"
    
    def close(self, exit_amount: float, exit_time: datetime = None):
        """
        Close the position.
        
        Args:
            exit_amount: The amount received after closing the position
            exit_time: Exit time (defaults to now)
        """
        self.exit_time = exit_time or datetime.now()
        self.status = "closed"
        self.profit = exit_amount - self.position_size
        self.profit_pct = (self.profit / self.position_size) * 100
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary for serialization."""
        return {
            "id": self.id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "status": self.status,
            "position_size": self.position_size,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "path": self.opportunity.path,
            "opportunity": {
                "timestamp": self.opportunity.timestamp.isoformat(),
                "start_amount": self.opportunity.start_amount,
                "end_amount": self.opportunity.end_amount,
                "profit": self.opportunity.profit,
                "profit_percentage": self.opportunity.profit_percentage,
                "is_profitable": self.opportunity.is_profitable
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create position from dictionary."""
        # Create a minimal opportunity object
        opportunity = ArbitrageOpportunity(
            timestamp=datetime.fromisoformat(data["opportunity"]["timestamp"]),
            start_amount=data["opportunity"]["start_amount"],
            end_amount=data["opportunity"]["end_amount"],
            profit=data["opportunity"]["profit"],
            profit_percentage=data["opportunity"]["profit_percentage"],
            prices={},  # We don't need to restore prices
            path=data["path"],
            is_profitable=data["opportunity"]["is_profitable"]
        )
        
        # Create position
        position = cls(
            opportunity=opportunity,
            position_size=data["position_size"],
            entry_time=datetime.fromisoformat(data["entry_time"])
        )
        
        # Set additional fields
        position.id = data["id"]
        position.status = data["status"]
        position.profit = data["profit"]
        position.profit_pct = data["profit_pct"]
        
        if data["exit_time"]:
            position.exit_time = datetime.fromisoformat(data["exit_time"])
        
        return position


class PositionManager:
    """Manages trading positions and capital allocation."""
    
    def __init__(self):
        """Initialize the position manager."""
        self.max_capital = config.MAX_CAPITAL
        self.position_size_pct = config.POSITION_SIZE_PCT
        self.max_positions = config.MAX_POSITIONS
        
        # Calculate position size
        self.position_size = self.max_capital * (self.position_size_pct / 100)
        
        # Track positions
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.allocated_capital = 0
        
        # Create data directory
        self.data_dir = config.DATA_DIR / 'positions'
        self.data_dir.mkdir(exist_ok=True)
        
        # Load existing positions
        self._load_positions()
        
        logger.info(f"Position Manager initialized with:")
        logger.info(f"  Max capital: ${self.max_capital:.2f}")
        logger.info(f"  Position size: ${self.position_size:.2f} ({self.position_size_pct:.1f}%)")
        logger.info(f"  Max positions: {self.max_positions}")
        logger.info(f"  Open positions: {len(self.open_positions)}")
        logger.info(f"  Allocated capital: ${self.allocated_capital:.2f}")
        logger.info(f"  Available capital: ${self.available_capital:.2f}")
    
    @property
    def available_capital(self) -> float:
        """Get available capital."""
        return self.max_capital - self.allocated_capital
    
    @property
    def available_positions(self) -> int:
        """Get number of available position slots."""
        return self.max_positions - len(self.open_positions)
    
    def can_open_position(self) -> bool:
        """Check if a new position can be opened."""
        return (
            self.available_capital >= self.position_size and
            self.available_positions > 0
        )
    
    def open_position(self, opportunity: ArbitrageOpportunity) -> Optional[Position]:
        """
        Open a new position.
        
        Args:
            opportunity: The arbitrage opportunity
            
        Returns:
            The new position or None if position cannot be opened
        """
        if not self.can_open_position():
            logger.warning("Cannot open position: insufficient capital or position slots")
            return None
        
        # Create position
        position = Position(
            opportunity=opportunity,
            position_size=self.position_size
        )
        
        # Update tracking
        self.open_positions.append(position)
        self.allocated_capital += self.position_size
        
        # Save positions
        self._save_positions()
        
        logger.info(f"Opened position {position.id} with size ${self.position_size:.2f}")
        logger.info(f"  Path: {' → '.join(opportunity.path)}")
        logger.info(f"  Expected profit: ${opportunity.profit:.2f} ({opportunity.profit_percentage:.2f}%)")
        logger.info(f"  Allocated capital: ${self.allocated_capital:.2f}")
        logger.info(f"  Available capital: ${self.available_capital:.2f}")
        
        return position
    
    def close_position(self, position_id: str, exit_amount: float) -> Optional[Position]:
        """
        Close a position.
        
        Args:
            position_id: The position ID
            exit_amount: The amount received after closing the position
            
        Returns:
            The closed position or None if position not found
        """
        # Find position
        position = None
        for pos in self.open_positions:
            if pos.id == position_id:
                position = pos
                break
        
        if not position:
            logger.warning(f"Cannot close position: position {position_id} not found")
            return None
        
        # Close position
        position.close(exit_amount)
        
        # Update tracking
        self.open_positions.remove(position)
        self.closed_positions.append(position)
        self.allocated_capital -= self.position_size
        
        # Save positions
        self._save_positions()
        
        logger.info(f"Closed position {position.id}")
        logger.info(f"  Profit: ${position.profit:.2f} ({position.profit_pct:.2f}%)")
        logger.info(f"  Allocated capital: ${self.allocated_capital:.2f}")
        logger.info(f"  Available capital: ${self.available_capital:.2f}")
        
        return position
    
    def get_position_summary(self) -> Dict:
        """Get a summary of all positions."""
        total_profit = sum(pos.profit for pos in self.closed_positions)
        avg_profit_pct = sum(pos.profit_pct for pos in self.closed_positions) / len(self.closed_positions) if self.closed_positions else 0
        
        return {
            "max_capital": self.max_capital,
            "position_size": self.position_size,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "allocated_capital": self.allocated_capital,
            "available_capital": self.available_capital,
            "total_profit": total_profit,
            "avg_profit_pct": avg_profit_pct
        }
    
    def _save_positions(self):
        """Save positions to file."""
        data = {
            "open_positions": [pos.to_dict() for pos in self.open_positions],
            "closed_positions": [pos.to_dict() for pos in self.closed_positions]
        }
        
        with open(self.data_dir / "positions.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_positions(self):
        """Load positions from file."""
        try:
            with open(self.data_dir / "positions.json", "r") as f:
                data = json.load(f)
            
            # Load open positions
            self.open_positions = [Position.from_dict(pos_data) for pos_data in data.get("open_positions", [])]
            
            # Load closed positions
            self.closed_positions = [Position.from_dict(pos_data) for pos_data in data.get("closed_positions", [])]
            
            # Recalculate allocated capital
            self.allocated_capital = sum(pos.position_size for pos in self.open_positions)
            
            logger.info(f"Loaded {len(self.open_positions)} open positions and {len(self.closed_positions)} closed positions")
        
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing positions found")
            self.open_positions = []
            self.closed_positions = []
            self.allocated_capital = 0


# Singleton instance
_position_manager = None

def get_position_manager() -> PositionManager:
    """Get the position manager singleton instance."""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager


if __name__ == "__main__":
    # Example usage
    manager = get_position_manager()
    print(f"Max capital: ${manager.max_capital:.2f}")
    print(f"Position size: ${manager.position_size:.2f} ({manager.position_size_pct:.1f}%)")
    print(f"Max positions: {manager.max_positions}")
    print(f"Open positions: {len(manager.open_positions)}")
    print(f"Allocated capital: ${manager.allocated_capital:.2f}")
    print(f"Available capital: ${manager.available_capital:.2f}")