import time
from typing import Dict, Any

class TemporalAwareness:
    """
    Keeps track of sequential step indices, chronological runtime,
    and adaptive time dilation based on data ingestion rates.
    """
    def __init__(self):
        self.tick_count = 0
        self.start_time = time.time()
        self.last_tick_time = self.start_time
        self.time_dilation_factor = 1.0
        
    def record_tick(self, input_size: int = 1):
        self.tick_count += 1
        now = time.time()
        elapsed = now - self.last_tick_time
        self.last_tick_time = now
        
        # If input size is very high, dilate time (meaning, slow down simulated clock transitions
        # to process denser information chunks before advancing chronological time state)
        if input_size > 10:
            self.time_dilation_factor = max(0.2, 10.0 / input_size)
        else:
            self.time_dilation_factor = 1.0
            
    def get_time_metrics(self) -> Dict[str, Any]:
        return {
            "tick_count": self.tick_count,
            "total_uptime": time.time() - self.start_time,
            "time_dilation": self.time_dilation_factor
        }
