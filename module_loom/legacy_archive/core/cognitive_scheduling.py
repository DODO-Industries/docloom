from typing import List, Dict, Any, Tuple
import heapq

class CognitiveScheduler:
    """
    Manages a priority queue of cognitive actions (tasks) to be
    executed by the main loop, regulated by target utility and deadlines.
    """
    def __init__(self):
        self.task_queue: List[Tuple[float, str, Dict[str, Any]]] = []
        
    def schedule_task(self, task_name: str, priority: float, payload: Dict[str, Any]):
        # heapq is a min-heap, so we push negative priority to get max priority first
        heapq.heappush(self.task_queue, (-priority, task_name, payload))
        
    def pop_next_task(self) -> Tuple[str, Dict[str, Any]]:
        if not self.task_queue:
            return "idle", {}
        neg_priority, task_name, payload = heapq.heappop(self.task_queue)
        return task_name, payload
        
    def is_empty(self) -> bool:
        return len(self.task_queue) == 0
        
    def reschedule_all(self, attention_focus: List[str]):
        """Boosts priority of tasks matching active focus concepts."""
        temp_queue = []
        while self.task_queue:
            neg_prio, name, payload = heapq.heappop(self.task_queue)
            prio = -neg_prio
            
            # Boost if payload relates to focus
            for focus in attention_focus:
                if focus in str(payload):
                    prio *= 1.5
                    break
                    
            temp_queue.append((prio, name, payload))
            
        for prio, name, payload in temp_queue:
            self.schedule_task(name, prio, payload)

    def register_task(self, task_name: str, weight: float):
        """Wrapper method to match CognitionPipeline's register_task."""
        self.schedule_task(task_name, weight, {})

    def tick_scheduler(self):
        """Wrapper method to match CognitionPipeline's tick_scheduler."""
        pass
