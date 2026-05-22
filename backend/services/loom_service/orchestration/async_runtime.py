import asyncio
from typing import Coroutine, Any, List

class AsyncRuntime:
    """
    Manages scheduling and execution of asynchronous tasks within the cognitive loop.
    """
    def __init__(self):
        self.pending_tasks: List[asyncio.Task] = []

    def run_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Schedules a coroutine task to execute concurrently."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        task = loop.create_task(coro)
        self.pending_tasks.append(task)
        return task

    async def gather_all(self) -> List[Any]:
        """Waits for all scheduled tasks to complete and returns their outcomes."""
        if not self.pending_tasks:
            return []
        
        tasks_to_run = list(self.pending_tasks)
        self.pending_tasks.clear()
        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        return results
