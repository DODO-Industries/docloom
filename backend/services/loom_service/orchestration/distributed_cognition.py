from typing import List, Dict, Any, Callable

class DistributedCognition:
    """
    Distributes complex tasks and queries to multiple parallel reasoning agents or worker tracks.
    """
    def __init__(self):
        self.workers: List[Callable[[Dict[str, Any]], Any]] = []

    def register_worker(self, worker_func: Callable[[Dict[str, Any]], Any]) -> None:
        """Adds a worker node to the distributed pool."""
        self.workers.append(worker_func)

    def process_query_distributed(self, query: Dict[str, Any]) -> List[Any]:
        """Runs the query against all worker nodes and aggregates the outputs."""
        results = []
        for worker in self.workers:
            try:
                res = worker(query)
                results.append(res)
            except Exception as e:
                results.append({"error": str(e)})
        return results
