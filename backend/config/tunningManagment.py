import os

class TuningManager:
    """
    Manages and parses model tuning parameters loaded from backend/.modelTunning.
    Allows dynamic adjustment of cognitive field parameters without code changes.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TuningManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, filepath=None):
        if hasattr(self, 'configs'):
            return
            
        if filepath is None:
            # Locate backend/.modelTunning by climbing directory tree
            curr = os.path.abspath(os.path.dirname(__file__))
            while curr != os.path.dirname(curr):
                candidate = os.path.join(curr, "backend", ".modelTunning")
                if os.path.exists(candidate):
                    filepath = candidate
                    break
                candidate_root = os.path.join(curr, ".modelTunning")
                if os.path.exists(candidate_root):
                    filepath = candidate_root
                    break
                curr = os.path.dirname(curr)

        self.configs = {}
        if filepath and os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        self.configs[k.strip()] = v.strip()

    def get_float(self, key, default):
        val = self.configs.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def get_int(self, key, default):
        val = self.configs.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

# Global singleton
tuning_manager = TuningManager()
