import os
import logging

logger = logging.getLogger(__name__)

class TuningManager:
    def __init__(self, filename=".modelTunning"):
        self.filename = filename
        self.config_path = self._find_config_file()
        self.params = {}
        self.last_loaded = 0
        self.reload_if_changed()

    def _find_config_file(self):
        # Start looking from the folder containing this file
        start_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = start_dir
        while True:
            candidate = os.path.join(current_dir, self.filename)
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(current_dir)
            if parent == current_dir:  # Root reached
                break
            current_dir = parent
        
        # Fallback to current working directory
        cwd_candidate = os.path.abspath(self.filename)
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
            
        # Fallback to backend/.modelTunning relative to cwd or standard layout
        fallback = os.path.join(os.getcwd(), "backend", self.filename)
        return fallback

    def reload_if_changed(self):
        if not self.config_path or not os.path.exists(self.config_path):
            return
        
        try:
            mtime = os.path.getmtime(self.config_path)
            if mtime > self.last_loaded:
                new_params = {}
                with open(self.config_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            parts = line.split("=", 1)
                            key = parts[0].strip()
                            val = parts[1].strip()
                            # Strip inline comments
                            if "#" in val:
                                val = val.split("#", 1)[0].strip()
                            new_params[key] = val
                self.params = new_params
                self.last_loaded = mtime
                logger.info(f"Loaded {len(self.params)} hyperparameters from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading hyperparameters from {self.config_path}: {e}")

    def get(self, key: str, default=None):
        self.reload_if_changed()
        return self.params.get(key, default)

    def get_float(self, key: str, default: float) -> float:
        val = self.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            logger.warning(f"Hyperparameter {key}={val} cannot be parsed as float, using default: {default}")
            return default

    def get_int(self, key: str, default: int) -> int:
        val = self.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            logger.warning(f"Hyperparameter {key}={val} cannot be parsed as int, using default: {default}")
            return default

# Global singleton
tuning_manager = TuningManager()
