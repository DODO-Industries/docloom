import sys
from datetime import datetime
from backend.config.envConfig import settings

# ANSI Escape Sequences for vivid colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def log_system(message: str):
    """Important system setup logs."""
    if settings.IMPT_LOGS:
        print(f"{BOLD}{BLUE}[{get_timestamp()}] [SYS] {message}{RESET}")

def log_info(message: str):
    """General informational logs."""
    if settings.API_LOGS:
        print(f"{CYAN}[{get_timestamp()}] [INFO] {message}{RESET}")

def log_success(message: str):
    """Green success logs for completed operations."""
    if settings.API_LOGS:
        print(f"{BOLD}{GREEN}[{get_timestamp()}] [SUCCESS] {message}{RESET}")

def log_error(message: str, error=None):
    """Bright red error logs."""
    err_str = f" - {str(error)}" if error else ""
    print(f"{BOLD}{RED}[{get_timestamp()}] [ERROR] {message}{err_str}{RESET}", file=sys.stderr)

def log_llm_metrics(model: str, gen_time: float, in_tokens: int, out_tokens: int, total_tokens: int = None, context: str = ""):
    """Beautiful vivid logging specifically for LLM timing and token usage."""
    if settings.LLM_LOGS:
        ctx_str = f"{context} | " if context else ""
        if total_tokens is None:
            total_tokens = (in_tokens or 0) + (out_tokens or 0)
            
        print(f"\n{BOLD}{MAGENTA}[{get_timestamp()}] [LLM STATS] {ctx_str}Model: {model}{RESET}")
        print(f"{YELLOW}  ├─ Time:  {gen_time:.2f}s{RESET}")
        print(f"{YELLOW}  ├─ IN:    {in_tokens or 0} tokens{RESET}")
        print(f"{YELLOW}  ├─ OUT:   {out_tokens or 0} tokens{RESET}")
        print(f"{YELLOW}  └─ TOTAL: {total_tokens or 0} tokens{RESET}\n")

def log_clause(category: str, summary: str, risk: str):
    """Formatter to pretty print extracted clauses."""
    if settings.API_LOGS:
        risk_color = RED if risk.lower() == 'high' else YELLOW if risk.lower() == 'medium' else GREEN
        print(f"    {BOLD}└─ [{risk_color}{risk.upper()}{RESET}{BOLD}] {CYAN}{category.upper()}{RESET}: {summary}")

