import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests never touch the network or a real key.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("TELEGRAM_TOKEN", None)
os.environ.pop("TELEGRAM_CHAT_ID", None)
