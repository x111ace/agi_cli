# src/run.py

import os
import sys
from pathlib import Path

def run():
    """Run the application."""
    from src.main import main
    main()

if __name__ == "__main__":
    run()