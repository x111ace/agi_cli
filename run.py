# src/run.py

from pathlib import Path
import sys
import os

def run():
    """Run the application."""
    from src.main import main
    main()

if __name__ == "__main__":
    run()