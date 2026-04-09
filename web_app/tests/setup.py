"""
Setup configuration for pytest plugins and test environment
"""
import sys
from pathlib import Path

# Add app root to path
sys.path.insert(0, str(Path(__file__).parent / 'app'))
sys.path.insert(0, str(Path(__file__).parent))
