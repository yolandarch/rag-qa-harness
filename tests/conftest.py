import sys
from pathlib import Path

# supaya "import app..." dan "import evals..." bekerja dari folder tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
