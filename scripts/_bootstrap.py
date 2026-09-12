"""Makes `import faa_cxer` work when a script in this directory is run
directly (`python scripts/foo.py`) without the package having been
`pip install`-ed first. Every script starts with:

    from _bootstrap import PROJECT_ROOT  # noqa: F401
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
