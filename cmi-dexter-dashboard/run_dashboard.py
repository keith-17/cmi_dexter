"""Start the dashboard using checkout-local dependencies."""
import os
import sys
from pathlib import Path

dashboard_root = Path(__file__).resolve().parent
os.chdir(dashboard_root)
packages_path = dashboard_root / 'packages-ready'
if packages_path.is_dir():
    sys.path.insert(0, str(packages_path))

from streamlit.web import cli

sys.argv = [
    'streamlit', 'run', str(dashboard_root / 'app.py'),
    '--server.port', '8501', '--server.address', '127.0.0.1',
    '--server.headless', 'true',
    '--global.developmentMode', 'false',
]
cli.main()
