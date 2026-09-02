"""Run the WJM demo web app:  python demo/run.py  (then open http://127.0.0.1:5000)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wjm_demo.app import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"WJM demo -> http://127.0.0.1:{port}")
    app.run(debug=bool(os.environ.get("WJM_DEMO_DEBUG")), port=port)
