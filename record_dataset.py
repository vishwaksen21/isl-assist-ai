from __future__ import annotations

import runpy
import sys


if __name__ == "__main__":
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        msg = str(e)
        if "libGL.so.1" in msg:
            print(
                "ERROR: OpenCV failed to import because libGL is missing (libGL.so.1).\n"
                "Fix (recommended in dev containers):\n"
                "  1) pip uninstall -y opencv-python\n"
                "  2) pip install -r requirements.txt\n\n"
                "Alternative fix (system packages):\n"
                "  sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0\n"
            )
            raise SystemExit(1)
        raise

    runpy.run_path("scripts/record_dataset.py", run_name="__main__")
