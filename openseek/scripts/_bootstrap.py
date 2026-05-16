"""Put openseek/src on sys.path so scripts run without an install step."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_ROOT)  # /home/user/Opentract
for p in (_PARENT, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
