"""
Centralized disease data loader.

Loads disease_data.json once at startup and provides it as a module-level
constant. Import from here instead of loading the JSON file separately in
views.py and serializers.py.
"""
import os
import json
from django.conf import settings

_DISEASE_DATA_PATH = os.path.join(settings.BASE_DIR, 'detector', 'disease_data.json')

try:
    with open(_DISEASE_DATA_PATH, 'r') as _f:
        DISEASE_INFO = json.load(_f)
except Exception as _e:
    print(f"Error loading disease_data.json: {_e}")
    DISEASE_INFO = {}
