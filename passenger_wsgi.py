"""Entry point for cPanel / Passenger.

Passenger imports this file and looks for `application`. Everything else — the settings, the
database, the first administrator — is the same as on a VPS; see deploy/README.md, section B.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Keep uploads and results out of the web root unless the host has been told otherwise.
os.environ.setdefault("GEOAI_DATA", os.path.join(HERE, "data"))

from portal import create_app                                        # noqa: E402

application = create_app()
