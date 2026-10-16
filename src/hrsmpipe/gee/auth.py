"""Earth Engine initialisation: service account, stored credentials, or interactive login."""

from __future__ import annotations

import json
import logging
import os

import ee

log = logging.getLogger(__name__)

_initialised: str | None = None


def _service_account_credentials():
    key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
        "EE_PRIVATE_KEY_FILE"
    )
    email = os.environ.get("EE_SERVICE_ACCOUNT")
    if not (key_file or email):
        return None
    if not key_file:
        raise RuntimeError(
            "EE_SERVICE_ACCOUNT is set but no key file: set GOOGLE_APPLICATION_CREDENTIALS"
        )
    if not email:
        with open(key_file) as f:
            email = json.load(f)["client_email"]
    return ee.ServiceAccountCredentials(email, key_file)


def initialize(project: str) -> None:
    """Initialise ``ee`` once per process.

    Order: service account if ``GOOGLE_APPLICATION_CREDENTIALS``/``EE_SERVICE_ACCOUNT`` is set,
    else stored interactive credentials, else ``ee.Authenticate()`` (then initialise again).
    """
    global _initialised
    if _initialised == project:
        return
    creds = _service_account_credentials()
    if creds is not None:
        log.info("initialising Earth Engine with a service account (project %s)", project)
        ee.Initialize(creds, project=project)
    else:
        try:
            ee.Initialize(project=project)
        except Exception as exc:  # no/expired stored credentials
            log.warning("stored Earth Engine credentials unusable (%s); authenticating", exc)
            ee.Authenticate()
            ee.Initialize(project=project)
    _initialised = project
