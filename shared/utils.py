"""
shared/oci_config.py
"""

import os
from google.adk.models.lite_llm import LiteLlm


def require_env(key: str) -> str:
    """Load a required environment variable or raise a clear error."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required env var: {key}")
    return val.strip()


def build_oci_model() -> LiteLlm:
    """Build a shared OCI LiteLlm instance from environment variables."""
    return LiteLlm(
        model="oci/xai.grok-4",
        oci_region=require_env("OCI_REGION"),
        oci_user=require_env("OCI_USER"),
        oci_fingerprint=require_env("OCI_FINGERPRINT"),
        oci_tenancy=require_env("OCI_TENANCY"),
        oci_compartment_id=require_env("OCI_COMPARTMENT_ID"),
        oci_key_file=require_env("OCI_KEY_FILE"),
        oci_serving_mode="ON_DEMAND",
    )
