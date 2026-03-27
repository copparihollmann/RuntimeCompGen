"""Tests for artifact bundling."""

from __future__ import annotations

import pytest
from compgen.runtime.bundle import BundleManifest


def test_bundle_manifest_defaults() -> None:
    """BundleManifest should have sensible defaults."""
    manifest = BundleManifest()
    assert manifest.version == "1.0"
    assert manifest.artifacts == {}


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_bundle_builder_creates_manifest() -> None:
    """BundleBuilder should create a manifest.json in the bundle directory."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_bundle_integrity() -> None:
    """Bundle should contain all referenced artifacts."""
