"""Tests that load a model (they take the `lm` fixture) are marked `model`; `-m "not model"` or
`-k "not model"` leaves only the text and tensor tests. Without CUDA they are skipped unless
KNOB_TEST_LOCAL=1, so a laptop run never downloads or runs a model."""
import os

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "model: loads KNOB_TEST_MODEL (run on the GPU box)")


def pytest_collection_modifyitems(config, items):
    off = not torch.cuda.is_available() and os.environ.get("KNOB_TEST_LOCAL") != "1"
    for item in items:
        if "lm" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.model)
            if off:
                item.add_marker(pytest.mark.skip(reason="loads a model; no CUDA here (KNOB_TEST_LOCAL=1 to force)"))
