"""Pytest configuration file"""

import pytest


def pytest_addoption(parser):
    """Add command-line parameters"""
    parser.addoption("--shared-fs", action="store_true", help="Assume shared filesystem for condor")


@pytest.fixture
def shared_fs(request):
    """Return value of shared filesystem parameter"""
    return request.config.getoption("--shared-fs")
