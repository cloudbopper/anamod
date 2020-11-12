"""Pytest configuration file"""

import pytest


from anamod.core.constants import POLL_BASED_TRACKING, EVENT_LOG_TRACKING


collect_ignore = ['setup.py']


def pytest_addoption(parser):
    """Add command-line parameters"""
    parser.addoption("--shared-fs", action="store_true", help="Assume shared filesystem for condor")
    parser.addoption("--tracking", choices=[POLL_BASED_TRACKING, EVENT_LOG_TRACKING], default=EVENT_LOG_TRACKING)


@pytest.fixture
def shared_fs(request):
    """Return value of condor shared filesystem parameter"""
    return request.config.getoption("--shared-fs")


@pytest.fixture
def tracking(request):
    """Return value of condor tracking parameter"""
    return request.config.getoption("--tracking")
