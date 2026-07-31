"""Pytest configuration and fixtures for the full test suite.

Includes a session-scoped teardown diagnostic that checks for leaked
resources after the full test suite completes. This is disabled by
default — set the environment variable DEVIN_TEARDOWN_DIAGNOSTIC=1
to enable verbose output.
"""
from __future__ import annotations

import os
import threading

import pytest


@pytest.fixture(scope="session", autouse=True)
def report_leaked_resources():
    """Session-scoped teardown diagnostic.

    Checks for leaked resources after the full test suite.
    Disabled by default — set DEVIN_TEARDOWN_DIAGNOSTIC=1 to enable.
    """
    yield

    if not os.environ.get("DEVIN_TEARDOWN_DIAGNOSTIC"):
        return

    import sys

    print("\n=== Teardown Resource Diagnostic ===")

    # Check threading.
    threads = threading.enumerate()
    non_daemon = [t for t in threads if not t.daemon and t != threading.main_thread()]
    print(f"Threads: {len(threads)} total, {len(non_daemon)} non-daemon")
    for t in non_daemon:
        print(f"  LEAKED: {t.name} (alive={t.is_alive()})")

    # Check asyncio tasks.
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                tasks = asyncio.all_tasks(loop)
                print(f"Asyncio tasks: {len(tasks)}")
                for t in tasks:
                    print(f"  LEAKED: {t}")
        except RuntimeError:
            print("Asyncio: no running event loop")
    except ImportError:
        print("Asyncio: not available")

    # Check multiprocessing.
    try:
        import multiprocessing
        children = multiprocessing.active_children()
        print(f"Multiprocessing children: {len(children)}")
        for c in children:
            print(f"  LEAKED: {c.name} (pid={c.pid})")
    except Exception as e:
        print(f"Multiprocessing: {e}")

    # Check subprocess.
    try:
        import subprocess
        # Just check if there are any known subprocess children.
        # This is a best-effort check.
    except Exception:
        pass

    print("=== Teardown Diagnostic Complete ===")
