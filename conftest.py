"""
Root pytest configuration.

Suppresses the Starlette httpx→httpx2 deprecation warning emitted by
`fastapi/testclient.py` when it imports `starlette.testclient`.

That warning is FastAPI's own `# noqa` — it is a forward-looking library
migration notice, not a problem in this project. It is silenced here so
test output stays clean.
"""


def pytest_configure(config):
    import warnings

    # Catch by category
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=Warning, module=r"starlette.*")
    warnings.filterwarnings("ignore", category=Warning, module=r"fastapi.*")

    # Catch by message (belt and braces)
    warnings.filterwarnings("ignore", message=r".*httpx.*starlette.*testclient.*")
