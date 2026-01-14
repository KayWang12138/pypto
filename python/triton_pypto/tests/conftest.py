import pytest


def pytest_addoption(parser):
    parser.addoption("--dynamic-only", action="store_true", default=False, help="Run only tests with dynamic=True")
    parser.addoption("--static-only", action="store_true", default=False, help="Run only tests with dynamic=False")
    parser.addoption("--unroll-factor", type=int, default=1, help="Dynamic grid loop unroll factor")
    parser.addoption("--device", type=int, default=None, help="NPU device ID")


def pytest_configure(config):
    """
    Perform validation on the configuration before the run starts.
    This is the correct place to check for conflicting flags.
    """
    if config.getoption("--dynamic-only") and config.getoption("--static-only"):
        pytest.exit("Configuration Error: Cannot use both --dynamic-only and --static-only", returncode=1)


def pytest_generate_tests(metafunc):
    if "dynamic" in metafunc.fixturenames:
        if metafunc.config.getoption("--dynamic-only"):
            params = [True]
        elif metafunc.config.getoption("--static-only"):
            params = [False]
        else:
            params = [True, False]
        ids = ["dynamic" if p else "static" for p in params]
        metafunc.parametrize("dynamic", params, ids=ids)


@pytest.fixture(autouse=True)
def device_id(pytestconfig, monkeypatch):
    dev_id = pytestconfig.getoption("--device")
    if dev_id is not None:
        monkeypatch.delenv("TILE_FWK_STEST_DEVICE_ID", raising=False)
        monkeypatch.setenv("TILE_FWK_DEVICE_ID", str(dev_id))
    yield dev_id


@pytest.fixture
def unroll_factor(pytestconfig):
    yield pytestconfig.getoption("--unroll-factor")
