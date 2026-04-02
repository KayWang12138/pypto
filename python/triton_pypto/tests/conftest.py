import pytest


def pytest_addoption(parser):
    parser.addoption("--device", type=int, default=None, help="NPU device ID")


@pytest.fixture(autouse=True)
def device_id(pytestconfig, monkeypatch):
    dev_id = pytestconfig.getoption("--device")
    if dev_id is not None:
        monkeypatch.delenv("TILE_FWK_STEST_DEVICE_ID", raising=False)
        monkeypatch.setenv("TILE_FWK_DEVICE_ID", str(dev_id))
    yield dev_id