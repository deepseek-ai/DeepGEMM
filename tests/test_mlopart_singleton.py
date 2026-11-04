import importlib.util
from pathlib import Path
import threading
from unittest.mock import patch

import pytest


def load_mlopart():
    path = Path(__file__).resolve().parents[1] / 'deep_gemm/utils/mlopart.py'
    spec = importlib.util.spec_from_file_location('mlopart_singleton_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concurrent_first_lookup_creates_one_client():
    module = load_mlopart()
    first_entered = threading.Event()
    second_entered = threading.Event()
    instances = []
    results = []

    class FakeMLOPart:
        def __init__(self, device_uuid):
            self.device_uuid = device_uuid
            self.available = True
            self.closed = False
            instances.append(self)
            if threading.current_thread().name == 'first':
                first_entered.set()
                second_entered.wait(1)
            else:
                second_entered.set()

        def close(self):
            self.closed = True

    def lookup():
        results.append(module.is_available(bytes(16)))

    with patch.object(module, 'MLOPart', FakeMLOPart):
        first = threading.Thread(target=lookup, name='first', daemon=True)
        second = threading.Thread(target=lookup, name='second', daemon=True)
        first.start()
        assert first_entered.wait(3)
        second.start()
        first.join(5)
        second.join(5)
        assert not first.is_alive() and not second.is_alive()
        module.release()

    assert results == [True, True]
    assert len(instances) == 1
    assert instances[0].closed


def test_release_waits_for_in_progress_construction():
    module = load_mlopart()
    constructing = threading.Event()
    finish_construction = threading.Event()
    release_started = threading.Event()
    released = threading.Event()
    instances = []

    class FakeMLOPart:
        def __init__(self, device_uuid):
            self.device_uuid = device_uuid
            self.available = True
            self.closed = False
            instances.append(self)
            constructing.set()
            assert finish_construction.wait(3)

        def close(self):
            self.closed = True

    def release():
        release_started.set()
        module.release()
        released.set()

    with patch.object(module, 'MLOPart', FakeMLOPart):
        starter = threading.Thread(target=module.is_available, args=(bytes(16),), daemon=True)
        stopper = threading.Thread(target=release, daemon=True)
        starter.start()
        assert constructing.wait(3)
        stopper.start()
        try:
            assert release_started.wait(3)
            assert not released.wait(0.2)
        finally:
            finish_construction.set()
            starter.join(5)
            stopper.join(5)
        assert not starter.is_alive() and not stopper.is_alive()
        assert released.is_set() and instances[0].closed


def test_failed_construction_can_be_retried():
    module = load_mlopart()
    attempts = []

    class FakeMLOPart:
        def __init__(self, device_uuid):
            attempts.append(device_uuid)
            if len(attempts) == 1:
                raise RuntimeError('injected failure')
            self.device_uuid = device_uuid
            self.available = True

        def close(self):
            pass

    with patch.object(module, 'MLOPart', FakeMLOPart):
        with pytest.raises(RuntimeError, match='injected failure'):
            module.is_available(bytes(16))
        assert module.is_available(bytes(16))
        module.release()
    assert len(attempts) == 2
