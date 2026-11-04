import importlib.util
import os
from pathlib import Path
import socket
import threading
from unittest.mock import patch


def test_concurrent_mlopart_responses_match_requests():
    # Load this Python module without importing the compiled DeepGEMM extension.
    path = Path(__file__).resolve().parents[1] / 'deep_gemm/utils/mlopart.py'
    spec = importlib.util.spec_from_file_location('mlopart_ipc_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            fd = os.dup(kwargs['pass_fds'][0])
            self.worker = threading.Thread(target=self.serve, args=(fd,), daemon=True)
            self.worker.start()

        @staticmethod
        def serve(fd):
            with socket.socket(fileno=fd) as sock:
                sock.sendall(module.READY)
                while request := sock.recv(module.REQUEST.size, socket.MSG_WAITALL):
                    num_bytes, domain = module.REQUEST.unpack(request)
                    read_fd, write_fd = os.pipe()
                    try:
                        os.write(write_fd, f'{num_bytes}:{domain}'.encode())
                        os.close(write_fd)
                        write_fd = -1
                        socket.send_fds(sock, [b'f'], [read_fd])
                    finally:
                        os.close(read_fd)
                        if write_fd != -1:
                            os.close(write_fd)

        def wait(self):
            self.worker.join(5)
            assert not self.worker.is_alive()
            return 0

    first_receiving = threading.Event()
    second_receiving = threading.Event()
    second_done = threading.Event()
    recv_fds = socket.recv_fds

    def reorder_receives(*args, **kwargs):
        if threading.current_thread().name == 'first':
            first_receiving.set()
            # Without serialization the second call can receive the first
            # reply; with serialization it waits for the first to finish.
            if second_receiving.wait(1):
                assert second_done.wait(3)
            return recv_fds(*args, **kwargs)
        second_receiving.set()
        result = recv_fds(*args, **kwargs)
        second_done.set()
        return result

    results = {}
    errors = []

    def request(client, name, size, domain):
        try:
            fd = client.create_memory(size, domain)
            try:
                results[name] = os.read(fd, 32).decode()
            finally:
                os.close(fd)
        except Exception as exc:
            errors.append(exc)

    with patch.object(module.subprocess, 'Popen', FakeProcess), patch.object(socket, 'recv_fds', reorder_receives):
        client = module.MLOPart(bytes(16))
        first = threading.Thread(target=request, args=(client, 'first', 1024, 0), name='first', daemon=True)
        second = threading.Thread(target=request, args=(client, 'second', 2048, 1), name='second', daemon=True)
        try:
            first.start()
            assert first_receiving.wait(3)
            second.start()
            first.join(5)
            second.join(5)
            assert not first.is_alive() and not second.is_alive()
        finally:
            client.close()

    assert not errors
    assert results == {'first': '1024:0', 'second': '2048:1'}
