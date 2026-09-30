"""An IPC wrapper for allocating localized memory via MLOPart"""
import ctypes
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional

NUM_LOCALITY_DOMAINS = 2
REQUEST = struct.Struct('qq')
READY = b'r'
PR_SET_PDEATHSIG = 1  # `<linux/prctl.h>`


class MLOPart:
    """
    An MLOPart client that allocates localized memory.
    It connects to the MLOPart pipe, and send the allocated handles via socket.
    """

    def __init__(self, device_uuid: bytes) -> None:
        self.device_uuid = device_uuid
        self.sock, client_sock = socket.socketpair()
        env = {k: v for k, v in os.environ.items() if k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT')}
        with client_sock:
            self.client = subprocess.Popen([sys.executable, '-I', __file__, str(client_sock.fileno()), f'GPU-{uuid.UUID(bytes=device_uuid)}'],
                                           env=env, pass_fds=(client_sock.fileno(), ), start_new_session=True)
        self.available = self.sock.recv(1) == READY
        if not self.available:
            warnings.warn('MLOPart is unavailable; localization has been disabled')

    def create_memory(self, num_bytes: int, domain_idx: int) -> int:
        assert self.available, 'MLOPart is unavailable, so no localized memory can be allocated'
        self.sock.sendall(REQUEST.pack(num_bytes, domain_idx))
        msg, fds, _, _ = socket.recv_fds(self.sock, 1, 1)
        assert msg == b'f', 'The MLOPart client exited'
        return fds[0]

    def close(self) -> None:
        """Quit the processes; the exported memory outlives them"""
        self.sock.close()
        self.client.wait()


_mlopart: Optional[MLOPart] = None


def _get(device_uuid: bytes) -> MLOPart:
    global _mlopart
    if _mlopart is None:
        _mlopart = MLOPart(device_uuid)
    assert _mlopart.device_uuid == device_uuid
    return _mlopart


def is_available(device_uuid: bytes) -> bool:
    return _get(device_uuid).available


def create_memory(device_uuid: bytes, num_bytes: int, domain_idx: int) -> int:
    return _get(device_uuid).create_memory(num_bytes, domain_idx)


def release() -> None:
    global _mlopart
    if _mlopart is not None:
        _mlopart.close()
        _mlopart = None


CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_LOCATION_TYPE_DEVICE, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1, 1, 1


class CUmemLocation(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('requestedHandleTypes', ctypes.c_int), ('location', CUmemLocation),
                ('win32HandleMetaData', ctypes.c_void_p), ('allocFlags', ctypes.c_uint64)]


def check(result: int) -> None:
    assert result == 0, f'CUDA driver error {result}'


def serve(sock: socket.socket) -> None:
    cuda = ctypes.CDLL('libcuda.so.1')
    check(cuda.cuInit(0))
    num_devices = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(num_devices)))
    assert num_devices.value == NUM_LOCALITY_DOMAINS, f'{num_devices.value} MLOPart devices'
    sock.sendall(READY)

    while request := sock.recv(REQUEST.size, socket.MSG_WAITALL):
        num_bytes, domain_idx = REQUEST.unpack(request)
        prop = CUmemAllocationProp(CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, CUmemLocation(CU_MEM_LOCATION_TYPE_DEVICE, domain_idx))
        handle, fd = ctypes.c_uint64(), ctypes.c_int()
        check(cuda.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(num_bytes), ctypes.byref(prop), ctypes.c_ulonglong(0)))
        check(cuda.cuMemExportToShareableHandle(ctypes.byref(fd), handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, ctypes.c_ulonglong(0)))
        socket.send_fds(sock, [b'f'], [fd.value])
        os.close(fd.value)
        check(cuda.cuMemRelease(handle))


def main(sock_fd: int, device: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    with tempfile.TemporaryDirectory(prefix='deep_gemm_mps_') as pipe_dir:
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = pipe_dir
        daemon_log = Path(pipe_dir, 'daemon.log')
        with daemon_log.open('w') as log:
            daemon = subprocess.Popen(['nvidia-cuda-mps-control', '-f'], env={**os.environ, 'CUDA_VISIBLE_DEVICES': device}, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, preexec_fn=lambda: libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM))
        try:
            while not Path(pipe_dir, 'control').exists():
                assert daemon.poll() is None, f'The MPS control daemon failed:\n{daemon_log.read_text()}'
                time.sleep(0.01)
            subprocess.run(['nvidia-cuda-mps-control'], input=f'start_server -uid {os.getuid()} -mlopart\n', text=True, check=True)
            with socket.socket(fileno=sock_fd) as sock:
                serve(sock)
        finally:
            daemon.terminate()
            daemon.wait()


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
