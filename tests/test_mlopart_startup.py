"""Exercise the startup handshake with owned CPU-only helper processes."""

import importlib.util
import os
import select
import signal
import socket
import subprocess
import sys
import time
import unittest
import uuid
import warnings
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'deep_gemm/utils/mlopart.py'
spec = importlib.util.spec_from_file_location('mlopart_startup_test', SOURCE)
mlopart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlopart)
real_popen = subprocess.Popen


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)

    def start_fake(self, program):
        def launch(args, **kwargs):
            fd = kwargs['pass_fds'][0]
            child = real_popen([sys.executable, '-c', program, str(fd)],
                               pass_fds=(fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.children.append(child)
            return child
        return launch

    def test_stalled_handshake_times_out_and_reaps_child(self):
        # The baseline eventually exits after three seconds, so a failed test
        # cannot leave a stuck helper. The fixed code should stop it earlier.
        fake = self.start_fake('import socket,sys,time; s=socket.socket(fileno=int(sys.argv[1])); time.sleep(3)')
        with mock.patch.dict(os.environ, {'DG_MLOPART_STARTUP_TIMEOUT': '0.1'}), \
             mock.patch.object(mlopart.subprocess, 'Popen', side_effect=fake), \
             warnings.catch_warnings():
            warnings.simplefilter('ignore')
            start = time.monotonic()
            client = mlopart.MLOPart(uuid.UUID(int=1).bytes)
            elapsed = time.monotonic() - start
            try:
                self.assertFalse(client.available)
                self.assertLess(elapsed, 2)
                self.assertIsNotNone(self.children[0].poll())
            finally:
                client.close()

    def test_successful_handshake_keeps_socket_usable(self):
        for timeout in ('0', '2'):
            with self.subTest(timeout=timeout):
                fake = self.start_fake("import socket,sys; s=socket.socket(fileno=int(sys.argv[1])); s.sendall(b'r'); s.recv(1)")
                with mock.patch.dict(os.environ, {'DG_MLOPART_STARTUP_TIMEOUT': timeout}), \
                     mock.patch.object(mlopart.subprocess, 'Popen', side_effect=fake):
                    client = mlopart.MLOPart(uuid.UUID(int=1).bytes)
                    try:
                        self.assertTrue(client.available)
                        self.assertIsNone(client.sock.gettimeout())
                    finally:
                        client.close()
                    self.assertIsNotNone(self.children[-1].poll())

    def test_invalid_timeouts_are_rejected_before_starting_helper(self):
        for value in ('-1', 'nan', 'inf', 'invalid'):
            with self.subTest(value=value), \
                 mock.patch.dict(os.environ, {'DG_MLOPART_STARTUP_TIMEOUT': value}), \
                 mock.patch.object(mlopart.subprocess, 'Popen') as popen:
                with self.assertRaises(ValueError):
                    mlopart.MLOPart(uuid.UUID(int=1).bytes)
                popen.assert_not_called()

    def test_popen_failure_closes_both_sockets(self):
        sockets = []
        real_socketpair = socket.socketpair

        def socketpair():
            pair = real_socketpair()
            sockets.extend(pair)
            return pair

        with mock.patch.dict(os.environ, {'DG_MLOPART_STARTUP_TIMEOUT': '2'}), \
             mock.patch.object(mlopart.socket, 'socketpair', side_effect=socketpair), \
             mock.patch.object(mlopart.subprocess, 'Popen', side_effect=OSError('spawn failed')), \
             self.assertRaisesRegex(OSError, 'spawn failed'):
            mlopart.MLOPart(uuid.UUID(int=1).bytes)
        self.assertEqual([s.fileno() for s in sockets], [-1, -1])

    def test_timeout_escalates_when_helper_ignores_terminate(self):
        child = mock.Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('helper', 1), 0, 0]
        peer_fds = []

        def launch(args, **kwargs):
            peer_fds.append(os.dup(kwargs['pass_fds'][0]))
            return child

        try:
            with mock.patch.dict(os.environ, {'DG_MLOPART_STARTUP_TIMEOUT': '0.05'}), \
                 mock.patch.object(mlopart.subprocess, 'Popen', side_effect=launch), \
                 warnings.catch_warnings():
                warnings.simplefilter('ignore')
                client = mlopart.MLOPart(uuid.UUID(int=1).bytes)
                try:
                    self.assertFalse(client.available)
                    child.terminate.assert_called_once()
                    child.kill.assert_called_once()
                finally:
                    client.close()
        finally:
            for fd in peer_fds:
                os.close(fd)

    def test_real_unresponsive_helper_is_killed_and_reaped(self):
        def launch(args, **kwargs):
            fd = kwargs['pass_fds'][0]
            ready_read, ready_write = os.pipe()
            program = (
                'import os,signal,socket,sys; '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                's=socket.socket(fileno=int(sys.argv[1])); '
                'os.write(int(sys.argv[2]), b"1"); s.recv(1)'
            )
            try:
                child = real_popen([sys.executable, '-c', program, str(fd), str(ready_write)],
                                   pass_fds=(fd, ready_write), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.children.append(child)
            finally:
                os.close(ready_write)
            try:
                self.assertTrue(select.select([ready_read], [], [], 5)[0])
                self.assertEqual(os.read(ready_read, 1), b'1')
            finally:
                os.close(ready_read)
            return child

        with mock.patch.dict(os.environ, {'DG_MLOPART_STARTUP_TIMEOUT': '0.1'}), \
             mock.patch.object(mlopart.subprocess, 'Popen', side_effect=launch), \
             warnings.catch_warnings():
            warnings.simplefilter('ignore')
            client = mlopart.MLOPart(uuid.UUID(int=1).bytes)
            try:
                self.assertFalse(client.available)
                self.assertEqual(self.children[-1].returncode, -signal.SIGKILL)
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
