"""CPU-only Linux process-identity tests. Every signalled process is created here."""
import signal,subprocess,sys,unittest
from a1_remote_supervisor import process_ticks,stop_owned_group


class OwnProcessIsolation(unittest.TestCase):
    def spawn(self,session=True):return subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=session)

    def cleanup(self,p):
        if p.poll() is None:p.terminate()
        p.wait(timeout=5)

    def test_only_own_group_stops_and_neighbour_survives(self):
        own=self.spawn();other=self.spawn()
        try:
            self.assertTrue(stop_owned_group(own,process_ticks(own.pid),signal.SIGTERM))
            own.wait(timeout=5);self.assertIsNone(other.poll())
        finally:self.cleanup(own);self.cleanup(other)

    def test_mismatched_start_identity_never_signalled(self):
        own=self.spawn()
        try:
            with self.assertRaises(AssertionError):stop_owned_group(own,'wrong',signal.SIGTERM)
            self.assertIsNone(own.poll())
        finally:self.cleanup(own)

    def test_nonisolated_group_never_signalled(self):
        own=self.spawn(session=False)
        try:
            with self.assertRaises(AssertionError):stop_owned_group(own,process_ticks(own.pid),signal.SIGTERM)
            self.assertIsNone(own.poll())
        finally:self.cleanup(own)


if __name__=='__main__':unittest.main()
