import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

fork_pdb = ForkedPdb()
stdin_lock = None
def init_fork_pdb(_stdin_lock):
    global stdin_lock
    stdin_lock = _stdin_lock

def set_trace():
    global stdin_lock
    stdin_lock.acquire()
    import os
    print(f"lock acquired: process {os.getpid()}")
    fork_pdb.set_trace()