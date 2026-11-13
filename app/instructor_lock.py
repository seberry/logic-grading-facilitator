"""Hold an OS lock while an instructor server owns a grading data directory."""
from pathlib import Path
import os


class InstructorDataLock:
    def __init__(self, data_dir: Path):
        self.path = data_dir / '.instructor-grader.lock'
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+b')
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b' ')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError('Another instructor grader is using this data folder. Open that grader or stop it before starting another.') from None
        self.handle = handle
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.close()  # OS releases the lock, also on process exit.
            self.handle = None
