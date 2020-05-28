import mtspy_cpp
import time


class thread_control():
    @classmethod
    def get_max_threads(csl):
        return mtspy_cpp.get_max_threads()

    @classmethod
    def get_num_threads(csl):
        return mtspy_cpp.get_num_threads()

    @classmethod
    def set_num_threads(cls, n):
        assert type(n) == int
        mtspy_cpp.set_num_threads(n)

    def __init__(self, num_threads, timer=False):
        self._cached_max_threads = self.get_max_threads()
        self._num_threads = min(num_threads, self._cached_max_threads)
        self._timer = timer

    def __enter__(self):
        self._start_time = time.perf_counter()
        self.set_num_threads(self._num_threads)
        return self

    def __exit__(self, type, value, traceback):
        self.set_num_threads(self._cached_max_threads)
        self._num_threads = self.get_num_threads()
        self.elapsed_time = time.perf_counter() - self._start_time
        if self._timer:
            print("Elapsed time (s): ", self.elapsed_time)

    @property
    def num_threads(self):
        return self._num_threads
