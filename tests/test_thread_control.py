from mtspy import thread_control
import multiprocessing


def test_thread_control():
    n = multiprocessing.cpu_count()
    thread_control.set_num_threads(n)

    assert(thread_control.get_max_threads() == n)

    # Request the use of less
    with thread_control(1, True):
        assert(thread_control.get_max_threads() == 1)

    # More threads than available
    with thread_control(n + 1, True):
        assert(thread_control.get_max_threads() == n)
