import argparse
import numpy
import sys

from mtspy import thread_control, matmat
from mtspy.utils import get_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "name", help="Matrix name from SuiteSparse Matrix Collection (group/name) eg: HB/west0479", type=str)
parser.add_argument("--nrows", default=10, help="Number of rows of the dense matrix", type=int)
parser.add_argument("--threads", default=1, help="Maximum number of threads to use", type=int)
parser.add_argument("--verbose", default=0, help="Use verbose mode", type=int)
args = parser.parse_args()

A = get_matrix(args.name, args.verbose)
m, n = A.shape
x = numpy.ones((m, args.nrows), A.dtype)

with thread_control(args.threads) as th:
    y0 = A @ x

sp_time = th.elapsed_time

with thread_control(args.threads) as th:
    y1 = matmat(A, x)

mt_time = th.elapsed_time

# TODO: Create summary and add to a csv
# Print summary
speedup = sp_time / mt_time
print("Speedup:", speedup)
