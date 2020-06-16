import argparse
import numpy
import sys

from mtspy import thread_control, matvec
from mtspy.utils import get_matrix

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Matrix name from SuiteSparse Collection eg: HB/west0479", type=str)
parser.add_argument("--threads", default=1, help="Maximum number of threads to use", type=int)
parser.add_argument("--verbose", default=0, help="Use verbose mode", type=int)
args = parser.parse_args()

thread_control.set_num_threads(args.threads)
A = get_matrix(args.name, args.verbose)
m, n = A.shape
x = numpy.ones(m, A.dtype)

with thread_control(args.threads) as th:
    y0 = A @ x

sp_time = th.elapsed_time
sp_gflops = (2 * A.nnz / sp_time) * 1e-9

with thread_control(args.threads) as th:
    y1 = matvec(A, x)

mt_time = th.elapsed_time
mt_gflops = (2 * A.nnz / mt_time) * 1e-9
speedup = sp_time / mt_time

# TODO: Create summary and add to a csv

print("\n=========================")
print("Sparse Matrix Data:")
print("Name: \t\t", args.name)
print("NNZ: \t\t", A.nnz)
print("nrows: \t\t", m)
print("ncols: \t\t", n)
print("Data Type: \t", A.dtype)
print("Index Type: \t", A.indices.dtype)


print("\n=========================")
print("SpMV Time")
print("mtspy(s):\t", mt_time)
print("Ref (s):\t", sp_time)
print("Sepeedup: \t", speedup)

print("\nEstimated GFLOPS")
print("mtspy: \t\t", mt_gflops)
print("Ref: \t\t", sp_gflops)

print("\n# threads: \t", thread_control.get_max_threads())
