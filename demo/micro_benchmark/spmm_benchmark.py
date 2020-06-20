import argparse
import numpy
import sys

from mtspy import thread_control, matmat
from mtspy.utils import get_csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Matrix name from SuiteSparse Collection, eg: HB/west0479", type=str)
parser.add_argument("--ncols", default=10, help="Number of columns of the dense matrix", type=int)
parser.add_argument("--threads", default=1, help="Maximum number of threads to use", type=int)
parser.add_argument("--verbose", default=0, help="Use verbose mode", type=int)
args = parser.parse_args()

thread_control.set_num_threads(args.threads)
A = get_csr_matrix(args.name, args.verbose)
m, n = A.shape
x = numpy.ones((n, args.ncols), A.dtype)

with thread_control(args.threads) as th:
    y0 = A @ x

sp_time = th.elapsed_time
sp_gflops = (2 * A.nnz * args.ncols / sp_time) * 1e-9

with thread_control(args.threads) as th:
    y1 = matmat(A, x)

mt_time = th.elapsed_time
mt_gflops = (2 * A.nnz * args.ncols / mt_time) * 1e-9

# TODO: add summary to a csv
# Print summary
speedup = sp_time / mt_time

print("\n=========================")
print("Sparse Matrix Data:")
print("Name: \t\t", args.name)
print("NNZ: \t\t", A.nnz)
print("nrows: \t\t", m)
print("ncols: \t\t", n)
print("Data Type: \t", A.dtype)
print("Index Type: \t", A.indices.dtype)


print("\nDense Matrix Data:")
print("nrows: \t\t", x.shape[0])
print("ncols: \t\t", x.shape[1])


print("\n=========================")
print("SpMM Time")
print("mtspy(s):\t", mt_time)
print("Ref (s):\t", sp_time)
print("Sepeedup: \t", speedup)

print("\nEstimated GFLOPS")
print("mtspy: \t\t", mt_gflops)
print("Ref: \t\t", sp_gflops)

print("\n# threads: \t", thread_control.get_max_threads())
