import os
import csv
import argparse
import numpy
import sys

from mtspy import thread_control, matvec
from mtspy.utils import get_csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Matrix name from SuiteSparse Collection eg: HB/west0479", type=str)
parser.add_argument("--threads", default=1, help="Maximum number of threads to use", type=int)
parser.add_argument("--verbose", default=0, help="Use verbose mode", type=int)
args = parser.parse_args()

thread_control.set_num_threads(args.threads)
A = get_csr_matrix(args.name, args.verbose)
m, n = A.shape
x = numpy.ones(m, A.dtype)

max_threads = thread_control.get_max_threads()

col_name = ["Name", "nnz", "rows", "cols", "dtype", "int type", "threads",
            "mtspy time(s)", "mtspy gflops", "scipy time(s)", "scipy gflops"]

for thread_number in range(1, max_threads + 1):
    with thread_control(thread_number) as th:
        y0 = A @ x

    sp_time = th.elapsed_time
    sp_gflops = (2 * A.nnz / sp_time) * 1e-9

    with thread_control(thread_number) as th:
        y1 = matvec(A, x)

    mt_time = th.elapsed_time
    mt_gflops = (2 * A.nnz / mt_time) * 1e-9
    speedup = sp_time / mt_time

    exists = os.path.exists('performance.csv')
    row_list = [args.name, A.nnz, m, n, A.dtype, A.indices.dtype,
                thread_number, mt_time, mt_gflops, sp_time, sp_gflops]

    with open('performance.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter=',')
        if not exists:
            writer.writerow(col_name)
        writer.writerow(row_list)
