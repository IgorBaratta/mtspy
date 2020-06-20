#!/bin/sh

matrix_list=('vanHeukelum/cage15' 'Schenk/nlpkkt200' 'Fluorem/HV15R' 'Janna/Queen_4147')

for matrix in "${matrix_list[@]}"
do
    for i in {1..5}
    do
        echo "Matrix $matrix, run number $i"
        python3 spmv_single_benchmark.py "$matrix" --threads=12
    done
done
