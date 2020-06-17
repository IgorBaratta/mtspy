#!/bin/sh

matrix_list=('vanHeukelum/cage15' 'Schenk/nlpkkt200' 'Fluorem/HV15R' 'Janna/Queen_4147')

for matrix in "${matrix_list[@]}"
do
    for i in {1..24}
    do
        python3 spmv_benchmark.py "$matrix" --threads=$i
    done
done