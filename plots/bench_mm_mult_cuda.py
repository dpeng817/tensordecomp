import cupy as cp
import time
import csv
import sys
"""
Lightweight script that benchmarks the performance of matrix-matrix multiplication.
Command line arguments: [d, k, num_samples]
"""

d = int(sys.argv[1])
k = int(sys.argv[2])
num_samples = int(sys.argv[3])
m1 = cp.full((d,d), 0.5)
m2 = cp.full((d,k), 0.5)
start = time.time()
for i in range(0, num_samples):
    cp.matmul(m1, m2)
end = time.time()
with open('data/all_k_mm_mult_cuda.csv', 'a') as f:
    writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(d),str(k),str(end-start),str(num_samples)])
with open('data/data_mm_mult_cuda.csv', 'a') as f:
    writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(d),str(k),str(end-start),str(num_samples)])