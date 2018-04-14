import cupy as cp
import time
import csv
import sys
"""
Lightweight script that benchmarks the performance of matrix-matrix multiplication.
Command line arguments: [d, k, num_samples]
"""
d = int(sys.argv[1])
num_samples = int(sys.argv[2])
start = time.time()
for i in range(0, num_samples):
    cp.random.standard_normal((d, d))
end = time.time()
with open('data/data_norm_m_creation.csv', 'a') as f:
    writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(d), str(end-start), str(num_samples)])
with open('data/all_norm_m_creation.csv', 'a') as f:
    writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(d), str(end-start), str(num_samples)])