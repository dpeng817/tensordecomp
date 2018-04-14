import numpy as np
import time
import csv
import sys
"""
Lightweight script that benchmarks the performance of matrix-matrix multiplication.
Command line arguments: [d, k, num_samples]
"""

d = int(sys.argv[1])
num_samples = int(sys.argv[2])
v1 = np.full((d), 0.5)
v2 = np.full((d), 0.5)
start = time.time()
for i in range(0, num_samples):
    np.dot(v1, v2)
end = time.time()
with open('data/data_dot_prod.csv', 'a') as f:
    writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(d),str(end-start),str(num_samples)])
with open('data/all_dot_prod.csv', 'a') as f:
    writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([str(d),str(end-start),str(num_samples)])
