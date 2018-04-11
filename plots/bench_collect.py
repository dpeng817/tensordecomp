import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess as sp
import os
import csv
"""
Set of functions for running scripts specified number of times and scraping data when benchmarking numpy functions
"""

def test_matrix_creation(max_dim_size, interval, num_samples, cores=1):
    """
    Purpose:
        Run matrix creation tests on dimension size, scaling up from 1 in intervals of interval
        Run tests using square matrices for consistency
    :param max_dim_size: largest dimension size to time
    :param interval: interval to increment by
    :param num_samples: number of samples to average over
    """
    dims = []
    times = []
#   create file
    with open('data/data_norm_m_creation.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
#       writer.writerow(['dimension', 'time', 'num_samples'])
#   write data
    for d in range(1, max_dim_size, interval * cores):
        procs = []
        for i in range(d, d + (interval * cores), interval):
            print(i)
            proc = sp.Popen('python3 bench_norm_m_creation.py ' + str(i) + ' ' + str(num_samples))
            procs.append(proc.pid)
        for proc in procs:
            os.waitpid(proc, 0)
#   gather data
    with open('data/data_norm_m_creation.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';',
                quotechar='|')
        for row in reader:
            print(row)
            dims.append(int(row[0]))
            times.append(float(row[1]) / float(row[2]))

#   plot data
    plt.xlabel("Matrix dimension (square matrix)")
    plt.ylabel("Time elapsed (sec)")
    plt.title('Random Gaussian Matrix Generation Benchmarking')
    plt.plot(range(1, max_dim_size, interval), times)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('figures/test_matrix_creation.eps', format='eps', dpi=1000)

def test_matrix_matrix_mult(max_d_size, max_k_size,
        d_interval, k_interval,
        num_samples, cores=1):
    """
    Purpose:
        Run dxd against dxk matrix multiplication tests, scaling up d in
        intervals of d_interval, each color represents different k
        :param max_d_size: dxd cross dxk, maximum d
        :param max_k_size: dxd cross dxk, maximum k
        :param d_interval: interval to increase d by
        :param k_interval: interval to increase k by
        :param num_samples: number of samples to test each point for
    """
    heatmap_data = {}
    plt.tight_layout()
    plt.figure(0)
    for k in range(1, max_k_size, k_interval):
        dims = []
        times = []
        with open('data/data_mm_mult.csv', 'w') as f:
            f.truncate()
        for d in range(1, max_dim_size, interval * cores):
            procs = []
            for i in range(d, d + (interval * cores), interval):
                print(i)
                proc = sp.Popen('python3 bench_mm_mult.py ' + str(i) + ' ' + str(k) + ' ' + str(num_samples))
                procs.append(proc.pid)
            for proc in procs:
                os.waitpid(proc, 0)
        with open('data/data_mm_mult.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';',
                    quotechar='|')
            for row in reader:
                print(row)
                dims.append(int(row[0]))
                times.append(float(row[2]) / float(row[3]))
        print(len(times))
        plt.plot(dims, times, label='k = ' + str(k))
        heatmap_data[k] = times 
    plt.xlabel("Matrix dimension (square matrix) for first")
    plt.ylabel("Time elapsed (sec)")
    plt.xscale('log')
    plt.title("Matrix by Matrix Multiplication Benchmark")
    plt.legend(loc='best')
    plt.savefig('figures/test_matrix_matrix_mult.eps', format='eps', dpi=1000)
    plt.figure(1)
    plt.xscale('linear')
    plt.xlim(0, int(max_d_size / d_interval))
    df = pd.DataFrame(heatmap_data)
    plt.pcolor(df.T)
    plt.ylabel("Second matrix dimension (k)")
    plt.xlabel("First matrix dimension (d/"+str(d_interval)+")")
    plt.colorbar()
    plt.title('Matrix by Matrix Multiplication Heatmap')
    plt.savefig('figures/est_matrix_matrix_mult_heatmap.eps', format='eps', dpi=1000)

def test_inner_product_mult(max_d_size, d_interval, num_samples, cores=1):
    """
    Purpose:
        Run d dot d vector inner product tests, scaling up d in intervals of d_interval
    :param max_d_size: max dimension of each vector
    :param d_interval: interval to increment by
    :param num_samples: number of samples to average over to obtain each point
    """
    #   create file
    with open('data/data_dot_prod.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    dims = []
    times = []
    for d in range(1, max_dim_size, interval * cores):
        procs = []
        for i in range(d, d + (interval * cores), interval):
            print(i)
            proc = sp.Popen('python3 bench_dot_prod.py ' + str(i) + ' ' + str(num_samples))
            procs.append(proc.pid)
        for proc in procs:
            os.waitpid(proc)
    with open('data_dot_product.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(dims)):
            writer.writerow(['dimension:('+str(dims[i])+'),time:'+str(times[i])+',num_samples:'+str(num_samples)])
#   gather data
    with open('data/data_dot_prod.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';',
                quotechar='|')
        for row in reader:
            dims.append(int(row[0]))
            times.append(float(row[1]) / float(row[2]))
    plt.figure(2)
    plt.xlabel("Size of each vector")
    plt.ylabel("Time elapsed (sec)")
    plt.plot(dims, times)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('figures/test_inner_product_mult.eps', format='eps', dpi=1000)

test_matrix_creation(1000000, 100, 20)
test_matrix_matrix_mult(1000000, 500, 100, 50, 20)
test_inner_product_mult(1000000, 100, 20)
