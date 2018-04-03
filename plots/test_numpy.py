import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
"""
This is a set of functions to create plots tracking the performance of various numpy functions that have importance to machine learning tasks.
"""


def test_matrix_creation(max_dim_size, interval, num_samples):
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
    dim = 1
    while dim <= max_dim_size:
        print(dim * dim)
        start = time.time()
        for i in range(0, num_samples):
            np.random.standard_normal((dim, dim))
        end = time.time()
        dims.append(dim)
        times.append((end - start) / num_samples)
        dim += interval
    plt.xlabel("Matrix dimension (square matrix)")
    plt.ylabel("Time elapsed (sec)")
    plt.title('Random Gaussian Matrix Generation Benchmarking')
    plt.plot(dims, times)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('test_matrix_creation.eps', format='eps', dpi=1000)
    with open('data_matrix_creation.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(dims)):
            writer.writerow(['dimension:('+str(dims[i])+'),time:'+str(times[i])+',num_samples:'+str(num_samples)])

def test_matrix_matrix_mult(max_d_size, max_k_size,
        d_interval, k_interval,
        num_samples):
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
    rows = []
    for k in range(1, max_k_size, k_interval):
        dims = []
        times = []
        for d in range(1, max_d_size, d_interval):
            print(d)
            m1 = np.full((d,d), 0.5)
            m2 = np.full((d, k), 0.5)
            start = time.time()
            for n in range(0, num_samples):
                np.matmul(m1, m2)
            end = time.time()
            dims.append(d)
            times.append((end - start) / num_samples)
            rows.append('dimension:('+str(d)+','+str(k)+'),time:'+str(end-start)+',num_samples:'+str(num_samples))
        plt.plot(dims, times, label='k = ' + str(k))
        heatmap_data[k] = times 
    plt.xlabel("Matrix dimension (square matrix) for first")
    plt.ylabel("Time elapsed (sec)")
    plt.xscale('log')
    plt.title("Matrix by Matrix Multiplication Benchmark")
    plt.legend(loc='best')
    plt.savefig('test_matrix_matrix_mult.eps', format='eps', dpi=1000)
    plt.figure(1)
    plt.xscale('linear')
    plt.xlim(0, int(max_d_size / d_interval))
    df = pd.DataFrame(heatmap_data)
    plt.pcolor(df.T)
    plt.ylabel("Second matrix dimension (k)")
    plt.xlabel("First matrix dimension (d/"+str(d_interval)+")")
    plt.colorbar()
    plt.title('Matrix by Matrix Multiplication Heatmap')
    plt.savefig('test_matrix_matrix_mult_heatmap.eps', format='eps', dpi=1000)
    with open('data_matrix_matrix_mult.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(rows)):
            writer.writerow([rows[i]])

def test_inner_product_mult(max_d_size, d_interval, num_samples):
    """
    Purpose:
        Run d dot d vector inner product tests, scaling up d in intervals of d_interval
    :param max_d_size: max dimension of each vector
    :param d_interval: interval to increment by
    :param num_samples: number of samples to average over to obtain each point
    """
    dims = []
    times = []
    for d in range(1, max_d_size, d_interval):
        v1 = np.full((d), 0.5)
        v2 = np.full((d), 0.5)
        start = time.time()
        for n in range(0, num_samples):
            np.dot(v1, v2)
        end = time.time()
        dims.append(d)
        times.append((end - start) / num_samples)
    plt.figure(2)
    plt.xlabel("Size of each vector")
    plt.ylabel("Time elapsed (sec)")
    plt.plot(dims, times)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('test_inner_product_mult.eps', format='eps', dpi=1000)
    with open('data_inner_product.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(dims)):
            writer.writerow(['dimension:('+str(dims[i])+'),time:'+str(times[i])+',num_samples:'+str(num_samples)])

test_matrix_creation(100, 10, 20)
test_matrix_matrix_mult(100, 10, 10, 1, 20)
test_inner_product_mult(100, 10, 20)
