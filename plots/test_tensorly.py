import tensorly.random as rnd
import matplotlib.pyplot as plt
import tensorly as tl
import time
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
"""
This is a set of functions for tracking the performance of various 
tensorly functions that have importance to machine learning tasks.
"""

def test_random_cp_creation(max_d_size, num_dims, d_interval, 
        max_rank, rank_interval, num_samples):
    """
    Purpose:
        benchmark the creation of a randomly generated CP decomposable tensor
        run tests using hypercube tensors for consistency
    :param max_d_size: maximum dimension size that each mode will reach
    :param num_dims: number of dimensions to test along
    :param d_interval: size of interval to jump by for each data point
    :param max_rank: maximum rank to test against
    :param rank_interval: size of interval for rank to jump by for each data point
    :param num_samples: number of items to sample over for each data point
    """
    for r in range(1, max_rank, rank_interval):
        dims = []
        times = []
        for d in range(1, max_d_size, d_interval):
            time_sum = 0
            for n in range(0, num_samples):
                shp = tuple([d] * num_dims)
                start = time.time()
                rnd.cp_tensor(shp, r)
                end = time.time()
                time_sum += end - start
            dims.append(d)
            times.append(time_sum / num_samples)
        plt.plot(dims, times, label='r = ' + str(r))
    plt.xlabel("Matrix dimension (square matrix)")
    plt.ylabel("Time elapsed (sec)")
    plt.title('Random CP-Decomposable Tensor Generation Benchmarking')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('test_random_cp_creation.eps', format='eps', dpi=1000)

def test_random_tucker_creation(max_d_size, num_dims, d_interval, 
        max_rank, rank_interval, num_samples):
    """
    Purpose:
        benchmark the creation of a randomly generated Tucker decomposable tensor
        run tests using hypercube tensors for consistency
    :param max_d_size: maximum dimension size that each mode will reach
    :param num_dims: number of dimensions to test along
    :param d_interval: size of interval to jump by for each data point
    :param max_rank: maximum rank to test against
    :param rank_interval: size of interval for rank to jump by for each data point
    :param num_samples: number of items to sample over for each data point
    """
    for r in range(1, max_rank, rank_interval):
        dims = []
        times = []
        for d in range(1, max_d_size, d_interval):
            time_sum = 0
            for n in range(0, num_samples):
                shp = tuple([d] * num_dims)
                start = time.time()
                rnd.tucker_tensor(shp, r)
                end = time.time()
                time_sum += end - start
            dims.append(d)
            times.append(time_sum / num_samples)
            plt.plot(dims, times, label='r = ' + str(r))
    plt.xlabel("Matrix dimension (square matrix)")
    plt.ylabel("Time elapsed (sec)")
    plt.title('Random Tucker-Decomposable Tensor Generation Benchmarking')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('test_random_tucker_creation.eps', format='eps', dpi=1000)

def test_cp_decomposition(max_d_size, num_dims, d_interval,
        max_rank, rank_interval, num_samples):
    """
    Purpose:
        benchmark the cp decomposition of a randomly generated CP decomposable tensor
        run tests using hypercube tensors for consistency
    :param max_d_size: maximum dimension size that each mode will reach
    :param num_dims: number of dimensions to test along
    :param d_interval: size of interval to jump by for each data point
    :param max_rank: maximum rank to test against
    :param rank_interval: size of interval for rank to jump by for each data point
    :param num_samples: number of items to sample over for each data point
    """
    rand_state = 5
    for r in range(1, max_rank, rank_interval):
        dims = []
        times = []
        for d in range(2, max_d_size, d_interval):
            time_sum = 0
            print(d)
            for n in range(0, num_samples):
                shp = tuple([d] * num_dims)
                t = rnd.cp_tensor(shp, r, full=True, random_state=rand_state)
                start = time.time()
                parafac(t, rank=r, tol=10e-6, random_state=rand_state)
                end = time.time()
                time_sum += end - start
            dims.append(d)
            times.append(time_sum / num_samples)
        plt.plot(dims, times, label='r = ' + str(r))
    plt.xlabel("Matrix dimension (square matrix)")
    plt.ylabel("Time elapsed (sec)")
    plt.legend(loc='best')
    plt.savefig('test_cp_decomposition.eps', format='eps', dpi=1000)

def test_tucker_decomposition(max_d_size, num_dims, d_interval,
        max_rank, rank_interval, num_samples):
    """
    Purpose:
        benchmark the tucker decomposition of a randomly generated tucker decomposable tensor
        run tests using hypercube tensors for consistency
    :param max_d_size: maximum dimension size that each mode will reach
    :param num_dims: number of dimensions to test along
    :param d_interval: size of interval to jump by for each data point
    :param max_rank: maximum rank to test against
    :param rank_interval: size of interval for rank to jump by for each data point
    :param num_samples: number of items to sample over for each data point
    """
    rand_state = 5
    for r in range(1, max_rank, rank_interval):
        dims = []
        times = []
        for d in range(r, max_d_size, d_interval):
            time_sum = 0
            print(d)
            for n in range(0, num_samples):
                shp = tuple([d] * num_dims)
                t = rnd.tucker_tensor(shp, r, full=True, random_state=rand_state)
                start = time.time()
                tucker(t, tol=10e-6, random_state=rand_state)
                end = time.time()
                time_sum += end - start
            dims.append(d)
            times.append(time_sum / num_samples)
        plt.plot(dims, times, label='r = ' + str(r))
    plt.xlabel("Matrix dimension (square matrix)")
    plt.ylabel("Time elapsed (sec)")
    plt.legend(loc='best')
    plt.savefig('test_tucker_decomposition.eps', format='eps', dpi=1000)

test_random_cp_creation(500, 4, 10, 5, 5, 20)
test_random_tucker_creation(500, 4, 10, 5, 5, 20)
test_tucker_decomposition(50, 4, 10, 4, 5, 20)
test_tucker_decomposition(50, 4, 10, 4, 5, 20)


            


