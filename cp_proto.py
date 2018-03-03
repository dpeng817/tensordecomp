import lin_alg_proto as la
import numpy as np

def fr_norm_tensor( tensor, approx_tensor ):
    """
    returns the frobenius norm of the difference of two tensors
    :param tensor: item 1
    :param approx_tensor: item 2
    :return : frobenius norm of the difference of the two
    """
    return np.linalg.norm( tensor - approx_tensor )

def recomp( factor_matrices, lambdas, orig_shape ):
    """
    function that recomposes a tensor from the factor matrices given
    :param factor_matrices: list of factor matrices
    """
#   Each factor matrix has exactly r column vectors, where r is the number of unique rank one components
    num_components = len( factor_matrices[0] )
    num_factors = len( factor_matrices )
    t_r = np.zeros( orig_shape )
    for i in range( 0 , num_components ):
        cur = lambdas[0] * factor_matrices[0][ : , i ]
        for j in range( 1 , num_factors ):
            cur = np.multiply.outer( cur , lambdas[j] * factor_matrices[j][ : , i ] )
        t_r = t_r + cur
    return t_r


def cp_decomp( tensor, num_factors, epochs ):
    """
    function to carry out a CP decomposition for a given tensor
    :param tensor: input tensor to carry out decomposition for
    :param num_factors: number of rank-one factors to fit for
    :param epochs: maximum number iterations
    :return : error rate, weight vector, factor matrices
    """
    shape = tensor.shape
    N = len( shape )
    factor_matrices = [None] * N
    total_elements = 1
#   total number of scalar elements in tensor
    for i in range(0, N):
        total_elements *= shape[i]
#   initialize factors to random values
    for i in range( 0 , N ):
        factor_matrices[i] = np.full( (shape[i],num_factors), 1 )
#   initialize previous cost to infinity so it always allows first iteration to pass
    prev_cost = 0
    cur_cost = float( "inf" )
    ep_passed = 0
    lambdas = np.zeros( num_factors )
    while True:
        for n in range( 0 , N ):
            v = np.full( (num_factors, num_factors) , 1 )
            for i in [x for x in range( 0 , N ) if x != n]:
                v = la.hadamard(
                        v ,
                        np.transpose( factor_matrices[i] ) @ factor_matrices[i]
                        )
            v_inv = np.linalg.pinv( v )
            k_temp = np.full( (1) , 1 )
            for i in [x for x in range( N - 1 , -1 , -1 ) if x != n]:
                k_temp = la.khatri_rao( k_temp , factor_matrices[i] )
            factor_matrices[n] = np.reshape( 
                    np.copy(tensor) , 
                    (shape[n], int (total_elements / shape[n])) ) @ k_temp @ v_inv
            lambdas[n] = np.linalg.norm( factor_matrices[n] )
            factor_matrices[n] = ( 1 / lambdas[n] ) * factor_matrices[n]
        est = recomp( factor_matrices , lambdas , shape )
        prev_cost = cur_cost
        cur_cost = fr_norm_tensor( tensor, est)
        ep_passed += 1
        if not cur_cost - prev_cost <= 0 and ep_passed < epochs:
            break
    print(ep_passed)
    return lambdas, factor_matrices


                        



t = np.full( (2,2,2), 1 )
c = cp_decomp(t, 3, 20)
print( recomp( c[1], c[0], t.shape))
