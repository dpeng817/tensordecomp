'''
This is a prototype module that defines static methods useful in tensor decomposition.
Defined in this class are the kronecker, khatri-rao, and hadamard products.
numpy is the only required package.
'''
import numpy as np


def hadamard( m1, m2 ):       
    """
    The Hadamard product is the entry-wise product of two matrices.
    This is equivalent to the multiply(a, b) operation in numpy,
    but is defined here for more precise naming.
    :param m1: input matrix 1
    :param m2: input matrix 2

    :return : resultant matrix of the same size
    """
    return np.multiply( m1, m2 )

def kronecker( m1, m2 ):
    """
    The Kronecker product is a generalization of the outer product,
    resulting in a block matrix. If A is an m by n matrix and B is a 
    p by q matrix, then the Kronecker Product is mp by nq.
    :param m1: input matrix 1 (m x n)
    :param m2: input matrix 2 (p x q)
    :return : block matrix from executing kronecker product (mp x nq)
    """
    s1 = m1.shape
    s2 = m2.shape
    m_r = np.empty( (s1[0] * s2[0] , s1[1] * s2[1]) )           # np.empty faster than np.zeros
    for r in range( 0 , s1[0] ):
        for c in range( 0 , s1[1] ):
            m_r[ s2[0] * r : s2[0] * (r+1) , s2[1] * c : s2[1] * (c+1) ] = m1[ r , c ] * m2
    return m_r

def khatri_rao( m1, m2 ):
    """
    The Khatri-Rao product is the column-wise Kronecker product,
    resulting in again a block matrix. If A is (m x n) and B is
    (p x n) then the resulting matrix is (mp x n).
    :param m1: input matrix 1 (m x n)
    :param m2: input matrix 2 (p x n)
    :return : block matrix from executing khatri-rao product (mp x n)
    """
    s1 = m1.shape
    s2 = m2.shape
    if len(s1) == 1:
        return m1 * m2
    m_r = np.empty( (s1[0] * s2[0] , s1[1]) )
    for c in range( 0 , s1[1]):
        for r in range( 0 , s1[0]):
            m_r[ s2[0] * r : s2[0] * (r+1) , c ] = m1[ r , c ] * m2[ : , c ]
    return m_r

