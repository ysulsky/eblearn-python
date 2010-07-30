# mode -*-python-*-

cimport cython

from _util cimport *
from _util import *

import_array()

def clear(np.ndarray m not None):
    cdef char *p
    cdef long size
    if not PyArray_ISCARRAY(m):
        return m.fill(0)
    p = m.data
    size = PyArray_SIZE(m) * PyArray_ITEMSIZE(m)
    memset(p, 0, size)

def sqmag(np.ndarray m not None):
    ''' sqmag(m) = |m|^2 '''
    cdef rtype_t *base
    cdef rtype_t x, acc = 0.
    cdef long i, size
    
    if not PyArray_ISCARRAY_RO(m) or (PyArray_TYPE(m) != NPY_RTYPE):
        return np.square(m).sum()
    
    base = <rtype_t*> m.data
    size = PyArray_SIZE(m)

    for i in range(size):
        x = base[i]
        acc += x*x

    return acc

def sqdist(np.ndarray a not None, np.ndarray b not None):
    ''' sqdist(a, b) = |a - b| '''
    cdef rtype_t *base_a, *base_b
    cdef rtype_t x, acc = 0.
    cdef long i, size
    
    assert (PyArray_SAMESHAPE(a, b)), "shapes don't match"
    
    if not PyArray_ISCARRAY_RO(a) or (PyArray_TYPE(a) != NPY_RTYPE) or \
       not PyArray_ISCARRAY_RO(b) or (PyArray_TYPE(b) != NPY_RTYPE):
        return sqmag(a - b)
    
    base_a = <rtype_t*> a.data
    base_b = <rtype_t*> b.data
    size   = PyArray_SIZE(a)
    
    for i in range(size):
        x = base_a[i] - base_b[i]
        acc += x*x
    
    return acc

cdef c_dtanh(rtype_t *x, long n, rtype_t *r):
    cdef long i
    cdef rtype_t v
    for i in range(n):
        v = x[i]
        if v < -12 or v > 12:
            r[i] = 0
        else:
            v = exp(-2. * v)
            r[i] = 4*v / ((v+1)*(v+1))

cdef c_ddtanh(rtype_t *x, long n, rtype_t *r):
    cdef long i
    cdef rtype_t u, v
    for i in range(n):
        v = x[i]
        if v < -12 or v > 12:
            r[i] = 0
        else:
            u = -2 * tanh(v)        # u = -2tanh(x)
            v = exp(-2. * v)
            v = 4*v / ((v+1)*(v+1)) # v = sech^2(x)
            r[i] = u * v

def dtanh(np.ndarray x not None, np.ndarray out = None):
    ''' dtanh(x) = sech^2(x) '''
    cdef rtype_t *inp, *outp, xi
    cdef long i, size
    cdef np.ndarray e

    if (PyArray_TYPE(x) != NPY_RTYPE):
        x = x.astype(rtype)
    if not PyArray_ISCARRAY_RO(x):
        x = x.copy()
    
    assert (out is None or PyArray_SAMESHAPE(x, out)), "shapes don't match"
    
    if out is None or \
       not PyArray_ISCARRAY(out) or (PyArray_TYPE(out) != NPY_RTYPE):
        e = PyArray_EMPTY(x.ndim, x.shape, NPY_RTYPE, 0)
    else:
        e = out
    
    c_dtanh(<rtype_t*> x.data, PyArray_SIZE(x), <rtype_t*> e.data)
    
    if out is None or out is e:
        return e

    out[:] = e
    return out

def ddtanh(np.ndarray x not None, np.ndarray out = None):
    ''' ddtanh(x) = -2 tanh(x) sech^2(x) '''
    cdef rtype_t *inp, *outp, xi
    cdef long i, size
    cdef np.ndarray e


    if (PyArray_TYPE(x) != NPY_RTYPE):
        x = x.astype(rtype)
    if not PyArray_ISCARRAY_RO(x):
        x = x.copy()
    
    assert (out is None or PyArray_SAMESHAPE(x, out)), "shapes don't match"
    
    if out is None or \
       not PyArray_ISCARRAY(out) or (PyArray_TYPE(out) != NPY_RTYPE):
        e = PyArray_EMPTY(x.ndim, x.shape, NPY_RTYPE, 0)
    else:
        e = out
    
    c_ddtanh(<rtype_t*> x.data, PyArray_SIZE(x), <rtype_t*> e.data)
    
    if out is None or out is e:
        return e

    out[:] = e
    return out


cdef rtype_t c_m1ldot(char *a, char *b, int n, int a_s0, int b_s0):
    cdef int i
    cdef rtype_t val = 0.
    cdef char *ai = a, *bi = b
    for i in range(n):
        val += (<rtype_t*>ai)[0] * (<rtype_t*>bi)[0]
        ai  += a_s0
        bi  += b_s0
    return val

cdef rtype_t c_m2ldot(char *a,  char *b,
                      int n0,   int n1,
                      int a_s0, int a_s1,
                      int b_s0, int b_s1):
    cdef int i, j
    cdef rtype_t val = 0.
    cdef char *ai, *aij
    cdef char *bi, *bij
    ai, bi = a, b
    for i in range(n0):
        aij, bij = ai, bi
        for j in range(n1):
            val += (<rtype_t*>aij)[0] * (<rtype_t*>bij)[0]
            aij += a_s1
            bij += b_s1
        ai += a_s0
        bi += b_s0
    return val

cdef rtype_t c_m3ldot(char *a,  char *b,
                      int n0,   int n1,   int n2,
                      int a_s0, int a_s1, int a_s2,
                      int b_s0, int b_s1, int b_s2):
    cdef int i, j, k
    cdef rtype_t val = 0.
    cdef char *ai, *aij, *aijk
    cdef char *bi, *bij, *bijk
    ai, bi = a, b
    for i in range(n0):
        aij, bij = ai, bi
        for j in range(n1):
            aijk, bijk = aij, bij
            for k in range(n2):
                val  += (<rtype_t*>aijk)[0] * (<rtype_t*>bijk)[0]
                aijk += a_s2
                bijk += b_s2
            aij += a_s1
            bij += b_s1
        ai += a_s0
        bi += b_s0
    return val

def m1ldot(np.ndarray a not None, np.ndarray b not None):
    ''' m2ldot(m1, m2) = sum_i (m1_i * m2_i) '''
    assert (a.ndim == 1 and b.ndim == 1), "wrong dimensions"
    assert (a.shape[0] == b.shape[0]),    "shapes don't match"
    if not PyArray_ISCARRAY_RO(a) or PyArray_TYPE(a) != NPY_RTYPE or \
       not PyArray_ISCARRAY_RO(b) or PyArray_TYPE(b) != NPY_RTYPE:
        return np.dot(a, b)
    return c_m1ldot(a.data, b.data, a.shape[0],
                    a.strides[0], b.strides[0])

m1ldot = np.dot # numpy has a faster dot-product

def m2ldot(np.ndarray a not None, np.ndarray b not None):
    ''' m2ldot(m1, m2) = sum_ij (m1_ij * m2_ij) '''
    assert (a.ndim == 2 and b.ndim == 2), "wrong dimensions"
    assert (a.shape[0] == b.shape[0] and 
            a.shape[1] == b.shape[1]),    "shapes don't match"
    if not PyArray_ISCARRAY_RO(a) or PyArray_TYPE(a) != NPY_RTYPE or \
       not PyArray_ISCARRAY_RO(b) or PyArray_TYPE(b) != NPY_RTYPE:
        return np.sum(a * b)
    return c_m2ldot(a.data, b.data,
                    a.shape[0], a.shape[1],
                    a.strides[0], a.strides[1],
                    b.strides[0], b.strides[1])

def m3ldot(np.ndarray a not None, np.ndarray b not None):
    ''' m2ldot(m1, m2) = sum_ijk (m1_ijk * m2_ijk) '''
    assert (a.ndim == 3 and b.ndim == 3), "wrong dimensions"
    assert(a.shape[0] == b.shape[0] and
           a.shape[1] == b.shape[1] and
           a.shape[2] == b.shape[2]),     "shapes don't match"
    if not PyArray_ISCARRAY_RO(a) or PyArray_TYPE(a) != NPY_RTYPE or \
       not PyArray_ISCARRAY_RO(b) or PyArray_TYPE(b) != NPY_RTYPE:
        return np.sum(a * b)
    return c_m3ldot(a.data, b.data,
                    a.shape[0], a.shape[1], a.shape[2],
                    a.strides[0], a.strides[1], a.strides[2],
                    b.strides[0], b.strides[1], b.strides[2])

def ldot(np.ndarray a not None, np.ndarray b not None):
    ''' ldot(a, b) = sum_i (a_i * b_i)
        where i runs over all dimensions '''
    cdef int ndim = a.ndim
    if ndim != b.ndim: ndim = -1
    if ndim == 1: return m1ldot(a, b)
    if ndim == 2: return m2ldot(a, b)
    if ndim == 3: return m3ldot(a, b)
    return np.sum(a * b)


cdef np.ndarray c_m2dotm1(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate):
    cdef int  i
    cdef int ni, nj
    
    cdef char *resi
    cdef char *m1i

    cdef int res_s0
    cdef int m1_s0, m1_s1
    cdef int m2_s0

    ni, nj = m1.shape[0], m1.shape[1]
    if res is None: res = PyArray_EMPTY(1, m1.shape, NPY_RTYPE, 0)
    
    res_s0 = res.strides[0]
    m1_s0  = m1.strides[0]
    m1_s1  = m1.strides[1]
    m2_s0  = m2.strides[0]
    
    m1i  = m1.data
    resi = res.data

    if accumulate:
        for i in range(ni):
            (<rtype_t*>resi)[0] += c_m1ldot(m1i, m2.data, nj, m1_s1, m2_s0)
        
            m1i  +=  m1_s0
            resi += res_s0
    else:
        for i in range(ni):
            (<rtype_t*>resi)[0]  = c_m1ldot(m1i, m2.data, nj, m1_s1, m2_s0)
        
            m1i  +=  m1_s0
            resi += res_s0        
    
    return res

cdef np.ndarray c_m4dotm2(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate):
    cdef int  i,  j
    cdef int ni, nj, nk, nl

    cdef char *resi, *resij
    cdef char *m1i, *m1ij

    cdef int res_s0, res_s1
    cdef int m1_s0, m1_s1, m1_s2, m1_s3
    cdef int m2_s0, m2_s1

    ni, nj, nk, nl = m1.shape[0], m1.shape[1], m1.shape[2], m1.shape[3]
    if res is None: res = PyArray_EMPTY(2, m1.shape, NPY_RTYPE, 0)
    
    res_s0 = res.strides[0]
    res_s1 = res.strides[1]
    m1_s0  =  m1.strides[0]
    m1_s1  =  m1.strides[1]
    m1_s2  =  m1.strides[2]
    m1_s3  =  m1.strides[3]
    m2_s0  =  m2.strides[0]
    m2_s1  =  m2.strides[1]
    
    m1i  = m1.data
    resi = res.data

    if accumulate:
        for i in range(ni):
            m1ij  =  m1i
            resij = resi
            for j in range(nj):
                (<rtype_t*>resij)[0] += \
                    c_m2ldot(m1ij, m2.data,
                             nk,   nl,
                             m1_s2, m1_s3,
                             m2_s0, m2_s1)
                m1ij  +=  m1_s1
                resij += res_s1
            m1i  +=  m1_s0
            resi += res_s0
    else:
        for i in range(ni):
            m1ij  =  m1i
            resij = resi
            for j in range(nj):
                (<rtype_t*>resij)[0] = \
                    c_m2ldot(m1ij, m2.data,
                             nk,   nl,
                             m1_s2, m1_s3,
                             m2_s0, m2_s1)
                m1ij  +=  m1_s1
                resij += res_s1
            m1i  +=  m1_s0
            resi += res_s0
    
    return res

cdef np.ndarray c_m6dotm3(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate):
    cdef int  i,  j,  k
    cdef int ni, nj, nk, nl, nm, nn
    
    cdef char *resi, *resij, *resijk
    cdef char *m1i, *m1ij, *m1ijk

    cdef int res_s0, res_s1, res_s2
    cdef int m1_s0, m1_s1, m1_s2, m1_s3, m1_s4, m1_s5
    cdef int m2_s0, m2_s1, m2_s2

    ni, nj, nk, nl, nm, nn = m1.shape[0], m1.shape[1], m1.shape[2], \
                             m1.shape[3], m1.shape[4], m1.shape[5]
    if res is None: res = PyArray_EMPTY(3, m1.shape, NPY_RTYPE, 0)
    
    res_s0 = res.strides[0]
    res_s1 = res.strides[1]
    res_s2 = res.strides[2]
    m1_s0  =  m1.strides[0]
    m1_s1  =  m1.strides[1]
    m1_s2  =  m1.strides[2]
    m1_s3  =  m1.strides[3]
    m1_s4  =  m1.strides[4]
    m1_s5  =  m1.strides[5]
    m2_s0  =  m2.strides[0]
    m2_s1  =  m2.strides[1]
    m2_s2  =  m2.strides[2]

    m1i  = m1.data
    resi = res.data

    if accumulate:
        for i in range(ni):
            m1ij  =  m1i
            resij = resi
            for j in range(nj):
                m1ijk  =  m1ij
                resijk = resij
                for k in range(nk):
                    (<rtype_t*>resijk)[0] += \
                        c_m3ldot(m1ijk, m2.data,
                                 nl, nm, nn,
                                 m1_s3, m1_s4, m1_s5,
                                 m2_s0, m2_s1, m2_s2)
                    m1ijk  +=  m1_s2
                    resijk += res_s2
                m1ij  +=  m1_s1
                resij += res_s1
            m1i  +=  m1_s0
            resi += res_s0
    else:
        for i in range(ni):
            m1ij  =  m1i
            resij = resi
            for j in range(nj):
                m1ijk  =  m1ij
                resijk = resij
                for k in range(nk):
                    (<rtype_t*>resijk)[0] = \
                        c_m3ldot(m1ijk, m2.data,
                                 nl, nm, nn,
                                 m1_s3, m1_s4, m1_s5,
                                 m2_s0, m2_s1, m2_s2)
                    m1ijk  +=  m1_s2
                    resijk += res_s2
                m1ij  +=  m1_s1
                resij += res_s1
            m1i  +=  m1_s0
            resi += res_s0
        
    return res


def m2dotm1(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res = None, bint accumulate = False):
    ''' m2dotm1(m1, m2[, res[, accumulate]]):
             res_i = m1_ij * m2_j
    '''
    assert (m1.ndim == 2 and m2.ndim == 1),    "wrong dimensions"
    assert (m1.shape[1] == m2.shape[0]),       "shapes don't match"
    if res is not None:
        assert (PyArray_ISWRITEABLE(res)),     "result isn't writeable"
        assert (res.ndim == 1 and
                res.shape[0] == m1.shape[0]),  "shapes don't match"
        assert (PyArray_TYPE(res) == NPY_RTYPE), "wrong result data type"
    assert (PyArray_TYPE(m1) == NPY_RTYPE and \
            PyArray_TYPE(m2) == NPY_RTYPE),      "wrong input data type" 
    return c_m2dotm1(m1, m2, res, accumulate)


def m4dotm2(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res = None, bint accumulate = False):
    ''' m4dotm2(m1, m2[, res[, accumulate]]):
             res_ij = sum_kl (m1_ijkl * m2_kl)
    '''
    assert (m1.ndim == 4 and m2.ndim == 2),    "wrong dimensions"
    assert (m1.shape[2] == m2.shape[0] and
            m1.shape[3] == m2.shape[1]),       "shapes don't match"
    if res is not None:
        assert (PyArray_ISWRITEABLE(res)),     "result isn't writeable"
        assert (res.ndim == 2 and
                res.shape[0] == m1.shape[0] and
                res.shape[1] == m1.shape[1]),  "shapes don't match"
        assert (PyArray_TYPE(res) == NPY_RTYPE), "wrong result data type"
    assert (PyArray_TYPE(m1) == NPY_RTYPE and \
            PyArray_TYPE(m2) == NPY_RTYPE),      "wrong input data type" 
    return c_m4dotm2(m1, m2, res, accumulate)


def m6dotm3(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res = None, bint accumulate = False):
    ''' m6dotm3(m1, m2[, res[, accumulate]]):
             res_ijk = sum_lmn (m1_ijklmn * m2_lmn)
    '''
    assert (m1.ndim == 6 and m2.ndim == 3),   "wrong dimensions"
    assert (m1.shape[3] == m2.shape[0] and
            m1.shape[4] == m2.shape[1] and
            m1.shape[5] == m2.shape[2]),      "shapes don't match"
    if res is not None:
        assert (PyArray_ISWRITEABLE(res)),    "result isn't writeable"
        assert (res.ndim == 3 and
                res.shape[0] == m1.shape[0] and
                res.shape[1] == m1.shape[1] and
                res.shape[2] == m1.shape[2]), "shapes don't match"
        assert (PyArray_TYPE(res) == NPY_RTYPE), "wrong result data type"
    assert (PyArray_TYPE(m1) == NPY_RTYPE and \
            PyArray_TYPE(m2) == NPY_RTYPE),      "wrong input data type" 
    return c_m6dotm3(m1, m2, res, accumulate)


def m2kdotmk(np.ndarray m1 not None, np.ndarray m2 not None,
             np.ndarray res = None, bint accumulate = False):
    ''' m2kdotmk(m1, m2[, res[, accumulate]]):
             res_{i} = sum_{j} (m1_{i,j} * m2_{j})
        where {i} ranges over all indices of res
        and   {j} ranges over all indices of m2
    '''
    cdef int k = m2.ndim
    if k == 1: return m2dotm1(m1, m2, res, accumulate)
    if k == 2: return m4dotm2(m1, m2, res, accumulate)
    if k == 3: return m6dotm3(m1, m2, res, accumulate)
    
    assert (m1.ndim == k*2), "wrong dimensions"
    if res is None: res = np.zeros(np.shape(m1)[:k], rtype)
    res_shape = np.shape(res)
    assert (res_shape == np.shape(m1)[:k] \
            and np.shape(m2) == np.shape(m1)[k:]), "shapes don't match"
    assert (PyArray_ISWRITEABLE(res)), "result isn't writeable"
    
    if accumulate:
        for i in np.ndindex(res_shape):
            res[i] += ldot(m1[i], m2)
    else:
        for i in np.ndindex(res_shape):
            res[i]  = ldot(m1[i], m2)
    return res


cdef np.ndarray c_m1extm1(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate):
    cdef char *m1_p0, *m2_p0, *res_p0, *res_p1
    cdef int   m1_s0,  m2_s0,  res_s0,  res_s1
    cdef int  i,  j
    cdef int ni, nj

    ni, nj = m1.shape[0], m2.shape[0]
    if res is None: res = np.empty((ni, nj), dtype=m1.descr)
    
    m1_s0 = m1.strides[0]
    m2_s0 = m2.strides[0]
    res_s0, res_s1 = res.strides[0], res.strides[1]

    m1_p0, res_p0 = m1.data, res.data

    if accumulate:
        for i in range(ni):
            m2_p0, res_p1 = m2.data, res_p0
            for j in range(nj):
                (<rtype_t*>res_p1)[0] += \
                    (<rtype_t*>m1_p0)[0] * (<rtype_t*>m2_p0)[0]
                m2_p0  += m2_s0
                res_p1 += res_s1
            m1_p0  += m1_s0
            res_p0 += res_s0
    else:
        for i in range(ni):
            m2_p0, res_p1 = m2.data, res_p0
            for j in range(nj):
                (<rtype_t*>res_p1)[0]  = \
                    (<rtype_t*>m1_p0)[0] * (<rtype_t*>m2_p0)[0]
                m2_p0  += m2_s0
                res_p1 += res_s1
            m1_p0  += m1_s0
            res_p0 += res_s0
    return res
    

cdef np.ndarray c_m2extm2(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate):
    cdef char *m1_p0, *m1_p1, *m2_p0, *m2_p1
    cdef int   m1_s0,  m1_s1,  m2_s0,  m2_s1
    cdef char *res_p0, *res_p1, *res_p2, *res_p3
    cdef int   res_s0,  res_s1,  res_s2,  res_s3
    cdef int  i,  j, k, l
    cdef int ni, nj, nk, nl

    ni, nj, nk, nl = m1.shape[0], m1.shape[1], m2.shape[0], m2.shape[1]
    if res is None: res = np.empty((ni, nj, nk, nl), dtype=m1.descr)
    
    m1_s0, m1_s1 = m1.strides[0], m1.strides[1]
    m2_s0, m2_s1 = m2.strides[0], m2.strides[1]
    res_s0, res_s1, res_s2, res_s3 = \
        res.strides[0], res.strides[1], res.strides[2], res.strides[3]
    
    m1_p0, res_p0 = m1.data, res.data
    
    if accumulate:
        for i in range(ni):
            m1_p1, res_p1 = m1_p0, res_p0
            for j in range(nj):
                m2_p0, res_p2 = m2.data, res_p1
                for k in range(nk):
                    m2_p1, res_p3 = m2_p0, res_p2
                    for l in range(nl):
                        (<rtype_t*>res_p3)[0] += \
                            (<rtype_t*>m1_p1)[0] * (<rtype_t*>m2_p1)[0]
                        m2_p1  += m2_s1
                        res_p3 += res_s3
                    m2_p0  += m2_s0
                    res_p2 += res_s2
                m1_p1  += m1_s1
                res_p1 += res_s1
            m1_p0  += m1_s0
            res_p0 += res_s0
    else:
       for i in range(ni):
            m1_p1, res_p1 = m1_p0, res_p0
            for j in range(nj):
                m2_p0, res_p2 = m2.data, res_p1
                for k in range(nk):
                    m2_p1, res_p3 = m2_p0, res_p2
                    for l in range(nl):
                        (<rtype_t*>res_p3)[0]  = \
                            (<rtype_t*>m1_p1)[0] * (<rtype_t*>m2_p1)[0]
                        m2_p1  += m2_s1
                        res_p3 += res_s3
                    m2_p0  += m2_s0
                    res_p2 += res_s2
                m1_p1  += m1_s1
                res_p1 += res_s1
            m1_p0  += m1_s0
            res_p0 += res_s0
    return res


cdef np.ndarray c_m3extm3(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate):
    cdef char *m1_p0, *m1_p1, *m1_p2, *m2_p0, *m2_p1, *m2_p2
    cdef int   m1_s0,  m1_s1,  m1_s2,  m2_s0,  m2_s1,  m2_s2
    cdef char *res_p0, *res_p1, *res_p2, *res_p3, *res_p4, *res_p5
    cdef int   res_s0,  res_s1,  res_s2,  res_s3,  res_s4,  res_s5
    cdef int  i,  j, k, l, m, n
    cdef int ni, nj, nk, nl, nm, nn

    ni, nj, nk, nl, nm, nn = m1.shape[0], m1.shape[1], m1.shape[2], \
                             m2.shape[0], m2.shape[1], m2.shape[2]
    if res is None: res = np.empty((ni, nj, nk, nl, nm, nn), dtype=m1.descr)
    
    m1_s0, m1_s1, m1_s2 = m1.strides[0], m1.strides[1], m1.strides[2]
    m2_s0, m2_s1, m2_s2 = m2.strides[0], m2.strides[1], m2.strides[2]
    res_s0, res_s1, res_s2, res_s3, res_s4, res_s5 = \
        res.strides[0], res.strides[1], res.strides[2], \
        res.strides[3], res.strides[4], res.strides[5]

    m1_p0, res_p0 = m1.data, res.data
    
    if accumulate:
        for i in range(ni):
            m1_p1, res_p1 = m1_p0, res_p0
            for j in range(nj):
                m1_p2, res_p2 = m1_p1, res_p1
                for k in range(nk):
                    m2_p0, res_p3 = m2.data, res_p2
                    for l in range(nl):
                        m2_p1, res_p4 = m2_p0, res_p3
                        for m in range(nm):
                            m2_p2, res_p5 = m2_p1, res_p4
                            for n in range(nn):
                                (<rtype_t*>res_p5)[0] += \
                                    (<rtype_t*>m1_p2)[0] * (<rtype_t*>m2_p2)[0]
                                m2_p2  += m2_s2
                                res_p5 += res_s5
                            m2_p1  += m2_s1
                            res_p4 += res_s4
                        m2_p0  += m2_s0
                        res_p3 += res_s3
                    m1_p2  += m1_s2
                    res_p2 += res_s2
                m1_p1  += m1_s1
                res_p1 += res_s1
            m1_p0  += m1_s0
            res_p0 += res_s0
    else:
        for i in range(ni):
            m1_p1, res_p1 = m1_p0, res_p0
            for j in range(nj):
                m1_p2, res_p2 = m1_p1, res_p1
                for k in range(nk):
                    m2_p0, res_p3 = m2.data, res_p2
                    for l in range(nl):
                        m2_p1, res_p4 = m2_p0, res_p3
                        for m in range(nm):
                            m2_p2, res_p5 = m2_p1, res_p4
                            for n in range(nn):
                                (<rtype_t*>res_p5)[0]  = \
                                    (<rtype_t*>m1_p2)[0] * (<rtype_t*>m2_p2)[0]
                                m2_p2  += m2_s2
                                res_p5 += res_s5
                            m2_p1  += m2_s1
                            res_p4 += res_s4
                        m2_p0  += m2_s0
                        res_p3 += res_s3
                    m1_p2  += m1_s2
                    res_p2 += res_s2
                m1_p1  += m1_s1
                res_p1 += res_s1
            m1_p0  += m1_s0
            res_p0 += res_s0
    return res



def m1extm1(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res = None, bint accumulate = False):
    ''' m1extm1(m1, m2[, res[, accumulate]]):
             res_ij = m1_i * m2_j
    '''
    assert (m1.ndim == 1 and m2.ndim == 1), "wrong dimensions"
    if res is not None:
        assert (PyArray_ISWRITEABLE(res)),  "result isn't writeable"
        assert (res.ndim == 2 and
                res.shape[0] == m1.shape[0] and
                res.shape[1] == m2.shape[0]), "shapes don't match"
        assert (PyArray_TYPE(res) == NPY_RTYPE), "wrong result data type"
    assert (PyArray_TYPE(m1) == NPY_RTYPE and \
            PyArray_TYPE(m2) == NPY_RTYPE),      "wrong input data type" 
    return c_m1extm1(m1, m2, res, accumulate)

def m2extm2(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res = None, bint accumulate = False):
    ''' m2extm2(m1, m2[, res[, accumulate]]):
             res_ijkl = m1_ij * m2_kl
    '''
    assert (m1.ndim == 2 and m2.ndim == 2), "wrong dimensions"
    if res is not None:
        assert (PyArray_ISWRITEABLE(res)),  "result isn't writeable"
        assert (res.ndim == 4 and
                res.shape[0] == m1.shape[0] and
                res.shape[1] == m1.shape[1] and
                res.shape[2] == m2.shape[0] and
                res.shape[3] == m2.shape[1]), "shapes don't match"
        assert (PyArray_TYPE(res) == NPY_RTYPE), "wrong result data type"
    assert (PyArray_TYPE(m1) == NPY_RTYPE and \
            PyArray_TYPE(m2) == NPY_RTYPE),      "wrong input data type" 
    return c_m2extm2(m1, m2, res, accumulate)

def m3extm3(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res = None, bint accumulate = False):
    ''' m3extm3(m1, m2[, res[, accumulate]]):
             res_ijklmn = m1_ijk * m2_lmn
    '''
    assert (m1.ndim == 3 and m2.ndim == 3), "wrong dimensions"
    if res is not None:
        assert (PyArray_ISWRITEABLE(res)),  "result isn't writeable"
        assert (res.ndim == 6 and
                res.shape[0] == m1.shape[0] and
                res.shape[1] == m1.shape[1] and
                res.shape[2] == m1.shape[2] and
                res.shape[3] == m2.shape[0] and
                res.shape[4] == m2.shape[1] and
                res.shape[5] == m2.shape[2]), "shapes don't match"
        assert (PyArray_TYPE(res) == NPY_RTYPE), "wrong result data type"
    assert (PyArray_TYPE(m1) == NPY_RTYPE and \
            PyArray_TYPE(m2) == NPY_RTYPE),      "wrong input data type" 
    return c_m3extm3(m1, m2, res, accumulate)

def mkextmk(np.ndarray m1 not None, np.ndarray m2 not None,
            np.ndarray res=None, bint accumulate=False):
    ''' mkextmk(m1, m2[, res[, accumulate]]):
             res_{i,j} = m1_{i} * m2_{j}
        where {i} ranges over all indices of m1
        and   {j} ranges over all indices of m2
    '''
    k = m1.ndim
    if k == 1: return m1extm1(m1, m2, res, accumulate)
    if k == 2: return m2extm2(m1, m2, res, accumulate)
    if k == 3: return m3extm3(m1, m2, res, accumulate)
    
    assert (k == m2.ndim), "wrong dimensions"
    if res is None: res = np.zeros(np.shape(m1) + np.shape(m2), m1.dtype)
    res_shape = np.shape(res)
    assert (res_shape[:k] == np.shape(m1) and
            res_shape[k:] == np.shape(m2)), "shapes don't match"
    if accumulate:
        for i in np.ndindex(res_shape):
            res[i] += m1[i[:k]] * m2[i[k:]]
    else:
        for i in np.ndindex(res_shape):
            res[i]  = m1[i[:k]] * m2[i[k:]]
    
    return res

def m2dotrows(np.ndarray m1 not None, np.ndarray m2 not None,
              np.ndarray res=None, bint accumulate=False):
    ''' m2dotrows(m1, m2[, res[, accumulate]]):
             res[i] = m1[i,:] . m2[i,:]
    '''
    cdef int i, m, n
    cdef int m1s0, m1s1, m2s0, m2s1
    cdef char *pm1, *pm2
    
    assert (m1.ndim == 2 and m2.ndim == 2), "wrong dimensions"
    
    m, n = m1.shape[0], m1.shape[1]
    assert (m == m2.shape[0] and n == m2.shape[1]), "shapes don't match"
    if res is None:
        if not PyArray_ISCARRAY_RO(m1) or PyArray_TYPE(m1) != NPY_RTYPE or \
           not PyArray_ISCARRAY_RO(m2) or PyArray_TYPE(m2) != NPY_RTYPE:
            return (m1 * m2).sum(1)
        res = PyArray_ZEROS(1, m1.shape, NPY_RTYPE, 0)
    else:
        assert (res.shape[0] == m), "shapes don't match"
        if not PyArray_ISCARRAY_RO(m1) or PyArray_TYPE(m1) != NPY_RTYPE or \
           not PyArray_ISCARRAY_RO(m2) or PyArray_TYPE(m2) != NPY_RTYPE or \
           not PyArray_ISCARRAY(res)  or PyArray_TYPE(res) != NPY_RTYPE:
            if accumulate: res[:] = (m1 * m2).sum(1)
            else:          res   += (m1 * m2).sum(1)
            return res
    
    pm1, pm2 = m1.data, m2.data
    m1s0, m1s1 = m1.strides[0], m1.strides[1]
    m2s0, m2s1 = m2.strides[0], m2.strides[1]
    if accumulate:
        for i in range(m):
            res[i] += c_m1ldot(pm1, pm2, n, m1s1, m2s1)
            pm1 += m1s0
            pm2 += m2s0
    else:
        for i in range(m):
            res[i]  = c_m1ldot(pm1, pm2, n, m1s1, m2s1)
            pm1 += m1s0
            pm2 += m2s0
    
    return res

def normrows(np.ndarray m not None):
    cdef int col, row, stride, rowsize
    cdef char *p
    cdef rtype_t x, v
    if not PyArray_ISCARRAY(m) or PyArray_TYPE(m) != NPY_RTYPE:
        for r in m: r /= sqrt(sqmag(r))
    else:
        rowsize = PyArray_SIZE(m) / m.shape[0]
        stride  = m.strides[0]
        p = m.data
        for row in range(m.shape[0]):
            v = 0.
            for col in range(rowsize):
                x = (<rtype_t*>p)[col]
                v += x * x
            v = sqrt(v)
            for col in range(rowsize):
                (<rtype_t*>p)[col] /= v
            p += stride


def mdotc(np.ndarray m not None, rtype_t c,
          np.ndarray res = None, bint accumulate=False):
    cdef long i, size
    cdef rtype_t *pm, *pr
    
    pm = <rtype_t*>m.data
    size = PyArray_SIZE(m)

    if res is None:
        if not PyArray_ISCARRAY_RO(m) or PyArray_TYPE(m) != NPY_RTYPE:
            return m*c
        res = PyArray_EMPTY(m.ndim, m.shape, NPY_RTYPE, 0)
        pr = <rtype_t*>res.data
        for i in range(size): pr[i] = pm[i] * c
        return res
    
    assert (PyArray_SAMESHAPE(m, res)), "shapes don't match"
    if not PyArray_ISCARRAY_RO(m) or PyArray_TYPE(m)   != NPY_RTYPE or \
       not PyArray_ISCARRAY(res)  or PyArray_TYPE(res) != NPY_RTYPE:
        if accumulate: res   += m*c
        else:          res[:] = m*c
        return res
    
    pr = <rtype_t*>res.data

    if accumulate:
        for i in range(size):
            pr[i] += pm[i] * c
    else:
        for i in range(size):
            pr[i]  = pm[i] * c
    
    return res

