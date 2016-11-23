import numpy as np
from stretch import stretch
from rfftmatrix import rfftr, rfftmatrix, rfftmultiply
from diskrand import diskrand

def combineIngredients(pine, fir, m):
    global piiineF, pF, h, a, b
    n = len(pine)
    assert n == len(fir)
    assert n <= m
    a = np.fft.ifft(np.fft.fft(pine) * np.fft.fft(fir), n = n)
    piiineF = np.fft.fft(stretch(pine, m))
    
    """
    assert np.all(np.abs(piiineF[n // 2 + 1:m - n // 2]) < 1e-13)
    pF = piiineF.copy()
    h = (m - n // 2 * 2) // 2
    x = diskrand(h) * 1000
    piiineF[n // 2 + 1:n // 2 + 1 + h] += x
    piiineF[m - n // 2 - h:m - n // 2] += x[::-1].conjugate()
    if n % 2 == 0:
        assert piiineF[n // 2].imag < 1e-13
        assert piiineF[m - n // 2].imag < 1e-13
        x = np.random.rand(1)[0] * 1000j
        piiineF[n // 2] += x
        piiineF[m - n // 2] -= x
    assert np.all(np.abs(piiineF.real) > 1e-13)
    if n % 2:
        if m % 2 == 0:
            assert np.all(np.abs(piiineF[1:m // 2].imag) > 1e-13)
            assert np.all(np.abs(piiineF[m // 2 + 1:].imag) > 1e-13)
        else:
            assert np.all(np.abs(piiineF[1:].imag) > 1e-13)
    """
    
    b = np.fft.ifft(piiineF * np.fft.fft(fir, n = m))
    if np.iscomplexobj(pine) or np.iscomplexobj(fir):
        return a, b
    else:
        #assert np.all(np.abs(a.imag) < 1e-11)
        #assert np.all(np.abs(b.imag) < 1e-11)
        return a.real.copy(), b.real.copy()

def arrayEqual(a, b, e = 1e-14):
    return np.all(np.abs(a - b) < e)

def rfftStretch(x, n):
    out = x[:n].copy()
    if 0 < n < len(x):
        out *= float(n) / len(x)
        if n % 2 == 0:
            out[-1] *= 2
    return out

def tileToLength(x, n, axis = -1):
    if axis < 0:
        axis += len(x.shape)
    m = x.shape[axis]
    outShape = list(x.shape)
    outShape[axis] = n
    out = np.empty(outShape, dtype = x.dtype)
    ps = (slice(None),) * axis
    
    for start in xrange(0, n, m):
        stop = min(start + m, n)
        out[ps + (slice(start, stop),)] = x[ps + (slice(stop - start),)]
    return out

def untile(x, n, axis = -1):
    if axis < 0:
        axis += len(x.shape)
    m = x.shape[axis]
    outShape = list(x.shape)
    outShape[axis] = n
    out = np.zeros(outShape, dtype = x.dtype)
    ps = (slice(None),) * axis
    
    for start in xrange(0, m, n):
        stop = min(start + n, m)
        out[ps + (slice(stop - start),)] += x[ps + (slice(start, stop),)]
    return out

def separate(a, b, r = None):
    global u, s, v, firCount, A, aF, bF
    n = len(a)
    m = len(b)
    assert n < m
    if r is None:
        r = n
    aF = rfftr(a)
    bF = rfftStretch(rfftr(b), n)
    
    A = rfftmatrix(m)[:n]
    rfftmultiply(aF, A)
    A = tileToLength(A, r)
    
    B = rfftmatrix(n)
    rfftmultiply(bF, B)
    B = tileToLength(B, r)
    A -= B

    u, s, v = np.linalg.svd(A)
    firCount = 1 + max(r - n, 0)
    firs = v[-firCount:].copy()
    
    pine = np.fft.rfft(untile(firs[-1], n))
    np.divide(np.fft.rfft(a), pine, out = pine)
    pine = np.fft.irfft(pine, n = n)
    
    return pine, firs

if __name__ == "__main__":
    for i in xrange(1000):
        n = np.random.randint(1, 10)
        m = np.random.randint(n + 1, 11)
        pine = np.random.rand(n)
        fir = np.random.rand(n)

        r = np.random.randint(1, n + 1)
        fir[r:] = 0
        
        a, b = combineIngredients(pine, fir, m)
        pineHat, firHat = separate(a, b, r)
        assert arrayEqual(firHat[0] / fir[:r], (firHat[0] / fir[:r])[0], e = 1e-5)
        assert arrayEqual(pineHat / pine, (pineHat / pine)[0], e = 1e-5)

    n = np.random.randint(1, 10)
    m = np.random.randint(n + 1, 11)
    pine = np.random.rand(n)
    fir = np.random.rand(n)
    a, b = combineIngredients(pine, fir, m)
    pineHat, firHats = separate(a, b, n + 20)
    
