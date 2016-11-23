import numpy as np

def rfftr(x):
    xF = np.fft.rfft(x)
    out = np.zeros(len(x))
    out[0] = xF[0].real
    out[1::2] = xF[1:].real
    out[2::2] = xF[1:(len(x) + 1) // 2].imag
    return out

def irfftr(x):
    xF = np.empty(len(x) // 2 + 1, dtype = np.complex)
    xF[0] = x[0]
    xF[1:] = x[1::2]
    xF[1:(len(x) + 1) // 2].imag = x[2::2]
    return np.fft.irfft(xF, n = len(x))

def rfftmatrix(n):
    out = np.empty((n, n))
    complexArea = getComplexArea(out)
    hRange = np.linspace(0, -2 * np.pi, num = n, endpoint = False)
    vRange = np.arange(1, (n + 1) // 2, dtype = np.float)[:, None]
    np.multiply(vRange, hRange, out = complexArea)
    np.cos(complexArea[0], out = complexArea[0])
    np.sin(complexArea[1], out = complexArea[1])
    out[0] = 1
    if n % 2 == 0:
        out[-1, ::2] = 1
        out[-1, 1::2] = -1
    return out

def rfftmultiply(a, b):
    aComplex = getComplexArea(a[:, None])
    bComplex = getComplexArea(b)
    tmp = aComplex[1] * bComplex[::-1]
    np.negative(tmp[0], out = tmp[0])
    bComplex *= aComplex[0]
    bComplex += tmp
    b[0] *= a[0]
    if len(a) % 2 == 0:
        b[-1] *= a[-1]

def getComplexArea(x):
    complexHeight = (len(x) + 1) // 2 - 1
    complexArea = x[1:1 + 2 * complexHeight]
    complexArea.shape = (complexHeight, 2) + x.shape[1:]
    complexArea = np.swapaxes(complexArea, 0, 1)
    return complexArea

def arrayEqual(a, b, e = 1e-14):
    return np.all(np.abs(a - b) < e)

if __name__ == "__main__":
    for i in xrange(100000):
        n = np.random.randint(1, 10)
        x = np.random.rand(n)
        assert arrayEqual(irfftr(rfftr(x)), x, e = 1e-13)
        assert arrayEqual(rfftmatrix(n).dot(x), rfftr(x), e = 1e-13)
        
    for i in xrange(10000):
        m = np.random.randint(10)
        n = np.random.randint(1, 10)
        a = np.random.rand(n)
        A = np.random.rand(n, m)
        aF = rfftr(a)
        AF = np.empty_like(A)
        for i in xrange(m):
            AF[:, i] = rfftr(A[:, i])
        rfftmultiply(aF, AF)
        B = np.empty_like(A)
        for i in xrange(m):
            B[:, i] = irfftr(AF[:, i])
        
        aFComplex = np.fft.rfft(a)
        BHat = np.empty_like(A)
        for i in xrange(m):
            BHat[:, i] = np.fft.irfft(np.fft.rfft(A[:, i]) * aFComplex, n = n)
        assert arrayEqual(B, BHat)
            
        
        
