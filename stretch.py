import numpy as np

def stretch(x, n):
    y = np.fft.fft(x)
    z = np.zeros(n, dtype = np.complex)
    lastNonzeroAbs = len(x) // 2
    for i in xrange(-lastNonzeroAbs // n * n, lastNonzeroAbs + 1, n):
        addFourierSlice(y, z, i)
    z *= n / float(len(x))
    return np.fft.ifft(z)

def addFourierSlice(src, dst, start):
    firstUnsafeAbs = (len(src) + 1) // 2
    safeStart = max(start, 1 - firstUnsafeAbs)
    safeStop = min(start + len(dst), firstUnsafeAbs)
    
    negStart = min(safeStart, 0)
    negStop = min(safeStop, 0)
    dst[negStart - start:negStop - start] += \
        src[negStart + len(src):negStop + len(src)]

    posStart = max(safeStart, 0)
    posStop = max(safeStop, 0)
    dst[posStart - start:posStop - start] += src[posStart:posStop]

    if len(src) % 2 == 0:
        nyquist = 0.5 * src[firstUnsafeAbs]
        
        negNyquist = -firstUnsafeAbs - start
        if negNyquist >= 0 and negNyquist < len(dst):
            dst[negNyquist] += nyquist
        
        posNyquist = firstUnsafeAbs - start
        if posNyquist >= 0 and posNyquist < len(dst):
            dst[posNyquist] += nyquist
