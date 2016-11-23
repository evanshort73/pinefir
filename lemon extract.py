import numpy as np
from scipy.io import wavfile
from separate import separate, combineIngredients

def normalize(x):
    out = x - np.mean(x)
    out /= np.max(np.abs(out))
    if len(out) % 2 == 0:
        outF = np.fft.rfft(out)
        outF[-1] = 0
        out = np.fft.irfft(outF)
    return out

rate, aWav = wavfile.read("a.wav")
rate, bWav = wavfile.read("b.wav")

n = len(aWav)
m = len(bWav)

a = aWav
b = bWav

pine, (fir,) = separate(a, b)
wavfile.write("pine.wav", rate, np.tile(normalize(pine), 20))
wavfile.write("fir.wav", rate, np.tile(normalize(fir), 20))

aHat, bHat = combineIngredients(pine, fir, m)
wavfile.write("aHat.wav", rate, np.concatenate((np.tile(normalize(a), 20), np.tile(normalize(aHat), 20))))
wavfile.write("bHat.wav", rate, np.concatenate((np.tile(normalize(b), 20), np.tile(normalize(bHat), 20))))

hats = [a, b]
for i in xrange(20):
    aHat, cHat = combineIngredients(pine, fir, m ** (i + 2) // n ** (i + 1))
    hats.append(cHat)
wavfile.write("hats.wav", rate, np.concatenate([np.tile(normalize(hat), 20) for hat in hats]))
