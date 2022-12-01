import numpy as np
import scipy.signal as ss
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

from NMF.models.nmf import EucNMF, KLNMF

plt.rcParams['figure.dpi'] = 200

fft_size, hop_size = 1024, 256

x, sr = librosa.load("data/darhla_c1_gr1_0001.wav", sr=44100)
# x = np.reshape(x, (-1, 1))
# print(sr)

print(sr, x.shape)

_, _, spectrogram = ss.stft(x, nperseg=fft_size, noverlap=fft_size - hop_size)
X = np.abs(spectrogram) ** 2

n_basis = 2

np.random.seed(111)
nmf = EucNMF(n_basis=n_basis)

basis, activation = nmf(X, iteration=1000)

for idx in range(n_basis):
    Z = basis[:, idx: idx + 1] * activation[idx: idx + 1, :]

    ratio = np.sqrt(Z / X)

    estimated_spectrogram = ratio * spectrogram
    _, estimated_signal = ss.istft(estimated_spectrogram, nperseg=fft_size, noverlap=fft_size - hop_size)
    estimated_signal = estimated_signal / np.abs(estimated_signal).max()
    sf.write('data/kl' + str(idx) + '.wav', estimated_signal, sr)

domain = 1
# for idx in range(n_basis):
#     estimated_spectrogram = (basis[:, idx: idx + 1] @ activation[idx: idx + 1, :])**(2 / domain)
#
#     estimated_power = np.abs(estimated_spectrogram)**2
#     estimated_power[estimated_power < 1e-12] = 1e-12
#     log_spectrogram = 10 * np.log10(estimated_power)
#
#     plt.figure(figsize=(10, 8))
#     plt.pcolormesh(log_spectrogram, cmap='jet')
#     plt.show()