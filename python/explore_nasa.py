import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# путь к одному файлу IMS
data_dir = Path(r"C:\projects\motor_fedlearning\data") / "1st_test"
files = sorted(data_dir.iterdir())
first_file = files[0]

signal = np.loadtxt(first_file)
print("Форма сигнала:", signal.shape)

channel = 0
y = signal[:, channel].astype(np.float32)

sr = 20000  # частота дискретизации IMS

S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                       hop_length=512, n_mels=64)
S_db = librosa.power_to_db(S_mel, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-спектрограмма мотора (канал 0)')
plt.tight_layout()
plt.show()


