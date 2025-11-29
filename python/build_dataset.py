import numpy as np
import librosa
from pathlib import Path

sr = 20000
n_mels = 64
n_fft = 1024
hop_length = 512

data_dir = Path(r"C:\projects\motor_fedlearning\data") / "1st_test"
files = sorted(data_dir.iterdir())

healthy_files = files[:3]      # условно "здоровые"
faulty_files = files[-3:]      # условно "больные"

def file_to_mel(path, channel=0):
    signal = np.loadtxt(path)
    y = signal[:, channel].astype(np.float32)
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S_mel, ref=np.max)
    return S_db

X = []
y_labels = []

for f in healthy_files:
    X.append(file_to_mel(f))
    y_labels.append(0)  # 0 = healthy

for f in faulty_files:
    X.append(file_to_mel(f))
    y_labels.append(1)  # 1 = faulty

X = np.stack(X, axis=0)
y_labels = np.array(y_labels)

print("X shape:", X.shape)
print("y shape:", y_labels.shape, "labels:", y_labels)
np.save("X_mel.npy", X)
np.save("y_labels.npy", y_labels)
