import numpy as np
import librosa
from pathlib import Path

sr = 20000
n_mels = 64
n_fft = 1024
hop_length = 512

data_dir = Path(r"C:\projects\motor_fedlearning\data")  # например, r"D:\bearing_dataset"

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

# 1st_test: первые файлы = здоровые, последние = поломка
test_folders = ["1st_test", "2nd_test", "3rd_test"]

for folder in test_folders:
    folder_path = data_dir / folder
    files = sorted(folder_path.iterdir())
    print(f"\n{folder}: {len(files)} файлов")
    
    # Первые 50% файлов → здоровые (class 0)
    num_healthy = len(files) // 2
    for i, f in enumerate(files[:num_healthy]):
        try:
            X.append(file_to_mel(f))
            y_labels.append(0)
            print(f"  {i+1}: {f.name} → Healthy")
        except:
            print(f"  Ошибка при загрузке {f.name}")
    
    # Вторые 50% файлов → поломка (class 1)
    for i, f in enumerate(files[num_healthy:]):
        try:
            X.append(file_to_mel(f))
            y_labels.append(1)
            print(f"  {num_healthy+i+1}: {f.name} → Faulty")
        except:
            print(f"  Ошибка при загрузке {f.name}")

X = np.stack(X, axis=0)
y_labels = np.array(y_labels)

print(f"\n✓ Загружено: X shape={X.shape}, y shape={y_labels.shape}")
print(f"  Класс 0 (здоровые): {(y_labels == 0).sum()}")
print(f"  Класс 1 (больные): {(y_labels == 1).sum()}")

np.save("X_mel_all.npy", X)
np.save("y_labels_all.npy", y_labels)
print("✓ Сохранено: X_mel_all.npy, y_labels_all.npy")
