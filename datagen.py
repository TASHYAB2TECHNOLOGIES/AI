import numpy as np
import os
import argparse

def generate_sine(fs, f0, duration, amp=1.0, phase=0.0):
    t = np.arange(0, duration, 1/fs)
    return t, amp * np.sin(2*np.pi*f0*t + phase)

def add_harmonics(signal, fs, freqs, amps):
    t = np.arange(len(signal)) / fs
    out = signal.copy()
    for f,a in zip(freqs, amps):
        out += a * np.sin(2*np.pi*f*t)
    return out

def generate_example(label, fs=5000, duration=1.0, mains_freq=50.0):
    t, v = generate_sine(fs, mains_freq, duration, amp=230.0)
    _, i = generate_sine(fs, mains_freq, duration, amp=1.0)

    if label == 'normal':
        v += np.random.normal(0, 1.0, size=v.shape)
        i += np.random.normal(0, 0.02, size=i.shape)

    elif label == 'short':
        idx = np.random.randint(int(0.2*fs), int(0.8*fs))
        span = int(0.01*fs)
        i[idx:idx+span] += 20.0 * np.hanning(span)
        v[idx:idx+span] *= 0.2
        v += np.random.normal(0, 2.0, size=v.shape)

    elif label == 'overload':
        i *= 2.5
        v = add_harmonics(v, fs, [150, 250], [20, 10])
        v += np.random.normal(0, 3.0, size=v.shape)

    elif label == 'open':
        i *= 0.0
        for _ in range(np.random.randint(1,4)):
            idx = np.random.randint(0, len(i)-int(0.02*fs))
            i[idx:idx+int(0.02*fs)] = 0.0
        v += np.random.normal(0, 1.5, size=v.shape)

    else:
        raise ValueError("Unknown label")

    return t, v, i

def generate_dataset(out_path='data/dataset.npz', classes=None, n_per_class=200, fs=5000, duration=0.5):
    if classes is None:
        classes = ['normal','short','overload','open']
    X = []
    y = []
    for label in classes:
        for _ in range(n_per_class):
            t,v,i = generate_example(label, fs=fs, duration=duration)
            X.append(np.stack([v,i], axis=0))
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, fs=fs)
    print("Saved dataset:", out_path)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dataset.npz")
    parser.add_argument("--per", type=int, default=300)
    args = parser.parse_args()
    generate_dataset(out_path=args.out, n_per_class=args.per, duration=0.5)
