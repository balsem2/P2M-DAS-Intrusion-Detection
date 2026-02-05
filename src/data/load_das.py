import h5py

def load_das(path):
    with h5py.File(path, "r") as f:
        print("Keys:", list(f.keys()))
        data = f["/Acquisition/Raw[0]/RawData"][:]
        print("Shape:", data.shape)
        return data

if __name__ == "__main__":
    data = load_das("data/raw/example.h5")
