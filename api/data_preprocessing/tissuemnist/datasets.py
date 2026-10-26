import os
import numpy as np
import torch.utils.data as data
from PIL import Image

class TissueMNIST_truncated(data.Dataset):
    def __init__(self, root, cache_data_set=None, dataidxs=None, train=True, transform=None, target_transform=None, download=True):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Make sure parent directory exists
        os.makedirs(self.root, exist_ok=True)
        npz_file = os.path.join(self.root, "tissuemnist.npz")

        if not os.path.exists(npz_file) and download:
            print(f"[TissueMNIST] tissuemnist.npz not found at {self.root}. Attempting auto-download...")
            try:
                import medmnist
                from medmnist import TissueMNIST
                # Download using official API
                _ = TissueMNIST(split="train", download=True, root=self.root)
                # locate tissuemnist.npz and copy if downloaded elsewhere
                home_dir = os.path.expanduser("~")
                possible_paths = [
                    os.path.join(home_dir, ".medmnist", "tissuemnist.npz"),
                    os.path.join(self.root, ".medmnist", "tissuemnist.npz"),
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        import shutil
                        shutil.copy(p, npz_file)
                        print(f"[TissueMNIST] Copied dataset npz from {p} to {npz_file}")
                        break
            except ImportError:
                import urllib.request
                import sys

                def download_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100.0, downloaded * 100.0 / total_size)
                        sys.stdout.write(
                            f"\r[TissueMNIST] Downloading... {percent:.1f}% "
                            f"({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)"
                        )
                    else:
                        sys.stdout.write(f"\r[TissueMNIST] Downloading... {downloaded / (1024*1024):.2f} MB")
                    sys.stdout.flush()

                # Direct Zenodo download link for tissuemnist.npz
                url = "https://zenodo.org/records/6496656/files/tissuemnist.npz?download=1"
                print(f"[TissueMNIST] Downloading tissuemnist.npz from Zenodo: {url}...")
                tmp_file = npz_file + ".tmp"
                try:
                    urllib.request.urlretrieve(url, tmp_file, download_progress)
                    print("\n[TissueMNIST] Download complete!")
                    if os.path.exists(npz_file):
                        os.remove(npz_file)
                    os.rename(tmp_file, npz_file)
                except Exception as e:
                    print(f"\n[TissueMNIST] Failed to download from primary link: {e}")
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                    fallback_url = "https://zenodo.org/records/10519652/files/tissuemnist.npz?download=1"
                    print(f"[TissueMNIST] Trying fallback Zenodo link: {fallback_url}...")
                    try:
                        urllib.request.urlretrieve(fallback_url, tmp_file, download_progress)
                        print("\n[TissueMNIST] Download complete!")
                        if os.path.exists(npz_file):
                            os.remove(npz_file)
                        os.rename(tmp_file, npz_file)
                    except Exception as e2:
                        print(f"\n[TissueMNIST] Fallback download failed: {e2}")
                        if os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        raise e2

        if not os.path.exists(npz_file):
            raise FileNotFoundError(
                f"[TissueMNIST] tissuemnist.npz not found at {npz_file}. "
                f"Please ensure it is downloaded or install medmnist with `pip install medmnist`."
            )

        # Load the npz file
        data_dict = np.load(npz_file)
        if self.train:
            self.data = data_dict['train_images']
            self.target = data_dict['train_labels'].squeeze().astype(np.int64)
        else:
            self.data = data_dict['test_images']
            self.target = data_dict['test_labels'].squeeze().astype(np.int64)

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        # TissueMNIST is grayscale, shape is (28, 28)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
