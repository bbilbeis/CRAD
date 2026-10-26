import os
import numpy as np
import torch.utils.data as data
from PIL import Image

class PathMNIST_truncated(data.Dataset):
    def __init__(self, root, cache_data_set=None, dataidxs=None, train=True, transform=None, target_transform=None, download=True):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Make sure parent directory exists
        os.makedirs(self.root, exist_ok=True)
        npz_file = os.path.join(self.root, "pathmnist.npz")

        if not os.path.exists(npz_file) and download:
            print(f"[PathMNIST] pathmnist.npz not found at {self.root}. Attempting auto-download...")
            # Try importing medmnist first
            try:
                import medmnist
                from medmnist import PathMNIST
                # Download using official API
                _ = PathMNIST(split="train", download=True, root=self.root)
                # locate pathmnist.npz and copy if downloaded elsewhere
                home_dir = os.path.expanduser("~")
                possible_paths = [
                    os.path.join(home_dir, ".medmnist", "pathmnist.npz"),
                    os.path.join(self.root, ".medmnist", "pathmnist.npz"),
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        import shutil
                        shutil.copy(p, npz_file)
                        print(f"[PathMNIST] Copied dataset npz from {p} to {npz_file}")
                        break
            except ImportError:
                # If medmnist package is not installed, fallback to direct Zenodo download
                import urllib.request
                import sys

                def _download_file(download_url, dest_path):
                    import subprocess
                    tmp_file = dest_path + ".tmp"
                    # 1. Try curl (most reliable for Zenodo redirects)
                    try:
                        print(f"[PathMNIST] Fetching with curl from: {download_url}")
                        subprocess.run(['curl', '-L', '--progress-bar', '-o', tmp_file, download_url], check=True)
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        os.rename(tmp_file, dest_path)
                        return
                    except Exception as curl_err:
                        print(f"[PathMNIST] curl download failed or not available ({curl_err}), trying urllib...")

                    # 2. Fallback to urllib
                    req = urllib.request.Request(
                        download_url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_file, "wb") as f_out:
                        total_length = resp.headers.get('content-length')
                        if total_length:
                            total_length = int(total_length)
                            downloaded = 0
                            while True:
                                chunk = resp.read(1024 * 1024)
                                if not chunk:
                                    break
                                downloaded += len(chunk)
                                f_out.write(chunk)
                                percent = min(100.0, downloaded * 100.0 / total_length)
                                sys.stdout.write(f"\r[PathMNIST] Downloading... {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_length / (1024*1024):.2f} MB)")
                                sys.stdout.flush()
                            sys.stdout.write("\n")
                        else:
                            f_out.write(resp.read())
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(tmp_file, dest_path)

                # Official Zenodo standard download link for PathMNIST (record 10519652)
                url = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
                print(f"[PathMNIST] Downloading pathmnist.npz from Zenodo: {url}...")
                try:
                    _download_file(url, npz_file)
                    print("[PathMNIST] Download complete!")
                except Exception as e:
                    print(f"[PathMNIST] Failed to download from primary link: {e}")
                    # Try fallback Zenodo URL if primary fails
                    fallback_url = "https://zenodo.org/records/6496656/files/pathmnist.npz?download=1"
                    print(f"[PathMNIST] Trying fallback Zenodo link: {fallback_url}...")
                    _download_file(fallback_url, npz_file)
                    print("[PathMNIST] Download complete!")

        if not os.path.exists(npz_file):
            raise FileNotFoundError(
                f"[PathMNIST] pathmnist.npz not found at {npz_file}. "
                f"Please ensure it is downloaded or install medmnist with `pip install medmnist`."
            )

        # Load the npz file
        data_dict = np.load(npz_file)
        if self.train:
            self.data = data_dict['train_images']
            # Labels are N x 1, squeeze to 1D array of class indices
            self.target = data_dict['train_labels'].squeeze().astype(np.int64)
        else:
            self.data = data_dict['test_images']
            self.target = data_dict['test_labels'].squeeze().astype(np.int64)

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.target = self.target[self.dataidxs]

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        # img is a numpy array of shape (28, 28, 3) and uint8
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
