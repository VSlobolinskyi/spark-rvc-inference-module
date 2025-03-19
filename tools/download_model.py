import requests
import zipfile
import argparse
from pathlib import Path
import tempfile
import shutil


def download_file(url: str, local_path: Path):
    """Download a file from a URL to a local path."""
    if local_path.exists():
        print(f"[SKIP] {local_path} already exists.")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url} -> {local_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"[DONE] Downloaded {local_path}")


def extract_and_move(zip_path: Path, logs_dir: Path, weights_dir: Path):
    """Extract a zip file and move .index files to logs and .pth files to assets/weights."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        print(f"[EXTRACT] Extracting {zip_path} to temporary directory {tmpdir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        # Walk through all extracted files recursively.
        for extracted_file in tmpdir.rglob("*"):
            if extracted_file.is_file():
                if extracted_file.suffix == ".index":
                    dest = logs_dir / extracted_file.name
                    print(f"[MOVE] Moving {extracted_file} to {dest}")
                    shutil.move(str(extracted_file), str(dest))
                elif extracted_file.suffix == ".pth":
                    dest = weights_dir / extracted_file.name
                    print(f"[MOVE] Moving {extracted_file} to {dest}")
                    shutil.move(str(extracted_file), str(dest))
    print("[CLEANUP] Extraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Download a model zip file, extract it, and place .index files in ./logs and .pth files in ./assets/weights."
    )
    parser.add_argument("url", type=str, help="URL of the zip file to download.")
    args = parser.parse_args()

    url = args.url
    zip_filename = Path(url).name
    zip_path = Path(zip_filename)

    # Download the zip archive.
    download_file(url, zip_path)

    # Define destination folders.
    logs_dir = Path("./logs")
    weights_dir = Path("./assets/weights")

    # Extract the archive and move files accordingly.
    extract_and_move(zip_path, logs_dir, weights_dir)

    # Delete the zip file.
    if zip_path.exists():
        print(f"[DELETE] Removing {zip_path}")
        zip_path.unlink()
    print("[COMPLETE] Model download and extraction complete.")


if __name__ == "__main__":
    main()
