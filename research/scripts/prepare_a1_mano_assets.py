"""Deploy an already licensed MANO archive; never requests or stores credentials."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from a1_mano import asset_root, MODEL_SHA256

ARCHIVE_SHA = "50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f"
REVISION = "933be97d6fa05c729656669640084251ce644b0a"
PUBLIC_FILES = {
    "manotorch/manolayer.py": "0a447b1cd4ddf889993b7e9d892cd003d752354e52d37a58295ca0616d941615",
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}


def write_verified(path, data, sha, private=False):
    assert hashlib.sha256(data).hexdigest() == sha, str(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if path.exists():
        assert path.read_bytes() == data, f"Existing different file retained: {path}"
    else:
        path.write_bytes(data)
    if private:
        path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    private = asset_root()
    archive = args.archive or private / "mano_v1_2.zip"
    with archive.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == ARCHIVE_SHA
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for side, sha in MODEL_SHA256.items():
            name = f"models/MANO_{side.upper()}.pkl"
            write_verified(private / name, z.read("mano_v1_2/" + name), sha, private=True)
        write_verified(private / "LICENSE.txt", z.read("mano_v1_2/LICENSE.txt"),
            "75fc4da03259504ce9a4ed2418ab98dd59647add7ff95e7691c4d55fb0e3cf63", private=True)
    root = Path(__file__).resolve().parents[1]
    reference = root / "references/manotorch_decoder"
    for name, sha in PUBLIC_FILES.items():
        path = reference / name
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/lixiny/manotorch/{REVISION}/{name}", timeout=60) as src:
                data = src.read()
        write_verified(path, data, sha)
    print(json.dumps({"asset_root": str(private), "archive_sha256": ARCHIVE_SHA,
                      "models": MODEL_SHA256, "reference_revision": REVISION}, indent=2))


if __name__ == "__main__":
    main()
