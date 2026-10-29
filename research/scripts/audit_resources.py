"""Read public metadata and documentation only; no dataset or model downloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SuperDex-bootstrap/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(8_000_001)
    if len(body) > 8_000_000:
        raise ValueError("Metadata exceeded 8 MB budget")
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    sources = {
        "supermimic_official": "https://supermimic.github.io/",
        "dexmimic_root": "https://api.github.com/repos/ZhexiLuo/DexMimic/contents",
        "dexmimic_commit": "https://api.github.com/repos/ZhexiLuo/DexMimic/commits?per_page=1",
        "oakink_readme": "https://raw.githubusercontent.com/oakink/OakInk2/dev/README.md",
        "oakink_commit": "https://api.github.com/repos/oakink/OakInk2/commits?per_page=1",
        "oakink_data_root": "https://huggingface.co/api/datasets/kelvin34501/OakInk-v2/tree/main?limit=100",
        "hot3d_readme": "https://raw.githubusercontent.com/facebookresearch/hot3d/main/README.md",
        "hot3d_clips_readme": "https://raw.githubusercontent.com/facebookresearch/hot3d/main/hot3d/clips/README.md",
        "hot3d_commit": "https://api.github.com/repos/facebookresearch/hot3d/commits?per_page=1",
        "hot3d_data_root": "https://huggingface.co/api/datasets/bop-benchmark/hot3d/tree/main?limit=100",
    }
    for name, url in sources.items():
        record = {"name": name, "url": url}
        try:
            body = fetch(url)
            path = args.output_dir / (name + ".txt")
            path.write_bytes(body)
            record.update(status="OK", bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), file=path.name)
            if name.endswith("_commit"):
                record["commit"] = json.loads(body)[0]["sha"]
            elif name == "dexmimic_root":
                record["entries"] = [x["name"] for x in json.loads(body)]
            elif name.endswith("_data_root"):
                record["entries"] = [{k:x.get(k) for k in ("path", "size", "type")} for x in json.loads(body)]
            elif name == "supermimic_official":
                record["github_links"] = sorted(set(re.findall(r'https://github\.com/[^\s"<>]+', body.decode())))
        except Exception as exc:
            record.update(status="ERROR", error=repr(exc))
        results.append(record)
        print(json.dumps({k:v for k,v in record.items() if k != "entries"}), flush=True)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "public metadata only", "resources": results}
    (args.output_dir / "resources.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
