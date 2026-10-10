"""Create a portable, verified data cache; raw archives remain on the preparing host."""
import argparse
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtsrb.data import digest, verify

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", type=Path, default=Path("data"))
p.add_argument("--output", type=Path, default=Path("dist/gtsrb-prepared.tar.gz"))
args = p.parse_args()
if (args.data/"test_access.json").exists():
    p.error("Do not package a test-exposed dataset as a fresh experiment")
meta = verify(args.data)
args.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.output, "w:gz") as tar:
    for name in ["manifest.json", "audit.json", *meta["cache_sha256"]]:
        tar.add(args.data/name, arcname="data/"+name)
checksum = digest(args.output)
args.output.with_suffix(args.output.suffix+".sha256").write_text(checksum+"  "+args.output.name+"\n")
print(f"{args.output}: {args.output.stat().st_size/1e6:.1f} MB; sha256 {checksum}")
