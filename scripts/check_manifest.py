"""Verify packaged source bytes against SOURCE_HASHES.sha256 (excluding itself)."""
from hashlib import sha256
from pathlib import Path


def verify(root: Path) -> int:
    manifest = root / 'SOURCE_HASHES.sha256'
    count = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split('  ', 1)
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(rel)
        if sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f'changed packaged file: {rel}')
        count += 1
    return count


if __name__ == '__main__':
    print(f'SOURCE_MANIFEST_PASS files={verify(Path(__file__).resolve().parents[1])}')
