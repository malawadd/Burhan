"""Platform-independent source-pin lookup; no model/data/experiment mutation."""
from pathlib import PurePosixPath


def canonical_relative_path(value):
    normalized = str(value).replace('\\', '/')
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts or ':' in normalized:
        raise ValueError('Expected a workspace-relative protected source identity')
    return path.as_posix()


def source_pin(pins, relative_path):
    expected = canonical_relative_path(relative_path)
    matches = {digest for path, digest in pins.items() if canonical_relative_path(path) == expected}
    if len(matches) != 1:
        raise ValueError('Missing or conflicting source identity after separator normalization')
    digest = matches.pop()
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
        raise ValueError('Expected one exact SHA256 source identity')
    return digest
