"""Canonical dataset keys for saved training-only mining records."""


def hardnegative_keys(records):
    return {row['image'].replace('\\','/'):row['locations'] for row in records}


def require_matched_hardnegatives(image_keys, hardneg):
    if not any(hardneg.get(key.replace('\\','/')) for key in image_keys):
        raise ValueError('No training hard-negative paths matched the canonical image keys')
