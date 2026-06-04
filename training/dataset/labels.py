import re
from dataclasses import dataclass


@dataclass
class ViolationLabel:
    video_id: str
    tracking_id: int
    roi: str
    start_frame: int
    annotation: int  # 0=violation, 1=compliance


def parse_train_label(label_str: str):
    m = re.match(r'V(\d+)I(\d+)S(\d)D\d+R\d+A(\d)', label_str)
    if not m:
        raise ValueError(f"Cannot parse label string: {label_str!r}")
    return (
        f"video_{int(m.group(1)):03d}",
        int(m.group(2)),
        'BOT' if m.group(3) == '1' else 'TOP',
        int(m.group(4)),
    )
