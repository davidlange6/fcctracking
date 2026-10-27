"""Minimal DD4hep BitFieldCoder: decode fields of 64-bit cellIDs from the encoding string."""

import numpy as np


class BitFieldCoder:
    """Parses e.g. "system:5,side:-2,layer:6,module:11,sensor:8" or "name:offset:width"."""

    def __init__(self, encoding):
        self.fields = {}
        offset = 0
        for item in encoding.split(","):
            parts = item.strip().split(":")
            if len(parts) == 3:
                name, offset, width = parts[0], int(parts[1]), int(parts[2])
            else:
                name, width = parts[0], int(parts[1])
            signed = width < 0
            self.fields[name] = (offset, abs(width), signed)
            offset += abs(width)

    def get(self, cell_ids, name, default=0):
        if name not in self.fields:
            return np.full(len(cell_ids), default, dtype=np.int64)
        offset, width, signed = self.fields[name]
        v = (np.asarray(cell_ids, dtype=np.uint64) >> np.uint64(offset)) & np.uint64((1 << width) - 1)
        v = v.astype(np.int64)
        if signed:
            v = np.where(v >= 1 << (width - 1), v - (1 << width), v)
        return v
