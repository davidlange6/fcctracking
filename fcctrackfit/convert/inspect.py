"""List collections, types and sizes in the first event(s) of an edm4hep file,
plus the CellIDEncoding metadata, to fill in a detector YAML.

Usage: python -m fcctrackfit.convert.inspect FILE [N_EVENTS]
"""

import sys

from podio import root_io


def main():
    path = sys.argv[1]
    n_events = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    reader = root_io.Reader(path)
    print("categories:", reader.categories)
    for i, event in enumerate(reader.get("events")):
        if i >= n_events:
            break
        print(f"--- event {i}")
        for name in sorted(event.getAvailableCollections()):
            coll = event.get(name)
            print(f"  {name:45s} {str(coll.getValueTypeName()):40s} {len(coll)}")
    if "metadata" in reader.categories:
        meta = reader.get("metadata")[0]
        for key in sorted(meta.parameters):
            if key.endswith("CellIDEncoding"):
                print(f"  {key}: {meta.get_parameter(key)}")


if __name__ == "__main__":
    main()
