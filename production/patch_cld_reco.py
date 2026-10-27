"""Write a copy of CLDReconstruction.py with an extra --digiOnly option.

--digiOnly runs the tracker digitisation (Tracking/TrackingDigi) but skips track
finding, the KF refit and the track-truth linking. That is all a training sample
needs (hits, hit-to-sim links, MC truth); the KF fit costs ~98 % of reconstruction
time. Every edit is anchored on an exact string, so a CLDConfig change fails here
instead of silently running full tracking.

Usage: python patch_cld_reco.py IN.py OUT.py
"""

import sys

EDITS = [
    (
        'tracking_group.add_argument("--truthTracking", action="store_true", default=False, help="Cheat tracking pattern recognition")\n',
        'tracking_group.add_argument("--truthTracking", action="store_true", default=False, help="Cheat tracking pattern recognition")\n'
        'parser_group.add_argument("--digiOnly", action="store_true", default=False, help="Tracker digitisation only: no track finding or fitting")\n',
    ),
    ("# tracking\nif reco_args.truthTracking:\n", "# tracking\nif reco_args.digiOnly:\n    pass\nelif reco_args.truthTracking:\n"),
    ('\nsequenceLoader.load("Tracking/Refit")\n', '\nif not reco_args.digiOnly:\n    sequenceLoader.load("Tracking/Refit")\n'),
    (
        'if not reco_args.native:\n    sequenceLoader.load("HighLevelReco/RecoMCTruthLink")\n',
        'if not reco_args.native and not reco_args.digiOnly:\n    sequenceLoader.load("HighLevelReco/RecoMCTruthLink")\n',
    ),
]


def main():
    src, dst = sys.argv[1:3]
    text = open(src).read()
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"patch_cld_reco: anchor not found exactly once in {src}:\n{old}")
        text = text.replace(old, new)
    open(dst, "w").write(text)


if __name__ == "__main__":
    main()
