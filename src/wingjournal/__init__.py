"""Wing Journal Markup (WJM) - paper-native structured information system.

CLI edition. This package currently implements the front of the processing
pipeline described in ``docs/SPEC-v0-draft.md``:

    image acquisition -> preprocess -> ArUco / fiducial detection
    -> page-boundary hypothesis -> perspective normalization

Everything downstream (handwriting recognition, markup parsing, the page /
document / semantic graphs, capture versioning) is on the roadmap.
"""

__version__ = "0.1.0"
