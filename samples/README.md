# Samples

Ready-to-print WJM writing sheets — US Letter (8.5 × 11 in), 300 DPI. Each has
four corner ArUco fiducials (`DICT_4X4_50`, IDs 0/1/2/3 = TL/TR/BR/BL) and the
empty two-row page-metadata block (spec §11):

```
DOCUMENT ID | PAGE ID | TOPIC TAGS
LEFT | ABOVE | BELOW | RIGHT
```

| File | |
|---|---|
| `wjm-writing-sheet-letter.pdf` / `.png` | blank body |
| `wjm-writing-sheet-letter-ruled.pdf` / `.png` | faint ruled lines in the body |

Print at 100 % scale (no "fit to page") so the fiducials keep their physical
size. Then photograph or scan and run `wingjournal ingest <image>`.

## Regenerate

```bash
wingjournal make-sheet --out samples/wjm-writing-sheet-letter.pdf
wingjournal make-sheet --out samples/wjm-writing-sheet-letter.png
wingjournal make-sheet --out samples/wjm-writing-sheet-letter-ruled.pdf --ruled
wingjournal make-sheet --out samples/wjm-writing-sheet-letter-ruled.png --ruled
```

`--paper a4|legal`, `--marker-mm`, and `--pages` (PDF only) are also available.
