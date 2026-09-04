# Wing Journal Markup (WJM)

**Status:** Project specification / design draft  
**Primary implementation language:** Python  
**Primary computer-vision stack:** OpenCV  
**Machine-readable page markers:** ArUco-compatible fiducials

## 1. Purpose

Wing Journal Markup (WJM) is a paper-native structured information system. It lets a person write naturally on ordinary paper while allowing the page to be scanned, identified, normalized, parsed, linked, versioned, compared against previous captures, and represented digitally.

The system is intended to support:

- bullet journaling and state changes;
- normal handwritten notes;
- structured boxes and nodes;
- hand-drawn data-flow/data-structure diagrams;
- graph relationships between boxes;
- page-to-page spatial relationships;
- logical documents spanning multiple pages, notebooks, or loose sheets;
- human-writable page identifiers and references;
- semantic topic tags;
- anchors and links;
- temporal data such as due dates, events, and ranges;
- contact information;
- static image/literal regions that must not be parsed;
- repeated rescanning of the same physical page;
- optional machine-readable fiducials;
- best-effort ingestion when fiducials are absent, damaged, misaligned, or unreadable.

The central design principle is:

> **The page is a persistent structured object. A photograph or scan is only an observation of that object.**

A new capture of an already-known page should update that page rather than create an unrelated OCR document.

---

## 2. System Architecture

```text
CAPTURE SOURCE
    |
    v
IMAGE ACQUISITION
    |
    v
FIDUCIAL + STRUCTURE DETECTION
    |
    v
PAGE BOUNDARY / ORIENTATION HYPOTHESES
    |
    v
PERSPECTIVE NORMALIZATION
    |
    v
VISUAL ELEMENT CATALOG
    |
    v
HANDWRITING / TEXT RECOGNITION
    |
    v
WJM MARKUP PARSER
    |
    v
PAGE / DOCUMENT / SEMANTIC GRAPH UPDATE
    |
    v
CAPTURE HISTORY + CHANGE RECONCILIATION
```

WJM maintains three related but distinct graph domains:

1. **Physical Page Graph**
   - left/right/above/below relationships;
   - page identity;
   - physical or logical arrangement.

2. **Semantic Content Graph**
   - nodes;
   - edges;
   - tags;
   - anchors;
   - references;
   - contacts;
   - temporal relationships.

3. **Capture / Revision Graph**
   - successive observations of a page;
   - change detection;
   - semantic reconciliation.

These graphs share persistent identifiers but should not be collapsed into one model.

---

## 3. Physical Notebook, Document, Page, and Capture

### 3.1 Physical Notebook

A physical notebook is only a physical container. It is **not automatically one logical document**.

One notebook may contain:

- multiple logical documents;
- unrelated notes;
- pages that are not yet assigned to a document;
- pages that later become linked into a document.

A logical document may span:

- multiple notebooks;
- loose paper;
- printed sheets;
- inserted pages;
- whiteboard captures;
- other page-like media.

### 3.2 Document

A document is a logical collection of pages.

Document membership can be:

- explicitly written in the page metadata block;
- inferred from spatial linkage;
- inherited through a connected component of linked pages.

Explicit and inferred values must remain distinguishable.

Example:

```text
document_id_explicit = null
document_id_resolved = "Research"
document_id_resolution_source = "spatial_graph"
```

### 3.3 Page

A `Page` is a persistent object that may have:

- handwritten page ID;
- optional machine page ID;
- logical document ID;
- topic tags;
- spatial relationships;
- multiple captures;
- structured elements;
- semantic nodes and edges;
- literal image regions.

### 3.4 Capture

A `Capture` is one image observation of a page.

It records:

- source;
- timestamp;
- raw image;
- normalized image;
- detected and inferred fiducials;
- page-boundary hypothesis;
- orientation;
- perspective transform;
- parsed elements;
- confidence;
- previous capture relationship.

---

## 4. Capture Sources

The Python codebase should use a source abstraction:

```python
class CaptureSource:
    def capture(self) -> "Image":
        ...
```

Initial implementations:

- `FileSource`
- `DirectorySource`
- `CameraSource`

Possible later sources:

- document camera;
- scanner;
- network camera;
- mobile upload;
- watched directory;
- clipboard image.

Example CLI:

```bash
wingjournal capture --camera 0
wingjournal ingest page.jpg
wingjournal ingest ./captures/
wingjournal show-page Research:P017
wingjournal history Research:P017
```

The processing pipeline must not depend on where the image came from.

---

## 5. Fiducial Philosophy

Fiducials are **strong evidence, not a requirement**.

They can help establish:

- page boundary;
- coordinate system;
- page orientation;
- perspective;
- scale;
- optional machine identity;
- optional overlay/template identity.

A page must remain ingestible when:

- markers are rotated;
- markers are badly aligned;
- markers are unevenly spaced;
- markers are partly obscured;
- one or more markers are missing;
- markers are not decodable;
- markers are merely empty squares;
- no markers exist.

---

## 6. Fiducial Deployment Modes

Supported physical deployment should include:

- directly printed fiducials;
- preprinted page templates;
- transparent overlays;
- reusable transparency masks;
- rigid scanning frames;
- temporary stickers;
- permanent stickers;
- adhesive corner tabs;
- mixed arrangements.

This allows an ordinary notebook page to become scannable without proprietary paper.

---

## 7. Misaligned Fiducials and Bounding Geometry

Marker placement is not assumed to be precise.

A sticker may be:

- independently rotated;
- skewed;
- too far inward or outward;
- not parallel to the page edge;
- not aligned with the other stickers.

The **collective marker positions** form a candidate quadrilateral.

```text
       [rotated marker]
             \________________ [marker]
             |                /
             |               /
      [marker]______________[rotated marker]
```

The marker constellation provides:

- top-left/top-right/bottom-left/bottom-right position candidates;
- page quadrilateral;
- perspective;
- relative page orientation;
- normalized page coordinate system.

The system must distinguish:

```text
marker_orientation != page_orientation
```

A badly rotated sticker can still be the top-left marker because its **position in the constellation** indicates that role.

Marker rotation is evidence, not authority.

---

## 8. Marker Identity

When permanent page-specific fiducials are used, the marker tuple may form a machine-readable page header.

Conceptually:

```text
TL -> document / namespace component
TR -> page identity component
BL -> orientation / type / metadata component
BR -> relationship / extension / integrity component
```

The exact bit/ID encoding is an implementation decision.

Core rule:

> **The tuple of markers may identify the page; one marker does not need to equal one page number.**

For reusable overlays, fiducials may represent only geometry/template information while handwritten metadata supplies logical page identity.

---

## 9. Capture Confidence Tiers

Graceful degradation:

```text
4 readable fiducials       -> very high confidence
3 readable fiducials       -> high confidence
2 readable fiducials       -> geometry + structure inference
1 readable fiducial        -> structure-dominant inference
0 readable fiducials       -> freeform structural inference
```

Unreadable but plausible square markers remain candidates.

```python
FiducialCandidate(
    bbox=(...),
    decoded=False,
    marker_id=None,
    inferred_role="TOP_LEFT",
    confidence=0.82
)
```

### 9.1 Capture Sharpness

A soft capture — motion blur, missed focus, too far — rectifies into a page
whose thin ink is unreadable, so sharpness is scored before any of it is
trusted.

Two signals, combined:

- **global** — variance of the Laplacian over the page (classic focus metric);
- **targeted** — edge acutance measured *at the known fiducials* (the corner
  ArUco markers and the metadata-block registration marks). Their ring/border
  edges are a known step, so a soft edge there is a soft photo, independent of
  page content. This is the deciding signal.

```python
SharpnessReport(
    score=0.34,                 # 0 soft .. 1 crisp
    laplacian_variance=41.0,
    probes=[
        SharpnessProbe("marker:0", 0.71, sharp=True),
        SharpnessProbe("registration:1", 0.09, sharp=False),
        ...
    ],
    blurry=True,
)
```

A live capture app **must** gate auto-shutter on `not blurry`; `ingest` records
the whole report on the `Capture` and flags a blurry page rather than pretending
it read clean.

---

## 10. Canonical Tag Syntax

All human-written identifiers use one of two forms.

Without spaces:

```text
#term
```

With spaces:

```text
#[term with spaces]
```

Examples:

```text
#P017
#auth
#AI
#Research
#[Data Science]
#[Project Alpha]
```

Canonical parsing:

```text
#auth            -> auth
#[Data Science]  -> Data Science
```

The brackets are quoting syntax for a tag containing spaces.

---

## 11. Page Metadata Block

A segmented title box at the **top of a page** is the canonical page metadata structure.

It may appear on any individual page.

Not all segments need to be filled.

Logical layout:

```text
+----------------+----------------+------------------------------+
| DOCUMENT ID    | PAGE ID        | TOPIC TAG IDS                |
+------------+------------+------------+--------------------------+
| LEFT       | ABOVE      | BELOW      | RIGHT                    |
+------------+------------+------------+--------------------------+
```

The important part is **segment order**, not equal widths.

### 11.1 Corner Registration Marks

The four corners of the printed metadata block carry a **concentric-square
registration mark** — a solid dark square, a bright square inside it, and a
small dark square at the centre:

```text
◎═══════╤═══════╤═══════◎
║ DOC ID │ PAGE ID│ TOPIC ║
╟──┬──┼──┬──┼───────╢
║ L│AB│BL│  RIGHT   ║
◎═══╧═══╧═══╧═══════◎
```

Detection reads the block two ways, in order:

1. **registration marks** — locate the four marks (nested dark→bright→dark
   contours, roughly square and concentric), order them TL/TR/BR/BL, and take
   that quad as the block. The cell grid follows from segment order. Solid ink,
   so this survives the dim or soft photos that erase the thin rules.
2. **ruled lines** — the morphology / projection fallback, for sheets printed
   before the marks, hand-drawn blocks, or overlays.

The marks also serve as sharpness probes (§9.1).

### 11.2 Adhesive Corner Stickers

Instead of a printed sheet, four **identical** adhesive ArUco stickers
(``CORNER_STICKER_ID``, a reserved id distinct from the printed sheet's
``0/1/2/3``) turn any page into a WJM page. Each sticker carries a solid
photo-corner **wedge** (its right-angle vertex tucks into the paper corner) and
a thin edge rule, both pointing the same way. ``make-stickers`` prints a grid of
them; the user rotates each 90° for its corner.

Because the stickers are identical, the id is not the corner — geometry (the
constellation) and the wedge direction give the role. Detection yields, per
sticker:

```python
CornerSticker(
    outward=[-0.6, -0.8],        # unit vector, marker centre -> page corner
    corner_point=[x, y],         # wedge tip = the page corner (tucked in), or extrapolated
    bracket_found=True,
    inferred_role="TOP_LEFT",
)
```

The wedge tips are the page quadrilateral (a ``corner_stickers`` boundary
hypothesis). The sticker ArUco is a **fixed physical size**, so its scale in
pixels gives px/mm and the constellation gives a page-size guess:

```python
PageSizeEstimate(
    width_mm=214.0, height_mm=277.0,
    best_match="letter", match_error_mm=6.0,
    method="corner_stickers",
)
```

This is a hint, not a measurement — good on a flat capture, rough under strong
perspective; ``match_error_mm`` carries the uncertainty and ``best_match`` is
``None`` when it is too ambiguous to name.

Schema:

```python
PageMetadata(
    document_id=None,
    page_id=None,
    topic_tags=[],
    left=None,
    above=None,
    below=None,
    right=None
)
```

### Row 1

1. `DOCUMENT ID`
2. `PAGE ID`
3. `TOPIC TAG IDS`

### Row 2

1. `LEFT`
2. `ABOVE`
3. `BELOW`
4. `RIGHT`

Example:

```text
+--------------+----------+-----------------------------+
| #Research    | #P017    | #AI #[Data Science]        |
+----------+----------+----------+-----------------------+
| #P016    |          | #P027    | #P018                |
+----------+----------+----------+-----------------------+
```

Parsed:

```python
PageMetadata(
    document_id="Research",
    page_id="P017",
    topic_tags=["AI", "Data Science"],
    left="P016",
    above=None,
    below="P027",
    right="P018"
)
```

Blank fields are valid.

---

## 12. Document Membership Propagation

Spatial linkage can propagate document membership.

Example:

```text
P17 --RIGHT--> P18 --BELOW--> P28
```

If only P17 explicitly contains:

```text
#Research
```

then P18 and P28 may resolve to the same document through the connected page graph.

Store explicit and resolved values separately:

```python
document_id_explicit = None
document_id_resolved = "Research"
document_id_resolution_source = "connected_component"
```

The system must not silently rewrite the physical page content.

If connected pages contain conflicting explicit document IDs, create a conflict rather than choosing silently.

---

## 13. Reciprocal Page Relationships

Spatial relationships are logically reciprocal.

If:

```text
P17.right = P18
```

the graph may infer:

```text
P18.left = P17
```

Store provenance:

```python
PageRelationship(
    source="P17",
    target="P18",
    relation="RIGHT",
    explicitly_declared=True
)

PageRelationship(
    source="P18",
    target="P17",
    relation="LEFT",
    explicitly_declared=False
)
```

Core metadata relationships:

- `LEFT`
- `ABOVE`
- `BELOW`
- `RIGHT`

Possible later graph-level relationships:

- `FACING`
- `BEFORE`
- `AFTER`
- `CONTINUATION`

---

## 14. Ordinary Boxes: First-Class Semantic Nodes

A simple rectangular box is a semantic node.

```text
+--------------------------+
| Authentication Service   |
|                          |
| Handles user sessions    |
| #backend                 |
| #[Project Alpha]         |
+--------------------------+
```

Its interior remains parseable.

It may contain:

- ordinary text;
- topic tags;
- anchors;
- references;
- temporal markup;
- contact data;
- bullets;
- other recognized semantics.

A tag inside a box belongs to that node unless another explicit scope rule overrides it.

Example:

```text
+------------------+
| Database         |
| #security        |
+------------------+
```

`#security` applies to `Database`.

---

## 15. Segmented Boxes

A segmented box contains deliberate internal dividers.

The page metadata block is one defined segmented-box form.

A semantic node may also use a title/body divider:

```text
+-------------------------+
| Authentication Service  |
+-------------------------+
| JWT validation          |
| Redis session cache     |
+-------------------------+
```

The upper region is the title.

The lower region is the body.

---

## 16. Literal / Static Image Box

WJM requires an escape mechanism equivalent to an escaped literal region.

A rectangular region whose **four corners are diagonally filled in black**, visually similar to old photo-album mounts, is a `LiteralAsset`.

Conceptually:

```text
◩--------------------------◪
|                           |
|  freehand drawing         |
|  photograph               |
|  arbitrary handwriting    |
|  any visual content       |
|                           |
◫--------------------------◧
```

The exact glyphs above are illustrative. Detection is based on the visual convention: **all four corners contain diagonal black corner fills**.

Processing:

```text
detect literal boundary
        |
        v
crop interior
        |
        v
store as image asset
        |
        v
DO NOT PARSE INTERIOR
```

The parser must not perform inside the literal region:

- OCR;
- bullet recognition;
- tag parsing;
- node extraction;
- line/edge extraction;
- anchor resolution;
- temporal parsing;
- contact extraction.

Object:

```python
LiteralAsset(
    page_id="P017",
    bbox=(...),
    image_asset="asset-id"
)
```

Literal boxes should be detected and masked **before detailed semantic parsing**.

---

## 17. Diagram Nodes and Edges

Ordinary boxes become graph nodes.

```text
+-----------------------+
| API Gateway           |
+-----------------------+
| REST endpoint         |
| auth + rate limiting  |
+-----------------------+
```

Possible representation:

```python
DiagramNode(
    id="node-123",
    title="API Gateway",
    body="REST endpoint\nauth + rate limiting",
    bbox=(...)
)
```

Lines between nodes become graph edges.

Initial visual vocabulary:

```text
------     generic / undirected connection
----->     directed
<-----     directed
<---->     bidirectional
- - -      weak / logical / optional relation
```

Example:

```text
+---------+       HTTPS        +----------+
| API     | -----------------> | Worker   |
+---------+                     +----------+
```

Representation:

```python
DiagramEdge(
    source="API",
    target="Worker",
    direction="forward",
    label="HTTPS"
)
```

Line endpoints and proximity to node boundaries determine attachment.

---

## 18. Bullet Journal Semantics

Bullet marks are first-class semantic objects.

Initial state vocabulary:

```text
•  open task
×  completed
>  migrated
<  scheduled
–  note
○  event
!  important
?  question / research
```

The vocabulary is extensible.

Example:

```text
• Implement parser
```

becomes:

```python
BulletItem(
    text="Implement parser",
    state="open"
)
```

If a later capture contains:

```text
× Implement parser
```

the system should reconcile this as:

```text
OPEN -> COMPLETED
```

instead of creating a duplicate object.

---

## 19. Anchors and Address Links

An anchor makes a page or semantic object addressable.

User-visible anchor names use the normal tag syntax:

```text
#AUTH
#[Authentication Service]
```

An anchor can apply to:

- a page;
- a node;
- a note;
- a task;
- a diagram region;
- another semantic object.

Internal fully qualified forms can use:

```text
document : page : anchor
```

Example:

```text
Research:P017:AUTH
```

A separate special hand-writable enclosure should identify an **address link/reference** so it is not interpreted as ordinary prose.

Conceptual forms:

```text
-> [#AUTH]
```

or:

```text
REF: #AUTH
```

The precise reference enclosure may be finalized later.

References can resolve:

- locally on the same page;
- to another page;
- to another document.

---

## 20. Topic Tags

Topic tags may appear:

- in page metadata;
- inside nodes;
- beside notes;
- beside tasks;
- within other semantic regions.

Examples:

```text
#AI
#python
#[Data Science]
#[Graph Theory]
```

Page-level topic tags are many-to-many.

---

## 21. Temporal Tags

Temporal markup should become structured data.

Potential explicit forms:

```text
[DUE: 2026-09-14]
[EVENT: 2026-09-18 14:00]
[RANGE: 2026-09-12 -> 2026-09-19]
```

Possible later natural forms:

```text
due Friday
next Tuesday
Sept 14
9/14-9/18
```

Example model:

```python
TemporalTag(
    type="due",
    start="2026-09-14"
)
```

or:

```python
TemporalTag(
    type="range",
    start="2026-09-12",
    end="2026-09-19"
)
```

---

## 22. Contact Information

A contact region may use a dedicated semantic box:

```text
+ CONTACT ------------------+
| Jane Smith                |
| jane@example.com          |
| 555-123-4567              |
| Acme Corp                 |
+---------------------------+
```

Representation:

```python
Contact(
    name="Jane Smith",
    email="jane@example.com",
    phone="555-123-4567",
    organization="Acme Corp"
)
```

Contacts may themselves be anchored and referenced elsewhere.

---

## 23. Page Boundary Detection Strategy

Boundary detection is progressive rather than binary.

The system should:

```text
1. Decode fiducials
2. Detect fiducial-like square contours
3. Catalog recognizable semantic structures
4. Generate page-boundary hypotheses
5. Score hypotheses
6. Select the strongest candidate
7. Normalize the page
8. Rerun detailed recognition
```

---

## 24. Decodable Fiducials

When readable ArUco markers exist:

- decode marker ID;
- store marker polygon;
- calculate center;
- store marker-local rotation;
- infer possible corner role;
- combine markers into candidate page frames.

A decoded marker is high-value evidence but does not alone dictate final page orientation.

---

## 25. Square / Fiducial Candidates

Square or near-square contours should initially be cataloged without assuming semantics.

Possible classifications:

- ordinary semantic box;
- checkbox;
- damaged fiducial;
- blank fiducial;
- decorative square;
- unknown.

Never use:

```text
square -> fiducial
```

Use context:

- proximity to inferred page corner;
- similarity in size to other candidates;
- content inside the quadrilateral;
- metadata-block placement;
- plausible page geometry;
- relationship to decoded markers.

---

## 26. Empty Corner Squares as Misaligned Fiducials

If no readable payload exists, empty square bounding boxes in the corners may be treated as probable misaligned/damaged fiducials **when the rest of the page contains recognizable WJM elements**.

Example:

```text
□                                  □

       metadata block

       notes / boxes / links

□                                  □
```

The existence of catalogable WJM structures inside the candidate frame makes the corner squares stronger fiducial evidence.

Example:

```python
FiducialCandidate(
    decoded=False,
    marker_id=None,
    inferred_role="TOP_RIGHT",
    reason="corner_geometry+content_containment",
    confidence=0.79
)
```

This supports:

- badly printed markers;
- blank marker stickers;
- obscured payloads;
- hand-drawn square scan guides;
- damaged markers;
- improperly aligned stickers.

---

## 27. Structural Catalog Before Final Rectification

In fallback mode, the system may perform a **coarse semantic catalog before final page rectification**.

This is intentional.

Useful pre-rectification structures include:

- metadata-box candidates;
- ordinary boxes;
- literal boxes;
- text clusters;
- bullet groups;
- arrows;
- long lines;
- topic tags;
- contact blocks;
- temporal blocks.

These elements help infer the page extent and orientation.

---

## 28. Freeform Page-Boundary Inference

When no useful fiducials exist:

1. catalog all identifiable elements;
2. compute their combined content region;
3. inspect surrounding blank margin;
4. determine dominant line/text orientation;
5. search for the metadata block;
6. search for corner square candidates;
7. generate plausible quadrilateral page frames;
8. score containment and alignment;
9. select the strongest page frame;
10. retain the inference method and confidence.

Conceptually:

```python
content_bounds = union_bbox(recognized_elements)
```

Example boundary result:

```python
PageBoundary(
    polygon=(...),
    method="structure_inferred",
    confidence=0.71
)
```

The system must preserve uncertainty rather than pretending the boundary was directly observed.

---

## 29. Mixed-Mode Reconstruction

One image may contain a mixture such as:

```text
TOP_LEFT     decoded ArUco
TOP_RIGHT    unreadable marker
BOTTOM_LEFT  missing
BOTTOM_RIGHT empty square
```

The system should combine all available evidence:

```text
decoded marker
+ geometric marker candidates
+ empty corner squares
+ metadata block
+ semantic content distribution
+ text orientation
+ box alignment
= page hypothesis
```

There is no hard "fiducial mode" versus "freeform mode."

---

## 30. Page Hypothesis Model

The detector should generate one or more candidate page hypotheses.

```python
PageHypothesis(
    polygon=(...),
    decoded_fiducials=1,
    inferred_fiducials=2,
    structural_elements=14,
    metadata_block_detected=True,
    content_containment_score=0.94,
    orientation_score=0.89,
    confidence=0.88
)
```

Alternative hypotheses may be retained for debugging and diagnostics.

---

## 31. Suggested Hypothesis Scoring

Possible positive evidence:

```text
decoded marker confidence
+ marker constellation geometry
+ square-candidate geometry
+ metadata-block location
+ content containment
+ dominant text orientation
+ box-edge consistency
+ plausible page aspect ratio
+ plausible content margins
+ marker-role consistency
+ page-ID recognition
```

Possible penalties:

```text
outlier content
contradictory orientation
relationship inconsistency
impossible quadrilateral
excessive clipping
poor WJM-structure containment
```

Weights should be configurable.

---

## 32. Orientation Fallthrough Logic

Orientation should be resolved in descending confidence order.

### Tier A — Marker Identity + Marker Geometry

If marker identities establish corner roles and the geometry is plausible:

```text
use marker roles
-> establish page orientation
-> rectify perspective
```

### Tier B — Marker Geometry, Ignore Bad Sticker Rotation

If markers decode but individual marker rotations are inconsistent:

```text
use marker centers / constellation
ignore contradictory local marker rotation
```

### Tier C — Partial Marker Geometry

With two or three useful markers:

```text
known marker positions
+ square candidates
+ metadata block
+ text orientation
+ page structures
```

are used to complete the missing frame.

### Tier D — Geometric Square Candidates

If payloads cannot be decoded:

```text
corner-like squares
+ content containment
+ metadata location
```

are used to infer a candidate frame.

### Tier E — Metadata-Driven Orientation

If the segmented metadata block is identified:

> The metadata block is defined as a top-of-page structure.

Therefore:

```text
metadata block -> strong TOP evidence
```

Text orientation is then used to determine 0°, 90°, 180°, or 270°.

### Tier F — Text and Structural Orientation

If no metadata block exists, use:

- dominant handwriting baseline;
- box orientation;
- title/body divider orientation;
- bullet-list direction;
- diagram text direction;
- long horizontal/vertical line distribution.

### Tier G — Best-Effort Content Envelope

If no stronger signal exists:

```text
catalog content
-> derive content envelope
-> generate plausible page orientation
-> choose orientation maximizing normal WJM layout
```

Mark the result low confidence.

---

## 33. Canonical Page-Boundary Fallthrough

```text
START
 |
 +-- Detect 4 valid fiducials?
 |      |
 |      +-- YES
 |            build marker quadrilateral
 |            validate against content
 |            score hypothesis
 |
 +-- Else detect 2-3 valid fiducials?
 |      |
 |      +-- YES
 |            combine with geometric square candidates
 |            combine with structural elements
 |
 +-- Detect fiducial-like square contours?
 |      |
 |      +-- YES
 |            generate marker-frame hypotheses
 |            validate content containment
 |
 +-- Detect metadata block?
 |      |
 |      +-- YES
 |            apply strong top-of-page evidence
 |
 +-- Catalog:
 |      boxes
 |      literal boxes
 |      bullets
 |      long lines
 |      arrows
 |      text clusters
 |      tags
 |
 +-- Build structural content envelope
 |
 +-- Infer candidate page quadrilateral(s)
 |
 +-- Reinspect squares near inferred corners
 |      |
 |      +-- plausible?
 |            promote to inferred fiducials
 |
 +-- Rescore hypotheses
 |
 +-- Select highest-confidence boundary/orientation
 |
 +-- Normalize perspective
 |
 +-- Rerun detailed recognition
 |
END
```

The logic is intentionally iterative.

A provisional page boundary may make previously ambiguous squares recognizable as fiducials, which can then improve the boundary estimate.

---

## 34. Iterative Boundary Refinement

Recommended passes:

```text
PASS 1
detect obvious machine markers and structures

PASS 2
infer rough page extent

PASS 3
reinterpret corner squares

PASS 4
refine quadrilateral and orientation

PASS 5
rectify page

PASS 6
rerun structural and semantic recognition
```

This is a key robustness feature.

---

## 35. Perspective Normalization

Once a page quadrilateral is selected:

1. order the four page corners;
2. compute homography;
3. warp to normalized page coordinates;
4. preserve the transform matrix;
5. perform detailed processing in normalized coordinates.

Conceptually:

```python
normalized = cv2.warpPerspective(
    raw_image,
    homography,
    output_size
)
```

Normalized coordinates make repeated scans directly comparable.

---

## 36. Literal-Box Processing Order

Literal boxes should be detected before detailed OCR and graph extraction.

```text
detect literal boxes
    |
    v
extract literal image assets
    |
    v
mask literal interiors
    |
    v
run detailed WJM parsing on remaining page
```

This prevents a drawing inside a literal box from accidentally becoming:

- graph nodes;
- arrows;
- OCR text;
- tags;
- bullets.

---

## 37. Repeated Captures and Versioning

A page may have many captures:

```text
Research:P017
    Capture 0001
    Capture 0002
    Capture 0003
```

Each new normalized capture should be compared with prior captures.

Detect:

- unchanged regions;
- new handwriting;
- modified handwriting;
- removed/crossed-out content;
- bullet state changes;
- node body changes;
- newly added graph edges;
- removed graph edges;
- new tags;
- altered metadata.

---

## 38. Semantic Reconciliation

The system should update existing semantic objects when possible.

First capture:

```text
• Implement parser
```

Later capture:

```text
× Implement parser
```

Desired result:

```text
BulletItem 391
OPEN -> COMPLETED
```

not a second unrelated task.

Matching can use:

- normalized position;
- OCR text similarity;
- handwriting similarity;
- neighboring elements;
- prior bounding box;
- topic context.

---

## 39. Identity Resolution

Possible page-identity evidence, approximately strongest first:

```text
permanent machine page identity
>
handwritten metadata page ID
>
resolved spatial graph identity
>
visual matching against previous captures
>
new unknown page
```

Identity conflicts must be surfaced.

Example:

```text
machine page = P017
handwritten page = P019
```

This must not be silently resolved.

---

## 40. Canonical Page Example

```text
■------------------------------------------------■

+--------------+----------+-----------------------+
| #Research    | #P017    | #AI #[Data Science]  |
+----------+----------+----------+-----------------+
| #P016    |          | #P027    | #P018          |
+----------+----------+----------+-----------------+

• Investigate embedding methods
× Install dependencies

+---------------------------+
| Vector Database           |
+---------------------------+
| Stores embeddings         |
| #database                 |
+-------------+-------------+
              |
              v
+---------------------------+
| Retrieval Service         |
| #retrieval                |
+---------------------------+

◩---------------------------◪
|                           |
|   freehand drawing        |
|   preserved as image      |
|                           |
◫---------------------------◧

■------------------------------------------------■
```

Conceptual parse:

```text
PAGE
|
+-- Metadata
|   +-- document = Research
|   +-- page = P017
|   +-- topics = AI, Data Science
|   +-- left = P016
|   +-- below = P027
|   +-- right = P018
|
+-- BulletItem
+-- BulletItem
|
+-- DiagramNode
|   +-- topic = database
|
+-- DiagramEdge
|
+-- DiagramNode
|   +-- topic = retrieval
|
+-- LiteralAsset
```

---

## 41. Canonical Data Model

```text
Document
Page
PageRelationship

Capture
CaptureDiff

Element
+-- Text
+-- BulletItem
+-- DiagramNode
+-- DiagramEdge
+-- Anchor
+-- Reference
+-- TopicTag
+-- TemporalTag
+-- Contact
+-- LiteralAsset
```

Common element fields:

```python
class Element:
    uuid
    page_id
    bbox
    created_at
    modified_at
    confidence
    source_capture
```

---

## 42. Suggested Page Model

```python
class Page:
    uuid: str

    document_id_explicit: str | None
    document_id_resolved: str | None

    page_id_explicit: str | None
    page_id_machine: str | None

    topic_tags: list[str]

    left: str | None
    above: str | None
    below: str | None
    right: str | None

    captures: list[str]
    elements: list[str]
```

---

## 43. Suggested Capture Model

```python
class Capture:
    uuid: str
    page_uuid: str | None

    timestamp: datetime
    source_type: str

    raw_image_path: str
    normalized_image_path: str

    page_boundary_method: str
    page_boundary_confidence: float

    orientation_degrees: int | None
    orientation_confidence: float

    homography: list[list[float]]

    detected_fiducials: list[str]
    inferred_fiducials: list[str]
    detected_elements: list[str]

    previous_capture_uuid: str | None
```

---

## 44. Suggested Page Relationship Model

```python
class PageRelationship:
    uuid: str
    source_page: str
    target_page: str
    relation: str

    explicitly_declared: bool
    source_capture: str
    confidence: float
```

---

## 45. Data Provenance

Resolved values should track their source.

```python
ResolvedValue(
    value="Research",
    source="handwritten_metadata",
    confidence=0.93
)
```

or:

```python
ResolvedValue(
    value="Research",
    source="spatial_inheritance",
    confidence=0.86
)
```

The system should distinguish what the user explicitly wrote from what software inferred.

---

## 46. Conflict Philosophy

Prefer:

```text
preserve evidence
+ retain provenance
+ store confidence
+ surface conflict
```

instead of silently guessing.

Examples:

- machine page ID conflicts with handwritten page ID;
- P17 says right=P18 while P18 explicitly says left=P19;
- linked pages carry different explicit document IDs;
- metadata orientation conflicts with strong machine-marker identity.

---

## 47. Proposed Python Package Layout

```text
wingjournal/
|
+-- capture/
|   +-- source.py
|   +-- camera.py
|   +-- file.py
|   +-- directory.py
|
+-- vision/
|   +-- aruco.py
|   +-- fiducial_candidates.py
|   +-- boundary.py
|   +-- orientation.py
|   +-- rectify.py
|   +-- contours.py
|   +-- boxes.py
|   +-- literal_boxes.py
|   +-- lines.py
|   +-- arrows.py
|   +-- symbols.py
|
+-- recognition/
|   +-- handwriting.py
|   +-- metadata.py
|   +-- tags.py
|   +-- bullets.py
|   +-- temporal.py
|   +-- contacts.py
|
+-- parser/
|   +-- page.py
|   +-- diagram.py
|   +-- markup.py
|   +-- references.py
|
+-- graph/
|   +-- pages.py
|   +-- documents.py
|   +-- anchors.py
|   +-- relationships.py
|   +-- semantic.py
|
+-- versioning/
|   +-- captures.py
|   +-- image_diff.py
|   +-- semantic_diff.py
|   +-- reconcile.py
|
+-- models/
|   +-- document.py
|   +-- page.py
|   +-- element.py
|   +-- capture.py
|   +-- relationship.py
|
+-- storage/
|   +-- database.py
|   +-- images.py
|   +-- assets.py
|
+-- scoring/
|   +-- boundary.py
|   +-- orientation.py
|   +-- confidence.py
|
+-- cli.py
```

---

## 48. Full Processing Pipeline

```text
IMAGE INPUT
    |
    v
PREPROCESS
    |
    +-- grayscale
    +-- contrast normalization
    +-- edge extraction
    +-- contour extraction
    |
    v
DECODE ARUCO
    |
    v
DETECT SQUARE/FIDUCIAL CANDIDATES
    |
    v
COARSE STRUCTURAL CATALOG
    |
    +-- metadata candidates
    +-- ordinary boxes
    +-- literal boxes
    +-- text regions
    +-- bullet regions
    +-- long lines
    +-- arrows
    |
    v
GENERATE PAGE HYPOTHESES
    |
    v
SCORE PAGE HYPOTHESES
    |
    v
CHOOSE BOUNDARY + ORIENTATION
    |
    v
RECTIFY
    |
    v
DETECT + MASK LITERAL ASSETS
    |
    v
DETAILED STRUCTURAL DETECTION
    |
    v
OCR / HANDWRITING RECOGNITION
    |
    v
WJM MARKUP PARSE
    |
    v
RESOLVE PAGE IDENTITY
    |
    v
COMPARE WITH PREVIOUS CAPTURE
    |
    v
UPDATE PAGE + DOCUMENT + SEMANTIC GRAPHS
```

---

## 49. Canonical Fallthrough Summary

```text
TRY MACHINE FIDUCIALS
        |
        v
TRY PARTIAL / DAMAGED FIDUCIAL GEOMETRY
        |
        v
CATALOG EMPTY SQUARE CANDIDATES
        |
        v
CATALOG WJM PAGE STRUCTURES
        |
        v
TRY METADATA-BLOCK ORIENTATION
        |
        v
TRY TEXT / BOX / LINE ORIENTATION
        |
        v
BUILD CONTENT ENVELOPE
        |
        v
INFER PAGE QUADRILATERAL
        |
        v
REINTERPRET CORNER SQUARES
        |
        v
RESCORE PAGE HYPOTHESES
        |
        v
SELECT HIGHEST-CONFIDENCE BOUNDARY
        |
        v
RECTIFY PAGE
        |
        v
RERUN DETAILED RECOGNITION
        |
        v
PARSE WJM
```

The governing rule is:

> **Machine markers are preferred, structural inference is always available, and all available visual evidence may cooperate to reconstruct the page.**

---

## 50. Design Principles

1. Paper remains the primary user interface.
2. Boxes are first-class semantic objects.
3. A four-diagonal-black-corner box is an escaped literal image region.
4. Literal-region interiors are never semantically parsed.
5. The segmented metadata block at the top of a page defines page metadata.
6. Metadata uses consistent `#term` and `#[term with spaces]` syntax.
7. A physical notebook is not equivalent to one logical document.
8. Spatial linkage creates a physical/logical page graph.
9. Document membership may propagate through linked pages.
10. Explicit values and inferred values remain distinguishable.
11. Ordinary boxes can become semantic graph nodes.
12. Lines and arrows can become graph edges.
13. Bullet-journal marks represent persistent stateful objects.
14. Anchors make semantic objects addressable.
15. References create links to anchors.
16. Topic, temporal, and contact information become structured data.
17. Fiducials improve capture but are not mandatory.
18. Fiducials may be badly aligned and independently rotated.
19. The marker constellation matters more than sticker alignment.
20. Undecodable or empty corner squares remain useful geometric evidence.
21. In no-fiducial mode, recognized WJM structures help infer the page boundary.
22. Empty corner squares gain fiducial significance when WJM content is found inside their candidate frame.
23. Boundary inference is iterative and can reinterpret prior ambiguous elements.
24. Repeated captures update a persistent Page instead of creating unrelated documents.
25. Confidence and provenance are first-class data.
26. The system should degrade gracefully under imperfect capture conditions.

