import numpy as np

from wingjournal.recognition.metadata import read_metadata_block
from wingjournal.recognition.metadata_block import MetadataBlock
from wingjournal.recognition.text import RecognizedRegion, RecognizedWord, TextRecognizer


class ScriptedRecognizer(TextRecognizer):
    """Returns text based on which cell rectangle it is handed."""

    name = "scripted"

    def __init__(self, by_x: dict[int, str]) -> None:
        self.by_x = by_x

    def available(self) -> bool:
        return True

    def recognize(self, image: np.ndarray) -> RecognizedRegion:
        # the scripted map is keyed by crop width as a cheap cell id
        text = self.by_x.get(image.shape[1], "")
        if not text:
            return RecognizedRegion.unreadable(self.name)
        return RecognizedRegion(
            text=text,
            words=[RecognizedWord(text=text, bbox=[0, 0, 1, 1], confidence=0.9)],
            backend=self.name,
            recognized=True,
        )


def test_read_metadata_block_maps_cells_to_fields():
    # cells distinguished by width
    block = MetadataBlock(
        bbox=[0, 0, 300, 40],
        row_divider_y=20,
        row1_cells=[[0, 0, 10, 20], [0, 0, 11, 20], [0, 0, 12, 20]],
        row2_cells=[[0, 0, 20, 20], [0, 0, 21, 20], [0, 0, 22, 20], [0, 0, 23, 20]],
        confidence=1.0,
    )
    rec = ScriptedRecognizer({
        10: "#Research", 11: "#P017", 12: "#AI #[Data Science]",
        20: "#P016", 22: "#P027", 23: "#P018",  # 21 (above) left blank
    })
    img = np.zeros((40, 300), np.uint8)
    reading = read_metadata_block(img, block, rec)

    md = reading.metadata
    assert md.document_id == "Research"
    assert md.page_id == "P017"
    assert md.topic_tags == ["AI", "Data Science"]
    assert (md.left, md.above, md.below, md.right) == ("P016", None, "P027", "P018")
    assert reading.confidence == 0.9


def test_read_metadata_block_all_blank_is_empty():
    block = MetadataBlock(bbox=[0, 0, 100, 40], row_divider_y=20,
                          row1_cells=[[0, 0, 5, 20]], row2_cells=[[0, 0, 6, 20]],
                          confidence=0.5)

    class Blank(TextRecognizer):
        name = "blank"

        def available(self):
            return True

        def recognize(self, image):
            return RecognizedRegion.unreadable(self.name)

    reading = read_metadata_block(np.zeros((40, 100), np.uint8), block, Blank())
    assert reading.metadata.page_id is None
    assert reading.confidence == 0.0
