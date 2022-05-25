from dataclasses import dataclass
from itertools import cycle
from random import choice, sample, seed, randint
from typing import List, Literal, Tuple, Optional
from copy import deepcopy
from music21 import note, stream, instrument, tempo
# import cv2

seed(48763)

# img = cv2.imread("rect.jpeg")
# img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
# cv2.imwrite('./preview.jpeg', img)


@dataclass
class Config:
    indexes: List[int]


# Make it configurable
config = Config(indexes=[0, 1, 0, 2])

# percussion pitch
SNARE_DRUM_PS = 38
BASS_DRUM_PS = 35


def gen_bass_drum(img_type: Literal['R', 'G', 'B'], measure_count=8) -> stream.Part:
    bass_part = stream.Part()
    bass_part.insert(0, instrument.BassDrum())
    measures = get_measures_by_type(img_type)
    indexes = cycle(config.indexes)
    for _ in range(measure_count):
        for note in measures[next(indexes)]:
            bass_part.append(deepcopy(note))
    return bass_part


def gen_snare_drum(measure_count=8) -> stream.Part:
    snare_part = stream.Part()
    snare_part.insert(tempo.MetronomeMark(number=180))
    snare_part.insert(0, instrument.SnareDrum())
    for _ in range(measure_count):
        snare_part.append(note.Rest())
        snare_part.append(note.Note(SNARE_DRUM_PS, quarterLength=1))
        snare_part.append(note.Rest())
        snare_part.append(note.Note(SNARE_DRUM_PS, quarterLength=1))
    return snare_part


# Maybe add a iterface to do that
def get_measures_by_type(
    img_type: Literal['R', 'G', 'B']
) -> Tuple[List[note.Note], List[note.Note], List[note.Note]]:
    if img_type == 'R':
        return gen_segment(2), gen_segment(), gen_segment()
    elif img_type == 'G':
        return gen_segment(randint(4, 6)), gen_segment(), gen_segment()
    else:
        return gen_segment(1), gen_segment(), gen_segment()


def get_segments() -> List[List[note.Note]]:
    ret = [gen_segment() for _ in range(500)]
    return ret


def gen_segment(n: Optional[int] = None) -> List[note.Note]:
    seg_len = randint(1, 7) if n is None else n
    quarter_lengthes = calculate_quarter_lengthes(seg_len)
    return [
        note.Note(35, quarterLength=0.5 * quarter_len)
        for quarter_len in quarter_lengthes
    ]


def calculate_quarter_lengthes(k: int, total: int = 8) -> List[int]:
    assert 1 <= k <= total
    cut_points = [0, *sorted(sample([*range(1, total)], k=k - 1)), total]
    results = []
    for l, r in zip(cut_points, cut_points[1:]):
        results.append(r - l)
    assert len(results) == k
    return results


s = stream.Stream()
# TODO: determine type by input
s.append([gen_bass_drum('B'), gen_snare_drum()])
s.show()
