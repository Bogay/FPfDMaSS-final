from dataclasses import dataclass
from itertools import cycle
from random import choice, sample, seed, randint
from typing import List, Literal, Tuple, Optional
from copy import deepcopy
from music21 import note, stream, instrument, tempo, chord
from numpy import random
# import cv2

seed(48763)
random.seed(12345)

# img = cv2.imread("rect.jpeg")
# img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
# cv2.imwrite('./preview.jpeg', img)


@dataclass
class Config:
    indexes: List[int]


# Make it configurable
config = Config(indexes=[0, 1, 0, 2])

MEASURE_COUNT = 16

# piano chords
C_maj = ['C4', 'E4', 'G4']
G_maj = ['G4', 'B4', 'D5']
F_maj = ['F4', 'A4', 'C5']
A_maj = ['A3', 'C4', 'E4']

# piano pitch
melody_pitches = [
    'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
    'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6',
]

# percussion pitch
SNARE_DRUM_PS = 38
BASS_DRUM_PS = 35


def gen_bass_drum(img_type: Literal['R', 'G', 'B']) -> stream.Part:
    bass_part = stream.Part()
    bass_part.insert(0, instrument.BassDrum())
    measures = get_measures_by_type(img_type)
    indexes = cycle(config.indexes)
    for _ in range(MEASURE_COUNT):
        for note in measures[next(indexes)]:
            bass_part.append(deepcopy(note))
    return bass_part


def gen_snare_drum() -> stream.Part:
    snare_part = stream.Part()
    snare_part.insert(tempo.MetronomeMark(number=180))
    snare_part.insert(0, instrument.SnareDrum())
    for _ in range(MEASURE_COUNT):
        snare_part.append(note.Rest())
        snare_part.append(note.Note(SNARE_DRUM_PS, quarterLength=1))
        snare_part.append(note.Rest())
        snare_part.append(note.Note(SNARE_DRUM_PS, quarterLength=1))
    return snare_part


def gen_chords() -> stream.Part():
    chords_part = stream.Part()
    for _ in range(MEASURE_COUNT // 4):
        chords_part.append(chord.Chord(C_maj, quarterLength=4))
        chords_part.append(chord.Chord(G_maj, quarterLength=4))
        chords_part.append(chord.Chord(F_maj, quarterLength=4))
        chords_part.append(chord.Chord(A_maj, quarterLength=4))
    return chords_part


def gen_melody(img_type: Literal['R', 'G', 'B']) -> stream.Part:
    melody_part = stream.Part()
    melody_part.insert(0, instrument.Piano())
    note_p, rest_p = 6.5/8.0, 1.5/8.0
    prev_index = choice(range(len(melody_pitches)))
    print(prev_index, melody_pitches[prev_index])
    for _mc in range(MEASURE_COUNT):
        for _n in range(8):
            if (_mc%8 == 7) and _n == 8-1:
                melody_part.append(note.Note('C5', quarterLength=0.5))
                continue
            rest_or_note = random.choice(['N', 'R'], p=[note_p, rest_p])
            if rest_or_note == 'N':
                melody_part.append(note.Note(melody_pitches[prev_index], quarterLength=0.5))
                rg = range(max(0, prev_index-3), min(len(melody_pitches), prev_index+3))
                pd = [0.1] + [0.8/(len(rg)-2) for _ in range(len(rg)-2)] + [0.1]
                prev_index = random.choice(rg, p=pd)
            else:
                melody_part.append(note.Rest(quarterLength=0.5))
    return melody_part


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
        note.Note(BASS_DRUM_PS, quarterLength=0.5 * quarter_len)
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
s.append([gen_bass_drum('B'), gen_snare_drum(), gen_chords(), gen_melody('B')])
s.show()
