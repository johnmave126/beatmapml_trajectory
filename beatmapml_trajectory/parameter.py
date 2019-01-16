from slider import Beatmap
from slider.mod import ar_to_ms, circle_radius, od_to_ms_300
from collections import namedtuple

Parameter = namedtuple('Parameter', ['ar', 'cs', 'od', 'hp'])


def normalize_parameter(beatmap: Beatmap) -> Parameter:
    """Return normalized parameter of a beatmap

    Args:
        beatmap (Beatmap): The beatmap to process

    Returns:
        A named tuple of normalized parameters
        ar: Normalized fade-in time in [0, 1] mapped from [0, 2400]
        cs: Normalized radius of note in [0, 1] mapped from [0, 55]
        od: Normalized window to hit 300 in [0, 1] mapped from [0, 106]
        hp: Normalized window to hit 300 in [0, 1] mapped from [0, 10]
    """
    ar_ms = ar_to_ms(beatmap.approach_rate)
    cs_px = circle_radius(beatmap.circle_size)
    od_ms = od_to_ms_300(beatmap.overall_difficulty)
    return (ar_ms / 2400, cs_px / 55, od_ms / 106, beatmap.hp_drain_rate / 10)
