from slider import Beatmap
from slider.beatmap import Circle, Slider, HitObject
from typing import List
import numpy as np
import math

from .slider_process import linearize

__all__ = [
    'make_inputs'
]

EPSILON = 1e-9


def check_2b(hit_objects: List[HitObject]) -> bool:
    """Check whether a map is 2B style

    Args:
        hit_objects (List[HitObject]): A list of hit objects sorted
                                       by start time

    Returns:
        True if the beatmap is 2B style, False otherwise
    """
    tick = -1
    for obj in hit_objects:
        if obj.time_s <= tick + EPSILON:
            return True
        tick = obj.time_s
        if isinstance(obj, Slider):
            tick = obj.end_time_s
    return False


def move_to(tick, position, frame_idx,
            next_tick, next_position,
            capture_rate, samples):
    if next_tick < math.floor(frame_idx / capture_rate) + EPSILON:
        # Skip this event since it is not captured
        return next_tick, next_position, frame_idx

    frame_before_arrival = math.floor(next_tick * capture_rate)
    speed = (next_position - position) / (next_tick - tick + EPSILON)

    direction = math.atan2(speed[1], speed[0])
    length = np.linalg.norm(speed, ord=2)
    samples[frame_idx:frame_before_arrival + 1, 0:2] = [direction, length]

    return next_tick, next_position, frame_before_arrival + 1


def fill_trajectory(hit_objects: List[HitObject],
                    capture_rate: int,
                    samples: np.ndarray):
    """Generate the trajectory and fill the samples buffer

    Args:
        hit_objects (List[HitObject]): A list of hit objects sorted
                                       by start time
        capture_rate (int): The capture rate of the trajectory in Hz
        samples (np.ndarray): The target buffer

    """
    tick = 0
    frame_idx = 0
    position = np.array(hit_objects[0].position, dtype=np.float32)

    for current_obj in hit_objects:
        # Move the cursor to the object
        next_tick = current_obj.time_s
        next_position = np.array(current_obj.position, dtype=np.float32)
        tick, position, frame_idx = move_to(tick, position, frame_idx,
                                            next_tick, next_position,
                                            capture_rate, samples)
        # Follow if the object is a slider
        if isinstance(current_obj, Slider):
            total_time = current_obj.end_time_s - current_obj.time_s
            one_pass_time = total_time / current_obj.repeat
            linearization = linearize(current_obj.curve, one_pass_time)

            current_delta_sign = 1
            current_pass_offset = 0
            for i in range(current_obj.repeat):
                ticks = (tick + current_delta_sign *
                         linearization[:, 2] + current_pass_offset)
                for j in range(linearization.shape[0]):
                    next_tick = ticks[j]
                    next_position = linearization[j, 0:2]

                    (tick,
                     position,
                     frame_idx) = move_to(tick, position, frame_idx,
                                          next_tick, next_position,
                                          capture_rate, samples)

                current_delta_sign = -current_delta_sign
                current_pass_offset = one_pass_time - current_pass_offset

    # Normalize the speed vector
    samples[:, 0:2] /= [math.pi, 3600]


def fill_keydown(hit_objects: List[HitObject],
                 capture_rate: int,
                 samples: np.ndarray):
    circle_time = np.array([o.time_s for o in hit_objects
                            if isinstance(o, Circle)],
                           dtype=np.float32)
    neareast_invoke_frames = np.rint(circle_time * capture_rate).astype(int)
    samples[neareast_invoke_frames, 2] = 1

    for slider in (o for o in hit_objects if isinstance(o, Slider)):
        nearest_start_frame = np.rint(slider.time_s * capture_rate).astype(int)
        nearest_end_frame = np.rint(
            slider.end_time_s * capture_rate).astype(int)
        samples[nearest_start_frame:nearest_end_frame + 1, 2] = 1


def fill_slider_tick(hit_objects: List[HitObject],
                     capture_rate: int,
                     samples: np.ndarray):
    for slider in (o for o in hit_objects if isinstance(o, Slider)):
        tick_frames = np.rint(np.linspace(slider.time_s * capture_rate,
                                          slider.end_time_s * capture_rate,
                                          slider.ticks)).astype(int)
        samples[tick_frames, 3] = 1


def make_inputs(beatmap: Beatmap, capture_rate: int=60) -> np.ndarray:
    """Make Auto Mod inputs of a beatmap

    Args:
        beatmap (Beatmap): The beatmap to process.
        capture_rate (int): The capture rate of the trajectory in Hz

    Returns:
        Trajectory of the map in dimension L x 4, where second axis
        is [D, R, T, S]

        L: Number of samples taken, approximately
           length of the map x capture_rate
        D: Direction of speed in [-pi, pi] mapped to [-1, 1]
        R: Norm of speed in [0, 3600] (px per second) mapped to [0, 1]
        T: Whether keydown should happen at this frame
        S: Whether the frame represents a tick in a slider

    Raises:
        NotImplementedError('Cannot make trajectory for 2B-style map'):
            If the beatmap is 2B style
    """
    hit_objecsts = sorted(beatmap.hit_objects_no_spinners,
                          key=lambda o: o.time)
    for obj in hit_objecsts:
        obj.time_s = obj.time.total_seconds()
        if isinstance(obj, Slider):
            obj.end_time_s = obj.end_time.total_seconds()

    # The trajectory is not defined for 2B style map
    if check_2b(hit_objecsts):
        raise NotImplementedError('Cannot make trajectory for 2B-style map')

    # Calculate beatmap length and frames required
    total_length = (hit_objecsts[-1].end_time_s if isinstance(
                    hit_objecsts[-1], Slider) else hit_objecsts[-1].time_s)
    # +1 for final empty frame
    total_frames = math.ceil(total_length * capture_rate) + 2

    # Create buffer to store result
    uniform_samples = np.zeros((total_frames, 4), dtype=np.float32)

    # Fill speed vectors in the buffer
    fill_trajectory(hit_objecsts, capture_rate, uniform_samples)

    # Fill keydown events in the buffer
    fill_keydown(hit_objecsts, capture_rate, uniform_samples)

    # Fill slider sections in the buffer
    fill_slider_tick(hit_objecsts, capture_rate, uniform_samples)

    return uniform_samples
