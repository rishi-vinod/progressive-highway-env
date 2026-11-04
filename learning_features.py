"""Stage 5 inputs and training rewards, derived only from observed traffic.

These functions describe distances and speeds; they never select or override
the neural network's action. Geometry matches the declared four-lane experiment.
"""

import numpy as np

from reactive import in_lane_or_entering, lane_number, traffic_rows

FEATURE_VERSION = 2
FEATURE_COUNT = 24


def lane_traffic(observation, lane):
    """Nearest front/rear bumper gaps and longitudinal speeds, in real units."""
    ego = observation[0]
    front_gap, rear_gap = 200.0, 200.0
    front_speed, rear_speed = 30.0, 0.0
    for car in traffic_rows(observation):
        if not in_lane_or_entering(car, lane):
            continue
        dx = float(car[1] - ego[1])
        if dx >= 0 and dx - 5 < front_gap:
            front_gap, front_speed = dx - 5, float(car[3])
        elif dx < 0 and -dx - 5 < rear_gap:
            rear_gap, rear_speed = -dx - 5, float(car[3])
    return front_gap, front_speed, rear_gap, rear_speed


def time_to_contact(gap, closing_speed):
    return float(np.clip(gap / closing_speed, 0, 10)) if closing_speed > 0.1 else 10.0


def encode_lanes(observation):
    """Ego motion plus left/current/right traffic; invariant to world-x drift."""
    ego = observation[0]
    lane = lane_number(observation)
    speed = float(ego[3])
    values = [speed / 30, (float(ego[2]) - 4 * lane) / 2, float(ego[4]) / 5]
    for candidate in (lane - 1, lane, lane + 1):
        if not 0 <= candidate < 4:
            values.extend([0, 0, 0, 0, 0, 0, 0])
            continue
        front, front_speed, rear, rear_speed = lane_traffic(observation, candidate)
        values.extend([1, front / 100, (front_speed - speed) / 30,
                       rear / 100, (rear_speed - speed) / 30,
                       time_to_contact(front, speed - front_speed) / 10,
                       time_to_contact(rear, rear_speed - speed) / 10])
    return np.clip(np.asarray(values, dtype=np.float32), -10, 10)


def lane_risk(observation, lane, *, include_rear=False):
    if not 0 <= lane < 4:
        return 1.0
    speed = max(0.0, float(observation[0, 3]))
    front, front_speed, rear, rear_speed = lane_traffic(observation, lane)
    headway_risk = max(0.0, 1 - front / max(10, 1.8 * speed))
    ttc_risk = max(0.0, 1 - time_to_contact(front, speed - front_speed) / 5)
    risk = max(headway_risk, ttc_risk)
    if include_rear:
        risk = max(risk, 1 - rear / max(10, rear_speed * 1.0),
                   1 - time_to_contact(rear, rear_speed - speed) / 5)
    return float(np.clip(risk, 0, 1))


def safety_reward(before, action, after, crashed):
    """Reward safe progress, penalize risky spacing before an eventual crash.

Collision is always -10, without the stock reward's on-road multiplier.
The training signal is hand-designed; the deployed policy is still learned.
"""
    if crashed:
        return -10.0
    reward = 0.5 * np.clip(float(after[0, 3]) / 30, 0, 1)
    reward -= lane_risk(after, lane_number(after))
    if int(action) in (0, 2):
        requested_lane = lane_number(before) + (-1 if int(action) == 0 else 1)
        reward -= 0.03 + lane_risk(before, requested_lane, include_rear=True)
    return float(reward)
