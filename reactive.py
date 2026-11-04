"""Stage 2 observation-only rules shared by reactive driving and takeover.

Geometry is a declared property of this four-lane straight-road experiment.
No environment, vehicle object, event schedule, or traffic intentions are read.
"""

from dataclasses import dataclass
import numpy as np

LANE_WIDTH_M = 4.0
LANES = 4
CAR_LENGTH_M = 5.0
CAR_WIDTH_M = 2.0
FRONT_GAP_M = 12.0
REAR_GAP_M = 12.0
HEADWAY_S = 1.5
TTC_S = 4.0
LANE_CHANGE_HORIZON_S = 3.0
CUT_IN_LOOKAHEAD_S = 2.0
TARGET_SPEED_MPS = 30.0
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = range(5)


def traffic_rows(obs):
    return obs[1:][obs[1:, 0] > 0.5]


def lane_number(obs):
    return int(np.clip(np.rint(obs[0, 2] / LANE_WIDTH_M), 0, LANES - 1))


def in_lane_or_entering(row, lane):
    """Include cars crossing toward a lane using observed lateral velocity."""
    center = lane * LANE_WIDTH_M
    now = float(row[2] - center)
    future = now + float(row[4]) * CUT_IN_LOOKAHEAD_S
    return abs(now) < 2.5 or abs(future) < 2.5 or now * future < 0


@dataclass
class LaneAssessment:
    safe: bool = True
    front_gap: float = float("inf")
    front_speed: float = TARGET_SPEED_MPS


def assess_lane(obs, lane):
    """Check front AND rear clearance, relative speeds, and projected overlap."""
    result = LaneAssessment(safe=0 <= lane < LANES)
    if not result.safe:
        return result
    ego = obs[0]
    for row in traffic_rows(obs):
        if not in_lane_or_entering(row, lane):
            continue
        dx = float(row[1] - ego[1])
        relative_vx = float(row[3] - ego[3])
        gap = abs(dx) - CAR_LENGTH_M
        if dx >= 0:
            if gap < result.front_gap:
                result.front_gap, result.front_speed = gap, float(row[3])
            closing = max(0.0, -relative_vx)
            required = max(FRONT_GAP_M, HEADWAY_S * float(ego[3]))
        else:
            closing = max(0.0, relative_vx)
            required = REAR_GAP_M + 0.7 * max(float(row[3]), 0.0)
        projected_gap = gap - closing * LANE_CHANGE_HORIZON_S
        if gap < required or projected_gap < 8.0 or (closing > 0 and gap / closing < TTC_S):
            result.safe = False
    return result


def danger(obs, gap_m=FRONT_GAP_M, ttc_s=TTC_S, cut_in_s=CUT_IN_LOOKAHEAD_S):
    """Detect immediate front/cut-in danger from positions and velocities only."""
    ego = obs[0]
    for row in traffic_rows(obs):
        dx, dy = float(row[1] - ego[1]), float(row[2] - ego[2])
        lateral_closing = -np.sign(dy) * float(row[4] - ego[4])
        lateral_overlap = abs(dy) < CAR_WIDTH_M + 0.5
        approaching = lateral_closing > 0 and abs(dy) - lateral_closing * cut_in_s < CAR_WIDTH_M + 0.5
        if not (lateral_overlap or approaching):
            continue
        if abs(dx) < CAR_LENGTH_M + 3.0:
            return True
        closing = float(ego[3] - row[3])
        gap = dx - CAR_LENGTH_M
        if dx > 0 and (gap < gap_m or (closing > 0 and gap / closing < ttc_s)):
            return True
    return False


def reactive_action(obs):
    """Prefer a clear adjacent lane when impeded; otherwise request SLOWER."""
    lane = lane_number(obs)
    current = assess_lane(obs, lane)
    urgent = danger(obs)
    impeded = current.front_gap < max(FRONT_GAP_M, HEADWAY_S * float(obs[0, 3]))
    # During a lane change keep the target already commanded by the actuator;
    # a new lane request would otherwise skip lanes. Still brake if danger grows.
    changing = abs(float(obs[0, 2]) - lane * LANE_WIDTH_M) > 0.3 or abs(float(obs[0, 4])) > 0.2
    if changing:
        return SLOWER if urgent else IDLE
    if urgent or impeded:
        choices = [(candidate, assess_lane(obs, candidate)) for candidate in (lane - 1, lane + 1)]
        safe = [(candidate, assessment) for candidate, assessment in choices if assessment.safe]
        if safe:
            best, _ = max(safe, key=lambda item: (item[1].front_gap, item[1].front_speed))
            return LANE_LEFT if best < lane else LANE_RIGHT
        return SLOWER
    return FASTER if float(np.linalg.norm(obs[0, 3:5])) < TARGET_SPEED_MPS - 0.5 else IDLE


def create_controller(seed):
    return reactive_action
