"""Observation-only cruising styles, underneath the shared emergency controller."""

from dataclasses import dataclass, asdict
import math
import statistics

from reactive import (TARGET_SPEED_MPS, IDLE, SLOWER, FASTER, LANE_LEFT,
                      LANE_RIGHT, assess_lane, lane_number, traffic_rows,
                      in_lane_or_entering, CAR_LENGTH_M, LANES)


@dataclass(frozen=True)
class DrivingStyle:
    speed_above_flow_mps: float
    pass_delay_s: float
    lane_change_cooldown_s: float
    minimum_pass_gain_mps: float
    pass_range_m: float
    headway_s: float
    comfort_deceleration_mps2: float


# Rage seeks progress aggressively; none of these settings relax emergency
# physics or the shared front/rear lane-clearance checks.
STYLE_CONFIG = {
    'Calm': DrivingStyle(0.0, 6.0, 20.0, 3.0, 80.0, 1.2, 2.0),
    'Quick': DrivingStyle(2.5, 3.0, 10.0, 1.0, 110.0, 1.0, 2.5),
    'Rage': DrivingStyle(TARGET_SPEED_MPS, 0.0, 1.5, 0.5, 150.0, 0.8, 3.0),
}
FLOW_RANGE_M = 100.0
EMPTY_ROAD_SPEED_MPS = 25.0
MINIMUM_FOLLOW_GAP_M = 10.0
SPEED_TOLERANCE_MPS = 1.25  # Half the actuator's 2.5 m/s target increment.
CRUISE_ACCELERATION_MPS2 = 3.0
GAP_RESTORE_SPEED_MPS = 2.0
PASS_HORIZON_S = 5.0
RAGE_FRONT_HEADWAY_S = 0.6
RAGE_REAR_HEADWAY_S = 0.4
MERGE_HORIZON_S = 2.5
MERGE_MIN_GAP_M = 8.0
MERGE_PROJECTED_GAP_M = 5.0


def follow_speed(front, cruise, speed, settings):
    """Match traffic, with a bounded correction when restoring headway."""
    if not math.isfinite(front.front_gap):
        return cruise
    gap_target = max(MINIMUM_FOLLOW_GAP_M, settings.headway_s * speed)
    remaining = front.front_gap - gap_target
    allowance = math.sqrt(2 * settings.comfort_deceleration_mps2 * max(0.0, remaining))
    correction = max(-GAP_RESTORE_SPEED_MPS, min(0.0, remaining) / settings.headway_s)
    return min(cruise, max(0.0, front.front_speed + allowance + correction))


def passing_lane_clear(obs, target, style):
    """Rage accepts tighter moving gaps, never side overlap or a fast rear catch."""
    if style != 'Rage':
        return assess_lane(obs, target).safe
    if not 0 <= target < LANES:
        return False
    ego = obs[0]
    for row in traffic_rows(obs):
        if not in_lane_or_entering(row, target):
            continue
        dx = float(row[1] - ego[1])
        gap = abs(dx) - CAR_LENGTH_M
        closing = max(0.0, float(ego[3] - row[3]) if dx >= 0 else float(row[3] - ego[3]))
        headway = RAGE_FRONT_HEADWAY_S if dx >= 0 else RAGE_REAR_HEADWAY_S
        speed = max(0.0, float(ego[3] if dx >= 0 else row[3]))
        if (gap < max(MERGE_MIN_GAP_M, headway * speed)
                or gap - closing * MERGE_HORIZON_S < MERGE_PROJECTED_GAP_M):
            return False
    return True


class AutonomousPlanner:
    def __init__(self, policy_dt):
        self.policy_dt = policy_dt
        self.cooldown = 0.0
        self.blocked_seconds = 0.0
        self.merge_lane = None
        self.info = {}

    def choose(self, obs, style, stable_lane, merge_clear):
        settings = STYLE_CONFIG[style]
        lane = lane_number(obs)
        speed = max(0.0, float(obs[0, 3]))
        front = assess_lane(obs, lane)
        speeds = [max(0.0, float(row[3])) for row in traffic_rows(obs)
                  if abs(float(row[1] - obs[0, 1])) <= FLOW_RANGE_M]
        flow = statistics.median(speeds) if speeds else EMPTY_ROAD_SPEED_MPS
        cruise = min(TARGET_SPEED_MPS, flow + settings.speed_above_flow_mps)
        centered = stable_lane(obs, lane)
        self.cooldown = max(0.0, self.cooldown - self.policy_dt)
        if self.merge_lane == lane and centered:
            self.merge_lane = None

        desired = follow_speed(front, cruise, speed, settings)
        # Look ahead to where this lane will constrain progress; a farther car
        # at the SAME speed can still make an adjacent lane a useful passing gap.
        current_progress = min(cruise, front.front_speed + max(0.0, front.front_gap - MINIMUM_FOLLOW_GAP_M) / PASS_HORIZON_S)
        impeded = (front.front_gap < settings.pass_range_m
                   and cruise - min(desired, current_progress) >= settings.minimum_pass_gain_mps)
        self.blocked_seconds = self.blocked_seconds + self.policy_dt if impeded else 0.0

        action, reason = IDLE, 'cruise'
        if (impeded and self.blocked_seconds + 1e-9 >= settings.pass_delay_s
                and self.cooldown <= 0 and centered and self.merge_lane is None):
            options = []
            for target in (lane - 1, lane + 1):
                other = assess_lane(obs, target)
                progress = min(cruise, other.front_speed + max(0.0, other.front_gap - MINIMUM_FOLLOW_GAP_M) / PASS_HORIZON_S)
                if (passing_lane_clear(obs, target, style)
                        and progress >= min(desired, current_progress) + settings.minimum_pass_gain_mps
                        and merge_clear(obs, target, speed, self.policy_dt)):
                    options.append((progress, other.front_gap, target))
            if options:
                self.merge_lane = max(options)[2]
                self.cooldown = settings.lane_change_cooldown_s
                self.blocked_seconds = 0.0
                action = LANE_LEFT if self.merge_lane < lane else LANE_RIGHT
                reason = 'pass_slower_traffic'

        if action == IDLE:
            if self.merge_lane is not None and not passing_lane_clear(obs, self.merge_lane, style):
                action, reason = SLOWER, 'merge_clearance_changed'
            elif speed > desired + SPEED_TOLERANCE_MPS:
                action, reason = SLOWER, 'follow_traffic'
            elif speed < desired - SPEED_TOLERANCE_MPS and self.merge_lane is None and centered:
                action, reason = FASTER, 'accelerate_to_style_speed'
        if self.merge_lane is not None:
            # During a pass, follow the destination lane rather than continuing
            # to slow for the car we are leaving. The shared emergency guard
            # still checks the actual occupied corridor on every decision.
            desired = min(speed, follow_speed(assess_lane(obs, self.merge_lane), cruise, speed, settings))
            if not passing_lane_clear(obs, self.merge_lane, style):
                desired = min(desired, max(0.0, speed - GAP_RESTORE_SPEED_MPS))
        self.info = dict(style=style, style_config=asdict(settings), flow_speed_mps=flow,
                         cruise_speed_mps=cruise, desired_speed_mps=desired,
                         requested_action=action, reason=reason,
                         cooldown_s=self.cooldown, blocked_seconds=self.blocked_seconds)
        return action
