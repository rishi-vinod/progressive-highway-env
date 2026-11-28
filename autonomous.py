"""Observation-only cruising styles, underneath the shared emergency controller."""

from dataclasses import dataclass, asdict
import math
import statistics

from reactive import (TARGET_SPEED_MPS, IDLE, SLOWER, FASTER, LANE_LEFT,
                      LANE_RIGHT, assess_lane, lane_number, traffic_rows)


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
    'Calm': DrivingStyle(0.0, 8.0, 20.0, 5.0, 70.0, 1.8, 2.0),
    'Quick': DrivingStyle(2.5, 4.0, 12.0, 2.5, 90.0, 1.5, 2.5),
    'Rage': DrivingStyle(TARGET_SPEED_MPS, 0.0, 2.0, 0.5, 120.0, 1.2, 3.0),
}
FLOW_RANGE_M = 100.0
EMPTY_ROAD_SPEED_MPS = 25.0
MINIMUM_FOLLOW_GAP_M = 10.0
SPEED_TOLERANCE_MPS = 1.25  # Half the actuator's 2.5 m/s target increment.
PASS_GAP_GAIN_M = 12.0


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

        impeded = (front.front_gap < settings.pass_range_m
                   and cruise - front.front_speed >= settings.minimum_pass_gain_mps)
        self.blocked_seconds = self.blocked_seconds + self.policy_dt if impeded else 0.0
        desired = cruise
        following_gap = max(MINIMUM_FOLLOW_GAP_M, settings.headway_s * speed)
        if math.isfinite(front.front_gap):
            # Approach at a closing speed removable by comfortable braking,
            # then match the leader at the desired headway. No 75% speed latch.
            remaining = front.front_gap - following_gap
            allowance = math.sqrt(2 * settings.comfort_deceleration_mps2 * max(0.0, remaining))
            desired = min(cruise, max(0.0, front.front_speed + allowance + min(0.0, remaining) / settings.headway_s))

        action, reason = IDLE, 'cruise'
        if (impeded and self.blocked_seconds + 1e-9 >= settings.pass_delay_s
                and self.cooldown <= 0 and centered and self.merge_lane is None):
            options = []
            for target in (lane - 1, lane + 1):
                other = assess_lane(obs, target)
                if (other.safe and other.front_gap >= front.front_gap + PASS_GAP_GAIN_M
                        and other.front_speed >= front.front_speed + settings.minimum_pass_gain_mps
                        and merge_clear(obs, target, speed, self.policy_dt)):
                    options.append((min(cruise, other.front_speed), other.front_gap, target))
            if options:
                self.merge_lane = max(options)[2]
                self.cooldown = settings.lane_change_cooldown_s
                self.blocked_seconds = 0.0
                action = LANE_LEFT if self.merge_lane < lane else LANE_RIGHT
                reason = 'pass_slower_traffic'

        if action == IDLE:
            if self.merge_lane is not None and not assess_lane(obs, self.merge_lane).safe:
                action, reason = SLOWER, 'merge_clearance_changed'
            elif speed > desired + SPEED_TOLERANCE_MPS:
                action, reason = SLOWER, 'follow_traffic'
            elif speed < desired - SPEED_TOLERANCE_MPS and self.merge_lane is None and centered:
                action, reason = FASTER, 'accelerate_to_style_speed'
        self.info = dict(style=style, style_config=asdict(settings), flow_speed_mps=flow,
                         cruise_speed_mps=cruise, desired_speed_mps=desired,
                         requested_action=action, reason=reason,
                         cooldown_s=self.cooldown, blocked_seconds=self.blocked_seconds)
        return action
