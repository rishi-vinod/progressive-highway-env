"""Optional observation-only action checks around a learned driving policy.

This is a rule-based guard, not learning. It does not inspect the environment,
hazard schedule, vehicle internals, or future simulator states.
"""

from reactive import (assess_lane, danger, lane_number, LANE_WIDTH_M,
                      LANE_LEFT, LANE_RIGHT, IDLE, SLOWER)

CENTER_TOLERANCE_M = 0.3
LATERAL_SPEED_TOLERANCE_MPS = 0.2


def checked_action(observation, requested):
    """Return an executed action and a reason; clear requests pass through."""
    requested = int(requested)
    if requested not in range(5):
        raise ValueError("Policy action must be one of the five discrete actions")
    lane = lane_number(observation)
    threatened = danger(observation)
    if requested in (LANE_LEFT, LANE_RIGHT):
        # Discrete lane requests are relative to the actuator's target lane.
        # Reject another request while lateral motion is still in progress.
        changing = (abs(float(observation[0, 2]) - lane * LANE_WIDTH_M) > CENTER_TOLERANCE_M
                    or abs(float(observation[0, 4])) > LATERAL_SPEED_TOLERANCE_MPS)
        if changing:
            return (SLOWER if threatened else IDLE), "lane_change_in_progress"
        target = lane + (-1 if requested == LANE_LEFT else 1)
        if not assess_lane(observation, target).safe:
            return SLOWER, "target_lane_not_clear"
    elif threatened and requested != SLOWER:
        return SLOWER, "current_lane_danger"
    return requested, None


class GuardedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.decision_info = None

    def __call__(self, observation):
        # Give the model its own copy so it cannot change the guard's inputs.
        requested = int(self.policy(observation.copy()))
        executed, reason = checked_action(observation, requested)
        self.decision_info = {"requested_action": requested, "executed_action": executed,
                              "guard_intervened": executed != requested, "guard_reason": reason}
        return executed
