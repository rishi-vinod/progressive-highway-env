"""Stage 3.5 hazard takeover. Rules only; the detector cannot access schedules."""

from reactive import danger, reactive_action, assess_lane, lane_number, LANE_LEFT, LANE_RIGHT, SLOWER

# All detector/handback thresholds are declared here, in metres and seconds.
DANGER_GAP_M = 16.0
DANGER_TTC_S = 4.0
CUT_IN_LOOKAHEAD_S = 2.0
SAFE_SECONDS_BEFORE_HANDBACK = 3.0


class AssistController:
    def __init__(self, enabled=True, policy_dt=1.0):
        self.enabled = enabled
        self.policy_dt = policy_dt
        self.active = False
        self.safe_seconds = 0.0
        self.takeovers = 0
        self.label = "human"

    def choose(self, observation, human_action):
        threatened = danger(observation, DANGER_GAP_M, DANGER_TTC_S, CUT_IN_LOOKAHEAD_S)
        if not self.enabled:
            self.active = False
            self.safe_seconds = 0.0
        elif threatened:
            if not self.active:
                self.takeovers += 1
            self.active = True
            self.safe_seconds = 0.0
        elif self.active:
            self.safe_seconds += self.policy_dt
            if self.safe_seconds >= SAFE_SECONDS_BEFORE_HANDBACK:
                self.active = False
        self.label = "assist" if self.active else "human"
        if not self.active:
            return int(human_action)
        action = reactive_action(observation)
        if action in (LANE_LEFT, LANE_RIGHT):
            target = lane_number(observation) + (-1 if action == LANE_LEFT else 1)
            if not assess_lane(observation, target).safe:
                action = SLOWER
        return action
