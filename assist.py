"""Stage 3.5 observation-only driving and transfer of control."""

from reactive import CAR_LENGTH_M, CAR_WIDTH_M, LANE_WIDTH_M, IDLE, traffic_rows
from last_moment import LastMomentController, INTERVENTION_CONFIG
from autonomous import AutonomousPlanner

SPEED_STEP_MPS = 2.5
ASSIST_TARGET_SPEEDS = [i * SPEED_STEP_MPS for i in range(13)]
OVERRIDE_HORIZON_S = 1.0
STYLES = ("Calm", "Quick", "Rage")


def overlap_interval(distance, velocity, extent):
    """Times when two car footprints overlap along one axis."""
    if abs(velocity) < 1e-6:
        return (0, OVERRIDE_HORIZON_S) if abs(distance) <= extent else (1, 0)
    a = (-extent - distance) / velocity
    b = (extent - distance) / velocity
    return min(a, b), max(a, b)


def stable_lane(obs, lane):
    return abs(float(obs[0, 2]) - lane * LANE_WIDTH_M) < 0.3 and abs(float(obs[0, 4])) < 0.25


def merge_clear(obs, target_lane, boost_speed, delay, maneuver_age=0.0):
    """Sweep the ego footprint between predicted positions for 2.5 seconds.

    Traffic keeps its observed velocity. Allow one decision for steering
    before a gradual speed increase. Reserve extra front/rear clearance.
    """
    _, x, y, speed, _ = map(float, obs[0])
    traffic = traffic_rows(obs)
    px = py = 0.0
    dt = 0.125
    def fraction(t):
        u = min(t / 1.5, 1.0)
        return u * u * (3 - 2 * u)

    initial = fraction(maneuver_age)
    for tick in range(1, 21):
        t = tick * dt
        progress = (fraction(t + maneuver_age) - initial) / max(1e-6, 1 - initial)
        lateral = (target_lane * LANE_WIDTH_M - y) * progress
        # Integrate a bounded 2 m/s² acceleration up to the proposed speed.
        elapsed = max(0.0, t - delay)
        rise = min(elapsed, max(0.0, boost_speed - speed) / 2)
        longitudinal = speed * t + rise * (2 * elapsed - rise)
        for _, tx, ty, tvx, tvy in traffic:
            dx = float(tx) - x + float(tvx) * (t - dt) - px
            dy = float(ty) - y + float(tvy) * (t - dt) - py
            x0, x1 = overlap_interval(dx, float(tvx) - (longitudinal - px) / dt, CAR_LENGTH_M + 2)
            y0, y1 = overlap_interval(dy, float(tvy) - (lateral - py) / dt, CAR_WIDTH_M + 0.3)
            if max(0, x0, y0) <= min(dt, x1, y1):
                return False
        px, py = longitudinal, lateral
    return True


class AssistController(LastMomentController):
    def __init__(self, enabled=True, policy_dt=1.0, style="Calm", config=INTERVENTION_CONFIG):
        if style not in STYLES:
            raise ValueError(f"Unknown driving style {style}")
        self.style = style
        self.enabled = enabled  # Automatic rescue permission (--assist on/off).
        self.policy_dt = policy_dt
        self.mode = 'manual'
        self.takeovers = 0
        self.label = 'human'
        self.init_rescue(config, policy_dt)
        self.autonomous = False
        self.planner = AutonomousPlanner(policy_dt)

    @property
    def active(self):
        return self.mode != 'manual'

    def toggle(self):
        # Emergency authority is released by the hysteresis/ramp, not a toggle.
        if self.mode in ('manual', 'auto'):
            self.mode = 'manual' if self.mode == 'auto' else 'auto'
            self.autonomous = self.mode == 'auto'
            self.planner = AutonomousPlanner(self.policy_dt)

    def choose(self, observation, human_action, *, acknowledge=False):
        if not self.autonomous:
            return self.choose_rescue(observation, human_action)
        request = (self.planner.choose(observation, self.style, stable_lane, merge_clear)
                   if self.mode in ('auto', 'blending') else IDLE)
        action = self.choose_rescue(observation, request, autonomous=True)
        if self.mode == 'override':
            # An emergency may choose a different lane. Don't retain a stale
            # cruise maneuver or keep issuing relative lane-change requests.
            self.planner.merge_lane = None
        self.decision_info.update(selected_style=self.style, autonomous=True,
                                  planner=dict(self.planner.info))
        self.label = 'assist'
        return action
