"""Stage 3.5 observation-only driving and transfer of control."""

from reactive import (CAR_LENGTH_M, CAR_WIDTH_M, LANES, LANE_WIDTH_M, LANE_LEFT, LANE_RIGHT, FASTER,
                      SLOWER, IDLE, TARGET_SPEED_MPS, assess_lane, lane_number,
                      traffic_rows, in_lane_or_entering, danger)

DANGER_GAP_M = 16.0
DANGER_TTC_S = 4.0
CUT_IN_LOOKAHEAD_S = 2.0
SAFE_SECONDS_BEFORE_HANDBACK = 3.0
LANE_CHANGE_COOLDOWN_S = 4.0
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


def projected_contact(obs, request):
    """Earliest simultaneous longitudinal and lateral overlap within one second.

    Traffic coasts at its observed velocity. For a new speed request use
    the mean of current and attainable target speeds as a short projection.
    A lane request projects a one-second lateral move toward its lane center.
    """
    _, x, y, vx, vy = map(float, obs[0])
    if request in (FASTER, SLOWER):
        index = round(vx / SPEED_STEP_MPS) + (1 if request == FASTER else -1)
        target = min(TARGET_SPEED_MPS, max(0, index * SPEED_STEP_MPS))
        vx = (vx + target) / 2
    elif request in (LANE_LEFT, LANE_RIGHT):
        lane = lane_number(obs) + (-1 if request == LANE_LEFT else 1)
        if 0 <= lane < LANES:
            vy = lane * LANE_WIDTH_M - y
    earliest = None
    for _, tx, ty, tvx, tvy in traffic_rows(obs):
        x0, x1 = overlap_interval(float(tx) - x, float(tvx) - vx, CAR_LENGTH_M)
        y0, y1 = overlap_interval(float(ty) - y, float(tvy) - vy, CAR_WIDTH_M)
        entry = max(0.0, x0, y0)
        if entry <= min(OVERRIDE_HORIZON_S, x1, y1):
            earliest = entry if earliest is None else min(earliest, entry)
    return earliest


def stable_lane(obs, lane):
    return abs(float(obs[0, 2]) - lane * LANE_WIDTH_M) < 0.3 and abs(float(obs[0, 4])) < 0.25


def rear_conflict(obs, lane, speed):
    for row in traffic_rows(obs):
        dx = float(obs[0, 1] - row[1])
        if dx > 0 and in_lane_or_entering(row, lane):
            closing = float(row[3]) - speed
            if closing > 0 and dx - CAR_LENGTH_M < closing * DANGER_TTC_S:
                return True
    return False


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


class AssistController:
    def __init__(self, enabled=True, policy_dt=1.0, style="Calm"):
        if style not in STYLES:
            raise ValueError(f"Unknown driving style {style}")
        self.style = style
        self.merge_lane = None
        self.merge_speed = 0.0
        self.enabled = enabled  # Automatic rescue permission (--assist on/off).
        self.policy_dt = policy_dt
        self.mode = 'manual'
        self.safe_seconds = 0.0
        self.takeovers = 0
        self.cooldown = 0.0
        self.label = 'human'

    @property
    def active(self):
        return self.mode != 'manual'

    def toggle(self):
        # A rescue requires a fresh safe arrow request to release the wheel.
        if self.mode in ('manual', 'auto'):
            self.mode = 'manual' if self.mode == 'auto' else 'auto'

    def choose(self, observation, human_action, *, acknowledge=False):
        lane = lane_number(observation)
        front = assess_lane(observation, lane)
        speed = float(observation[0, 3])
        centered = stable_lane(observation, lane)
        threatened = danger(observation, DANGER_GAP_M, DANGER_TTC_S, CUT_IN_LOOKAHEAD_S)
        closing = max(0.0, speed - front.front_speed)
        # Leave time for the discrete actuator to respond to a slowing queue.
        # Include the distance used while reducing the closing speed, plus
        # room to follow if the leader continues braking.
        follow_gap = max(DANGER_GAP_M, speed * 1.5)
        threatened |= closing > 2 and front.front_gap < follow_gap + closing * closing / 6
        self.cooldown = max(0.0, self.cooldown - self.policy_dt)
        self.contact_s = projected_contact(observation, human_action)
        imminent = self.contact_s is not None
        reject_request = False

        if self.mode == 'manual' and self.enabled and imminent:
            reject_request = human_action != IDLE and projected_contact(observation, IDLE) is None
            self.mode = 'override'
            self.safe_seconds = 0.0
            self.takeovers += 1
        elif self.mode in ('override', 'ready'):
            if imminent or not centered:
                self.mode = 'override'
                self.safe_seconds = 0.0
            elif self.mode == 'ready' and acknowledge:
                self.mode = 'manual'
            else:
                self.safe_seconds += self.policy_dt
                if self.safe_seconds + 1e-9 >= SAFE_SECONDS_BEFORE_HANDBACK:
                    self.mode = 'ready'

        self.label = 'assist' if self.active else 'human'
        if not self.active:
            self.merge_lane = None
            return int(human_action)
        if reject_request:
            return IDLE  # Continuing the current path avoids the requested conflict.

        # The simulator continues steering to its lane target during subsequent
        # speed commands. Recheck the observed corridor before each increase.
        if self.merge_lane is not None:
            self.merge_age += self.policy_dt
            if centered and lane == self.merge_lane:
                self.merge_lane = None
            elif merge_clear(observation, self.merge_lane, self.merge_speed, 0, self.merge_age):
                return FASTER if speed < self.merge_speed - SPEED_STEP_MPS / 2 else IDLE

        speeds = [float(row[3]) for row in traffic_rows(observation)]
        flow = sorted(speeds)[len(speeds) // 2] if speeds else 25.0
        style = self.style if self.mode == 'auto' else 'Calm'
        desired = min(TARGET_SPEED_MPS, flow + (0.5 if style == "Calm" else 4))
        if style == "Rage":
            desired = TARGET_SPEED_MPS
        pass_range = 45 if style == "Quick" else 75
        passing = style != "Calm" and front.front_gap < pass_range and front.front_speed < desired - 2
        if (threatened or imminent or passing) and centered and self.cooldown <= 0 and self.merge_lane is None:
            options = []
            for offset, action in ((-1, LANE_LEFT), (1, LANE_RIGHT)):
                target = lane + offset
                if not 0 <= target < LANES:
                    continue
                other = assess_lane(observation, target)
                required = min(front.front_gap, DANGER_GAP_M) if threatened else front.front_gap
                if other.front_gap <= required + 12:
                    continue
                # Ordinary passes use the standard headway checks. In a rescue
                # or Rage pass also evaluate narrower moving gaps end to end.
                if not (other.safe or imminent or style == "Rage"):
                    continue
                for goal in (speed, min(speed + SPEED_STEP_MPS, TARGET_SPEED_MPS)):
                    if merge_clear(observation, target, goal, self.policy_dt):
                        options.append((other.front_gap, action, target, goal))
                        break
            if options:
                _, action, self.merge_lane, self.merge_speed = max(options)
                self.merge_age = 0.0
                self.cooldown = 2.5 if style == "Rage" else LANE_CHANGE_COOLDOWN_S
                return action

        # Approach the leader's speed in small target increments. Only ask
        # for extra clearance when bumpers are already dangerously close.
        if front.front_gap < follow_gap + (closing + SPEED_STEP_MPS) ** 2 / 6:
            desired = min(desired, front.front_speed)
            if front.front_gap < CAR_LENGTH_M:
                desired = max(0, desired - (CAR_LENGTH_M - front.front_gap))
        if speed > desired + SPEED_STEP_MPS / 2:
            if threatened or not rear_conflict(observation, lane, speed - SPEED_STEP_MPS):
                return SLOWER
        if centered and speed < desired - SPEED_STEP_MPS / 2 and not threatened:
            return FASTER
        return IDLE
