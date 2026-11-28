"""Observation-only emergency intervention; SI units and simulation-time timers."""

from dataclasses import asdict, dataclass
import math

from reactive import (CAR_LENGTH_M, CAR_WIDTH_M, LANE_WIDTH_M, LANES,
                      IDLE, SLOWER, LANE_LEFT, LANE_RIGHT, assess_lane,
                      lane_number, traffic_rows)


@dataclass(frozen=True)
class InterventionConfig:
    """Tune emergency thresholds here; saved with every decision for auditing."""

    alpha: float = 0.9
    a_max: float = 8.0                 # Maximum emergency braking, m/s².
    a_lat_max: float = 4.0             # Lateral acceleration estimate, m/s².
    t_margin: float = 0.2              # Steering response allowance, seconds.
    TTC_release: float = 3.0
    release_brake_fraction: float = 0.3
    safe_hold_s: float = 0.5
    handback_ramp_s: float = 0.5
    lateral_clearance_m: float = 0.3
    centered_tolerance_m: float = 0.3
    lateral_speed_tolerance_mps: float = 0.25

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Intervention thresholds must be finite and positive")
        if not 0 < self.release_brake_fraction < self.alpha <= 1:
            raise ValueError("Require 0 < release_brake_fraction < alpha <= 1")


INTERVENTION_CONFIG = InterventionConfig()


def finite(value):
    """JSON null denotes infinity (no finite collision time / overlap)."""
    return value if math.isfinite(value) else None


def object_risks(obs, request, config):
    """Compute every ahead object's bumper gap and constant-velocity risk.

    Parallel traffic is logged but only considered threatening if it overlaps
    our corridor now, at longitudinal contact, or in a requested target lane.
    Zero/negative gaps with closing motion are already beyond braking recovery.
    """
    ego = obs[0]
    requested_lane = lane_number(obs) + (-1 if request == LANE_LEFT else 1)
    risks = []
    for index, row in enumerate(traffic_rows(obs), 1):
        dx = float(row[1] - ego[1])
        if dx < 0:
            continue
        gap = dx - CAR_LENGTH_M
        closing = float(ego[3] - row[3])
        ttc = max(0.0, gap) / closing if closing > 0 else math.inf
        required = (closing ** 2 / (2 * gap) if gap > 0 else math.inf) if closing > 0 else 0.0
        dy = float(row[2] - ego[2])
        extent = CAR_WIDTH_M + config.lateral_clearance_m
        at_contact = dy + float(row[4] - ego[4]) * ttc if math.isfinite(ttc) else dy
        relevant = abs(dy) <= extent or abs(at_contact) <= extent or dy * at_contact < 0
        if request in (LANE_LEFT, LANE_RIGHT) and 0 <= requested_lane < LANES:
            relevant |= abs(float(row[2]) - requested_lane * LANE_WIDTH_M) <= extent
        clearance = max(0.0, extent - min(abs(dy), abs(at_contact)))
        if dy * at_contact < 0:
            clearance = extent
        t_steer = math.sqrt(2 * clearance / config.a_lat_max)
        risks.append(dict(object_row=index, gap_m=gap, closing_speed_mps=closing,
                          ttc=ttc, a_req=required, t_steer=t_steer,
                          dy_clear_m=clearance, relevant=bool(relevant)))
    return risks


class LastMomentController:
    """Shared emergency physics for human driving and every autonomous style."""

    def init_rescue(self, config, policy_dt):
        if not math.isfinite(policy_dt) or policy_dt <= 0:
            raise ValueError("policy_dt must be finite and positive")
        self.config = config
        self.time_s = 0.0
        self.safe_since = None
        self.blend_since = None
        self.rescue_lane = None
        self.rescue_acceleration = 0.0
        self.interventions = []
        self.decision_info = None

    def choose_rescue(self, obs, human_action, *, autonomous=False):
        c = self.config
        now = self.time_s
        self.time_s += self.policy_dt
        risks = object_risks(obs, human_action, c)
        relevant = [r for r in risks if r['relevant']]
        ttc = min((r['ttc'] for r in relevant), default=math.inf)
        a_req = max((r['a_req'] for r in relevant), default=0.0)
        brake_trigger = a_req >= c.alpha * c.a_max
        steer_trigger = any(r['closing_speed_mps'] > 0 and
                            r['ttc'] <= r['t_steer'] + c.t_margin for r in relevant)
        triggered = brake_trigger or steer_trigger
        self.contact_s = finite(ttc)
        lane = lane_number(obs)
        centered = (abs(float(obs[0, 2]) - lane * LANE_WIDTH_M) < c.centered_tolerance_m
                    and abs(float(obs[0, 4])) < c.lateral_speed_tolerance_mps)
        event = None
        if self.mode in ('manual', 'auto') and (self.enabled or autonomous) and triggered:
            self.mode = 'override'
            self.takeovers += 1
            self.safe_since = None
            self.rescue_lane = lane
            self.interventions.append(dict(id=self.takeovers, start_time_s=now,
                                           TTC_s=finite(ttc), a_req_mps2=finite(a_req),
                                           braking_impossible=a_req > c.a_max,
                                           maneuver=None, handback_time_s=None))
            event = 'intervention_started'

        action = int(human_action)
        control = None
        maneuver = 'driver'
        if self.mode in ('override', 'blending'):
            # Complete the rescue before starting handback. Once blending,
            # allow the driver to begin a safe lane change without fighting it.
            settled = self.mode == 'blending' or (centered and lane == self.rescue_lane)
            safe = (ttc > c.TTC_release and a_req < c.release_brake_fraction * c.a_max
                    and settled)
            if not safe:
                if self.mode == 'blending':
                    event = 'handback_cancelled'
                self.mode = 'override'
                self.safe_since = self.blend_since = None
            elif self.safe_since is None:
                self.safe_since = now
            elif self.mode == 'override' and now - self.safe_since + 1e-9 >= c.safe_hold_s:
                self.mode = 'blending'
                self.blend_since = now
                event = 'handback_started'

            if self.mode == 'blending' and now - self.blend_since + 1e-9 >= c.handback_ramp_s:
                self.mode = 'auto' if autonomous else 'manual'
                self.interventions[-1]['handback_time_s'] = now
                event = 'handback_completed'
                self.rescue_lane = None
            else:
                if self.mode == 'override':
                    # Prefer an open escape when braking cannot remove closing
                    # speed, or when the steering deadline is reached first.
                    if triggered and centered and (a_req > c.a_max or not brake_trigger):
                        choices = []
                        for target in (lane - 1, lane + 1):
                            assessment = assess_lane(obs, target)
                            if assessment.safe:
                                choices.append((assessment.front_gap, target))
                        if choices:
                            self.rescue_lane = max(choices)[1]
                    # Max braking is real acceleration, not a discrete SLOWER
                    # target step. Also brake during escape to reduce impact risk.
                    self.rescue_acceleration = -c.a_max if any(
                        r['closing_speed_mps'] > 0 for r in relevant) else 0.0
                maneuver = ('steer_left' if self.rescue_lane < lane else 'steer_right'
                            if self.rescue_lane > lane else 'brake' if self.rescue_acceleration < 0 else 'hold')
                action = (LANE_LEFT if self.rescue_lane < lane else LANE_RIGHT
                          if self.rescue_lane > lane else SLOWER if self.rescue_acceleration < 0 else IDLE)
                weight = 1.0 if self.mode == 'override' else max(0.0, 1 - (now - self.blend_since) / c.handback_ramp_s)
                end_weight = weight if self.mode == 'override' else max(0.0, weight - self.policy_dt / c.handback_ramp_s)
                control = dict(weight=weight, end_weight=end_weight, duration_s=self.policy_dt,
                               acceleration=self.rescue_acceleration, target_lane=self.rescue_lane,
                               a_max=c.a_max, a_lat_max=c.a_lat_max,
                               human_action=int(human_action), blending=self.mode == 'blending')
                if self.interventions[-1]['maneuver'] is None:
                    self.interventions[-1]['maneuver'] = maneuver

        self.label = 'assist' if self.active else 'human'
        self.decision_info = dict(mode=self.mode, simulation_time_s=now,
                                  TTC_s=finite(ttc), a_req_mps2=finite(a_req),
                                  braking_impossible=a_req > c.a_max,
                                  brake_trigger=brake_trigger, steer_trigger=steer_trigger,
                                  maneuver=maneuver, event=event,
                                  return_control_to='autonomous' if autonomous else 'human',
                                  intervention=(dict(self.interventions[-1]) if self.interventions else None),
                                  config=asdict(c), rescue_control=control,
                                  objects=[{k: finite(v) if isinstance(v, float) else v
                                            for k, v in r.items()} for r in risks])
        return action
