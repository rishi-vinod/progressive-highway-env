# Progressive highway driving

A local, free, open-source driving simulation built one stage at a time on macOS.
**Stages 0-5, including Stage 3.5, are implemented.**
Parts 1-2 and Stages 1-4 (including 3.5) do **not** learn. Only Stage 5 learns.
No accounts, API keys, paid services, model downloads, or runtime network calls.

## Run the guarded learned demo

To see the trained car with the new guard, run these commands from the project
folder using the saved model on this machine:

```sh
source .venv/bin/activate
python 05_train_policy.py drive runs/stage5_20261103T201030071303Z_09dbd8ed/policy.zip --guard --seed 46
```

Use `source .venv/bin/activate`, rather than executing `.venv/bin/activate` by
itself, to activate Python in your current terminal. The demo opens a pygame
window. The terminal prints `RULE-BASED GUARD ENABLED`,
`controller=learned_guarded`, and `GUARD ACTIVE` when a request is overridden.
Seed 46 exercises the guard and completed all 40 simulated seconds in headless
verification. Close the window or press Ctrl+C to stop.

The command prints `Display mode: WINDOWED | offscreen_rendering=False |
real_time_rendering=True` before driving. Saved training configs contain
`offscreen_rendering=True`; the scenario factory now overrides saved display
flags to match the requested mode. This fixes the earlier bug where the guarded
demo printed steps but never opened a window. Windowed playback is paced in real
time; `--headless` remains window-free and runs without display pacing.

`00_random_drive.py` is the original Stage 0 random-action demo. It does not load
the trained model or enable the guard. Increasing `--max-steps` to 1000 does not
change its controller or extend the environment's 40-second episode duration;
a collision can end it sooner. For keyboard driving with takeover, use
`python 03_5_takeover.py --seed 46 --assist on` instead (rules, no learning).

The model path above refers to a local artifact ignored by Git. On a fresh
checkout, follow Stage 5 below to train locally and use the model path it prints.

## Setup (Part 1)

Run commands from this repository's root. Python 3.12 is required for this setup;
the intended project range is 3.9-3.12, but these dependency pins were verified
only on Python 3.12.15. Do not assume this lock works on older Python versions.

For a fresh setup (skip venv creation if `.venv` already exists):

```sh
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
which python
python --version
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

Before installing, check that `which python` points into this project's `.venv`.
The requirements file was generated with `pip freeze`, including automatic
dependencies. Pip itself is listed below but is excluded by ordinary `pip freeze`.
Only PyPI is used for installation. The `--no-cache-dir` option avoids writing a
pip download cache outside the project.

Base libraries for Stages 0-4: highway-env, gymnasium, numpy,
pygame-ce, and matplotlib (plus their automatically installed dependencies).
The pinned versions are highway-env 1.12.1, gymnasium 1.4.0, numpy 2.5.4,
pygame-ce 2.5.8, and matplotlib 3.11.2.

Base-environment dependencies (the complete post-Stage-5 inventory is below):

```text
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
Farama-Notifications==0.0.6
fonttools==4.67.0
kiwisolver==1.5.1
packaging==26.3
pandas==3.0.6
pillow==13.0.0
pip==26.2.1
pyparsing==3.3.3
python-dateutil==2.9.0.post0
six==1.17.0
typing_extensions==4.16.0
```

This highway-env version directly requires pygame-ce as well as pandas.
Use only the pygame-ce distribution; the Python import remains `import pygame`.
Do not install the separate pygame distribution alongside pygame-ce because they
share the same import namespace and conflict. `requirements.txt` includes only
pygame-ce for this namespace. The user replaced pygame with pygame-ce in the
virtual environment. Stages 0-4 use these base packages. Stage 5 additionally uses
the separately approved CPU training dependencies in `requirements-stage5.txt`.

## Random driving (Part 2)

One command to verify the installation without a window:

```sh
source .venv/bin/activate && python 00_random_drive.py --headless --seed 0 --max-steps 5
```

Run the windowed version yourself:

```sh
source .venv/bin/activate
python 00_random_drive.py --seed 0 --max-steps 100
```

Working means a highway window with the ego car and other traffic, plus terminal
lines containing step, action number/name, reward, and collision flag. The ego
car takes random actions and may crash: this is expected. The program stops at
a collision, the stock 40-second episode limit, or the requested step limit,
whichever comes first. Close the window or press Ctrl+C to stop early; cleanup
runs in `finally`. Windowed behavior is tested by the user, not the assistant.

`--headless` selects `render_mode=None` and never calls render. The default is
`render_mode="human"`. `--seed` defaults to 0 and seeds both reset and action
sampling. Repeat the same command with the pinned environment for repeatable
traffic, actions, and rewards. `--max-steps` must be positive. Matplotlib's local
cache is placed under ignored `runs/matplotlib`; Fontconfig's cache is directed
to ignored `runs/cache`.

The stock action mapping is 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER.
Stock traffic contains 50 other cars. The script prints the actual mapping and
traffic count rather than assuming them silently. Reward is only printed;
there is no training or parameter update.

### Installed API verification

Read locally, without web access or library edits:

- `.venv/lib/python3.12/site-packages/highway_env-1.12.1.dist-info/METADATA`,
  Usage section: `import highway_env`, then `gym.register_envs(highway_env)`.
- `highway_env/__init__.py` inside site-packages: import registers highway-v0.
- `gymnasium/envs/registration.py`: `register_envs` is an explicit no-op marker.
- `highway_env/envs/highway_env.py`: default DiscreteMetaAction, 50 other vehicles,
  40-second time limit, and collision termination.
- `highway_env/envs/common/action.py`: the five discrete actions above.
- `highway_env/envs/common/abstract.py`: reset recreates action spaces;
  `info['crashed']` reports collision; close sets `done` and closes the viewer.
- `highway_env/envs/common/graphics.py`: the window close event calls env.close().
- `farama_notifications/__init__.py`: bundled static notifications, no network fetch.

## Stage 1 — stay in lane and keep speed

`01_rule_based_drive.py` is a deterministic baseline. It aims for **30 m/s
(108 km/h)** in its starting lane, with a 0.5 m/s speed tolerance. It uses only
the ego velocity from the current observation: FASTER below the target band,
SLOWER above it, and IDLE inside it. IDLE retains the library's lane/speed target;
it does not stop the car. There are no lane-change commands or traffic avoidance,
so crashes in traffic are expected. There is no learning.

Run with a window yourself:

```sh
source .venv/bin/activate
python 01_rule_based_drive.py --seed 0 --max-steps 40
```

Run a short headless test:

```sh
source .venv/bin/activate
python 01_rule_based_drive.py --headless --seed 0 --max-steps 5
```

Working means the ego car stays in its starting lane, accelerates from the stock
25 m/s starting speed toward 30 m/s, and prints action/reward/collision/speed each
step. It stops at a collision, the 40-second episode limit, or the step limit.
Window close and Ctrl+C clean up the environment. A completed run writes
`episode.json` with its seed, environment config, metric definitions, and episode
metrics under a uniquely named `runs/stage1_.../` directory. A window-closed
episode is labeled `window_closed`; Ctrl+C does not log an unfinished episode.

The observation configuration is explicit and printed at startup:

```json
{
  "type": "Kinematics",
  "features": ["presence", "x", "y", "vx", "vy"],
  "vehicles_count": 51,
  "absolute": true,
  "normalize": false,
  "clip": false,
  "order": "sorted",
  "see_behind": true,
  "observe_intentions": false,
  "include_obstacles": false
}
```

Row 0 is ego. All positions are world coordinates in metres; all velocities are
in m/s. Traffic rows are sorted by absolute longitudinal lane distance, which is
the installed library's meaning of `order="sorted"` (not Euclidean distance).
The observation has capacity for ego plus all 50 stock traffic vehicles, but the
stock perception radius still limits which cars appear. Empty rows have
`presence=0`. Rear traffic is included and intentions are excluded. The baseline
ignores all traffic rows; later reactive controllers may use them.

### Shared metrics and fixed-seed evaluation

Run the full 50-episode evaluation (always `render_mode=None`):

```sh
source .venv/bin/activate
python evaluate.py
```

This uses exactly seeds **0 through 49**, a fresh controller/environment per
episode, 50 stock traffic vehicles, a 40-second horizon, 15 Hz physics, and 1 Hz
controller decisions. It prints a per-episode table and an aggregate summary.
Crashes end episodes early; they count as collisions, not evaluation failures.
Do not change seeds, timing, traffic, or metric definitions when comparing stages.
The full-project action range now includes targets `[0, 5, 10, 15, 20, 25, 30]`
m/s, so braking can reach zero. Earlier Stage 1 runs used `[20, 25, 30]`; compare
new runs against the new baseline, not historical results made with that range.

For a quick check instead of the full benchmark:

```sh
python evaluate.py --smoke-test --max-steps 3
```

This runs only seeds 0 and 1 and is explicitly labeled **SMOKE TEST**. A custom
`--max-steps` can shorten evaluation; step-limited episodes are counted separately
in the summary. Shortened runs are not comparable to the default benchmark.

Per-episode definitions are shared by the demo and evaluator:

- **Collision:** whether ego's collision flag was ever true during the episode.
- **Closest distance (m):** minimum Euclidean center-to-center distance between
  ego and any other car, including the initial state and every physics tick at
  15 Hz. This is a sampled minimum, not a continuous-time interpolation or bumper
  gap. All road vehicles count, including those outside the observation radius.
- **Average speed (m/s):** trapezoidal integration of ego speed at physics ticks,
  divided by elapsed simulated time. Includes all ticks through the end of the
  terminating controller step, matching stock episode termination timing.

The summary reports collision count/rate, the minimum closest distance across
episodes, mean episode closest distance, and mean episode average speed. Each
episode receives equal weight in those means. Missing nearby cars yield a JSON
null/CSV blank distance rather than infinity (stock traffic normally prevents it).

Each evaluation creates a new `runs/evaluation_.../` directory containing a JSON
record for each finished seed, `episodes.csv` (collision yes/no), and
`summary.json` with the config, requested/completed seeds, metric definitions,
and completion status. Ctrl+C preserves completed episodes and marks the run
incomplete. Unique directories and exclusive file creation prevent overwrites.
These metrics logs are not scenario/replay files.

### Reusing the runner for future authorized controllers

`driving_common.py` holds the environment, fixed seeds, metrics, and episode loop.
The measurement environment subclasses stock `HighwayEnv` and `Road`, delegating
physics to the installed library and sampling afterward; no installed files are
edited. Ground truth is used only for metrics. The baseline gets a copy of the
observation, never the environment, vehicle objects, `info`, or metrics.

The plugin contract is `create_controller(seed) -> callable(observation) -> int`.
The seed allows future controllers to initialize local randomness. For example,
the explicit form of the current benchmark is:

```sh
python evaluate.py --controller 01_rule_based_drive:create_controller
```

When another stage is authorized, its local module can expose that same factory
and use the same evaluator. `run_episode` also accepts an environment factory
for future scenario environments that preserve the metric hooks. The loader
imports local Python code; only select trusted project controller modules.

Headless correctness checks (standard-library unittest, no extra packages):

```sh
python test_stage1.py
```

The checks cover observation-only decisions, physical units and sorting,
intermediate-tick distance minima, collision accumulation, time-averaged speed,
unchanged stock dynamics, repeatability, fixed seeds, and cleanup on controller
failure. Windowed rendering remains for the user to test.

Stage 1 API verification used installed highway-env 1.12.1 / Gymnasium 1.4.0:
`KinematicObservation` in `envs/common/observation.py`, sorting and physics in
`road/road.py`, action mappings in `envs/common/action.py`, discrete speed and
lane keeping in `vehicle/controller.py`, and reset/step/render/close hooks in
`envs/common/abstract.py`. These paths are relative to the installed
`highway_env` package under `.venv/lib/python3.12/site-packages/`.

## Stage 2 — reactive driving on stock traffic

```sh
source .venv/bin/activate
python 02_reactive_drive.py --seed 0
python 02_reactive_drive.py --headless --seed 0 --max-steps 8
python evaluate.py --controller 02_reactive_drive:create_controller
```

`reactive.py` uses only the current observation. It checks front gap, closing
speed/time-to-collision, adjacent-lane front AND rear gaps, and projected closing
over a lane change. It also includes cars visibly moving toward a lane. It
requests a clear adjacent lane when impeded, otherwise SLOWER; on a clear road it
cruises toward 30 m/s. It does not read internal vehicle intentions, event
schedules, ground truth, or learned weights. Named thresholds are at the top of
the file. Working means traffic-dependent braking/lane changes; crashes remain
possible. Discrete SLOWER reduces a speed target, not an instantaneous emergency stop.

## Stage 3 — developer-forced hazards

```sh
python 03_forced_events.py --seed 10 --hazard hard_brake
python 03_forced_events.py --seed 10 --hazard cut_in_left
python 03_forced_events.py --headless --seed 10 --max-steps 8
```

`hazards.py` subclasses the library's IDM vehicle for hard braking and cut-ins;
installed files are unchanged. With no `--hazard`, kind and trigger time come
from the seed. Initial scenarios stage one actor ahead or in a neighboring lane
at reset, maintaining 50 traffic cars and avoiding initial overlap. At a road
edge a seeded cut-in is mirrored to the available side and the resolved event
is logged. Normal traffic is unchanged. The actor switches behavior when
physics time reaches `trigger_time_s`, or when ego travel reaches
`trigger_distance_m` for distance-scheduled scenarios.

Hard braking lasts three seconds by default, with up to 6 m/s² deceleration.
A forced cut-in uses the library steering controller and may be unsafe by
design. Runtime developer commands affect an existing car without teleporting it.
Working means the same seed repeats the same initial scene and scheduled hazard.

## Stage 3.5 — you drive; assistance can take over

```sh
python 03_5_takeover.py --seed 10 --assist on
python 03_5_takeover.py --seed 10 --assist off
python 03_5_takeover.py --headless --seed 10 --assist on --max-steps 40
python benchmark.py --stage 3.5
```

In the user-run window:

- LEFT/RIGHT request a lane change.
- UP/DOWN request faster/slower speed; releasing a key retains the current target.
- A toggles assistance ON/OFF; S saves a scenario snapshot; Esc exits.
- The banner reads **ASSIST ACTIVE** when the computer has control.

The keyboard mapping feeds highway-env's manual-control mode. During assistance,
normal action control resumes and human requests cannot overwrite the rescue.
`assist.py` detects danger from the current observation only. It calls the Stage 2
rules, checks target-lane clearance ahead and behind, and brakes if blocked.
Control returns after three seconds of continuously safe detector readings.
Thresholds are named constants. No AI training occurs here, and collision
avoidance is a goal, not a guarantee.

The default window lasts at most 200 decisions / 40 simulated seconds. The
headless version uses the Stage 1 rule as a stand-in driver, and the benchmark
runs the same 50 seeds once OFF and once ON. It reports triggered hazards,
collisions, avoided percentage, secondary collisions, closest distance, average
speed, and takeovers. A hazard is counted avoided only if it triggered and ego
survived the full horizon without any collision. Early exits, step-limited runs,
and crashes with other cars do not count as successful avoidance.

Collision partners come from the simulator's collision solver, including impacts
applied on the next physics tick, and count only when ego actually crashes.
A secondary collision involves a car other than a triggered hazard actor.

## Stage 4 — local developer command panel

The user selected **pygame-only**. Gradio and huggingface_hub are not installed
or used. No server, port, account, analytics, deployment, or network request is needed.

```sh
python 04_developer_panel.py --seed 0
python 04_developer_panel.py --headless --seed 0 --max-steps 10 --command "make the car in front brake hard"
```

The simulation uses rgb_array frames in the window with a developer panel at
the side. Type a command and press Enter:

- `make the car in front brake hard` or `brake hard`
- `force a cut-in from the left lane` or `cut-in from left`
- `cut-in from right`
- `save scenario`, `pause`, `resume`, `help`

This is an allowlisted keyword/regex parser, never an LLM or arbitrary code
executor. Missing lanes or eligible cars produce a helpful rejection rather
than teleporting vehicles. The reactive controller drives the ego car. Saving
and exiting write new local scenarios; the window stays open after an episode
ends so its result can be inspected or saved. Headless `--command` inputs are
submitted before step 1; pause/resume control the interactive window only.

## Full benchmark and timing profiles

```sh
python benchmark.py
python benchmark.py --stage 2
python benchmark.py --stage 3.5
python benchmark.py --smoke-test --max-steps 5
```

The full command runs seeds 0-49 for Stages 0, 1, 2, 3, 3.5 OFF/ON, and 4:
350 episodes total. Every run uses the same three physics-tick metrics and a
40-second horizon unless terminated by a collision. Stages 0-3 use 1 Hz policy
decisions; interactive Stages 3.5/4 and their matching benchmarks use 5 Hz for
keyboard/assist responsiveness. Physics remains 15 Hz throughout. OFF/ON trials
use identical timing and seeded scenes. Do not attribute differences between
1 Hz and 5 Hz profiles solely to the controller; all configs are saved explicitly.
These frequencies are configuration choices, not measured latency improvements.

Each stage gets per-seed scenario JSON, episode CSV, and a summary under a new
`runs/benchmark_.../` directory. `--smoke-test` uses just seeds 0 and 1.
`--max-steps` explicitly caps policy decisions and marks shortened episodes.
Stage 4 benchmark uses the reactive driver without typed developer commands;
use saved command scenarios for specific panel-driven experiments.

Measured on November 3, 2026, with the pinned environment and all 50 seeds:

- Random actions: 36/50 collisions; mean episode speed 21.649 m/s.
- Stay-in-lane baseline: 50/50 collisions; 29.050 m/s.
- Reactive stock traffic: 0/50 collisions; 22.804 m/s.
- Forced hazards at 1 Hz: 9/50 collisions; 82% avoided; one secondary collision;
  22.751 m/s.
- Takeover at 5 Hz, assist OFF: 50/50 collisions, 48 hazards triggered, 0% avoided,
  16 secondary collisions; closest center distance 2.027 m; 29.262 m/s.
- Takeover at 5 Hz, assist ON: 3/50 collisions, 50 hazards triggered, **94% avoided**,
  zero secondary collisions; closest center distance 3.794 m; 22.987 m/s;
  365 takeovers across the 50 episodes.
- Stage 4 stock traffic at 5 Hz: 0/50 collisions; 22.774 m/s.

These are finite seeded experiments, not safety guarantees. The avoidance metric
requires both a triggered hazard and completion without any ego collision;
an episode ending before the hazard triggers is not an avoided hazard.
The local audit verified all 350 episodes and their aggregate collision counts:
`runs/benchmark_20261103T151818762627Z_ec2dd452/summary.json`.
Run directories are ignored by Git; these files exist only on the machine that
ran the benchmark. A fresh checkout can regenerate them with `python benchmark.py`.

## Scenario saving and verified replay

New-stage demos and `benchmark.py` save versioned JSON containing environment
config, seed, initial state, initial hazard schedule, every submitted command
with a UTC wall-clock timestamp and effective simulation step, and every policy
step's car IDs, positions, speeds, headings, collision flags, ego action, reward,
and controller identity. Takeover logs also include human requests and hazard
outcomes. Metrics sample every physics tick; full car records sample each policy
step. These are separate sampling rates.

```sh
python scenarios.py runs/SCENARIO_DIRECTORY/scenario.json
```

Replace the example path with a printed scenario path (benchmark seed JSON files
also work). Replay is headless: it resets the same seed/config, reconstructs
events and commands at the same steps, and replays the logged ego actions,
including human actions. It checks every recorded car state, reward, termination
flag, metric, and hazard outcome within a small floating-point tolerance. It
fails on mismatch. This verifies the simulation trajectory; it does not prove
that a changed controller would choose the same actions. Reproducibility assumes
the pinned dependency versions. No JSON contains executable Python.

Stage 0/1's original standalone demos retain their original log formats; run
`benchmark.py --stage 0` or `--stage 1` for the shared replayable scenario format.

## Stage 5 — advanced CPU reinforcement learning

This is the **only learning stage**. The user approved installation of
stable-baselines3 and PyTorch. Extra dependencies are pinned separately:

```sh
source .venv/bin/activate
which python
pip install --no-cache-dir -r requirements-stage5.txt
python 05_train_policy.py train --timesteps 10000
python 05_train_policy.py evaluate runs/TRAINING_DIRECTORY/policy.zip
python 05_train_policy.py drive runs/TRAINING_DIRECTORY/policy.zip --seed 0
```

For a new installation, obtain approval before the optional pip command. Replace
the model path with the path printed by training. Training starts from random
weights locally; no model download occurs. DQN uses a small network, a bounded
replay buffer, one CPU thread, and explicit `device="cpu"` for both training and
inference. TensorBoard/cloud tracking are disabled. Short runs validate the
pipeline and do not promise a better driver. `training.json` reports actual
transitions, gradient updates, whether weights changed, device, and seed split.

Training cycles seeds **1000-1049**. Checkpoints are validated on **2000-2004**
every 2,000 transitions and at the end. Selection prefers fewer collisions, then
higher average speed. The selected checkpoint becomes `policy.zip`; individual
checkpoints are retained. Evaluation uses the fixed held-out **0-49** and compares
the frozen learned policy to the reactive rule with identical metrics and config.
Validation is a small selection set, not evidence of universal safety.

Version 2 training converts the current observation into 24 features: ego motion
and left/current/right lane availability, nearest front/rear gaps, relative
speeds, and time to contact. Nearby cars entering a lane are included using their
observed lateral velocity. This reduces the moving input slots the small network
must interpret. No vehicle internals, scripted intentions, or ground-truth
metrics enter the policy. Version 1 checkpoints still load with their original
51-by-5 encoder; loading one does not improve its learned weights.

The training wrapper rewards forward speed (`0.5 * speed / 30`), subtracts risk
from short headway or time to contact, and penalizes risky lane-change requests
and small unnecessary lane-change costs. A collision always returns **-10**.
See `learning_features.py` for the formulas. These are hand-designed learning
signals. By default the neural network chooses all five actions without a
rule-based override. The optional `--guard` mode described below checks its
requests before execution and is explicitly labeled **learned policy + guard**.
Scenario/evaluation logs retain the simulator's raw
reward; training episode returns use this shaped reward and are not comparable.
Per-training-episode metrics and progress CSV stay in the local training folder.

`drive` opens a window for you to inspect a saved policy; it does not train.
Add `--headless --max-steps 5` for a window-free smoke test. Working means the
model loads, reports its selected actions, and saves a replayable scenario at
the end. Crashes remain possible. Close the window or press Ctrl+C to exit.

Additional options:

```sh
python 05_train_policy.py train --timesteps 1200 --hazards
python 05_train_policy.py train --timesteps 1200 --scenario runs/SCENARIO_DIRECTORY/scenario.json
python 05_train_policy.py evaluate runs/TRAINING_DIRECTORY/policy.zip --hazards
python 05_train_policy.py evaluate runs/TRAINING_DIRECTORY/policy.zip --smoke-test
```

By default evaluation reruns both controllers. Optional
`--reuse-reactive runs/PREVIOUS_EVALUATION_DIRECTORY/reactive` copies a historical
stock-traffic baseline only after checking every seed, exact environment config,
controller label, and completed-episode status. This avoids repeating unchanged
baseline work; the summary explicitly records its source and reuse. The learned
policy always runs fresh episodes. Reuse compares against that saved baseline,
not any subsequent edits to the reactive controller.

Scenario training recreates its seed, initial schedule, and command events on
each reset. It **does not train on logged per-step records or logged actions**.
Runtime hazard commands record their selected actor ID and target lane. If a
learned trajectory has already made that actor unavailable, training skips that
event and reports `unavailable_scheduled_events` in step info; it never resurrects
or teleports cars.
Training scenarios using held-out seeds 0-49 or validation seeds 2000-2004 are
rejected: generate a scenario with a training seed such as 1000 instead.
Validation uses independent seeded traffic (and seeded hazards with `--hazards`),
not the exact typed-event sequence of a supplied training scenario.
The optional learned takeover guardian
is not enabled; Stage 3.5 continues to use the explicit reactive rules.

### Measured Stage 5 correction (November 3, 2026)

The original 1,200-transition model crashed in **50/50 stock-traffic evaluation
episodes**. Its actions were 80.3% FASTER and 0% SLOWER. This was a poor learned
policy, not evidence that merely loading a model makes it a competent driver.

After the feature/reward changes above and 10,000 CPU training transitions
(9,744 total gradient updates), validation selected the **6,000-step checkpoint**
(5,743 updates). Checkpoints at 2,000/4,000/6,000/8,000/10,000 steps had respectively
2/1/0/1/0 collisions across the five validation seeds. The 6,000-step checkpoint
won the speed tie-break among the two crash-free validation candidates.

The selected checkpoint was then frozen and tested on all 50 evaluation seeds:

- Learned model: **2/50 collisions (4%)**, minimum center distance **2.680 m**,
  mean episode speed **21.850 m/s**. It chose SLOWER on 18.9% of steps.
- Reactive baseline: **0/50 collisions**, minimum center distance **3.998 m**,
  mean episode speed **22.804 m/s**. These previously verified baseline records
  were reused after exact seed/config/completion checks; the model's 50 episodes
  were newly simulated. The output summary records the baseline's source.

This is 48 fewer crashes in this fixed test, **not a guarantee or a complete
elimination of crashes**. Remaining collision seeds are **10 and 46**; both logs
end after a LANE_RIGHT action. The learned model still underperformed the reactive
rule on collision count, so the takeover demo continues to use the rule controller.
These tests concern stock traffic; they do not establish learned-policy safety
against the forced hazards. Changes were tested together, so no individual
feature or reward change is credited with a separately measured improvement.

Local artifacts (ignored by Git):

- Model and training provenance: `runs/stage5_20261103T201030071303Z_09dbd8ed/`.
- Audited evaluation and per-step scenarios:
  `runs/stage5_evaluation_20261103T204416859865Z_c3a4a4f2/`.
- The earlier unsuccessful model remains in
  `runs/stage5_20261103T151713908654Z_42d79d87/`.

Watch the corrected checkpoint locally:

```sh
source .venv/bin/activate
python 05_train_policy.py drive runs/stage5_20261103T201030071303Z_09dbd8ed/policy.zip --seed 0
```

This seed completed its full 40-second headless evaluation without a collision.
The assistant verified loading, headless driving, and full-episode replay;
window rendering remains a user-run check. A fresh checkout must train locally
because model ZIPs and run artifacts are not committed.

### Optional guard for learned driving

Replaying the two remaining failures showed unsafe right-lane requests: in seed
10 another car was alongside ego; in seed 46 a faster car was about 2 m behind
in the target lane. The model's collision penalty had not reliably prevented
these requests. The installed simulator checks road reachability when changing
lanes; it does not reject a discrete request because another car occupies that
lane.

`policy_guard.py` adds explicit rules around the saved network. It uses only the
current observation and the same front/rear gap, relative-speed, projected-gap,
and time-to-collision checks used by the reactive controller. Unsafe lane changes
are replaced by SLOWER. Repeated lane-change requests during lateral motion are
held with IDLE, or SLOWER when danger is detected. Dangerous IDLE/FASTER requests
are also replaced by SLOWER. Clear requests pass through. No weights are changed.

Run the guarded window yourself, including a previously failing seed:

```sh
source .venv/bin/activate
python 05_train_policy.py drive runs/stage5_20261103T201030071303Z_09dbd8ed/policy.zip --guard --seed 10
```

The terminal announces `RULE-BASED GUARD ENABLED`, labels steps
`controller=learned_guarded`, and prints `GUARD ACTIVE` with the requested action,
executed action, and reason whenever it intervenes. The car should complete the
40-second seed-10 run; close the window or press Ctrl+C to exit. Window behavior
is a user-run check. Omit `--guard` to inspect the raw learned policy, including
its documented failures. The guard is optional and disabled by default.

Headless checks and evaluation:

```sh
python test_policy_guard.py
python validate_policy_guard.py runs/stage5_20261103T201030071303Z_09dbd8ed/policy.zip
python 05_train_policy.py evaluate runs/stage5_20261103T201030071303Z_09dbd8ed/policy.zip --guard
```

Guard validation uses seeds 2005-2009, separate from training and model selection;
known failure seeds 10 and 46 are reported separately as regressions. The main
evaluation still uses the fixed 0-49 seeds and shared metrics. Once failures from
that set have informed a fix, it is a regression benchmark, not a fresh unseen
test of generalization. Results must be reported as **learned policy + guard**;
they do not measure an improvement to the neural network itself.

Each guarded scenario step adds `decision_info`: requested action, executed
action, intervention flag, and reason. Episode and evaluation summaries count
guard interventions separately from Stage 3.5 takeovers. Replay preserves the
executed trajectory and decision records. `verify_runs.py` checks their action
and intervention-count consistency. Evaluation metadata also records the model
file's relative path and SHA-256 fingerprint. `policy_diagnostics.py` distinguishes
model requests from executed actions for guarded runs. No guard can guarantee
avoidance after an already-unavoidable traffic situation develops.

Measured with the same saved model, 50 stock-traffic seeds, and 40-second horizon:

- Raw learned policy (previous recorded benchmark): **2/50 collisions**, mean
  episode speed **21.850 m/s**, minimum center distance **2.680 m**.
- Learned policy + guard (new run): **0/50 collisions**, mean episode speed
  **21.613 m/s**, minimum center distance **3.990 m**. The guard overrode
  **99 of 2,000 actions (4.95%)**. Overrides are not counts of crashes prevented.
- Five separate validation seeds and both known failure regressions also finished
  without a collision. The 26 headless unit/integration tests passed. Replays of
  both formerly failing seeds reproduced their complete guarded trajectories.

The weights were not retrained for this change. These measurements establish the
guarded system's result on the tested scenes, not universal collision avoidance
or an improvement to the network alone. The stock-traffic test does not establish
performance against developer-forced hazards. The reactive comparison again used
previously verified records with matching environment settings.

Local guarded benchmark:
`runs/stage5_evaluation_20261110T050021461945Z_86d3ced6/summary.json`.
Separate validation and crash regressions:
`runs/guard_validation_20261110T045913472701Z_728b9f01/summary.json`.

Installed optional versions: stable-baselines3 2.9.2 (MIT) and torch 2.15.0.
PyTorch's installed metadata lists its composite open-source license expression:
Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0,
ISC, MIT, and Zlib. Both APIs were checked against installed package source.

After the approved Stage 5 install, every installed distribution outside the
named allowed libraries is listed here (automatic dependencies plus pip):

```text
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
Farama-Notifications==0.0.6
filelock==4.2.7
fonttools==4.67.0
fsspec==2026.9.0
Jinja2==3.1.6
kiwisolver==1.5.1
MarkupSafe==3.0.4
mpmath==1.3.0
networkx==3.7
packaging==26.3
pandas==3.0.6
pillow==13.0.0
pip==26.2.1
pyparsing==3.3.3
python-dateutil==2.9.0.post0
setuptools==84.2.1
six==1.17.0
sympy==1.14.0
typing_extensions==4.16.0
```

## Verification and file guide

```sh
python test_stage1.py
python test_project.py
python verify_runs.py runs/BENCHMARK_DIRECTORY
```

Both suites are headless. The newer tests cover rear clearance, cut-in closing
speed, sustained-safe handback, assist disable, event triggering by time and
distance, replay/tamper detection, command parsing, collision attribution,
braking below 20 m/s, and disjoint training/evaluation seeds. UI window inspection
and keyboard interaction remain user-run checks; the assistant does not open
pygame windows. No latency or rendering-performance improvement is claimed.

- `reactive.py`: observation-only rules.
- `assist.py`: danger detection and takeover state.
- `hazards.py`: vehicle subclasses, scenario environment, collision attribution.
- `scenarios.py`: shared scenario records and replay verification.
- `project_runner.py`, `benchmark.py`: demos and common stage comparisons.
- `local_ui.py`: local pygame window and regex parser.
- `learning.py`: optional CPU DQN training and held-out evaluation.
- `learning_features.py`: observation-derived features and training-only rewards.
- `policy_guard.py`: optional, explicitly labeled rule checks around learned actions.
- `validate_policy_guard.py`: separate validation seeds and known crash regressions.
- `inspect_policy_failures.py`: headless replay with pre-crash observation diagnostics.
- `policy_diagnostics.py`: action counts and crash counts from saved local runs.

All outputs and model ZIPs remain ignored under `runs/`. No installed library
was edited. No cloud logging, hosting, paid services, or runtime network calls
are used. The project is a simulator demonstration, not software for a real car.
