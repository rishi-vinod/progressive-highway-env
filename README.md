# Progressive highway driving

A local, free, open-source driving simulation built one stage at a time on macOS.
**Stage 0 (Parts 1 and 2) and Stage 1 are implemented. Stages 2-5 are planned only.**
Parts 1-2 and Stages 1-4 (including 3.5) do **not** learn. Only Stage 5 learns.
No accounts, API keys, paid services, model downloads, or runtime network calls.

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

Allowed libraries for the current stage: highway-env, gymnasium, numpy,
pygame-ce, and matplotlib (plus their automatically installed dependencies).
The pinned versions are highway-env 1.12.1, gymnasium 1.4.0, numpy 2.5.4,
pygame-ce 2.5.8, and matplotlib 3.11.2.

Every other installed distribution (automatic dependencies, plus pip):

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
virtual environment. Stage 1 uses pygame-ce; no further dependencies are needed.

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

## Roadmap — later stages only when requested

1. **Stage 1: implemented.** Stay in lane and keep speed, real-unit observations,
   shared physics-tick metrics, and fixed-seed headless evaluation (see above).
2. **Stage 2: reactive stock-traffic driving.** Use only current observed positions
   and speeds for gap/time-to-collision braking and lane changes; never inspect
   traffic's internal intentions. Check target lanes ahead and behind.
3. **Stage 3: repeatable forced hazards.** Subclass vehicle classes for hard braking
   and cut-ins, triggered by simulation time or distance. Stress-test Stage 2.
4. **Stage 3.5: human driving with hazard takeover.** Keyboard driving and assist
   ON/OFF, seeded random Stage 3 hazards, named observation-only danger thresholds,
   Stage 2 rescue with front/rear lane checks (brake if blocked), visible
   ASSIST ACTIVE notice, and handback after a sustained safe interval. Log human
   versus assist actions and the hazard schedule. Compare assist OFF/ON on the
   same 50 seeds using a Stage 1 stand-in driver or recorded keypress replay.
   Report hazards triggered, collisions, percentage avoided, secondary collisions,
   closest distance, average speed, and takeover count. Avoidance is not guaranteed.
5. **Stage 4 — advanced/later: local developer UI.** rgb_array simulation frames
   beside a rule-based keyword/regex command panel and save-scenario. Before
   building, explain Gradio's license and dependencies (including huggingface_hub)
   and ask the user to choose Gradio or pygame-only. If chosen, disable Gradio
   analytics before import and launch only on 127.0.0.1 with share=False.
6. **Stage 5 — advanced/later: local CPU reinforcement learning.** Request approval
   before installing stable-baselines3 and CPU-only PyTorch; keep dependencies in
   requirements-stage5.txt. Short training with negative collision reward,
   separate training and held-out seeds, and comparison to rules using identical
   metrics. Optional learned guardian for Stage 3.5. Only this stage learns.

## Scenario saving and replay — future stage work

Stage 1 implements the shared metrics and evaluation runner described above.
Stages 2-5 and scenario/replay functionality remain unimplemented. Preserve the
Stage 1 metric definitions and fixed 50 seeds in future stage comparisons.

JSON scenarios will save environment config, seed, every typed command with both
wall-clock timestamp and effective simulation step, and per-step position, speed,
and heading for all cars plus the ego action. Stage 3.5 adds acting controller and
hazard schedule. Replay resets the seed and applies the same commands/events at
the same steps, then verifies the result against the saved run. Scenario training
recreates seeds and event schedules; it does not train on per-step log records.

Scripts print to the terminal and keep library caches under ignored `runs/`.
Stage 1 metrics use local JSON/CSV files there; future scenarios will also stay local.
No cloud logging, hosting, paid services, or runtime network calls are permitted.
