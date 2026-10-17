# Progressive highway driving

A local, free, open-source driving simulation built one stage at a time on macOS.
**Parts 1 and 2 are implemented. Stages 1-5 are planned only.**
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
virtual environment; the earlier headless smoke tests predate that replacement.

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

## Roadmap — build only when requested

1. **Stage 1: rule-based baseline.** Stay in lane and keep speed, with expected
   crashes. Explicitly configure and print observations in real units
   (normalization off, sufficient vehicles, distance sorting). Add shared metrics
   and the fixed-seed headless evaluator with this first stage.
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

## Shared evaluation and scenarios — future stage work

No Stage 1-5 code, evaluation runner, or scenario/replay implementation is included
in Parts 1-2. Add these as their stages are authorized.

Every stage will log per episode: collision, minimum center-to-center distance to
another car across the episode, and average ego speed. Ground-truth vehicle
positions are allowed for metrics and logs, never as controller inputs. Use the
same fixed 50 seeds (0 through 49) for the headless evaluation script and print a
summary table. Define shared sampling and averaging conventions when Stage 1 is
built, and preserve them for all comparisons.

JSON scenarios will save environment config, seed, every typed command with both
wall-clock timestamp and effective simulation step, and per-step position, speed,
and heading for all cars plus the ego action. Stage 3.5 adds acting controller and
hazard schedule. Replay resets the seed and applies the same commands/events at
the same steps, then verifies the result against the saved run. Scenario training
recreates seeds and event schedules; it does not train on per-step log records.

The current script prints to the terminal and keeps library caches under ignored
`runs/`. Future scenario and metrics output will use local JSON/CSV files there.
No cloud logging, hosting, paid services, or runtime network calls are permitted.
