# Project instructions

## COST RULES (strict, apply to every stage)
- Allowed: Python 3.12 (project range 3.9-3.12), highway-env, gymnasium, numpy, pygame-ce, matplotlib.
- Use pygame-ce only, as required by highway-env. Its Python import remains `import pygame`. Do not install the separate pygame distribution alongside pygame-ce because they share the same import namespace and conflict.
- Optional, Stage 4 only: Gradio. Advanced, Stage 5 only: stable-baselines3 and PyTorch (CPU only).
- Packages that the allowed libraries install automatically as their own dependencies are fine, but after install, list for me every package in the environment that is not one of the allowed ones.
- Before adding ANY other library, stop and ask me first and tell me its license.
- No paid services, cloud services, API keys, accounts, or logins. No LLM, no AI API calls, no model downloads.
- The chatbot panel is a simple rule-based parser (keywords/regex), nothing more.
- No cloud logging or tracking (no wandb, no Hugging Face Hub). Local files only (JSON, CSV; local TensorBoard is fine).
- No network calls at runtime. The only network use is pip install from PyPI.
- Gradio, if used: set the environment variable GRADIO_ANALYTICS_ENABLED=False BEFORE importing gradio (it sends analytics to api.gradio.app by default), launch with server_name="127.0.0.1" and share=False. No deployment or hosting.
- Everything runs locally on a Mac. No GPU required.

## COMMAND SAFETY RULES (strict)
- I run you in a sandboxed workspace-write mode with approvals on. You MAY run commands yourself inside this project folder, but ONLY commands of the types below. Do not add per-command explanations or safety commentary; just follow these rules.
- Allowed command types only: mkdir, cd, ls, cat, grep, head, tail, touch, pwd, cp and mv (inside the project folder only, never overwriting an existing file without telling me), python3.12 --version, python3.12 -m venv .venv, source .venv/bin/activate, which python, python --version (inside the venv), pip install / pip install --upgrade pip / pip show / pip list / pip freeze (only inside the activated .venv, and only for allowed packages from PyPI), running the project's own Python scripts (HEADLESS ONLY, see below), and git status / diff / log.
- Always confirm the .venv is active (which python points inside the project) before any pip install.
- NEVER use: sudo, su, rm (never delete anything; if a file must be removed, tell me and I will do it), curl or wget, brew or any system-wide install, chmod -R, or any change outside the project folder (including ~/.zshrc, system settings, other folders).
- pip may only install from PyPI: no git+ URLs, no --index-url or --extra-index-url, no --trusted-host, no local archives from elsewhere.
- No git init, git add, git commit, git push, remotes, or GitHub login. I handle all committing and pushing myself. Do not commit unless I ask.
- No commands that open network ports, except Gradio on 127.0.0.1 in Stage 4.
- Never read or print secrets or personal files (~/.ssh, environment variables, keychain).
- Do NOT run any script that opens a pygame window (render_mode="human"). Smoke-test with headless runs (render_mode=None) and give me the exact command to run the windowed version myself.
- If you need something not on the allowed list, or a command asks for approval and is not on the list, STOP and ask me instead of running or requesting it.

## HOW TO WORK WITH ME
- Create and edit files directly in the project folder. After running commands, show me the output and what you checked.
- Always use python3.12 (never python3, which is 3.13.6 on my Mac and out of range) to create the venv.
- This project is a PUBLIC GitHub repo. Use relative paths only; never write absolute personal paths or any personal info, keys, or secrets into any file.
- Clean, well-commented, easy-to-read Python with helpful debug prints. Explain the reason behind any design decision in one or two sentences.
- Mark advanced/later goals clearly. Never edit the installed library; subclass or configure instead.
- Verify the highway-env API against the INSTALLED version (pip show, read its source or README in .venv) instead of relying on memory.
- Create an AGENTS.md in the project folder containing the COST RULES, COMMAND SAFETY RULES, and HOW TO WORK WITH ME sections of this prompt, so the rules persist across stages.
- README.md explains how to run each stage and states clearly that Stages 1-4 (including 3.5) do NOT learn; only Stage 5 learns. Update it as each stage is added.
- At the end of each stage, tell me exactly what to run and what "working" should look like, then STOP and wait for me.

## Scope and future requirements
- Parts 1 and 2 only are authorized initially. Build ONE later stage at a time, only when requested, in its own file.
- Stages 1-4 (including 3.5) do NOT learn. Only Stage 5 learns.
- Controllers must use only current observations; ground-truth positions may be used for metrics and logging only.
- Every stage uses the same fixed 50 seeds and metrics: collision, minimum center-to-center distance to another car across the episode, and average speed. Add the headless evaluation script when Stage 1 is authorized.
- Future JSON scenarios must contain env config, seed, typed commands with wall-clock timestamps and effective simulation steps, and per-step positions, speeds, headings of all cars plus ego action. Replay must reset with the same seed, apply commands at the same steps, and verify the result.
- Stage 3.5 additionally logs acting controller and seeded hazard schedule, compares assist OFF/ON over 50 seeds with a stand-in driver, and reports avoidance percentages rather than guarantees. Lane changes must check ahead and behind; handback requires a sustained safe interval.
- Before Stage 4, explain Gradio's license and extra dependencies (including huggingface_hub), and let the user choose Gradio or pygame-only.
- Before Stage 5, request installation approval; use separate requirements-stage5.txt, CPU-only training, and held-out evaluation seeds.
