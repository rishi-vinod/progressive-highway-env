"""Local pygame views for human takeover and the developer command panel.

Only run_window opens a display. Parsing and headless runs never open a port.
"""

import copy
import re
import time

from driving_common import ENV_CONFIG
from scenarios import Session
from hazards import seeded_schedule
from assist import AssistController
from reactive import reactive_action, IDLE, LANE_LEFT, LANE_RIGHT, FASTER, SLOWER
import pygame


def parse_command(text):
    """A small allowlisted parser. No eval, model, API, shell, or network calls."""
    normalized = " ".join(text.lower().strip().split())
    if normalized in ("save", "save scenario", "save-scenario"):
        return {"operation": "save"}
    if normalized in ("pause", "resume"):
        return {"operation": normalized}
    if normalized == "help":
        return {"operation": "help"}
    if re.fullmatch(r"(?:make (?:the )?car (?:in )?front )?(?:brake(?: hard)?|hard brake)", normalized):
        return {"operation": "hazard", "kind": "hard_brake"}
    match = re.fullmatch(r"(?:force (?:a )?)?cut[ -]in from (?:the )?(left|right)(?: lane)?", normalized)
    if match:
        return {"operation": "hazard", "kind": f"cut_in_{match.group(1)}"}
    raise ValueError("Commands: brake hard; cut-in from left/right; save scenario; pause; resume; help")


def execute_command(session, text):
    try:
        parsed = parse_command(text)
        if parsed["operation"] == "hazard":
            session.command(text, {"kind": parsed["kind"], "duration_s": 3.0})
        else:
            session.command(text)
        return parsed
    except ValueError:
        # Invalid/no-target commands are recorded as no-ops, so every submitted
        # command has both a timestamp and an effective step in the scenario.
        session.command(text)
        raise


def run_window(stage, seed=0, max_steps=200, assist_on=True):
    """User-run only. rgb_array feeds one local pygame window with controls."""
    settings = copy.deepcopy(ENV_CONFIG)
    # Human input is sampled at 5 Hz. Benchmark files record their own frequency;
    # the OFF/ON benchmark uses a matched profile on both sides.
    settings.update({"policy_frequency": 5, "screen_width": 900, "screen_height": 400})
    session = Session(seed, seeded_schedule(seed) if stage == "3.5" else [],
                      render_mode="rgb_array", config=settings)
    helper = AssistController(assist_on, policy_dt=0.2)
    message = "Arrow keys drive. A toggles assist. Esc exits." if stage == "3.5" else "Type help or a developer command."
    text = ""
    history = [message]
    pending_action = IDLE
    paused = False
    running = True
    saved = []
    try:
        pygame.init()
        screen = pygame.display.set_mode((1280, 600))
        pygame.display.set_caption(f"Progressive Highway - Stage {stage} - local only")
        font = pygame.font.Font(None, 24)
        small = pygame.font.Font(None, 20)
        clock = pygame.time.Clock()
        # Render to an offscreen surface first; highway-env does not own events.
        frame = session.env.render()
        session.env.unwrapped.enable_auto_render = False
        next_step = time.monotonic() + 0.2
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif stage == "3.5":
                        keys = {pygame.K_LEFT: LANE_LEFT, pygame.K_RIGHT: LANE_RIGHT,
                                pygame.K_UP: FASTER, pygame.K_DOWN: SLOWER}
                        if event.key in keys:
                            pending_action = keys[event.key]
                            session.command(f"key:{pygame.key.name(event.key)}")
                        elif event.key == pygame.K_a:
                            helper.enabled = not helper.enabled
                            session.command(f"assist {'on' if helper.enabled else 'off'}")
                        elif event.key == pygame.K_s:
                            session.command("save scenario")
                            saved.append(session.save(extra={"takeovers": helper.takeovers}))
                            history.append(f"Saved snapshot {len(saved)}")
                    elif event.key == pygame.K_RETURN:
                        try:
                            command = execute_command(session, text)
                            operation = command["operation"]
                            history.append("> " + text)
                            if operation == "save":
                                saved.append(session.save())
                                history.append(f"Saved snapshot {len(saved)}")
                            elif operation in ("pause", "resume"):
                                paused = operation == "pause"
                            elif operation == "help":
                                history.extend(["brake hard", "cut-in from left", "cut-in from right", "save scenario", "pause / resume"])
                            else:
                                history.append("Hazard queued for next step")
                        except ValueError as error:
                            history.append(str(error))
                        text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif stage == "4" and event.unicode.isprintable() and len(text) < 100:
                        text += event.unicode
            now = time.monotonic()
            if running and not paused and not session.ended and session.step_number < max_steps and now >= next_step:
                human_action = pending_action if stage == "3.5" else None
                if stage == "3.5":
                    action = helper.choose(session.observation.copy(), pending_action)
                    label = helper.label
                else:
                    action, label = reactive_action(session.observation.copy()), "rule"
                session.step(int(action), label, human_action)
                pending_action = IDLE
                frame = session.env.render()
                session.env.unwrapped.enable_auto_render = False
                next_step = now + 0.2
            if paused:
                next_step = now + 0.2
            screen.fill((18, 24, 32))
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (15, 80))
            active = stage == "3.5" and helper.active
            title = "ASSIST ACTIVE" if active else "HUMAN CONTROL" if stage == "3.5" else "DEVELOPER CONTROL PANEL"
            screen.blit(font.render(title, True, (255, 190, 70) if active else (110, 220, 170)), (20, 20))
            status = f"step {session.step_number} | assist {'ON' if helper.enabled else 'OFF'} | " if stage == "3.5" else f"step {session.step_number} | "
            status += "episode ended - save or Esc" if session.ended or session.step_number >= max_steps else "paused" if paused else "running"
            screen.blit(small.render(status, True, (220, 225, 230)), (20, 50))
            instructions = ["LEFT/RIGHT: change lane", "UP/DOWN: faster/slower", "A: assist ON/OFF", "S: save scenario", "Esc: quit"] if stage == "3.5" else ["Developer commands", "brake hard", "cut-in from left/right", "save scenario", "pause / resume", "Enter submits; Esc quits"]
            for index, line in enumerate(instructions):
                screen.blit(small.render(line, True, (220, 225, 230)), (940, 30 + index * 25))
            for index, line in enumerate(history[-10:]):
                # Wrap to fit the panel, rather than hiding long error messages.
                for part in range(min(2, (len(line) + 34) // 35)):
                    screen.blit(small.render(line[part * 35:(part + 1) * 35], True, (170, 190, 210)), (940, 210 + index * 28 + part * 13))
            if stage == "4":
                pygame.draw.rect(screen, (45, 55, 70), (930, 535, 335, 45))
                screen.blit(small.render(text[-35:] + "_", True, (255, 255, 255)), (940, 550))
            screen.blit(small.render("No learning in Stages 1-4. Local files only.", True, (180, 190, 200)), (20, 530))
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if not session.ended and session.step_number < max_steps:
            session.end_reason = "window_closed"
        try:
            path = session.save(extra={"stage": stage, "takeovers": helper.takeovers,
                                       "assist_enabled": helper.enabled, "ui_policy_frequency": 5})
            print(f"Saved scenario: {path}")
        finally:
            session.close()
            pygame.quit()
