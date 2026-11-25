"""Local pygame views for human takeover and the developer command panel.

Only run_window opens a display. Parsing and headless runs never open a port.
"""

import copy
import re
import time

from driving_common import ENV_CONFIG
from scenarios import Session
from hazards import seeded_schedule
from assist import AssistController, ASSIST_TARGET_SPEEDS
from reactive import reactive_action, IDLE, LANE_LEFT, LANE_RIGHT, FASTER, SLOWER
import pygame

UI_FREQUENCY_HZ = 30
OLD_UI_FREQUENCY_HZ = 5
DRIVE_WINDOW = (1120, 390)
ROAD_RECT = pygame.Rect(0, 80, DRIVE_WINDOW[0], 200)
DRIVE_TOGGLE = pygame.Rect(20, 309, 210, 50)


def draw_dashboard(screen, frame, session, helper, started, ended, font, small, now):
    screen.fill((18, 24, 32))
    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), ROAD_RECT.topleft)
    titles = {"manual": "YOU ARE DRIVING", "auto": "SELF-DRIVING",
              "override": "MANUAL OVERRIDE  |  Computer driving", "ready": "Computer driving"}
    title = "Episode ended" if ended else titles[helper.mode]
    screen.blit(font.render(title, True, (245, 245, 245)), (20, 16))
    hint = "Press an arrow to start" if not started else "Left / Right change lane     Hold Up / Down to adjust speed"
    screen.blit(small.render(hint, True, (190, 205, 215)), (20, 48))
    ego = session.env.unwrapped.vehicle  # Dashboard only.
    speeds = f"Speed {ego.speed * 3.6:.0f} km/h     Target {ego.target_speed * 3.6:.0f} km/h"
    screen.blit(font.render(speeds, True, (245, 245, 245)), (755, 26))
    pygame.draw.rect(screen, (37, 99, 82) if helper.active else (58, 70, 84), DRIVE_TOGGLE, border_radius=8)
    screen.blit(font.render(f"Self-driving   {'ON' if helper.active else 'OFF'}", True, (255, 255, 255)), (36, 325))
    if helper.mode == "ready" and not ended and int(now * 2) % 2 == 0:
        screen.blit(font.render("TAKE CONTROL  |  Press an arrow", True, (255, 155, 55)), (255, 325))
    screen.blit(small.render("A  Toggle      S  Save      Esc  Exit", True, (190, 205, 215)), (840, 329))


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
    # Keep --max-steps' previous wall-clock duration (it counted 5 Hz steps).
    display_steps = max_steps * UI_FREQUENCY_HZ // OLD_UI_FREQUENCY_HZ
    settings = copy.deepcopy(ENV_CONFIG)
    # Render a fresh physics state at 30 Hz; the old 5 Hz UI repeated each
    # frame on a 60 Hz display. Benchmarks retain their matched saved profile.
    settings.update({"simulation_frequency": UI_FREQUENCY_HZ,
                     "policy_frequency": UI_FREQUENCY_HZ,
                     "screen_width": 900, "screen_height": 400})
    if stage == "3.5":
        settings["action"]["target_speeds"] = ASSIST_TARGET_SPEEDS.copy()
        settings.update(screen_width=ROAD_RECT.width, screen_height=ROAD_RECT.height, scaling=6)
    session = Session(seed, seeded_schedule(seed) if stage == "3.5" else [],
                      render_mode="rgb_array", config=settings)
    helper = AssistController(assist_on, policy_dt=1 / UI_FREQUENCY_HZ)
    message = "You have control. Arrow keys drive." if stage == "3.5" else "Type help or a developer command."
    text = ""
    history = [message]
    pending_action = IDLE
    started = stage != "3.5"
    paused = False
    running = True
    saved = []
    try:
        pygame.init()
        screen = pygame.display.set_mode(DRIVE_WINDOW if stage == "3.5" else (1280, 600))
        pygame.display.set_caption(f"Progressive Highway - Stage {stage} - local only")
        font = pygame.font.Font(None, 24)
        small = pygame.font.Font(None, 20)
        clock = pygame.time.Clock()
        # Render to an offscreen surface first; highway-env does not own events.
        frame = session.env.render()
        session.env.unwrapped.enable_auto_render = False
        next_step = time.monotonic() + 1 / UI_FREQUENCY_HZ
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif (stage == "3.5" and event.type == pygame.MOUSEBUTTONDOWN
                      and event.button == 1 and DRIVE_TOGGLE.collidepoint(event.pos)):
                    helper.toggle()
                    started = True
                    session.command("click:self-driving")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif stage == "3.5":
                        keys = {pygame.K_LEFT: LANE_LEFT, pygame.K_RIGHT: LANE_RIGHT,
                                pygame.K_UP: FASTER, pygame.K_DOWN: SLOWER}
                        if event.key in keys:
                            started = True
                            pending_action = keys[event.key]
                            session.command(f"key:{pygame.key.name(event.key)}")
                        elif event.key == pygame.K_a:
                            helper.toggle()
                            started = True
                            session.command("key:self-driving")
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
            if running and started and not paused and not session.ended and session.step_number < display_steps and now >= next_step:
                human_action = pending_action if stage == "3.5" else None
                if stage == "3.5":
                    if pending_action == IDLE and helper.mode == "manual":
                        held = pygame.key.get_pressed()
                        if held[pygame.K_UP] != held[pygame.K_DOWN]:
                            human_action = FASTER if held[pygame.K_UP] else SLOWER
                    action = helper.choose(session.observation.copy(), human_action,
                                           acknowledge=pending_action != IDLE)
                    label = helper.label
                else:
                    action, label = reactive_action(session.observation.copy()), "rule"
                info = {"mode": helper.mode, "projected_contact_s": helper.contact_s} if stage == "3.5" else None
                session.step(int(action), label, human_action, info)
                pending_action = IDLE
                frame = session.env.render()
                session.env.unwrapped.enable_auto_render = False
                next_step = now + 1 / UI_FREQUENCY_HZ
            if paused:
                next_step = now + 1 / UI_FREQUENCY_HZ
            if stage == "3.5":
                draw_dashboard(screen, frame, session, helper, started, session.ended or
                               session.step_number >= display_steps, font, small, now)
                pygame.display.flip()
                clock.tick(60)
                continue
            screen.fill((18, 24, 32))
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (15, 80))
            screen.blit(font.render("DEVELOPER CONTROL PANEL", True, (110, 220, 170)), (20, 20))
            status = f"step {session.step_number} | "
            if not started:
                status += "Press an arrow to start driving"
            else:
                status += "episode ended - save or Esc" if session.ended or session.step_number >= display_steps else "paused" if paused else "running"
            screen.blit(small.render(status, True, (220, 225, 230)), (20, 50))
            instructions = ["Developer commands", "brake hard", "cut-in from left/right", "save scenario", "pause / resume", "Enter submits; Esc quits"]
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
        if not session.ended and session.step_number < display_steps:
            session.end_reason = "window_closed"
        try:
            path = session.save(extra={"stage": stage, "takeovers": helper.takeovers,
                                       "assist_enabled": helper.enabled,
                                       "ui_policy_frequency": UI_FREQUENCY_HZ})
            print(f"Saved scenario: {path}")
        finally:
            session.close()
            pygame.quit()
