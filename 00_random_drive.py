"""Part 2: random driving only; no learning, controllers, or runtime network calls."""

import argparse
import os
from pathlib import Path

# Keep Matplotlib's import-time cache inside the project, including headless runs.
os.environ["MPLCONFIGDIR"] = str(Path("runs") / "matplotlib")
# Fontconfig requires an absolute cache location; derive it from a relative
# project path at runtime, without storing any machine-specific path in source.
os.environ["XDG_CACHE_HOME"] = str((Path("runs") / "cache").resolve())

import gymnasium as gym
import highway_env
import pygame

# Verified in the installed highway-env 1.12.1 usage guide. The import registers
# environments; Gymnasium 1.4.0 keeps register_envs as an explicit no-op marker.
gym.register_envs(highway_env)


def nonnegative_int(value: str) -> int:
    """Accept valid seeds, with an argparse error for negative values."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def positive_int(value: str) -> int:
    """Reject a step budget that would never drive the car."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--max-steps", type=positive_int, default=100)
    parser.add_argument(
        "--headless", action="store_true", help="Use render_mode=None (no window)."
    )
    args = parser.parse_args()
    env = None

    try:
        env = gym.make("highway-v0", render_mode=None if args.headless else "human")
        env.reset(seed=args.seed)
        # reset() recreates the spaces, so seed the action space AFTER reset.
        env.action_space.seed(args.seed)
        base_env = env.unwrapped
        action_names = base_env.action_type.actions
        print(
            f"highway-v0 | seed={args.seed} | headless={args.headless} | "
            f"other_cars={len(base_env.road.vehicles) - 1}"
        )
        print(f"Actions: {action_names}")
        print("Random actions only; no learning.")

        for step in range(1, args.max_steps + 1):
            # The stock viewer handles its close button by calling env.close(),
            # which sets done. Do not start another step after a window close.
            if base_env.done:
                print("Window closed.")
                break
            action = int(env.action_space.sample())
            _, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step:03d} action={action} ({action_names[action]}) "
                f"reward={reward:.6f} collision={bool(info['crashed'])}",
                flush=True,
            )
            if base_env.done:
                print("Window closed.")
                break
            if terminated or truncated:
                print(f"Episode ended: terminated={terminated}, truncated={truncated}")
                break
        else:
            print(f"Reached --max-steps={args.max_steps}.")
    except KeyboardInterrupt:
        print("\nInterrupted with Ctrl+C.")
    except pygame.error:
        # Some close events arrive mid-render, after pygame has shut down.
        # Suppress only that shutdown case; expose unrelated rendering errors.
        if env is None or not env.unwrapped.done:
            raise
        print("Window closed during rendering.")
    finally:
        if env is not None:
            env.close()
        pygame.quit()
        print("Environment closed.")


if __name__ == "__main__":
    main()
