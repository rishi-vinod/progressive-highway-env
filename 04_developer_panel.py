"""Stage 4: pygame-only developer command panel. No server, AI, or learning."""

import argparse
import copy
from driving_common import ENV_CONFIG, positive_int, nonnegative_int
from scenarios import Session
from local_ui import run_window, execute_command
from reactive import reactive_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--max-steps", type=positive_int, default=200)
    parser.add_argument("--command", action="append", default=[], help="Headless command at step 1; may repeat.")
    args = parser.parse_args()
    if not args.headless:
        run_window("4", args.seed, args.max_steps)
        return
    config = copy.deepcopy(ENV_CONFIG)
    config["policy_frequency"] = 5
    session = Session(args.seed, config=config)
    try:
        for text in args.command:
            execute_command(session, text)
        for _ in range(args.max_steps):
            session.step(reactive_action(session.observation.copy()), "rule")
            if session.ended:
                break
        print(f"Saved headless panel scenario: {session.save(extra={'stage': '4'})}")
        print(session.result())
    finally:
        session.close()


if __name__ == "__main__":
    main()
