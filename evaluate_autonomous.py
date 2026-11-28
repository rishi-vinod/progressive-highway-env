"""Headless 50-seed style comparison; stock traffic, local JSON/CSV only."""

import argparse
import copy

from assist import AssistController, ASSIST_TARGET_SPEEDS, STYLES
from driving_common import ENV_CONFIG, EVALUATION_SEEDS, new_run_directory, write_json, write_episode_csv
from evaluate import summarize
from reactive import IDLE, lane_number
from scenarios import Session


def evaluate(style):
    directory = new_run_directory('autonomous_' + style.lower())
    settings = copy.deepcopy(ENV_CONFIG)
    settings.update(policy_frequency=5, simulation_frequency=15)
    settings['action']['target_speeds'] = ASSIST_TARGET_SPEEDS.copy()
    results, lane_changes, takeovers, flow_errors = [], [], [], []
    for seed in EVALUATION_SEEDS:
        session = Session(seed, config=settings)
        helper = AssistController(style=style, policy_dt=0.2)
        helper.toggle()
        changed = 0
        episode_flow_errors = []
        lane = lane_number(session.observation)
        try:
            while not session.ended:
                action = helper.choose(session.observation, IDLE)
                session.step(action, helper.label, IDLE, helper.decision_info)
                episode_flow_errors.append(float(session.observation[0, 3]) - helper.planner.info['flow_speed_mps'])
                new_lane = lane_number(session.observation)
                changed += int(new_lane != lane)
                lane = new_lane
            result = session.result()
            results.append(result)
            lane_changes.append(changed)
            takeovers.append(helper.takeovers)
            flow_errors.append(sum(episode_flow_errors) / len(episode_flow_errors))
            write_json(directory / f'seed_{seed:02d}.json', session.document(
                dict(style=style, lane_transitions=changed, emergency_interventions=helper.takeovers)))
            print(f'{style} seed={seed:02d} collision={result.collision} '
                  f'speed={result.average_speed_mps:.3f}m/s lane_transitions={changed}', flush=True)
        finally:
            session.close()
    summary = summarize(results)
    summary.update(style=style, seeds=list(EVALUATION_SEEDS), complete=True,
                   mean_lane_transitions=sum(lane_changes) / len(results),
                   emergency_interventions=sum(takeovers),
                   mean_speed_relative_to_flow_mps=sum(flow_errors) / len(flow_errors),
                   total_simulation_s=sum(r.elapsed_s for r in results))
    write_json(directory / 'summary.json', summary)
    write_episode_csv(directory / 'episodes.csv', results)
    print(summary, flush=True)
    print(f'Saved: {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', choices=STYLES, required=True)
    evaluate(parser.parse_args().style)
