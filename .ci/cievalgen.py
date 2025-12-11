#!/usr/bin/env python3

"""Generate CI evaluation YAML file for GitLab CI/CD."""

import argparse
import json
from textwrap import dedent, indent

EVALSET_TEMPLATE = dedent(
    """
    {evalset}:
        stage: evalsets
        tags: [basalt-evaluation]
        needs: ["build"]
        extends: .run-dataset
        resource_group: at111
        parallel:
            matrix:
            - DATASET: [{datasets}]
    """
)

TIMING_EVALSET_TEMPLATE = dedent(
    """
    {evalset}:
        stage: timing
        tags: [basalt-timing-evaluation]
        needs: ["build"]
        extends: .time-dataset
        parallel:
            matrix:
            - DATASET: [{datasets}]
    """
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "description", type=str, help="Path to the evaluation set description JSON file"
    )
    parser.add_argument(
        "evalsets",
        type=str,
        help="Comma-separated list (without spaces) of selected evalsets to"
        "generate CI file from, e.g., 'euroc-v1,quickset1,EMH05'",
    )
    parser.add_argument(
        "timing_evalsets",
        type=str,
        help="Same as evalsets but for timing evaluation.",
    )
    parser.add_argument(
        "template",
        type=str,
        help="Path to the CI YAML template to fill in",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output the filled in YAML file",
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="Set to 1 by default for deterministic evaluation. ",
    )
    parser.add_argument(
        "--timing_deterministic",
        type=int,
        default=1,
        help="Set to 1 by default for deterministic timing evaluation.",
    )
    parser.add_argument(
        "--timing_repetitions",
        type=int,
        default=1,
        help="Runs can be performed multiple times for warming up the system.",
    )

    return parser.parse_args()


def add_custom_evalset(evalsets, evaluations):
    custom_evalset = []
    for i, string in enumerate(list(evalsets)):
        if string in evaluations["evalsets"]:
            continue

        if string in evaluations["sequences"]:
            custom_evalset.append(string)
        else:
            print(f"Ignoring invalid evalset/sequence: {string}")

        evalsets.remove(string)
    if len(custom_evalset) > 0:
        evaluations["evalsets"]["custom"] = custom_evalset
        evalsets.append("custom")


def main():
    "Main function"
    args = parse_args()
    description = args.description
    evalsets = args.evalsets.split(",")
    timing_evalsets = args.timing_evalsets.split(",")
    template = args.template
    output = args.output
    deterministic = args.deterministic
    timing_deterministic = args.timing_deterministic
    timing_repetitions = args.timing_repetitions

    tcontents = open(template, "r", encoding="utf-8").read()
    evaluations = json.load(open(description, "r", encoding="utf-8"))
    timing_evaluations = evaluations.copy()

    jobs = []
    tjobs = []

    # Optionally add a `custom` evalset if individual sequences were specified
    add_custom_evalset(evalsets, evaluations)
    add_custom_evalset(timing_evalsets, timing_evaluations)

    for evalset in evalsets:
        datasets = ", ".join(evaluations["evalsets"][evalset])
        job = EVALSET_TEMPLATE.format(evalset=evalset, datasets=datasets)
        jobs.append(job)
    jobs_str = f"".join(jobs).strip()

    for evalset in timing_evalsets:
        datasets = ", ".join(timing_evaluations["evalsets"][evalset])
        job = TIMING_EVALSET_TEMPLATE.format(evalset=f"timing:{evalset}", datasets=datasets)
        tjobs.append(job)
    tjobs_str = f"".join(tjobs).strip()


    contents = tcontents.format(
        evalset_list=args.evalsets,
        evalsets_jobs=jobs_str,
        deterministic=deterministic,
        timing_evalset_list=args.timing_evalsets,
        timing_jobs=tjobs_str,
        timing_deterministic=timing_deterministic,
        timing_repetitions=timing_repetitions,
    )
    open(output, "w", encoding="utf-8").write(contents)


if __name__ == "__main__":
    main()
