#!/usr/bin/env python3

"""Generate CI evaluation YAML file for GitLab CI/CD."""

import argparse
import json
from textwrap import dedent, indent


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
        "generate CI file from, e.g., 'euroc-v1,quickset1'",
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

    return parser.parse_args()


def main():
    "Main function"
    args = parse_args()
    description = args.description
    evalsets = args.evalsets.split(",")
    template = args.template
    output = args.output
    deterministic = args.deterministic

    tcontents = open(template, "r", encoding="utf-8").read()
    evaluations = json.load(open(description, "r", encoding="utf-8"))

    evalset_jobs = []

    # Optionally add custom evalset if individual sequences were specified
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

    for evalset in evalsets:
        datasets = ", ".join(evaluations["evalsets"][evalset])

        evalset_job = dedent(
            f"""
            {evalset}:
              stage: evalsets
              tags: [basalt-evaluation]
              needs: ["build"]
              extends: .run-dataset
              parallel:
                  matrix:
                    - DATASET: [{datasets}]
            """
        )
        evalset_jobs.append(evalset_job)

    evalset_jobs_str = f"".join(evalset_jobs).strip()
    contents = tcontents.format(
        evalset_list=args.evalsets,
        evalsets_jobs=evalset_jobs_str,
        deterministic=deterministic,
    )
    open(output, "w", encoding="utf-8").write(contents)


if __name__ == "__main__":
    main()
