<!--
Copyright 2026, Mateo de Mayo
SPDX-License-Identifier: BSD-3-Clause

Author: Mateo de Mayo <mateo.demayo@tum.de>
-->

# CI Runner Configuration

There are three types of runner tags:

- `basalt-docker`
- `basalt-evaluation`
- `basalt-timing-evaluation`

We describe the CI setup by explaining each tag.

## Runner Tags

### Tag `basalt-docker`

- Runner used for crosscompiling jobs to non-x86 platforms (e.g.,
  `aarch64-raspberry-bookworm`, `aarch64-radxa-jammy`).
- These are usually based on compilations guides like <docs/Raspberry.md> or
  <docs/RadxaZero.md>.
- They require a docker executor since a replicable build environment is
  important.

### Tag `basalt-evaluation`

- Runner used for getting accuracy (and other) metrics by running the system on
  different datasets.
- These runners hould have all datasets pre-downloaded and accessible.
- Furthermore these runners run a series of commands that you will need to be
  available, a non-extensive list is: xrtslam-metrics (with complete target
  directory), python3, xrtmet, jq, curl, unzip, tree, head, zip, fish, 7z,
  pandoc.
- We assume the runner runs on a fast disk (e.g., an SSD), and the datasets can
  live on the same disk or on a different disk (e.g., an HDD). If a different
  disk, we assume it is a slower disk and thus we copy them before running, so
  enough space for the biggest datasets should be available for all jobs to
  succeed.
- The dataset location should be set in `.gitlab-runner/config.toml`. See below.

### Tag `basalt-timing-evaluation`

- Runner used to measure timing performance in isolation.
- Similar requirements to the `basalt-evaluation` tag.
- However, a single concurrent job should be allowed to avoid contention between
  jobs.

## Runner Configuration

### Docker executor (`basalt-docker`)

- Configure a new runner
- Set it to docker executor
- Set the `basalt-docker` tag in the [web
  ui](https://gitlab.freedesktop.org/mateosss/basalt/-/settings/ci_cd#js-runners-settings)

### Shell executor (`basalt-evaluation`)

- Configure a new runner
- Set it to shell executor
- For the `report job`, add the `xrtmet` envvar pointing to your installation of
  [xrtslam-metrics](https://gitlab.freedesktop.org/mateosss/xrtslam-metrics),
  with a complete `$xrtmet/test/data/targets` and a `$xrtmet/.venv` virtualenv
  set up.
- Add envvars for the dataset locations (i.e., use the `gitlab-runner register
--env` option or add it to `.gitlab-runner.config.toml` manually):
  ```toml
  [[runners]]
  concurrent=1
  ...
  executor = "shell"
  environment = [
    "xrtmet=/path/to/xrtslam-metrics",
    "msdmo=/path/to/monado-slam-datasets/M_monado_datasets/MO_odyssey_plus",
    "msdmi=/path/to/monado-slam-datasets/M_monado_datasets/MI_valve_index",
    "msdmg=/path/to/monado-slam-datasets/M_monado_datasets/MG_reverb_g2",
    "euroc=/path/to/euroc",
    "tumvi=/path/to/tumvi",
  ]
  ```
- You should specify envvars for all `device`s specified in
  <.ci/evaluation.json> `sequences` field.
- If you are not interested in `basalt-timing-evaluation`, you can set
  `concurrent=3` instead.
- Set the `basalt-evaluation` tag in the [web
  ui](https://gitlab.freedesktop.org/mateosss/basalt/-/settings/ci_cd#js-runners-settings)
- Provide a `READ_PROJECT_TOKEN`. The token is set up once per repo. In case you
  need to setup a new repository, you need to create a project access token with
  `read_api` (you can use any role, I used `reporter`). Then you go to CI/CD ->
  Variables and set a hidden/masked variable with that token called:
  `READ_PROJECT_TOKEN`

### Shell executor (`basalt-timing-evaluation`)

- Same as `basalt-evaluation` setup but requires `concurrent=1`
- Set oldest_first for `gate` resource_group. This will gate pipelines to run
  sequentially while still allowing parallel intra-pipeline jobs. You will need
  an [access
  token](https://gitlab.freedesktop.org/mateosss/basalt/-/settings/access_tokens).

```bash
  curl --request PUT \
    --header "PRIVATE-TOKEN: <private_token>" \
    --data "process_mode=oldest_first" \
    --url "https://gitlab.freedesktop.org/api/v4/projects/mateosss%2Fbasalt/resource_groups/gate"
```

### Running all runners in same computer

- This is possible, you will need to register the two runners (one shell, one
  docker) on the same computer.
- You will need to give two tags to the shell runner (`basalt-evaluation` and
  `basalt-timing-evaluation`)
- You will be limited to using `concurrent=1` if you want to do timing
  evaluation.

## Running Jobs

Most jobs are meant to be manually triggered for now to reduce wasted
computation.

### Triggering builds and static analysis

- You can trigger any build manually
- You can also trigger `clang-format` and `clangd-tidy` jobs

### Triggering an evaluation

For triggering an evaluation you can click into the `create-evaluation` job and
press play.

Furthermore you can configure the evaluation with the following
variables in the web UI before launching it:

- `EVALSETS`: A comma separated list of evalsets from <.ci/evaluations.json>.
  E.g. to run all datasets,
  `"msdmg,msdmio,msdmipb,msdmipp,msdmipt,msdmo,euroc-v1,euroc-v2,euroc-mh,tumvi-room"`.
  (default `"quickset1"`)
- `DETERMINISTIC`: Whether to run the deterministic/reproducible pipeline or the
  faster non-deterministic one. (default `"1"`)
- `NUM_THREADS`: Maximum number of threads to use for the run. Useful to queue
  many runs simultaneously. (default `"0"`)
- `TIMING_EVALSETS`: Same as `EVALSETS` but for timing jobs. (default
  `"EMH02,TR2,MOO02,MIO02,MGO02"`)
- `TIMING_DETERMINISTIC`: Same as `DETERMINISTIC` but for timing jobs (default
  `"1"`)
- `TIMING_REPETITIONS`: Timing jobs are repeated more than one time to
  understand inter-run variability (default `"3"`)
- `TIMING_NUM_THREADS`: Same as `NUM_THREADS` but for timing jobs (default
  `"0"`)

## Evaluating

Just create a branch for each experiment you want to perform and run the CI.
You can then see the generated html page and download all the generated files.

## Remaining TODOs

- [ ] Make this work with any runner, not just a custom "shell" one. Consider:
  - while having a single runner use a single build job first instead of parallelizing
    We don't use a build artifact because we don't want to share the build among runners for now (due to -march=native, maybe using march=x86-64-v3 would be reasonable?)
    the basalt_vio binary is relatively small, so we could download in each parallel job in the future
- [ ] Report metrics per evalset and not just all together
