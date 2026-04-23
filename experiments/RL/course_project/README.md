# Course Project: Policy Comparison on Open Duck Mini

This folder provides a reproducible pipeline for comparing on-policy and off-policy methods.

Environment variants:
- `simple`: baseline reward (existing setup)
- `imitation`: adds reference-motion style shaping reward

## Main Track (PPO vs SAC)

### 1) Train

Run from repository root:

```bash
python experiments/RL/course_project/train_compare.py \
  --algos ppo sac \
  --env simple \
  --seeds 0 1 2 \
  --timesteps 500000 \
  --device auto \
  --run-name ppo_vs_sac_main
```

This creates:

- `experiments/RL/course_project/runs/<run-name>/models/...`
- `experiments/RL/course_project/runs/<run-name>/monitor/...`
- `experiments/RL/course_project/runs/<run-name>/tensorboard/...`

### 2) Evaluate

```bash
python experiments/RL/course_project/evaluate_compare.py \
  --run-dir experiments/RL/course_project/runs/ppo_vs_sac_main \
  --episodes 10 \
  --max-steps 2000 \
  --deterministic
```

Outputs:

- `evaluation/episodes.csv`
- `evaluation/summary.json`

### 3) Build markdown table for report

```bash
python experiments/RL/course_project/summarize_results.py \
  --episodes-csv experiments/RL/course_project/runs/ppo_vs_sac_main/evaluation/episodes.csv \
  --output experiments/RL/course_project/runs/ppo_vs_sac_main/evaluation/results_table.md
```

## Supplementary Track (A2C vs TQC)

### 1) Train

```bash
python experiments/RL/course_project/train_compare.py \
  --algos a2c tqc \
  --env simple \
  --seeds 0 1 2 \
  --timesteps 500000 \
  --device auto \
  --run-name a2c_vs_tqc_supp
```

### 2) Evaluate

```bash
python experiments/RL/course_project/evaluate_compare.py \
  --run-dir experiments/RL/course_project/runs/a2c_vs_tqc_supp \
  --episodes 10 \
  --max-steps 2000 \
  --deterministic
```

### 3) Build markdown table

```bash
python experiments/RL/course_project/summarize_results.py \
  --episodes-csv experiments/RL/course_project/runs/a2c_vs_tqc_supp/evaluation/episodes.csv \
  --output experiments/RL/course_project/runs/a2c_vs_tqc_supp/evaluation/results_table.md
```

## Suggested report metrics

- Learning curves (TensorBoard scalar `rollout/ep_rew_mean`)
- Final episode return (mean +/- std over seeds)
- Episode survival length
- Mean forward speed

## Notes

- The legacy Open Duck env uses relative model paths. `env_factory.py` handles this so scripts can be launched from repo root.
- Start with shorter training for sanity check, e.g. `--timesteps 50000`.

## Visualize Robot Behavior (MuJoCo window)

```bash
python experiments/RL/course_project/visualize_policy.py \
  --run-dir experiments/RL/course_project/runs/quick_check_all \
  --env simple \
  --algo ppo \
  --seed 0 \
  --episodes 3 \
  --max-steps 2000
```

Replace `--algo` with `sac`, `a2c`, or `tqc` to compare gaits.

To test the imitation-reward environment with PPO:

```bash
python experiments/RL/course_project/train_compare.py \
  --algos ppo \
  --env imitation \
  --seeds 0 \
  --timesteps 100000 \
  --device cpu \
  --run-name ppo_imitation_probe
```

## Record MP4

```bash
python experiments/RL/course_project/record_policy.py \
  --run-dir experiments/RL/course_project/runs/main_ppo_sac_300k \
  --env simple \
  --algo ppo \
  --seed 2 \
  --episodes 5 \
  --max-steps 2000 \
  --fps 50
```

Default output path:
- `experiments/RL/course_project/runs/<run-name>/videos/<algo>_seed_<seed>.mp4`
