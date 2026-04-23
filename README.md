# CS 4180/5180 Final Project — PPO vs. SAC for Open Duck Mini Locomotion

**Author:** Yan Yijun &nbsp;·&nbsp; **Course:** CS 4180/5180 RL (Spring 2026) &nbsp;·&nbsp; **Branch:** `v2`

This fork of the [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) project studies on-policy vs. off-policy deep RL (PPO, SAC, A2C, TQC) for bipedal locomotion, with an additional reward-engineering ablation. **All course-project deliverables live in a single directory:**

### 📁 `experiments/RL/course_project/`

| File | Purpose |
|---|---|
| **[`report_aaai.pdf`](experiments/RL/course_project/report_aaai.pdf)** | **Final paper (AAAI format, 5 pages).** Start here. |
| [`report_aaai.tex`](experiments/RL/course_project/report_aaai.tex) / [`references.bib`](experiments/RL/course_project/references.bib) | LaTeX source |
| [`imitation_env.py`](experiments/RL/course_project/imitation_env.py) | Primary training env (reward-bugfixed imitation reward, 17 terms) |
| [`imitation_env_walk.py`](experiments/RL/course_project/imitation_env_walk.py) | Walk env variant (relaxed forward gate, reweighted for locomotion) |
| [`env_factory.py`](experiments/RL/course_project/env_factory.py) | Env registration for `simple` / `imitation` / `walk` variants |
| [`train_compare.py`](experiments/RL/course_project/train_compare.py) | Main training entry point (PPO / SAC / A2C / TQC × seeds) |
| [`evaluate_compare.py`](experiments/RL/course_project/evaluate_compare.py) | Rollout evaluation (return, ep-length, forward speed) |
| [`continue_train.py`](experiments/RL/course_project/continue_train.py) | Warm-start training from a saved SB3 checkpoint |
| [`plot_learning_curves.py`](experiments/RL/course_project/plot_learning_curves.py) | TensorBoard → PNG/PDF learning-curve plots |
| [`record_policy.py`](experiments/RL/course_project/record_policy.py) | Record a trained policy to MP4 |
| [`make_comparison_video.py`](experiments/RL/course_project/make_comparison_video.py) | Build the 2×2 PPO/SAC/A2C/TQC side-by-side video |
| [`summarize_results.py`](experiments/RL/course_project/summarize_results.py) | Build markdown evaluation tables |
| [`README.md`](experiments/RL/course_project/README.md) | Step-by-step reproduction instructions |

### 🧪 Experimental runs (under `experiments/RL/course_project/runs/`)

Every run directory contains `models/` (SB3 `.zip` checkpoints), `tensorboard/`, `monitor/`, `evaluation/` (eval JSON/CSV), `learning_curves_{return,length}.{pdf,png}`, and `videos/` with a recorded rollout.

| Run | What it contains | Referenced in report as |
|---|---|---|
| `main_ppo_sac_300k/` | PPO vs SAC, 3 seeds × 300k (primary comparison) | Table 2, Figure 1 |
| `supp_a2c_tqc_150k/` | A2C vs TQC, 3 seeds × 150k (supplementary) | Table 3, Figure 2 |
| `ppo_imitation_bugfix_1_5M/` | PPO 3 seeds × 1.5M on imitation (reward-hacking study) | Table 4 |
| `ppo_walk_1_5M/` | PPO single seed × 1.5M on walk variant (breaks reward hacking) | Table 5, Figure 3 |
| `ppo_walk_3_5M/` | PPO continued to 3.5M (speed–stability tradeoff) | Table 5, Figure 4 |
| `walk_all_algos_300k/` | SAC/A2C/TQC on walk variant (extra ablation, single seed) | Ablation disclaimer |
| plus earlier exploratory runs (`ppo_footsteps_*`, `ppo_imitation_walkfix_v{1..5}*`, `smoke_*`, …) | | Not in final report; kept for transparency |

### ▶️ Reproducing the main results

```bash
# Main: PPO vs SAC, 3 seeds × 300k
python experiments/RL/course_project/train_compare.py \
  --algos ppo sac --env imitation --seeds 0 1 2 \
  --timesteps 300000 --device cpu --run-name main_ppo_sac_300k

# Supplementary: A2C vs TQC, 3 seeds × 150k
python experiments/RL/course_project/train_compare.py \
  --algos a2c tqc --env imitation --seeds 0 1 2 \
  --timesteps 150000 --device cpu --run-name supp_a2c_tqc_150k

# Reward-ablation walking policy: PPO, 1.5M on walk variant
python experiments/RL/course_project/train_compare.py \
  --algos ppo --env walk --seeds 0 --timesteps 1500000 \
  --device cpu --run-name ppo_walk_1_5M

# Evaluate any run
python experiments/RL/course_project/evaluate_compare.py \
  --run-dir experiments/RL/course_project/runs/main_ppo_sac_300k \
  --episodes 10 --max-steps 2000 --deterministic
```

### 🎥 Demo videos

- Walking-then-falling PPO policy: https://youtu.be/DXRsKvcgk8M
- 2×2 side-by-side comparison: [`experiments/RL/course_project/figures/four_algo_comparison_labeled.mp4`](experiments/RL/course_project/figures/four_algo_comparison_labeled.mp4)

### 🔧 What changed from upstream

Reward-engineering bug fixes in `experiments/RL/new/{footsteps,simple,placo_imitate}_env.py` and `experiments/RL/course_project/imitation_env.py` (missing squares in smoothness terms, abs() trap in yaw, saturated gait scaling, unnormalized weights). See commit `b8135e9` for the full diff. These fixes apply to all algorithms in all comparisons above.

---

# Open Duck Mini v2

<table>
  <tr>
    <td> <img src="https://github.com/user-attachments/assets/2a407765-70ad-48dd-8a5d-488f82503716" alt="1" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/3b8fe350-73a9-4c9f-ad29-efc781be7aee" alt="2" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/fd7e5949-1492-4d31-851f-feaa9b695557" alt="3" width="300px" ></td>
   </tr> 
</table>

We are making a miniature version of the BDX Droid by Disney. It is about 42 centimeters tall with its legs extended.
The full BOM cost should be under $400 !

This repo is kind of a hub where we centralize all resources related to this project. This is a working repo, so there are a lot of undocumented scripts :) We'll try to clean things up at some point.


# State of sim2real

https://github.com/user-attachments/assets/58721d0f-2f95-4088-8900-a5d02f41bba7

https://github.com/user-attachments/assets/4129974a-9d97-4651-9474-c078043bb182

https://github.com/user-attachments/assets/a0afcd38-15d8-40c6-8171-a619107406b8


# Updates

> Update 02/04/2024: You can try two policies we trained : [this one](BEST_WALK_ONNX.onnx) and [this one](BEST_WALK_ONNX_2.onnx)
> Run with the following arguments :
> python v2_rl_walk_mujoco.py --onnx_model_path ~/BEST_WALK_ONNX_2.onnx

> Update 15/03/2025: join our discord server to get help or show us your duck :) https://discord.gg/UtJZsgfQGe

> Update 07/02/2025: Big progress on sim2real, see videos above :)

> Update 24/02/2025: Working hard on sim2real ! 

> Update 07/02/2025 : We are writing documentation on the go, but the design and BOM should not change drastically. Still missing the "expression" features, but they can be added after building the robot!

> Update 22/01/2025 : The mechanical design is pretty much finalized (fixing some mistakes here and there). The current version does not include all the "expression" features we want to include in the final robot (LEDs for the eyes, a camera, a speaker and a microphone). We are now working on making it walk with reinforcement learning !

# Community 

![duck_collage](https://github.com/user-attachments/assets/e240c06e-769f-4c87-b65f-189a442cf1e9)

Join our discord community ! https://discord.gg/UtJZsgfQGe

# CAD

https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo (Warning, outdated. Has not been updated in a while)

# RL stuff

We are switching to Mujoco Playground, see this [repo](https://github.com/apirrone/Open_Duck_Playground)

https://github.com/user-attachments/assets/037a1790-7ac1-4140-b154-2c901d20d5f5


## Reference motion generation for imitation learning 

https://github.com/user-attachments/assets/4cb52e17-99a5-47a8-b841-4141596b7afb

See [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

## Actuator identification 

We used Rhoban's [BAM](https://github.com/Rhoban/bam)

# BOM

https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing

Chinese: https://zihao-ai.feishu.cn/wiki/AfAtw69vRigXaRk5UkbcrAiLnJw?from=from_copylink

# Build Guide

> New : you can now use the Tnkr guide ! https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home

Chinese: https://zihao-ai.feishu.cn/wiki/space/7488517034406625281

## Print Guide

See [print_guide](docs/print_guide.md).

## Assembly Guide

See [assembly guide (incomplete)](docs/assembly_guide.md).

# Embedded runtime

This repo contains the code to run the policies on the onboard computer (Raspberry pi zero 2w) https://github.com/apirrone/Open_Duck_Mini_Runtime

# Training your own policies

If you want to train your own policies, and contribute to making the ducks walk nicely, see [this document](docs/sim2real.md)

> Thanks a lot to HuggingFace and Pollen Robotics for sponsoring this project !
