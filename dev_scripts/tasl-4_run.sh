./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Reach-Direct-v0 \
    --num_envs 25600 \
    --headless \
    --max_iterations 1500 \
    agent.num_steps_per_env=128 \
    agent.algorithm.num_mini_batches=8 \
    agent.algorithm.learning_rate=5.0e-2 \
    agent.algorithm.num_learning_epochs=20


# RESUME
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#     scripts/reinforcement_learning/rsl_rl/train.py \
#     --task Isaac-G1-Reach-Direct-v0 \
#     --num_envs 25600 \
#     --headless \
#     --max_iterations 1500 \
#     agent.num_steps_per_env=128 \
#     agent.algorithm.num_mini_batches=80000 \
#     agent.algorithm.learning_rate=5.0e-2 \
#     --resume \
#     --load_run <run-folder-name> \
#     --checkpoint <ckpt-path>