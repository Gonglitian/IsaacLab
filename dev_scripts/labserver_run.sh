CUDA_VISIBLE_DEVICES=3,4 ./isaaclab.sh -p -m torch.distributed.run --standalone --nproc_per_node=2 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Reach-Direct-v0 \
    --num_envs 60000 \
    --max_iterations 10000 \
    --headless \
    agent.save_interval=100 \
    --distributed

# RESUME
# CUDA_VISIBLE_DEVICES=3,4 ./isaaclab.sh -p -m torch.distributed.run --standalone --nproc_per_node=2 \
#     scripts/reinforcement_learning/rsl_rl/train.py \
#     --task Isaac-G1-Reach-Direct-v0 \
#     --num_envs 60000 \
#     --max_iterations 10000 \
#     --headless \
#     agent.save_interval=100 \
#     --distributed \
#     --resume \
#     --load_run <run-folder-name> \
#     --checkpoint <ckpt-path>