for seed in 67890 78906 89067 90678 6789; do
    python train_PEBBLE.py  env=quadruped_walk seed=$seed agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=30000 max_feedback=2000 reward_batch=$1 reward_update=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 reward_schedule=$2
done
