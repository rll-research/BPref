for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_PPO.py --env metaworld_sweep-into-v2 --seed $seed --lr 0.00003 --batch-size 256 --n-envs 32 --ent-coef 0.0 --n-steps 250 --total-timesteps 4000000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92
done
