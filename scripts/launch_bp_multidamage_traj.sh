cd /home/marcello/workspace/boolean_nca_cc
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=7   # optional

python experiments/bp_multi_damage_trajectory.py \
  --run-id 7cyn12ne \
  --methods bp \
  --damage-injection-mode threshold_reinject \
  --reinject-threshold 0.99 \
  --reinject-cooldown-steps 15 \
  --damage-start-offset 6 \
  --max-damage-per-circuit 9 \
  --damage-prob 40 \
  --n-circuits 256 \
  --bp-epochs 250 \
  --reversible-bias -10