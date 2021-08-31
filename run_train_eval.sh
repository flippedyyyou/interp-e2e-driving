CUDA_VISIBLE_DEVICES=0 python3 train_eval.py \
  --root_dir logs \
  --experiment_name latent_sac2 \
  --gin_file params.gin \
  --gin_param load_carla_env.port=2000
