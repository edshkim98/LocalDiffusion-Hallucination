"""Train the conditional diffusion model on in-distribution data only.

Example:
    python train.py --config configs/mnist_train.yaml
"""
import argparse

from ddpm import GaussianDiffusion, Trainer
from utils import build_model, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train the conditional diffusion model.')
    parser.add_argument('--config', default='configs/mnist_train.yaml', help='path to the training config yaml')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config)

    model = build_model(config)
    diffusion = GaussianDiffusion(
        config,
        model,
        image_size=config['img_size'],
        timesteps=config['timestep'],
        beta_schedule=config.get('scheduler', 'sigmoid'),
        objective=config['pred_objective'],
        auto_normalize=False,
    )
    trainer = Trainer(
        config,
        diffusion,
        folder=None,
        train_batch_size=config.get('batch_size', 64),
        train_lr=config.get('lr', 1e-4),
        train_num_steps=config.get('train_num_steps', 100000),
        save_and_sample_every=config.get('save_and_sample_every', config['timestep'] // 4),
    )
    trainer.train()
