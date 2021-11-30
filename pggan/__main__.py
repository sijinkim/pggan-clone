import argparse
from pathlib import Path

import torch
import yaml

from pggan.dataset import CelebAHQ
from pggan.models import Generator
from pggan.models import Discriminator
from pggan.worker import Trainer
from pggan.worker import Validator


def train(args):
    config_file = args.config_file
    assert config_file.exists(), f"The given config_file is not exists. {args.config_file}"

    with open(config_file) as config_io:
        config_text = config_io.read()

        hparams = yaml.load(
            stream=config_text,
            Loader=yaml.FullLoader,
        )
        print('[Hyper Params]')
        print(config_text)

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # set checkpoint path
    args.checkpoint_root = Path(args.checkpoint_root)
    if not args.checkpoint_root.exists():
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Load Datasets
    if not args.data_root.joinpath('train').exists() or not args.data_root.joinpath('valid').exists():
        raise FileNotFoundError(f"'train' or 'valid' not found in {args.data_root}")

    train_dataset = CelebAHQ(
        data_root=args.data_root,
        split='train',
    )
    valid_dataset = CelebAHQ(
        data_root=args.data_root,
        split='valid',
    )
    print(f'Dataset size:\n\tTrain: {len(train_dataset)}\n\tValid: {len(valid_dataset)}')

    # Load Models
    generator = Generator(
        **hparams['model']['generator']
    )
    generator.to(device)
    discriminator = Discriminator(
        **hparams['model']['discriminator']
    )
    discriminator.to(device)

    print(generator)
    print(discriminator)

    # Load Optimizer
    try:
        optimizer = getattr(torch.optim, hparams['optimizer']['method'])
    except AttributeError:
        raise
        # Uncomment below if custom optimizer is implemented.
        # optimizer = getattr(pggan.optimizer, hparams['optimizer']['method'])
    except Exception:
        raise

    optimizer = optimizer(
        params=[{"params": model.parameters() for model in []}],
        ** hparams['optimizer']['kwargs']
    )
    print('[Optimizer]')
    print(optimizer)

    # Load Criterion
    try:
        criterion = getattr(torch.nn, hparams['criterion']['method'])
    except AttributeError:
        raise
        # Uncomment below if custom criterion is implemented.
        # criterion = getattr(pggan.criterion, hparams['criterion']['method'])
    except Exception:
        raise AttributeError

    criterion = criterion(
        ** hparams['criterion']['kwargs']
    )
    print('[Loss Function]')
    print(criterion)

    start_epoch = 1

    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)
        assert args.checkpoint.exists(), "Given checkpoint not exists."

        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['model']['generator'])
        discriminator.load_state_dict(checkpoint['model']['discriminator'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    trainer = Trainer(
        dataset=train_dataset,
        generator=generator,
        discriminator=discriminator,
        optimizer=optimizer
    )
    validator = Validator(
        dataset=valid_dataset,
    )
    for epoch in range(start_epoch, args.epoch):
        train_loss = 0
        num_batches = len(train_dataset) / hparams['batch_size']
        for train_loss_batch in trainer.train():
            train_loss += train_loss_batch
        avg_train_loss = train_loss / num_batches
        print(avg_train_loss)

        valid_loss = 0
        num_batches = len(valid_dataset) / hparams['batch_size']
        for valid_loss_batch in validator.validate():
            valid_loss += valid_loss_batch
        avg_valid_loss = valid_loss / num_batches

        if validator.is_best_score(avg_valid_loss):
            state = trainer.get_state()
            state['epoch'] = epoch
            torch.save(state, args.checkpoint_root.joinpath('best.pt'))

        if args.checkpoint_period % epoch == 0:
            state = trainer.get_state()
            state['epoch'] = epoch
            torch.save(state, args.checkpoint_root.joinpath(f'{epoch:04d}.pt'))


def inference(args):
    pass


def main():
    parser = argparse.ArgumentParser(
        prog="ProgressiveGAN"
    )
    sub_parser = parser.add_subparsers()

    train_parser = sub_parser.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('--epoch', type=int, required=True)
    train_parser.add_argument('--config_file', type=Path, default='./config_default.yml')
    train_parser.add_argument('--data_root', type=Path, required=True)
    train_parser.add_argument('--checkpoint_root', type=Path)
    train_parser.add_argument('--checkpoint_period', type=Path)
    train_parser.add_argument('--checkpoint', type=Path)

    inference_parser = sub_parser.add_parser('inference')
    inference_parser.set_defaults(func=inference)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
