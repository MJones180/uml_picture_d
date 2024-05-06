"""
This script trains a neural network model.

Portions of the code from this file are adapted from:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
"""

import torch
from torchvision.transforms import v2
from utils.hdf_loader import HDFLoader
from utils.json import json_write
from utils.load_network import load_network
from utils.path import (copy_files, delete_dir, delete_file, make_dir,
                        path_parent)
from utils.printing_and_logging import step_ri, title

# Constants for the different available loss and optimizers functions.
# Each value should correspond to the function's name in PyTorch.
# nn.<loss_function>
LOSS_FUNCTIONS = {
    'mae': 'L1Loss',
    'mse': 'MSELoss',
}
# Optimizers are currently restricted to the learning rate (`lr`) parameter.
# torch.optim.<optimizer_function>
OPTIMIZERS = {
    'adagrad': 'Adagrad',
    'adam': 'Adam',
    'rmsprop': 'RMSprop',
    'sgd': 'SGD',
}


def model_train_parser(subparsers):
    """
    Example commands:
    python3 main.py model_train \
        v1a training_03_05_global validation_03_05_global \
        test mae adam 1e-3 250 \
        --batch-size 64 --overwrite-existing --only-best-epoch
    python3 main.py model_train \
        v1a training_03_05_ind validation_03_05_ind \
        test mae adam 1e-3 250 \
        --batch-size 64 --overwrite-existing --only-best-epoch
    """
    subparser = subparsers.add_parser(
        'model_train',
        help='train a new model',
    )
    subparser.set_defaults(main=model_train)
    subparser.add_argument(
        'tag',
        help=('unique tag given to this model, all epochs will be saved under'
              'this tag in the `/output/trained_models` directory'),
    )
    subparser.add_argument(
        'training_dataset_name',
        help='name of the training dataset, will look in `/data/processed/`',
    )
    subparser.add_argument(
        'validation_dataset_name',
        help='name of the validation dataset, will look in `/data/processed/`',
    )
    subparser.add_argument(
        'network_name',
        help=('name of the python script containing the network (without the '
              '`.py`), must be located in the `/src/networks` folder'),
    )
    subparser.add_argument(
        'loss',
        type=str.lower,
        choices=LOSS_FUNCTIONS.keys(),
        help='loss function to use',
    )
    subparser.add_argument(
        'optimizer',
        type=str.lower,
        choices=OPTIMIZERS.keys(),
        help='optimizer function to use',
    )
    subparser.add_argument(
        'learning_rate',
        type=float,
        help='learning rate of the optimizer',
    )
    subparser.add_argument(
        'epochs',
        type=int,
        help='total number of epochs to train for',
    )
    subparser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='number of samples per batch',
    )
    subparser.add_argument(
        '--disable-shuffle',
        action='store_true',
        help='do not shuffle data at the start of each epoch',
    )
    subparser.add_argument(
        '--drop-last',
        action='store_true',
        help='drop the last incomplete batch for each epoch if there is one',
    )
    subparser.add_argument(
        '--overwrite-existing',
        action='store_true',
        help='if existing model with tag, delete before training',
    )
    subparser.add_argument(
        '--randomly-flip-images',
        action='store_true',
        help=('50%% chance of flipping images horizontally and '
              '50%% chance of flipping images vertically'),
    )
    epoch_save_group = subparser.add_mutually_exclusive_group()
    epoch_save_group.add_argument(
        '--epoch-save-steps',
        type=int,
        metavar='n',
        help='every n epochs and the most recent epoch will be saved',
    )
    epoch_save_group.add_argument(
        '--only-best-epoch',
        action='store_true',
        help=('only the best epoch as based on the validation '
              'dataset will be saved'),
    )


def model_train(cli_args):
    title('Model train script')

    step_ri('Creating the new model directory')
    tag = cli_args['tag']
    print(f'Tag: {tag}')
    output_model_path = f'../output/trained_models/{tag}'
    if cli_args['overwrite_existing']:
        print('Deleting old model if one exists')
        delete_dir(output_model_path)
    make_dir(output_model_path)

    step_ri('Loading in the training and validation datasets')

    def _fetch_loader(arg):
        return HDFLoader(f'../data/processed/{cli_args[arg]}/data.h5')

    train_dataset = _fetch_loader('training_dataset_name')
    validation_dataset = _fetch_loader('validation_dataset_name')

    step_ri('Copying over the normalization values from the training dataset')
    copy_files(f'{path_parent(train_dataset.get_path())}/norm.json',
               f'{output_model_path}/norm.json')

    step_ri('Saving all CLI args')
    json_write(f'{output_model_path}/args.json', cli_args)

    step_ri('Creating the torch `DataLoader` for training and validation')
    batch_size = cli_args['batch_size']
    drop_last = cli_args['drop_last']
    shuffle = not cli_args['disable_shuffle']
    print(f'Batch size: {batch_size}')
    print(f'Drop last: {drop_last}')
    print(f'Shuffle: {shuffle}')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )
    print('Validation dataset will not shuffle')
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=False,
    )

    step_ri('Loading the network')
    network_name = cli_args['network_name']
    model = load_network(network_name)()
    print(model)

    step_ri('Setting the loss function')
    loss_name = cli_args['loss']
    loss_function = getattr(torch.nn, LOSS_FUNCTIONS[loss_name])()
    print(f'{loss_name} [{loss_function}]')

    step_ri('Setting the optimizer')
    optimizer = getattr(torch.optim, OPTIMIZERS[cli_args['optimizer']])
    # Currently, the only configurable parameter for each optimizer is the lr
    optimizer = optimizer(model.parameters(), lr=cli_args['learning_rate'])
    print(optimizer)

    step_ri('Setting the image transforms')
    image_transforms = None
    if cli_args['randomly_flip_images']:
        image_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    print(image_transforms)

    step_ri('Creating CSV to track loss')
    loss_file = f'{output_model_path}/epoch_loss.csv'
    with open(loss_file, 'w') as loss_writer:
        loss_writer.write('epoch, training_loss, validation_loss')

    step_ri('Preparing to train')
    epoch_count = cli_args['epochs']
    print(f'Epochs: {epoch_count}')
    training_batches = len(train_loader)
    print(f'Training batches per epoch: {training_batches}')
    validation_batches = len(validation_loader)
    print(f'Validation batches per epoch: {validation_batches}')
    epoch_save_steps = cli_args['epoch_save_steps']
    only_best_epoch = cli_args['only_best_epoch']
    if epoch_save_steps:
        print(f'Will save every {epoch_save_steps} epochs')
    if only_best_epoch:
        best_epoch_path = None
        best_val_loss = 1e10
        print('Will only save best epoch')
    print()

    for epoch_idx in range(1, epoch_count + 1):
        print(f'EPOCH {epoch_idx}/{epoch_count}')

        # Turn gradient tracking on
        model.train(True)
        total_train_loss = 0
        for i, data in enumerate(train_loader):
            inputs, outputs_truth = data
            if image_transforms is not None:
                inputs = image_transforms(inputs)
            # Zero gradients for every batch
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = loss_function(outputs, outputs_truth)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / training_batches

        # Set the model to evaluation mode (disables dropout)
        model.eval()

        total_val_loss = 0
        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, data in enumerate(validation_loader):
                inputs, outputs_truth = data
                outputs = model(inputs)
                loss = loss_function(outputs, outputs_truth)
                total_val_loss += loss
        avg_val_loss = total_val_loss / validation_batches

        with open(loss_file, 'a+') as loss_writer:
            out_line = f'\n{epoch_idx}, {avg_train_loss}, {avg_val_loss}'
            loss_writer.write(out_line)

        current_epoch_path = f'{output_model_path}/epoch_{epoch_idx}'

        def _save_epoch():
            torch.save(model.state_dict(), current_epoch_path)

        if epoch_save_steps is not None:
            # Always save the current epoch for progress
            _save_epoch()
            # If the last epoch isn't a checkpoint, then delete it
            last_epoch = epoch_idx - 1
            if last_epoch % epoch_save_steps:
                delete_file(f'{output_model_path}/epoch_{last_epoch}', True)
        elif only_best_epoch is True:
            if avg_val_loss < best_val_loss:
                # Save the new epoch
                _save_epoch()
                # Delete the last saved epoch sense it is no longer the best
                if best_epoch_path is not None:
                    delete_file(best_epoch_path, True)
                best_epoch_path = current_epoch_path
                best_val_loss = avg_val_loss
        else:
            _save_epoch()
