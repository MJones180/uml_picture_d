"""
This script trains a neural network model.

Portions of the code from this file are adapted from:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

The training and validation dataset must have their inputs pre-normalized.
"""

from time import time
import torch
from torchvision.transforms import v2
from utils.constants import (ARGS_F, DS_RAW_INFO_F, EPOCH_LOSS_F,
                             LOSS_FUNCTIONS, NORM_F, OPTIMIZERS, OUTPUT_P,
                             PROC_DATA_P, TAG_LOOKUP_F, TRAINED_MODELS_P)
from utils.json import json_load, json_write
from utils.load_network import load_network
from utils.model import Model
from utils.path import (copy_files, delete_dir, delete_file, make_dir,
                        path_exists)
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.torch_hdf_ds_loader import DSLoaderHDF


def model_train_parser(subparsers):
    """
    Example commands:
        python3 main.py model_train \
            v1a train_fixed_10nm_gl val_fixed_10nm_gl \
            test mae adam 1e-3 250 \
            --batch-size 64 --overwrite-existing \
            --only-best-epoch --early-stopping 10
    """
    subparser = subparsers.add_parser(
        'model_train',
        help='train a new model',
    )
    subparser.set_defaults(main=model_train)
    subparser.add_argument(
        'tag',
        help='unique tag given to this model under which epochs will be saved',
    )
    subparser.add_argument(
        'training_ds',
        help='name of the training dataset',
    )
    subparser.add_argument(
        'validation_ds',
        help='name of the validation dataset',
    )
    shared_argparser_args(subparser, ['network_name'])
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
    subparser.add_argument(
        '--early-stopping',
        type=int,
        metavar='n',
        help=('stop training if performance does not improve after n epochs, '
              'this is based on the validation loss'),
    )
    subparser.add_argument(
        '--loss-improvement-threshold',
        type=float,
        help='threshold to determine if loss has improved',
    )
    subparser.add_argument(
        '--max-threads',
        type=int,
        help='limit the number of threads PyTorch can use',
    )
    subparser.add_argument(
        '--init-weights',
        nargs=2,
        metavar=('[pretrained model tag]', '[pretrained model epoch]'),
        help=('init the weights from a pretrained model, the network '
              'structure must be the same'),
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

    max_threads = cli_args['max_threads']
    if max_threads:
        step_ri('Limiting available PyTorch threads')
        torch.set_num_threads(max_threads)
        print(f'Set to {max_threads} threads')

    step_ri('Creating the new model directory')
    tag = cli_args['tag']
    print(f'Tag: {tag}')
    output_model_path = f'{TRAINED_MODELS_P}/{tag}'
    if cli_args['overwrite_existing']:
        print('Deleting old model if one exists')
        delete_dir(output_model_path)
    make_dir(output_model_path)

    step_ri('Loading in the training and validation datasets')

    def _fetch_loader(arg):
        return DSLoaderHDF(cli_args[arg])

    train_dataset = _fetch_loader('training_ds')
    validation_dataset = _fetch_loader('validation_ds')

    step_ri('Copying over the normalization values from the training dataset')
    train_ds_tag = cli_args['training_ds']
    train_ds_folder = f'{PROC_DATA_P}/{train_ds_tag}'
    copy_files(f'{train_ds_folder}/{NORM_F}', f'{output_model_path}/{NORM_F}')

    step_ri('Copying over the raw ds values from the training dataset')
    copy_files(f'{train_ds_folder}/{DS_RAW_INFO_F}',
               f'{output_model_path}/{DS_RAW_INFO_F}')

    step_ri('Saving all CLI args')
    json_write(f'{output_model_path}/{ARGS_F}', cli_args)

    step_ri('Creating the torch `DataLoader` for training and validation')
    batch_size = cli_args['batch_size']
    drop_last = cli_args.get('drop_last', False)
    shuffle = not cli_args.get('disable_shuffle', False)
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

    init_weights = cli_args.get('init_weights')
    if init_weights is not None:
        step_ri('Initializing weights from trained model')
        print(f'Trained model: {init_weights[0]}, Epoch: {init_weights[1]}')
        pt_model = Model(*init_weights, suppress_logs=True).get_model()
        model.load_state_dict(pt_model.state_dict())

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
    if cli_args.get('randomly_flip_images'):
        image_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    print(image_transforms)

    step_ri('Creating a CSV file to track loss')
    loss_file = f'{output_model_path}/{EPOCH_LOSS_F}'
    with open(loss_file, 'w') as loss_writer:
        loss_writer.write('epoch, training_loss, validation_loss')

    step_ri('Writing to the tag lookup')
    tag_lookup_path = f'{OUTPUT_P}/{TAG_LOOKUP_F}'
    # Load in the current lookup if one already exists
    if path_exists(tag_lookup_path):
        tag_lookup = json_load(tag_lookup_path)
    else:
        tag_lookup = {}
    # Add all the cli arguments associated with this tag for easy reference
    tag_lookup[tag] = cli_args
    json_write(tag_lookup_path, tag_lookup)

    step_ri('Epoch counts')
    epoch_count = cli_args['epochs']
    print(f'Epochs: {epoch_count}')
    training_batches = len(train_loader)
    print(f'Training batches per epoch: {training_batches}')
    validation_batches = len(validation_loader)
    print(f'Validation batches per epoch: {validation_batches}')

    step_ri('Saving preferences')
    best_val_loss_epoch = 0
    best_val_loss = 1e10
    epoch_save_steps = cli_args.get('epoch_save_steps')
    only_best_epoch = cli_args['only_best_epoch']
    early_stopping = cli_args['early_stopping']
    if epoch_save_steps:
        print(f'Will save every {epoch_save_steps} epochs')
    if only_best_epoch:
        best_epoch_path = None
        print('Will save only best epoch based on the loss function')
    if early_stopping:
        print('Early stopping enabled - will stop if loss does not improve '
              f'after {early_stopping} epochs')

    step_ri('Beginning training')
    for epoch_idx in range(1, epoch_count + 1):
        step(f'EPOCH {epoch_idx}/{epoch_count}')
        start_time = time()

        # Turn gradient tracking on
        model.train(True)
        total_train_loss = 0
        for data in train_loader:
            inputs, outputs_truth = data
            if image_transforms is not None:
                inputs = image_transforms(inputs)
            # Zero gradients for every batch
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients, need to convert both inputs
            # to floats (float32) so that the backward prop works for MSE
            loss = loss_function(outputs.float(), outputs_truth.float())
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
            for data in validation_loader:
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

        threshold = cli_args.get('loss_improvement_threshold')
        if threshold is None:
            threshold = 7.5e-6
        # More stable version of = avg_val_loss < best_val_loss
        loss_improved = best_val_loss - avg_val_loss > threshold

        if epoch_save_steps is not None:
            # Always save the current epoch for progress
            _save_epoch()
            # If the last epoch isn't a checkpoint, then delete it
            last_epoch = epoch_idx - 1
            if last_epoch % epoch_save_steps:
                delete_file(f'{output_model_path}/epoch_{last_epoch}', True)
        elif only_best_epoch is True:
            if loss_improved:
                # Save the new epoch
                _save_epoch()
                # Delete the last saved epoch sense it is no longer the best
                if best_epoch_path is not None:
                    delete_file(best_epoch_path, True)
                best_epoch_path = current_epoch_path
        else:
            _save_epoch()

        print('Validation Loss: ', float(avg_val_loss))
        epochs_since_improvement = epoch_idx - best_val_loss_epoch
        difference_from_best = abs(float(best_val_loss - avg_val_loss))

        # Pretty logs for loss
        if loss_improved:
            print(f'{difference_from_best} from best')
            best_val_loss_epoch = epoch_idx
            best_val_loss = avg_val_loss
        else:
            print(f'{difference_from_best} off best')
            print('Performance has not increased in '
                  f'{epochs_since_improvement} epoch(s)')

        # Handle the early stopping
        if early_stopping and epochs_since_improvement >= early_stopping:
            print('Ending training due to early stopping')
            break

        print(f'Time: {time() - start_time}')
        dec_print_indent()
