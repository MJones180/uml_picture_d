"""
This script trains a neural network model.

Portions of the code from this file are adapted from:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

The training and validation dataset must have their inputs pre-normalized.
"""

from time import time
import torch
from torchvision.transforms import v2
from utils.cli_args import save_cli_args
from utils.constants import (EPOCH_LOSS_F, EXTRA_VARS_F, LOSS_FUNCTIONS,
                             OPTIMIZERS, OUTPUT_P, PROC_DATA_P, TAG_LOOKUP_F,
                             TRAINED_MODELS_P)
from utils.json import json_load, json_write
from utils.load_network import load_network
from utils.model import Model
from utils.path import (copy_files, delete_dir, delete_file, make_dir,
                        path_exists)
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.torch_grab_device import torch_grab_device
from utils.torch_hdf_ds_loader import DSLoaderHDF


def model_train_parser(subparsers):
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
        '--lr-auto-annealing',
        nargs=2,
        type=float,
        help=('auto change the learning rate when progress begins to plateau; '
              'the initial learning rate is taken from the '
              '`learning_rate` arg; the learning rate is reduced each time '
              'the `early_stopping` arg is met; two variables should be '
              'passed: final learning rate, learning rate divide factor'),
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
    subparser.add_argument(
        '--init-weights-select',
        nargs='+',
        metavar=('[pretrained model tag], [pretrained model epoch], '
                 '*[layer name]'),
        help=('init the weights from a pretrained model; each layer name '
              'listed will be checked if it is contained within a layer from '
              'the models state dict, if it is, then the layer will be used'),
    )
    subparser.add_argument(
        '--transfer-learning-train-layers',
        nargs='+',
        help='freeze all layers, except passed layers, during training',
    )
    subparser.add_argument(
        '--transfer-learning-batchnorm',
        action='store_true',
        help=('can be used with the `--transfer-learning-train-layers` arg '
              'to unfreeze all `BatchNorm2d` layers'),
    )
    subparser.add_argument(
        '--save-post-training-loss',
        action='store_true',
        help=('calculate the loss of the training dataset after weights for '
              'the epoch have been finalized, this will increase computation '
              'time as all the training batches will have to be iterated '
              'over again'),
    )
    shared_argparser_args(subparser, ['force_cpu'])
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

    step_ri('Copying over the extra variables from the training dataset')
    train_ds_tag = cli_args['training_ds']
    copy_files(f'{PROC_DATA_P}/{train_ds_tag}/{EXTRA_VARS_F}',
               f'{output_model_path}/{EXTRA_VARS_F}')

    step_ri('Saving all CLI args')
    save_cli_args(output_model_path, cli_args, 'model_train')

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

    step_ri('Grabbing the device')
    device = torch_grab_device(cli_args.get('force_cpu'))

    step_ri('Loading the network')
    network_name = cli_args['network_name']
    model = load_network(network_name)().to(device)
    print(model)

    init_weights = cli_args.get('init_weights')
    if init_weights is not None:
        step_ri('Initializing weights from trained model')
        print(f'Trained model: {init_weights[0]}, Epoch: {init_weights[1]}')
        pt_model = Model(*init_weights, suppress_logs=True).model
        model.load_state_dict(pt_model.state_dict())

    init_weights_select = cli_args.get('init_weights_select')
    if init_weights_select is not None:
        step_ri('Initializing some weights from trained model')
        model_tag, model_epoch, *pt_layers = init_weights_select
        print(f'Trained model: {model_tag}, Epoch: {model_epoch}')
        pt_model = Model(model_tag, model_epoch, suppress_logs=True).model
        # Grab the state from the pretrained model
        pt_state = pt_model.state_dict()
        # The state for the new model that will be trained
        model_state = model.state_dict()
        for k, v in pt_state.items():
            for layer in pt_layers:
                # If the passed layer's name is contained in the new layer's
                # name then use its data
                if layer in k:
                    model_state[k] = pt_state[k]
                    break
        # Set the new state
        model.load_state_dict(model_state)

    transfer_learn_layers = cli_args.get('transfer_learning_train_layers')
    if transfer_learn_layers:
        step_ri('Preparing network for transfer learning')

        def _set_param_grad(params, val):
            for param in params:
                param.requires_grad = val

        print('Freezing all layers by default')
        _set_param_grad(model.parameters(), False)
        unfreeze_batchnorm = cli_args.get('transfer_learning_batchnorm')
        if unfreeze_batchnorm:
            print('Unfreezing all BatchNorm2d layers')
        print(f'Unfreezing select layers: {transfer_learn_layers}')
        for name, module in model.named_modules():
            is_batch_norm = (unfreeze_batchnorm
                             and isinstance(module, torch.nn.BatchNorm2d))
            is_unfrozen_layer = name in transfer_learn_layers
            if is_batch_norm or is_unfrozen_layer:
                _set_param_grad(module.parameters(), True)
        step('Layer frozen status')
        for name, param in model.named_parameters():
            frozen_str = 'Unfrozen' if param.requires_grad else 'Frozen'
            print(f'Layer: {name} - {frozen_str}')
        dec_print_indent()

    step_ri('Setting the loss function')
    loss_name = cli_args['loss']
    loss_function = getattr(torch.nn, LOSS_FUNCTIONS[loss_name])()
    print(f'{loss_name} [{loss_function}]')

    step_ri('Setting the optimizer')
    optimizer = getattr(torch.optim, OPTIMIZERS[cli_args['optimizer']])
    # Currently, the only configurable parameter for each optimizer is the lr
    base_learning_rate = cli_args['learning_rate']
    optimizer = optimizer(model.parameters(), lr=base_learning_rate)
    print(optimizer)

    early_stopping = cli_args['early_stopping']
    if early_stopping:
        step_ri('Early stopping enabled')
        print('Will stop if loss does not improve '
              f'after {early_stopping} epochs')

    lr_auto_annealing = cli_args.get('lr_auto_annealing')
    if lr_auto_annealing:
        step_ri('Will use learning rate annealing')
        final_lr = lr_auto_annealing[0]
        lr_scale_factor = lr_auto_annealing[1]
        print(f'Final learning rate: {final_lr}')
        print(f'Learning rate scale factor: {lr_scale_factor}')
        print('Will switch learning rate every time loss does not '
              f'improve after {early_stopping} epochs')
        upcoming_lrs = []
        next_lr = base_learning_rate / lr_scale_factor
        while next_lr > final_lr:
            upcoming_lrs.append(next_lr)
            next_lr /= lr_scale_factor
        print(f'Will use the following learning rates: {upcoming_lrs}')

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
    loss_keys = 'epoch, training_loss, validation_loss'
    save_post_training_loss = cli_args.get('save_post_training_loss')
    if save_post_training_loss:
        loss_keys += ', post_training_loss'
    with open(loss_file, 'w') as loss_writer:
        loss_writer.write(loss_keys)

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
    if epoch_save_steps:
        print(f'Will save every {epoch_save_steps} epochs')
    if only_best_epoch:
        best_epoch_path = None
        print('Will save only best epoch based on the loss function')

    step_ri('Beginning training')
    for epoch_idx in range(1, epoch_count + 1):
        step(f'EPOCH {epoch_idx}/{epoch_count}')
        start_time = time()

        # Turn gradient tracking on
        model.train(True)
        total_train_loss = 0
        for inputs, outputs_truth in train_loader:
            inputs = inputs.to(device)
            outputs_truth = outputs_truth.to(device)
            if image_transforms is not None:
                inputs = image_transforms(inputs)
            # Zero gradients for every batch
            optimizer.zero_grad(set_to_none=True)
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
            for inputs, outputs_truth in validation_loader:
                inputs = inputs.to(device)
                outputs_truth = outputs_truth.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, outputs_truth)
                total_val_loss += loss
        avg_val_loss = total_val_loss / validation_batches

        # Iterate through the training dataset again and compute the loss now
        # that the weights for this epoch have been finalized. If there is
        # dropout, the previous `avg_train_loss` will probably be higher than
        # the `avg_val_loss`
        if save_post_training_loss:
            total_post_train_loss = 0
            with torch.no_grad():
                for inputs, outputs_truth in train_loader:
                    inputs = inputs.to(device)
                    outputs_truth = outputs_truth.to(device)
                    outputs = model(inputs)
                    loss = loss_function(outputs, outputs_truth)
                    total_post_train_loss += loss
            avg_post_train_loss = total_post_train_loss / training_batches

        with open(loss_file, 'a+') as loss_writer:
            out_line = f'\n{epoch_idx}, {avg_train_loss}, {avg_val_loss}'
            if save_post_training_loss:
                out_line += f', {avg_post_train_loss}'

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
            if lr_auto_annealing and len(upcoming_lrs) > 0:
                current_lr = optimizer.param_groups[0]['lr']
                next_lr = upcoming_lrs.pop(0)
                print(f'Updating learning rate from {current_lr} -> {next_lr}')
                optimizer.param_groups[0]['lr'] = next_lr
                # This epoch may not be the best val loss, but the learning rate
                # has changed, so make this epoch a new starting point
                best_val_loss_epoch = epoch_idx
            else:
                print('Ending training due to early stopping')
                break

        print(f'Time: {time() - start_time}')
        dec_print_indent()
