from datetime import datetime
from networks.basic import Basic
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from utils.arg_types import dir_path
from utils.hdf_loader import HDFLoader


def model_train_parser(subparsers):
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
    # Batch size
    # Loss function – Simple string, no parameters needed
    # Learning rate
    # Optimizer
    # Transforms
    # Epochs
    # Epoch save steps
    #     Or, only best epochs?
    # shuffle
    # Output


def model_train(cli_args):
    pass


transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])


def ModelTrain(dataset, output, batch_size=128, shuffle=False):

    train_dataset = HDFLoader(dataset)
    # This can probably be optimized
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    # Add torchvision support

    model = Basic()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = transforms(inputs)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # # Disable gradient computation and reduce memory consumption.
        # with torch.no_grad():
        #     for i, vdata in enumerate(validation_loader):
        #         vinputs, vlabels = vdata
        #         voutputs = model(vinputs)
        #         vloss = loss_fn(voutputs, vlabels)
        #         running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss', {
            'Training': avg_loss,
            'Validation': avg_vloss
        }, epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
