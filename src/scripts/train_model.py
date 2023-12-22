import argparse
from datetime import datetime
from h5py import File
from networks.basic import Basic
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from utils.arg_types import dir_path

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])


class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = None

    def __load_dataset_if_None__(self):
        if self.dataset is None:
            self.dataset = File(self.path, 'r')

    def __getitem__(self, index):
        self.__load_dataset_if_None__()
        # print(self.dataset['outputs'][index])
        inputs = self.dataset['inputs'][index]
        outputs = self.dataset['outputs'][index]
        return inputs, outputs

    def __len__(self):
        self.__load_dataset_if_None__()
        return len(self.dataset['inputs'])


def ModelTrain(dataset, output, batch_size=128, shuffle=False):

    train_dataset = H5Dataset(dataset)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument(
        'dataset',
        help='path to the dataset to use for training',
    )
    parser.add_argument(
        'output',
        type=dir_path,
        help='path to the output dir',
    )
    # parser.add_argument(
    #     '-n',
    #     '--networkName',
    #     metavar='',
    #     help='name of the network to use for training',
    # )
    # parser.add_argument(
    #     '-o',
    #     '--optimizer',
    #     metavar='',
    #     default='adam',
    #     help="Keras optimizer, default is 'adam'",
    # )
    # parser.add_argument(
    #     '-r',
    #     '--lr',
    #     metavar='',
    #     nargs='+',
    #     help=(
    #         'learning rate of the optimizer as an exponent of base 10 (int), '
    #         'by passing None the optimizer\'s default value will be used '
    #         '(default); lr annealing is also supported if a constant lr is not '
    #         'desired, this can be accomplished by passing three parameters in '
    #         'the format `<starting lr> <epoch steps> <change amount>` where '
    #         '`starting lr` is an exponent of base 10 (int), `epoch steps` is '
    #         'the number of epochs between updating the lr (int), and `change '
    #         'amount` is the value to update the lr by in scientific notation '
    #         '(float; write as ints – no decimal points); an example annealing '
    #         'lr is `-2 5 99e-2` – it starts with a lr of 10**-2 (0.01) and '
    #         'gets updated every 5 epochs by being multiplied by 99e-2 (0.99)'),
    # )
    # parser.add_argument(
    #     '-l',
    #     '--loss',
    #     metavar='',
    #     default='mean_squared_error',
    #     help="Keras loss, default is 'mean_squared_error'",
    # )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        metavar='',
        default=128,
        help='number of samples to train each batch, default 32',
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='shuffle at the start of each epoch',
    )
    # parser.add_argument(
    #     '-e',
    #     '--epochs',
    #     type=int,
    #     metavar='',
    #     default=100,
    #     help=('goal epoch to train until (starts at 0), default 100; '
    #           'based on total epochs trained – not the starting epoch'),
    # )
    # parser.add_argument(
    #     '-s',
    #     '--epochSaveSteps',
    #     type=int,
    #     metavar='',
    #     default=1,
    #     help=('number of steps between saving epochs, default 1; '
    #           'first and last epoch are always saved, even if program '
    #           'ends early; saving occurs with respect to the total'),
    # )
    # parser.add_argument(
    #     '-d',
    #     '--validationDataset',
    #     metavar='',
    #     help='path to the dataset to use for validation',
    # )
    ModelTrain(**vars(parser.parse_args()))

