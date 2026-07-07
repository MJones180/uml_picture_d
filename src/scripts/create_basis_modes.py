import numpy as np
import torch
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.printing_and_logging import step_ri, title


def create_basis_modes_parser(subparsers):
    subparser = subparsers.add_parser(
        'create_basis_modes',
        help='create a set of basis modes using PCA',
    )
    subparser.set_defaults(main=create_basis_modes)
    subparser.add_argument(
        'output_tag',
        help='tag to give to the modes; will be put in the raw data dir',
    )
    subparser.add_argument(
        'number_modes',
        type=int,
        help='number of basis modes to create',
    )
    subparser.add_argument(
        '--raw-data-tags',
        nargs='*',
        help='raw simulated data tags to create the basis modes with',
    )
    subparser.add_argument(
        '--table-names',
        help=('names of the tables in the raw data files; if multiple tables '
              'are given, then the data will be concat in the order passed'),
    )
    subparser.add_argument(
        '--mask',
        nargs=2,
        help=('apply a mask to the data; two args expected: '
              'mask tag; HDF table name'),
    )


def create_basis_modes(cli_args):
    title('Create basis modes script')

    step_ri('Data information')
    raw_data_tags = cli_args.get('raw_data_tags')
    table_names = cli_args.get('table_names')
    number_modes = cli_args['number_modes']
    output_tag = cli_args['output_tag']
    print(f'Will load data from: {raw_data_tags}')
    print(f'Table names: {table_names}')
    print(f'Number of modes: {number_modes}')
    print(f'Output tag: {output_tag}')

    mask = cli_args.get('mask')
    if mask:
        step_ri('Will apply a mask to the data')
        mask_tag, mask_table = mask
        mask_path = raw_sim_data_chunk_paths(mask_tag)[0]
        print(f'Loading in the mask from {mask_path} ({mask_table})')
        mask_data = read_hdf(mask_path)[mask_table][:]

    step_ri('Loading in raw datafiles')
    all_table_data = {table_name: [] for table_name in table_names}
    for raw_data_tag in raw_data_tags:
        for data_path in raw_sim_data_chunk_paths(raw_data_tag):
            print(f'Loading in data from {data_path}')
            data = read_hdf(data_path)
            for table_name in table_names:
                table_data = data[table_name][:]
                if mask:
                    table_data = table_data[..., mask_data]
                all_table_data[table_name].extend(table_data)

    step_ri('Joining together all the data')
    for table_name, table_data in table_names:
        table_data = np.array(table_data)
        print(f'{table_name}: {table_data.shape}')
    all_table_data = np.concat(*[all_table_data[name] for name in table_names],
                               axis=-1)
    print(f'Merged shape: {all_table_data.shape}')

    step_ri('Computing PCA modes')
    # Subtract off the mean
    all_table_data -= np.mean(all_table_data, axis=0)
    # The PCA modes are calculated in torch since `svd_lowrank` doesn't
    # require every mode to be computed; convert to a torch tensor
    tensor_data = torch.from_numpy(all_table_data)
    # Compute the SVD with torch
    with torch.no_grad():
        _, sing_vals, modes = torch.svd_lowrank(tensor_data, q=number_modes)
    # Convert the data back to numpy
    modes = modes.cpu().numpy()
    sing_vals = sing_vals.cpu().numpy()
    # Go from shape (pixels, modes) to (modes, pixels)
    modes = modes.T
    print(f'Modes shape: {modes.shape}')

    step_ri('Writing out modes')
    datafile_path = f'{RAW_DATA_P}/{output_tag}/0_{DATA_F}'
    print(f'Path: {datafile_path}')
    HDFWriteModule(datafile_path).create_and_write_hdf_simple({'modes': modes})
