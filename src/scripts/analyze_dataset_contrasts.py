import numpy as np
from utils.constants import DATA_F, RANDOM_P, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir
from utils.plots.plot_dh_contrast import plot_dh_contrast
from utils.printing_and_logging import step_ri, title


def analyze_dataset_contrasts_parser(subparsers):
    subparser = subparsers.add_parser(
        'analyze_dataset_contrasts',
        help='analyze the dark hole contrasts in a dataset',
    )
    subparser.set_defaults(main=analyze_dataset_contrasts)
    subparser.add_argument(
        'raw_data_tag',
        help='tag of the raw dataset; will only use first datafile',
    )
    subparser.add_argument(
        'mask_tag',
        help='tag of the raw dataset containing the dark zone mask',
    )
    subparser.add_argument(
        'mask_table_name',
        help='dark zone mask table name',
    )
    subparser.add_argument(
        'unocc_tag',
        help=('tag of the raw dataset containing the unocc intensity image; '
              'there should be no error in the system'),
    )
    subparser.add_argument(
        'unocc_table_name',
        help='unocc image table name',
    )
    subparser.add_argument(
        'tot_iterations',
        type=int,
        help='total number of iterations per DH',
    )
    subparser.add_argument(
        'n_iteration',
        type=int,
        help='iteration number to look at the contrasts for (1-indexed)',
    )
    subparser.add_argument(
        '--filter-contrasts',
        nargs=2,
        help=('calculate the number of rows which meet the contrast '
              'requirement; two arguments expected: threshold (in log units) '
              '[max_pixel, mean]'),
    )
    subparser.add_argument(
        '--save-filter-mask',
        help=('should be used with the `--filter-contrasts` argument; save '
              'the filter mask as a raw dataset; one argument expected: tag'),
    )
    subparser.add_argument(
        '--plot-contrasts',
        nargs='*',
        help='plot the contrast; arguments expected: vmin, vmax, *idxs to plot',
    )


def analyze_dataset_contrasts(cli_args):
    title('Analyze dataset contrasts script')

    step_ri('Loading intensity')
    raw_data_tag = cli_args['raw_data_tag']
    print(f'Tag: {raw_data_tag}')
    modes_path = raw_sim_data_chunk_paths(raw_data_tag)[0]
    print(f'Modes path: {modes_path}')
    tot_iterations = cli_args['tot_iterations']
    print(f'Iterations per DH: {tot_iterations}')
    n_iteration = cli_args['n_iteration']
    print(f'Will use iteration (1-indexed): {n_iteration}')
    print('Calculating intensity from `sci_r` and `sci_i`')
    data = read_hdf(modes_path)
    intensity = (data['sci_r'][n_iteration - 1::tot_iterations]**2 +
                 data['sci_i'][n_iteration - 1::tot_iterations]**2)
    print(f'Intensity shape: {intensity.shape}')

    step_ri('Loading mask')
    mask_tag = cli_args['mask_tag']
    print(f'Tag: {mask_tag}')
    mask_path = raw_sim_data_chunk_paths(mask_tag)[0]
    print(f'Mask path: {mask_path}')
    mask_table_name = cli_args['mask_table_name']
    print(f'Mask table name: {mask_table_name}')
    mask_data = read_hdf(mask_path)[mask_table_name][:]
    print(f'Mask shape: {mask_data.shape}')

    step_ri('Applying mask')
    intensity = intensity[:, mask_data]
    print(f'Intensity shape: {intensity.shape}')

    step_ri('Loading unocc image')
    unocc_tag = cli_args['unocc_tag']
    print(f'Tag: {unocc_tag}')
    unocc_path = raw_sim_data_chunk_paths(unocc_tag)[0]
    print(f'Unocc path: {unocc_path}')
    unocc_table_name = cli_args['unocc_table_name']
    print(f'Unocc table name: {unocc_table_name}')
    unocc_data = read_hdf(unocc_path)[unocc_table_name][:]
    print(f'Unocc shape: {unocc_data.shape}')

    step_ri('Normalizing')
    unocc_peak_intensity = np.max(unocc_data)
    print(f'Peak intensity from unocc: {unocc_peak_intensity}')
    print('Dividing intensity by peak')
    intensity = intensity / unocc_peak_intensity

    step_ri('Max Pixel Contrast / DH')
    max_per_dh = np.max(intensity, axis=1)
    print(f'Global Arithmetic DH Mean: {np.log10(np.mean(max_per_dh))}')
    print(f'Global Darkest Pixel (Min): {np.log10(np.min(max_per_dh))}')
    print(f'Global Brightest Pixel (Max): {np.log10(np.max(max_per_dh))}')

    step_ri('Arithmetic Mean Contrast / DH')
    avg_per_dh = np.mean(intensity, axis=1)
    print(f'Global DH Mean: {np.log10(np.mean(avg_per_dh))}')
    print(f'Deepest DH (Min): {np.log10(np.min(avg_per_dh))}')
    print(f'Brightest DH (Max): {np.log10(np.max(avg_per_dh))}')

    filter_contrasts = cli_args.get('filter_contrasts')
    if filter_contrasts is not None:
        step_ri('Filtering contrasts')
        threshold, filter_type = filter_contrasts
        threshold = float(threshold)
        print(f'Threshold: {threshold}')
        base_filter_str = 'Will filter based on'
        if filter_type == 'max_pixel':
            print(f'{base_filter_str} max value in each DH')
            values = max_per_dh
        else:
            print(f'{base_filter_str} arithmetic mean of each DH')
            values = avg_per_dh
        mask = np.log10(values) < threshold
        valid_rows = np.sum(mask)
        total_rows = intensity.shape[0]
        percent = valid_rows / total_rows * 100
        print(f'Valid: {valid_rows}/{total_rows} ({percent:0.2f}%)')
        save_filter_mask = cli_args.get('save_filter_mask')
        if save_filter_mask:
            step_ri('Saving filter mask')
            print(f'Raw tag: {save_filter_mask}')
            output_dir = f'{RAW_DATA_P}/{save_filter_mask}'
            make_dir(output_dir)
            path = f'{output_dir}/0_{DATA_F}'
            print(f'Writing to output HDF datafile: {path}')
            out_data = {'mask': mask.astype(bool)}
            HDFWriteModule(path).create_and_write_hdf_simple(out_data)

    plot_contrasts = cli_args.get('plot_contrasts')
    if plot_contrasts is not None:
        step_ri('Plotting contrasts')
        vmin, vmax, *idxs_to_plot = plot_contrasts
        print(f'Plot vmin: {vmin}')
        print(f'Plot vmax: {vmax}')
        print(f'Will plot indices: {idxs_to_plot}')
        output_dir = f'{RANDOM_P}/{raw_data_tag}'
        make_dir(output_dir)
        print(f'Will output plots at: {output_dir}')
        for idx in idxs_to_plot:
            intensity_2d = np.zeros(mask_data.shape)
            intensity_2d[mask_data] = intensity[int(idx)]
            plot_dh_contrast(intensity_2d, vmin, vmax, f'Index {idx}',
                             f'{output_dir}/{idx}.png')
