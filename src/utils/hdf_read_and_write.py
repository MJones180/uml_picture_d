from h5py import File


def read_hdf(path, mode='r'):
    """
    Read data from an HDF file.

    Parameters
    ----------
    path : str
        Path to the HDF file.
    mode : str
        The mode to open the HDF file in.

    Returns
    -------
    h5py object
        The HDF file's contents.
    """
    return File(path, mode)


class HDFWriteModule():

    def __init__(self, path):
        """
        Create an HDF object for easily creating and writing to an HDF file.

        Parameters
        ----------
        path : str
            Path to the HDF file (including name).
        """
        self.path = path

    def create_write_hdf(self):
        """
        Create an output HDF file to write to.

        Returns
        -------
        h5py File object
            HDF file to write to
        """
        return File(self.path, 'w')

    def create_and_write_hdf_simple(self, data):
        """
        Create an HDF file and write data to it. Will overwrite any existing
        HDF files. Data will be written row by row from the start of the file.

        Parameters
        ----------
        data : dict
            Dictionary of values to write.
        """
        with self.create_write_hdf() as file:
            for table, values in data.items():
                file[table] = values

    def create_and_write_hdf_simple_with_metadata(self, data):
        """
        Create an HDF file and write data to it. Will overwrite any existing
        HDF files. Data will be written row by row from the start of the file.
        The data dictionary must have the form of:
            { table: { 'metadata': { meta_key: str_val }, 'data': [ ... ] } }

        Parameters
        ----------
        data : dict
            Dictionary of values to write.
        """
        with self.create_write_hdf() as file:
            for table, values in data.items():
                metadata = values['metadata']
                file[table] = values['data']
                for meta_key, meta_val in metadata.items():
                    file[table].attrs[meta_key] = meta_val

    def create_and_curry_write_group(self):
        """
        Create an HDF file and return a function that will allow groups of
        data to be written.

        Returns
        -------
        Function
            Parameters
            ----------
            group_name : str
                Name of the HDF grouping
            data : dict
                Dictionary of values to write.
        """
        output = self.create_write_hdf()

        def write_data(group_name, data):
            group = output.create_group(group_name)
            for key, val in data.items():
                group[key] = val

        return write_data
