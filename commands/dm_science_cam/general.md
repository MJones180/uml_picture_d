# Commands Lookup

## Data Simulation

Poke single actuators at a time on both DMs:

    python3 main_scnp.py sim_data single_actuator_poke_test v84_DMs_to_sci_cam 600e-9 \
        --single-actuator-pokes 50e-9 0 \
        --no-aberrations 1 1

Poke single actuators at a time on both DMs and add in small aberrations on each row:

    python3 main_scnp.py sim_data test_dataset v84_DMs_to_sci_cam 600e-9 \
        --single-actuator-pokes 50e-9 0 \
        --rand-amount-per-zernike 1904 2 24 " -1e-9" 1e-9 \
        --aberrations-and-dm-union
