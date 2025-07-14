# Commands Lookup

This file contains commands to simulate data that involves both DMs and the science camera.
However, data simulation is actually being done in the `piccsim` library (in IDL).
Therefore, all these commands are not being used anymore.

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

Randomly set each actuator with a valid height in the stroke range:

    # 100 rows, random heights between [10, 20] nm on DM 1, [5, 25] nm on DM 2
    python3 main_scnp.py sim_data test_dataset v84_DMs_to_sci_cam 600e-9 \
        --rand-actuator-heights 100 10e-9 20e-9 5e-9 25e-9 \
        --no-aberrations 1 1

Poke specific actuators:

    # Two actuators on the first DM with heights of 20 nm
    python3 main_scnp.py sim_data test_dataset v84_DMs_to_sci_cam 600e-9 \
        --explicit-actuator-heights 0 10 20e-9 0 300 20e-9 \
        --no-aberrations 1 1

    # No actuators being poked
    python3 main_scnp.py sim_data no_aberrations_dm_cam v84_DMs_to_sci_cam 600e-9 \
        --explicit-actuator-heights 0 0 0 --no-aberrations 1 1 \
        --save-full-intensity --save-full-ef --save-plots True True False False
