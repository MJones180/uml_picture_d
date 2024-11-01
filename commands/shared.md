## Shared Commands

Print out the layers, trainable neurons, and time it takes to run a network:

    # Example on `dfc3` network
    python3 main.py network_info dfc3 --benchmark 1000

Output information on a dataset:

    # Example on `train_com50nm_gl_diff` dataset
    python3 main.py dataset_info \
        train_com50nm_gl_diff \
        --verify-network-compatability dfc3 \
        --plot-example-images --plot-outputs-hist

Plot the training vs validation loss for all epochs in a trained model:

    # Example on `com50nm_gl_diff_v2_1` model
    python3 main.py plot_model_loss com50nm_gl_diff_v2_1

Prune the `tag_lookup.json` file to remove old models:

    python3 main.py prune_tag_lookup

Interactively view plots after `model_test` has been run:

    # Example on ran50nm_single_diff_v2_3 model, epoch 31 
    python3 main.py interactive_model_test_plots ran50nm_single_diff_v2_3 31 fixed_50nm_range_processed --scatter-plot 5 5 --zernike-plots 2 24

A native PyTorch model can be exported to TorchScript and ONNX by doing the following command:

    python3 main.py export_model data_groups_approx_2 last val_fixed_2000_and_random_group_ranges_approx --benchmark 5000

Generate a CSV with timesteps and Zernike coefficients to run a control loop on:

    # 1 ms steps for a total of 2 seconds, terms 2 through 24, constant amplitude on term 3
    python3 main.py gen_zernike_time_steps \
        single_term_3_10 0.001 2000 2 24 --single-zernike-constant-value 3 1e-9
