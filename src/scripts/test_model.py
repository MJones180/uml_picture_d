import json

# NEED TO DENORMALIZE THE DATA FOR TESTING.

# THIS IS SOMETHING THAT SHOULD BE READ IN FROM THE EPOCH, NOT THE DATASET,
# BUT THIS IS SOMETHING THAT CAN CHANGE LATER ONCE THINGS BECOME MORE OFFICIAL
# with open('normalization.json') as f:
#     norm_data = json.load(f)

# def min_max_denormalize(data, max_min_diff, min_x):
#     return (data * max_min_diff) + min_x

# if MIN_MAX_NORM['global_input_norm']:
#     min_x, max_min_diff, normed = _min_max_norm(input_data)
#     norm_values['input_min_x'] = min_x
#     norm_values['input_max_min_diff'] = max_min_diff
#     input_data = normed
# if MIN_MAX_NORM['global_output_norm']:
#     min_x, max_min_diff, normed = _min_max_norm(output_data)
#     # For the output normalization, it is easier if there is a norm value for
#     # every single element
#     min_x = np.repeat(min_x, output_data.shape[1])
#     max_min_diff = np.repeat(max_min_diff, output_data.shape[1])
#     norm_values['output_min_x'] = min_x
#     norm_values['output_max_min_diff'] = max_min_diff
#     output_data = normed
# if MIN_MAX_NORM['individual_output_norm']:
#     min_x, max_min_diff, normed = _min_max_norm(output_data, globally=False)
#     norm_values['output_min_x'] = min_x
#     norm_values['output_max_min_diff'] = max_min_diff
#     output_data = normed
