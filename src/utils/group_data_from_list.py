from utils.terminate_with_message import terminate_with_message


def group_data_from_list(data, args_per_group, error_message):
    elements = len(data)
    if elements % args_per_group != 0:
        terminate_with_message(error_message)
    groups = elements // args_per_group
    grouped_data = []
    for group_idx in range(groups):
        starting_idx = group_idx * args_per_group
        ending_idx = starting_idx + args_per_group
        grouped_data.append(data[starting_idx:ending_idx])
    return grouped_data
