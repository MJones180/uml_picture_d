from networks.basic import Basic
import torch

random_data = torch.rand((1, 1, 31, 31))

model = Basic()
print(model)

total = 0
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        params = parameter.numel()
        print(name, params)
        total += params
print(f'Total Trainable Params: {total}')

print(model(random_data))
