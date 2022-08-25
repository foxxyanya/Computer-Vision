import torch


class LinearClassificationModel(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearClassificationModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred