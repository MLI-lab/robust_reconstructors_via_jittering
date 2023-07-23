# %%
import torch

# %%
def get_loss_fn(name, kwargs):
    if name == "mse":
        return MSELossWrapper(**kwargs)
    elif name == "mse_error_reg":
        return ErrorRegularizedLoss(**kwargs, base_loss=torch.nn.MSELoss())
    else:
        print(f"Unknown loss_fn_name: {name}. Choosing MSELoss.")
        return MSELossWrapper(**kwargs)

class MSELossWrapper(torch.nn.MSELoss):
    def __init__(self):
        super(MSELossWrapper, self).__init__()

    def forward(self, prediction, targets_inputs):
        return super(MSELossWrapper, self).forward(prediction, targets_inputs[0])

class ErrorRegularizedLoss(torch.nn.Module):
    def __init__(self, base_param, reg_param, base_loss, random_row, random_row_grad_output="normal"):
        super(ErrorRegularizedLoss, self).__init__()
        self._base_param = float(base_param)
        self._reg_param = float(reg_param)
        self._base_loss = base_loss
        self._random_row = random_row
        self._random_row_grad_output = random_row_grad_output

    def _jacobian(self, y, x):
        if self._random_row:
            if self._random_row_grad_output == "one_hot":
                indices = torch.randint(low=0, high=x.shape[1], size=(x.shape[0],), device=x.device) # for each batch size an index where 1 is
                grad_outputs = torch.nn.functional.one_hot(indices, num_classes = x.shape[1])
            else:
                grad_outputs = torch.randn_like(y) #/ np.sqrt(x.shape[1])
            grad, = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, retain_graph=True, create_graph=True)
            jac = grad
        else:
            grad_outputs = torch.zeros_like(y).detach()
            jac = torch.zeros( (x.shape[0], y.shape[1], x.shape[1])).detach()
            grad_outputs = torch.zeros_like(y).detach()
            for i in range(y.shape[1]):
                grad_outputs[:,i] = 1
                grad, = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, retain_graph=True, create_graph=True)
                jac[:,i] = grad
                grad_outputs[:,i] = 0
        return jac
 
    def forward(self, prediction, targets_inputs):        
        targets, input, model = targets_inputs
        x = input.clone().detach().requires_grad_(True)
        _, prediction2 = model(x)
        return self._base_param*self._base_loss(prediction, targets) + self._reg_param * self._jacobian(prediction2, x).pow(2).mean()
