import torch
import torch.nn.functional as F

class AttackerStep:
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        raise NotImplementedError

    def step(self, x, g):
        raise NotImplementedError

    def random_perturb(self, x):
        raise NotImplementedError

    def get_best(self, x, loss, x_new, loss_new):
        loss_per_ex = torch.sum(loss.view(x.shape[0], -1), dim=1)
        loss_new_per_ex = torch.sum(loss_new.view(x_new.shape[0], -1), dim=1)
        replace = loss_per_ex < loss_new_per_ex
        replace_e = replace.view(-1, *([1]*(len(x.shape)-1))).expand(x.shape)
        return torch.where(replace_e, x_new, x), torch.where(replace_e, loss_new, loss)

class L2Step(AttackerStep):
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return self.orig_input + diff

    def step(self, x, g):
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x, mode="uniform_in_sphere"):
        if mode == "uniform_in_sphere":
            l = len(x.shape) - 1
            rp = torch.randn_like(x)
            rp2 = rp.view(x.shape[0], -1)
            u = torch.rand(rp2.shape[0], device=x.device)
            n = rp2.shape[-1]
            perturb = rp * torch.pow(u, 1.0/n).view(-1, *([1]*l)) / torch.norm(rp2, dim=1).view(-1, *([1]*l))
            return x + self.eps*perturb
        elif mode == "uniform_on_sphere":
            l = len(x.shape) - 1
            rp = torch.randn_like(x)
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
            return x + self.eps * rp / (rp_norm + 1e-10)
        else:
            raise Exception(f"mode: {mode} not supported for L2-type perturbations.")


class LinfStep(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return self.orig_input + diff

    def step(self, x, g):
        return x + self.step_size * torch.sign(g)

    def random_perturb(self, x, mode):
        return x + 2 * (torch.rand_like(x) - 0.5) * self.eps

STEPS = {
    '2': L2Step,
    'inf' : LinfStep
}

class AttackerModel(torch.nn.Module):
    def __init__(self, model):
        super(AttackerModel, self).__init__()
        self.model = model

    def forward(self, x, target=None, make_adv=False, constraint="2", eps=0.1, step_size=0.1, iterations=10, 
                random_start=False, random_restarts=5, random_mode="uniform_in_sphere", use_best=False, restart_use_best=True, data_transform=None):

        if make_adv:
            prev_training = bool(self.training)
            self.eval()

            ###### (adv. training begin)
            orig_input = x.clone()
            criterion = torch.nn.MSELoss(reduction='none')

            step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
            step = step_class(eps=eps, orig_input=orig_input, step_size=step_size) 

            def calc_loss(inp, target):
                if data_transform is not None:
                    inp, target = data_transform(inp, target)
                output = self.model(inp)
                if output.shape != target.shape:
                    #output = F.interpolate(output, target.shape[-2:])
                    target = F.interpolate(target, output.shape[-2:])
                return criterion(output, target), output

            # Main function for making adversarial examples
            def get_adv_examples(x, losses):
                # keeping track of best candidates
                best_x = x.clone().detach()
                best_loss = losses.clone().detach()

                # Random start (to escape certain types of gradient masking)
                if random_start:
                    x = step.random_perturb(x, random_mode)
                    losses, _ = calc_loss(x, target)
                    best_x, best_loss = step.get_best(best_x, best_loss, x, losses) if use_best else (x, losses)

                # PGD iterates
                for i in range(iterations):
                    x = x.clone().detach().requires_grad_(True)
                    losses, _ = calc_loss(x, target)
                    assert losses.shape[0] == x.shape[0], \
                            'Shape of losses must match input!'

                    loss = torch.mean(losses)

                    grad, = torch.autograd.grad(loss, [x])

                    with torch.no_grad():
                        best_x, best_loss = step.get_best(best_x, best_loss, x, losses) if use_best else (x, losses)

                        x = step.step(x, grad)
                        x = step.project(x)

                losses, _ = calc_loss(x, target)
                best_x, best_loss = step.get_best(best_x, best_loss, x, losses) if use_best else (x, losses)
                if torch.any(losses != best_loss):
                    print(f"Loss improved from {losses.sum()} to {best_loss.sum()}")

                return best_x.clone().detach(), best_loss.clone().detach()

            x_init = x.clone().detach()
            loss_init, _ = calc_loss(x_init, target)

            if random_restarts:
                best_x = x_init
                best_loss = loss_init

                for _ in range(random_restarts):
                    new_x, new_loss = get_adv_examples(x_init, loss_init)
                    best_x, best_loss = step.get_best(best_x, best_loss, new_x, new_loss) if restart_use_best else (new_x, new_loss)

                adv_ret = best_x
            else:
                adv_ret,_ = get_adv_examples(x_init, loss_init)

            ###### (adv. training end)
            if(prev_training):
                self.train()
            inp = adv_ret
        else:
            inp = x

        model_input = data_transform(inp, None)[0] if data_transform is not None else inp
        return inp, self.model(model_input)