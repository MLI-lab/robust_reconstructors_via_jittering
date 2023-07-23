import torch
import torch.nn.functional as F

def train(dataloader, model, device, loss_fn, optimizer, make_adv, **attack_kwargs):
    model.train()
    loss_total = 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        _, pred = model(X.float(), target=y.float(), make_adv=make_adv, **attack_kwargs)
        loss = loss_fn(pred, (y, X, model))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return loss_total / len(dataloader)

def test(dataloader, model, device, loss_fn, make_adv, **attack_kwargs):
    model.eval()
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        inp, pred = model(X.float(), target=y.float(), make_adv=make_adv, **attack_kwargs)
        test_loss += loss_fn(pred, (y, X, model)).item() 
    return test_loss / len(dataloader)