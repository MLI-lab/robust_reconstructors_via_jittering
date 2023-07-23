
def train(dataloader, model, device, loss_fn, optimizer, make_adv, data_transform, **attack_kwargs):
    model.train()
    loss_total = 0
    for i, (X, Xorig, y) in enumerate(dataloader):
        X, Xorig, y = X.to(device), Xorig.to(device), y.to(device)

        if data_transform is not None:
            _, _, target, mask, mean, std = data_transform(Xorig, y, return_meta_data=True, fixed_mask=None, fixed_mean=None, fixed_std=None)
            def data_transform_fixed_mask(Xn, yn):
                return data_transform(Xn, yn, return_meta_data=False, fixed_mask=mask, fixed_mean=None, fixed_std=None)
        else:
            target = y
            data_transform_fixed_mask = None

        _, pred = model(X, target=target, make_adv=make_adv, data_transform=data_transform_fixed_mask, **attack_kwargs)

        if pred.shape != target.shape:
            print(f"Shapes do not match!")
        
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
    return loss_total / len(dataloader)

def test(dataloader, model, device, loss_fn, make_adv, data_transform, **attack_kwargs):
    model.eval()
    test_loss = 0
    for X, Xorig, y in dataloader:
        X, Xorig, y = X.to(device), Xorig.to(device), y.to(device)

        if data_transform is not None:
            _, _, target, mask, mean, std = data_transform(Xorig, y, return_meta_data=True, fixed_mask=None, fixed_mean=None, fixed_std=None)
            def data_transform_fixed_mask(Xn, yn):
                return data_transform(Xn, yn, return_meta_data=False, fixed_mask=mask, fixed_mean=None, fixed_std=None)
        else:
            target = y
            data_transform_fixed_mask = None

        _, pred = model(X, target=target, make_adv=make_adv, data_transform=data_transform_fixed_mask, **attack_kwargs)

        test_loss += loss_fn(pred, target).item()

    return test_loss / len(dataloader)