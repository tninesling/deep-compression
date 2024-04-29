import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

prune_amount = 0.2


def count_unpruned_weights(model):
    unpruned_weights_count = 0

    # Loop over all the layers in the model
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            unpruned_weights_count += len(module.weight.data.nonzero())

    print(f"Total unpruned weights: {unpruned_weights_count}")


def no_prune(model):
    return model


def l1_unstructured_prune(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_amount)
            prune.remove(module, "weight")
    return model


def random_unstructured_prune(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.random_unstructured(module, name="weight", amount=prune_amount)
            prune.remove(module, "weight")
    return model


def global_unstructured_prune(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((getattr(model, name), "weight"))
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, "weight")
    return model


def ln_structured_prune(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module, name="weight", amount=0.1, n=float("-inf"), dim=0
            )
            prune.remove(module, "weight")
    return model


def _ln_structured_prune(model, amount, n):
    count_unpruned_weights(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=0)
            prune.remove(module, "weight")
    count_unpruned_weights(model)
    return model


def l1_structured_prune_0_1(model):
    return _ln_structured_prune(model, 0.1, 1)


def l1_structured_prune_0_05(model):
    return _ln_structured_prune(model, 0.05, 1)


def l2_structured_prune_0_1(model):
    return _ln_structured_prune(model, 0.1, 2)


def l2_structured_prune_0_05(model):
    return _ln_structured_prune(model, 0.05, 2)


def l3_structured_prune_0_1(model):
    return _ln_structured_prune(model, 0.1, 3)


def l3_structured_prune_0_05(model):
    return _ln_structured_prune(model, 0.05, 3)
