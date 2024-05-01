import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import scipy.sparse as sp

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
            sparse_weight = sp.csr_matrix(module.weight.detach().numpy())
            module.weight.data = torch.from_numpy(sparse_weight.toarray())
    return model


def random_unstructured_prune(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.random_unstructured(module, name="weight", amount=prune_amount)
            prune.remove(module, "weight")
            sparse_weight = sp.csr_matrix(module.weight.detach().numpy())
            module.weight.data = torch.from_numpy(sparse_weight.toarray())
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
            sparse_weight = sp.csr_matrix(module.weight.detach().numpy())
            module.weight.data = torch.from_numpy(sparse_weight.toarray())
    return model

def ln_structured_prune(model, amount, n):
    print(f"Pruning {amount * 100}% of weights using L{n} norm")
    count_unpruned_weights(model)
    for _, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            prune.ln_structured(module, name="weight", amount=amount, n=n, dim=0)
            prune.remove(module, "weight")
            sparse_weight = sp.csr_matrix(module.weight.detach().numpy())
            module.weight.data = torch.from_numpy(sparse_weight.toarray())
    count_unpruned_weights(model)
    return model

def l1_structured_prune_one_percent(model):
    return ln_structured_prune(model, 0.01, 1)

def l1_structured_prune_two_percent(model):
    return ln_structured_prune(model, 0.02, 1)

def l1_structured_prune_three_percent(model):
    return ln_structured_prune(model, 0.03, 1)

def l1_structured_prune_four_percent(model):
    return ln_structured_prune(model, 0.04, 1)

def l1_structured_prune_five_percent(model):
    return ln_structured_prune(model, 0.05, 1)
