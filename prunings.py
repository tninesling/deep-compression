import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

prune_amount = 0.2

def no_prune(model):
    return model
    
def l1_unstructured_prune(model):
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      prune.l1_unstructured(module, name="weight", amount=prune_amount)
      prune.remove(module, 'weight')
  return model
  
def random_unstructured_prune(model):
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      prune.random_unstructured(module, name="weight", amount=prune_amount)
      prune.remove(module, 'weight')
  return model
  
def global_unstructured_prune(model):
  parameters_to_prune = []
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        parameters_to_prune.append((getattr(model, name), 'weight'))
  prune.global_unstructured(
      parameters_to_prune,
      pruning_method=prune.L1Unstructured,
      amount=prune_amount
  )
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')
  return model

def ln_structured_prune(model):
  for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
      prune.ln_structured(module, name="weight", amount=prune_amount, n=2, dim=0)
      prune.remove(module, 'weight')
  return model

