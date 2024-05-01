import matplotlib.pyplot as plt
import pandas as pd

"""
Model Size vs Accuracy Chart
"""
data = pd.read_csv("results.csv")
unpruned = data[data["pruning"] == "no_prune"]
# float8 and int8 end up with the same size and accuracy, so this prevents overlapping labels
not_float8 = unpruned[unpruned["quantization"] != "quanto_float8_quantize"]
models = not_float8.sort_values(by="size (MB)").groupby("model")

data_type_for_quantization = {
    "no_quantize": "float32",
    "quanto_int4_quantize": "int4",
    "quanto_int8_quantize": "int8",
    "quanto_float8_quantize": "float8",
}
marker_styles = {"lenet5": "^", "lenet300": "s", "alexnet": "o", "vgg16": "D"}

fig, ax = plt.subplots()
for name, model in models:
    sizes = model["size (MB)"].values
    accuracies = model["accuracy % (top5)"].values
    quantizations = model["quantization"].values
    data_types = list(map(lambda q: data_type_for_quantization[q], quantizations))

    ax.plot(sizes, accuracies, marker=marker_styles[name], label=name)
    for i in range(len(sizes)):
        ax.text(sizes[i], accuracies[i], data_types[i], ha="center")

ax.set_title("Model Size vs. Top 5 Accuracy (No Pruning)")
ax.set_xlabel("Size (MB)")
ax.set_ylabel("Accuracy (%)")

ax.legend()
plt.show()

"""
Model Size vs Accuracy Chart (LeNets only)
"""
lenets = not_float8[not_float8["model"].str.startswith("lenet")]
models = lenets.sort_values(by="size (MB)").groupby("model")

fig, ax = plt.subplots()
for name, model in models:
    sizes = model["size (MB)"].values
    accuracies = model["accuracy % (top5)"].values
    quantizations = model["quantization"].values
    data_types = list(map(lambda q: data_type_for_quantization[q], quantizations))

    ax.plot(sizes, accuracies, marker=marker_styles[name], label=name)
    for i in range(len(sizes)):
        ax.text(sizes[i], accuracies[i], data_types[i], ha="center")

ax.set_title("Model Size vs. Top 5 Accuracy (No Pruning)")
ax.set_xlabel("Size (MB)")
ax.set_ylabel("Accuracy (%)")

ax.legend()
plt.show()

"""
Model Size vs Accuracy Chart (With Pruning)
"""
models = data.sort_values(by="size (MB)").groupby(["model", "pruning"])
interesting_combinations = {
    "alexnet": [
        "l1_structured_prune_five_percent",
        "no_prune",
    ],
    "lenet5": [],
    "lenet300": [],
    "vgg16": [
        "l1_structured_prune_one_percent",
        "no_prune",
        "random_unstructured_prune",
    ],
}

fig, ax = plt.subplots()
for (name, pruning), model in models:
    if pruning not in interesting_combinations[name]:
        continue
    sizes = model["size (MB)"].values
    accuracies = model["accuracy % (top5)"].values
    quantizations = model["quantization"].values
    data_types = list(map(lambda q: data_type_for_quantization[q], quantizations))

    ax.plot(sizes, accuracies, marker=marker_styles[name], label=f"{name} + {pruning}")
    for i in range(len(sizes)):
        ax.text(sizes[i], accuracies[i], data_types[i], ha="center")

ax.set_title("Model Size vs. Top 5 Accuracy")
ax.set_xlabel("Size (MB)")
ax.set_ylabel("Accuracy (%)")

ax.legend()
plt.show()
