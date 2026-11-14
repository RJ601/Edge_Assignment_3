"""
prune_unstructured.py
=====================
Question 2, part 2 + 4: UNSTRUCTURED (weight-level) magnitude pruning with
iterative pruning and fine-tuning.

Idea
----
In a trained network many individual weights are close to zero and barely
affect the output. Magnitude pruning sets the SMALLEST weights (by absolute
value, "L1") to exactly zero. The model keeps the same shape; it just becomes
"sparse". Because zeros are scattered, normal CPUs still do the same amount
of work, so the benefit shows up in the number of non-zero parameters, not
in speed.

Pipeline
--------
  1. Load the trained FP32 ResNet-18 (from Question 1's baseline).
  2. Repeat for N iterations (iterative pruning):
        a. Prune a bit more, GLOBALLY across all conv + fully connected
           layers (the smallest weights anywhere in the network go first).
           Pruning creates a "mask" (1 = keep, 0 = pruned) for each layer.
        b. Fine-tune with the masks active, so the network adapts.
  3. A final, longer fine-tune.
  4. Make pruning PERMANENT: multiply each weight by its mask once and delete
     the mask, leaving an ordinary model whose pruned weights are exact zeros.
  5. Evaluate and save.

The sparsity ladder: with target 80% and 5 iterations the network is
16% -> 32% -> 48% -> 64% -> 80% sparse after iterations 1..5.

Run:  python prune_unstructured.py
"""

import torch
import torch.nn.utils.prune as prune

import config
from common import (get_train_loader, get_test_loader, evaluate, load_model,
                    set_seed, ensure_dirs, require_file)
from prune_utils import (prunable_layers, prunable_sparsity, count_parameters,
                         fine_tune)


def prune_step(layers, amount_of_remaining: float) -> None:
    """
    Prune `amount_of_remaining` of the weights that are STILL unpruned.

    global_unstructured ranks the weights of ALL listed layers together and
    zeroes the smallest ones. When called again on already-pruned layers,
    torch keeps the old mask and applies the new pruning only to the weights
    that survived, so `amount` is a fraction of what remains.
    """
    prune.global_unstructured(
        layers,
        pruning_method=prune.L1Unstructured,  # rank by absolute value
        amount=amount_of_remaining,
    )


def make_pruning_permanent(layers) -> None:
    """
    Fold each mask into its weight and delete the mask.

    Before: module.weight_orig (full values) + module.weight_mask (0/1).
    After : module.weight is a normal parameter with exact zeros.
    """
    for module, name in layers:
        prune.remove(module, name)


def main() -> None:
    set_seed()
    ensure_dirs()
    require_file(config.FP32_PATH, "Run train_baseline.py first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Unstructured pruning on: {device}")

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    # Step 1: load the trained original model
    model = load_model("fp32").to(device)
    original_accuracy = evaluate(model, test_loader, device)
    print(f"Original FP32 accuracy: {original_accuracy:.2f}%")

    # The (module, "weight") pairs that will be pruned: every conv and fc layer
    layers = [(module, "weight") for _, module in prunable_layers(model)]

    # Step 2: iterative pruning + fine-tuning
    previous_sparsity = 0.0
    for iteration in range(1, config.UNSTRUCT_ITERATIONS + 1):
        # Cumulative sparsity we want after this iteration (16%, 32%, ... 80%)
        target = config.UNSTRUCT_TARGET_SPARSITY * iteration / config.UNSTRUCT_ITERATIONS
        # Convert "reach `target` overall" into "prune this fraction of what is left"
        amount = (target - previous_sparsity) / (1.0 - previous_sparsity)
        previous_sparsity = target

        print(f"\n--- Iteration {iteration}/{config.UNSTRUCT_ITERATIONS}: "
              f"target sparsity {100 * target:.0f}% ---")
        prune_step(layers, amount)
        print(f"Sparsity now: {prunable_sparsity(model):.2f}% | accuracy right "
              f"after pruning: {evaluate(model, test_loader, device):.2f}%")

        fine_tune(model, train_loader, config.UNSTRUCT_FT_EPOCHS_PER_ITER,
                  config.FINE_TUNE_LR, device, label=f"iteration {iteration}")
        print(f"Accuracy after fine-tuning: "
              f"{evaluate(model, test_loader, device):.2f}%")

    # Step 3: final, longer fine-tune (masks still active)
    print("\n--- Final fine-tuning ---")
    fine_tune(model, train_loader, config.UNSTRUCT_FINAL_FT_EPOCHS,
              config.FINE_TUNE_LR, device, label="final")

    # Step 4: make the pruning permanent
    make_pruning_permanent(layers)

    # Step 5: evaluate and save
    final_accuracy = evaluate(model, test_loader, device)
    total, nonzero = count_parameters(model)
    print(f"\nFinal unstructured-pruned accuracy: {final_accuracy:.2f}% "
          f"(original {original_accuracy:.2f}%)")
    print(f"Sparsity of conv+fc weights: {prunable_sparsity(model):.2f}%")
    print(f"Parameters: {total:,} total, {nonzero:,} non-zero")

    torch.save(model.cpu().state_dict(), config.UNSTRUCTURED_PATH)
    print(f"Saved unstructured-pruned model to {config.UNSTRUCTURED_PATH}")


if __name__ == "__main__":
    main()
