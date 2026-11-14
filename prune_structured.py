"""
prune_structured.py
===================
Question 2, part 3 + 4: STRUCTURED CHANNEL pruning, architecture
reconstruction and fine-tuning.

Idea
----
Instead of zeroing single weights, structured pruning removes WHOLE
channels (filters). The network becomes physically smaller, so it needs
less memory and less computation on ordinary CPUs and GPUs.

Which channels are pruned?
--------------------------
Every ResNet-18 block looks like:

    x --> conv1 -> bn1 -> relu -> conv2 -> bn2 --> (+) --> relu
    |                                                ^
    +----------------- shortcut ---------------------+

The block's OUTPUT must keep its width, because it is added to the shortcut
and feeds the next block. But the channels INSIDE the block (the output of
conv1) can be removed freely. Removing channel c inside a block means:

    conv1 : delete filter c                 (one fewer output channel)
    bn1   : delete entry c of weight, bias, running_mean, running_var
    conv2 : delete input-channel c of every filter

Those three edits keep every tensor dimension consistent. The importance
of a channel is the L1 norm (sum of absolute values) of its conv1 filter:
filters with small weights contribute least, so they are removed first.

Reconstruction
--------------
We do NOT leave zeroed channels in a big model. We build a NEW, thinner
ResNet18CIFAR (via `mid_channels`) and copy the surviving weights into it.
A check then proves the thin model computes exactly what the original model
would compute with those channels switched off.

Pipeline
--------
  1. Load the trained FP32 ResNet-18.
  2. Rank and select channels in each block.
  3. Build the thinner model and copy the surviving weights.
  4. Verify the reconstruction.
  5. Fine-tune the thin model.
  6. Evaluate and save (the checkpoint also stores the channel counts).

Run:  python prune_structured.py
"""

import copy

import torch

import config
from common import (BasicBlock, ResNet18CIFAR, get_train_loader, get_test_loader,
                    evaluate, load_model, set_seed, ensure_dirs, require_file)
from prune_utils import count_parameters, fine_tune, channel_rows


def blocks_of(model: ResNet18CIFAR):
    """Return [(name, BasicBlock), ...] for the 8 residual blocks, in order."""
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, BasicBlock)]


def select_channels_to_keep(block: BasicBlock, prune_ratio: float) -> torch.Tensor:
    """
    Decide which inner channels of `block` survive.

    Importance of channel c = sum of |weights| of conv1's filter c.
    The (1 - prune_ratio) most important channels are kept (at least 1).

    Returns:
        A sorted tensor with the indices of the channels to keep.
    """
    importance = block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
    channels = importance.numel()
    keep_count = max(1, int(round(channels * (1.0 - prune_ratio))))
    keep = torch.topk(importance, keep_count).indices
    return torch.sort(keep).values  # sorted keeps the original channel order


def build_pruned_model(dense: ResNet18CIFAR, keep_lists: list) -> ResNet18CIFAR:
    """
    Build the thinner network and copy the surviving weights into it.

    Args:
        dense:      the original trained model.
        keep_lists: one tensor of kept channel indices per block (8 tensors).

    Returns:
        A new ResNet18CIFAR whose blocks have only the kept inner channels.
    """
    pruned = ResNet18CIFAR(mid_channels=[len(keep) for keep in keep_lists])
    old_state = dense.state_dict()
    new_state = pruned.state_dict()

    # (a) Layers that were not touched: copy as they are. Only the 3 tensors
    #     per block handled in (b) have a different shape, and those are
    #     overwritten below.
    for key, value in old_state.items():
        if value.shape == new_state[key].shape:
            new_state[key] = value.clone()

    # (b) The pruned parts of every block
    for (name, _), keep in zip(blocks_of(dense), keep_lists):
        new_state[f"{name}.conv1.weight"] = old_state[f"{name}.conv1.weight"][keep]
        for bn_key in ("weight", "bias", "running_mean", "running_var"):
            new_state[f"{name}.bn1.{bn_key}"] = old_state[f"{name}.bn1.{bn_key}"][keep]
        new_state[f"{name}.conv2.weight"] = old_state[f"{name}.conv2.weight"][:, keep]

    pruned.load_state_dict(new_state)
    return pruned


def switch_off_channels(dense: ResNet18CIFAR, keep_lists: list) -> ResNet18CIFAR:
    """
    Return a copy of `dense` where the pruned channels are only SWITCHED OFF
    (zeroed), not removed. Used purely to verify the reconstruction.

    A channel is fully silent when its conv1 filter AND its bn1 scale and
    shift are all zero (zeroing only the filter would not be enough, because
    the batch-norm shift would still produce a non-zero output).
    """
    masked = copy.deepcopy(dense)
    with torch.no_grad():
        for (_, block), keep in zip(blocks_of(masked), keep_lists):
            removed = torch.ones(block.conv1.out_channels, dtype=torch.bool)
            removed[keep] = False
            block.conv1.weight[removed] = 0.0
            block.bn1.weight[removed] = 0.0
            block.bn1.bias[removed] = 0.0
    return masked


def verify_reconstruction(dense: ResNet18CIFAR, pruned: ResNet18CIFAR,
                          keep_lists: list) -> None:
    """
    Prove the thin model equals the original with those channels switched off.

    Both models are run on the same random images; their outputs should match
    up to floating-point rounding.
    """
    reference = switch_off_channels(dense, keep_lists).cpu().eval()
    pruned = pruned.cpu().eval()
    images = torch.randn(8, 3, 32, 32)
    with torch.inference_mode():
        expected = reference(images)
        actual = pruned(images)

    difference = float((expected - actual).abs().max())
    relative = difference / max(1.0, float(expected.abs().max()))
    print(f"Reconstruction check: largest output difference {difference:.2e} "
          f"(relative {relative:.2e})")
    if relative > 1e-4:
        raise RuntimeError("Reconstructed model does not match the masked original!")
    print("Reconstruction check: passed")


def main() -> None:
    set_seed()
    ensure_dirs()
    require_file(config.FP32_PATH, "Run train_baseline.py first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Structured pruning on: {device}")

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    # Step 1: load the trained original model (CPU) and measure it
    dense = load_model("fp32")
    original_accuracy = evaluate(dense.to(device), test_loader, device)
    dense = dense.cpu()
    print(f"Original FP32 accuracy: {original_accuracy:.2f}%")

    # Step 2: choose the channels to keep in each of the 8 blocks
    keep_lists = [select_channels_to_keep(block, config.STRUCT_PRUNE_RATIO)
                  for _, block in blocks_of(dense)]

    # Step 3: build the physically smaller network
    pruned = build_pruned_model(dense, keep_lists)
    for row in channel_rows(dense, pruned):
        print(f"  {row['block']:<10} inner channels "
              f"{row['channels_before']:>3} -> {row['channels_after']:>3}")

    # Step 4: verify it matches the original with those channels switched off
    verify_reconstruction(dense, pruned, keep_lists)

    total_before, _ = count_parameters(dense)
    total_after, _ = count_parameters(pruned)
    print(f"Parameters: {total_before:,} -> {total_after:,} "
          f"({100 * (1 - total_after / total_before):.1f}% fewer)")

    # Step 5: fine-tune the thin model to recover accuracy
    pruned = pruned.to(device)
    print(f"Accuracy right after pruning: "
          f"{evaluate(pruned, test_loader, device):.2f}%")
    fine_tune(pruned, train_loader, config.STRUCT_FT_EPOCHS,
              config.FINE_TUNE_LR, device, label="structured")

    # Step 6: evaluate and save. The checkpoint stores the channel counts too,
    # because load_model() needs them to rebuild the thin architecture.
    final_accuracy = evaluate(pruned, test_loader, device)
    print(f"\nFinal structured-pruned accuracy: {final_accuracy:.2f}% "
          f"(original {original_accuracy:.2f}%)")

    torch.save({
        "mid_channels": [len(keep) for keep in keep_lists],
        "state_dict": pruned.cpu().state_dict(),
    }, config.STRUCTURED_PATH)
    print(f"Saved structured-pruned model to {config.STRUCTURED_PATH}")


if __name__ == "__main__":
    main()
