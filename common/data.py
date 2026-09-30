"""
This is a shared data-generation utility for the grid-cell task ladder (Task 0, 1, 1.5, 2/3).

Task 0 needs:
  - N_COLORS x N_SHAPES "items", each represented as a one-hot vector
    (color one-hot concatenated with shape one-hot).
  - Each item is assigned to a location in a continuous 2D square [0, 1] x [0, 1].
  - The assignment is ARBITRARY (meaning that it is shuffled), not aligned to the one-hot axes,
    so the network cannot read off (x, y) directly from which color/shape
    units are on -- it has to learn the layout.

Everything here is  framework-light (numpy in, numpy/torch out) so the same item 
and location table can be reused by Task 1 (comparison),
Task 1.5 (vector between items), and Task 2/3 (k-step paths),
which all start from the same underlying item space! 
"""
import numpy as np
import torch

"""
    The below builds the item space: n_colors * n_shapes items, each a (color_idx, shape_idx)
    pair, assigned to a location on an arbitrary n_colors x n_shapes grid of points covering [0, 1] x [0, 1].

    It returns a dict with:
      n_items      : int, total number of items (n_colors * n_shapes)
      color_idx    : (n_items,) int array, which color each item has
      shape_idx    : (n_items,) int array, which shape each item has
      locations    : (n_items, 2) float array, each item's (x, y) in [0, 1]^2
      grid_shape   : (n_colors, n_shapes), for reshaping into images later

    The mapping from (color_idx, shape_idx) to a grid cell is shuffled using
    `seed`, so color and shape do NOT line up with x and y. Two different
    seeds give two different arbitrary layouts (useful for a shuffled-structure
    control, or for training multiple networks on different layouts).
    """
def make_item_space(n_colors: int = 16, n_shapes: int = 16, seed: int = 0):
  rng = np.random.default_rng(seed)
    
    n_items = n_colors * n_shapes
    color_idx = np.repeat(np.arange(n_colors), n_shapes) # np.arrange creates array of color IDs, np.repeat duplicates each
    shape_idx = np.tile(np.arange(n_shapes), n_colors) # np.tile duplicates the entire array sequence 

    # a regular grid of candidate locations covering the unit square, using cell centers so no location sits exactly on the boundary.
    xs = (np.arange(n_colors) + 0.5) / n_colors
    ys = (np.arange(n_shapes) + 0.5) / n_shapes
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    candidate_locations = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # this shuffles which item gets which candidate location. this is the code that breaks the color=x, shape=y alignment.
    perm = rng.permutation(n_items)
    locations = candidate_locations[perm]

    return {
        "n_items": n_items,
        "n_colors": n_colors,
        "n_shapes": n_shapes,
        "color_idx": color_idx,
        "shape_idx": shape_idx,
        "locations": locations.astype(np.float32),
        "grid_shape": (n_colors, n_shapes),
    }

# CHECKED UP TO HERE THAT CODE RUNS ON COLAB

    """
    The below builds the (n_items, n_colors + n_shapes) one-hot input matrix: each row is
    a color one-hot concatenated with a shape one-hot, following Fascianelli-style two-one-hot inputs.
    """
def one_hot_inputs(item_space: dict) -> torch.Tensor: # just indicating that output will be a PyTorch tensor object 
    n_items = item_space["n_items"]
    n_colors = item_space["n_colors"]
    n_shapes = item_space["n_shapes"]

    color_oh = np.zeros((n_items, n_colors), dtype=np.float32)
    color_oh[np.arange(n_items), item_space["color_idx"]] = 1.0

    shape_oh = np.zeros((n_items, n_shapes), dtype=np.float32)
    shape_oh[np.arange(n_items), item_space["shape_idx"]] = 1.0

    inputs = np.concatenate([color_oh, shape_oh], axis=1)
    return torch.from_numpy(inputs)
