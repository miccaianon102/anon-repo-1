"""
Space-filling curve utilities for point serialization:
  - Hilbert curve (get_hilbert_sort_order)
  - Morton / Z-order curve (get_morton_sort_order)
  - Transposed variants used by PTv3 ablation.
"""

import torch
import torch.nn.functional as F



# Hilbert Curve Helpers

def _right_shift(binary: torch.Tensor, k: int = 1, axis: int = -1) -> torch.Tensor:
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return F.pad(binary[tuple(slicing)], (k, 0), value=0)


def _gray2binary(gray: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shift = 2 ** (torch.tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode_hilbert(locs: torch.Tensor, num_dims: int = 3, num_bits: int = 16) -> torch.Tensor:
    """Encode integer coordinates into Hilbert indices (Skilling's method)."""
    locs = locs.long().clamp(0, 2 ** num_bits - 1)
    orig_shape = locs.shape
    device = locs.device

    bitpack_mask = (1 << torch.arange(0, 8, device=device))
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if not locs.is_contiguous():
        locs = locs.contiguous()

    locs_uint8 = locs.view(torch.uint8).reshape((*orig_shape[:-1], num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0).byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[..., dim, bit]
            gray[..., 0, bit + 1:] = torch.logical_xor(gray[..., 0, bit + 1:], mask[..., None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[..., None]).repeat(*([1] * (mask.dim())), gray.shape[-1] - bit - 1),
                torch.logical_xor(gray[..., 0, bit + 1:], gray[..., dim, bit + 1:]),
            )
            gray[..., dim, bit + 1:] = torch.logical_xor(gray[..., dim, bit + 1:], to_flip)
            gray[..., 0, bit + 1:] = torch.logical_xor(gray[..., 0, bit + 1:], to_flip)

    gray = gray.swapaxes(-2, -1).reshape((*orig_shape[:-1], num_bits * num_dims))
    hh_bin = _gray2binary(gray)

    extra = 64 - num_bits * num_dims
    padded = F.pad(hh_bin, (extra, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((*orig_shape[:-1], 8, 8)) * bitpack_mask)
        .sum(-1).type(torch.uint8)
    )
    return hh_uint8.view(torch.int64).squeeze(-1)


def get_hilbert_sort_order(coords: torch.Tensor, num_bits: int = 16):
    """Sort 3-D point cloud by Hilbert index."""
    coords_t = coords.permute(0, 2, 1)          # [B, N, 3]
    min_v = coords_t.min(dim=1, keepdim=True)[0]
    max_v = coords_t.max(dim=1, keepdim=True)[0]
    norm  = (coords_t - min_v) / (max_v - min_v + 1e-6)
    int_coords = (norm * (2 ** num_bits - 1)).long().contiguous()  # [B, N, 3]

    code = encode_hilbert(int_coords, num_dims=3, num_bits=num_bits)
    sort_idx   = torch.argsort(code, dim=1)
    unsort_idx = torch.argsort(sort_idx, dim=1)
    return sort_idx, unsort_idx



# Morton / Z-order Curve


def _interleave_bits(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                     num_bits: int) -> torch.Tensor:
    code = torch.zeros_like(x, dtype=torch.long)
    for i in range(num_bits):
        code |= ((x >> i) & 1) << (3 * i)
        code |= ((y >> i) & 1) << (3 * i + 1)
        code |= ((z >> i) & 1) << (3 * i + 2)
    return code


def get_morton_sort_order(coords: torch.Tensor, num_bits: int = 16):
    """Sort points by Morton (Z-order) code."""
    assert 3 * num_bits <= 63, "num_bits too large for int64"
    if coords.shape[1] == 3 and coords.dim() == 3:
        coords_t = coords.permute(0, 2, 1)
    else:
        coords_t = coords  # already [B, N, 3]

    min_v = coords_t.amin(dim=1, keepdim=True)
    max_v = coords_t.amax(dim=1, keepdim=True)
    norm  = (coords_t - min_v) / (max_v - min_v).clamp(min=1e-6)
    scale = (1 << num_bits) - 1
    ic    = (norm * scale).round().clamp(0, scale).long()

    code = _interleave_bits(ic[..., 0], ic[..., 1], ic[..., 2], num_bits)
    sort_idx   = torch.argsort(code, dim=-1)
    unsort_idx = torch.argsort(sort_idx, dim=-1)
    return sort_idx, unsort_idx


def get_trans_hilbert_sort_order(coords: torch.Tensor, num_bits: int = 16):
    """Hilbert on (y, z, x) permutation."""
    coords_t = coords[:, [1, 2, 0], :] if coords.shape[1] == 3 else coords[:, :, [1, 2, 0]]
    return get_hilbert_sort_order(coords_t, num_bits)


def get_trans_zorder_sort_order(coords: torch.Tensor, num_bits: int = 16):
    """Morton on (y, z, x) permutation."""
    coords_t = coords[:, [1, 2, 0], :] if coords.shape[1] == 3 else coords[:, :, [1, 2, 0]]
    return get_morton_sort_order(coords_t, num_bits)
