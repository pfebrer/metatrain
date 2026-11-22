"""
Conversion of atomic-basis operator targets (Hamiltonians, overlaps, density
matrices, ...) from their block form (``TensorMap``) to dense matrices in the orbital
basis.

An operator ``O`` on an atom-centered basis ``|i, l, n, m; S>`` (atom ``i`` of the
cell translated by ``S``) is stored as per-atom blocks (samples ``system, atom``) for
the on-site part and per-pair blocks (samples ``system, first_atom, second_atom,
cell_shift_a, cell_shift_b, cell_shift_c``) for the rest. The block of the pair
``(i, j, S)`` holds ``<i, l_1, n_1, m_1; 0| O |j, l_2, n_2, m_2; S>``, with ``S`` the
cell shift of the neighbor list (``r_ij = r_j - r_i + S @ cell``).

:func:`blocks_to_matrix` assembles, for every structure, the real-space matrices
``O[S][I, J]`` (``I``, ``J`` the orbitals of the structure, ordered by atom and then as
given by the atomic basis, :func:`get_atomic_basis`) and :func:`bloch_sum` the
k-space matrices ``O(k) = sum_S exp(2 pi i k.S) O[S]`` (``k`` fractional).

Both the coupled (keys ``o3_lambda, o3_sigma``, properties ``l_1, l_2, n_1, n_2``)
and the uncoupled (keys ``o3_lambda_1, o3_lambda_2``, components ``o3_mu_1,
o3_mu_2``, properties ``n_1, n_2``) forms are accepted, either sparse (atom types in
the keys) or densified (:func:`densify_atomic_basis_target`). The coupled blocks are
uncoupled with the Clebsch-Gordan coefficients of featomic's ``cg_couple``, which
:func:`~metatrain.utils.data.spherical_helpers.couple_tensor_blocks` uses to couple
them.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


AtomicBasis = Dict[int, List[Tuple[int, int]]]
"""The shells ``(l, n)`` of every atomic type, in the order of the orbitals (each
shell spanning ``2 l + 1`` orbitals, ``m = -l, ..., l``)."""


def get_atomic_basis(layouts: Union[TensorMap, List[TensorMap]]) -> AtomicBasis:
    """
    The atomic basis of the (sparse) layouts of atomic-basis operator targets, per-atom
    (keys with ``atom_type``) or per-pair (keys with ``first_atom_type`` and
    ``second_atom_type``), coupled or uncoupled. The shells of every type are sorted by
    ``(l, n)``: the orbitals are in the order ``(l, n, m)``.

    :param layouts: one or several layouts, e.g. the ``layout`` of the ``TargetInfo``
        of the node and of the edge target. Any targets (such as tensors with
        samples) with the same metadata can be passed as well.
    :return: the shells ``(l, n)`` of every atomic type.
    """
    if not isinstance(layouts, list):
        layouts = [layouts]
    shells: Dict[int, set] = {}
    for layout in layouts:
        names = layout.keys.names
        if "atom_type" in names:
            type_names = ["atom_type", "atom_type"]
        elif "first_atom_type" in names and "second_atom_type" in names:
            type_names = ["first_atom_type", "second_atom_type"]
        else:
            raise ValueError(
                "the atomic basis is read from sparse layouts (keys with 'atom_type' "
                f"or 'first_atom_type' and 'second_atom_type'), got keys {names}"
            )
        for key, block in layout.items():
            z1, z2 = (int(key[name]) for name in type_names)
            for l1, l2, _, n1, n2 in _shell_pairs(key.values, names, block):
                shells.setdefault(z1, set()).update((l1, n) for n in n1.tolist())
                shells.setdefault(z2, set()).update((l2, n) for n in n2.tolist())
    return {z: sorted(s) for z, s in sorted(shells.items())}


def blocks_to_matrix(
    tensors: Union[TensorMap, List[TensorMap]],
    systems: List[System],
    basis: AtomicBasis,
    system_ids: Optional[torch.Tensor] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    The real-space matrices ``O[S]`` of every structure, from the blocks of an
    atomic-basis operator: usually a per-atom (on-site) and a per-pair (off-site)
    target, e.g. the predictions of a model.

    The pairs can be given in both directions or in one only: a pair ``(i, j, S)``
    whose opposite ``(j, i, -S)`` is in no block also gives ``O[-S][J, I] =
    O[S][I, J]^T`` (real symmetric operators). The blocks are written as given:
    on-site blocks are not symmetrized, and a pair given in both directions gives both
    from its own values. The contributions of the ``o3_lambda`` of a coupled operator
    add up; every pair should be given by at most one sample otherwise. Matrix
    elements of no pair are zero.

    Properties of shells that the atomic types of a sample do not have (the padding
    of a densified target, or of a prediction) are ignored, as are non-finite values
    (missing values of a padded target).

    The output is differentiable with respect to the values of the blocks.

    :param tensors: the blocks: per-atom and per-pair ``TensorMap``, coupled or
        uncoupled, sparse or densified.
    :param systems: the structures, whose atomic types define the orbitals.
    :param basis: the atomic basis, e.g. from :func:`get_atomic_basis`.
    :param system_ids: the value of the ``system`` sample dimension of every structure
        of ``systems``, by default ``0, ..., len(systems) - 1``.
    :return: per structure, ``(shifts, matrices)``: the ``(n_shifts, 3)`` cell shifts
        ``S`` and the ``(n_shifts, n_orbitals, n_orbitals)`` matrices ``O[S]``. The
        shifts are those of the samples and their opposites.
    """
    if not isinstance(tensors, list):
        tensors = [tensors]
    tensors = [t for t in tensors if len(t.keys) > 0]
    if len(tensors) == 0:
        raise ValueError("blocks_to_matrix needs at least one non-empty TensorMap")
    device = tensors[0].block(0).values.device
    dtype = tensors[0].block(0).values.dtype

    # orbitals: the first orbital of every shell of every type, and of every atom of
    # the batch in its structure
    shell_offsets: Dict[int, Dict[Tuple[int, int], int]] = {}
    sizes = torch.zeros(max(basis) + 1, dtype=torch.long)
    for z, shells in basis.items():
        offset, table = 0, {}
        for ell, n in shells:
            table[(ell, n)] = offset
            offset += 2 * ell + 1
        shell_offsets[z] = table
        sizes[z] = offset
    types = torch.cat([s.types for s in systems]).to(device=device, dtype=torch.long)
    missing = sorted(set(types.tolist()) - set(basis))
    if missing:
        raise ValueError(f"atomic types {missing} are not in the atomic basis")
    n_atoms = torch.tensor([len(s) for s in systems], device=device)
    atom_start = torch.cumsum(n_atoms, 0) - n_atoms
    frame_of_atom = torch.repeat_interleave(
        torch.arange(len(systems), device=device), n_atoms
    )
    atom_sizes = sizes.to(device)[types]
    cumulative = torch.cumsum(atom_sizes, 0) - atom_sizes
    orbital_start = cumulative - cumulative[atom_start][frame_of_atom]
    n_orbitals = torch.zeros(len(systems), dtype=torch.long, device=device).index_add(
        0, frame_of_atom, atom_sizes
    )

    if system_ids is None:
        system_ids = torch.arange(len(systems))
    system_ids = system_ids.to(device=device, dtype=torch.long)
    order = torch.argsort(system_ids)
    samples = [
        _pair_samples(block, system_ids, order, atom_start)
        for tensor in tensors
        for block in tensor.blocks()
    ]
    frames = torch.cat([s[0] for s in samples])
    shifts = torch.cat([s[3] for s in samples])

    # integer codes of (structure, shift) (sorted structure-major) and of pairs
    base = int(shifts.abs().max()) if len(shifts) > 0 else 0
    width = 2 * base + 1
    n_total = len(types)

    def shift_code(s: torch.Tensor) -> torch.Tensor:
        s = s + base
        return (s[:, 0] * width + s[:, 1]) * width + s[:, 2]

    def pair_code(i: torch.Tensor, j: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (i * n_total + j) * width**3 + shift_code(s)

    codes = torch.unique(
        torch.cat(
            [
                frames * width**3 + shift_code(shifts),
                frames * width**3 + shift_code(-shifts),
            ]
        )
    )
    code_frames = codes // width**3
    rest = codes % width**3
    code_shifts = (
        torch.stack([rest // width**2, (rest // width) % width, rest % width], 1) - base
    )
    n_shifts = torch.zeros(len(systems), dtype=torch.long, device=device).index_add(
        0, code_frames, torch.ones_like(code_frames)
    )
    first_code = torch.cumsum(n_shifts, 0) - n_shifts
    matrix_sizes = n_shifts * n_orbitals**2
    matrix_start = torch.cumsum(matrix_sizes, 0) - matrix_sizes
    pair_codes = [pair_code(s[1], s[2], s[3]) for s in samples]
    pairs = torch.unique(torch.cat(pair_codes))

    # the blocks of every pair of atomic types (of all tensors: on-site samples are
    # the pairs (i, i, 0)), and the samples of the pair types in them
    groups: Dict[Tuple[int, int], List[Tuple]] = {}
    s_index = 0
    for tensor in tensors:
        names = tensor.keys.names
        for key, block in tensor.items():
            frame, first, second, shift = samples[s_index]
            codes_b = pair_codes[s_index]
            s_index += 1
            if len(frame) == 0:
                continue
            for z1, z2, rows in _type_groups(key, names, types[first], types[second]):
                selected = codes_b if rows is None else codes_b[rows]
                groups.setdefault((z1, z2), []).append(
                    (key, names, block, rows, selected)
                )

    # every pair type: the orbital blocks of its pairs, summed over its blocks (one
    # matmul each), and their positions in the matrices (and the transposed ones for
    # the pairs whose opposite is absent). No element is written twice: one scatter.
    indices: List[torch.Tensor] = []
    values_out: List[torch.Tensor] = []
    for (z1, z2), entries in groups.items():
        n1, n2 = int(sizes[z1]), int(sizes[z2])
        first_codes = entries[0][4]
        same = all(
            len(e[4]) == len(first_codes) and bool(torch.equal(e[4], first_codes))
            for e in entries[1:]
        )
        union = (
            first_codes if same else torch.unique(torch.cat([e[4] for e in entries]))
        )
        blocks = torch.zeros((len(union), n1 * n2), dtype=dtype, device=device)
        for key, names, block, rows, selected in entries:
            columns, uncoupling = _uncoupling_map(
                key, names, block, shell_offsets, z1, z2, device, dtype
            )
            if len(columns) == 0:
                continue
            values = block.values.reshape(len(block.samples), -1)
            v = values[:, columns] if rows is None else values[rows][:, columns]
            v = torch.where(torch.isfinite(v), v, 0.0) @ uncoupling
            if same:
                blocks = blocks + v
            else:
                blocks = blocks.index_add(0, torch.searchsorted(union, selected), v)

        # the pairs (i, j, S) of the union, decoded
        shift = union % width**3
        ij = union // width**3
        i, j = ij // n_total, ij % n_total
        shift = (
            torch.stack([shift // width**2, (shift // width) % width, shift % width], 1)
            - base
        )
        frame = frame_of_atom[i]
        n_f = n_orbitals[frame]
        here = torch.searchsorted(codes, frame * width**3 + shift_code(shift))
        there = torch.searchsorted(codes, frame * width**3 + shift_code(-shift))
        start = (matrix_start[frame] + (here - first_code[frame]) * n_f**2)[
            :, None, None
        ]
        start_t = (matrix_start[frame] + (there - first_code[frame]) * n_f**2)[
            :, None, None
        ]
        i0 = orbital_start[i][:, None, None]
        j0 = orbital_start[j][:, None, None]
        n_f = n_f[:, None, None]
        a = torch.arange(n1, device=device)[:, None]
        b = torch.arange(n2, device=device)[None, :]
        indices.append((start + (i0 + a) * n_f + j0 + b).reshape(-1))
        values_out.append(blocks.reshape(-1))
        mirror = ~torch.isin(pair_code(j, i, -shift), pairs)
        indices.append((start_t + (j0 + b) * n_f + i0 + a)[mirror].reshape(-1))
        values_out.append(blocks.reshape(-1, n1, n2)[mirror].reshape(-1))

    # in place on a tensor without history: no copy, and the backward is a gather
    out = torch.zeros(int(matrix_sizes.sum()), dtype=dtype, device=device)
    out.index_put_((torch.cat(indices),), torch.cat(values_out))

    result = []
    for f in range(len(systems)):
        size, r_f = int(n_orbitals[f]), int(n_shifts[f])
        o_f, c_f = int(matrix_start[f]), int(first_code[f])
        matrices = out[o_f : o_f + r_f * size * size].reshape(r_f, size, size)
        result.append((code_shifts[c_f : c_f + r_f], matrices))
    return result


def bloch_sum(
    shifts: torch.Tensor, matrices: torch.Tensor, kpoints: torch.Tensor
) -> torch.Tensor:
    """
    The k-space matrices ``O(k) = sum_S exp(2 pi i k.S) O[S]`` of a structure.

    :param shifts: ``(n_shifts, 3)`` cell shifts ``S``, see :func:`blocks_to_matrix`.
    :param matrices: ``(n_shifts, n, n)`` real matrices ``O[S]``.
    :param kpoints: ``(n_k, 3)`` k-points, in fractional coordinates (of the
        reciprocal cell).
    :return: ``(n_k, n, n)`` complex matrices.
    """
    phase = 2 * torch.pi * (kpoints.to(torch.float64) @ shifts.to(torch.float64).T)
    flat = matrices.reshape(len(shifts), -1)
    shape = (len(kpoints), *matrices.shape[1:])
    real = (torch.cos(phase).to(flat.dtype) @ flat).reshape(shape)
    imag = (torch.sin(phase).to(flat.dtype) @ flat).reshape(shape)
    return torch.complex(real, imag)


def _type_groups(
    key: Labels, key_names: List[str], z1: torch.Tensor, z2: torch.Tensor
) -> List[Tuple[int, int, Optional[torch.Tensor]]]:
    """
    The samples of a block by pair of atomic types: from the keys of a sparse
    target, from the atoms of a densified one.

    :param key: the block's key.
    :param key_names: the names of the keys.
    :param z1: the type of the first atom of every sample.
    :param z2: the type of the second atom of every sample.
    :return: per pair of types, ``(z1, z2, rows)``, ``rows`` ``None`` for all samples.
    """
    if "atom_type" in key_names:
        return [(int(key["atom_type"]), int(key["atom_type"]), None)]
    if "first_atom_type" in key_names:
        return [(int(key["first_atom_type"]), int(key["second_atom_type"]), None)]
    pair_types = z1 * 1024 + z2
    unique = torch.unique(pair_types)
    if len(unique) == 1:
        p = int(unique[0])
        return [(p // 1024, p % 1024, None)]
    return [
        (p // 1024, p % 1024, torch.nonzero(pair_types == p)[:, 0])
        for p in unique.tolist()
    ]


_UNCOUPLING_MAPS: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


def _uncoupling_map(
    key: Labels,
    key_names: List[str],
    block: TensorBlock,
    shell_offsets: Dict[int, Dict[Tuple[int, int], int]],
    z1: int,
    z2: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The linear map from the values of a sample of a block (flattened) to its
    ``(n_1 n_2)`` orbital block between atoms of types ``z1`` and ``z2``: ``U[m_1,
    m_2] = sum_mu C[m_1, m_2, mu] y[mu]`` for coupled blocks, a placement for
    uncoupled ones. Only the values of shells both types have are used. Cached per
    metadata.

    :param key: the block's key.
    :param key_names: the names of the keys.
    :param block: the block.
    :param shell_offsets: the first orbital of every shell of every type.
    :param z1: the type of the first atom.
    :param z2: the type of the second atom.
    :param device: the device of the map.
    :param dtype: the dtype of the map.
    :return: ``(columns, map)``: the used values (indices in the flattened values) and
        the ``(n_columns, n_1 n_2)`` map.
    """
    properties = block.properties
    signature = (
        tuple(key_names),
        tuple(key.values.tolist()),
        tuple(properties.names),
        tuple(map(tuple, properties.values.tolist())),
        z1,
        z2,
        tuple(sorted(shell_offsets[z1].items())),
        tuple(sorted(shell_offsets[z2].items())),
        str(device),
        dtype,
    )
    if signature in _UNCOUPLING_MAPS:
        return _UNCOUPLING_MAPS[signature]
    t1, t2 = shell_offsets[z1], shell_offsets[z2]
    n1 = sum(2 * ell + 1 for ell, _ in t1)
    n2 = sum(2 * ell + 1 for ell, _ in t2)
    n_properties = len(properties)
    columns, targets, values = [], [], []
    for l1, l2, props, radial_1, radial_2 in _shell_pairs(key.values, key_names, block):
        if "o3_lambda" in key_names:
            cg = _cg_uncoupling(l1, l2, int(key["o3_lambda"]))
        else:
            cg = torch.eye((2 * l1 + 1) * (2 * l2 + 1), dtype=torch.float64)
            cg = cg.reshape(2 * l1 + 1, 2 * l2 + 1, -1)
        m1, m2, mu = torch.meshgrid(*[torch.arange(d) for d in cg.shape], indexing="ij")
        for p, r1, r2 in zip(
            props.tolist(), radial_1.tolist(), radial_2.tolist(), strict=True
        ):
            o1, o2 = t1.get((l1, r1)), t2.get((l2, r2))
            if o1 is None or o2 is None:
                continue
            columns.append((mu * n_properties + p).reshape(-1))
            targets.append(((o1 + m1) * n2 + o2 + m2).reshape(-1))
            values.append(cg.reshape(-1))
    if len(columns) == 0:
        result = (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros((0, n1 * n2), dtype=dtype, device=device),
        )
    else:
        used, inverse = torch.unique(torch.cat(columns), return_inverse=True)
        matrix = torch.zeros((len(used), n1 * n2), dtype=torch.float64)
        matrix.index_put_(
            (inverse, torch.cat(targets)), torch.cat(values), accumulate=True
        )
        result = (used.to(device), matrix.to(device=device, dtype=dtype))
    _UNCOUPLING_MAPS[signature] = result
    return result


def _shell_pairs(
    key: torch.Tensor, key_names: List[str], block: TensorBlock
) -> List[Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    The properties of a block, grouped by ``(l_1, l_2)``.

    :param key: the values of the block's key.
    :param key_names: the names of the keys.
    :param block: the block.
    :return: per group, ``(l_1, l_2, properties, n_1, n_2)``: the indices of the
        properties in the block and their ``n_1`` and ``n_2``.
    """
    p = block.properties
    if "o3_lambda" in key_names:
        if not {"l_1", "l_2", "n_1", "n_2"} <= set(p.names):
            raise ValueError(
                "coupled atomic-basis blocks need properties 'l_1', 'l_2', 'n_1', "
                f"'n_2', got {p.names}"
            )
        ls = torch.stack([p.column("l_1"), p.column("l_2")], 1)
        groups = []
        for l1, l2 in torch.unique(ls, dim=0).tolist():
            props = torch.nonzero((ls[:, 0] == l1) & (ls[:, 1] == l2))[:, 0]
            groups.append(
                (l1, l2, props, p.column("n_1")[props], p.column("n_2")[props])
            )
        return groups
    if "o3_lambda_1" in key_names and "o3_lambda_2" in key_names:
        if not {"n_1", "n_2"} <= set(p.names):
            raise ValueError(
                f"uncoupled atomic-basis blocks need properties 'n_1', 'n_2', "
                f"got {p.names}"
            )
        l1 = int(key[key_names.index("o3_lambda_1")])
        l2 = int(key[key_names.index("o3_lambda_2")])
        props = torch.arange(len(p))
        return [(l1, l2, props, p.column("n_1"), p.column("n_2"))]
    raise ValueError(
        "atomic-basis operator blocks need keys 'o3_lambda' (coupled) or "
        f"'o3_lambda_1' and 'o3_lambda_2' (uncoupled), got {key_names}"
    )


def _pair_samples(
    block: TensorBlock,
    system_ids: torch.Tensor,
    order: torch.Tensor,
    atom_start: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The structure (index in the systems), the first and the second atom (indices in
    the batch) and the cell shift of every sample of a per-atom or per-pair block.

    :param block: the block.
    :param system_ids: the ``system`` label of every structure.
    :param order: ``argsort(system_ids)``.
    :param atom_start: the index of the first atom of every structure in the batch.
    :return: ``(frames, first, second, shifts)``.
    """
    s = block.samples
    labels = s.column("system").to(torch.long)
    position = torch.searchsorted(system_ids[order], labels).clamp(max=len(order) - 1)
    frames = order[position]
    if not bool((system_ids[frames] == labels).all()):
        raise ValueError(
            "the 'system' samples of the blocks are not all in `system_ids`"
        )
    if "atom" in s.names:
        first = s.column("atom").to(torch.long)
        second = first
        shifts = torch.zeros((len(s), 3), dtype=torch.long, device=labels.device)
    elif "first_atom" in s.names:
        first = s.column("first_atom").to(torch.long)
        second = s.column("second_atom").to(torch.long)
        shifts = torch.stack(
            [s.column(f"cell_shift_{c}").to(torch.long) for c in "abc"], 1
        )
    else:
        raise ValueError(
            "atomic-basis operator blocks need per-atom ('atom') or per-pair "
            f"('first_atom', 'second_atom', 'cell_shift_*') samples, got {s.names}"
        )
    return frames, atom_start[frames] + first, atom_start[frames] + second, shifts


@lru_cache(maxsize=None)
def _cg_uncoupling(l1: int, l2: int, o3_lambda: int) -> torch.Tensor:
    """
    The real Clebsch-Gordan coefficients ``C[m_1, m_2, mu]`` of featomic's
    ``cg_couple`` (as used by
    :func:`~metatrain.utils.data.spherical_helpers.couple_tensor_blocks`):
    ``y[mu] = sum_{m_1, m_2} C[m_1, m_2, mu] U[m_1, m_2]``. ``C`` is orthogonal,
    so the uncoupled block is ``U[m_1, m_2] = sum_lambda sum_mu C[m_1, m_2, mu]
    y_lambda[mu]``.

    :param l1: the angular momentum of the first orbital.
    :param l2: the angular momentum of the second orbital.
    :param o3_lambda: the coupled angular momentum.
    :return: ``(2 l1 + 1, 2 l2 + 1, 2 o3_lambda + 1)`` float64 coefficients.
    """
    from featomic.torch.clebsch_gordan._coefficients import (
        calculate_cg_coefficients,
        cg_couple,
    )

    cg = calculate_cg_coefficients(
        l1 + l2,
        cg_backend="python-sparse",
        arrays_backend="torch",
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    d1, d2 = 2 * l1 + 1, 2 * l2 + 1
    eye = torch.eye(d1 * d2, dtype=torch.float64).reshape(d1 * d2, d1, d2, 1)
    coupled = cg_couple(eye, [o3_lambda], cg, "python-sparse")[0]
    return coupled[:, :, 0].reshape(d1, d2, 2 * o3_lambda + 1)
