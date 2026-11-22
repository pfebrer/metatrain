import metatensor.torch as mts
import numpy as np
import pytest
import torch
from featomic.torch.clebsch_gordan._coefficients import calculate_cg_coefficients
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data import DiskDataset
from metatrain.utils.data.atom_pair_helpers import get_single_direction_edges
from metatrain.utils.data.atomic_basis_helpers import densify_atomic_basis_target
from metatrain.utils.data.matrix_helpers import (
    bloch_sum,
    blocks_to_matrix,
    get_atomic_basis,
)
from metatrain.utils.data.spherical_helpers import (
    couple_tensor_blocks,
    uncouple_tensor_blocks,
)

from ...conftest import RESOURCES_PATH


BASIS = {1: [(0, 0), (1, 0)], 8: [(0, 0), (0, 1), (1, 0), (2, 0)]}
PAIR_NAMES = [
    "system",
    "first_atom",
    "second_atom",
    "cell_shift_a",
    "cell_shift_b",
    "cell_shift_c",
]
# one direction of the pairs of every structure, with self-images (i, i, S != 0)
PAIRS = [
    [
        (0, 1, (0, 0, 0)),
        (0, 2, (0, 0, 0)),
        (1, 2, (0, 0, 0)),
        (0, 0, (1, 0, 0)),
        (1, 2, (0, 1, 0)),
        (0, 1, (-1, 0, 1)),
        (2, 2, (0, 0, 1)),
    ],
    [(0, 1, (0, 0, 0))],
]


def _size(z):
    return sum(2 * ell + 1 for ell, _ in BASIS[z])


def _systems():
    torch.manual_seed(0)
    return [
        System(
            types=torch.tensor([8, 1, 1]),
            positions=torch.rand(3, 3, dtype=torch.float64) * 3.0,
            cell=3.0 * torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            types=torch.tensor([1, 8]),
            positions=torch.rand(2, 3, dtype=torch.float64),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        ),
    ]


def _expected(systems):
    """random real-space matrices {S: O[S]} with O[-S] = O[S]^T, and the blocks of
    the on-site part and of one direction of the pairs."""
    generator = torch.Generator().manual_seed(1)
    out, onsite, pairs = [], [], []
    for system, frame_pairs in zip(systems, PAIRS, strict=True):
        types = system.types.tolist()
        sizes = [_size(z) for z in types]
        starts = np.cumsum([0] + sizes)
        n = starts[-1]
        matrices = {}

        def at(shift, n=n, matrices=matrices):
            return matrices.setdefault(shift, torch.zeros(n, n, dtype=torch.float64))

        frame_onsite = []
        for i, size in enumerate(sizes):
            a = torch.randn(size, size, generator=generator, dtype=torch.float64)
            a = a + a.T
            at((0, 0, 0))[starts[i] : starts[i + 1], starts[i] : starts[i + 1]] = a
            frame_onsite.append(a)
        frame_blocks = []
        for i, j, shift in frame_pairs:
            b = torch.randn(
                sizes[i], sizes[j], generator=generator, dtype=torch.float64
            )
            if i == j and shift == (0, 0, 0):
                raise ValueError("on-site pair")
            opposite = tuple(-s for s in shift)
            at(shift)[starts[i] : starts[i + 1], starts[j] : starts[j + 1]] = b
            at(opposite)[starts[j] : starts[j + 1], starts[i] : starts[i + 1]] = b.T
            frame_blocks.append(b)
        out.append(matrices)
        onsite.append(frame_onsite)
        pairs.append(frame_blocks)
    return out, onsite, pairs


def _shells(z):
    """(l, n, first orbital) of the shells of type z"""
    out, offset = [], 0
    for ell, n in BASIS[z]:
        out.append((ell, n, offset))
        offset += 2 * ell + 1
    return out


def _uncoupled(systems, entries, per_atom):
    """uncoupled TensorMap from entries [(system, i, j, shift, block)]"""
    blocks = {}
    for f, i, j, shift, block in entries:
        z1, z2 = int(systems[f].types[i]), int(systems[f].types[j])
        sample = [f, i] if per_atom else [f, i, j, *shift]
        for l1, n1, o1 in _shells(z1):
            for l2, n2, o2 in _shells(z2):
                key = (l1, l2, 1, 1, z1) if per_atom else (l1, l2, 1, 1, z1, z2)
                entry = blocks.setdefault(key, {})
                sub = block[o1 : o1 + 2 * l1 + 1, o2 : o2 + 2 * l2 + 1]
                entry.setdefault(tuple(sample), {})[(n1, n2)] = sub
    names = ["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"]
    names += ["atom_type"] if per_atom else ["first_atom_type", "second_atom_type"]
    keys, tensor_blocks = [], []
    for key, samples in sorted(blocks.items()):
        l1, l2 = key[0], key[1]
        properties = sorted({p for values in samples.values() for p in values})
        values = torch.zeros(len(samples), 2 * l1 + 1, 2 * l2 + 1, len(properties))
        values = values.to(torch.float64)
        for s, sample_values in enumerate(samples.values()):
            for p, prop in enumerate(properties):
                values[s, :, :, p] = sample_values[prop]
        tensor_blocks.append(
            TensorBlock(
                values=values,
                samples=Labels(
                    ["system", "atom"] if per_atom else PAIR_NAMES,
                    torch.tensor(list(samples), dtype=torch.int32),
                ),
                components=[
                    Labels(["o3_mu_1"], torch.arange(-l1, l1 + 1).reshape(-1, 1)),
                    Labels(["o3_mu_2"], torch.arange(-l2, l2 + 1).reshape(-1, 1)),
                ],
                properties=Labels(["n_1", "n_2"], torch.tensor(properties)),
            )
        )
        keys.append(key)
    return TensorMap(
        Labels(names, torch.tensor(keys, dtype=torch.int32)), tensor_blocks
    )


def _targets(systems, onsite, pairs, both_directions, form):
    node_entries = [
        (f, i, i, (0, 0, 0), a)
        for f, frame in enumerate(onsite)
        for i, a in enumerate(frame)
    ]
    edge_entries = []
    for f, (frame_pairs, frame_blocks) in enumerate(zip(PAIRS, pairs, strict=True)):
        for (i, j, shift), b in zip(frame_pairs, frame_blocks, strict=True):
            edge_entries.append((f, i, j, shift, b))
            if both_directions:
                edge_entries.append((f, j, i, tuple(-s for s in shift), b.T))
    tensors = [
        _uncoupled(systems, node_entries, per_atom=True),
        _uncoupled(systems, edge_entries, per_atom=False),
    ]
    if form == "uncoupled":
        return tensors
    cg = calculate_cg_coefficients(
        4,
        cg_backend="python-sparse",
        arrays_backend="torch",
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    tensors = [couple_tensor_blocks(t, cg) for t in tensors]
    if form == "coupled":
        return tensors
    assert form == "dense"
    return [densify_atomic_basis_target(t, t) for t in tensors]


def _check(out, expected):
    assert len(out) == len(expected)
    for (shifts, matrices), frame in zip(out, expected, strict=True):
        got = {tuple(s): m for s, m in zip(shifts.tolist(), matrices, strict=True)}
        assert set(got) == set(frame)
        for shift, m in frame.items():
            torch.testing.assert_close(got[shift], m, rtol=0, atol=1e-12)


@pytest.mark.parametrize("form", ["uncoupled", "coupled", "dense"])
@pytest.mark.parametrize("both_directions", [False, True])
def test_blocks_to_matrix(form, both_directions):
    systems = _systems()
    expected, onsite, pairs = _expected(systems)
    tensors = _targets(systems, onsite, pairs, both_directions, form)
    if form != "dense":
        assert get_atomic_basis(tensors) == BASIS
    _check(blocks_to_matrix(tensors, systems, BASIS), expected)


def test_system_ids():
    systems = _systems()
    expected, onsite, pairs = _expected(systems)
    tensors = _targets(systems, onsite, pairs, False, "coupled")
    labels = {0: 7, 1: 3}
    relabelled = []
    for tensor in tensors:
        blocks = []
        for block in tensor.blocks():
            values = block.samples.values.clone()
            values[:, 0] = torch.tensor([labels[int(v)] for v in values[:, 0]])
            blocks.append(
                TensorBlock(
                    values=block.values,
                    samples=Labels(block.samples.names, values),
                    components=block.components,
                    properties=block.properties,
                )
            )
        relabelled.append(TensorMap(tensor.keys, blocks))
    out = blocks_to_matrix(relabelled, systems, BASIS, system_ids=torch.tensor([7, 3]))
    _check(out, expected)
    with pytest.raises(ValueError, match="system_ids"):
        blocks_to_matrix(relabelled, systems, BASIS)


def test_gradients():
    systems = _systems()
    _, onsite, pairs = _expected(systems)
    tensors = _targets(systems, onsite, pairs, False, "coupled")
    leaves = [
        b.values.clone().requires_grad_(True) for t in tensors for b in t.blocks()
    ]
    blocks = iter(leaves)
    tensors = [
        TensorMap(
            t.keys,
            [
                TensorBlock(
                    values=next(blocks),
                    samples=b.samples,
                    components=b.components,
                    properties=b.properties,
                )
                for b in t.blocks()
            ],
        )
        for t in tensors
    ]
    out = blocks_to_matrix(tensors, systems, BASIS)
    generator = torch.Generator().manual_seed(2)
    weights = [torch.randn(m.shape, generator=generator, dtype=m.dtype) for _, m in out]
    sum((w * m).sum() for w, (_, m) in zip(weights, out, strict=True)).backward()
    # every value of every block enters the matrices
    assert all(bool((leaf.grad != 0).all()) for leaf in leaves)


def test_bloch_sum():
    systems = _systems()
    expected, onsite, pairs = _expected(systems)
    tensors = _targets(systems, onsite, pairs, False, "coupled")
    shifts, matrices = blocks_to_matrix(tensors, systems, BASIS)[0]
    k = torch.tensor([[0.0, 0.0, 0.0], [0.25, -0.5, 0.125], [0.3, 0.1, 0.7]])
    h = bloch_sum(shifts, matrices, k)
    assert h.shape == (3, *matrices.shape[1:]) and h.dtype == torch.complex128
    torch.testing.assert_close(h, h.conj().transpose(-1, -2))
    for kk, hk in zip(k.tolist(), h, strict=True):
        reference = sum(
            np.exp(2j * np.pi * np.dot(kk, shift)) * m.numpy()
            for shift, m in expected[0].items()
        )
        np.testing.assert_allclose(hk.numpy(), reference, atol=1e-12)
    torch.testing.assert_close(h[0].real, matrices.sum(0))


def test_scfbench_hamiltonian():
    """the edge Hamiltonian of the scfbench resource: symmetric, and the same from
    one direction of the pairs, from both, and uncoupled."""
    dataset = DiskDataset(RESOURCES_PATH / "scfbench_2_bidirectional_edges.zip")
    samples = [dataset[0], dataset[1]]
    name = "mtt::matrix_edges::ham_coup"
    tensor = mts.join(
        [s[name] for s in samples], axis="samples", different_keys="union"
    )
    keep = [i for i, b in enumerate(tensor.blocks()) if len(b.samples) > 0]
    tensor = TensorMap(
        Labels(tensor.keys.names, tensor.keys.values[keep]),
        [tensor.block(i).copy(deep=False) for i in keep],
    )
    systems = [s["system"] for s in samples]
    basis = get_atomic_basis(tensor)
    out = blocks_to_matrix(tensor, systems, basis)
    for (shifts, matrices), system in zip(out, systems, strict=True):
        n = sum(sum(2 * ell + 1 for ell, _ in basis[int(z)]) for z in system.types)
        assert shifts.tolist() == [[0, 0, 0]] and matrices.shape == (1, n, n)
        torch.testing.assert_close(matrices[0], matrices[0].T, rtol=0, atol=1e-6)

    single = blocks_to_matrix(get_single_direction_edges(tensor), systems, basis)
    cg = calculate_cg_coefficients(
        4,
        cg_backend="python-dense",
        arrays_backend="torch",
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    uncoupled = blocks_to_matrix(uncouple_tensor_blocks(tensor, cg), systems, basis)
    for (_, m), (_, m_single), (_, m_uncoupled) in zip(
        out, single, uncoupled, strict=True
    ):
        torch.testing.assert_close(m_single, m, rtol=0, atol=1e-6)
        torch.testing.assert_close(m_uncoupled, m, rtol=0, atol=1e-12)
