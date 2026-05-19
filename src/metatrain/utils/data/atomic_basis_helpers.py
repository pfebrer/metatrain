from typing import Callable, Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System

from ..data import DatasetInfo
from .target_info import TargetInfo


# ===== General utilities


def reindex_to_batch_index(
    tensor: TensorMap,
    system_ids: torch.tensor,
) -> TensorMap:
    """
    Reindex the system ids in the samples of the blocks of the tensor to have batch ids.

    :param tensor: the input TensorMap with system ids in the samples to reindex.
    :param system_ids: Tensor containing the actual system ids for each sample in the
        input ``tensor``.

    :return: the output TensorMap with batch ids in the samples instead of system ids.
    """
    id_mapping = torch.ones(system_ids.max().item() + 1, dtype=int) * -1
    for new_id, old_id in enumerate(system_ids):
        id_mapping[old_id] = new_id

    blocks = []
    for block in tensor:
        system_ids = block.samples.values[:, 0]
        batch_id = id_mapping[system_ids]
        block = TensorBlock(
            samples=Labels(
                block.samples.names,
                torch.hstack(
                    [
                        batch_id.reshape(-1, 1),
                        block.samples.values[:, 1:],
                    ]
                ),
            ),
            components=block.components,
            properties=block.properties,
            values=block.values,
        )
        blocks.append(block)
    return TensorMap(tensor.keys, blocks)


def get_per_atom_sample_labels(
    systems: List[System],
) -> Labels:
    """
    Builds the atom sample labels for the input ``systems``, assuming the systems have a
    batch index (0, ..., n_batch - 1).

    :param systems: List of systems to build the per-atom sample labels for.
    :return: The per-atom sample labels.
    """
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=systems[0].device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=systems[0].device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )
    return sample_labels


def get_per_atom_pair_sample_labels(
    systems: List[System],
    sample_labels: Labels,
    nl_options: NeighborListOptions,
) -> Labels:
    """
    Create per-pair sample labels from center and neighbor atom indices and cell shifts.

    Each row in the returned Labels corresponds to one directed edge (center → neighbor)
    in the neighbor list, identified in the same way as a standard metatensor neighbor
    list: by system index, center atom index, neighbor atom index, and the integer cell
    shift vector ``(cell_shift_a, cell_shift_b, cell_shift_c)``.

    :param systems: List of systems in the batch.
    :param sample_labels: Labels for all atoms in the batch, with dimensions
        ``["system", "atom"]``, as returned by :func:`systems_to_batch`.
    :param nl_options: Options for the neighbor list used to enumerate edges.
    :return: Labels with columns ``["system", "first_atom", "second_atom",
        "cell_shift_a", "cell_shift_b", "cell_shift_c"]``, shape ``(n_edges, 6)``.
    """
    device = systems[0].positions.device
    nl_values_list: List[torch.Tensor] = []
    num_edges: List[int] = []
    node_offsets_list: List[int] = []

    node_counter = 0
    for system in systems:
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values
        nl_values_list.append(nl_values)

        system_size = len(system)
        node_offsets_list.append(node_counter)
        num_edges.append(nl_values.shape[0])
        node_counter += system_size

    nl_values = torch.cat(nl_values_list)
    centers = nl_values[:, 0]
    neighbors = nl_values[:, 1]
    cell_shifts = nl_values[:, 2:]

    # Compute the offsets
    total_edges = sum(num_edges)
    num_edges_tensor = torch.tensor(num_edges, device=device, dtype=torch.long)
    node_offsets = torch.tensor(node_offsets_list, device=device, dtype=torch.long)
    edge_offsets = torch.repeat_interleave(
        node_offsets, num_edges_tensor, output_size=total_edges
    ).to(dtype=centers.dtype)

    # Offset the centers and neighbors
    centers = centers + edge_offsets
    neighbors = neighbors + edge_offsets

    sample_values = sample_labels.values  # (n_atoms, 2): [system, atom]
    center_values = sample_values[centers]  # (n_edges, 2): [system, first_atom]
    neighbor_values = sample_values[neighbors]  # (n_edges, 2): [system, second_atom]

    pair_values = torch.cat(
        [
            center_values[:, :1],  # system        (n_edges, 1)
            center_values[:, 1:],  # first_atom    (n_edges, 1)
            neighbor_values[:, 1:],  # second_atom   (n_edges, 1)
            cell_shifts,  # a, b, c       (n_edges, 3)
        ],
        dim=1,
    )

    return Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=pair_values,
        assume_unique=True,
    )


# ===== Densification utilities (keys with atom types to samples)


def _densify_per_atom_atomic_basis_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the per-atom atomic basis target by moving the "atom_type" key dimension to
    the samples, creating a padded property dimension according to the maximum property
    size for each irrep.

    :param tensor: the per-atom atomic basis target TensorMap to densify.
    :param layout: the layout TensorMap defining the global basis set.
    :param fill_value: the value to use for filling in the padded values when
        densifying.

    :return: the densified per-atom atomic basis target TensorMap.
    """

    # First ensure that the tensor has all keys present in the layout tensor (i.e. the
    # global basis set definition). If any blocks aren't present, they are added as
    # zero-sample blocks with the correct components and properties.
    blocks = []
    for key, layout_block in layout.items():
        if key in tensor.keys:
            existing_block = tensor.block(key)
            block = TensorBlock(
                values=existing_block.values,
                samples=existing_block.samples,
                components=existing_block.components,
                properties=existing_block.properties,
            )
        else:
            block = layout_block.copy()
            assert len(block.samples) == 0
        blocks.append(block)

    tensor = TensorMap(layout.keys, blocks)

    # Now densification can be done.

    # =====
    # TODO: the following is a manual densification, but this will be replaced with
    # `keys_to_samples(..., fill_value)` once publicly available in
    # metatensor-operations.
    # return mts.keys_to_samples(tensor, fill_value=fill_value, sort_samples=True)
    # =====

    # =====
    # For now, implement a manual densification:
    # =====

    # First, identify the "atom_type"-like and non-"atom_type"-like key dimensions.
    type_indices = [
        i for i, name in enumerate(tensor.keys.names) if name.endswith("atom_type")
    ]
    non_type_indices = [
        i for i, name in enumerate(tensor.keys.names) if not name.endswith("atom_type")
    ]
    type_names = [tensor.keys.names[i] for i in type_indices]

    # Using the layout TensorMap, build the union of the property labels values across
    # all atom types
    union_properties = {}
    for key, block in layout.items():
        key_vals = tuple([key.values[i].item() for i in non_type_indices])
        if key_vals not in union_properties:
            union_properties[key_vals] = block.properties
        else:
            union_properties[key_vals] = union_properties[key_vals].union(
                block.properties
            )

    # For each block, pad the properties using the dense properties
    padded_blocks = []
    for key, block in tensor.items():
        key_vals = tuple([key.values[i].item() for i in non_type_indices])
        properties = union_properties[key_vals]

        # Create a values array filled with the fill value
        padded_values = torch.full(
            (
                len(block.samples),
                *[len(c) for c in block.components],
                len(properties),
            ),
            fill_value,
            dtype=block.values.dtype,
        )

        # Now broadcast the existing values to the new shape
        properties_mask = properties.select(block.properties)
        padded_values[..., properties_mask] = block.values
        padded_block = TensorBlock(
            values=padded_values,
            samples=block.samples,
            components=block.components,
            properties=properties,
        )
        padded_blocks.append(padded_block)

    tensor = TensorMap(tensor.keys, padded_blocks)

    # Now move the "atom_type"-like key dimension to the samples and remove them
    tensor = tensor.keys_to_samples(type_names, sort_samples=True)
    for name in type_names:
        tensor = mts.remove_dimension(tensor, "samples", name)

    return tensor


def densify_atomic_basis_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the atomic basis target by moving any "atom_type"-like key dimensions to the
    samples, creating a padded property dimension according to the maximum property size
    for each irrep.

    :param tensor: the atomic basis target TensorMap to densify.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the output).
    :param fill_value: the value to use for filling in the padded values when
        densifying.

    :return: the densified atomic basis target TensorMap.
    """
    if "atom" in tensor.sample_names:
        return _densify_per_atom_atomic_basis_target(tensor, layout, fill_value)
    else:
        return _densify_per_atom_atomic_basis_target(tensor, layout, fill_value)

    raise NotImplementedError(
        "Currently only densification of per-atom atomic basis targets is implemented."
    )


def _pad_block(
    block: TensorBlock,
    axis: str,
    padded_labels: Labels,
    fill_value: float,
) -> TensorBlock:
    """
    Takes a TensorBlock and pads it along the specified axis with zeros, using the
    provided padded_labels to determine the metadata structure of the new block.

    Any of the samples/properties that are already present in the block will be filled
    with the original values, while the new samples/properties will be filled with
    zeros.

    :param block: The TensorBlock to pad.
    :param axis: Either ``"samples"`` or ``"properties"``, specifying which axis to
        pad.
    :param padded_labels: The target Labels defining the full (padded) extent of the
        chosen axis.
    :param fill_value: Value used to fill positions that have no corresponding entry
        in the original block.
    :return: A new TensorBlock padded to the requested shape along the chosen axis.
    """

    def get_idxs_pad_idxs_original(
        padded_labels: Labels, original_labels: Labels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # First find the intersection
        intersection = padded_labels.intersection(original_labels)
        # Get the indices of the intersection in the padded_labels
        nonzero_idxs_padded = padded_labels.select(intersection)
        # Get the indices of the intersection in the original_labels
        nonzero_idxs_original = original_labels.select(intersection)

        return nonzero_idxs_padded, nonzero_idxs_original

    if axis == "samples":
        samples = padded_labels
        properties = block.properties

        block_values = torch.full(
            (len(samples), *[len(c) for c in block.components], len(properties)),
            fill_value=fill_value,
            dtype=block.values.dtype,
        )
        nonzero_idxs_padded, nonzero_idxs_original = get_idxs_pad_idxs_original(
            samples, block.samples
        )
        assert len(nonzero_idxs_padded) == len(nonzero_idxs_original)
        if len(nonzero_idxs_original) > 0:
            block_values[nonzero_idxs_padded] = block.values[nonzero_idxs_original]

    else:
        assert axis == "properties"
        samples = block.samples
        properties = padded_labels

        block_values = (
            torch.ones(
                (len(samples), *[len(c) for c in block.components], len(properties)),
                like=block.values,
            )
            * fill_value
        )

        block_values = torch.full(
            (len(samples), *[len(c) for c in block.components], len(properties)),
            fill_value=fill_value,
            dtype=block.values.dtype,
        )
        nonzero_idxs_padded, nonzero_idxs_original = get_idxs_pad_idxs_original(
            properties, block.properties
        )
        assert len(nonzero_idxs_padded) == len(nonzero_idxs_original)
        if len(nonzero_idxs_original) > 0:
            block_values[..., nonzero_idxs_padded] = block.values[
                ..., nonzero_idxs_original
            ]

    return TensorBlock(
        samples=samples,
        components=block.components,
        properties=properties,
        values=block_values,
    )


def _pad_samples_per_atom_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
) -> TensorMap:
    sample_labels = get_per_atom_sample_labels(systems)
    new_blocks = []
    for block in tensor:
        new_blocks.append(_pad_block(block, "samples", sample_labels, torch.nan))

    return TensorMap(tensor.keys, new_blocks)


def _pad_samples_per_atom_pair_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    nl_options: NeighborListOptions,
) -> TensorMap:
    # Build the samples labels
    sample_labels = get_per_atom_sample_labels(systems)
    pair_sample_labels = get_per_atom_pair_sample_labels(
        systems,
        sample_labels,
        nl_options,
    )

    # Pad the samples according to the neighborlist pair samples
    new_blocks = []
    for block in tensor:
        new_blocks.append(_pad_block(block, "samples", pair_sample_labels, torch.nan))

    return TensorMap(tensor.keys, new_blocks)


def pad_samples_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    nl_options: NeighborListOptions,
) -> TensorMap:
    """
    Pad the samples of the atomic basis target to have the same number of samples for
    each block.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to pad.
    :param nl_options: Options for the neighbor list used to enumerate edges for
        per-atom-pair targets.

    :return: the padded atomic basis target TensorMap
    """

    if "atom" in tensor.sample_names:
        return _pad_samples_per_atom_atomic_basis_target(systems, tensor)
    elif "first_atom" in tensor.sample_names:
        return _pad_samples_per_atom_pair_atomic_basis_target(
            systems, tensor, nl_options
        )
    raise NotImplementedError(
        "Currently only padding of per-atom atomic basis targets is implemented."
    )


def prepare_atomic_basis_targets(
    systems: List[System],
    system_ids: torch.Tensor,
    tensor: TensorMap,
    layout: TensorMap,
    nl_options: Optional[NeighborListOptions] = None,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Prepare the atomic basis targets for batching by reindexing to batch ids, densifying
    (moving "atom_type" key dimensions to the samples) and padding the samples.

    :param systems: List of systems in the batch.
    :param system_ids: Tensor containing the system ids for each sample in the input
        ``tensor``.
    :param tensor: the atomic basis target TensorMap to prepare.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the output).
    :param nl_options: Options for the neighbor list.
    :param fill_value: the value to use for filling in the padded values when densifying
        and padding. Default is NaN, but can be set to 0 if desired (e.g. for
        classification targets).
    :return: the prepared atomic basis target TensorMap with batch ids, densified and
        padded.
    """

    # Reindex to batch ids
    tensor = reindex_to_batch_index(tensor, system_ids)

    # Densify: "atom type" key dimensions -> samples
    tensor = densify_atomic_basis_target(tensor, layout, fill_value)

    # Pad samples
    tensor = pad_samples_atomic_basis_target(systems, tensor, nl_options)

    return tensor


# ===== Sparsification utilities (atom types back to keys)


def _compute_sparse_properties(layout: TensorMap) -> TensorMap:
    """Compute the properties of the sparse tensor map that results from
    sparsifying a given dense tensor map.

    This can be computed once and for all for a given layout, and be reused to
    sparsify as many padded tensor maps as needed with the function
    :func:`sparsify_atomic_basis_target`.

    :param layout: the layout ``TensorMap`` defining the global basis set.
    :return: The ``TensorMap`` to be used to go from the dense to the sparse
      representation. The values of each block are the indices of the properties
      to select from a padded tensor map.
    """
    non_type_indices: list[int] = []
    for i, name in enumerate(layout.keys.names):
        if not name.endswith("atom_type"):
            non_type_indices.append(i)

    # Build union of properties across atom types for each non-type key — same
    # logic as _densify_per_atom_atomic_basis_target.
    union_properties: dict[str, Labels] = {}
    for key, block in layout.items():
        k = str([key.values[i].item() for i in non_type_indices])
        if k not in union_properties:
            union_properties[k] = block.properties
        else:
            union_properties[k] = union_properties[k].union(block.properties)

    blocks: list[TensorBlock] = []
    for key, block in layout.items():
        padded_properties = union_properties[
            str([key.values[i].item() for i in non_type_indices])
        ]

        blocks.append(
            TensorBlock(
                values=padded_properties.select(block.properties).reshape(1, -1),
                samples=Labels(["_"], torch.zeros((1, 1), dtype=torch.int64)),
                components=[],
                properties=block.properties,
            )
        )

    return TensorMap(layout.keys, blocks)


def _sparsify_per_atom_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    sparse_properties: TensorMap,
    atom_types_batch: Optional[torch.Tensor] = None,
) -> TensorMap:
    """
    Sparsify the per-atom atomic basis target by creating blocks with an explicit
    "atom_type" dimension. The dense blocks of the input ``tensor`` are therefore sliced
    according to atom type of each atom in the samples.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param sparse_properties: A TensorMap containing the properties to select
        from the dense tensor map to create each block in the sparse layout.
    :param atom_types_batch: Optional tensor containing the atom types for each sample
        in the batch. If not provided, these are inferred from the systems.
    :return: the sparsified atomic basis target TensorMap
    """
    if atom_types_batch is None:
        # Get the atom types for each sample in the batch from the systems
        atom_types_batch = torch.cat(
            [system.types for system in systems],
            dim=0,
        )

    device = tensor.device
    atom_types_batch = atom_types_batch.to(device=device)

    unique_types: List[int] = (
        torch.unique(atom_types_batch).to(torch.int64).cpu().tolist()
    )

    # Build per-type boolean masks and sample labels as parallel lists (indexed
    # by position in unique_types). Using parallel lists avoids Dict[int, Labels]
    # which may not be supported by all TorchScript backends.
    sample_indices = torch.arange(len(atom_types_batch), device=device)
    type_masks: List[torch.Tensor] = []
    type_samples: List[Labels] = []
    ref_block = tensor.block_by_id(0)
    for atom_type in unique_types:
        mask = sample_indices[atom_types_batch == atom_type]
        type_masks.append(mask)
        type_samples.append(
            Labels(
                names=["system", "atom"],
                values=ref_block.samples.values[mask],
            )
        )

    sparse_properties = sparse_properties.to(device=device)

    # Build the sparsified tensormap by looping over the dense one
    # and splitting each block into one block per atom type.
    new_keys: list[list[int]] = []
    sparse_blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        key_values: List[int] = key.values.to(torch.int64).tolist()
        for i_type, atom_type in enumerate(unique_types):
            new_key = key_values + [atom_type]

            # Get the corresponding layout block to know which properties to select
            block_position = sparse_properties.keys.position(new_key)
            if block_position is None:
                # This irrep doesn't exist for this atom type in the layout
                continue
            assert block_position is not None  # for torchscript
            layout_block = sparse_properties.block_by_id(block_position)

            # Select samples
            values = block.values[type_masks[i_type]]
            # Select properties
            properties_mask = layout_block.values.ravel()
            # Do block.values[..., properties_mask] in a torchscriptable way.
            if block.values.ndim == 3:
                values = values[:, :, properties_mask]
            elif block.values.ndim == 4:
                values = values[:, :, :, properties_mask]
            else:
                raise ValueError(
                    "Tensorblocks with more than 2 component dimensions can't be "
                    "sparsified with the current implementation."
                )

            sparse_block = TensorBlock(
                values=values,
                samples=type_samples[i_type],
                components=block.components,
                properties=layout_block.properties,
            )

            new_keys.append(new_key)
            sparse_blocks.append(sparse_block)

    tensor = TensorMap(
        Labels(
            names=tensor.keys.names + ["atom_type"],
            values=torch.tensor(new_keys, device=device),
        ),
        sparse_blocks,
    )

    return tensor


def _sparsify_per_atom_pair_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    sparse_properties: TensorMap,
    atom_types_batch: Optional[torch.Tensor] = None,
) -> TensorMap:
    """
    Sparsify the per-atom-pair atomic basis target by creating blocks with explicit
    "first_atom_type" and "second_atom_type" dimensions.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param sparse_properties: A TensorMap containing the properties to select.
    :param atom_types_batch: Optional concatenated atom-type tensor for the batch.
    :return: the sparsified atomic basis target TensorMap
    """
    if atom_types_batch is None:
        # Get the atom types for each sample in the batch from the systems
        atom_types_batch = torch.cat(
            [system.types for system in systems],
            dim=0,
        )

    device = tensor.device
    atom_types_batch = atom_types_batch.to(device=device)

    unique_types: List[int] = (
        torch.unique(atom_types_batch).to(torch.int64).cpu().tolist()
    )

    sparse_properties = sparse_properties.to(device=device)

    # Pre-compute per-system atom offsets so we can look up types from
    # atom_types_batch with a single vectorized indexing step.
    n_systems = len(systems)
    system_lengths = torch.tensor(
        [len(s.types) for s in systems], dtype=torch.long, device=device
    )
    offsets = torch.zeros(n_systems, dtype=torch.long, device=device)
    if n_systems > 1:
        offsets[1:] = torch.cumsum(system_lengths[:-1], dim=0)

    new_keys: List[List[int]] = []
    sparse_blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        key_values: List[int] = key.values.to(torch.int64).tolist()

        # samples columns: system(0), first_atom(1), second_atom(2), cell_shift_*(3-5)
        sample_vals = block.samples.values.long()
        sys_col = sample_vals[:, 0]
        first_atom_col = sample_vals[:, 1]
        second_atom_col = sample_vals[:, 2]

        # Vectorized type lookup: avoids per-sample Python loops
        first_types = atom_types_batch[offsets[sys_col] + first_atom_col]
        second_types = atom_types_batch[offsets[sys_col] + second_atom_col]

        for first_atom_type in unique_types:
            for second_atom_type in unique_types:
                mask = (first_types == first_atom_type) & (
                    second_types == second_atom_type
                )
                new_key = key_values + [first_atom_type, second_atom_type]

                block_position = sparse_properties.keys.position(new_key)
                if block_position is None:
                    continue
                assert block_position is not None  # for torchscript
                layout_block = sparse_properties.block_by_id(block_position)

                # Select samples
                values = block.values[mask]
                # Select properties
                properties_mask = layout_block.values.ravel()
                # Do block.values[..., properties_mask] in a torchscriptable way.
                if block.values.ndim == 3:
                    values = values[:, :, properties_mask]
                elif block.values.ndim == 4:
                    values = values[:, :, :, properties_mask]
                else:
                    raise ValueError(
                        "Tensorblocks with more than 2 component dimensions can't be "
                        "sparsified with the current implementation."
                    )

                sparse_block = TensorBlock(
                    values=values,
                    samples=Labels(
                        names=[
                            "system",
                            "first_atom",
                            "second_atom",
                            "cell_shift_a",
                            "cell_shift_b",
                            "cell_shift_c",
                        ],
                        values=block.samples.values[mask],
                    ),
                    components=block.components,
                    properties=layout_block.properties,
                )

                new_keys.append(new_key)
                sparse_blocks.append(sparse_block)

    tensor = TensorMap(
        Labels(
            names=tensor.keys.names + ["first_atom_type", "second_atom_type"],
            values=torch.tensor(new_keys, device=device),
        ),
        sparse_blocks,
    )

    return tensor


def sparsify_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    layout: Optional[TensorMap] = None,
    atom_types_batch: Optional[torch.Tensor] = None,
    sparse_properties: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Sparsify the atomic basis target by creating blocks with an explicit "atom_type"
    dimension. The dense blocks of the input ``tensor`` are therefore sliced according
    to atom type of each atom in the samples.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the sparsified output). Required if
        ``sparse_properties`` is not provided.
    :param atom_types_batch: Optional tensor containing the atom types for each sample
        in the batch. If not provided, these are inferred from the systems.
    :param sparse_properties: A TensorMap containing the properties to select
        from the dense tensor map to create each block in the sparse layout.
        If not provided, it is computed from ``layout``.

    :return: the sparsified atomic basis target TensorMap
    """
    if sparse_properties is None:
        if layout is None:
            raise ValueError("Either `layout` or `sparse_properties` must be provided.")
        sparse_properties = _compute_sparse_properties(layout)

    if "atom" in tensor.sample_names:
        return _sparsify_per_atom_atomic_basis_target(
            systems, tensor, sparse_properties, atom_types_batch
        )
    elif "first_atom" in tensor.sample_names:
        return _sparsify_per_atom_pair_atomic_basis_target(
            systems, tensor, sparse_properties, atom_types_batch
        )

    raise NotImplementedError(
        "Currently only sparsification of per-atom atomic basis targets is implemented."
    )


# ===== dataloader transforms


def get_reindex_to_batch_index_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> Callable:
    """
    Get a function that reindexes the systems to have batch ids.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.

    :return: A function that takes in systems, targets and extra data, and returns the
        systems, targets and extra data with reindexed batch ids.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Transform function that reindexes the systems to have batch ids, modifying
        in-place. Only applied to atomic basis targets and extra data.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with reindexed system ids.
        """
        assert "mtt::aux::system_index" in extra
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:
                targets[name] = reindex_to_batch_index(
                    tensor,
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64),
                )

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].is_atomic_basis
            ):
                extra[name] = reindex_to_batch_index(
                    tensor,
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64),
                )

        return systems, targets, extra

    return transform


def get_prepare_atomic_basis_targets_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
    nl_options: Optional[NeighborListOptions] = None,
) -> Tuple[Callable, Callable]:
    """
    Get a function that prepares the atomic basis targets for batching by reindexing to
    batch ids, densifying and padding.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.
    :param nl_options: Options for the neighbor list used to enumerate edges for
        per-atom-pair targets.

    :return: A function that takes in systems, targets and extra data, and returns the
        systems, targets and extra data with prepared atomic basis targets.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Transform function that prepares the atomic basis targets for batching by
        reindexing to batch ids, densifying and padding, modifying in-place.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with prepared atomic basis targets.
        """
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:
                assert "mtt::aux::system_index" in extra
                system_ids = (
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64)
                )

                targets[name] = prepare_atomic_basis_targets(
                    systems,
                    system_ids,
                    tensor,
                    target_info_dict[name].layout,
                    nl_options,
                    fill_value=torch.nan,
                )

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].is_atomic_basis
            ):
                assert "mtt::aux::system_index" in extra
                system_ids = (
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64)
                )

                extra[name] = prepare_atomic_basis_targets(
                    systems,
                    system_ids,
                    tensor,
                    extra_data_info_dict[name].layout,
                    nl_options,
                    fill_value=torch.nan,
                )

        return systems, targets, extra

    # Precompute property masks for all atomic basis targets and extra data.
    # This avoids calling layout_properties.select(block.properties) every
    # time we want to sparsify a batch.
    sparse_properties: dict[str, TensorMap] = {}
    for name, info in target_info_dict.items():
        if info.is_atomic_basis:
            sparse_properties[name] = _compute_sparse_properties(info.layout)
    for name, info in extra_data_info_dict.items():
        if info.is_atomic_basis:
            sparse_properties[name] = _compute_sparse_properties(info.layout)

    def reverse_transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Reverse transform function that unpads, undensifies and reindexes the atomic
        basis targets, modifying in-place.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with unprepared atomic basis
            targets.
        """
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:
                if name in sparse_properties:
                    sparse_properties[name] = sparse_properties[name].to(tensor.device)
                targets[name] = sparsify_atomic_basis_target(
                    systems,
                    tensor,
                    target_info_dict[name].layout,
                    sparse_properties=sparse_properties.get(name),
                )

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].is_atomic_basis
            ):
                if name in sparse_properties:
                    sparse_properties[name] = sparse_properties[name].to(tensor.device)
                extra[name] = sparsify_atomic_basis_target(
                    systems,
                    tensor,
                    extra_data_info_dict[name].layout,
                    sparse_properties=sparse_properties.get(name),
                )

        return systems, targets, extra

    return transform, reverse_transform


# ===== DatasetInfo manipulation utilities


def densify_atomic_basis_dataset_info(dataset_info: DatasetInfo) -> DatasetInfo:
    """
    Densify the atomic basis target layouts in the TargetInfos of the input
    ``dataset_info`` by moving any

    :param dataset_info: the input DatasetInfo with TargetInfos containing atomic basis
        targets to densify.
    :return: a new DatasetInfo with the same information as the input, but with the
        atomic basis targets densified.
    """

    return DatasetInfo(
        length_unit=dataset_info.length_unit,
        atomic_types=dataset_info.atomic_types,
        targets={
            target_name: (
                TargetInfo(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    layout=densify_atomic_basis_target(
                        target_info.layout, target_info.layout
                    ),
                    description=target_info.description,
                )
                if target_info.is_atomic_basis
                else target_info
            )
            for target_name, target_info in dataset_info.targets.items()
        },
    )
