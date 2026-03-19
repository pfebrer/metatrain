import itertools
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from metatomic.torch import System

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap

from .target_info import TargetInfo


def reindex_to_batch_index(
    tensor: TensorMap,
    system_ids: torch.tensor,
) -> TensorMap:
    """Reindex the system ids in the samples of the blocks of the tensor to have batch
    ids.
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


def get_reindex_to_batch_index_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> Callable:
    """
    Get a function that reindexes the systems to have batch ids.

    :return: A function that takes in systems, targets and extra data, and returns
        the systems, targets and extra data with reindexed batch ids.
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


def densify_atomic_basis_per_atom_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the per-atom atomic basis target by moving the "atom_type" key dimension to
    the samples, creating a padded property dimension according to the maximum property
    size for each irrep.
    """

    # First ensure that the tensor has all keys present in the layout tensor (i.e. the
    # global basis set definition). If any blocks aren't present, they are added as
    # zero-sample blocks with the correct components and properties.
    blocks = []
    for key, layout_block in layout.items():
        if key in tensor.keys:
            block = tensor.block(key)
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
    # return mts.keys_to_samples(tensor, fill_value=fill_value)
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
    non_type_names = [tensor.keys.names[i] for i in non_type_indices]

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
        padded_values[:, :, properties_mask] = block.values
        padded_block = TensorBlock(
            values=padded_values,
            samples=block.samples,
            components=block.components,
            properties=properties,
        )
        padded_blocks.append(padded_block)

    tensor = TensorMap(tensor.keys, padded_blocks)

    # Now move the "atom_type"-like key dimension to the samples and remove them
    tensor = tensor.keys_to_samples(type_names)
    for name in type_names:
        tensor = mts.remove_dimension(tensor, "samples", name)

    return tensor


def densify_atomic_basis_per_pair_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the per-atom atomic basis target by moving the "atom_type" key dimension to
    the samples, creating a padded property dimension according to the maximum property
    size for each irrep.
    """
    raise NotImplementedError(
        "Densification of per-pair atomic basis targets not yet implemented."
    )


def get_densify_atomic_basis_targets_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> Callable:
    """
    Get a function that densifies the atomic basis targets.

    :return: A function that takes in systems, targets and extra data, and returns
        the systems, targets and extra data with densified atomic basis targets.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Transform function that densifies the atomic basis targets, modifying
        in-place.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with densified atomic basis
            targets.
        """
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:

                # TODO: in next PR, add support for per-pair targets (both Cartesian and
                # coupled product basis)
                if tensor.keys.names == ["o3_lambda", "o3_sigma", "atom_type"]:
                    targets[name] = densify_atomic_basis_per_atom_target(
                        tensor, target_info_dict[name].layout
                    )

                else:
                    raise NotImplementedError(
                        f"Densification of atomic basis target {name} with keys "
                        f"{tensor.keys.names} not yet implemented."
                    )

        for name, tensor in extra.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:

                # TODO: in next PR, add support for per-pair targets (both Cartesian and
                # coupled product basis)
                if tensor.keys.names == ["o3_lambda", "o3_sigma", "atom_type"]:
                    extra[name] = densify_atomic_basis_per_atom_target(
                        tensor, target_info_dict[name].layout
                    )

                else:
                    raise NotImplementedError(
                        f"Densification of atomic basis target {name} with keys "
                        f"{tensor.keys.names} not yet implemented."
                    )

        return systems, targets, extra

    return transform


def slice_samples_atomic_basis_per_atom_target(
    atom_types_batch: torch.Tensor,
    tensor: TensorMap,
    layout: TensorMap,
) -> TensorMap:
    """
    According to the basis set implicitly stored in the ``layout``, slice the samples of
    each irrep to only keep the atoms with a type that has basis functions for the given
    irrep.
    """
    new_blocks: List[TensorBlock] = []
    for key_, block in tensor.items():
        o3_lambda_ = key_.values[0].item()
        o3_sigma_ = key_.values[1].item()

        valid_types_for_irrep: List[int] = []
        for key_i in range(len(layout)):
            o3_lambda = layout.keys.values[key_i][0].item()
            o3_sigma = layout.keys.values[key_i][1].item()
            atom_type = layout.keys.values[key_i][2].item()
            if o3_lambda_ == o3_lambda and o3_sigma_ == o3_sigma:
                valid_types_for_irrep.append(atom_type)

        valid_types_for_irrep = torch.tensor(valid_types_for_irrep)
        sample_mask = torch.isin(atom_types_batch, valid_types_for_irrep)
        new_blocks.append(
            TensorBlock(
                values=block.values[sample_mask],
                samples=Labels(
                    block.samples.names,
                    block.samples.values[sample_mask],
                ),
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(tensor.keys, new_blocks)


def sparsify_atomic_basis_target_per_atom(
    atom_types_batch: torch.Tensor,
    tensor: TensorMap,
    layout: TensorMap,
) -> TensorMap:
    """
    Sparsify the per-atom atomic basis target by creating blocks with an explicit
    "atom_type" dimension. The dense blocks of the input ``tensor`` are therefore sliced
    according to atom type of each atom in the samples.
    """
    # First sparsify by moving the "atom_type" from the samples to the keys
    unique_types = torch.unique(atom_types_batch)
    atom_type_masks: Dict[int, torch.Tensor] = {}
    for atom_type in unique_types:
        atom_type_masks[atom_type.item()] = atom_types_batch == atom_type

    new_keys: List[torch.Tensor] = []
    new_blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        for atom_type in unique_types:

            new_key = torch.cat([key.values[:2], atom_type.view(1)], dim=0)
            new_block = TensorBlock(
                values=block.values[atom_type_masks[atom_type.item()]],
                samples=Labels(
                    block.samples.names,
                    block.samples.values[atom_type_masks[atom_type.item()]],
                ),
                components=block.components,
                properties=block.properties,
            )

            new_keys.append(new_key)
            new_blocks.append(new_block)

    tensor = TensorMap(
        Labels(names=tensor.keys.names + ["atom_type"], values=torch.vstack(new_keys)),
        new_blocks,
    )

    # Now unpad the properties
    new_keys: List[torch.Tensor] = []
    unpadded_blocks: List[TensorBlock] = []
    for key, block in tensor.items():

        if key not in layout.keys:
            continue

        new_keys.append(key.values)
        layout_block = layout[key]
        properties_mask = layout_block.properties.select(block.properties)
        new_block = TensorBlock(
            values=block.values[:, :, properties_mask],
            samples=block.samples,
            components=block.components,
            properties=layout_block.properties,
        )
        unpadded_blocks.append(new_block)

    return TensorMap(
        Labels(tensor.keys.names, torch.vstack(new_keys)),
        unpadded_blocks
    )
