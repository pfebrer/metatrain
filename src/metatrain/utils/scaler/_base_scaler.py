"""
Contains the ``BaseScaler`` class. This is intended for eventual porting to metatomic.
The class ``Scaler`` wraps this to be compatible with metatrain-style objects.
"""

import itertools
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


FixedScalerWeights = dict[
    str, Union[float, dict[int, float], dict[int, dict[int, float]]]
]


class BaseScaler(torch.nn.Module):
    """
    Fits a scaler for a dict of targets. Scales are computed as the per-property (and
    therefore per-block) standard deviations. By default, the scales are also
    computed per atomic type for per-atom targets.

    The :py:method:`accumulate` method is used to accumulate the necessary quantities
    based on the training data, and the :py:method:`fit` method is used to fit the model
    based on the accumulated quantities. These should both be called before the
    :py:method:`forward` method is called to compute the scales at inference
    time.

    :param atomic_types: List of atomic types to use in the composition model.
    :param layouts: Dict of zero-sample layout :py:class:`TensorMap` corresponding to
        each target. The keys of the dict are the target names, and the values are
        :py:class:`TensorMap` objects with the zero-sample layout for each target.
    :param per_property_for_atom_pair_targets: Whether to fit per-property scales for
        atom-pair targets.
    """

    # Needed for torchscript compatibility
    target_names: List[str]
    scales: Dict[str, TensorMap]
    sample_kinds: Dict[str, str]
    type_to_index: torch.Tensor
    type_pair_to_index: torch.Tensor
    N: Dict[str, TensorMap]
    Y2: Dict[str, TensorMap]
    per_property_N: Dict[str, TensorMap]
    per_property_Y2: Dict[str, TensorMap]
    per_property_scales: Dict[str, TensorMap]  # per-property scales
    per_target_scales: Dict[str, TensorMap]  # per-target scales
    multi_property_target_names: List[str]
    per_property_for_atom_pair_targets: bool

    def __init__(
        self,
        atomic_types: List[int],
        layouts: Dict[str, TensorMap],
        per_property_for_atom_pair_targets: bool = True,
    ) -> None:
        super().__init__()

        self.per_property_for_atom_pair_targets = per_property_for_atom_pair_targets
        self.atomic_types = torch.as_tensor(atomic_types, dtype=torch.int32)
        self.target_names = []
        self.sample_kinds = {}
        self.N = {}
        self.Y2 = {}
        self.scales = {}
        self.per_property_N = {}
        self.per_property_Y2 = {}
        self.per_property_scales = {}
        self.per_target_scales = {}
        self.multi_property_target_names = []

        # go from an atomic type to its position in `self.atomic_types`
        self.register_buffer(
            "type_to_index", torch.empty(max(self.atomic_types) + 1, dtype=torch.long)
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.type_to_index[atomic_type] = i

        # go from an (first_atom_type, second_atom_type) pair to a flat index
        # in [0, n_types^2), where the flat index is i * n_types + j with
        # i = type_to_index[first_atom_type], j = type_to_index[second_atom_type].
        # This 2D buffer avoids recomputing the flat index via type_to_index lookups
        # and two arithmetic operations every time we need it.
        n_types = len(self.atomic_types)
        max_type = int(self.atomic_types.max().item())
        type_pair_to_index = torch.full(
            (max_type + 1, max_type + 1), -1, dtype=torch.long
        )
        for i, at_i in enumerate(self.atomic_types.tolist()):
            for j, at_j in enumerate(self.atomic_types.tolist()):
                type_pair_to_index[at_i, at_j] = i * n_types + j
        self.register_buffer("type_pair_to_index", type_pair_to_index)

        # Add targets based on provided layouts
        for target_name, layout in layouts.items():
            self.add_output(target_name, layout)

    def add_output(self, target_name: str, layout: TensorMap) -> None:
        """
        Adds a new target to the composition model.

        :param target_name: Name of the target to add.
        :param layout: Layout of the target as a :py:class:`TensorMap`.
        """
        if target_name in self.target_names:
            raise ValueError(f"target {target_name} already exists in the model.")

        self.target_names.append(target_name)
        valid_sample_names = [
            ["system"],
            [
                "system",
                "atom",
            ],
            [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
        ]

        if layout.sample_names == valid_sample_names[0]:
            self.sample_kinds[target_name] = "per_structure"
            samples = Labels(["atomic_type"], torch.tensor([[-1]]))

        elif layout.sample_names == valid_sample_names[1]:
            self.sample_kinds[target_name] = "per_atom"
            samples = Labels(
                ["atomic_type"], torch.arange(len(self.atomic_types)).reshape(-1, 1)
            )

        elif layout.sample_names == valid_sample_names[2]:
            self.sample_kinds[target_name] = "per_atom_pair"
            n_types = len(self.atomic_types)
            pair_values = torch.tensor(
                list(itertools.product(self.atomic_types.tolist(), repeat=2)),
                dtype=torch.int32,
            ).reshape(n_types * n_types, 2)
            samples = Labels(["first_atomic_type", "second_atomic_type"], pair_values)

        else:
            raise ValueError(
                "unknown sample kind. TensorMap has sample names"
                f" {layout.sample_names} but expected one of "
                f"{valid_sample_names}."
            )

        # Initialize TensorMaps for the quantities to accumulate for this target.

        # First, the full scales. These are the multiplication of per-target and
        # per-property (if applicable) scales, stored with the same layout as the
        # targets for convenient application. For single-block, single-property targets,
        # these scales are just the per-target scales as the per-property scales are by
        # definition 1.
        self.N[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.Y2[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.scales[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )

        # Store the per-target scales separately, as these are needed to be
        # applied/removed separately from the per-property scales, i.e. during training
        self.per_target_scales[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )

        if len(layout.keys) > 1 or len(layout[0].properties) > 1:
            is_atom_pair = "first_atom" in layout.sample_names
            if not is_atom_pair or self.per_property_for_atom_pair_targets:
                self.multi_property_target_names.append(target_name)

        # Next, per-property scales. These have a single value per-block and
        # per-property in the target, which is also separately computed for each atomic
        # type for per-atom targets. These are only computed for targets with multiple
        # blocks or multiple properties.
        self.per_property_N[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.per_property_Y2[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )
        self.per_property_scales[target_name] = TensorMap(
            layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.ones(
                        len(samples),
                        len(block.properties),
                        dtype=torch.float64,
                    ),
                    samples=samples,
                    components=[],
                    properties=block.properties,
                )
                for block in layout
            ],
        )

    def _compute_N_and_Y2(
        self,
        systems: List[System],
        target_name: str,
        target: TensorMap,
        per_property: bool,
        mask: Optional[TensorMap],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        N_list = []
        Y2_list = []
        for key, block in target.items():
            Y_block = block.to(device=self.N[target_name][0].values.device)
            Y = Y_block.values

            if per_property:
                # Compute sum over all axes except the property axis
                dim = list(range(0, Y.dim() - 1))
            else:
                # Compute sum over all axes
                dim = list(range(0, Y.dim()))

            # First get the mask
            if mask is None:  # inferred from target
                mask_vals = ~torch.isnan(Y_block.values)
            else:  # mask provided
                mask_vals = mask[key].values

            # Set any NaNs to zero so they don't contribute to the sum
            mask_vals = mask_vals.to(Y.dtype)
            Y[torch.isnan(Y)] = 0.0

            # Now handle the different cases: per-target vs per-property and
            # per-structure vs per-atom. Hnadle the mask in all cases.
            if self.sample_kinds[target_name] == "per_structure":
                N = mask_vals.sum(dim=dim)
                Y2 = torch.sum((Y * mask_vals) ** 2, dim=dim)

            elif self.sample_kinds[target_name] == "per_atom":
                block_types = torch.cat([system.types for system in systems])

                # Initialize N and Y2 tensors for this block, which will store the
                # values for each atomic type. For per-property scales, these have shape
                # (n_types, n_properties), and for per-target scales, these have shape
                # (n_types,).
                if per_property:
                    shape = [len(self.atomic_types), len(Y_block.properties)]
                else:
                    shape = [len(self.atomic_types)]
                N = torch.zeros(
                    tuple(shape),
                    dtype=torch.long,
                    device=Y.device,
                )
                Y2 = torch.zeros(
                    tuple(shape),
                    dtype=Y.dtype,
                    device=Y.device,
                )

                if "atom_type" in key.names:
                    atomic_type = key["atom_type"]
                    i = self.type_to_index[atomic_type]

                    N[i] = mask_vals.sum(dim=dim)
                    Y2[i] = torch.sum(
                        (Y * mask_vals) ** 2,
                        dim=dim,
                    )
                else:
                    for i, atomic_type in enumerate(self.atomic_types):
                        type_mask = block_types == atomic_type

                        N[i] = mask_vals[type_mask].sum(dim=dim)
                        Y2[i] = torch.sum(
                            (Y[type_mask] * mask_vals[type_mask]) ** 2,
                            dim=dim,
                        )

            else:
                assert self.sample_kinds[target_name] == "per_atom_pair"

                n_types = len(self.atomic_types)
                n_pairs = n_types * n_types

                if per_property:
                    shape = [n_pairs, len(Y_block.properties)]
                else:
                    shape = [n_pairs]

                N = torch.zeros(tuple(shape), dtype=torch.long, device=Y.device)
                Y2 = torch.zeros(tuple(shape), dtype=Y.dtype, device=Y.device)

                if "first_atom_type" in key.names and "second_atom_type" in key.names:
                    # All samples in this block share the same type pair encoded in
                    # the key — look up the flat index directly.
                    first_type = key["first_atom_type"]
                    second_type = key["second_atom_type"]
                    flat_idx = self.type_pair_to_index[first_type, second_type].item()
                    N[flat_idx] = mask_vals.sum(dim=dim)
                    Y2[flat_idx] = torch.sum((Y * mask_vals) ** 2, dim=dim)
                else:
                    # Infer the type pair for each sample from the atom indices stored
                    # in the block samples.
                    first_atom_col = Y_block.samples.names.index("first_atom")
                    second_atom_col = Y_block.samples.names.index("second_atom")

                    n_samples = Y_block.samples.values.shape[0]
                    if n_samples > 0:
                        # Build a concatenated type array and per-system atom offsets.
                        # All lookups are done on CPU to avoid potential device issues
                        # with system.types, then moved to Y.device.
                        all_types_cpu = torch.cat([s.types.cpu() for s in systems])
                        system_lengths_cpu = torch.tensor(
                            [len(s.types) for s in systems], dtype=torch.long
                        )
                        offset_cpu = torch.zeros(len(systems), dtype=torch.long)
                        if len(systems) > 1:
                            offset_cpu[1:] = torch.cumsum(
                                system_lengths_cpu[:-1], dim=0
                            )

                        system_indices_cpu = Y_block.samples.values[:, 0].long().cpu()
                        first_atom_indices_cpu = (
                            Y_block.samples.values[:, first_atom_col].long().cpu()
                        )
                        second_atom_indices_cpu = (
                            Y_block.samples.values[:, second_atom_col].long().cpu()
                        )

                        # Resolve atom indices → types, preserving original sample order
                        block_first_types = all_types_cpu[
                            offset_cpu[system_indices_cpu] + first_atom_indices_cpu
                        ].to(Y.device)
                        block_second_types = all_types_cpu[
                            offset_cpu[system_indices_cpu] + second_atom_indices_cpu
                        ].to(Y.device)

                        # Map each sample's (first_type, second_type) → flat pair index
                        sample_pair_indices = self.type_pair_to_index[
                            block_first_types, block_second_types
                        ]

                        # Accumulate N and Y2 per unique type pair
                        for flat_idx in sample_pair_indices.unique().tolist():
                            pair_mask = sample_pair_indices == flat_idx
                            N[flat_idx] = mask_vals[pair_mask].sum(dim=dim)
                            Y2[flat_idx] = torch.sum(
                                (Y[pair_mask] * mask_vals[pair_mask]) ** 2,
                                dim=dim,
                            )
                    # If n_samples == 0, N and Y2 remain zero (correct).

            N_list.append(N)
            Y2_list.append(Y2)

        return N_list, Y2_list

    def accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """
        Takes a batch of targets, and for each target accumulates the necessary
        quantities, i.e. the sum over the squared samples (Y2), and the number of
        samples overall (N). This function computes a single scale per target for
        per-structure targets, or a scale per atomic type for per-atom quantities.

        :param systems: List of systems corresponding to the targets.
        :param targets: Dict of names to targets to accumulate. The names (keys) should
            be a subset of the target names used during fitting.
        :param extra_data: Optional dict of extra data, e.g., masks for the targets
            (e.g., for padded samples).
        """

        if extra_data is None:
            extra_data = {}

        device = list(targets.values())[0][0].values.device
        dtype = list(targets.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # accumulate per-target N and Y2 quantities
        for target_name, target in targets.items():
            mask = None
            if target_name + "_mask" in extra_data:
                mask = extra_data[target_name + "_mask"]

            N_list, Y2_list = self._compute_N_and_Y2(
                systems=systems,
                target_name=target_name,
                target=target,
                per_property=False,
                mask=mask,
            )

            # Stack N and Y2
            N = torch.stack(N_list).sum(dim=0).reshape(-1, 1)
            Y2 = torch.stack(Y2_list).sum(dim=0).reshape(-1, 1)

            # Store for each block, repeating the same values and copying along the
            # properties dimension
            for key in self.N[target_name].keys:
                self.N[target_name][key].values[:] += N.repeat(
                    1, self.N[target_name][key].values.shape[1]
                )
                self.Y2[target_name][key].values[:] += Y2.repeat(
                    1, self.Y2[target_name][key].values.shape[1]
                )

    def accumulate_per_property(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> None:
        """
        Takes a batch of targets, and for each target accumulates the necessary
        quantities, i.e. the sum over the squared samples (per_property_Y2), and the
        number of samples overall (per_property_N). This function computes per-block and
        per-property scales. If the target is per-atom, scales are computed separately
        for each atomic type.

        :param systems: List of systems corresponding to the targets.
        :param targets: Dict of names to targets to accumulate. The names (keys) should
            be a subset of the target names used during fitting.
        :param extra_data: Optional dict of extra data, e.g., masks for the targets
            (e.g., for padded samples).
        """

        if extra_data is None:
            extra_data = {}

        device = list(targets.values())[0][0].values.device
        dtype = list(targets.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # Only accumulate targets with multiple properties
        targets = {
            target_name: target
            for target_name, target in targets.items()
            if target_name in self.multi_property_target_names
        }

        # Remove the per-target scales from the targets before accumulating per-property
        # quantities, so that we only accumulate the pure per-property correction
        # factors.
        targets = self.forward(
            systems,
            targets,
            remove=True,
            use_per_target_scales=True,
            use_per_property_scales=False,
        )

        # accumulate per-property quantities
        for target_name, target in targets.items():
            mask = None
            if target_name + "_mask" in extra_data:
                mask = extra_data[target_name + "_mask"]

            N_list, Y2_list = self._compute_N_and_Y2(
                systems=systems,
                target_name=target_name,
                target=target,
                per_property=True,
                mask=mask,
            )
            for key, N, Y2 in zip(target.keys, N_list, Y2_list, strict=True):
                self.per_property_N[target_name][key].values[:] += N
                self.per_property_Y2[target_name][key].values[:] += Y2

    def fit(
        self,
        fixed_weights: Optional[FixedScalerWeights] = None,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        per-target scales.

        :param fixed_weights: Optional dict of fixed weights to apply to the scales of
            each target. The keys of the dict are the target names, and the values are
            either a single float value to be applied to all atomic types, or a dict
            mapping atomic type (int) to weight (float). If not provided, all scales
            will be computed based on the accumulated quantities.
        :param targets_to_fit: Optional list of target names to fit. If not provided,
            all targets will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        if fixed_weights is None:
            fixed_weights = {}

        # fit and store per-target scales
        for target_name in targets_to_fit:
            if target_name in fixed_weights:
                self._set_fixed_weights(target_name, fixed_weights[target_name])
            else:
                for key in self.scales[target_name].keys:
                    scales_vals = (
                        self.Y2[target_name].block(key).values
                        / self.N[target_name].block(key).values
                    ) ** 0.5
                    self.scales[target_name][key].values[:] = scales_vals
                    self.per_target_scales[target_name][key].values[:] = scales_vals

        # NaNs can arise if a target has zero samples in the training data, so we set
        # the scale to 1.0 to avoid issues during training.
        for target_name in targets_to_fit:
            for key in self.scales[target_name].keys:
                self.scales[target_name][key].values[:] = torch.nan_to_num(
                    self.scales[target_name][key].values, nan=1.0
                )
                self.per_target_scales[target_name][key].values[:] = torch.nan_to_num(
                    self.per_target_scales[target_name][key].values, nan=1.0
                )

    def fit_per_property(
        self,
        targets_to_fit: Optional[List[str]] = None,
    ) -> None:
        """
        Based on the pre-accumulated quantities from the training data, computes the
        per-block, per-property scales for each target. This only applies to targets
        with multiple properties. If a target is per-atom, scales are computed for each
        atomic type separately.

        :param targets_to_fit: Optional list of target names to fit. If not provided,
            all targets will be fitted.
        """
        if targets_to_fit is None:
            targets_to_fit = self.target_names

        # Only fit and store per-property scales for targets with multiple properties
        targets_to_fit = [
            target_name
            for target_name in targets_to_fit
            if target_name in self.multi_property_target_names
        ]

        # fit per-block, per-property scales
        for target_name in targets_to_fit:
            blocks = []
            for key in self.per_property_N[target_name].keys:
                N_block = self.per_property_N[target_name][key]
                Y2_block = self.per_property_Y2[target_name][key]

                N_values = N_block.values
                Y2_values = Y2_block.values

                if self.sample_kinds[target_name] == "per_structure":
                    assert len(Y2_block.samples) == 1

                # Set a sensible default in case we don't compute a scale below
                block = TensorBlock(
                    values=torch.ones_like(Y2_block.values),
                    samples=Y2_block.samples,
                    components=Y2_block.components,
                    properties=Y2_block.properties,
                )

                # Now iterate over all the atomic types in this block. For per-structure
                # targets, this is just one iteration as we do not compute
                # per-atomic-type
                for type_index in range(len(Y2_block.samples)):
                    N_values_type = N_values[type_index].unsqueeze(0)
                    Y2_values_type = Y2_values[type_index].unsqueeze(0)

                    # Compute std without Bessel's correction
                    scale_vals_type = torch.sqrt(Y2_values_type / N_values_type)

                    scale_vals_type = scale_vals_type.contiguous()
                    block.values[type_index][:] = scale_vals_type

                # Update the full scales by multiplying the per-target scales with the
                # new per-property scales
                self.scales[target_name][key].values[:] = (
                    self.per_target_scales[target_name][key].values * block.values
                )

                # If any scales are zero or NaN, set them to 1.0
                block.values[torch.isnan(block.values)] = 1.0
                self.scales[target_name][key].values[
                    torch.isnan(self.scales[target_name][key].values)
                ] = 1.0

                blocks.append(block)

            # Store the per-property scales
            self.per_property_scales[target_name] = TensorMap(
                self.per_property_Y2[target_name].keys.to(
                    device=scale_vals_type.device
                ),
                blocks,
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, TensorMap],
        remove: bool,
        use_per_target_scales: bool = False,
        use_per_property_scales: bool = False,
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Applies/removes scales to/from the outputs.

        If both ``use_per_target_scales`` and ``use_per_property_scales`` are True,
        applies/removes the full scales. If either ``use_per_target_scales`` or
        ``use_per_property_scales`` is True (but not both), applies/removes only the
        corresponding scales.

        In both cases, if the target is per-atom, scales are applied/removed separately
        for each atomic type.

        :param systems: List of systems corresponding to the outputs.
        :param outputs: Dict of names outputs to which scales should be applied/removed.
            The names (keys) should be a subset of the target names used during fitting.
            If ``use_per_property_scales`` is True and ``use_per_target_scales`` is
            False, only targets with multiple properties (i.e. > 1 block or >= 1 block
            with > 1 property) will be scaled, with single-property targets left
            unchanged.
        :param remove: If True, removes the scaling (i.e., divides by the scales). If
            False, applies the scaling (i.e., multiplies by the scales).
        :param use_per_target_scales: If True, applies/removes per-target scales.
        :param use_per_property_scales: If True, applies/removes per-block, per-property
            scales.
        :param selected_atoms: Optional labels for selected atoms. If provided, scales
            will be applied/removed only to the selected atoms, and the appropriate
            scales will be selected based on the atomic types of the selected atoms.
        :returns: A dictionary with the scaled outputs for each system.
        :raises ValueError: If no scales have been computed or if `outputs` keys contain
            unsupported keys.
        """

        device = list(outputs.values())[0][0].values.device
        dtype = list(outputs.values())[0][0].values.dtype
        self._sync_device_dtype(device, dtype)

        # Build the scaled outputs for each output
        predictions: Dict[str, TensorMap] = {}
        for output_name in outputs:
            if output_name not in self.target_names:
                # just return output as is (e.g., auxiliary outputs)
                predictions[output_name] = outputs[output_name]
                continue

            if not use_per_target_scales and use_per_property_scales:
                if output_name not in self.multi_property_target_names:
                    # per-property scales for targets with only one property are all by
                    # definition 1, so we can skip applying them
                    predictions[output_name] = outputs[output_name]
                    continue

            output_tmap = outputs[output_name]

            prediction_blocks: List[TensorBlock] = []
            for key, output_block in output_tmap.items():
                # Find the scales block and check metadata
                if use_per_target_scales and use_per_property_scales:
                    # Apply full scales
                    scales_block = self.scales[output_name].block(key)
                elif use_per_target_scales and not use_per_property_scales:
                    # Apply per-target scales
                    scales_block = self.per_target_scales[output_name].block(key)
                elif not use_per_target_scales and use_per_property_scales:
                    # Apply per-property scales
                    scales_block = self.per_property_scales[output_name].block(key)
                else:
                    raise ValueError(
                        "At least one of `use_per_target_scales` or "
                        "`use_per_property_scales` must be True."
                    )

                assert scales_block.properties == output_block.properties, (
                    f"Properties of scales block {scales_block.properties} "
                    f"do not match output block {output_block.properties} "
                    f"for key {key}."
                )

                scaled_vals = output_block.values

                # unsqueeze scales_block.values to make broadcasting work
                # (components are missing in scales_block)
                scales_block_values = scales_block.values
                for _ in range(scaled_vals.dim() - 2):
                    scales_block_values = scales_block_values.unsqueeze(1)

                if self.sample_kinds[output_name] == "per_structure":
                    # Scale the values of the output block
                    if remove:  # remove the scaler
                        scaled_vals = scaled_vals / scales_block_values[0]
                    else:  # apply the scaler
                        scaled_vals = scaled_vals * scales_block_values[0]

                    prediction_block = TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )

                    # Gradients are scaled by the same factor(s) as the values
                    if len(output_block.gradients_list()) > 0:
                        for parameter, gradient in output_block.gradients():
                            if len(gradient.gradients_list()) != 0:
                                raise NotImplementedError(
                                    "gradients of gradients are not supported"
                                )

                            if remove:  # remove the scaler
                                scaled_gradient_vals = (
                                    gradient.values / scales_block_values[0]
                                )
                            else:
                                scaled_gradient_vals = (
                                    gradient.values * scales_block_values[0]
                                )

                            prediction_block.add_gradient(
                                parameter=parameter,
                                gradient=TensorBlock(
                                    values=scaled_gradient_vals,
                                    samples=gradient.samples,
                                    components=gradient.components,
                                    properties=gradient.properties,
                                ),
                            )

                elif self.sample_kinds[output_name] == "per_atom":
                    output_block_types = torch.cat([system.types for system in systems])
                    if "atom_type" in key.names:
                        atom_type = key["atom_type"]
                        output_block_types = torch.tensor(
                            [atom_type] * torch.sum(output_block_types == atom_type)
                        )

                    if selected_atoms is not None:
                        # Scale each atomic type separately, also handling selected
                        # atoms and/or potential reordering
                        system_indices = output_block.samples.values[:, 0]
                        atom_indices = output_block.samples.values[:, 1]
                        system_lengths = torch.tensor(
                            [len(s.types) for s in systems],
                            dtype=torch.long,
                            device=device,
                        )
                        offset = torch.cat(
                            [
                                torch.zeros(1, dtype=torch.long, device=device),
                                torch.cumsum(system_lengths[:-1], dim=0),
                            ]
                        )
                        output_block_types = output_block_types[
                            offset[system_indices] + atom_indices
                        ]

                    # TODO: gradients of per-atom targets are not supported
                    if len(output_block.gradients_list()) > 0:
                        raise NotImplementedError(
                            "scaling of gradients is not implemented for per-atom "
                            f"target '{output_name}'"
                        )

                    # Scale the values of the output block
                    if remove:  # remove the scaler
                        scaled_vals = (
                            scaled_vals
                            / scales_block_values[
                                self.type_to_index[output_block_types]
                            ]
                        )
                    else:  # apply the scaler
                        scaled_vals = (
                            scaled_vals
                            * scales_block_values[
                                self.type_to_index[output_block_types]
                            ]
                        )

                    prediction_block = TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )

                else:
                    assert self.sample_kinds[output_name] == "per_atom_pair"

                    # Build a flat pair index.
                    n_samples = output_block.samples.values.shape[0]
                    if (
                        "first_atom_type" in key.names
                        and "second_atom_type" in key.names
                    ):
                        fa_type = int(key["first_atom_type"])
                        sa_type = int(key["second_atom_type"])
                        flat_idx = int(self.type_pair_to_index[fa_type, sa_type].item())
                        pair_flat_indices = torch.full(
                            (n_samples,),
                            flat_idx,
                            dtype=torch.long,
                            device=device,
                        )
                    else:
                        system_lengths = torch.tensor(
                            [len(s.types) for s in systems],
                            dtype=torch.long,
                            device=device,
                        )
                        offset = torch.cat(
                            [
                                torch.zeros(1, dtype=torch.long, device=device),
                                torch.cumsum(system_lengths[:-1], dim=0),
                            ]
                        )
                        all_types = torch.cat([s.types for s in systems]).to(device)

                        system_indices = output_block.samples.values[:, 0]
                        first_atom_col = output_block.samples.names.index("first_atom")
                        second_atom_col = output_block.samples.names.index(
                            "second_atom"
                        )
                        first_atom_indices = output_block.samples.values[
                            :, first_atom_col
                        ]
                        second_atom_indices = output_block.samples.values[
                            :, second_atom_col
                        ]

                        first_types = all_types[
                            offset[system_indices] + first_atom_indices
                        ]
                        second_types = all_types[
                            offset[system_indices] + second_atom_indices
                        ]

                        # Map each sample's (first_type, second_type) → flat pair index
                        pair_flat_indices = self.type_pair_to_index[
                            first_types, second_types
                        ]

                    if len(output_block.gradients_list()) > 0:
                        raise NotImplementedError(
                            "scaling of gradients is not implemented for atom-pair "
                            f"target '{output_name}'"
                        )

                    if remove:  # remove the scaler
                        scaled_vals = (
                            scaled_vals / scales_block_values[pair_flat_indices]
                        )
                    else:  # apply the scaler
                        scaled_vals = (
                            scaled_vals * scales_block_values[pair_flat_indices]
                        )

                    prediction_block = TensorBlock(
                        values=scaled_vals,
                        samples=output_block.samples,
                        components=output_block.components,
                        properties=output_block.properties,
                    )

                prediction_blocks.append(prediction_block)

            predictions[output_name] = TensorMap(
                outputs[output_name].keys,
                prediction_blocks,
            )

        return predictions

    def apply_onsite_scales_for_offsite(
        self,
        node_target_name: str,
        edge_target_name: str,
        mean: str = "geometric",
    ) -> None:
        """
        Override the per-target (and per-property, when applicable) scales of an
        edge (atom-pair) target using a proxy derived from the corresponding per-atom
        (node) target's scales.

        The proxy for a type pair (Z_I, Z_J) is computed as either the geometric mean
        ``sqrt(s_I * s_J)`` or the arithmetic mean ``(s_I + s_J) / 2`` of the node
        scales, controlled by the ``mean`` parameter.

        **Coupled basis** (non-``atom_type`` key names include ``o3_lambda`` and
        ``o3_sigma``): every edge block uses the invariant
        ``(o3_lambda=0, o3_sigma=1)`` node block as proxy for per-target scales.
        Per-property scales are looked up by replacing all ``_2``-suffixed property
        dimension values with the matching ``_1``-suffixed values (Z_I contribution)
        and vice-versa (Z_J contribution).

        **Uncoupled basis** (other non-``atom_type`` key structures, e.g.
        ``["l_1", "l_2", …]``): for an edge block with physics keys
        ``(k_1=A, k_2=B, …)`` the diagonal node blocks ``(k_1=A, k_2=A, …)``
        and ``(k_1=B, k_2=B, …)`` are used for Z_I and Z_J respectively.  The
        same ``_1``/``_2`` replacement logic applies for per-property scales.

        Must be called after :meth:`fit` and :meth:`fit_per_property`.

        :param node_target_name: Name of the per-atom (node) target.
        :param edge_target_name: Name of the per-atom-pair (edge) target.
        :param mean: Aggregation function for the proxy. Either ``"geometric"``
            (``sqrt(s_I * s_J)``) or ``"arithmetic"`` (``(s_I + s_J) / 2``).
        """
        if mean not in ("geometric", "arithmetic"):
            raise ValueError(
                f"offsite_proxy_mean must be 'geometric' or 'arithmetic', got '{mean}'"
            )
        if node_target_name not in self.target_names:
            raise ValueError(f"Node target '{node_target_name}' not found in scaler.")
        if edge_target_name not in self.target_names:
            raise ValueError(f"Edge target '{edge_target_name}' not found in scaler.")

        node_pt = self.per_target_scales[node_target_name]
        edge_pt = self.per_target_scales[edge_target_name]
        node_pp = self.per_property_scales[node_target_name]
        edge_pp = self.per_property_scales[edge_target_name]

        edge_key_names: List[str] = list(edge_pt.keys.names)
        node_key_names: List[str] = list(node_pt.keys.names)

        # Physics (non-atom-type) key names for the edge target
        edge_phys_names: List[str] = [
            n for n in edge_key_names if not n.endswith("atom_type")
        ]

        # Detect coupled vs. uncoupled basis
        is_coupled = "o3_lambda" in edge_phys_names and "o3_sigma" in edge_phys_names

        # For uncoupled: identify _1-suffixed physics key names
        phys_1_names: List[str] = (
            [] if is_coupled else [n for n in edge_phys_names if n.endswith("_1")]
        )

        # Whether we also update per-property scales
        do_per_property = edge_target_name in self.multi_property_target_names

        # Identify _1/_2 property dimension pairs (for the per-property proxy)
        prop_pair_bases: List[str] = []
        prop_cols_1: Dict[str, int] = {}
        prop_cols_2: Dict[str, int] = {}
        if do_per_property and len(edge_pt.keys) > 0:
            first_block = edge_pt.block_by_id(0)
            edge_prop_names: List[str] = list(first_block.properties.names)
            for pname in edge_prop_names:
                if pname.endswith("_1"):
                    base = pname[:-2]
                    if base + "_2" in edge_prop_names:
                        prop_pair_bases.append(base)
                        prop_cols_1[base] = edge_prop_names.index(pname)
                        prop_cols_2[base] = edge_prop_names.index(base + "_2")

        # Build new per-target and per-property scale blocks for the edge target.
        # The outer loop is over blocks (one per physics key); the inner loop is
        # over sample rows (type pairs).  This handles both sparsified targets
        # (where first_atom_type / second_atom_type appear in the block keys) and
        # densified targets (where they appear only in the sample labels).
        new_pt_blocks: List[TensorBlock] = []
        new_pp_blocks: List[TensorBlock] = []

        for edge_key in edge_pt.keys:
            edge_pt_block = edge_pt.block(edge_key)
            edge_pp_block = edge_pp.block(edge_key)

            # Physics key values from this edge block (no atom-type dims)
            edge_phys_vals: Dict[str, int] = {
                n: int(edge_key[n]) for n in edge_phys_names
            }

            # For uncoupled: compute the diagonal physics-key substitutions once
            # per edge block (they depend on the physics key, not on the type pair).
            if not is_coupled:
                node_phys_for_I: Dict[str, int] = dict(edge_phys_vals)
                node_phys_for_J: Dict[str, int] = dict(edge_phys_vals)
                for n1 in phys_1_names:
                    n2 = n1[:-2] + "_2"
                    if n2 in edge_phys_vals:
                        node_phys_for_I[n2] = edge_phys_vals[n1]  # _2 ← _1
                        node_phys_for_J[n1] = edge_phys_vals[n2]  # _1 ← _2

            # For the per-property proxy, the property index mapping is the same
            # for all type pairs, so precompute it once per edge block.
            if do_per_property:
                edge_pp_props = edge_pp_block.properties
                n_edge_props = len(edge_pp_props)
                edge_prop_vals_tensor = edge_pp_props.values  # (n_props, n_dims)

                # For each edge property build the "diagonal" node property
                # indices: Z_I uses (_1 repeated), Z_J uses (_2 repeated).
                node_prop_I_list: List[List[int]] = []
                node_prop_J_list: List[List[int]] = []
                for p_idx in range(n_edge_props):
                    prop_row: List[int] = edge_prop_vals_tensor[p_idx].tolist()
                    npi: List[int] = list(prop_row)
                    npj: List[int] = list(prop_row)
                    for base in prop_pair_bases:
                        npi[prop_cols_2[base]] = prop_row[prop_cols_1[base]]
                        npj[prop_cols_1[base]] = prop_row[prop_cols_2[base]]
                    node_prop_I_list.append(npi)
                    node_prop_J_list.append(npj)

            # Start with clones of the existing values; we will overwrite
            # individual rows as we iterate over type pairs.
            new_pt_vals = edge_pt_block.values.clone()
            new_pp_vals = edge_pp_block.values.clone() if do_per_property else None

            # ---- Inner loop: one update per type pair (sample row) ----
            for sample_idx, sample_entry in enumerate(edge_pt_block.samples):
                Z_I = int(sample_entry["first_atomic_type"])
                Z_J = int(sample_entry["second_atomic_type"])
                i_I = int(self.type_to_index[Z_I].item())
                i_J = int(self.type_to_index[Z_J].item())

                # Determine node block key for Z_I and Z_J
                if is_coupled:
                    node_key_vals_I: List[int] = []
                    node_key_vals_J: List[int] = []
                    for n in node_key_names:
                        if n == "o3_lambda":
                            node_key_vals_I.append(0)
                            node_key_vals_J.append(0)
                        elif n == "o3_sigma":
                            node_key_vals_I.append(1)
                            node_key_vals_J.append(1)
                        elif n == "atom_type":
                            node_key_vals_I.append(Z_I)
                            node_key_vals_J.append(Z_J)
                        else:
                            node_key_vals_I.append(0)
                            node_key_vals_J.append(0)
                else:
                    node_key_vals_I = [
                        Z_I
                        if n == "atom_type"
                        else node_phys_for_I.get(n, edge_phys_vals.get(n, 1))
                        for n in node_key_names
                    ]
                    node_key_vals_J = [
                        Z_J
                        if n == "atom_type"
                        else node_phys_for_J.get(n, edge_phys_vals.get(n, 1))
                        for n in node_key_names
                    ]

                pos_I = node_pt.keys.position(node_key_vals_I)
                pos_J = node_pt.keys.position(node_key_vals_J)

                if pos_I is None or pos_J is None:
                    key_info = dict(
                        zip(edge_key_names, edge_key.values.tolist(), strict=True)
                    )
                    logging.warning(
                        "onsite_scales_for_offsite: could not find node block "
                        f"for (Z_I={Z_I}, Z_J={Z_J}) in edge block {key_info};"
                        " leaving scales unchanged for this type pair."
                    )
                    continue  # leave this row unchanged (clone keeps old value)

                node_block_I_pt = node_pt.block_by_id(pos_I)
                node_block_J_pt = node_pt.block_by_id(pos_J)

                # Per-target proxy (uniform across all properties in the block)
                s_I = node_block_I_pt.values[i_I, 0]
                s_J = node_block_J_pt.values[i_J, 0]
                if mean == "geometric":
                    new_pt_vals[sample_idx, :] = torch.sqrt(s_I.abs() * s_J.abs())
                else:
                    new_pt_vals[sample_idx, :] = (s_I.abs() + s_J.abs()) / 2

                # Per-property proxy
                if do_per_property and new_pp_vals is not None:
                    node_block_I_pp = node_pp.block_by_id(pos_I)
                    node_block_J_pp = node_pp.block_by_id(pos_J)

                    pp_proxy = torch.ones(
                        n_edge_props,
                        dtype=edge_pp_block.values.dtype,
                        device=edge_pp_block.values.device,
                    )
                    if len(prop_pair_bases) > 0:
                        for p_idx in range(n_edge_props):
                            idx_I_pp = node_block_I_pp.properties.position(
                                node_prop_I_list[p_idx]
                            )
                            idx_J_pp = node_block_J_pp.properties.position(
                                node_prop_J_list[p_idx]
                            )
                            if idx_I_pp is not None and idx_J_pp is not None:
                                s_pp_I = node_block_I_pp.values[i_I, idx_I_pp]
                                s_pp_J = node_block_J_pp.values[i_J, idx_J_pp]
                                if mean == "geometric":
                                    pp_proxy[p_idx] = torch.sqrt(
                                        s_pp_I.abs() * s_pp_J.abs()
                                    )
                                else:
                                    pp_proxy[p_idx] = (s_pp_I.abs() + s_pp_J.abs()) / 2
                            # else: leave as 1.0

                    new_pp_vals[sample_idx, :] = pp_proxy

            new_pt_blocks.append(
                TensorBlock(
                    values=new_pt_vals,
                    samples=edge_pt_block.samples,
                    components=edge_pt_block.components,
                    properties=edge_pt_block.properties,
                )
            )
            if do_per_property and new_pp_vals is not None:
                new_pp_blocks.append(
                    TensorBlock(
                        values=new_pp_vals,
                        samples=edge_pp_block.samples,
                        components=edge_pp_block.components,
                        properties=edge_pp_block.properties,
                    )
                )

        # ---- Update TensorMaps in-place ----
        self.per_target_scales[edge_target_name] = TensorMap(
            edge_pt.keys, new_pt_blocks
        )
        if do_per_property:
            self.per_property_scales[edge_target_name] = TensorMap(
                edge_pp.keys, new_pp_blocks
            )

        # Recompute full scales = per_target * per_property
        current_pp = self.per_property_scales[edge_target_name]
        new_full_blocks: List[TensorBlock] = []
        for key in self.per_target_scales[edge_target_name].keys:
            pt_b = self.per_target_scales[edge_target_name].block(key)
            pp_b = current_pp.block(key)
            full_vals = torch.nan_to_num(pt_b.values * pp_b.values, nan=1.0)
            new_full_blocks.append(
                TensorBlock(
                    values=full_vals,
                    samples=pt_b.samples,
                    components=pt_b.components,
                    properties=pt_b.properties,
                )
            )
        self.scales[edge_target_name] = TensorMap(
            self.per_target_scales[edge_target_name].keys, new_full_blocks
        )

    def _set_fixed_weights(
        self,
        target_name: str,
        weights: Union[float, Dict[int, float], Dict[int, Dict[int, float]]],
    ) -> None:
        """
        Apply fixed weights to the scales of a given target.

        :param target_name: Name of the target to which fixed weights should be applied.
        :param weights: Either a single float value to be applied to all rows, a dict
            mapping atomic type (int) to weight (float) for per-atom targets, or a
            nested dict ``{Z1: {Z2: weight}}`` for per-atom-pair targets.
        """
        sample_kind = self.sample_kinds[target_name]

        # Fixed weights set the per-target scale; fit_per_property() then multiplies
        # by per-property scales to produce the full scales.  There is therefore no
        # restriction on the number of blocks or properties: the same per-type (or
        # per-type-pair) scalar is broadcast across all properties, and per-property
        # variation is handled normally afterwards.

        if sample_kind == "per_atom_pair":
            atomic_types = self.atomic_types.tolist()
            all_pairs = list(itertools.product(atomic_types, repeat=2))

            if isinstance(weights, dict):
                # Nested {Z1: {Z2: weight}} form.
                flat: Dict[tuple, float] = {}
                for at_i, inner in weights.items():
                    if not isinstance(inner, dict):
                        raise ValueError(
                            f"Fixed scaling weights for per-atom-pair target "
                            f"'{target_name}' must be a float or a nested dict "
                            f"{{Z1: {{Z2: weight}}}}, got a flat dict instead."
                        )
                    for at_j, w in inner.items():
                        flat[(int(at_i), int(at_j))] = float(w)
                missing_pairs = set(all_pairs) - set(flat.keys())
                if missing_pairs:
                    raise ValueError(
                        f"Fixed scaling weights for per-atom-pair target "
                        f"'{target_name}' are missing the following type pairs: "
                        f"{missing_pairs}"
                    )
                pair_weights = flat
            elif isinstance(weights, float):
                logging.info(
                    "Fixed scaling weights provided as a single float for "
                    f"per-atom-pair target '{target_name}'. The same weight will "
                    "be applied to all type pairs."
                )
                pair_weights = {(at_i, at_j): weights for at_i, at_j in all_pairs}
            else:
                raise ValueError(
                    f"Fixed scaling weights for per-atom-pair target '{target_name}' "
                    f"must be a float or a nested dict {{Z1: {{Z2: weight}}}}."
                )

            blocks = []
            for key in self.Y2[target_name].keys:
                Y2_block = self.Y2[target_name][key]
                block = TensorBlock(
                    values=torch.empty_like(Y2_block.values),
                    samples=Y2_block.samples,
                    components=Y2_block.components,
                    properties=Y2_block.properties,
                )
                for at_i, at_j in all_pairs:
                    idx = self.type_pair_to_index[at_i, at_j].item()
                    # Broadcast the scalar across all properties.
                    block.values[idx, :] = pair_weights[(at_i, at_j)]
                blocks.append(block)

            self.scales[target_name] = TensorMap(
                self.Y2[target_name].keys.to(device=blocks[0].values.device),
                blocks,
            )
            self.per_target_scales[target_name] = self.scales[target_name].copy()

        else:
            # per_structure / per_atom: validate the weight format up front, then
            # apply the same per-type scalars to every block (broadcast across all
            # properties so fit_per_property can multiply in per-property variation).
            if isinstance(weights, dict):
                if sample_kind == "per_structure":
                    raise ValueError(
                        "Fixed scaling weights as a dict are not supported for "
                        f"per-structure target '{target_name}'"
                    )
                for atomic_type in self.atomic_types.tolist():
                    if int(atomic_type) not in weights:
                        raise ValueError(
                            f"Atomic type {atomic_type} is missing from the fixed "
                            f"scaling weights for target '{target_name}'"
                        )
            elif not isinstance(weights, float):
                raise ValueError(
                    f"Fixed scaling weights for '{target_name}' must be either a "
                    "float or a dict of int to float."
                )
            else:
                if sample_kind == "per_atom":
                    logging.info(
                        "Fixed scaling weights provided as a single float for "
                        f"per-atom target '{target_name}'. The same weight will be "
                        "applied to all atomic types."
                    )

            blocks = []
            for key in self.Y2[target_name].keys:
                Y2_block = self.Y2[target_name][key]
                block = TensorBlock(
                    values=torch.empty_like(Y2_block.values),
                    samples=Y2_block.samples,
                    components=Y2_block.components,
                    properties=Y2_block.properties,
                )
                if isinstance(weights, dict):
                    for atom_type, weight in weights.items():
                        # Broadcast the scalar across all properties.
                        block.values[self.type_to_index[atom_type], :] = weight
                else:
                    block.values[:] = weights
                blocks.append(block)

            self.scales[target_name] = TensorMap(
                self.Y2[target_name].keys.to(device=blocks[0].values.device),
                blocks,
            )
            self.per_target_scales[target_name] = self.scales[target_name].copy()

    def _sync_device_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        # manually move the TensorMap dicts:

        self.atomic_types = self.atomic_types.to(device=device)
        self.type_to_index = self.type_to_index.to(device=device)
        self.type_pair_to_index = self.type_pair_to_index.to(device=device)
        self.N = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.N.items()
        }
        self.Y2 = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.Y2.items()
        }
        self.scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.scales.items()
        }
        self.per_target_scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_target_scales.items()
        }
        self.per_property_N = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_property_N.items()
        }
        self.per_property_Y2 = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_property_Y2.items()
        }
        self.per_property_scales = {
            target_name: tm.to(device=device, dtype=dtype)
            for target_name, tm in self.per_property_scales.items()
        }
