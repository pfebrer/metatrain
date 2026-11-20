from typing import Any, Optional

import torch
from e3nn import o3
from metatensor.torch import Labels, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System
from sphericart.torch import SphericalHarmonics

from metatrain.experimental.mace.utils.mts import (
    e3nn_to_tensormap,
    get_e3nn_mts_layout,
    target_info_to_e3nn_irreps,
)
from metatrain.experimental.mace.utils.structures import create_batch
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.atom_pair_helpers import get_pair_sample_labels
from metatrain.utils.data.atomic_basis_helpers import get_per_atom_sample_labels

from ..abc import ExtenderInterface
from .documentation import Hypers
from .utils.edge_embeddings import RadialEmbeddingBlock
from .utils.edge_operations import MACEEdgeState
from .utils.structures import get_edge_vectors_and_lengths


class Graph2Mat(ExtenderInterface[Hypers]):
    """
    Passes the inputs to the outputs without any modification.

    :param hypers: A dictionary with the extender's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        # Get the information about the output targets from the dataset info
        out_names = hypers["outputs"]
        if isinstance(out_names, str):
            out_names = [out_names]
        if out_names is None:
            raise ValueError("Graph2mat extender requires at least one output target")

        # Modules to embed edges
        self.spherical_harmonics = SphericalHarmonics(
            self.hypers["edge_lmax"],
        )
        sh_irreps = o3.Irreps.spherical_harmonics(self.hypers["edge_lmax"])
        n_basis = 10
        self.radial_embeddings = RadialEmbeddingBlock(
            r_max=self.hypers["cutoff"],
            num_bessel=n_basis,
            num_polynomial_cutoff=10,
        )
        radial_irreps = o3.Irreps(f"{n_basis}x0e")

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        # Build the target infos that the extender will request.
        self.out_targets = {}
        self._input_target_infos = {}
        operations = []
        targets = dataset_info.targets
        for out_name in out_names:
            target = targets[out_name]

            self.out_targets[out_name] = target
            node_irreps = self.hypers["node_hidden_irreps"]
            layout = get_e3nn_mts_layout(
                o3.Irreps(node_irreps),
                "mtt::aux::graph2mat::node_states",
                {
                    "type": "spherical",
                    "sample_kind": "atom",
                    "properties_name": "feature",
                },
            )
            layout = TensorMap(
                keys=Labels(
                    names=[name for name in layout.keys.names if name != "i_irrep"],
                    values=layout.keys.values[:, :-1],
                ),
                blocks=[block.copy() for block in layout.blocks()],
            )
            self._input_target_infos["mtt::aux::graph2mat::node_states"] = TargetInfo(
                layout
            )
            edge_irreps = target_info_to_e3nn_irreps(target)
            edge_message_irreps = self.hypers["edge_message_irreps"] or edge_irreps
            operations.append(
                MACEEdgeState(
                    irreps={
                        "node_feats_irreps": o3.Irreps(node_irreps),
                        "edge_attrs_irreps": sh_irreps,
                        "edge_feats_irreps": radial_irreps,
                        "message_irreps": o3.Irreps(edge_message_irreps),
                        "edge_targets_irreps": o3.Irreps(edge_irreps),
                    }
                )
            )

        self.operations = torch.nn.ModuleList(operations)

    def requested_extender_inputs(
        self, outputs: Optional[dict[str, ModelOutput]] = None
    ) -> dict[str, TargetInfo]:
        if outputs is None:
            return self._input_target_infos

        req_inputs: dict[str, TargetInfo] = {}
        for out_name, (in_name, target_info) in zip(
            self.out_targets, self._input_target_infos.items(), strict=True
        ):
            if out_name in outputs:
                req_inputs[in_name] = target_info

        return req_inputs

    def requested_neighbor_lists(
        self,
    ) -> list[NeighborListOptions]:
        return [self.requested_nl]

    def supported_outputs(self) -> dict[str, TargetInfo]:
        """
        Returns the supported outputs for the extender.

        :return: A dictionary containing the supported outputs
          for the extender.
        """
        return self.out_targets

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:

        device = systems[0].positions.device

        if selected_atoms is not None:
            raise NotImplementedError(
                "Graph2mat extender does not support selected_atoms yet."
            )

        batch_data = create_batch(
            systems=systems,
            neighbor_list_options=self.requested_nl,
            atomic_types_to_species_index=torch.zeros(
                max(self.dataset_info.atomic_types) + 1,
                device=device,
                dtype=torch.int64,
            ),
            n_types=len(self.dataset_info.atomic_types),
            head_index=0,
        )

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch_data["positions"],
            edge_index=batch_data["edge_index"],
            shifts=batch_data["shifts"],
        )

        per_atom_samples = get_per_atom_sample_labels(
            systems,
            system_ids=torch.arange(len(systems), device=device, dtype=torch.int64),
        )
        pair_samples = get_pair_sample_labels(
            sample_labels=per_atom_samples,
            centers=batch_data["edge_index"][0],
            neighbors=batch_data["edge_index"][1],
            cell_shifts=batch_data["unit_shifts"].T,
        )

        vectors_spherical = self.spherical_harmonics(vectors)
        radial_embeddings = self.radial_embeddings(
            lengths,
            batch_data["node_attrs"],
            batch_data["edge_index"],
            torch.tensor(self.dataset_info.atomic_types, device=device),
        )

        batch_data["edge_attrs"] = vectors_spherical
        batch_data["edge_feats"] = radial_embeddings

        in_names = list(self._input_target_infos.keys())
        out_names = list(self.out_targets.keys())
        return_dict: dict[str, TensorMap] = {}
        for i, operation in enumerate(self.operations):
            in_name = in_names[i]
            out_name = out_names[i]
            if out_name in outputs:
                node_states = inputs[in_name]
                n_samples = node_states.block(0).samples.values.shape[0]
                values = torch.concat(
                    [
                        block.values.reshape(n_samples, -1)
                        for block in node_states.blocks()
                    ],
                    dim=1,
                )
                edge_values = operation(batch_data, values)

                return_dict[out_name] = e3nn_to_tensormap(
                    edge_values,
                    pair_samples,
                    self.out_targets[out_name].layout.to(device),
                )

        return return_dict

    def get_checkpoint(self, best_state_dict: Optional[dict[str, Any]] = None) -> dict:
        """
        Get the checkpoint of the extender.

        :return: A dictionary containing the state of the extender.
        """
        state_dict = self.state_dict()
        return {
            "extender_name": "graph2mat",
            "extender_ckpt_version": self.__checkpoint_version__,
            "extender_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": state_dict,
        }

    @classmethod
    def load_checkpoint(
        cls, checkpoint: dict[str, Any], context: str
    ) -> "Graph2MatHook":
        """
        Load the checkpoint of the extender.

        :param checkpoint: A dictionary containing the state of the extender.
        """
        if checkpoint["extender_ckpt_version"] != cls.__checkpoint_version__:
            raise ValueError(
                f"Invalid checkpoint version for the extender: "
                f"expected {cls.__checkpoint_version__}, got "
                f"{checkpoint['extender_ckpt_version']}"
            )

        extender = cls(
            hypers=checkpoint["extender_data"]["hypers"],
            dataset_info=checkpoint["extender_data"]["dataset_info"],
        )

        extender.load_state_dict(checkpoint["model_state_dict"])
        return extender
