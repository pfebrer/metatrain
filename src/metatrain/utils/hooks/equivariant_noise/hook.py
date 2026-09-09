from typing import Optional

import torch

from metatensor.torch import Labels, TensorMap, TensorBlock
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo

from ..abc import HookInterface
from .documentation import Hypers


class Noise(HookInterface[Hypers]):
    """

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        self.hypers = hypers

        # Get the information about the output targets from the dataset info
        out_names = hypers["outputs"]
        if isinstance(out_names, str):
            out_names = [out_names]
        if out_names is None:
            raise ValueError("Noise hook requires at least one output target")

        # Get the information for the input targets.
        input_names = hypers.get("inputs")
        if isinstance(input_names, str):
            input_names = [input_names]
        elif input_names is None:
            input_names = [
                f"mtt::aux::noise::{out_name.replace('mtt::', '')}"
                for out_name in out_names
            ]

        if len(input_names) != len(out_names):
            raise ValueError(
                f"Noise hook expects the same number of input and output "
                f"targets, but got {len(input_names)} inputs and "
                f"{len(out_names)} outputs"
            )

        # Build the target infos that the hook will request.
        self.out_targets = {}
        self._input_target_infos = {}
        targets = dataset_info.targets
        for in_name, out_name in zip(input_names, out_names, strict=True):
            if out_name in dataset_info.targets:
                target = dataset_info.targets[out_name]

                input_target_info = TargetInfo(
                    layout=TensorMap(
                        keys=Labels(
                            names=["o3_lambda", "o3_sigma"],
                            values=torch.tensor(
                                [[0, 1], [1, 1]],
                                dtype=torch.int64
                            )
                        ),
                        blocks=[
                            TensorBlock(
                                samples=target.layout.block(0).samples,
                                components=[
                                    Labels(
                                        ["o3_mu"],
                                        torch.tensor([[0]], dtype=torch.int64)
                                    )
                                ],
                                properties=target.layout.block(0).properties,
                                values=torch.empty((0, 1, target.layout.block(0).properties.values.shape[0]), dtype=torch.float32)
                            ),
                            TensorBlock(
                                samples=target.layout.block(0).samples,
                                components=[
                                    Labels(
                                        ["o3_mu"],
                                        torch.tensor([[-1], [0], [1]], dtype=torch.int64)
                                    )
                                ],
                                properties=target.layout.block(0).properties,
                                values=torch.empty((0, 3, target.layout.block(0).properties.values.shape[0]), dtype=torch.float32)
                            )
                        ]
                    ),
                )

                self._input_target_infos[in_name] = input_target_info
                self.out_targets[out_name] = target     
            else:
                raise ValueError(
                    f"Noise hook expects output target '{out_name}' to be present "
                    f"in the dataset info, but it is not found. Available targets: "
                    f"{list(targets.keys())}"
                )

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_hook_inputs(self, outputs: dict[str, ModelOutput]) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :param outputs: Dictionary of requested outputs. These can contain outputs that
            are handled by other hooks or the main model. In that case, the hook
            ignores those outputs.
        :return: A list of requested input names.
        """
        req_inputs: dict[str, ModelOutput] = {}
        for out_name, (in_name, target_info) in zip(
            self.out_targets, self._input_target_infos.items(), strict=True
        ):
            if out_name in outputs:
                req_inputs[in_name] = ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind=target_info.sample_kind,
                )

        return req_inputs

    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return {
            out_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind=target_info.sample_kind,
            )
            for out_name, target_info in self.out_targets.items()
        }

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        return_dict: dict[str, TensorMap] = {}
        for in_name, out_name in zip(
            self._input_target_infos, self.out_targets, strict=True
        ):
            if out_name in outputs:
                l1_values = inputs[in_name].block(1).values
                print("Mean noise level:", abs(l1_values).mean())
                return_dict[out_name] = TensorMap(
                    keys=self.out_targets[out_name].layout.keys,
                    blocks=[
                        TensorBlock(
                            values=inputs[in_name].block(0).values.sum(dim=1) + l1_values.sum(dim=1),
                            samples=inputs[in_name].block(0).samples,
                            components=self.out_targets[out_name].layout.block(0).components,
                            properties=self.out_targets[out_name].layout.block(0).properties,
                        )
                    ],
                )

        return return_dict
