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
        self.noise_max_ell = hypers.get("max_ell", 1)

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
                                [[ell, 1] for ell in range(self.noise_max_ell + 1)],
                                dtype=torch.int64
                            )
                        ),
                        blocks=[
                            TensorBlock(
                                samples=target.layout.block(0).samples,
                                components=[
                                    Labels(
                                        ["o3_mu"],
                                        torch.arange(-ell, ell + 1, dtype=torch.int64).reshape(-1, 1)
                                    )
                                ],
                                properties=target.layout.block(0).properties,
                                values=torch.empty((0, 2 * ell + 1, target.layout.block(0).properties.values.shape[0]), dtype=torch.float32)
                            )
                            for ell in range(self.noise_max_ell + 1)
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
                layout = self.out_targets[out_name].layout.to(systems[0].positions.device)
                contribs = [
                    inputs[in_name].block(ell).values.sum(dim=1)
                    for ell in range(self.noise_max_ell + 1)
                ]
                print("Mean contribs:", [round(abs(values).mean().item(), 3) for values in contribs])
                return_dict[out_name] = TensorMap(
                    keys=layout.keys,
                    blocks=[
                        TensorBlock(
                            values=torch.stack(contribs, dim=1).sum(dim=1),
                            samples=inputs[in_name].block(0).samples,
                            components=layout.block(0).components,
                            properties=layout.block(0).properties,
                        )
                    ],
                )

        return return_dict
