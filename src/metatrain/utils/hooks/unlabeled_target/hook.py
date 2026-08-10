from typing import Optional

from metatensor.torch import Labels, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info

from ..abc import HookInterface
from .documentation import Hypers


class UnlabeledTarget(HookInterface[Hypers]):
    """
    Passes the inputs to the outputs without any modification.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        self.hypers = hypers

        self._input_target_infos = {
            name: get_generic_target_info(name, target)
            for name, target in hypers["targets"].items()
        }

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_inputs(self) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :return: A list of requested input names.
        """
        return {
            in_name: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                sample_kind=target_info.sample_kind,
            )
            for in_name, target_info in self._input_target_infos.items()
        }

    def supported_outputs(self) -> dict[str, ModelOutput]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return {}

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        return inputs
