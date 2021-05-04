from .base import Segmentation_MP, log
from torch_points3d.modules.RandLANet import *


class RandLANetSeg(Segmentation_MP):
    # Segmentation_MP is UnetBasedModel
    """ Unet base implementation of RandLANet
    """
    def __init__(self, option, model_type, dataset, modules):

        Segmentation_MP.__init__(self, option, model_type, dataset, modules)

        self._num_classes = dataset.num_classes
        self._weight_classes = dataset.weight_classes
        self._use_category = getattr(option, "use_category", False)

        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError("Dataset does not specify needed "
                                 "class_to_segments property")
            self._num_categories = len(dataset.class_to_segments.keys())
            log.info(f"Using category information for "
                     f"the predictions with ${self._num_categories}")
        else:
            self._num_categories = 0
            log.info(f"Category information is not going to be used")

        # Name specs.
        self.loss_names = ["loss_seg"]
        self.visual_names = ["data_visual"]

        self.input = None
        self.labels = None
        self.batch_idx = None
        self.category = None

        self.loss_seg = None
        self.data_visual = None

        ...

    def set_input(self, data, device):
        ...

    def forward(self, *args, **kwargs) -> Any:
        ...

    def backward(self) -> Any:
        self.loss_seg.backward()
