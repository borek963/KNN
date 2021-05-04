import copy

from .base import UnetBasedModel, log
from torch_points3d.modules.RandLANet import *
from torch_points3d.datasets.segmentation import IGNORE_LABEL


from torch_points3d.core.common_modules.dense_modules import Conv1D


class RandLANetSeg(UnetBasedModel):
    # Segmentation_MP is UnetBasedModel
    """ Unet base implementation of RandLANet
    """
    def __init__(self, option, model_type, dataset, modules):

        UnetBasedModel.__init__(self, option, model_type, dataset, modules)

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

        # REEEEEEEEEEEEEAAAAAAAAAAAAAAALLLLLYYYYYY NOT SURE ABOUT THIS
        last_mlp_opt = copy.deepcopy(option.mlp_cls)
        self.FC_layer = Seq()
        last_mlp_opt.nn[0] += self._num_categories
        # Adding layers specified in pointnet2.yaml - mlp_cls
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(MLP(channels=[last_mlp_opt.nn[i - 1],
                                               last_mlp_opt.nn[i]]))
        # Specify dropout of last FC layer (mlp_cls)
        if last_mlp_opt.dropout:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt.dropout))
        # last FC layer - num of classes = out dim
        self.FC_layer.append(MLP([last_mlp_opt.nn[-1], self._num_classes]))

    def set_input(self, data, device):
        # if len(data.pos.shape) < 3:
        #     raise ValueError(f"Position data shape should be 3, {len(data.pos.shape)} - given!")

        data = data.to(device)

        x = data.x.transpose(1, 2).contiguous() if (data.x is not None) else None
        self.input = Data(x=x, pos=data.pos)
        self.labels = torch.flatten(data.y).long() if (data.y is not None) else None  # [B * N]
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        self.category = data.category if self._use_category else ...

    def forward(self, *args, **kwargs) -> Any:
        data = self.model(self.input)
        last_feature = data.x

        self.output = self.FC_layer(last_feature).transpose(1, 2).contiguous().view((-1, self._num_classes))

        if self._weight_classes is not None:
            self._weight_classes = self._weight_classes.to(self.output.device)

        # Compute loss Cross Entropy
        if self.labels is not None:
            self.loss_seg = F.cross_entropy(
                self.output, self.labels, weight=self._weight_classes, ignore_index=IGNORE_LABEL
            )
        self.data_visual = self.input
        self.data_visual.y = torch.reshape(self.labels, data.pos.shape[0:2])
        self.data_visual.pred = torch.max(self.output, -1)[1].reshape(data.pos.shape[0:2])

        return self.output


    def backward(self) -> Any:
        self.loss_seg.backward()
