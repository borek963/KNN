
import logging
import copy

from torch_points3d.modules.pointnet2 import *
from torch_points3d.models.base_architectures import UnetBasedModel
from torch_points3d.core.common_modules.dense_modules import Conv1D
from torch_points3d.core.common_modules.base_modules import Seq
from torch_points3d.datasets.segmentation import IGNORE_LABEL

log = logging.getLogger(__name__)


class MyPointNet2(UnetBasedModel):

    def __init__(self, option, model_type, dataset, modules):

        # Pointnet++ is UnetBased model, call init method of unet model
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

        # ---------------------------------------------------
        # Specification of last MLP based on
        # mlp_cls opt in "mypointnet2" in "pointnet2.yaml"
        last_mlp_opt = copy.deepcopy(option.mlp_cls)

        # A sequential container. Modules will be added to
        # it in the order they are passed in the constructor
        # (Torch classic method)
        self.FC_layer = Seq()
        last_mlp_opt.nn[0] += self._num_categories

        # Adding layers specified in pointnet2.yaml - mlp_cls
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(Conv1D(last_mlp_opt.nn[i - 1],
                                        last_mlp_opt.nn[i], bn=True, bias=False))

        # Specify dropout of last FC layer (mlp_cls)
        if last_mlp_opt.dropout:
            self.FC_layer.append(torch.nn.Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.append(Conv1D(last_mlp_opt.nn[-1], self._num_classes,
                                    activation=None, bias=True, bn=False))
        # -------------------------------------------------------------------

        # Name specs.
        self.loss_names = ["loss_seg"]
        self.visual_names = ["data_visual"]

        self.input = None
        self.labels = None
        self.batch_idx = None
        self.category = None

        self.loss_seg = None
        self.data_visual = None

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        Sets:
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        if len(data.pos.shape) != 3:
            raise ValueError(f"Position data shape should be 3, {len(data.pos.shape)} - given!")
        data = data.to(device)

        x = data.x.transpose(1, 2).contiguous() if (data.x is not None) else None
        self.input = Data(x=x, pos=data.pos)
        self.labels = torch.flatten(data.y).long() if (data.y is not None) else None  # [B * N]
        self.batch_idx = torch.arange(0, data.pos.shape[0]).view(-1, 1).repeat(1, data.pos.shape[1]).view(-1)
        self.category = data.category if self._use_category else ...

    def forward(self, *args, **kwargs):
        r"""
            Forward pass of the network
            self.input:
                x -- Features [B, C, N]
                pos -- Points [B, N, 3]
        """
        data = self.model(self.input)
        last_feature = data.x

        if self._use_category:
            # splitting categorical data to more columns
            cat_one_hot = F.one_hot(self.category, self._num_categories).float().transpose(1, 2)
            # concatenates given tensors (dim over which)
            last_feature = torch.cat((last_feature, cat_one_hot), dim=1)

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

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()
