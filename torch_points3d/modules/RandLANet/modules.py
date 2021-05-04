import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from torch_points3d.core.spatial_ops import *
from torch_points3d.core.base_conv.message_passing import *


class RandlaKernel(MessagePassing):
    # Initialized by RandLaConv
    """
        Implements both the Local Spatial Encoding and Attentive Pooling blocks from
        RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/pdf/1911.11236
    """
    def __init__(self, point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        MessagePassing.__init__(self, aggr="add")

        self.point_pos_nn = MLP(point_pos_nn)
        self.attention_nn = MLP(attention_nn)
        self.global_nn = MLP(global_nn)  # == down_conv_nn from .yaml file

    def forward(self, x, pos, edge_index):
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    def message(self, x_j, pos_i, pos_j):
        if x_j is None:
            x_j = pos_j

        # This is in paper - Figure 3. PART of LocSE block (Local Spatial Encoding)
        # compute relative position encoding
        # in paper equation (1)
        vij = pos_i - pos_j
        dij = torch.norm(vij, dim=1).unsqueeze(1)
        relPointPos = torch.cat([pos_i, pos_j, vij, dij], dim=1)
        rij = self.point_pos_nn(relPointPos)  # (r_i^k)

        # This is in paper - Figure 3. PART of LocSE block (Local Spatial Encoding)
        # concatenate position encoding with feature vector for feature augmentation
        fij_hat = torch.cat([x_j, rij], dim=1)

        # This is in paper - Figure 3. Attentive Pooling block
        # attentive pooling
        # in paper equation (2) and (3)
        g_fij = self.attention_nn(fij_hat)
        s_ij = F.softmax(g_fij, -1)
        msg = s_ij * fij_hat

        return msg

    def update(self, aggr_out):
        return self.global_nn(aggr_out)


class RandlaConv(BaseConvolutionDown):
    # In pointnet++ - this is defined in "message_passing.py"
    # (PointNet++ uses Multiscale version of base class)
    # -------------------------------------------------------
    def __init__(self, ratio=None, k=None, *args, **kwargs):
        # Random sampling implementation and find K nearest neighbors
        # In PointNet++ FPSSampler and MultiscaleRadiusNeighbourFinder are used.
        super(RandlaConv, self).__init__(
            RandomSampler(ratio),
            KNNNeighbourFinder(k),
            *args,
            **kwargs
        )

        # kwargs = key-worded list of args, never used them before, definitely WILL
        if kwargs.get("index") == 0 and kwargs.get("nb_feature") is not None:
            kwargs["point_pos_nn"][-1] = kwargs.get("nb_feature")
            kwargs["attention_nn"][0] = kwargs["attention_nn"][-1] = kwargs.get("nb_feature") * 2
            kwargs["down_conv_nn"][0] = kwargs.get("nb_feature") * 2

        # PointNet++ uses PointConv from: "/torch_geometric/nn/conv/point_conv.py"
        self._conv = RandlaKernel(*args, global_nn=kwargs["down_conv_nn"], **kwargs)
        # Probably not needed TODO
        self._ratio = ratio
        self._k = k

    # Return instance of conv layer defined in __init__
    def conv(self, x, pos, edge_index, batch):
        return self._conv(x, pos, edge_index)


class DilatedResidualBlock(BaseResnetBlock):
    # This is defined for PointNet++ in dense.py
    # (in GitHub implementation of RandLaNet: "dilated_res_block")
    # ------------------------------------------
    def __init__(
        self,
        indim,
        outdim,
        ratio1,
        ratio2,
        point_pos_nn1,
        point_pos_nn2,
        attention_nn1,
        attention_nn2,
        global_nn1,
        global_nn2,
        *args,
        **kwargs
    ):
        if kwargs.get("index") == 0 and kwargs.get("nb_feature") is not None:
            indim = kwargs.get("nb_feature")
        super(DilatedResidualBlock, self).__init__(indim, outdim, outdim)
        self.conv1 = RandlaConv(
            ratio1,
            16,  # KNN - k value
            point_pos_nn=point_pos_nn1,
            attention_nn=attention_nn1,
            down_conv_nn=global_nn1,
            *args,
            **kwargs
        )
        kwargs["nb_feature"] = None
        self.conv2 = RandlaConv(
            ratio2,
            16,  # KNN - k value
            point_pos_nn=point_pos_nn2,
            attention_nn=attention_nn2,
            down_conv_nn=global_nn2,
            *args,
            **kwargs
        )

    def convs(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        return data


class RandLANetRes(torch.nn.Module):
    def __init__(self, indim, outdim, ratio, point_pos_nn, attention_nn, down_conv_nn, *args, **kwargs):
        super(RandLANetRes, self).__init__()

        self._conv = DilatedResidualBlock(
            indim,
            outdim,
            ratio[0],
            ratio[1],
            point_pos_nn[0],
            point_pos_nn[1],
            attention_nn[0],
            attention_nn[1],
            down_conv_nn[0],
            down_conv_nn[1],
            *args,
            **kwargs
        )

    def forward(self, data):
        return self._conv.forward(data)
