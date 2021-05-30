# --------------------------------------------------------- #
#                                                           #
#   Project: 3D Point Cloud Semantic Segmentation           #
#   University: Brno University of Technology               #
#   Year: 2021                                              #
#                                                           #
#   Authors:                                                #
#       Bořek Reich    <xreich06@stud.fit.vutbr.cz>         #
#       Martin Chládek <xchlad16@stud.fit.vutbr.cz>         #
#                                                           #
# --------------------------------------------------------- #

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_points3d.core.base_conv.message_passing import *


class RandlaAgrKernel(MessagePassing):
    def __init__(self, point_pos_nn=None, attention_nn=None, global_nn=None, *args, **kwargs):
        MessagePassing.__init__(self, aggr="add")

        self.point_pos_nn = MLP(point_pos_nn)
        self.attention_nn = MLP(attention_nn)
        self.global_nn = MLP(global_nn)  # == down_conv_nn from .yaml file

    def forward(self, x, pos, edge_index):
        # print(f"x.size(): {x.size()}, \n"
        #       f"pos0.size(): {pos[0].size()}, \n"
        #       f"pos1.size(): {pos[1].size()}")
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    def message(self, x_k, pos_i, pos_k):
        if x_k is None:
            x_k = pos_k

        # This is in paper - Figure 3. PART of LocSE block (Local Spatial Encoding)
        # compute relative position encoding
        # in paper equation (1)
        v_ik = pos_i - pos_k

        # print(f"vij: {vij}, pos_i: {pos_i}, pos_j: {pos_j}")
        relPointPos = torch.cat([pos_i, pos_k, v_ik, torch.norm(v_ik, dim=1).unsqueeze(1)], dim=1)
        r_ik = self.point_pos_nn(relPointPos)  # (r_i^k)

        # This is in paper - Figure 3. PART of LocSE block (Local Spatial Encoding)
        # concatenate position encoding with feature vector for feature augmentation
        f_ik_hat = torch.cat([x_k, r_ik], dim=1)

        # This is in paper - Figure 3. Attentive Pooling block
        # attentive pooling
        # in paper equation (2) and (3)
        g_f_ik = self.attention_nn(f_ik_hat)
        s_ik = F.softmax(g_f_ik, -1)

        return s_ik * f_ik_hat

    def update(self, aggr_out):
        # Shared MLP
        return self.global_nn(aggr_out)


class RandLANetRes(BaseResnetBlockDown):
    def __init__(self,
                 indim, convdim, outdim,
                 ratio,
                 point_pos_nn,
                 attention_nn,
                 down_conv_nn,
                 *args, **kwargs):
        super(RandLANetRes, self).__init__(
            sampler=RandomSampler(ratio),
            neighbour_finder=KNNNeighbourFinder(16),
            indim=indim, convdim=convdim, outdim=outdim,
            *args
        )
        self.point_pos_nn = point_pos_nn
        self.attention_nn = attention_nn
        self.down_conv_nn = down_conv_nn
        self.indim = indim
        self.convdim = convdim
        self.outdim = outdim
        kwargs["nb_feature"] = None

        # First agr block = LocSE + Pooling
        self.conv1 = RandlaAgrKernel(
            point_pos_nn=point_pos_nn[0],
            attention_nn=attention_nn[0],
            global_nn=down_conv_nn[0],
            *args,
            **kwargs
        )
        # Second agr block = LocSE + Pooling
        self.conv2 = RandlaAgrKernel(
            point_pos_nn=point_pos_nn[1],
            attention_nn=attention_nn[1],
            global_nn=down_conv_nn[1],
            *args,
            **kwargs
        )

    # Called by conv method of BaseResNet
    def convs(self, x, pos, edge_index):
        data = self.conv1(x, pos, edge_index)
        data = self.conv2(data, pos, edge_index)
        return data, pos, edge_index, None
