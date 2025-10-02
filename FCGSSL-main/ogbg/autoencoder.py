from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from utils import noise_fn, get_activation

from torch_geometric.utils import remove_self_loops, to_undirected
from utils import get_activation, noise_fn
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)


class MaskLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, mask_tokens=None):
        if mask_tokens is not None:
            features = features[mask_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias

        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.out_proj = nn.Linear(128, 1)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, dist, edge_index):
        dist = self.dense(dist)
        dist = self.activation_fn(dist)
        dist = self.layer_norm(dist)
        dist = self.out_proj(dist)
        
        edge_index, dist = to_undirected(edge_index=edge_index, edge_attr=dist, reduce="mean")        
        return dist.squeeze(), edge_index


class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, num_atom_type=0, args=None):
        super(GraphAutoEncoder, self).__init__()
        self.args = args
        self.encoder = encoder

        self.mask_ratio = args.mask_ratio
        self.edge_drop = args.edge_drop
        self.noise_val = args.noise_val

        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.masked_ssl_loss = float(args.masked_ssl_loss)
        self.ssl_sample = args.ssl_sample
        self.ssl_temp = args.ssl_temp

        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = num_atom_type
        self.alpha_l = args.alpha_l

        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(heads=args.embed_dim, activation_fn=args.task_head_activation)

        self.node_pred2 = MaskLMHead(args.embed_dim, output_dim=args.max_freqs, activation_fn=args.task_head_activation)

        # kitlee
        # self.enc_mask_token = nn.Parameter(torch.zeros(1, args.feat_dim))

    def forward(self, batch, x_drop_pro):
        
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        snorm_n = batch.snorm_n

        u = None
        PE = None
        e, u = batch.EigVals.clone(), batch.EigVecs.clone()
        mask_u = torch.isnan(u)
        u[mask_u] = 0   # [n_sum, max_freqs]
        mask_e = torch.isnan(e)
        e[mask_e] = 0
        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)

        sample_graphs_ids = batch.x_indexes
        sample_graphs_num_nodes = batch.x_num_nodes
        sample_graphs_num_edges = batch.x_num_edges
        sample_graphs_node_offset = batch.x_node_offset
        sample_graphs_edge_offset = batch.x_edge_offset


        # print('WWWWWWWWWWWWWWWWWWWWWWWW')
        # print(sum(sample_graphs_num_nodes))
        # print(sum(sample_graphs_num_edges))

        # x_masked, u_masked, mask_tokens = self.encoding_mask_noise(x=x, u=u, mask_ratio=self.mask_ratio)
        
        # PE_noise = torch.linalg.norm(u_masked[edge_index[0]] - u_masked[edge_index[1]], dim=-1)
        # enc_rep, pe = self.encoder(x, x_masked, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE, PE_noise=PE_noise)
        
        # enc_rep = self.node_pred(enc_rep, mask_tokens)
        # pe_loss = 0.0
        # reconstruct_dist, _ = self.pe_reconstruct_heads(pe, edge_index)
        # atom_loss = self.cal_atom_loss(pred_node=enc_rep, target_atom=x, mask_tokens=mask_tokens,
        #                         loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)
        # pe_loss = self.cal_pe_loss(reconstruct_dis=reconstruct_dist, target_dis=PE, edge_index=edge_index, mask_tokens=mask_tokens)
        # loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss
        # return loss


        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(x.shape)
        # print(edge_drop_pro_num.shape)

        edge_index_pe = edge_index

        drop_edge_union_index, drop_edge_union_pe, drop_edge_union, drop_edge_inter_index, drop_edge_inter_pe, drop_edge_inter, drop_edge_union_mask, drop_edge_inter_mask = self.encoding_edge_drop(edge_index, x_drop_pro, sample_graphs_ids, sample_graphs_num_edges, sample_graphs_edge_offset, u, self.edge_drop)
        mask_nodes_union, x_union, mask_nodes_inter, x_inter = self.encoding_node_mask(x, x_drop_pro, sample_graphs_ids, sample_graphs_num_nodes, sample_graphs_node_offset, mask_ratio=self.mask_ratio, replace_ratio=0.0)

        enc_feature, enc_structure, enc_both = self.encoder(x, x_union, x_inter, edge_index, drop_edge_union_index, drop_edge_inter_index, drop_edge_union_mask, drop_edge_inter_mask, PE=PE, PE_drop_union=drop_edge_union_pe, PE_drop_inter=drop_edge_inter_pe, edge_attr=edge_attr, snorm_n=snorm_n)

        enc_rep = self.node_pred(enc_feature, mask_nodes_union)
        atom_loss = self.cal_atom_loss(pred_node=enc_rep, target_atom=x, mask_tokens=mask_nodes_union, loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)

        pe = enc_structure[edge_index_pe[0][drop_edge_union]] * enc_structure[edge_index_pe[1][drop_edge_union]]
        pe = self.node_pred2(pe)
        target_pe = u[edge_index_pe[0][drop_edge_union]] * u[edge_index_pe[1][drop_edge_union]]
        pe_loss = self.cal_atom_loss2(pred_node=pe, target_atom=target_pe, mask_tokens=None, loss_fn=self.atom_recon_type, alpha_l=self.alpha_l)

        ssl_loss = self.cal_ssl_loss(enc_both, enc_feature, enc_structure)
        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss + self.masked_ssl_loss * ssl_loss

        return loss




    def encoding_edge_drop_help(self, u, drop_edge, edge_index):

        mask = torch.ones(len(edge_index[0]), dtype=torch.bool, device=u.device)
        mask[drop_edge] = False  

        row = edge_index[0][mask]
        col = edge_index[1][mask]
    
        edge_drop_index = torch.stack([row, col], dim=0) 

        # edge_index_pe, _ = remove_self_loops(edge_drop_index, None)
        # edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
        edge_index_pe = edge_drop_index
        PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)  # [e_sum, 1]   # PE 对应的是 边的 表示

        return edge_drop_index, PE, mask


    # kitlee
    def encoding_edge_drop(self, edge_index, x_drop_pro, sample_graphs_ids, sample_graphs_num_edges, sample_graphs_edge_offset, u, mask_ratio):

        drop_edge_num = []
        drop_edge_qua = []

        for i in range(len(sample_graphs_ids)):
            
            id = sample_graphs_ids[i]
            num_edge = sample_graphs_num_edges[i]
            offset = sample_graphs_edge_offset[i]

            edge_drop_pro_num = x_drop_pro[id]['edge_drop_pro_num']
            edge_drop_pro_qua = x_drop_pro[id]['edge_drop_pro_qua']

            drop_num = int(num_edge * mask_ratio)
            if drop_num == 0:
                drop_num = 1

            if edge_drop_pro_num.shape[0] == 0:
                continue

            # print("#########$$$$$$$$$$$")
            # print(edge_drop_pro_num.shape)
            # print(drop_num)

            temp_drop_edge_num = torch.multinomial(edge_drop_pro_num, num_samples=drop_num, replacement=False).to(edge_index.device)
            temp_drop_edge_qua = torch.multinomial(edge_drop_pro_qua, num_samples=drop_num, replacement=False).to(edge_index.device)

            drop_edge_num.append(temp_drop_edge_num + offset)
            drop_edge_qua.append(temp_drop_edge_qua + offset)

        drop_edge_num = torch.cat(drop_edge_num)
        drop_edge_qua = torch.cat(drop_edge_qua)

        drop_edge_union = torch.unique(torch.cat((drop_edge_num, drop_edge_qua)))
        drop_edge_inter = torch.unique(drop_edge_num[torch.isin(drop_edge_num, drop_edge_qua)])

        drop_edge_union_index, drop_edge_union_pe, drop_edge_union_mask = self.encoding_edge_drop_help(u, drop_edge_union, edge_index)
        drop_edge_inter_index, drop_edge_inter_pe, drop_edge_inter_mask = self.encoding_edge_drop_help(u, drop_edge_inter, edge_index)

        return drop_edge_union_index, drop_edge_union_pe, drop_edge_union, drop_edge_inter_index, drop_edge_inter_pe, drop_edge_inter, drop_edge_union_mask, drop_edge_inter_mask


    def encoding_node_mask_help(self, x, mask_nodes, mask_ratio, replace_ratio):

        # num_mask_nodes = len(mask_nodes)


        # if replace_ratio > 0:
        #     num_noise_nodes = int(replace_ratio * num_mask_nodes)  # 要被其他节点的特征 替换的数目
        #     perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        #     token_nodes = mask_nodes[perm_mask[: int(mask_ratio * num_mask_nodes)]] # 选择了使用0添加掩码的节点
        #     noise_nodes = mask_nodes[perm_mask[-int(replace_ratio * num_mask_nodes):]] # 选择了使用其他节点特征替换的节点
        #     noise_to_be_chosen = torch.randperm(x.shape[0], device=x.device)[:num_noise_nodes] # 选择要替换的特征节点

        #     out_x = x.clone()
        #     out_x[token_nodes] = 0.0
        #     out_x[noise_nodes] = x[noise_to_be_chosen]
        # else:
        #     out_x = x.clone()
        #     token_nodes = mask_nodes
        #     out_x[mask_nodes] = 0.0

        # out_x[token_nodes] += self.enc_mask_token  # 给添加0掩码的 添加可学习参数

        # return out_x.to(x.device)

        # print("###################")
        # print(x.shape)

        # print(mask_nodes)

        out_x = x.clone()
        out_x[mask_nodes, 0] = self.num_atom_type
        # out_x[mask_nodes] = 0.0
        # out_x[mask_nodes] += self.enc_mask_token
        return out_x.to(x.device)

  # kitlee
    def encoding_node_mask(self, x, x_drop_pro, sample_graphs_ids, sample_graphs_num_nodes, sample_graphs_node_offset, mask_ratio, replace_ratio):


        # sample_num = int(mask_ratio * x.shape[0])
        # mask_nodes_num = torch.multinomial(node_mask_pro_num, num_samples=sample_num, replacement=False).to(x.device)
        # mask_nodes_qua = torch.multinomial(node_mask_pro_qua, num_samples=sample_num, replacement=False).to(x.device)


        mask_nodes_num = []
        mask_nodes_qua = []

        for i in range(len(sample_graphs_ids)):
            
            id = sample_graphs_ids[i]
            num_node = sample_graphs_num_nodes[i]
            offset = sample_graphs_node_offset[i]

            node_drop_pro_num = x_drop_pro[id]['node_mask_pro_num']
            node_drop_pro_qua = x_drop_pro[id]['node_mask_pro_qua']

            mask_num = int(num_node * mask_ratio)
            if mask_num == 0:
                mask_num = 1

            if node_drop_pro_num.shape[0] == 0:
                continue

            temp_mask_node_num = torch.multinomial(node_drop_pro_num, num_samples=mask_num, replacement=False).to(x.device)
            temp_mask_node_qua = torch.multinomial(node_drop_pro_qua, num_samples=mask_num, replacement=False).to(x.device)

            # mask_nodes_num.extend([x + offset for x in temp_mask_node_num])
            # mask_nodes_qua.extend([x + offset for x in temp_mask_node_qua])

            mask_nodes_num.append(temp_mask_node_num + offset)
            mask_nodes_qua.append(temp_mask_node_qua + offset)

        mask_nodes_num = torch.cat(mask_nodes_num)
        mask_nodes_qua = torch.cat(mask_nodes_qua)

        mask_nodes_union = torch.unique(torch.cat((mask_nodes_num, mask_nodes_qua)))
        mask_nodes_inter = torch.unique(mask_nodes_num[torch.isin(mask_nodes_num, mask_nodes_qua)])

        x_union = self.encoding_node_mask_help(x, mask_nodes_union, mask_ratio, replace_ratio)
        x_inter = self.encoding_node_mask_help(x, mask_nodes_inter, mask_ratio, replace_ratio)

        return mask_nodes_union, x_union, mask_nodes_inter, x_inter
          
    # def encoding_mask_noise(self, x, u, mask_ratio):
    #     num_nodes = x.shape[0]
    #     perm = torch.randperm(num_nodes, device=x.device)

    #     # random masking
    #     num_mask_nodes = int(mask_ratio * num_nodes)
    #     mask_nodes = perm[: num_mask_nodes]

    #     out_x = x.clone()
    #     out_x[mask_nodes, 0] = self.num_atom_type

    #     u_masked = None
    #     u_masked = u.clone()
    #     pos_noise = noise_fn(self.noise_val, len(mask_nodes), u.size(1)).to(u_masked.device)
    #     u_masked[mask_nodes] += pos_noise

    #     return out_x, u_masked, mask_nodes
    

    def embed(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        snorm_n = batch.snorm_n
        
        u = None
        PE = None
        e, u = batch.EigVals.clone(), batch.EigVecs.clone()
        mask_u = torch.isnan(u)
        u[mask_u] = 0   # [n_sum, max_freqs]
        mask_e = torch.isnan(e)
        e[mask_e] = 0

        PE = torch.linalg.norm(u[edge_index[0]] - u[edge_index[1]], dim=-1)

        enc_rep, _ = self.encoder.embed(x, edge_index, edge_attr=edge_attr, snorm_n=snorm_n, PE=PE)
                
        return enc_rep


    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index, mask_tokens):
        edge_index, target_dis = to_undirected(edge_index=edge_index, edge_attr=target_dis, reduce="mean")
        row = edge_index[0]
        idx = torch.isin(row, mask_tokens)

        reconstruct_dis = reconstruct_dis[idx]
        target_dis = target_dis[idx]

        pe_reconstruct_loss = F.smooth_l1_loss(
            reconstruct_dis,
            target_dis,
            reduction="mean",
            beta=1.0,
        )

        return pe_reconstruct_loss

    def cal_atom_loss(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):
        target_atom = target_atom[mask_tokens, 0]

        if loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
            target_atom = F.one_hot(target_atom, num_classes=self.num_atom_type).float()
            atom_loss = criterion(pred_node, target_atom)
        elif loss_fn == "mse":
            target_atom = F.one_hot(target_atom, num_classes=self.num_atom_type).float()
            atom_loss = self.mse_loss(pred_node, target_atom)
        else:
            criterion = nn.CrossEntropyLoss()
            atom_loss = criterion(pred_node, target_atom)

        return atom_loss

    def cal_atom_loss2(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):


        if mask_tokens is not None:
            target_atom = target_atom[mask_tokens]

        if loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
            atom_loss = criterion(pred_node, target_atom)
        elif loss_fn == "mse":
            atom_loss = self.mse_loss(pred_node, target_atom)
        else:
            criterion = nn.CrossEntropyLoss()
            atom_loss = criterion(pred_node, target_atom)

        return atom_loss

    def sce_loss(self, x, y, alpha=1):
        x = F.normalize(x.float(), p=2, dim=-1)
        y = F.normalize(y.float(), p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss
    
    def cal_ssl_loss(self, emb1s, emb2s, emb3s):

        def cal_loss(emb1, emb2):
            temp = self.ssl_temp
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss
        


        perm = torch.randperm(len(emb1s), device=emb1s.device)
        selected_nodes = perm[: self.ssl_sample]  # 选择要添加掩码的节点

        emb1 = F.normalize(emb1s[selected_nodes], dim=1)
        emb2 = F.normalize(emb2s[selected_nodes], dim=1)
        emb3 = F.normalize(emb3s[selected_nodes], dim=1)

        # emb1 = emb1s[selected_nodes]
        # emb2 = emb2s[selected_nodes]
        
        return cal_loss(emb1, emb2) + cal_loss(emb1, emb3)