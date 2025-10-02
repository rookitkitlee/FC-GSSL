import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.utils.features import get_atom_feature_dims
from ogb.graphproppred.mol_encoder import BondEncoder

from utils import get_activation
from pos_enc import PEG
from conv import GatedGCNConv


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_atom_type):
        super(AtomEncoder, self).__init__()
        full_atom_feature_dims = get_atom_feature_dims()
        full_atom_feature_dims[0] = num_atom_type+1
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class GraphEncoder(torch.nn.Module):
    def __init__(self, out_dim, args=None):
        super().__init__()
        emb_dim = args.embed_dim
        self.x_embedding = AtomEncoder(emb_dim, args.num_atom_type)
        self.edge_embedding = BondEncoder(emb_dim)

        self.pe_enc = PEG(args=args)

        self.num_layer = args.enc_gnn_layer
        emb_dim = args.embed_dim
        self.gnn_dropout = args.gnn_dropout
        self.gnn_edge_dropout = args.gnn_edge_dropout
        
        self.gnns = nn.ModuleList()
        if self.num_layer == 1:
            self.gnns.append(GatedGCNConv(in_channels=args.embed_dim, out_channels=args.embed_dim, args=args))
        else:
            self.gnns.append(GatedGCNConv(in_channels=args.embed_dim, out_channels=args.embed_dim, args=args))
            self.activations = nn.ModuleList()
            self.activations_edge = nn.ModuleList()
            self.activations_pe = nn.ModuleList()
            for layer in range(self.num_layer - 1):
                self.gnns.append(GatedGCNConv(in_channels=args.embed_dim, out_channels=args.embed_dim, args=args))        
                self.activations.append(get_activation(args.gnn_activation))
                self.activations_edge.append(get_activation(args.gnn_activation))
                self.activations_pe.append(get_activation(args.gnn_activation))

        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms_edge = torch.nn.ModuleList()
        self.batch_norms_pe = torch.nn.ModuleList()
        
        for layer in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_edge.append(torch.nn.BatchNorm1d(emb_dim))
            self.batch_norms_pe.append(torch.nn.BatchNorm1d(emb_dim))


    # def forward(self, x, x_masked, edge_index, edge_attr=None, snorm_n=None, PE=None, PE_noise=None):
    #     x = self.x_embedding(x)
    #     x_masked = self.x_embedding(x_masked)
    #     e = self.edge_embedding(edge_attr) if edge_attr is not None else None

    #     pe = self.pe_enc(PE)
    #     pe_noise = self.pe_enc(PE_noise)

    #     h_list = [x]
    #     e_h_list = [e]
    #     e_pe_list = [e]
    #     h_masked_list = [x_masked]
    #     pe_list = [pe]
    #     pe_noise_list = [pe_noise]
    #     for layer in range(self.num_layer):
    #         h_masked, e_h, pe = self.gnns[layer](h_masked_list[layer], pe_list[layer], edge_index, e_h_list[layer], snorm_n)

    #         h, e_pe, pe_noise = self.gnns[layer](h_list[layer], pe_noise_list[layer], edge_index, e_pe_list[layer], snorm_n)

    #         h_masked = self.batch_norms[layer](h_masked)
    #         e_h = self.batch_norms_edge[layer](e_h)
    #         h = self.batch_norms[layer](h)
    #         e_pe = self.batch_norms_edge[layer](e_pe)
            
    #         if layer != self.num_layer - 1:
    #             h = self.activations[layer](h)
    #             e_pe = self.activations_edge[layer](e_pe)
    #             h_masked = self.activations[layer](h_masked)
    #             e_h = self.activations_edge[layer](e_h)
            
    #         h_masked = F.dropout(h_masked, p=self.gnn_dropout, training=self.training)
    #         e_h = F.dropout(e_h, p=self.gnn_edge_dropout, training=self.training)
    #         pe = F.dropout(pe, p=self.gnn_edge_dropout, training=self.training)

    #         h = F.dropout(h, p=self.gnn_dropout, training=self.training)
    #         e_pe = F.dropout(e_pe, p=self.gnn_edge_dropout, training=self.training)
    #         pe_noise = F.dropout(pe_noise, p=self.gnn_edge_dropout, training=self.training)
            
    #         h_masked_list.append(h_masked)
    #         h_list.append(h)
    #         e_h_list.append(e_h)
    #         e_pe_list.append(e_pe)
    #         pe_noise_list.append(pe_noise)
    #         pe_list.append(pe)
        
    #     return h_masked_list[-1], pe_noise_list[-1]


    def forward(self, x, x_union, x_inter, edge_index, edge_index_drop_union, edge_index_drop_inter, drop_edge_union_mask, drop_edge_inter_mask, PE=None, PE_drop_union=None, PE_drop_inter=None, edge_attr=None, snorm_n=None):


        x = self.x_embedding(x)
        x_union = self.x_embedding(x_union)
        x_inter = self.x_embedding(x_inter)
        e = self.edge_embedding(edge_attr) if edge_attr is not None else None

        # print("###########################")
        # print(e.shape)

        pe = self.pe_enc(PE) 
        pe_drop_union = self.pe_enc(PE_drop_union)
        pe_drop_inter = self.pe_enc(PE_drop_inter)

        h_list_feature = [x_union]
        pe_list_feature = [pe]
        edge_index_feature = edge_index
        e_list_feature = [e]

        # print('-------------------------')
        # print(h_list_feature[-1].shape)
        # print(pe_list_feature[-1].shape)
        # print(e_list_feature[-1].shape)
        # print(edge_index_feature.shape)

        h_list_structure = [x]
        pe_list_structure = [pe_drop_union]
        edge_index_structure = edge_index_drop_union
        e_list_structure = [e[drop_edge_union_mask]]

        # print('-------------------------')
        # print(h_list_structure[-1].shape)
        # print(pe_list_structure[-1].shape)
        # print(e_list_structure[-1].shape)
        # print(edge_index_structure.shape)

        h_list_both = [x_inter]
        pe_list_both = [pe_drop_inter]
        edge_index_both = edge_index_drop_inter
        e_list_both = [e[drop_edge_inter_mask]]


        # print('-------------------------')
        # print(h_list_both[-1].shape)
        # print(pe_list_both[-1].shape)
        # print(e_list_both[-1].shape)
        # print(edge_index_both.shape)




        for layer in range(self.num_layer):

            h_feature, e_feature, pe_feature = self.gnns[layer](h_list_feature[layer], pe_list_feature[layer], edge_index_feature, e_list_feature[layer], snorm_n)
            h_structure, e_structure, pe_structure = self.gnns[layer](h_list_structure[layer], pe_list_structure[layer], edge_index_structure, e_list_structure[layer], snorm_n)
            h_both, e_both, pe_both = self.gnns[layer](h_list_both[layer], pe_list_both[layer], edge_index_both, e_list_both[layer], snorm_n)

            h_feature = self.batch_norms[layer](h_feature)
            h_structure = self.batch_norms[layer](h_structure)
            h_both = self.batch_norms[layer](h_both)


            # print("-------------------------")
            # print(e_feature.shape)
            # print(e_structure.shape)
            # print(e_both.shape)

            e_feature = self.batch_norms_edge[layer](e_feature)
            # if e_structure.shape[0] > 1:
            #     e_structure = self.batch_norms_edge[layer](e_structure)
            e_structure = self.batch_norms_edge[layer](e_structure)
            e_both = self.batch_norms_edge[layer](e_both)

            
            if layer != self.num_layer - 1:

                h_feature = self.activations[layer](h_feature)
                h_structure = self.activations[layer](h_structure)
                h_both = self.activations[layer](h_both)

                e_feature = self.activations_edge[layer](e_feature)
                e_structure = self.activations_edge[layer](e_structure)
                e_both = self.activations_edge[layer](e_both)
              
            
            h_feature = F.dropout(h_feature, p=self.gnn_dropout, training=self.training)
            e_feature = F.dropout(e_feature, p=self.gnn_edge_dropout, training=self.training)
            pe_feature = F.dropout(pe_feature, p=self.gnn_edge_dropout, training=self.training)

            h_structure = F.dropout(h_structure, p=self.gnn_dropout, training=self.training)
            e_structure = F.dropout(e_structure, p=self.gnn_edge_dropout, training=self.training)
            pe_structure = F.dropout(pe_structure, p=self.gnn_edge_dropout, training=self.training)

            h_both = F.dropout(h_both, p=self.gnn_dropout, training=self.training)
            e_both = F.dropout(e_both, p=self.gnn_edge_dropout, training=self.training)
            pe_both = F.dropout(pe_both, p=self.gnn_edge_dropout, training=self.training)



            # h_feature = F.normalize(h_feature, dim=1)
            # e_feature = F.normalize(e_feature, dim=1)
            # pe_feature = F.normalize(pe_feature, dim=1)

            # h_structure = F.normalize(h_structure, dim=1)
            # e_structure = F.normalize(e_structure, dim=1)
            # pe_structure = F.normalize(pe_structure, dim=1)

            # h_both = F.normalize(h_both, dim=1)
            # e_both = F.normalize(e_both, dim=1)
            # pe_both = F.normalize(pe_both, dim=1)



           
            h_list_feature.append(h_feature)
            e_list_feature.append(e_feature)
            pe_list_feature.append(pe_feature)
            
            h_list_structure.append(h_structure)
            e_list_structure.append(e_structure)
            pe_list_structure.append(pe_structure)

            h_list_both.append(h_both)
            e_list_both.append(e_both)
            pe_list_both.append(pe_both)

        return h_list_feature[-1], h_list_structure[-1], h_list_both[-1]



    def embed(self, x, edge_index, edge_attr=None, snorm_n=None, PE=None):
        x = self.x_embedding(x)
        e = self.edge_embedding(edge_attr) if edge_attr is not None else None
        pe = self.pe_enc(PE)

        h_list = [x]
        e_list = [e]
        pe_list = [pe]
        for layer in range(self.num_layer):
            h, e_h, pe = self.gnns[layer](h_list[layer], pe_list[layer], edge_index, e_list[layer], snorm_n)

            h = self.batch_norms[layer](h)
            e_h = self.batch_norms_edge[layer](e_h)
            
            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
                e_h = self.activations_edge[layer](e_h)
            
            h = F.dropout(h, p=self.gnn_dropout, training=self.training)
            e_h = F.dropout(e_h, p=self.gnn_edge_dropout, training=self.training)
            pe = F.dropout(pe, p=self.gnn_edge_dropout, training=self.training)
            


            # h = F.normalize(h, dim=1)
            # e_h = F.normalize(e_h, dim=1)
            # pe = F.normalize(pe, dim=1)


            h_list.append(h)
            e_list.append(e_h)
            pe_list.append(pe)
        
        return h_list[-1], pe_list[-1]