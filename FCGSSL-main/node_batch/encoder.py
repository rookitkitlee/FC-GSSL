import torch
import torch.nn as nn

from pos_enc import PEG
from utils import get_activation
from conv import GATConv
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, encoder, out_dim, args=None):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(args.embed_dim, out_dim)
    
    def forward(self, x, edge_index, eigvals=None, eigvecs=None, mask_tokens=None, PE=None):
        h, pe = self.encoder.embed(x, edge_index, eigvals, eigvecs, mask_tokens, PE)
        pred = self.classifier(h)
        return pred, pe



class GraphEncoder(torch.nn.Module):
    def __init__(self, out_dim, args=None):
        super().__init__()
        self.gnn_dropout = args.gnn_dropout

        self.num_layer = args.enc_gnn_layer
        emb_dim = args.embed_dim
        hid_dim = emb_dim // args.heads

        self.gnns = nn.ModuleList()
        if self.num_layer == 1:
            self.gnns.append(GATConv(in_channels=args.feat_dim, out_channels=out_dim, heads=args.heads, args=args))
        else:
            self.gnns.append(GATConv(in_channels=args.feat_dim, out_channels=hid_dim, heads=args.heads, args=args))
            self.activations = nn.ModuleList()
            for layer in range(self.num_layer - 1):
                self.gnns.append(GATConv(in_channels=emb_dim, out_channels=hid_dim, heads=args.heads, args=args))        
                self.activations.append(get_activation(args.gnn_activation))

        self.pe_enc = PEG(args=args)
        

    def forward(self, x, x_union, x_inter, edge_index, edge_index_drop_union, edge_index_drop_inter, PE=None, PE_drop_union=None, PE_drop_inter=None):


        pe = self.pe_enc(PE) 
        pe_drop_union = self.pe_enc(PE_drop_union)
        pe_drop_inter = self.pe_enc(PE_drop_inter)

        h_list_feature = [x_union]
        pe_list_feature = [pe]
        edge_index_feature = edge_index

        h_list_structure = [x]
        pe_list_structure = [pe_drop_union]
        edge_index_structure = edge_index_drop_union

        h_list_both = [x_inter]
        pe_list_both = [pe_drop_inter]
        edge_index_both = edge_index_drop_inter



        for layer in range(self.num_layer):


            h_feature = F.dropout(h_list_feature[layer], p=self.gnn_dropout, training=self.training)
            h_feature, pe_feature = self.gnns[layer](h_feature, pe_list_feature[layer], edge_index_feature)

            h_structure = F.dropout(h_list_structure[layer], p=self.gnn_dropout, training=self.training)
            h_structure, pe_structure = self.gnns[layer](h_structure, pe_list_structure[layer], edge_index_structure)

            h_both = F.dropout(h_list_both[layer], p=self.gnn_dropout, training=self.training)
            h_both, pe_both = self.gnns[layer](h_both, pe_list_both[layer], edge_index_both)

            if layer != self.num_layer - 1:
                h_feature = self.activations[layer](h_feature)
                h_structure = self.activations[layer](h_structure)
                h_both = self.activations[layer](h_both)

            
            h_list_feature.append(h_feature)
            pe_list_feature.append(pe_feature)
            
            h_list_structure.append(h_structure)
            pe_list_structure.append(pe_structure)

            h_list_both.append(h_both)
            pe_list_both.append(pe_both)


        return h_list_feature[-1], h_list_structure[-1], h_list_both[-1]

    def embed(self, x, edge_index, PE=None):
        pe = self.pe_enc(PE)

        h_list = [x]
        pe_list = [pe]
        for layer in range(self.num_layer):
            h = F.dropout(h_list[layer], p=self.gnn_dropout, training=self.training)
            h, pe = self.gnns[layer](h, pe_list[layer], edge_index)

            if layer != self.num_layer - 1:
                h = self.activations[layer](h)
            
            h_list.append(h)
            pe_list.append(pe)
        
        return h_list[-1], pe_list[-1]