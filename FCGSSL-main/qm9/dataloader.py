import torch
from torch_geometric.loader.dataloader import Collater


def collate_fn_with_index(graphs):


    collater = Collater(dataset=None)
    batch = collater(batch=graphs)
    tab_sizes_n = [graphs[i].num_nodes for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    batch.snorm_n = snorm_n

    batch.x_indexes = [graphs[i].index for i in range(len(graphs))]

    batch.x_num_nodes = [graphs[i].num_nodes for i in range(len(graphs))]
    batch.x_num_edges = [graphs[i].edge_attr.shape[0] for i in range(len(graphs))]

    batch.x_node_offset = []
    temp_offset = 0
    for i in range(len(batch.x_num_nodes)):
        batch.x_node_offset.append(temp_offset)
        temp_offset += batch.x_num_nodes[i]

    batch.x_edge_offset = []
    temp_offset = 0
    for i in range(len(batch.x_num_edges)):
        batch.x_edge_offset.append(temp_offset)
        temp_offset += batch.x_num_edges[i]

    return batch

def collate_fn(graphs):
    collater = Collater(dataset=None)
    batch = collater(batch=graphs)
    tab_sizes_n = [graphs[i].num_nodes for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    batch.snorm_n = snorm_n
    
    return batch