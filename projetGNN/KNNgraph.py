from torch_geometric.transforms import KNNGraph

graphKNN = KNNGraph(k = 5,
        loop = False,
        force_undirected = True,
        flow= 'source_to_target',
        cosine = False,
        num_workers = 1)