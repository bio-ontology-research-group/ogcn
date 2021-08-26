import networkx
import dgl

def dgl_to_networkx(dglGraph):
    g = dgl.to_homogeneous(dglGraph)
    return dgl.to_networkx(g)


def dgl_to_networkx_file(path_dgl_graph):
    graphs, data_dict = dgl.load_graphs(path_dgl_graph)
    g_dgl = graphs[0]
    g_networkx = dgl_to_networkx(g_dgl)

    networkx.write_gpickle(g_networkx, "../data/go_cat.pkl")
