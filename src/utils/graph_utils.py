import pandas as pd
import igraph as ig
import gravis as gv
from pathlib import Path

def load_graph(kg_dir_path: str, 
               node_col: str, 
               source_col: str, 
               target_col: str, 
               node_type_col: str = 'type', 
               edge_type_col: str = 'type', 
               directed=True, 
               remove_node_types: list[str]=None, 
               remove_edge_types: list[str]=None, 
               return_df: bool=False):
    
    edges_path = Path(f"{kg_dir_path}/edges.parquet")
    nodes_path = Path(f"{kg_dir_path}/nodes.parquet")

    if not edges_path.exists() or not nodes_path.exists():
        raise FileNotFoundError(f"Graph files not found in {kg_dir_path}. Please check the directory.")

    edges_df = pd.read_parquet(edges_path)
    nodes_df = pd.read_parquet(nodes_path)
    
    if remove_node_types is not None:
        nodes_df = nodes_df[~nodes_df[node_type_col].isin(remove_node_types)]
        nodes_df.reset_index(inplace=True, drop=True)
        edges_df = edges_df[edges_df[source_col].isin(nodes_df[node_col]) & edges_df[target_col].isin(nodes_df[node_col])]
        edges_df.reset_index(inplace=True, drop=True)

    if remove_edge_types is not None:
        edges_df = edges_df[~edges_df[edge_type_col].isin(remove_edge_types)]
        edges_df.reset_index(inplace=True, drop=True)
        nodes_df = nodes_df[nodes_df[node_col].isin(edges_df[source_col]) | nodes_df[node_col].isin(edges_df[target_col])]
        nodes_df.reset_index(inplace=True, drop=True)

    G = ig.Graph(directed=directed)
    G.add_vertices(nodes_df[node_col])
    G.add_edges(edges_df[[source_col, target_col]].values)

    for col in nodes_df.columns:
        G.vs[col] = nodes_df[col]
    for col in edges_df.columns:
        G.es[col] = edges_df[col]

    if return_df:
        return G, nodes_df, edges_df
    return G


def graph_find_related_nodes_to(nodes: list[str], words: list[str]):
    if isinstance(words, str):
        words = [words.split(" ")]
    related_nodes = set()
    for word in words:
        related_nodes = related_nodes.union([node for node in nodes if word.lower() in node.split(" ")])
    return list(related_nodes)


def graph_reasoning_paths_to_text(reasoning_paths: list[list[str]]):
    rps_text = []
    for rp in reasoning_paths:
        res = []
        i = 0
        while i < len(rp):
            res.append('(' + rp[i] + ')')
            if i < len(rp) - 1:
                res.append("--")
                res.append(rp[i+1])
                res.append("->")
            i+=2
        rps_text.append(" ".join(res))
    rps_text = "\n".join([f"Path {i+1}: {path}" for i, path in enumerate(rps_text)])
    return rps_text

def graph_reasoning_paths_to_nodes(G: ig.Graph, nodes: list[str]):
    reasoning_paths = []
    to = G.vs.select(name_notin=nodes)
    for node in nodes:
        paths = G.get_all_shortest_paths(node, to, mode='in')
        for path in paths:
            if len(path) <= 0: continue
            path = list(reversed(path))
            new_path = []
            for i in range(len(path) - 1):
                src = G.vs[path[i]]
                trg = G.vs[path[i+1]]
                edge = G.es.find(_source=src.index, _target=trg.index)
                new_path += [src['name'], edge['type']]
            new_path += [G.vs[path[-1]]['name']]
            reasoning_paths.append(new_path)
    return reasoning_paths


def visualize_graph(graph: ig.Graph, node_type_to_color=None, source=None, target=None):
    if node_type_to_color is None:
        node_type_to_color = {}
    g_vis = ig.Graph(directed=graph.is_directed())
    def shorten(string, max_length):
        if len(string) > max_length:
            n = int(max_length / 2 - 3)
            string = string[:n] + " ... " + string[-n:]
        return string
    for node in graph.vs:
        node_id = node["name"]
        node_type = node["type"]
        node_properties = {
            k: node[k]
            for k in node.attribute_names()
            if k not in ["name", "type"] and node[k] not in [None, "", []]
        }
        node_properties_str = "\n".join(
            f" <b>{k}:</b> {shorten(str(v), 120)}"
            for k, v in sorted(node_properties.items())
        )
        hover = f"<b>id:</b>{node_id}\n<b>type:</b>{node_type}\n<b>properties:</b>\n{node_properties_str}"
        hover = shorten(hover, 1000)
        coords = {}
        size = None
        try:
            coords["x"] = node["x"]
            coords["y"] = node["y"]
        except Exception:
            pass
        if source is not None:
            try:
                if node.index == source or node["name"] == source:
                    coords["x"] = 0 if target is None else -500
                    coords["y"] = 0
                    size = 20
            except KeyError:
                pass
        if target is not None:
            try:
                if node.index == target or node["name"] == target:
                    coords["x"] = 0 if source is None else +500
                    coords["y"] = 0
                    size = 20
            except KeyError:
                pass

        g_vis.add_vertex(
            name=node["name"],
            hover=hover,
            click="$hover",
            color=node_type_to_color.get(node["type"], None),
            size=size,
            **coords,
        )

    for edge in graph.es:
        try:
            hover = edge["type"]
        except Exception:
            hover = None
        g_vis.add_edge(
            edge.source,
            edge.target,
            hover=hover,
            kwds={
                "type": edge["type"],
            }
        )
    fig = gv.d3(
        g_vis,
        node_label_data_source="name",
        edge_label_data_source="type",
        many_body_force_strength=-1500,
        edge_curvature=0.1,
        node_hover_neighborhood=True,
    )
    return fig
