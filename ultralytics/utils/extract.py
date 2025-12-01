import torch
import typing
from torch.fx import GraphModule
from torch.fx.node import Node


def graph_get_node_by_name(gm: GraphModule, name: str):
    fx_graph = gm.graph
    for node in fx_graph.nodes:
        if node.name == name:
            return node
    assert 0, f"error can not find the node {name}"
    return None


def get_internal_nodes(graph_module: GraphModule, start_nodes: typing.List[str], end_nodes: typing.List[str]):
    needed_nodes = set()
    inputs_nodes = set()
    outputs_nodes = set()

    worklist = list()
    for n_name in end_nodes:
        node = graph_get_node_by_name(graph_module, n_name)
        worklist.append(node)
        outputs_nodes.add(node)

    while len(worklist) > 0:
        node = worklist.pop()
        if node in needed_nodes:
            continue
        needed_nodes.add(node)
        if node.name not in start_nodes:
            worklist.extend(node.all_input_nodes)
        if node.name in start_nodes:
            inputs_nodes.update(node.all_input_nodes)
    return inputs_nodes, needed_nodes, outputs_nodes


def extract_subgraph(gm: GraphModule, start_nodes: typing.List[str], end_nodes: typing.List[str]) -> GraphModule:
    """
    从完整图中提取指定起点到终点的子图
    gm: 需要切子图的 graph module
    start_nodes: subgraph 的起始 node.name
    end_nodes: subgraph 的末尾 node.name
    需要保证 start_nodes 和 end_nodes 完全覆盖子图的起始节点和末尾节点
    """
    # 创建新图
    new_graph = torch.fx.Graph()
    env = {}  # 节点映射环境

    inputs_nodes, needed_nodes, outputs_nodes = get_internal_nodes(gm, start_nodes, end_nodes)
    # 1. 添加输入占位符
    for node in inputs_nodes:
        new_node = new_graph.placeholder(node.name, type_expr=node.type)
        new_node.meta = node.meta.copy()
        env[node] = new_node

    # 2. 添加中间节点
    # 找到从起点到终点的所有节点
    # all_nodes = set()
    # queue = collections.deque(end_nodes)

    # while queue:
    #     node = queue.popleft()
    #     if node in all_nodes:
    #         continue
    #     all_nodes.add(node)

    #     # 添加依赖节点
    #     for input_node in node.all_input_nodes:
    #         if input_node not in all_nodes:
    #             queue.append(input_node)

    # TODO: 按原始顺序添加节点
    for node in gm.graph.nodes:
        if node in needed_nodes and node not in inputs_nodes:
            # 复制节点
            new_node = new_graph.node_copy(
                node, 
                lambda n: env[n] if n in env else new_graph.placeholder(n.name)
            )
            new_node.meta = node.meta.copy()
            env[node] = new_node

    # 3. 添加输出
    output_values = [env[n] for n in outputs_nodes]
    new_graph.output(output_values[0] if len(output_values) == 1 else tuple(output_values))

    # 4. 创建子图模块
    return torch.fx.GraphModule(gm, new_graph)