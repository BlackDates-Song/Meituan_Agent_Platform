"""
图数据加载和处理模块
使用 DGL 库加载 Meituan TRD 数据集的图结构数据
"""

import dgl
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphDataLoader:
    """图数据加载器"""
    
    def __init__(self, graph_path: str):
        """
        初始化图数据加载器
        
        Args:
            graph_path: graph.bin 文件路径
        """
        self.graph_path = Path(graph_path)
        self.graphs = None
        self.graph = None
        self.node_types = None
        self.edge_types = None
        
    def load_graph(self) -> dgl.DGLGraph:
        """
        加载图数据
        
        Returns:
            DGL 图对象
        """
        if not self.graph_path.exists():
            raise FileNotFoundError(f"图数据文件不存在: {self.graph_path}")
            
        logger.info(f"正在加载图数据: {self.graph_path}")
        
        try:
            # 使用 DGL 加载图数据
            self.graphs, labels = dgl.load_graphs(str(self.graph_path))
            
            # 通常第一个图就是我们需要的
            self.graph = self.graphs[0]
            
            logger.info(f"✓ 图数据加载成功")
            self._print_graph_info()
            
            return self.graph
            
        except Exception as e:
            logger.error(f"加载图数据失败: {str(e)}")
            raise
            
    def _print_graph_info(self):
        """打印图的基本信息"""
        if self.graph is None:
            logger.warning("图尚未加载")
            return
            
        logger.info("=" * 60)
        logger.info("图结构信息")
        logger.info("=" * 60)
        
        # 判断是异构图还是同构图
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            logger.info("图类型: 异构图 (Heterogeneous Graph)")
            
            # 节点类型和数量
            logger.info(f"\n节点类型 ({len(self.graph.ntypes)}):")
            for ntype in self.graph.ntypes:
                num_nodes = self.graph.num_nodes(ntype)
                logger.info(f"  - {ntype}: {num_nodes:,} 个节点")
                
            # 边类型和数量
            logger.info(f"\n边类型 ({len(self.graph.etypes)}):")
            for etype in self.graph.canonical_etypes:
                num_edges = self.graph.num_edges(etype)
                logger.info(f"  - {etype}: {num_edges:,} 条边")
                
            # 节点特征
            logger.info(f"\n节点特征:")
            for ntype in self.graph.ntypes:
                ndata = self.graph.nodes[ntype].data
                if len(ndata) > 0:
                    logger.info(f"  {ntype}:")
                    for key, value in ndata.items():
                        logger.info(f"    - {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    logger.info(f"  {ntype}: 无特征")
                    
            # 边特征
            logger.info(f"\n边特征:")
            for etype in self.graph.canonical_etypes:
                edata = self.graph.edges[etype].data
                if len(edata) > 0:
                    logger.info(f"  {etype}:")
                    for key, value in edata.items():
                        logger.info(f"    - {key}: shape {value.shape}, dtype {value.dtype}")
                        
        else:
            logger.info("图类型: 同构图 (Homogeneous Graph)")
            logger.info(f"节点数: {self.graph.num_nodes():,}")
            logger.info(f"边数: {self.graph.num_edges():,}")
            
            # 节点特征
            if len(self.graph.ndata) > 0:
                logger.info(f"\n节点特征:")
                for key, value in self.graph.ndata.items():
                    logger.info(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                    
            # 边特征
            if len(self.graph.edata) > 0:
                logger.info(f"\n边特征:")
                for key, value in self.graph.edata.items():
                    logger.info(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                    
        logger.info("=" * 60)
        
    def get_node_ids(self, node_type: Optional[str] = None) -> torch.Tensor:
        """
        获取节点ID列表
        
        Args:
            node_type: 节点类型（异构图需要指定）
            
        Returns:
            节点ID张量
        """
        if self.graph is None:
            raise RuntimeError("请先加载图数据")
            
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            if node_type is None:
                raise ValueError("异构图需要指定节点类型")
            return torch.arange(self.graph.num_nodes(node_type))
        else:
            return torch.arange(self.graph.num_nodes())
            
    def get_neighbors(self, node_id: int, node_type: Optional[str] = None,
                     edge_type: Optional[Tuple] = None) -> List[int]:
        """
        获取节点的邻居
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            edge_type: 边类型（异构图）
            
        Returns:
            邻居节点ID列表
        """
        if self.graph is None:
            raise RuntimeError("请先加载图数据")
            
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            if edge_type is None:
                raise ValueError("异构图需要指定边类型")
            successors = self.graph.successors(node_id, etype=edge_type)
        else:
            successors = self.graph.successors(node_id)
            
        return successors.tolist()
        
    def sample_subgraph(self, seed_nodes: List[int], 
                       num_layers: int = 2,
                       fanout: int = 10,
                       node_type: Optional[str] = None) -> dgl.DGLGraph:
        """
        采样子图（用于小批量训练）
        
        Args:
            seed_nodes: 种子节点列表
            num_layers: 采样层数
            fanout: 每层采样的邻居数
            node_type: 节点类型
            
        Returns:
            采样的子图
        """
        if self.graph is None:
            raise RuntimeError("请先加载图数据")
            
        seed_nodes = torch.LongTensor(seed_nodes)
        
        # 使用邻居采样
        sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout] * num_layers)
        
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            if node_type is None:
                raise ValueError("异构图需要指定节点类型")
            dataloader = dgl.dataloading.DataLoader(
                self.graph,
                {node_type: seed_nodes},
                sampler,
                batch_size=len(seed_nodes)
            )
        else:
            dataloader = dgl.dataloading.DataLoader(
                self.graph,
                seed_nodes,
                sampler,
                batch_size=len(seed_nodes)
            )
            
        # 获取第一个批次
        for input_nodes, output_nodes, blocks in dataloader:
            return blocks
            
    def get_user_poi_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取用户-商家边
        
        Returns:
            (用户ID数组, 商家ID数组)
        """
        if self.graph is None:
            raise RuntimeError("请先加载图数据")
            
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            # 尝试找到用户-商家的边类型
            for etype in self.graph.canonical_etypes:
                if 'user' in etype[0].lower() and 'poi' in etype[2].lower():
                    src, dst = self.graph.edges(etype=etype)
                    return src.numpy(), dst.numpy()
            raise ValueError("未找到用户-商家边类型")
        else:
            # 同构图，返回所有边
            src, dst = self.graph.edges()
            return src.numpy(), dst.numpy()
            
    def export_adjacency_matrix(self, output_path: str, 
                               node_type: Optional[str] = None,
                               edge_type: Optional[Tuple] = None):
        """
        导出邻接矩阵（稀疏格式）
        
        Args:
            output_path: 输出文件路径
            node_type: 节点类型
            edge_type: 边类型
        """
        if self.graph is None:
            raise RuntimeError("请先加载图数据")
            
        import scipy.sparse as sp
        
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            if edge_type is None:
                raise ValueError("异构图需要指定边类型")
            adj = self.graph.adj(etype=edge_type, scipy_fmt='csr')
        else:
            adj = self.graph.adj(scipy_fmt='csr')
            
        sp.save_npz(output_path, adj)
        logger.info(f"邻接矩阵已保存到: {output_path}")


class GraphFeatureExtractor:
    """图特征提取器"""
    
    def __init__(self, graph: dgl.DGLGraph):
        """
        初始化特征提取器
        
        Args:
            graph: DGL 图对象
        """
        self.graph = graph
        
    def compute_node_degrees(self, node_type: Optional[str] = None) -> torch.Tensor:
        """
        计算节点度数
        
        Args:
            node_type: 节点类型
            
        Returns:
            度数张量
        """
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            if node_type is None:
                raise ValueError("异构图需要指定节点类型")
            return self.graph.in_degrees(etype=node_type)
        else:
            return self.graph.in_degrees()
            
    def compute_pagerank(self, num_iterations: int = 10,
                        node_type: Optional[str] = None) -> np.ndarray:
        """
        计算 PageRank 分数
        
        Args:
            num_iterations: 迭代次数
            node_type: 节点类型
            
        Returns:
            PageRank 分数数组
        """
        # 简化的 PageRank 实现
        if isinstance(self.graph, dgl.DGLHeteroGraph):
            if node_type is None:
                raise ValueError("异构图需要指定节点类型")
            num_nodes = self.graph.num_nodes(node_type)
        else:
            num_nodes = self.graph.num_nodes()
            
        # 初始化 PageRank 值
        pr = np.ones(num_nodes) / num_nodes
        damping = 0.85
        
        for _ in range(num_iterations):
            # 简化实现，实际应使用图的邻接矩阵
            pr_new = (1 - damping) / num_nodes + damping * pr
            pr = pr_new
            
        return pr


def main():
    """主函数 - 示例用法"""
    # 项目根目录
    project_root = Path(__file__).parent.parent
    graph_path = project_root / "Meituan_TRD" / "graph.bin"
    
    # 创建加载器
    loader = GraphDataLoader(graph_path)
    
    # 加载图
    graph = loader.load_graph()
    
    # 创建特征提取器
    # extractor = GraphFeatureExtractor(graph)
    
    logger.info("\n图数据加载接口测试完成！")


if __name__ == "__main__":
    main()
