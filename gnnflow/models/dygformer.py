import numpy as np
import torch
import torch.nn as nn

from gnnflow.models.modules.layers import EdgePredictor
from gnnflow.models.modules.time_encoder import TimeEncoder
from gnnflow.dygformer_sampler import NeighborSampler


class NeighborCooccurrenceEncoder(nn.Module):
    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str):
        super().__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray,
                dst_padded_nodes_neighbor_ids: np.ndarray):
        src = src_padded_nodes_neighbor_ids
        dst = dst_padded_nodes_neighbor_ids
        src_feats = np.zeros((src.shape[0], src.shape[1], self.neighbor_co_occurrence_feat_dim), dtype=np.float32)
        dst_feats = np.zeros((dst.shape[0], dst.shape[1], self.neighbor_co_occurrence_feat_dim), dtype=np.float32)
        for i in range(src.shape[0]):
            src_set = set(src[i].tolist())
            dst_set = set(dst[i].tolist())
            inter = src_set.intersection(dst_set)
            if len(inter) == 0:
                continue
            for j in range(src.shape[1]):
                if src[i, j] in inter:
                    src_feats[i, j, 0] = 1.0
            for j in range(dst.shape[1]):
                if dst[i, j] in inter:
                    dst_feats[i, j, 0] = 1.0
        return torch.from_numpy(src_feats).to(self.device), torch.from_numpy(dst_feats).to(self.device)


class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(attention_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 4, attention_dim),
        )
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class DyGFormer(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 neighbor_sampler: NeighborSampler, time_feat_dim: int,
                 channel_embedding_dim: int, patch_size: int = 1,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 max_input_sequence_length: int = 512, device: str = 'cpu'):
        super().__init__()
        self.is_dygformer = True
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
            device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(self.patch_size * self.node_feat_dim, self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(self.patch_size * self.edge_feat_dim, self.channel_embedding_dim, bias=True),
            'time': nn.Linear(self.patch_size * self.time_feat_dim, self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(self.patch_size * self.neighbor_co_occurrence_feat_dim, self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4
        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim,
                               num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(self.num_channels * self.channel_embedding_dim, self.node_feat_dim, bias=True)
        self.edge_predictor = EdgePredictor(self.node_feat_dim)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times)
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times)

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(src_node_ids, node_interact_times,
                               src_nodes_neighbor_ids_list, src_nodes_edge_ids_list,
                               src_nodes_neighbor_times_list,
                               patch_size=self.patch_size,
                               max_input_sequence_length=self.max_input_sequence_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(dst_node_ids, node_interact_times,
                               dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list,
                               dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size,
                               max_input_sequence_length=self.max_input_sequence_length)

        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids)

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times,
                              padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                              padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                              time_encoder=self.time_encoder)

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
            src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(src_padded_nodes_neighbor_node_raw_features,
                             src_padded_nodes_edge_raw_features,
                             src_padded_nodes_neighbor_time_features,
                             src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(dst_padded_nodes_neighbor_node_raw_features,
                             dst_padded_nodes_edge_raw_features,
                             dst_padded_nodes_neighbor_time_features,
                             dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](
            src_patches_nodes_neighbor_co_occurrence_features)

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](
            dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        patches_nodes_neighbor_node_raw_features = torch.cat(
            [src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat(
            [src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat(
            [src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat(
            [src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        patches_data = torch.stack(patches_data, dim=2)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches,
                                            self.num_channels * self.channel_embedding_dim)

        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        src_patches_data = patches_data[:, : src_num_patches, :]
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray,
                     padded_nodes_edge_ids: np.ndarray, padded_nodes_neighbor_times: np.ndarray,
                     time_encoder: TimeEncoder):
        node_raw_features = self.node_raw_features[padded_nodes_neighbor_ids]
        edge_raw_features = self.edge_raw_features[padded_nodes_edge_ids]
        time_diffs = node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
        time_raw_features = time_encoder(torch.from_numpy(time_diffs).to(self.device))
        return node_raw_features, edge_raw_features, time_raw_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor,
                    padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor,
                    padded_nodes_neighbor_co_occurrence_features: torch.Tensor,
                    patch_size: int = 1):
        batch_size, max_seq_len, _ = padded_nodes_neighbor_node_raw_features.shape
        num_patches = max_seq_len // patch_size
        def to_patches(x):
            x = x.reshape(batch_size, num_patches, patch_size, -1)
            return x.reshape(batch_size, num_patches, patch_size * x.shape[-1])
        return (to_patches(padded_nodes_neighbor_node_raw_features),
                to_patches(padded_nodes_edge_raw_features),
                to_patches(padded_nodes_neighbor_time_features),
                to_patches(padded_nodes_neighbor_co_occurrence_features))

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                      nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1,
                      max_input_sequence_length: int = 256):
        assert max_input_sequence_length - 1 > 0
        max_seq_length = 0
        for idx in range(len(nodes_neighbor_ids_list)):
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                n = len(nodes_neighbor_ids_list[idx])
                padded_nodes_neighbor_ids[idx, 1: n + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: n + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: n + 1] = nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def edge_predict_from_batch(self, target_nodes: np.ndarray, ts: np.ndarray):
        num_edges = len(target_nodes) // 3
        src = target_nodes[:num_edges]
        dst = target_nodes[num_edges:2 * num_edges]
        neg = target_nodes[2 * num_edges:]
        ts_pos = ts[:num_edges]

        src_emb, dst_emb = self.compute_src_dst_node_temporal_embeddings(src, dst, ts_pos)
        neg_emb, _ = self.compute_src_dst_node_temporal_embeddings(neg, neg, ts_pos)

        h = torch.cat([src_emb, dst_emb, neg_emb], dim=0)
        return self.edge_predictor(h, neg_samples=1)
