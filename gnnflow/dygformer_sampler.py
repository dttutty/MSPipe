import numpy as np


class NeighborSampler:
    def __init__(self, src_nodes: np.ndarray, dst_nodes: np.ndarray,
                 edge_ids: np.ndarray, timestamps: np.ndarray,
                 num_nodes: int):
        self.num_nodes = num_nodes
        self._neighbors = [[] for _ in range(num_nodes)]
        for s, d, e, t in zip(src_nodes, dst_nodes, edge_ids, timestamps):
            self._neighbors[int(s)].append((t, d, e))
            self._neighbors[int(d)].append((t, s, e))

        self._times = []
        self._nbrs = []
        self._eids = []
        for entries in self._neighbors:
            if not entries:
                self._times.append(np.array([], dtype=np.float32))
                self._nbrs.append(np.array([], dtype=np.int64))
                self._eids.append(np.array([], dtype=np.int64))
                continue
            entries.sort(key=lambda x: x[0])
            times = np.array([e[0] for e in entries], dtype=np.float32)
            nbrs = np.array([e[1] for e in entries], dtype=np.int64)
            eids = np.array([e[2] for e in entries], dtype=np.int64)
            self._times.append(times)
            self._nbrs.append(nbrs)
            self._eids.append(eids)

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray,
                                    node_interact_times: np.ndarray):
        neighbor_ids_list = []
        edge_ids_list = []
        neighbor_times_list = []
        for node_id, ts in zip(node_ids, node_interact_times):
            node_id = int(node_id)
            times = self._times[node_id]
            if times.size == 0:
                neighbor_ids_list.append(np.array([], dtype=np.int64))
                edge_ids_list.append(np.array([], dtype=np.int64))
                neighbor_times_list.append(np.array([], dtype=np.float32))
                continue
            idx = np.searchsorted(times, ts, side="left")
            neighbor_ids_list.append(self._nbrs[node_id][:idx])
            edge_ids_list.append(self._eids[node_id][:idx])
            neighbor_times_list.append(times[:idx])

        return neighbor_ids_list, edge_ids_list, neighbor_times_list
