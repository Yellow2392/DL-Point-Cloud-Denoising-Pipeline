import numpy as np
import open3d as o3d

def knn(query, reference, k=1):
    # Búsqueda vectorizada de los k vecinos más cercanos de 'query' dentro de 'reference'
    reference = np.ascontiguousarray(reference, dtype=np.float64)
    query = np.ascontiguousarray(query, dtype=np.float64)

    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(reference))
    nns.knn_index()
    idx, dist2 = nns.knn_search(o3d.core.Tensor(query), k)

    idx = idx.numpy().astype(np.int64)
    dist = np.sqrt(dist2.numpy())

    if k == 1:
        return idx[:, 0], dist[:, 0]
    return idx, dist
