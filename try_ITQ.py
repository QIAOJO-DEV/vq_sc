import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from collections import Counter
import itertools
import torch
from torchvision import transforms
from PIL import Image
import lpips
from py_lightning_code.modules.CNNVQGAN import VQVAE
from py_lightning_code.utils.general import get_config_from_file, initialize_from_config, setup_callbacks

def itq_binary_codes(X, m=13, n_iter=20, random_state=42):
    """
    ITQ: PCA -> iterative quantization
    X: (n, d) float array, assumed real-valued features
    m: number of bits
    Returns:
      B: (n, m) in {-1, +1}
      S: (n, m) continuous projections Y @ R (before sign)
      PCA_obj: fitted PCA (scikit-learn)
      R: final rotation matrix (m, m)
    """
    n, d = X.shape
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    # PCA to m dims
    pca = PCA(n_components=m, svd_solver='auto', random_state=random_state)
    Y = pca.fit_transform(Xc)  # (n, m)

    # initialize R as random orthogonal or identity
    rng = np.random.RandomState(random_state)
    # random orthogonal via QR
    A = rng.randn(m, m)
    Q, _ = np.linalg.qr(A)
    R = Q

    for it in range(n_iter):
        Z = Y.dot(R)               # (n, m)
        B = np.where(Z >= 0, 1.0, -1.0)  # {-1, +1}
        # solve orthogonal Procrustes: R = UV^T where USV^T = Y^T B
        M = Y.T.dot(B)            # (m, m)
        U, _, Vt = np.linalg.svd(M)
        R = U.dot(Vt)
    S = Y.dot(R)
    B_final = np.where(S >= 0, 1.0, -1.0)
    return B_final, S, pca, R

def bits_to_ints(bits01, bit_order='msb_left'):
    """
    bits01: (n, m) array of 0/1
    bit_order: 'msb_left' means bits01[:,0] is MSB, bits01[:,-1] is LSB
    returns ints array shape (n,)
    """
    n, m = bits01.shape
    if bit_order == 'msb_left':
        powers = (1 << np.arange(m-1, -1, -1)).astype(np.int64)
    else:
        powers = (1 << np.arange(0, m)).astype(np.int64)
    return (bits01.astype(np.int64) * powers).sum(axis=1)

def int_to_pm1_vec(k, m):
    """convert integer 0..2^m-1 to +/-1 vector length m (msb_left)"""
    bits = np.array(list(np.binary_repr(k, width=m))).astype(np.int8)
    # bits are '0'/'1' as ascii digits; convert
    bits = (bits - ord('0')) if isinstance(bits[0], np.str_) else bits
    # but simpler:
    # we'll build via bit ops
    v = np.empty(m, dtype=np.int8)
    for i in range(m):
        shift = m - 1 - i
        v[i] = 1 if ((k >> shift) & 1) else -1
    return v

def int_to_pm1_matrix(idxs, m):
    """vectorized conversion for array of ints -> (len(idxs), m) of +/-1"""
    idxs = np.asarray(idxs, dtype=np.int64)
    n = idxs.shape[0]
    mat = np.empty((n, m), dtype=np.int8)
    for i in range(m):
        shift = m - 1 - i
        mat[:, i] = ((idxs >> shift) & 1) * 2 - 1
    return mat

def neighbors_by_hamming(code_int, m, radius):
    """generate ints within Hamming distance <= radius from code_int (including itself).
       For small radius (<=3) it's OK.
    """
    neighbors = []
    bits_positions = list(range(m))
    for r in range(radius+1):
        for comb in itertools.combinations(bits_positions, r):
            new = code_int
            for pos in comb:
                shift = m - 1 - pos
                new ^= (1 << shift)
            neighbors.append(new)
    return neighbors

def local_unique_assignment(S, bits01, m, max_radius=2):
    """
    Ensure unique integer codes for each row by local reassignment.
    S: (n, m) continuous projections (float), used to compute distances to +/-1 vertices.
    bits01: (n, m) initial 0/1 bits (may have collisions)
    m: number of bits
    max_radius: initial Hamming radius to search for replacement vertices (increase if needed)
    Returns:
      bits01_unique, ints_unique
    """
    n = S.shape[0]
    ints = bits_to_ints(bits01, bit_order='msb_left')
    counts = Counter(ints.tolist())
    dup_codes = [code for code, cnt in counts.items() if cnt > 1]
    assigned = {}  # int -> index assigned
    ints_unique = -np.ones(n, dtype=np.int64)

    # First assign all unique codes straight away
    for i in range(n):
        code = ints[i]
        if counts[code] == 1:
            ints_unique[i] = code
            assigned[code] = i

    # Set of available code ints
    all_codes = set(range(1 << m))
    available = all_codes - set(assigned.keys())

    # Process each duplicated code group
    # We'll sort groups by size descending to handle big groups first
    dup_groups = {}
    for code in dup_codes:
        dup_groups[code] = np.where(ints == code)[0]

    groups_sorted = sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))

    for code, idxs in groups_sorted:
        # For this group, we will find a candidate set = union of neighbors within radius
        radius = 0
        candidate_set = set()
        while radius <= max_radius and len(candidate_set) < len(idxs):
            # expand neighbors
            for idx in idxs:
                neighs = neighbors_by_hamming(code, m, radius)
                for nb in neighs:
                    if nb in available:
                        candidate_set.add(nb)
            if len(candidate_set) >= len(idxs):
                break
            radius += 1

        if len(candidate_set) < len(idxs):
            # still not enough candidates: expand more for this group up to radius=m
            for r2 in range(max_radius+1, m+1):
                for idx in idxs:
                    neighs = neighbors_by_hamming(code, m, r2)
                    for nb in neighs:
                        if nb in available:
                            candidate_set.add(nb)
                if len(candidate_set) >= len(idxs):
                    radius = r2
                    break

        candidate_list = sorted(list(candidate_set))
        if len(candidate_list) < len(idxs):
            # worst-case fallback: use nearest available among ALL available codes (rare)
            candidate_list = sorted(list(available))

        # Build cost matrix: rows = idxs (points), cols = candidate_list (vertices)
        cand_mat = int_to_pm1_matrix(candidate_list, m).astype(np.float32)  # values ±1
        points = S[idxs].astype(np.float32)  # continuous projections
        # costs = squared euclidean distance between points and cand_mat
        # cost_ij = ||points[i] - cand_mat[j]||^2 = ||p||^2 + ||c||^2 - 2 p·c
        p_norm2 = (points ** 2).sum(axis=1)[:, None]  # (g,1)
        c_norm2 = (cand_mat ** 2).sum(axis=1)[None, :]  # (1, k)
        dot = points.dot(cand_mat.T)  # (g, k)
        costs = p_norm2 + c_norm2 - 2.0 * dot
        # Solve assignment (Hungarian). If more candidates than points, just take first k cols? No: use hungarian on rectangular matrix.
        row_ind, col_ind = linear_sum_assignment(costs)
        # Note: linear_sum_assignment gives min cost matching of size = min(g,k) assigning each row to one col.
        # We need to ensure every row gets assigned; if k < g (rare), we used fallback earlier to ensure k>=g.
        for r_idx, c_idx in zip(row_ind, col_ind):
            point_idx = idxs[r_idx]
            assigned_code = candidate_list[c_idx]
            ints_unique[point_idx] = assigned_code
            # update availability
            if assigned_code in available:
                available.remove(assigned_code)
            assigned[assigned_code] = point_idx

    # For any still-unassigned (shouldn't happen), assign arbitrary available codes
    unassigned_points = np.where(ints_unique < 0)[0]
    if len(unassigned_points) > 0:
        av = sorted(list(available))
        if len(av) < len(unassigned_points):
            raise RuntimeError("Not enough available codes to assign uniquely.")
        for i, pidx in enumerate(unassigned_points):
            code = av[i]
            ints_unique[pidx] = code
            available.remove(code)
            assigned[code] = pidx

    # Build bits01 from ints_unique
    bits01_unique = ((int_to_pm1_matrix(ints_unique, m) + 1) // 2).astype(np.int8)
    return bits01_unique, ints_unique

# -----------------------
# Top-level wrapper
# -----------------------
def compute_unique_itq_codes(X, m=13, n_iter=50, max_radius=2, verbose=True):
    """
    X: (n, d) codebook
    Returns:
      bits01_unique: (n, m) 0/1
      ints_unique: (n,)
      stats: dict with collision counts etc.
    """
    n, d = X.shape
    B_pm1, S, pca, R = itq_binary_codes(X, m=m, n_iter=n_iter)
    bits01 = ((B_pm1 + 1) // 2).astype(np.int8)  # 0/1

    ints = bits_to_ints(bits01, bit_order='msb_left')
    cnt = Counter(ints.tolist())
    n_collisions = sum(v-1 for v in cnt.values() if v > 1)
    unique_before = len([c for c in cnt.values() if c == 1])
    if verbose:
        print(f"ITQ produced {len(cnt)} distinct codes out of {1<<m} possible.")
        print(f"Collisions (duplicate assignments) count = {n_collisions}. Unique rows before fix: {unique_before}/{n}")

    # If no collisions, return directly
    if n_collisions == 0 and len(cnt) == n:
        return bits01, ints, {'collisions_before':0, 'distinct_before':len(cnt)}

    # Else run local unique assignment to resolve collisions
    bits01_unique, ints_unique = local_unique_assignment(S, bits01, m, max_radius=max_radius)
    cnt_after = Counter(ints_unique.tolist())
    if verbose:
        print(f"After local assignment: distinct codes = {len(cnt_after)} (should equal n={n}).")
        dup_after = sum(v-1 for v in cnt_after.values() if v>1)
        print(f"Collisions after fix: {dup_after}")
    stats = {'collisions_before': n_collisions, 'distinct_before': len(cnt), 'distinct_after': len(cnt_after)}
    return bits01_unique, ints_unique, stats
def calculate_matrix_distance(x,bits):
    dist_x=torch.cdist(x,x)
    bits_i = bits.unsqueeze(1)  # (8192,1,13)
    bits_j = bits.unsqueeze(0)  # (1,8192,13)
    dist_matrix = (bits_i ^ bits_j).sum(dim=-1)
    alpha = dist_x.mean() / dist_matrix.float().mean()
    loss = torch.mean(torch.abs(dist_x - dist_matrix.float() * alpha))
    return loss
def reorder_codebook_by_index(codebook_tensor, ints_unique):
    """
    返回新的码本张量 new_codebook，其中 new_codebook[i] = 原始码本向量
    对应 ints_unique == i 的向量。
    """
    n_codes, d = codebook_tensor.shape
    new_codebook = torch.empty_like(codebook_tensor)

    ints_tensor = torch.from_numpy(ints_unique).long()  # 确保 PyTorch 索引类型
    new_codebook[ints_tensor] = codebook_tensor

    return new_codebook

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
        # -------------------- 1. 加载 VQ-VAE 码本 --------------------
    ckpt_path = '/home/data/haoyi_project/vq_sc/checkpoints/cnn_w_error_0.01_top_500_channel_loss-epoch=880.ckpt'
    config_path = '/home/data/haoyi_project/vq_sc/config/control_cnn_w_error_0.01_top_500_channel_loss.yaml'

    config = get_config_from_file(config_path)
    vqvae = initialize_from_config(config.model)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    vqvae.load_state_dict(ckpt['state_dict'])
    vqvae.eval()
    codebook_tensor = vqvae.model.quantize_b.embedding.weight.detach()  # (8192, 64)
    n_codes = codebook_tensor.shape[0]
    n_bits = 10
    indices = torch.arange(n_codes, dtype=torch.int32).unsqueeze(1)  # (8192,1)
    bits = ((indices >> torch.arange(n_bits)) & 1).to(torch.uint8)   # (8192, 13)

    initial_loss = calculate_matrix_distance(codebook_tensor, bits)
    print("初始汉明矩阵和欧式距离矩阵的距离:", initial_loss.item())
    X_np = codebook_tensor.numpy()
    bits01_unique, ints_unique, stats = compute_unique_itq_codes(
        X_np, m=n_bits, n_iter=20, max_radius=2, verbose=True
    )
    new_codebook = reorder_codebook_by_index(codebook_tensor, ints_unique)
    a=calculate_matrix_distance(codebook_tensor, torch.from_numpy(bits01_unique))
    new_loss = calculate_matrix_distance(new_codebook, bits)
    print("重新分配后的汉明矩阵和欧式距离矩阵的距离:", new_loss.item())
    torch.save(new_codebook, '/home/data/haoyi_project/vq_sc/reassign_codebook/cnn_w_error_0.01_top_500_channel_loss-epoch=880_codebook_b.pt')
