import torch


def star(matrix, eta, known_rank=None):
    """Apply STAR SVD truncation with nuclear-norm preservation to a 2D matrix."""
    if matrix.ndim != 2:
        return matrix

    if known_rank is not None:
        # Low-rank SVD for speed if rank is known.
        u, s, v = torch.svd_lowrank(matrix, q=int(known_rank) + 5)
        s = s[: int(known_rank)]
    else:
        u, s, vt = torch.linalg.svd(matrix, full_matrices=False)

    if s.numel() == 0:
        return matrix

    sum_tot_s = torch.sum(s)
    cumulative_tot_s = torch.cumsum(s, dim=0)
    target = sum_tot_s * (eta / 100.0)
    rank_remain = torch.searchsorted(cumulative_tot_s, target).item() + 1
    rank_remain = max(1, min(rank_remain, s.shape[0]))

    u_truncated = u[:, :rank_remain]
    if known_rank is not None:
        v_truncated = v[:, :rank_remain]
    else:
        vt_truncated = vt[:rank_remain, :]
    remain_s = s[:rank_remain]

    sum_remain_s = torch.sum(remain_s)
    if sum_remain_s == 0:
        return matrix

    scaled_remain_s = (sum_tot_s / sum_remain_s) * remain_s

    if known_rank is not None:
        truncated_matrix = u_truncated @ torch.diag_embed(scaled_remain_s) @ v_truncated.T
    else:
        truncated_matrix = u_truncated @ torch.diag_embed(scaled_remain_s) @ vt_truncated

    return truncated_matrix


def compute_and_sum_star(task_vectors, config):
    """Apply STAR to each task vector and then average across tasks."""
    device = config.device
    eta = float(config.method.eta)
    if eta <= 1.0:
        eta = eta * 100.0
    known_rank = getattr(config.method, "known_rank", None)

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector.vector[key].to(device)
                if vec.ndim == 2 and "text_projection" not in key:
                    vec = star(vec, eta, known_rank=known_rank)

                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)

    return new_vector
