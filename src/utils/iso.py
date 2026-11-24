import torch


def iso_c(task_vectors, config):
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            tvs = [task_vector.vector[key].to(device) for task_vector in task_vectors]
            new_vector[key] = sum(tvs) / len(tvs)

            is_2d_matrix = len(task_vectors[0].vector[key].shape) == 2
            if is_2d_matrix and "text_projection" not in key:
                new_vector[key] *= len(tvs)
                u, s, v = torch.linalg.svd(new_vector[key], full_matrices=False)
                s_mean = torch.ones_like(s) * s.mean()

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u,
                        torch.diag(s_mean),
                        v,
                    )
                )

    return new_vector


@torch.no_grad()
def iso_cts(task_vectors, config):
    device = config.device
    new_vector = {}

    print("Computing SVD...")
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape
        is_2d_matrix = len(shape_) == 2 and "text_projection" not in key

        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector.vector[key].to(device)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            continue

        print(f"Computing common space using sum for {key}...")
        combined_w = sum(task_vector.vector[key].to(device) for task_vector in task_vectors)

        common_space_index_s = int(min(shape_) * config.method.common_space_fraction)
        task_specific_total_space = round(
            (min(shape_) - common_space_index_s) / len(config.DATASETS)
        ) * len(config.DATASETS)
        common_space_index_s = min(shape_) - task_specific_total_space

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]

        n_dims_per_task = int((min(shape_) - common_space_index_s) / len(config.DATASETS))
        for i, task_vector in enumerate(task_vectors):
            w = task_vector.vector[key].to(device)
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=device)
                combined_space_s = torch.zeros_like(s_ts, device=device)
                combined_space_v = torch.zeros_like(v_ts, device=device)

            start = i * n_dims_per_task
            end = (i + 1) * n_dims_per_task
            combined_space_u[:, start:end] = u_ts[:, :n_dims_per_task]
            combined_space_s[start:end] = s_ts[:n_dims_per_task]
            combined_space_v[start:end, :] = v_ts[:n_dims_per_task, :]

        start = len(config.DATASETS) * n_dims_per_task
        end = start + common_space_index_s
        combined_space_u[:, start:end] = common_space_u
        combined_space_s[start:end] = common_space_s
        combined_space_v[start:end, :] = common_space_v

        u_combined_u, _, v_combined_u = torch.linalg.svd(
            combined_space_u, full_matrices=False
        )
        u_combined_v, _, v_combined_v = torch.linalg.svd(
            combined_space_v, full_matrices=False
        )
        combined_space_u = u_combined_u @ v_combined_u
        combined_space_v = u_combined_v @ v_combined_v

        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        new_vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )

    return new_vector
