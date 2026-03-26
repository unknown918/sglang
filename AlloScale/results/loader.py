import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_profile(
    filename,
    num_layers,
    max_tokens,
    max_iterations,
):
    with open(filename, "rb") as f:
        prefix_sum = np.fromfile(
            f,
            dtype=np.int32,
            count=max_iterations
        )

        data_count = num_layers * max_tokens * 6
        profile_data = np.fromfile(
            f,
            dtype=np.int32,
            count=data_count
        )
        profile_data = profile_data.reshape(
            num_layers, max_tokens, 6
        )

        time_count = num_layers * max_iterations
        profile_time = np.fromfile(
            f,
            dtype=np.float64,
            count=time_count
        )
        profile_time = profile_time.reshape(
            num_layers, max_iterations
        )

    return prefix_sum, profile_data, profile_time


if __name__ == "__main__":
    filename = "./profile_on_rank0.bin"

    num_layers = 10
    max_tokens = 1024 * 1024
    max_iterations = 100

    token_count, data, times = load_profile(
        filename,
        num_layers,
        max_tokens,
        max_iterations
    )

    layer_id = 1
    valid_iteration = max_iterations - 33

    # =========================
    # 时间（秒）
    # =========================
    times = np.round(times[layer_id][:valid_iteration] / 1000, 3)

    num_experts = 160

    # =========================
    # Step 1: iteration-level统计
    # =========================
    activation_per_iter = np.zeros((valid_iteration, num_experts), dtype=np.float32)

    cum = np.cumsum(token_count)
    cum = np.concatenate([[0], cum])

    for i in range(valid_iteration):
        start = cum[i]
        end = cum[i + 1]

        expert_ids = data[layer_id][start:end]
        flat_ids = expert_ids.reshape(-1)

        counts = np.bincount(flat_ids, minlength=num_experts)
        activation_per_iter[i] = counts

    # =========================
    # Step 2: 构建时间 bin
    # =========================
    num_bins = 400
    time_bins = np.linspace(times.min(), times.max(), num_bins + 1)

    # =========================
    # Step 3: iteration → bin
    # =========================
    bin_indices = np.digitize(times, time_bins) - 1

    # =========================
    # Step 4: 时间聚合
    # =========================
    activation_matrix = np.zeros((num_experts, num_bins), dtype=np.float32)
    token_sum = np.zeros(num_bins)

    for i in range(valid_iteration):
        b = bin_indices[i]

        if 0 <= b < num_bins:
            activation_matrix[:, b] += activation_per_iter[i]
            token_sum[b] += activation_per_iter[i].sum()

    # =========================
    # Step 5: token 归一化
    # =========================
    nonzero_mask = token_sum > 0
    token_sum_safe = token_sum.copy()
    token_sum_safe[token_sum_safe == 0] = 1

    activation_matrix = activation_matrix / token_sum_safe

    # =========================
    # ⭐ Step 6: 只保留非空 bin（核心）
    # =========================
    activation_matrix = activation_matrix[:, nonzero_mask]
    time_bins_left = time_bins[:-1][nonzero_mask]
    time_bins_right = time_bins[1:][nonzero_mask]
    token_sum = token_sum[nonzero_mask]

    num_bins = activation_matrix.shape[1]  # 更新 bin 数

    # =========================
    # Step 7: expert 归一化
    # =========================
    row_max = activation_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    activation_matrix = activation_matrix / row_max

    # =========================
    # Step 8: log scale（避免0问题）
    # =========================
    eps = 1e-6
    activation_matrix = np.log1p(activation_matrix + eps)

    # =========================
    # Step 9: 绘图
    # =========================
    plt.figure(figsize=(16, 8))

    ax = sns.heatmap(
        activation_matrix,
        cmap="viridis",
        cbar=True
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Expert ID")
    ax.set_title("MoE Expert Activation")

    # =========================
    # x 轴标签
    # =========================
    step = max(1, num_bins // 10)
    xticks = np.arange(0, num_bins, step)

    labels = [
        f"{time_bins_left[i]:.2f}-{time_bins_right[i]:.2f}"
        for i in xticks
    ]

    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # =========================
    # y 轴
    # =========================
    yticks = np.arange(0, num_experts, 16)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels(yticks)

    plt.tight_layout()
    # plt.show()
    plt.savefig("profile.pdf", format="pdf")
