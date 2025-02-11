import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple
import imageio
import os


def save_residuals(tracker: dict, filepath: str = "residuals.pkl") -> None:
    """
    Save loss/residual trackers to a pickle file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(tracker, f)


def load_residuals(filepath: str = "residuals.pkl") -> dict:
    """
    Load loss/residual trackers from a pickle file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def plot_residuals(
    train_tracker: List[float],
    test_tracker: List[float],
    test_BCE_tracker: List[float],
    test_KLD_tracker: List[float],
    test_CEP_tracker: List[float],
    init_index: int = 0,
    save_fig_address: str = "./residuals.pdf",
) -> None:
    """
    Plot training and test losses.
    """
    plt.figure()
    plt.plot(train_tracker[init_index:], label="Training Total Loss")
    plt.plot(test_tracker[init_index:], label="Test Total Loss")
    plt.plot(test_BCE_tracker[init_index:], label="Test BCE Loss")
    plt.plot(test_KLD_tracker[init_index:], label="Test KLD Loss")
    plt.plot(test_CEP_tracker[init_index:], label="Test CEP Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Residuals")
    plt.legend()
    plt.savefig(save_fig_address)
    plt.show()


def get_equidistant_points(p1: np.ndarray, p2: np.ndarray, parts: int) -> List[Tuple[float, ...]]:
    """
    Returns equidistant points between two points.
    """
    return list(zip(*[np.linspace(p1[i], p2[i], parts + 1) for i in range(len(p1))]))


def sample_data_on_line(x0: torch.Tensor, x1: torch.Tensor, number_of_points: int) -> torch.Tensor:
    """
    Returns a set of equally spaced latent points along the line between x0 and x1.
    """
    delta = (x1 - x0) / (number_of_points - 1)
    points = torch.stack([x0 + i * delta for i in range(number_of_points)], dim=0)
    return points.cpu()


def save_gif(decoded_objects: np.ndarray, file_path_root: str, indicator: str, speed: int = 5) -> None:
    """
    Save a series of images as an animated GIF.
    """
    temp_filename = f"{file_path_root}{indicator}.png"
    images = []
    for array_ in decoded_objects:
        plt.imshow(array_, origin="lower", cmap="viridis")
        plt.colorbar(shrink=0.5)
        plt.savefig(temp_filename)
        images.append(imageio.imread(temp_filename))
        plt.clf()
    plt.close()
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    gif_filename = f"{file_path_root}{indicator}.gif"
    imageio.mimsave(gif_filename, images, fps=speed)


def latent_traversal(
    points_mean: np.ndarray,
    points_std: np.ndarray,
    start_id: int,
    end_id: int,
    k_neighbor_ratio: float = 0.1,
    distance_euclidean: bool = False,
    plot_results_2d: bool = True,
) -> List[int]:
    """
    Computes a latent traversal path (using a shortest path algorithm via igraph).
    """
    import igraph
    n_samples = points_mean.shape[0]
    k = int(0.05 * n_samples)
    dist_array = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if distance_euclidean:
                dist_array[i, j] = np.linalg.norm(points_mean[i] - points_mean[j])
            else:
                dist_array[i, j] = np.linalg.norm(points_mean[i] - points_mean[j])
    adjacency = (dist_array > 0).astype(int)
    g = igraph.Graph.Adjacency(adjacency.tolist())
    g.es["weight"] = dist_array[dist_array.nonzero()]
    path = g.get_shortest_paths(start_id, to=end_id, weights=g.es["weight"])
    if plot_results_2d:
        plt.scatter(points_mean[:, 0], points_mean[:, 1], s=1)
        path_ids = path[0]
        plt.plot(points_mean[path_ids, 0], points_mean[path_ids, 1], "-r")
        plt.scatter(points_mean[start_id, 0], points_mean[start_id, 1], c="green", s=10)
        plt.scatter(points_mean[end_id, 0], points_mean[end_id, 1], c="yellow", s=10)
        plt.title("Latent Traversal Path")
        plt.show()
    return path[0]


def calculate_lower_dimensions(latent_vectors: torch.Tensor, labels: np.ndarray, N: int = 1000):
    """
    Computes lower-dimensional embeddings using TSNE, UMAP, and PCA.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    latent_np = latent_vectors.cpu().detach().numpy()
    if latent_np.shape[0] > N:
        indices = np.random.choice(latent_np.shape[0], N, replace=False)
        X = latent_np[indices]
        Y = labels[indices]
    else:
        X = latent_np
        Y = labels
    tsne_proj = TSNE(n_components=3).fit_transform(X)
    umap_proj = umap.UMAP(random_state=42, n_components=latent_np.shape[1]).fit_transform(X)
    pca_proj = PCA(n_components=3).fit_transform(X)
    return tsne_proj, umap_proj, pca_proj, Y


def plot_lower_dimension(
    embedding: np.ndarray, labels, size_dot: int = 1, projection: str = "2d", save_str: str = "plot.pdf"
) -> None:
    """
    Plots a 2D or 3D projection of latent embeddings.
    """
    if projection == "2d":
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], s=size_dot, c=labels, cmap="viridis", marker=".")
    elif projection == "3d":
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=size_dot, c=labels, cmap="viridis", marker=".")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.title(f"{projection.upper()} Projection")
    plt.savefig(save_str)
    plt.show()


def generate_synthetic_data(trainer, num_additional_data: int, images_per_traversal: int = 20) -> np.ndarray:
    """
    Generates synthetic data by performing latent space linear traversal.
    """
    k = num_additional_data // images_per_traversal
    synthetic_data_all = []
    model = trainer.model
    device = trainer.device
    with torch.no_grad():
        for x, _ in trainer.testloader:
            x = x.to(device)
            _, _, _, _, z = model(x)
            latent_vectors = z.cpu()
            break
    if latent_vectors is None:
        raise ValueError("No latent vectors found.")
    for i in range(k):
        indices = np.random.choice(range(latent_vectors.size(0)), 2, replace=False)
        x0 = latent_vectors[indices[0]]
        x1 = latent_vectors[indices[1]]
        line = sample_data_on_line(x0, x1, images_per_traversal)
        decoded = model.decoder(line.to(device)).detach().cpu().numpy()
        synthetic_data_all.append(decoded)
    synthetic_data_all = np.concatenate(synthetic_data_all, axis=0)
    return synthetic_data_all
