import numpy as np

def remove_pc0(data):
    """
    Remove the effect of the first PCA component (PC0) from a batch of samples.
    data: array of shape (N, H, W)
    
    Returns:
        data_clean: same shape, PC0 removed
    """

    # Flatten each sample
    N = data.shape[0]
    X = data.reshape(N, -1)

    # Center the data
    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean

    # PCA
    U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
    pc0_vector = Vh[0, :]
    pc0_scores = U[:, 0] * S[0]

    # Contribution of PC0 to each sample
    pc0_contribution = np.outer(pc0_scores, pc0_vector)

    # Remove PC0
    X_clean = X_centered - pc0_contribution

    # Add back the mean to stay in original feature space
    X_clean += X_mean

    # Reshape back to original
    data_clean = X_clean.reshape(data.shape)

    return data_clean