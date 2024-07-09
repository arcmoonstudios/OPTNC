# root/data_processing/augmentor.py
# Implements data augmentation techniques to increase dataset diversity

import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import QuantileTransformer

class DataAugmentor:
    def __init__(self, noise_level=0.05, time_warp_factor=0.2):
        self.noise_level = noise_level
        self.time_warp_factor = time_warp_factor
        self.qt = QuantileTransformer(output_distribution='normal')

    def add_noise(self, data):
        # Add random noise to the data
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise

    def time_warp(self, data):
        # Apply time warping to the data
        original_steps = np.arange(data.shape[1])
        warped_steps = np.linspace(0, data.shape[1] - 1, num=data.shape[1])
        warped_steps += np.random.normal(0, self.time_warp_factor, size=warped_steps.shape)
        warped_steps = np.sort(warped_steps)
        warped_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            interpolator = interp1d(original_steps, data[i, :], kind='linear', fill_value='extrapolate')
            warped_data[i, :] = interpolator(warped_steps)
        return warped_data

    def mixup(self, data, labels, alpha=0.2):
        # Implement mixup augmentation
        indices = np.random.permutation(len(data))
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]
        lam = np.random.beta(alpha, alpha, len(data))
        lam = np.max([lam, 1 - lam], axis=0)
        mixed_data = lam.reshape(-1, 1) * data + (1 - lam.reshape(-1, 1)) * shuffled_data
        mixed_labels = lam.reshape(-1, 1) * labels + (1 - lam.reshape(-1, 1)) * shuffled_labels
        return mixed_data, mixed_labels

    def generate_synthetic_samples(self, data, n_samples):
        # Generate synthetic samples using quantile transformation
        synthetic_data = self.qt.fit_transform(data)
        synthetic_data = np.random.multivariate_normal(
            mean=np.mean(synthetic_data, axis=0),
            cov=np.cov(synthetic_data, rowvar=False),
            size=n_samples
        )
        return self.qt.inverse_transform(synthetic_data)

    def augment(self, data, labels=None, augmentation_factor=2):
        # Combine multiple augmentation techniques
        augmented_data = [data]
        augmented_labels = [labels] if labels is not None else None
        
        for _ in range(augmentation_factor - 1):
            aug_data = self.add_noise(data)
            aug_data = self.time_warp(aug_data)
            augmented_data.append(aug_data)
            if labels is not None:
                augmented_labels.append(labels)
        
        augmented_data = np.concatenate(augmented_data, axis=0)
        
        if labels is not None:
            augmented_labels = np.concatenate(augmented_labels, axis=0)
            augmented_data, augmented_labels = self.mixup(augmented_data, augmented_labels)
        
        synthetic_samples = self.generate_synthetic_samples(data, len(data) * (augmentation_factor - 1))
        augmented_data = np.concatenate([augmented_data, synthetic_samples], axis=0)
        
        if labels is not None:
            synthetic_labels = np.repeat(labels, augmentation_factor - 1, axis=0)
            augmented_labels = np.concatenate([augmented_labels, synthetic_labels], axis=0)
            return augmented_data, augmented_labels
        
        return augmented_data
