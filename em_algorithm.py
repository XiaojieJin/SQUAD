import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import norm

# Read CSV file
df = pd.read_csv('/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/obs_result_fov124.csv')

# Store original cell order
original_cell_order = df['fov_cell'].values

# Verify that cell names are unique
if len(original_cell_order) != len(set(original_cell_order)):
    raise ValueError("Warning: Duplicate cell names found in the data!")

# Replace "doublet" with 1 and "not doublet" with 0 in scrublet_1 column
# df['scrublet_1'] = df['scrublet_1'].map({'doublet': 1, 'not doublet': 0})

# Select feature columns for analysis
feature_columns = ['EOR', 'cell_size_score', 'sensitivity_2', 'saturation_2', 
                  'solidity', 'circularity', 'sgcc', 'scrublet_1']

invert_features = ['EOR','scrublet_1']  # Add other features that need inversion if needed

# Verify all feature columns exist in the dataframe
missing_columns = [col for col in feature_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in data: {missing_columns}")

# Preprocess features that need inversion
for feature in invert_features:
    df[feature] = -df[feature]

# Preprocess cell_size_score (convert to negative absolute deviation from mean)
mean_size = df['cell_size_score'].mean()
df['cell_size_score'] = -np.abs(df['cell_size_score'] - mean_size)

# Perform MinMax normalization for all features
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])


# Calculate number of rows needed for subplots
n_features = len(feature_columns)
total_plots = n_features + 2  # Add 2 for meta_score and meta_score_pca
n_rows = (total_plots + 2) // 3  # Calculate number of rows needed

# Create figures for both distributions
plt.figure(figsize=(15, 4*n_rows))  # Figure 1: Probability distributions
fig_raw = plt.figure(figsize=(15, 4*((n_features + 2)//3)))  # Figure 2: Raw data with GMM fits

# Create a dictionary to store results for each feature
results = {}

# Process each feature separately
for idx, feature in enumerate(feature_columns):
    print(f"\nProcessing feature: {feature}")
    
    # Extract feature data and verify no NaN values
    X = df[feature].values
    if np.any(np.isnan(X)):
        print(f"Warning: NaN values found in feature: {feature}, replacing with 0")
        X = np.nan_to_num(X, nan=0)
    
    X = X.reshape(-1, 1)  # Reshape to 2D array for GMM
    
    # Create and train GMM model
    n_components = 2  # Use 2 Gaussian distributions
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Calculate probabilities for each sample belonging to each Gaussian distribution
    probs = gmm.predict_proba(X)
    
    # Get probabilities for each sample belonging to the main Gaussian distribution
    main_component_probs = probs[:, np.argmax(gmm.weights_)]
    
    # Verify probability calculation didn't change the order
    if len(main_component_probs) != len(df):
        raise ValueError(f"Length mismatch in probability calculation for feature: {feature}")
    
    # Add results to the original dataframe
    df[f'{feature}_prob'] = main_component_probs
    
    # Store results
    results[feature] = {
        'weights': gmm.weights_,
        'means': gmm.means_,
        'covariances': gmm.covariances_,
        'main_component_prob': main_component_probs
    }
    
    # Print statistical information
    print(f"Main Gaussian component weight: {gmm.weights_[np.argmax(gmm.weights_)]:.4f}")
    print(f"Average conditional probability: {np.mean(main_component_probs):.4f}")
    print(f"Standard deviation of conditional probability: {np.std(main_component_probs):.4f}")
    
    # Plot probability distribution (Figure 1)
    plt.figure(1)
    plt.subplot(n_rows, 3, idx + 1)
    plt.hist(main_component_probs, bins=50, alpha=0.7, density=True)
    plt.title(f'Distribution of {feature}_locfdr')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    
    # Add mean and median text
    mean_val = np.mean(main_component_probs)
    median_val = np.median(main_component_probs)
    plt.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    # Plot raw data distribution with GMM fits (Figure 2)
    plt.figure(2)
    plt.subplot((n_features + 2)//3, 3, idx + 1)
    
    # Plot histogram of raw data
    hist, bins, _ = plt.hist(X, bins=50, density=True, alpha=0.7, label='Raw Data')
    
    # Generate points for the Gaussian curves
    x = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    
    # Plot each Gaussian component
    for i in range(n_components):
        if i == np.argmax(gmm.weights_):
            color = 'orange'  # 主要高斯分布用橘色
        else:
            color = 'green'   # 次要高斯分布用绿色
        pdf = norm.pdf(x, gmm.means_[i], np.sqrt(gmm.covariances_[i][0])) * gmm.weights_[i]
        plt.plot(x, pdf, '--', color=color, label=f'Gaussian {i+1}')
    
    # Plot the sum of Gaussians
    total_pdf = np.zeros_like(x)
    for i in range(n_components):
        total_pdf += norm.pdf(x, gmm.means_[i], np.sqrt(gmm.covariances_[i][0])) * gmm.weights_[i]
    plt.plot(x, total_pdf, 'r-', label='GMM fit')
    
    plt.title(f'Raw Distribution of {feature}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

# Calculate meta_score as the product of conditional probabilities
prob_columns = [f'{feature}_prob' for feature in feature_columns]
meta_score = df[prob_columns].prod(axis=1)
df['meta_score'] = meta_score

# Calculate meta_score_pca using PCA
# Get probability matrix
prob_matrix = df[prob_columns].values

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(prob_matrix)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Find number of components needed to explain >90% variance
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
print(f"\nNumber of components needed to explain >90% variance: {n_components_90}")
print("Explained variance ratios:", pca.explained_variance_ratio_)

# Calculate meta_score_pca as the mean of selected components
meta_score_pca = np.mean(pca_result[:, :n_components_90], axis=1)

# Normalize meta_score_pca to [0,1]
scaler = MinMaxScaler()
meta_score_pca_normalized = scaler.fit_transform(meta_score_pca.reshape(-1, 1)).flatten()
df['meta_score_pca'] = meta_score_pca_normalized

# Plot meta_score distribution (Figure 1)
plt.figure(1)
plt.subplot(n_rows, 3, n_features + 1)
plt.hist(meta_score, bins=50, alpha=0.7, density=True)
plt.title('Distribution of meta_score')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Add mean and median text for meta_score
mean_val = np.mean(meta_score)
median_val = np.median(meta_score)
plt.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

# Plot meta_score_pca distribution
plt.figure(1)
plt.subplot(n_rows, 3, n_features + 2)
plt.hist(meta_score_pca_normalized, bins=50, alpha=0.7, density=True)
plt.title('Distribution of meta_score_pca')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# Add mean and median text for meta_score_pca
mean_val = np.mean(meta_score_pca_normalized)
median_val = np.median(meta_score_pca_normalized)
plt.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

# Adjust layout for both figures
plt.figure(1)
plt.tight_layout()
plt.savefig('/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/probability_distributions4.png', dpi=300, bbox_inches='tight')

plt.figure(2)
plt.tight_layout()
plt.savefig('/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/raw_distributions_with_gmm4.png', dpi=300, bbox_inches='tight')

plt.close('all')

# Select only the original features and their corresponding probability columns
output_columns = ['fov_cell']  # Start with cell name column
for feature in feature_columns:
    output_columns.extend([feature, f'{feature}_prob'])
output_columns.extend(['meta_score', 'meta_score_pca'])

# Add quality column based on meta_score_pca
################################################################
# df['quality'] = (df['meta_score_pca'] > 0.6).astype(int)
################################################################
df['quality'] = (df['meta_score'] > 0.01).astype(int)
output_columns.append('quality')

# Verify final cell order matches original order
if not np.array_equal(df['fov_cell'].values, original_cell_order):
    raise ValueError("Cell order has changed during processing!")

# Save selected columns to CSV file
df[output_columns].to_csv('/fs/ess/PAS1475/Xiaojie/spatialQC/test_data/obs_result_fov122_with_probs42.csv', index=False)

# Print verification message
print("\nVerification Summary:")
print(f"Number of cells processed: {len(df)}")
print(f"Number of features processed: {len(feature_columns)}")
print("All cell-score correspondences maintained") 