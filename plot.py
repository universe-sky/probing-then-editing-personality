import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

def plot_embeddings(pos_embds, neg_embds, ext_embds, output_dir='./'):
    """
    Visualize the embeddings of different personality classes using t-SNE (t-distributed Stochastic Neighbor Embedding).
    
    Args:
        pos_embds: Embeddings for the 'agreeableness' class.
        neg_embds: Embeddings for the 'neuroticism' class.
        ext_embds: Embeddings for the 'extraversion' class.
        output_dir: Directory to save the visualized embeddings in PNG and PDF formats (default is current directory).
    """
    # Extract embeddings from the last layer (only the last layer)
    pos_embds_last_layer = pos_embds.layers[-1].cpu().numpy()
    neg_embds_last_layer = neg_embds.layers[-1].cpu().numpy()
    ext_embds_last_layer = ext_embds.layers[-1].cpu().numpy()

    # Combine embeddings from all three classes into one array
    embeddings = np.vstack([pos_embds_last_layer, neg_embds_last_layer, ext_embds_last_layer])

    # Use t-SNE for dimensionality reduction (from high dimensions to 2D)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create the plot figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the t-SNE reduced embeddings with transparency (alpha)
    scatter_tsne = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                              c=np.concatenate([np.ones(pos_embds_last_layer.shape[0]), 
                                                np.zeros(neg_embds_last_layer.shape[0]), 
                                                np.full(ext_embds_last_layer.shape[0], 2)]),  # Class labels: 0 for neuroticism, 1 for agreeableness, 2 for extraversion
                              cmap='viridis', alpha=0.7)

    # Set the title and axis labels
    ax.set_title('', fontsize=14)
    ax.set_xlabel('', fontname='Avenir', fontsize=16)
    ax.set_ylabel('', fontname='Avenir', fontsize=16)

    # Set the font size for tick marks
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Add a legend with class labels
    ax.legend(handles=scatter_tsne.legend_elements()[0], 
              labels=['Agreeableness', 'Neuroticism', 'Extraversion'], 
              prop={'family': 'Avenir', 'size': 16})

    # Save the plot in PNG and PDF formats
    plt.savefig(f"{output_dir}/layer_31.png", format='png')
    plt.savefig(f"{output_dir}/layer_31.pdf", format='pdf', bbox_inches='tight')
    plt.close()

##PCA (Alternative Visualization using PCA)
# The following code (commented out) shows how to use PCA instead of t-SNE for dimensionality reduction
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from sklearn.decomposition import PCA

# def plot_embeddings(pos_embds, neg_embds, ext_embds, output_dir='./'):
#     # Extract embeddings from the last layer (only the last layer)
#     pos_embds_last_layer = pos_embds.layers[-1].cpu().numpy()
#     neg_embds_last_layer = neg_embds.layers[-1].cpu().numpy()
#     ext_embds_last_layer = ext_embds.layers[-1].cpu().numpy()

#     # Combine embeddings from all three classes into one array
#     embeddings = np.vstack([pos_embds_last_layer, neg_embds_last_layer, ext_embds_last_layer])

#     # Use PCA for dimensionality reduction (from high dimensions to 2D)
#     pca = PCA(n_components=2)
#     reduced_embeddings = pca.fit_transform(embeddings)

#     # Create the plot figure
#     plt.figure(figsize=(10, 8))

#     # Plot each class with a different color
#     plt.scatter(reduced_embeddings[:pos_embds_last_layer.shape[0], 0], reduced_embeddings[:pos_embds_last_layer.shape[0], 1], c='blue', label='Agreeableness', alpha=0.6)
#     plt.scatter(reduced_embeddings[pos_embds_last_layer.shape[0]:pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0], 0], reduced_embeddings[pos_embds_last_layer.shape[0]:pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0], 1], c='red', label='Neuroticism', alpha=0.6)
#     plt.scatter(reduced_embeddings[pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0]:, 0], reduced_embeddings[pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0]:, 1], c='green', label='Extraversion', alpha=0.6)

#     # Set title and labels
#     plt.title("PCA Visualization of Model's Last Layer Embeddings", fontsize=16)
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
    
#     # Add a legend
#     plt.legend()

#     # Save the plot in PNG and PDF formats
#     plt.colorbar()
#     plt.savefig(f"{output_dir}/embedding_visualization.png", format='png')
#     plt.savefig(f"{output_dir}/embedding_visualization.pdf", format='pdf')
#     plt.close()
