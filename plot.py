import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

def plot_embeddings(pos_embds, neg_embds, ext_embds, output_dir='./'):
    # 获取最后一层的嵌入（只取最后一层）
    pos_embds_last_layer = pos_embds.layers[-1].cpu().numpy()
    neg_embds_last_layer = neg_embds.layers[-1].cpu().numpy()
    ext_embds_last_layer = ext_embds.layers[-1].cpu().numpy()

    # 合并正类、负类、外向类最后一层的嵌入
    embeddings = np.vstack([pos_embds_last_layer, neg_embds_last_layer, ext_embds_last_layer])

    # 使用 t-SNE 对嵌入进行降维，方便在二维空间中可视化
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # t-SNE 可视化，增加透明度
    scatter_tsne = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                              c=np.concatenate([np.ones(pos_embds_last_layer.shape[0]), 
                                                np.zeros(neg_embds_last_layer.shape[0]), 
                                                np.full(ext_embds_last_layer.shape[0], 2)]),
                              cmap='viridis', alpha=0.7)

    # 设置标题和标签
    ax.set_title('', fontsize=14)
    ax.set_xlabel('', fontname='Avenir', fontsize=16)
    ax.set_ylabel('', fontname='Avenir', fontsize=16)

    # 设置坐标刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=14)  # 设置坐标轴刻度的字体大小

    # 添加图例
    ax.legend(handles=scatter_tsne.legend_elements()[0], 
              labels=['Agreeableness', 'Neuroticism', 'Extraversion'], 
              prop={'family': 'Avenir', 'size': 16})

    # 保存为PNG和PDF格式
    plt.savefig(f"{output_dir}/layer_31.png", format='png')
    plt.savefig(f"{output_dir}/layer_31.pdf", format='pdf', bbox_inches='tight')
    plt.close()

##PCA
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from sklearn.decomposition import PCA

# def plot_embeddings(pos_embds, neg_embds, ext_embds, output_dir='./'):
#     # 获取最后一层的嵌入（只取最后一层）
#     pos_embds_last_layer = pos_embds.layers[-1].cpu().numpy()
#     neg_embds_last_layer = neg_embds.layers[-1].cpu().numpy()
#     ext_embds_last_layer = ext_embds.layers[-1].cpu().numpy()

#     # 合并正类、负类、外向类最后一层的嵌入
#     embeddings = np.vstack([pos_embds_last_layer, neg_embds_last_layer, ext_embds_last_layer])

#     # 使用 PCA 对嵌入进行降维，方便在二维空间中可视化
#     pca = PCA(n_components=2)
#     reduced_embeddings = pca.fit_transform(embeddings)

#     # 创建图形
#     plt.figure(figsize=(10, 8))
    
#     # 为不同类别绘制不同的颜色
#     plt.scatter(reduced_embeddings[:pos_embds_last_layer.shape[0], 0], reduced_embeddings[:pos_embds_last_layer.shape[0], 1], c='blue', label='Agreeableness', alpha=0.6)
#     plt.scatter(reduced_embeddings[pos_embds_last_layer.shape[0]:pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0], 0], reduced_embeddings[pos_embds_last_layer.shape[0]:pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0], 1], c='red', label='Neuroticism', alpha=0.6)
#     plt.scatter(reduced_embeddings[pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0]:, 0], reduced_embeddings[pos_embds_last_layer.shape[0]+neg_embds_last_layer.shape[0]:, 1], c='green', label='Extraversion', alpha=0.6)

#     # 设置标题和标签
#     plt.title("PCA Visualization of Model's Last Layer Embeddings", fontsize=16)
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
    
#     # 添加图例
#     plt.legend()

#     # 保存为PNG和PDF格式
#     plt.colorbar()
#     plt.savefig(f"{output_dir}/embedding_visualization.png", format='png')
#     plt.savefig(f"{output_dir}/embedding_visualization.pdf", format='pdf')
#     plt.close()
