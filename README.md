# Equivariant Cortical Mesh Segmentation

Official PyTorch Geometric based implementation of:

<b>Utility of Equivariant Message Passing in Cortical Mesh Segmentation</b>

Dániel Unyi, Ferdinando Insalata, Petar Veličković, Bálint Gyires-Tóth

https://arxiv.org/abs/2206.03164

<b>Abstract</b>: The automated segmentation of cortical areas has been a long-standing challenge in medical image analysis. The complex geometry of the cortex is commonly represented as a polygon mesh, whose segmentation can be addressed by graph-based learning methods. When cortical meshes are misaligned across subjects, current methods produce significantly worse segmentation results, limiting their ability to handle multi-domain data. In this paper, we investigate the utility of E(n)-equivariant graph neural networks (EGNNs), comparing their performance against plain graph neural networks (GNNs). Our evaluation shows that GNNs outperform EGNNs on aligned meshes, due to their ability to leverage the presence of a global coordinate system. On misaligned meshes, the performance of plain GNNs drop considerably, while E(n)-equivariant message passing maintains the same segmentation results. The best results can also be obtained by using plain GNNs on realigned data (co-registered meshes in a global coordinate system).
