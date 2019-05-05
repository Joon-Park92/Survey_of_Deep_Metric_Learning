# 👏 Survey of Deep Metric Learning

Traditionally, they have defined metrics in a variety of ways, including Euclidean distance and cosine similarity.


💡I hope that many people will learn about metric learning through this repository.

---
### 1️⃣ Euclidean-based metric

- Dimensionality Reduction by Learning an Invariant Mapping (Contrastive) (CVPR 2006) [[Paper]](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)[[Caffe]](https://github.com/wujiyang/Contrastive-Loss)[[Tensorflow]](https://github.com/ardiya/siamesenetwork-tensorflow)[[Keras]](https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py)[[Pytorch1]](https://github.com/delijati/pytorch-siamese/blob/master/contrastive.py)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)

- FaceNet: A Unified Embedding for Face Recognition and Clustering (Triplet) (CVPR 2015) [[Paper]](https://arxiv.org/abs/1503.03832)[[Tensorflow]](https://github.com/omoindrot/tensorflow-triplet-loss)[[Pytorch]](https://github.com/bnulihaixia/Deep_metric)

- Regressive Virtual Metric Learning (NIPS 2015) [[Paper]](https://papers.nips.cc/paper/5687-regressive-virtual-metric-learning)
(Strictly speaking, they used Mahalanobis distance)

- Deep Metric Learning via Lifted Structured Feature Embedding (LSSS) (CVPR 2016) [[Paper]](https://arxiv.org/abs/1511.06452)[[Chainer]](https://github.com/ronekko/deep_metric_learning)[[Caffe]](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)[[Pytorch1]](https://github.com/zhengxiawu/pytorch_deep_metric_learning)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)

- Improved Deep Metric Learning with
Multi-class N-pair Loss Objective (N-pair) (NIPS 2016) [[Paper]](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)[[Pytorch]](https://github.com/ChaofWang/Npair_loss_pytorch)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Beyond triplet loss: a deep quadruplet network for person re-identification (Quadruplet) (CVPR 2017) [[Paper]](https://cvip.computing.dundee.ac.uk/papers/Chen_CVPR_2017_paper.pdf)

- Deep Metric Learning via Facility Location[[Paper]](https://arxiv.org/abs/1612.01213)[[Tensorflow]](https://github.com/CongWeilin/cluster-loss-tensorflow)

- No Fuss Distance Metric Learning using Proxies (Proxy NCA) (ICCV 2017) [[Paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf)[[Pytorch1]](https://github.com/dichotomies/proxy-nca)[[Pytorch2]](https://github.com/bnulihaixia/Deep_metric)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Deep Metric Learning with Angular Loss [[Paper]](https://arxiv.org/abs/1708.01682)[[Tensorflow]](https://github.com/geonm/tf_angular_loss)[[Chainer]](https://github.com/ronekko/deep_metric_learning)

- Ranked List Loss for Deep Metric Learning (CVPR 2019) [[Paper]](https://arxiv.org/abs/1903.03238)

- Hardness-Aware Deep Metric Learning (CVPR 2019) [[Paper]](https://arxiv.org/abs/1903.05503)[[Tensorflow]](https://github.com/wzzheng/HDML)

---
### 2️⃣ Similarity-based metric

- Deep Metric Learning for Practical Person Re-Identification [[Paper]](https://arxiv.org/abs/1407.4979)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)[[Pytorch]](https://github.com/bnulihaixia/Deep_metric)

- Learning Deep Embeddings with Histogram Loss (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6464-learning-deep-embeddings-with-histogram-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)[[Pytorch]](https://github.com/valerystrizh/pytorch-histogram-loss)[[Caffe]](https://github.com/madkn/HistogramLoss)

- Learning Deep Disentangled Embeddings With the F-Statistic Loss (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7303-learning-deep-disentangled-embeddings-with-the-f-statistic-loss)[[Tensorflow]](https://github.com/kridgeway/f-statistic-loss-nips-2018)

---
### 3️⃣ Integrated framework

- Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7293-adapted-deep-embeddings-a-synthesis-of-methods-for-k-shot-inductive-transfer-learning)[[Tensorflow]](https://github.com/tylersco/adapted_deep_embeddings)

---
### 4️⃣ Ensemble method

- Deep Randomized Ensembles for Metric Learning (ECCV 2018) [[Paper]](https://arxiv.org/abs/1808.04469)[[Pytorch]](https://github.com/littleredxh/DREML)

- Attention-based Ensemble for Deep Metric Learning (ECCV 2018) [[Paper]](https://arxiv.org/abs/1804.00382)

- Deep Metric Learning with Hierarchical
Triplet Loss (ECCV 2018) [[Paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ge_Deep_Metric_Learning_ECCV_2018_paper.pdf)

---
### 5️⃣ Related works

- Distance Metric Learning for Large Margin Nearest Neighbor Classification (NIPS 2005) [[Paper]](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification)[[Journal]](http://jmlr.csail.mit.edu/papers/volume10/weinberger09a/weinberger09a.pdf)

- Metric Learning by Collapsing Classes (NIPS 2005) [[Paper]](https://papers.nips.cc/paper/2947-metric-learning-by-collapsing-classes)

- Discriminative Metric Learning by Neighborhood Gerrymandering (NIPS 2014) [[Paper]](https://papers.nips.cc/paper/5385-discriminative-metric-learning-by-neighborhood-gerrymandering)

- Log-Hilbert-Schmidt metric between positive definite operators on Hilbert spaces (NIPS 2014) [[Paper]](https://papers.nips.cc/paper/5457-log-hilbert-schmidt-metric-between-positive-definite-operators-on-hilbert-spaces)

- Sample complexity of learning Mahalanobis distance metrics (NIPS 2015) [[Paper]](https://arxiv.org/abs/1505.02729)

- Improved Error Bounds for Tree Representations of Metric Spaces (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6431-improved-error-bounds-for-tree-representations-of-metric-spaces)

- What Makes Objects Similar: A Unified Multi-Metric Learning Approach (NIPS 2016) [[Paper]](https://papers.nips.cc/paper/6192-what-makes-objects-similar-a-unified-multi-metric-learning-approach)

- Learning Low-Dimensional Metrics (NIPS 2017) [[Paper]](https://papers.nips.cc/paper/7002-learning-low-dimensional-metrics)

- Generative Local Metric Learning for Kernel Regression (NIPS 2017) [[Paper]](https://papers.nips.cc/paper/6839-generative-local-metric-learning-for-kernel-regression)

- Persistence Fisher Kernel: A Riemannian Manifold
Kernel for Persistence Diagrams (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/8205-persistence-fisher-kernel-a-riemannian-manifold-kernel-for-persistence-diagrams)

- Bilevel Distance Metric Learning for Robust Image Recognition (NIPS 2018) [[Paper]](https://papers.nips.cc/paper/7674-bilevel-distance-metric-learning-for-robust-image-recognition)



---
### Milestone

[x] Add Euclidean-based metric

[x] Add Similarity-based metric

[ ] Add Ensemble-based metric

[ ] Add brief descriptions