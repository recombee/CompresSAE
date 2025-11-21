# [*The Future is Sparse*](https://arxiv.org/abs/2505.11388)
## Embedding Compression for Scalable Retrieval in Recommender Systems
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3705328.3748147-blue)](https://doi.org/10.1145/3705328.3748147) [![arXiv](https://img.shields.io/badge/arXiv-2505.11388-b31b1b.svg)](https://arxiv.org/abs/2505.11388) By [Recombee Research](https://www.recombee.com/research)

Recommender systems embeddings are growing. Sparsity is here to help. 

### 💡 Watch our RecSys 2025 [spotlight talk (YouTube)](https://www.youtube.com/watch?v=Hma0PSOGUw8) to learn more, or check out the conference [poster](assets/poster.pdf).

![Results](./assets/results.jpg)
Our learnable sparse compression algorithm, **CompresSAE**, achieves a superior compression-retrieval accuracy trade-off, outperforming equally sized Matryoshka embeddings and approaching uncompressed embedding performance with 12× fewer parameters.

## Model Architecture (CompresSAE)
![CompresSAE Architecture](./assets/model.jpg)
CompresSAE is a sparse autoencoder (SAE) that maps dense embeddings into high-dimensional, sparsely activated vectors optimized for fast similarity search.

Two inference modes allow a trade-off between latency and accuracy: fast retrieval computes similarity in the sparse compressed space, while high-accuracy retrieval uses similarity in the dense reconstructed space.

See [model.py](./model.py) for implementation.

## Citation
If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{10.1145/3705328.3748147,
author = {Kasalick\'{y}, Petr and Spi\v{s}\'{a}k, Martin and Van\v{c}ura, Vojt\v{e}ch and Bohun\v{e}k, Daniel and Alves, Rodrigo and Kord\'{\i}k, Pavel},
title = {The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems},
year = {2025},
isbn = {9798400713644},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3705328.3748147},
doi = {10.1145/3705328.3748147},
abstract = {Industry-scale recommender systems face a core challenge: representing entities with high cardinality, such as users or items, using dense embeddings that must be accessible during both training and inference. However, as embedding sizes grow, memory constraints make storage and access increasingly difficult. We describe a lightweight, learnable embedding compression technique that projects dense embeddings into a high-dimensional, sparsely activated space. Designed for retrieval tasks, our method reduces memory requirements while preserving retrieval performance, enabling scalable deployment under strict resource constraints. Our results demonstrate that leveraging sparsity is a promising approach for improving the efficiency of large-scale recommenders. We release our code at .},
booktitle = {Proceedings of the Nineteenth ACM Conference on Recommender Systems},
pages = {1099–1103},
numpages = {5},
keywords = {Embedding Compression, Sparse Autoencoders},
location = {
},
series = {RecSys '25}
}
```

## License
[MIT License](./LICENSE)