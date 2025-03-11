# 向量数据库索引参数推荐工具 (VecIndexFinder)

这是一个向量数据库索引参数推荐工具，帮助用户为向量数据库表选择最优的索引参数配置。该工具支持从数据库表中采样数据，测试不同索引参数的性能，并推荐最佳参数组合。

## 主要功能

- 数据采样：从向量数据库表中采样一定比例的数据
- 索引测试：支持多种索引类型（IVF-FLAT, IVF-PQ, HNSW, DiskANN等）
- 参数优化：测试不同参数组合的性能（QPS和召回率）
- 性能分析：基于测试结果推荐最优参数配置
- 结果可视化：生成性能对比图表

## 支持的数据库

- Milvus
- OpenGauss
- PGVector
- 可扩展支持更多数据库

## 安装

### 环境要求

- Python 3.8+
- 支持的向量数据库客户端

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-repo/VecIndexFinder.git
cd VecIndexFinder

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python index_finder.py --db milvus --table my_vectors --indices ivfflat,hnsw
```

### 完整参数说明

```
用法: python index_finder.py [选项]

选项:
  --db TYPE                      向量数据库类型 (milvus, opengauss, pgvector)
  --table NAME                   数据表名称
  --dim INT                      向量维度
  --metric TYPE                  距离度量类型 (l2, ip, cosine)
  --indices TYPES                要测试的索引类型列表，逗号分隔 (ivfflat,hnsw,ivfpq,diskann)
  --sample-ratio FLOAT           索引数据采样比例 (0-1)
  --sample-method METHOD         采样方法 (random, stratified, cluster)
  --query-json FILE              查询数据JSON文件路径
  --query-count INT              自动采样的查询数量
  --topk INT                     查询返回的结果数
  --concurrency INT              并发查询数
  --output DIR                   输出目录
  --config FILE                  配置文件路径
  --optimize-for METRIC          优化目标 (qps, recall, combined)
  --weight-qps FLOAT             QPS权重 (0-1)
  --weight-recall FLOAT          召回率权重 (0-1)
  --verbose                      详细输出模式
```

### 示例

1. 基本用法（使用默认配置）:

```bash
python index_finder.py --db milvus --table my_vectors --indices ivfflat,hnsw
```

2. 指定采样比例和查询数据:

```bash
python index_finder.py --db milvus --table my_vectors --indices ivfflat,hnsw --sample-ratio 0.1 --query-count 5000
```

3. 使用自定义配置文件:

```bash
python index_finder.py --db milvus --table my_vectors --config my_config.yml
```

4. 优化QPS（速度优先）:

```bash
python index_finder.py --db milvus --table my_vectors --optimize-for qps --weight-qps 0.8 --weight-recall 0.2
```

5. 优化召回率（准确度优先）:

```bash
python index_finder.py --db milvus --table my_vectors --optimize-for recall --weight-qps 0.2 --weight-recall 0.8
```

### 自定义查询数据

用户可以提供自定义的查询数据集（JSON格式），格式要求为：

```json
{
  "vectors": [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...],
* [Annoy](https://github.com/spotify/annoy) ![https://img.shields.io/github/stars/spotify/annoy?style=social](https://img.shields.io/github/stars/spotify/annoy?style=social)
* [FLANN](http://www.cs.ubc.ca/research/flann/) ![https://img.shields.io/github/stars/flann-lib/flann?style=social](https://img.shields.io/github/stars/flann-lib/flann?style=social)
* [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html): LSHForest, KDTree, BallTree
* [Weaviate](https://github.com/weaviate/weaviate) ![https://img.shields.io/github/stars/weaviate/weaviate?style=social](https://img.shields.io/github/stars/weaviate/weaviate?style=social)
* [PANNS](https://github.com/ryanrhymes/panns) ![https://img.shields.io/github/stars/ryanrhymes/panns?style=social](https://img.shields.io/github/stars/ryanrhymes/panns?style=social)
* [NearPy](http://pixelogik.github.io/NearPy/) ![https://img.shields.io/github/stars/pixelogik/NearPy?style=social](https://img.shields.io/github/stars/pixelogik/NearPy?style=social)
* [KGraph](https://github.com/aaalgo/kgraph) ![https://img.shields.io/github/stars/aaalgo/kgraph?style=social](https://img.shields.io/github/stars/aaalgo/kgraph?style=social)
* [NMSLIB (Non-Metric Space Library)](https://github.com/nmslib/nmslib) ![https://img.shields.io/github/stars/nmslib/nmslib?style=social](https://img.shields.io/github/stars/nmslib/nmslib?style=social): SWGraph, HNSW, BallTree, MPLSH
* [hnswlib (a part of nmslib project)](https://github.com/nmslib/hnsw) ![https://img.shields.io/github/stars/nmslib/hnsw?style=social](https://img.shields.io/github/stars/nmslib/hnsw?style=social)
* [RPForest](https://github.com/lyst/rpforest) ![https://img.shields.io/github/stars/lyst/rpforest?style=social](https://img.shields.io/github/stars/lyst/rpforest?style=social)
* [FAISS](https://github.com/facebookresearch/faiss) ![https://img.shields.io/github/stars/facebookresearch/faiss?style=social](https://img.shields.io/github/stars/facebookresearch/faiss?style=social)
* [DolphinnPy](https://github.com/ipsarros/DolphinnPy) ![https://img.shields.io/github/stars/ipsarros/DolphinnPy?style=social](https://img.shields.io/github/stars/ipsarros/DolphinnPy?style=social)
* [Datasketch](https://github.com/ekzhu/datasketch) ![https://img.shields.io/github/stars/ekzhu/datasketch?style=social](https://img.shields.io/github/stars/ekzhu/datasketch?style=social)
* [nndescent](https://github.com/brj0/nndescent) ![https://img.shields.io/github/stars/brj0/nndescent?style=social](https://img.shields.io/github/stars/brj0/nndescent?style=social)
* [PyNNDescent](https://github.com/lmcinnes/pynndescent) ![https://img.shields.io/github/stars/lmcinnes/pynndescent?style=social](https://img.shields.io/github/stars/lmcinnes/pynndescent?style=social)
* [MRPT](https://github.com/teemupitkanen/mrpt) ![https://img.shields.io/github/stars/teemupitkanen/mrpt?style=social](https://img.shields.io/github/stars/teemupitkanen/mrpt?style=social)
* [NGT](https://github.com/yahoojapan/NGT) ![https://img.shields.io/github/stars/yahoojapan/NGT?style=social](https://img.shields.io/github/stars/yahoojapan/NGT?style=social): ONNG, PANNG, QG
* [SPTAG](https://github.com/microsoft/SPTAG) ![https://img.shields.io/github/stars/microsoft/SPTAG?style=social](https://img.shields.io/github/stars/microsoft/SPTAG?style=social)
* [PUFFINN](https://github.com/puffinn/puffinn) ![https://img.shields.io/github/stars/puffinn/puffinn?style=social](https://img.shields.io/github/stars/puffinn/puffinn?style=social)
* [N2](https://github.com/kakao/n2) ![https://img.shields.io/github/stars/kakao/n2?style=social](https://img.shields.io/github/stars/kakao/n2?style=social)
* [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
* [Vearch](https://github.com/vearch/vearch) ![https://img.shields.io/github/stars/vearch/vearch?style=social](https://img.shields.io/github/stars/vearch/vearch?style=social)
* [Elasticsearch](https://github.com/elastic/elasticsearch) ![https://img.shields.io/github/stars/elastic/elasticsearch?style=social](https://img.shields.io/github/stars/elastic/elasticsearch?style=social): HNSW
* [Elastiknn](https://github.com/alexklibisz/elastiknn) ![https://img.shields.io/github/stars/alexklibisz/elastiknn?style=social](https://img.shields.io/github/stars/alexklibisz/elastiknn?style=social)
* [ExpANN](https://github.com/jacketsj/expANN) ![https://img.shields.io/github/stars/jacketsj/expANN?style=social](https://img.shields.io/github/stars/jacketsj/expANN?style=social)
* [OpenSearch KNN](https://github.com/opensearch-project/k-NN) ![https://img.shields.io/github/stars/opensearch-project/k-NN?style=social](https://img.shields.io/github/stars/opensearch-project/k-NN?style=social)
* [DiskANN](https://github.com/microsoft/diskann) ![https://img.shields.io/github/stars/microsoft/diskann?style=social](https://img.shields.io/github/stars/microsoft/diskann?style=social): Vamana, Vamana-PQ
* [Vespa](https://github.com/vespa-engine/vespa) ![https://img.shields.io/github/stars/vespa-engine/vespa?style=social](https://img.shields.io/github/stars/vespa-engine/vespa?style=social)
* [scipy](https://docs.scipy.org/doc/scipy/reference/spatial.html): cKDTree
* [vald](https://github.com/vdaas/vald) ![https://img.shields.io/github/stars/vdaas/vald?style=social](https://img.shields.io/github/stars/vdaas/vald?style=social)
* [Qdrant](https://github.com/qdrant/qdrant) ![https://img.shields.io/github/stars/qdrant/qdrant?style=social](https://img.shields.io/github/stars/qdrant/qdrant?style=social)
* [HUAWEI(qsgngt)](https://github.com/WPJiang/HWTL_SDU-ANNS.git)
* [Milvus](https://github.com/milvus-io/milvus) ![https://img.shields.io/github/stars/milvus-io/milvus?style=social](https://img.shields.io/github/stars/milvus-io/milvus?style=social): [Knowhere](https://github.com/milvus-io/knowhere)
* [Zilliz(Glass)](https://github.com/hhy3/pyglass)
* [pgvector](https://github.com/pgvector/pgvector) ![https://img.shields.io/github/stars/pgvector/pgvector?style=social](https://img.shields.io/github/stars/pgvector/pgvector?style=social)
* [pgvecto.rs](https://github.com/tensorchord/pgvecto.rs) ![https://img.shields.io/github/stars/tensorchord/pgvecto.rs?style=social](https://img.shields.io/github/stars/tensorchord/pgvecto.rs?style=social)
* [RediSearch](https://github.com/redisearch/redisearch) ![https://img.shields.io/github/stars/redisearch/redisearch?style=social](https://img.shields.io/github/stars/redisearch/redisearch?style=social)
  * [pg_embedding](https://github.com/neondatabase/pg_embedding) ![https://img.shields.io/github/stars/pg_embedding/pg_embedding?style=social](https://img.shields.io/github/stars/neondatabase/pg_embedding?style=social)
* [Descartes(01AI)](https://github.com/xiaoming-01ai/descartes)
* [kgn](https://github.com/Henry-yan/kgn)

Data sets
=========

We have a number of precomputed data sets in HDF5 format. All data sets have been pre-split into train/test and include ground truth data for the top-100 nearest neighbors.

| Dataset                                                           | Dimensions | Train size | Test size | Neighbors | Distance  | Download                                                                   |
| ----------------------------------------------------------------- | ---------: | ---------: | --------: | --------: | --------- | -------------------------------------------------------------------------- |
| [DEEP1B](http://sites.skoltech.ru/compvision/noimi/)              |         96 |  9,990,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/deep-image-96-angular.hdf5) (3.6GB)
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) (217MB) |
| [GIST](http://corpus-texmex.irisa.fr/)                            |        960 |  1,000,000 |     1,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/gist-960-euclidean.hdf5) (3.6GB)          |
| [GloVe](http://nlp.stanford.edu/projects/glove/)                  |         25 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-25-angular.hdf5) (121MB)            |
| GloVe                                                             |         50 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-50-angular.hdf5) (235MB)            |
| GloVe                                                             |        100 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-100-angular.hdf5) (463MB)           |
| GloVe                                                             |        200 |  1,183,514 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/glove-200-angular.hdf5) (918MB)           |
| [Kosarak](http://fimi.uantwerpen.be/data/)                        |      27,983 |     74,962 |       500 |       100 | Jaccard   | [HDF5](http://ann-benchmarks.com/kosarak-jaccard.hdf5) (33MB)             |
| [MNIST](http://yann.lecun.com/exdb/mnist/)                        |        784 |     60,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/mnist-784-euclidean.hdf5) (217MB)         |
| [MovieLens-10M](https://grouplens.org/datasets/movielens/10m/)  |      65,134 |     69,363 |       500 |       100 | Jaccard   | [HDF5](http://ann-benchmarks.com/movielens10m-jaccard.hdf5) (63MB)             |
| [NYTimes](https://archive.ics.uci.edu/ml/datasets/bag+of+words)   |        256 |    290,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/nytimes-256-angular.hdf5) (301MB)         |
| [SIFT](http://corpus-texmex.irisa.fr/)                           |        128 |  1,000,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/sift-128-euclidean.hdf5) (501MB)          |
| [Last.fm](https://github.com/erikbern/ann-benchmarks/pull/91)     |         65 |    292,385 |    50,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/lastfm-64-dot.hdf5) (135MB)               |

Results
=======

These are all as of April 2023, running all benchmarks on a r6i.16xlarge machine on AWS with `--parallelism 31` and hyperthreading disabled. All benchmarks are single-CPU.

glove-100-angular
-----------------

![glove-100-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/glove-100-angular.png)

sift-128-euclidean
------------------

![glove-100-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/sift-128-euclidean.png)

fashion-mnist-784-euclidean
---------------------------

![fashion-mnist-784-euclidean](https://raw.github.com/erikbern/ann-benchmarks/master/results/fashion-mnist-784-euclidean.png)

nytimes-256-angular
-------------------

![nytimes-256-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/nytimes-256-angular.png)

gist-960-euclidean
-------------------

![gist-960-euclidean](https://raw.github.com/erikbern/ann-benchmarks/master/results/gist-960-euclidean.png)

glove-25-angular
----------------

![glove-25-angular](https://raw.github.com/erikbern/ann-benchmarks/master/results/glove-25-angular.png)

TODO: update plots on <http://ann-benchmarks.com>.

Install
=======

The only prerequisite is Python (tested with 3.10.6) and Docker.

1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Run `python install.py` to build all the libraries inside Docker containers (this can take a while, like 10-30 minutes).

Running
=======

1. Run `python run.py` (this can take an extremely long time, potentially days)
2. Run `python plot.py` or `python create_website.py` to plot results.
3. Run `python data_export.py --out res.csv` to export all results into a csv file for additional post-processing.

You can customize the algorithms and datasets as follows:

* Check that `ann_benchmarks/algorithms/{YOUR_IMPLEMENTATION}/config.yml` contains the parameter settings that you want to test
* To run experiments on SIFT, invoke `python run.py --dataset glove-100-angular`. See `python run.py --help` for more information on possible settings. Note that experiments can take a long time. 
* To process the results, either use `python plot.py --dataset glove-100-angular` or `python create_website.py`. An example call: `python create_website.py --plottype recall/time --latex --scatter --outputdir website/`. 

Including your algorithm
========================

Add your algorithm in the folder `ann_benchmarks/algorithms/{YOUR_IMPLEMENTATION}/` by providing

- [ ] A small Python wrapper in `module.py`
- [ ] A Dockerfile named `Dockerfile` 
- [ ] A set of hyper-parameters in `config.yml`
- [ ] A CI test run by adding your implementation to `.github/workflows/benchmarks.yml`

Check the [available implementations](./ann_benchmarks/algorithms/) for inspiration.

Principles
==========

* Everyone is welcome to submit pull requests with tweaks and changes to how each library is being used.
* In particular: if you are the author of any of these libraries, and you think the benchmark can be improved, consider making the improvement and submitting a pull request.
* This is meant to be an ongoing project and represent the current state.
* Make everything easy to replicate, including installing and preparing the datasets.
* Try many different values of parameters for each library and ignore the points that are not on the precision-performance frontier.
* High-dimensional datasets with approximately 100-1000 dimensions. This is challenging but also realistic. Not more than 1000 dimensions because those problems should probably be solved by doing dimensionality reduction separately.
* Single queries are used by default. ANN-Benchmarks enforces that only one CPU is saturated during experimentation, i.e., no multi-threading. A batch mode is available that provides all queries to the implementations at once. Add the flag `--batch` to `run.py` and `plot.py` to enable batch mode. 
* Avoid extremely costly index building (more than several hours).
* Focus on datasets that fit in RAM. For billion-scale benchmarks, see the related [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) project.
* We mainly support CPU-based ANN algorithms. GPU support exists for FAISS, but it has to be compiled with GPU support locally and experiments must be run using the flags `--local --batch`. 
* Do proper train/test set of index data and query points.
* Note that we consider that set similarity datasets are sparse and thus we pass a **sorted** array of integers to algorithms to represent the set of each user.


Authors
=======

Built by [Erik Bernhardsson](https://erikbern.com) with significant contributions from [Martin Aumüller](http://itu.dk/people/maau/) and [Alexander Faithfull](https://github.com/ale-f).

Related Publication
==================

Design principles behind the benchmarking framework are described in the following publications: 

- M. Aumüller, E. Bernhardsson, A. Faithfull:
[ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms](https://arxiv.org/abs/1807.05614). Information Systems 2019. DOI: [10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006)
-   M. Aumüller, E. Bernhardsson, A. Faithfull: [Reproducibility protocol for ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor search algorithms](https://itu.dk/people/maau/additional/2022-ann-benchmarks-reproducibility.pdf), [Artifacts](https://doi.org/10.5281/zenodo.4607761).

Related Projects
================

- [big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) is a benchmarking effort for billion-scale approximate nearest neighbor search as part of the [NeurIPS'21 Competition track](https://neurips.cc/Conferences/2021/CompetitionTrack).

