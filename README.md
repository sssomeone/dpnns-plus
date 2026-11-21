# DPNNS-Plus

Code of the paper

> **Delaunay-Based Nearest Neighbor Search with Grid-Accelerated Queries and Parallel Construction**<br>
> Pengfei Wang, Shiqing Xin, Changhe Tu<br>
> [arXiv:2611.15320](https://arxiv.org/abs/2611.15320)

DPNNS-Plus is an accelerated version of DP-NNS
([code](https://github.com/sssomeone/DPNN), [paper](https://doi.org/10.1109/TPAMI.2025.3610211)):
exact nearest-neighbor and k-nearest-neighbor search for 3D points on the
successor DAG of an incremental Delaunay construction. It is fastest when the
data points lie on a surface and the queries fill the surrounding volume.
Compared with DP-NNS it adds:

- **Grid-accelerated queries**: a uniform grid stores, per cell, a point on
  the search path of every query in the cell, so a nearest-neighbor query
  skips the beginning of its path. Results are unchanged.
- **Parallel construction**: the points are inserted level by level of a
  BRIO order, with the points of each level inserted concurrently, which
  keeps the logarithmic expected query cost.
- **Query reordering**: query batches are sorted along a Morton curve for
  cache locality.

## Build

Requirements: a C++20 compiler, CMake ≥ 3.20, [CGAL](https://www.cgal.org/) ≥ 5.4
(with the Boost headers) and [oneTBB](https://github.com/oneapi-src/oneTBB).

```bash
# Ubuntu:  sudo apt install cmake libcgal-dev libboost-dev libtbb-dev
# conda:   conda install -c conda-forge cmake cgal-cpp tbb-devel
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/dpnn_example                     # 200K points on a torus, 1M queries
./build/dpnn_example points.xyz          # your points, one "x y z" per line
```

Tested on Ubuntu 22.04 (GCC 11) and Windows 11 (clang-cl). On Windows, put
`tbb12.dll` and `tbbmalloc.dll` on the `PATH` or next to the executable.

## Usage

```cpp
#include <dpnn/dpnn.hpp>

std::vector<dpnn::Point> points = ...;              // {x, y, z}
dpnn::NearestNeighbor index(points);                // parallel construction

std::uint32_t i = index.query({0.1, 0.2, 0.3});     // index of the nearest point
std::vector<std::uint32_t> nn  = index.query(queries);         // batch, one id per query
std::vector<std::uint32_t> knn = index.query_knn(queries, 8);  // 8 ids per query
```

Link with `add_subdirectory(dpnns-plus)` and `target_link_libraries(app PRIVATE dpnn)`.
See `include/dpnn/dpnn.hpp` for the full interface.

## Citation

```bibtex
@article{wang2026delaunay,
  title   = {Delaunay-Based Nearest Neighbor Search with Grid-Accelerated Queries and Parallel Construction},
  author  = {Wang, Pengfei and Xin, Shiqing and Tu, Changhe},
  journal = {arXiv preprint arXiv:2611.15320},
  year    = {2026}
}

@article{10.1145/3811271,
  author = {Wang, Pengfei and Guo, Qinghao and Zhao, Haisen and Xin, Shiqing and Chen, Shuangmin and Tu, Changhe and Wang, Wenping},
  title = {Manifold k-NN: Accelerated k-NN Queries for Manifold Point Clouds},
  year = {2026},
  issue_date = {July 2026},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {45},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3811271},
  doi = {10.1145/3811271},
  journal = {ACM Trans. Graph.},
  month = jul,
  articleno = {49},
  numpages = {12}
}

@ARTICLE{11165079,
  author={Wang, Pengfei and Song, Jiantao and Xin, Shiqing and Chen, Shuangmin and Tu, Changhe and Wang, Wenping and Wang, Jiaye},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Efficient Nearest Neighbor Search Using Dynamic Programming},
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Delaunay triangulation;density peak clustering;farthest point sampling;nearest neighbor search;voronoi diagram},
  doi={10.1109/TPAMI.2025.3610211}}
```

## License

MIT
