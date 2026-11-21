// Delaunay-based exact nearest-neighbor search in 3D.
//
// The index is the successor DAG of an incremental Delaunay construction
// (DP-NNS), built in parallel with a level-consistent insertion order, plus a
// uniform start grid that lets each nearest-neighbor query skip the beginning
// of its walk.  Batches of queries are processed in Morton order on CPU threads.
//
// This header has no third-party dependency; link against the `dpnn` library.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace dpnn {

// A 3D point as {x, y, z}.
using Point = std::array<double, 3>;

// Exact nearest-neighbor index over a fixed set of 3D points.  Every query
// returns indices into the input array.  Duplicate input points are indexed
// once; a query then returns one of their indices.
class NearestNeighbor {
public:
    // Builds the index.  threads = 1 builds serially, threads > 1 uses that
    // many worker threads, threads = 0 chooses the number of workers.
    explicit NearestNeighbor(const std::vector<Point>& points, int threads = 0);

    // Same, from a flat array of num_points xyz triples.
    NearestNeighbor(const double* xyz, std::size_t num_points, int threads = 0);

    ~NearestNeighbor();
    NearestNeighbor(NearestNeighbor&&) noexcept;
    NearestNeighbor& operator=(NearestNeighbor&&) noexcept;
    NearestNeighbor(const NearestNeighbor&) = delete;
    NearestNeighbor& operator=(const NearestNeighbor&) = delete;

    // Number of indexed (distinct) points.
    std::size_t size() const;

    // Nearest neighbor of one point.
    std::uint32_t query(const Point& p) const;
    std::uint32_t query(double x, double y, double z) const;

    // Nearest neighbor of every point of a batch, in the order of `queries`.
    // The batch is processed in Morton order for cache locality, on all
    // worker threads when `parallel` is true.
    std::vector<std::uint32_t> query(const std::vector<Point>& queries,
                                     bool parallel = true) const;
    // Same, into a caller-owned buffer of queries.size() ids.  Reusing one
    // buffer across batches avoids the page faults of a fresh allocation.
    void query(const std::vector<Point>& queries, std::uint32_t* out,
               bool parallel = true) const;

    // The k nearest neighbors, sorted by distance.  For a batch, the result
    // is row-major: query i occupies [i*k, (i+1)*k).
    std::vector<std::uint32_t> query_knn(const Point& p, std::size_t k) const;
    std::vector<std::uint32_t> query_knn(const std::vector<Point>& queries,
                                         std::size_t k, bool parallel = true) const;
    // Same, into a caller-owned buffer of queries.size() * k ids.
    void query_knn(const std::vector<Point>& queries, std::size_t k, std::uint32_t* out,
                   bool parallel = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dpnn
