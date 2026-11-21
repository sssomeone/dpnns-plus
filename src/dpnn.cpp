// Delaunay-based exact nearest-neighbor search in 3D: implementation.
// Public API: include/dpnn/dpnn.hpp
//
//  * Preprocessing inserts the points into a Delaunay triangulation in a
//    BRIO order (random levels, Hilbert order inside each level).  When a
//    point is inserted, it becomes a successor of every earlier point it is
//    adjacent to; these successor lists form the successor DAG of DP-NNS.
//    The insertion runs in parallel one level at a time: the levels keep
//    their order, the points of a level are inserted concurrently.
//  * A uniform start grid stores, per cell, a node that lies on the walk of
//    every query in the cell, so that a nearest-neighbor query skips the
//    beginning of its walk.
//  * Batches of queries are processed in Morton order for cache locality.
#include "dpnn/dpnn.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/spatial_sort.h>
#include <CGAL/Bbox_3.h>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_arena.h>

namespace {

// Filtered exact predicates: with inexact orientation and in-sphere tests the
// insertion can loop forever on point sets with many cospherical points.
using K       = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Vb      = CGAL::Triangulation_vertex_base_with_info_3<unsigned, K>;
using Cb      = CGAL::Delaunay_triangulation_cell_base_3<K>;
using Tds     = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag>;

// vector::resize value-initializes every element, a long serial memset for
// arrays that are about to be overwritten in parallel.  This allocator
// default-initializes instead, which leaves trivial types untouched.
template <class T>
struct DefaultInitAllocator : std::allocator<T> {
    using std::allocator<T>::allocator;
    template <class U> struct rebind { using other = DefaultInitAllocator<U>; };
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};
template <class T> using RawVector = std::vector<T, DefaultInitAllocator<T>>;

// Locks of the parallel insertion, one per triangulation vertex.  CGAL hands
// the lock structure a reference to the point of a vertex, whose address is
// fixed for the lifetime of the vertex, so hashing that address gives each
// vertex its own lock word (a table of 2^bits words).  The priority rule is
// that of CGAL's Spatial_lock_grid_3<Tag_priority_blocking>: a thread waits
// for a lower-priority holder and backs off from a higher-priority one, so
// waits never form a cycle.  A word stores the id of the holding thread.
class VertexLocks {
public:
    VertexLocks(const CGAL::Bbox_3& bbox, int bits)
        : bbox_(bbox), bits_(bits),
          words_(static_cast<std::atomic<uint32_t>*>(
              std::calloc(std::size_t(1) << bits, sizeof(std::atomic<uint32_t>)))) {
        if (!words_) throw std::bad_alloc();
    }
    ~VertexLocks() { std::free(words_); }
    VertexLocks(const VertexLocks&) = delete;
    VertexLocks& operator=(const VertexLocks&) = delete;

    const CGAL::Bbox_3& get_bbox() const { return bbox_; }

    template <typename P3>
    bool try_lock(const P3& p, int /*lock_radius*/ = 0) {
        const uint32_t word = index(p);
        State& me = state();
        uint32_t holder = words_[word].load(std::memory_order_relaxed);
        if (holder == me.id) return true;
        for (;;) {
            if (holder == 0) {
                if (words_[word].compare_exchange_weak(holder, me.id, std::memory_order_acquire,
                                                       std::memory_order_relaxed)) {
                    me.locked.push_back(word);
                    return true;
                }
                continue;                       // holder now holds the new owner
            }
            if (holder > me.id) return false;  // higher priority: back off
            std::this_thread::yield();          // lower priority: it will release soon
            holder = words_[word].load(std::memory_order_relaxed);
        }
    }

    template <typename P3>
    bool is_locked_by_this_thread(const P3& p) {
        return words_[index(p)].load(std::memory_order_relaxed) == state().id;
    }

    void unlock_all_points_locked_by_this_thread() {
        State& me = state();
        for (const uint32_t word : me.locked) words_[word].store(0, std::memory_order_release);
        me.locked.clear();
    }

    template <typename P3>
    void unlock_all_tls_locked_locations_but_one_point(const P3& p) {
        State& me = state();
        const uint32_t keep = index(p);
        bool kept = false;
        for (const uint32_t word : me.locked) {
            if (word == keep) { kept = true; continue; }
            words_[word].store(0, std::memory_order_release);
        }
        me.locked.clear();
        if (kept) me.locked.push_back(keep);
    }

private:
    struct State {
        const VertexLocks* owner = nullptr;
        uint32_t id = 0;
        std::vector<uint32_t> locked;
    };
    State& state() {
        thread_local State st;
        if (st.owner != this) {
            static std::atomic<uint32_t> next_id{0};
            st.owner = this;
            st.id = ++next_id;
            if (st.id == 0) st.id = ++next_id;
            st.locked.clear();
        }
        return st;
    }
    template <typename P3>
    uint32_t index(const P3& p) const {
        const uint64_t a = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(&p));
        return static_cast<uint32_t>((a * UINT64_C(0x9E3779B97F4A7C15)) >> (64 - bits_));
    }

    CGAL::Bbox_3 bbox_;
    int bits_;
    std::atomic<uint32_t>* words_;
};

using Delaunay = CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Default, VertexLocks>;
using Vertex_handle = Delaunay::Vertex_handle;
using Cell_handle = Delaunay::Cell_handle;

struct Vec3 {
    double X = 0, Y = 0, Z = 0;
    Vec3() = default;
    Vec3(double x, double y, double z) : X(x), Y(y), Z(z) {}
    double x() const { return X; }
    double y() const { return Y; }
    double z() const { return Z; }
};

uint64_t mix64(uint64_t value) noexcept {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

uint64_t point_fingerprint(double x, double y, double z) noexcept {
    // +0.0 and -0.0 are the same point.
    const uint64_t bx = (x == 0.0) ? 0 : std::bit_cast<uint64_t>(x);
    const uint64_t by = (y == 0.0) ? 0 : std::bit_cast<uint64_t>(y);
    const uint64_t bz = (z == 0.0) ? 0 : std::bit_cast<uint64_t>(z);
    return mix64(bx) ^ std::rotl(mix64(by), 21) ^ std::rotl(mix64(bz), 42);
}

using Point_with_info = std::pair<Point_3, unsigned>;
using Accessor        = CGAL::First_of_pair_property_map<Point_with_info>;
using Sort_traits_3   = CGAL::Spatial_sort_traits_adapter_3<K, Accessor>;

// The insertion order of CGAL::spatial_sort (BRIO with Hilbert order inside
// each level, median policy), computed with more parallelism.  CGAL shuffles
// the points serially, sorts the levels one after another and forks only at
// the first octant split.  This version replays the same shuffle on an index
// array and gathers in parallel, sorts the levels concurrently and forks the
// median recursion at every level above a cutoff.  The splits are the same
// nth_element calls on the same ranges, so the order is identical to CGAL's.
class ParallelSpatialSort {
public:
    using It = std::vector<Point_with_info>::iterator;

    void operator()(std::vector<Point_with_info>& pts) const {
        const std::size_t n = pts.size();
        if (n < 2) return;
        // CGAL::cpp98::random_shuffle(begin, end, rng): swap i with rng(i + 1).
        std::vector<uint32_t> perm(n);
        for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
        {
            boost::rand48 random;
            boost::random_number_generator<boost::rand48, std::ptrdiff_t> rng(random);
            for (std::ptrdiff_t i = 1; i < static_cast<std::ptrdiff_t>(n); ++i)
                std::swap(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(rng(i + 1))]);
        }
        std::vector<Point_with_info> shuffled(n);
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, 16384),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) shuffled[i] = pts[perm[i]];
            });
        pts.swap(shuffled);
        // Multiscale_sort(threshold 64, ratio 0.125): disjoint BRIO levels.
        std::vector<std::pair<It, It>> levels;
        It b = pts.begin(), e = pts.end();
        for (;;) {
            if (e - b >= kThreshold) {
                const It m = b + static_cast<std::ptrdiff_t>(double(e - b) * kRatio);
                levels.emplace_back(m, e);
                e = m;
            } else {
                levels.emplace_back(b, e);
                break;
            }
        }
        tbb::parallel_for(std::size_t(0), levels.size(), [&](std::size_t l) {
            sort<0, false, false, false>(levels[l].first, levels[l].second);
        });
    }

    static constexpr std::ptrdiff_t kThreshold = 64;
    static constexpr double kRatio = 0.125;

private:
    static constexpr std::ptrdiff_t kLimit = 8;          // CGAL's 3D threshold_hilbert
    static constexpr std::ptrdiff_t kFork = 1 << 13;

    template <int a, bool up>
    It split(It b, It e) const {
        return CGAL::internal::hilbert_split(b, e, CGAL::internal::Hilbert_cmp_3<Sort_traits_3, a, up>(k_));
    }

    template <int x, bool upx, bool upy, bool upz>
    void sort(It begin, It end) const {
        constexpr int y = (x + 1) % 3, z = (x + 2) % 3;
        if (end - begin <= kLimit) return;
        if (end - begin < kFork) {
            serial_.template recursive_sort<x, upx, upy, upz>(begin, end);
            return;
        }
        const It m0 = begin, m8 = end;
        const It m4 = split<x, upx>(m0, m8);
        It m2, m6;
        tbb::parallel_invoke([&] { m2 = split<y, upy>(m0, m4); },
                             [&] { m6 = split<y, !upy>(m4, m8); });
        It m1, m3, m5, m7;
        tbb::parallel_invoke([&] { m1 = split<z, upz>(m0, m2); },
                             [&] { m3 = split<z, !upz>(m2, m4); },
                             [&] { m5 = split<z, upz>(m4, m6); },
                             [&] { m7 = split<z, !upz>(m6, m8); });
        tbb::parallel_invoke([&] { sort<z, upz, upx, upy>(m0, m1); },
                             [&] { sort<y, upy, upz, upx>(m1, m2); },
                             [&] { sort<y, upy, upz, upx>(m2, m3); },
                             [&] { sort<x, upx, !upy, !upz>(m3, m4); },
                             [&] { sort<x, upx, !upy, !upz>(m4, m5); },
                             [&] { sort<y, !upy, upz, !upx>(m5, m6); },
                             [&] { sort<y, !upy, upz, !upx>(m6, m7); },
                             [&] { sort<z, !upz, !upx, upy>(m7, m8); });
    }

    Sort_traits_3 k_;
    CGAL::Hilbert_sort_median_3<Sort_traits_3, CGAL::Sequential_tag> serial_{k_, kLimit};
};

// Boundaries of the BRIO levels of the insertion order of n points, from 0 to
// n: level l spans [bounds[l], bounds[l + 1]).  Same recursion as above.
std::vector<std::size_t> brio_level_bounds(std::size_t n) {
    std::vector<std::size_t> bounds{n};
    for (std::size_t e = n; e >= std::size_t(ParallelSpatialSort::kThreshold);) {
        e = static_cast<std::size_t>(double(e) * ParallelSpatialSort::kRatio);
        bounds.push_back(e);
    }
    if (bounds.back() != 0) bounds.push_back(0);
    std::reverse(bounds.begin(), bounds.end());
    return bounds;
}

uint64_t morton_expand21(uint64_t v) noexcept {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8)  & 0x100f00f00f00f00fULL;
    v = (v | v << 4)  & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2)  & 0x1249249249249249ULL;
    return v;
}

// On a multi-socket machine the build runs on the first NUMA node.
int auto_numa_node() {
    const std::vector<tbb::numa_node_id> nodes = tbb::info::numa_nodes();
    return nodes.size() > 1 ? nodes.front() : tbb::task_arena::automatic;
}

// Below a few thousand points the fixed costs of the parallel insertion
// exceed the insertion itself; cap the number of workers by the size.
int threads_for_size(std::size_t n) {
    return static_cast<int>(std::min<std::size_t>(n / 512, 1024));
}

// Incremental Delaunay insertion stops scaling before all cores of a large
// machine are useful because of lock contention.
int auto_build_threads(int numa_id, bool big_build) {
    tbb::task_arena::constraints constraints;
    constraints.set_max_threads_per_core(1);
    if (numa_id != tbb::task_arena::automatic) constraints.set_numa_id(numa_id);
    const int available = std::max(1, tbb::info::default_concurrency(constraints));
    if (big_build && available >= 64) return 64;
    if (available >= 32) return 32;
    if (available >= 16) return 16;
    if (available >= 8)  return 8;
    return available;
}

// A node of the successor DAG: its point and the range of its successors in
// the flat successor array, which lists them in increasing insertion rank.
struct alignas(32) Node {
    double x, y, z;
    uint32_t firstChild, endChild;
};

inline double sqDist(double ax, double ay, double az, double bx, double by, double bz) noexcept {
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz);
}

// ============================================================================
// The index: successor DAG, start grid and queries.  Node indices follow the
// insertion order sigma (BRIO); `rank` is the order in which the insertions
// committed, which defines the direction of the successor edges.
// ============================================================================
class DpnnCore {
public:
    DpnnCore(const std::vector<Vec3>& input, int threads) {
        const std::size_t N = input.size();
        if (N > std::numeric_limits<uint32_t>::max())
            throw std::length_error("dpnn supports at most UINT32_MAX points");
        const int size_cap = threads_for_size(N);
        const bool serial = threads == 1 || size_cap < 2;
        // Builds of millions of points gain more from a second socket than
        // they lose to cross-node traffic; smaller builds stay on one node.
        const bool big_build = !serial && N >= 4000000;
        const int numa_id = (serial || big_build) ? tbb::task_arena::automatic : auto_numa_node();
        const int nthreads = serial ? 1 : std::min(
            size_cap, threads <= 0 ? auto_build_threads(numa_id, big_build) : threads);
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                               static_cast<std::size_t>(std::max(1, nthreads)));

        auto build = [&] {
            std::vector<Point_with_info> pts(N);
            std::vector<uint64_t> fingerprints(N);
            std::atomic<bool> all_finite{true};
            auto fill_point = [&](std::size_t i) {
                const double x = input[i].x(), y = input[i].y(), z = input[i].z();
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                    all_finite.store(false, std::memory_order_relaxed);
                pts[i] = Point_with_info(Point_3(x, y, z), static_cast<unsigned>(i));
                fingerprints[i] = point_fingerprint(x, y, z);
            };
            if (serial) {
                for (std::size_t i = 0; i < N; ++i) fill_point(i);
            } else {
                tbb::parallel_for(tbb::blocked_range<std::size_t>(0, N),
                    [&](const tbb::blocked_range<std::size_t>& r) {
                        for (std::size_t i = r.begin(); i != r.end(); ++i) fill_point(i);
                    });
            }
            if (!all_finite.load(std::memory_order_relaxed))
                throw std::invalid_argument("dpnn input coordinates must be finite");

            // Duplicates: keep the occurrence with the smallest index.
            if (serial) std::sort(fingerprints.begin(), fingerprints.end());
            else        tbb::parallel_sort(fingerprints.begin(), fingerprints.end());
            const bool maybe_duplicates =
                std::adjacent_find(fingerprints.begin(), fingerprints.end()) != fingerprints.end();
            std::vector<uint64_t>().swap(fingerprints);
            if (maybe_duplicates) {
                auto lexicographic = [](const Point_with_info& a, const Point_with_info& b) {
                    if (a.first.x() != b.first.x()) return a.first.x() < b.first.x();
                    if (a.first.y() != b.first.y()) return a.first.y() < b.first.y();
                    if (a.first.z() != b.first.z()) return a.first.z() < b.first.z();
                    return a.second < b.second;
                };
                if (serial) std::sort(pts.begin(), pts.end(), lexicographic);
                else        tbb::parallel_sort(pts.begin(), pts.end(), lexicographic);
                auto same_point = [](const Point_with_info& a, const Point_with_info& b) {
                    return a.first.x() == b.first.x() && a.first.y() == b.first.y() &&
                           a.first.z() == b.first.z();
                };
                pts.erase(std::unique(pts.begin(), pts.end(), same_point), pts.end());
            }

            // Insertion order sigma: BRIO levels, Hilbert order inside each.
            if (serial) CGAL::spatial_sort(pts.begin(), pts.end(), Sort_traits_3());
            else        ParallelSpatialSort()(pts);

            const std::size_t n = pts.size();
            std::vector<Vec3> sortedPts(n);
            std::vector<int> sortedIds(n);
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, 16384),
                [&](const tbb::blocked_range<std::size_t>& r) {
                    for (std::size_t i = r.begin(); i != r.end(); ++i) {
                        sortedPts[i] = Vec3(pts[i].first.x(), pts[i].first.y(), pts[i].first.z());
                        sortedIds[i] = static_cast<int>(pts[i].second);
                    }
                });
            std::vector<Point_with_info>().swap(pts);
            if (serial) buildSerial(sortedPts, std::move(sortedIds));
            else        buildParallel(sortedPts, std::move(sortedIds), nthreads);
        };

        if (serial) {
            build();
        } else {
            tbb::task_arena::constraints constraints;
            constraints.set_max_concurrency(nthreads).set_max_threads_per_core(1);
            if (numa_id != tbb::task_arena::automatic) constraints.set_numa_id(numa_id);
            tbb::task_arena arena(constraints);
            arena.execute(build);
        }
        buildStartGrid();
    }

    std::size_t size() const { return nodes.size(); }

    // ---- nearest neighbor ----

    uint32_t nearest(const Vec3& p) const { return static_cast<uint32_t>(ids[walk(p)]); }

    // Batch of Q queries (xyz triples); result[i] answers query i.
    void nearestBatch(const double* xyz, std::size_t Q, bool parallel, uint32_t* result) const {
        if (Q == 0) return;
        thread_local std::vector<uint32_t> order_tls;   // the calling thread's, see queryOrder
        queryOrder(xyz, Q, parallel, order_tls);
        const std::vector<uint32_t>& order = order_tls;
        auto run = [&](std::size_t position) {
            const std::size_t qi = order[position];
            result[qi] = nearest(Vec3(xyz[3 * qi], xyz[3 * qi + 1], xyz[3 * qi + 2]));
        };
        forBatch(Q, parallel, run);
    }

    // ---- k nearest neighbors ----

    // Exact Manifold k-NN for one query: `out` receives the k ids in
    // ascending distance.  The query descends the DAG greedily from the root
    // to the nearest neighbor, recording its path, then expands the r-th
    // nearest point found so far (adds all its successors) in round r, and
    // adds the nodes of the path, deepest first, as the expansion reaches
    // their rank.  The k-NN query starts at the root: its expansion revisits
    // the part of the walk that a start node of the grid would skip.
    void knnInto(const Vec3& p, std::size_t k, uint32_t* out) const {
        if (k == 1) { out[0] = nearest(p); return; }
        KnnScratch& scratch = knn_scratch_.local();
        scratch.prepare(nodes.size(), k);
        const uint32_t epoch = ++scratch.epoch;
        uint32_t* const stamp = scratch.stamp.data();
        double* const best_d = scratch.best_d.data();
        uint32_t* const best_n = scratch.best_n.data();
        std::size_t best_count = 0;
        const double px = p.x(), py = p.y(), pz = p.z();

        // Insert `node` into the window, sorted by distance.  Equal distances
        // keep their discovery order: a newcomer goes after every entry at its
        // distance, and once the window is full a candidate tied with the k-th
        // is not taken.  This keeps the traversal exact when distances tie: a
        // node tied with the one being expanded lands behind it and is still
        // expanded in a later round.
        auto add = [&](uint32_t node) {
            if (stamp[node] == epoch) return;
            stamp[node] = epoch;
            const Node& n = nodes[node];
            const double distance = sqDist(n.x, n.y, n.z, px, py, pz);
            if (best_count == k && distance >= best_d[k - 1]) return;
            std::size_t i = (best_count < k) ? best_count : k - 1;
            for (; i > 0; --i) {
                const double previous = best_d[i - 1];
                if (previous <= distance) break;
                best_d[i] = previous;
                best_n[i] = best_n[i - 1];
            }
            best_d[i] = distance;
            best_n[i] = node;
            if (best_count < k) ++best_count;
        };

        uint32_t current = root;
        double minimum = sqDist(nodes[current].x, nodes[current].y, nodes[current].z, px, py, pz);
        std::size_t path_count = 0;
        for (bool improved = true; improved;) {
            improved = false;
            if (path_count == scratch.path.size()) scratch.path.resize(2 * path_count);
            scratch.path[path_count++] = current;
            const Node& node = nodes[current];
            for (uint32_t edge = node.firstChild; edge < node.endChild; ++edge) {
                const uint32_t child = children[edge];
                const Node& c = nodes[child];
                const double distance = sqDist(c.x, c.y, c.z, px, py, pz);
                if (distance < minimum) {
                    minimum = distance;
                    current = child;
                    improved = true;
                    break;
                }
            }
        }

        const uint32_t* const path = scratch.path.data();
        add(path[path_count - 1]);
        std::ptrdiff_t path_pos = static_cast<std::ptrdiff_t>(path_count) - 2;
        for (std::size_t round = 1; round < k; ++round) {
            if (round > best_count)
                throw std::logic_error("Manifold k-NN candidate frontier became empty");
            const uint32_t selected = best_n[round - 1];
            while (path_pos >= 0 &&
                   rank[path[static_cast<std::size_t>(path_pos + 1)]] >= rank[selected]) {
                add(path[static_cast<std::size_t>(path_pos)]);
                --path_pos;
            }
            const Node& node = nodes[selected];
            for (uint32_t edge = node.firstChild; edge < node.endChild; ++edge)
                add(children[edge]);
        }
        if (best_count != k)
            throw std::logic_error("Manifold k-NN returned fewer than k candidates");
        for (std::size_t i = 0; i < k; ++i) out[i] = static_cast<uint32_t>(ids[best_n[i]]);
    }

    // Batch of Q queries; row i of `result` (k ids) answers query i.
    void knnBatch(const double* xyz, std::size_t Q, std::size_t k, bool parallel,
                  uint32_t* result) const {
        if (Q == 0 || k == 0) return;
        thread_local std::vector<uint32_t> order_tls;
        queryOrder(xyz, Q, parallel, order_tls);
        const std::vector<uint32_t>& order = order_tls;
        auto run = [&](std::size_t position) {
            const std::size_t qi = order[position];
            knnInto(Vec3(xyz[3 * qi], xyz[3 * qi + 1], xyz[3 * qi + 2]), k, result + qi * k);
        };
        forBatch(Q, parallel, run);
    }

private:
    RawVector<Node>       nodes;      // the DAG, indexed in insertion order sigma
    RawVector<uint32_t>   children;   // successor lists, each in increasing rank
    std::vector<int>      ids;        // node -> input index
    std::vector<uint32_t> rank;       // node -> commit rank
    uint32_t              root = 0;   // the node of rank 0

    // Per-thread k-NN scratch: an epoch-stamped membership array (cleared in
    // O(1) by bumping the epoch), the sorted top-k window and the greedy path.
    struct KnnScratch {
        std::vector<uint32_t> stamp;
        std::vector<double>   best_d;
        std::vector<uint32_t> best_n;
        std::vector<uint32_t> path;
        uint32_t              epoch = 0;

        void prepare(std::size_t node_count, std::size_t k) {
            if (stamp.size() != node_count) {
                stamp.assign(node_count, 0);
                epoch = 0;
            } else if (epoch == std::numeric_limits<uint32_t>::max()) {
                std::fill(stamp.begin(), stamp.end(), 0u);
                epoch = 0;
            }
            if (best_d.size() < k) { best_d.resize(k); best_n.resize(k); }
            if (path.size() < 64) path.resize(64);
        }
    };
    mutable tbb::enumerable_thread_specific<KnnScratch> knn_scratch_;

    // Runs fn(position) for every position of a batch, in chunks of queries
    // that are consecutive in Morton order.
    template <class Fn>
    static void forBatch(std::size_t Q, bool parallel, Fn&& fn) {
        if (!parallel) {
            for (std::size_t position = 0; position < Q; ++position) fn(position);
            return;
        }
        constexpr std::size_t kGrain = 256;
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, Q, kGrain),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t position = r.begin(); position != r.end(); ++position) fn(position);
            }, tbb::simple_partitioner());
    }

    // ------------------------------------------------------------------------
    // Query order.  The batch is sorted by 24-bit Morton codes (8 bits per
    // axis) over the box of the start grid (or the box of the batch without a
    // grid; queries outside the box are clamped to it), so that consecutive
    // queries walk the same part of the DAG.  The sort is a two-pass LSD radix
    // sort of the codes in four parallel phases: keys and first histogram,
    // first scatter, second histogram, second scatter.  The order only affects
    // locality, never results.  The scratch buffers belong to the calling
    // thread: a fresh multi-megabyte buffer per batch costs page faults that
    // exceed the sort itself.
    // ------------------------------------------------------------------------
    void queryOrder(const double* xyz, std::size_t Q, bool parallel,
                    std::vector<uint32_t>& order) const {
        order.resize(Q);
        if (Q < 2) {
            for (std::size_t i = 0; i < Q; ++i) order[i] = static_cast<uint32_t>(i);
            return;
        }
        double lo[3], scale[3];
        if (gridRes > 0) {
            for (int a = 0; a < 3; ++a) {
                lo[a] = gridLo[a];
                scale[a] = 256.0 / (gridStep[a] * gridRes);
            }
        } else {
            struct Box { double lo[3], hi[3]; };
            auto grow = [](Box b, const double* q) {
                for (int a = 0; a < 3; ++a) { b.lo[a] = std::min(b.lo[a], q[a]); b.hi[a] = std::max(b.hi[a], q[a]); }
                return b;
            };
            const Box first = {{xyz[0], xyz[1], xyz[2]}, {xyz[0], xyz[1], xyz[2]}};
            Box box = first;
            if (parallel) {
                box = tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, Q, 4096), first,
                    [&](const tbb::blocked_range<std::size_t>& r, Box b) {
                        for (std::size_t i = r.begin(); i != r.end(); ++i) b = grow(b, xyz + 3 * i);
                        return b;
                    },
                    [](Box a, const Box& b) {
                        for (int d = 0; d < 3; ++d) { a.lo[d] = std::min(a.lo[d], b.lo[d]); a.hi[d] = std::max(a.hi[d], b.hi[d]); }
                        return a;
                    });
            } else {
                for (std::size_t i = 0; i < Q; ++i) box = grow(box, xyz + 3 * i);
            }
            for (int a = 0; a < 3; ++a) {
                const double span = box.hi[a] - box.lo[a];
                lo[a] = box.lo[a];
                scale[a] = span > 0 ? 256.0 / span : 0.0;
            }
        }
        // The calling thread's buffers, bound to references that the parallel
        // lambdas capture (naming a thread_local inside a lambda run by a worker
        // would refer to the worker's own instance).
        thread_local std::vector<uint64_t> keys_tls, tmp_tls;
        thread_local std::vector<uint32_t> hist_tls, base_tls;
        std::vector<uint64_t>& keys = keys_tls;
        std::vector<uint64_t>& tmp = tmp_tls;
        std::vector<uint32_t>& hist = hist_tls;
        std::vector<uint32_t>& base = base_tls;
        keys.resize(Q);
        tmp.resize(Q);
        auto key_of = [&](std::size_t i) {
            uint64_t code = 0;
            for (int a = 0; a < 3; ++a) {
                const double t = (xyz[3 * i + a] - lo[a]) * scale[a];
                const uint64_t c = t <= 0 ? 0 : t >= 255.0 ? 255 : static_cast<uint64_t>(t);
                code |= morton_expand21(c) << a;
            }
            return (code << 32) | static_cast<uint64_t>(i);
        };
        constexpr int kDigit = 12;
        constexpr std::size_t kBuckets = std::size_t(1) << kDigit;
        const std::size_t blocks = parallel
            ? std::max<std::size_t>(1, std::min<std::size_t>(
                  static_cast<std::size_t>(tbb::this_task_arena::max_concurrency()), Q / 4096))
            : 1;
        const std::size_t block_len = (Q + blocks - 1) / blocks;
        auto for_blocks = [&](auto&& fn) {
            if (blocks == 1) { fn(0); return; }
            tbb::parallel_for(std::size_t(0), blocks, fn, tbb::static_partitioner());
        };
        hist.assign(blocks * kBuckets, 0u);
        base.resize(kBuckets);
        // Exclusive prefix over (bucket, block), bucket-major.
        auto prefix = [&]() {
            std::fill(base.begin(), base.end(), 0u);
            for (std::size_t b = 0; b < blocks; ++b) {
                const uint32_t* h = hist.data() + b * kBuckets;
                for (std::size_t d = 0; d < kBuckets; ++d) base[d] += h[d];
            }
            uint32_t offset = 0;
            for (std::size_t d = 0; d < kBuckets; ++d) {
                const uint32_t c = base[d];
                base[d] = offset;
                offset += c;
            }
            for (std::size_t b = 0; b < blocks; ++b) {
                uint32_t* h = hist.data() + b * kBuckets;
                for (std::size_t d = 0; d < kBuckets; ++d) {
                    const uint32_t c = h[d];
                    h[d] = base[d];
                    base[d] += c;
                }
            }
        };
        const int shift0 = 32, shift1 = 32 + kDigit;
        for_blocks([&](std::size_t b) {          // keys and the first histogram
            uint32_t* h = hist.data() + b * kBuckets;
            for (std::size_t i = b * block_len, e = std::min(Q, i + block_len); i < e; ++i) {
                keys[i] = key_of(i);
                ++h[(keys[i] >> shift0) & (kBuckets - 1)];
            }
        });
        prefix();
        for_blocks([&](std::size_t b) {          // first scatter
            uint32_t* h = hist.data() + b * kBuckets;
            for (std::size_t i = b * block_len, e = std::min(Q, i + block_len); i < e; ++i)
                tmp[h[(keys[i] >> shift0) & (kBuckets - 1)]++] = keys[i];
        });
        std::fill(hist.begin(), hist.end(), 0u);
        for_blocks([&](std::size_t b) {          // second histogram
            uint32_t* h = hist.data() + b * kBuckets;
            for (std::size_t i = b * block_len, e = std::min(Q, i + block_len); i < e; ++i)
                ++h[(tmp[i] >> shift1) & (kBuckets - 1)];
        });
        prefix();
        for_blocks([&](std::size_t b) {          // second scatter, straight into the order
            uint32_t* h = hist.data() + b * kBuckets;
            for (std::size_t i = b * block_len, e = std::min(Q, i + block_len); i < e; ++i)
                order[h[(tmp[i] >> shift1) & (kBuckets - 1)]++] = static_cast<uint32_t>(tmp[i]);
        });
    }

    // ------------------------------------------------------------------------
    // Start grid.  A uniform G^3 grid over the data box grown by half its span
    // on every side.  The walk of a point visits node c exactly when c is the
    // nearest neighbor of the point among the points inserted up to c, that
    // is, when the point lies in the Voronoi cell of c right after the
    // insertion of c.  If the walks of all eight corners of a grid cell visit
    // c, the corners lie in that Voronoi cell; the Voronoi cell is convex, so
    // the whole grid cell does, and c is on the walk of every query in it.
    // Each grid cell stores the deepest such node, and a nearest-neighbor walk
    // starts there instead of at the root; from c on, it takes the same steps.
    // ------------------------------------------------------------------------
    std::vector<uint32_t> gridStart;   // cell -> start node
    int gridRes = 0;                   // cells per axis; 0 = no grid
    double gridLo[3] = {0, 0, 0}, gridStep[3] = {1, 1, 1};

    struct Step { uint32_t rank, node; };
    static constexpr uint32_t kPathSlots = 32;

    // Greedy walk of (px, py, pz) from `cur`, a node on its walk: move to the
    // first successor (in rank order) that is closer, until none is.
    uint32_t walkFrom(uint32_t cur, double px, double py, double pz) const {
        double minDis = sqDist(nodes[cur].x, nodes[cur].y, nodes[cur].z, px, py, pz);
        for (bool moved = true; moved;) {
            moved = false;
            const uint32_t end = nodes[cur].endChild;
            for (uint32_t i = nodes[cur].firstChild; i < end; ++i) {
                const Node& c = nodes[children[i]];
                const double d = sqDist(c.x, c.y, c.z, px, py, pz);
                if (d < minDis) {
                    minDis = d;
                    cur = children[i];
                    moved = true;
                    break;
                }
            }
        }
        return cur;
    }

    // Nearest node of a query, from the start node of its grid cell (the
    // root without a grid or outside it).
    uint32_t walk(const Vec3& p) const {
        return walkFrom(gridStartFor(p.x(), p.y(), p.z()), p.x(), p.y(), p.z());
    }

    // The same walk, recording the nodes met in `out` (at most kPathSlots;
    // past that the last slot keeps the deepest node, which keeps the record
    // sorted by rank).  Returns the count.
    uint32_t walkRecord(double px, double py, double pz, uint32_t cur, Step* out) const {
        double minDis = sqDist(nodes[cur].x, nodes[cur].y, nodes[cur].z, px, py, pz);
        uint32_t count = 0;
        out[count++] = {rank[cur], cur};
        for (bool moved = true; moved;) {
            moved = false;
            const uint32_t end = nodes[cur].endChild;
            for (uint32_t i = nodes[cur].firstChild; i < end; ++i) {
                const uint32_t c = children[i];
                const double d = sqDist(nodes[c].x, nodes[c].y, nodes[c].z, px, py, pz);
                if (d < minDis) {
                    minDis = d;
                    cur = c;
                    moved = true;
                    out[count < kPathSlots ? count++ : kPathSlots - 1] = {rank[c], c};
                    break;
                }
            }
        }
        return count;
    }

    // Start nodes of a G^3 grid over gridLo + [0, G) * step: per cell, the
    // deepest node on the walks of all eight corners.  With G even, the G/2
    // grid is computed first.  A corner lies in up to eight coarse cells, and
    // the start nodes of all of them are on its walk; the walk begins at the
    // shallowest of them.  Every fine cell lies in one coarse cell, whose start
    // node its eight corners share, so the deepest node they share comes after
    // the beginning of each of their records.
    void gridStarts(int G, const double step[3], std::vector<uint32_t>& start) const {
        std::vector<uint32_t> coarse;
        const int Gc = (G % 2 == 0 && G / 2 >= 8) ? G / 2 : 0;
        if (Gc) {
            const double coarseStep[3] = {2 * step[0], 2 * step[1], 2 * step[2]};
            gridStarts(Gc, coarseStep, coarse);
        }
        const std::size_t C = std::size_t(G) + 1;
        auto coarse_cell = [&](std::size_t i, std::size_t j, std::size_t k) {
            const std::size_t c = std::size_t(Gc) - 1;
            return std::min(i, c) + std::min(j, c) * Gc + std::min(k, c) * std::size_t(Gc) * Gc;
        };
        // Shallowest start among the coarse cells that contain grid point (i, j, k).
        auto corner_start = [&](std::size_t i, std::size_t j, std::size_t k) {
            std::size_t lo[3], hi[3];
            const std::size_t idx[3] = {i, j, k};
            for (int a = 0; a < 3; ++a) {
                hi[a] = std::min(idx[a] / 2, std::size_t(Gc) - 1);
                lo[a] = (idx[a] % 2 == 0 && idx[a] > 0) ? idx[a] / 2 - 1 : hi[a];
            }
            uint32_t best = coarse[coarse_cell(hi[0], hi[1], hi[2])];
            for (std::size_t x = lo[0]; x <= hi[0]; ++x)
                for (std::size_t y = lo[1]; y <= hi[1]; ++y)
                    for (std::size_t z = lo[2]; z <= hi[2]; ++z) {
                        const uint32_t v = coarse[coarse_cell(x, y, z)];
                        if (rank[v] < rank[best]) best = v;
                    }
            return best;
        };
        // Uninitialized: the walks write what the cells read.
        std::unique_ptr<Step[]> path_store(new Step[C * C * C * kPathSlots]);
        std::unique_ptr<uint8_t[]> length_store(new uint8_t[C * C * C]);
        Step* const paths = path_store.get();
        uint8_t* const lengths = length_store.get();
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, C * C * C, 512),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t c = r.begin(); c != r.end(); ++c) {
                    const std::size_t i = c % C, j = (c / C) % C, k = c / (C * C);
                    const uint32_t from = Gc ? corner_start(i, j, k) : root;
                    lengths[c] = static_cast<uint8_t>(walkRecord(
                        gridLo[0] + step[0] * double(i), gridLo[1] + step[1] * double(j),
                        gridLo[2] + step[2] * double(k), from, paths + c * kPathSlots));
                }
            });
        start.assign(std::size_t(G) * G * G, root);
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, std::size_t(G) * G * G, 1024),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t cell = r.begin(); cell != r.end(); ++cell) {
                    const std::size_t i = cell % G, j = (cell / G) % G, k = cell / (std::size_t(G) * G);
                    const Step* p[8];
                    uint32_t len[8];
                    for (int b = 0; b < 8; ++b) {
                        const std::size_t c = (i + (b & 1)) + (j + ((b >> 1) & 1)) * C + (k + (b >> 2)) * C * C;
                        p[b] = paths + c * kPathSlots;
                        len[b] = lengths[c];
                    }
                    uint32_t best = root;
                    for (uint32_t t = len[0]; t-- > 0;) {    // deepest first
                        const Step s = p[0][t];
                        bool everywhere = true;
                        for (int b = 1; b < 8 && everywhere; ++b) {
                            const Step* q = std::lower_bound(p[b], p[b] + len[b], s.rank,
                                [](const Step& x, uint32_t value) { return x.rank < value; });
                            everywhere = q != p[b] + len[b] && q->rank == s.rank;
                        }
                        if (everywhere) { best = s.node; break; }
                    }
                    if (Gc) {          // only matters when a record overflowed
                        const uint32_t parent = coarse[coarse_cell(i / 2, j / 2, k / 2)];
                        if (rank[parent] > rank[best]) best = parent;
                    }
                    start[cell] = best;
                }
            });
    }

    // G is chosen from the size: about one cell per three points, a multiple
    // of 8 from 16 to 64 (at most 1 MB of start nodes).  Below 8192 points no
    // grid is built, as the walks are short anyway.
    void buildStartGrid() {
        const std::size_t N = nodes.size();
        if (N < 8192) return;
        const int G = std::clamp(8 * static_cast<int>(std::lround(std::cbrt(double(N) / 3.0) / 8.0)), 16, 64);
        constexpr double kMargin = 0.5;
        struct Box { double lo[3], hi[3]; };
        const Box empty = {{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}};
        const Box box = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, N, 16384), empty,
            [&](const tbb::blocked_range<std::size_t>& r, Box b) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const double c[3] = {nodes[i].x, nodes[i].y, nodes[i].z};
                    for (int d = 0; d < 3; ++d) { b.lo[d] = std::min(b.lo[d], c[d]); b.hi[d] = std::max(b.hi[d], c[d]); }
                }
                return b;
            },
            [](Box a, const Box& b) {
                for (int d = 0; d < 3; ++d) { a.lo[d] = std::min(a.lo[d], b.lo[d]); a.hi[d] = std::max(a.hi[d], b.hi[d]); }
                return a;
            });
        for (int d = 0; d < 3; ++d) {
            const double span = std::max(box.hi[d] - box.lo[d], 1e-12);
            gridLo[d] = box.lo[d] - kMargin * span;
            gridStep[d] = (1 + 2 * kMargin) * span / G;
        }
        gridStarts(G, gridStep, gridStart);
        gridRes = G;
    }

    // Cell of a point, or -1 outside the grid.
    std::ptrdiff_t gridCell(double px, double py, double pz) const {
        if (gridRes == 0) return -1;
        const double fx = (px - gridLo[0]) / gridStep[0], fy = (py - gridLo[1]) / gridStep[1],
                     fz = (pz - gridLo[2]) / gridStep[2];
        if (!(fx >= 0 && fy >= 0 && fz >= 0 && fx < gridRes && fy < gridRes && fz < gridRes)) return -1;
        return static_cast<std::ptrdiff_t>(std::size_t(fx) + std::size_t(fy) * gridRes +
                                           std::size_t(fz) * std::size_t(gridRes) * gridRes);
    }

    uint32_t gridStartFor(double px, double py, double pz) const {
        const std::ptrdiff_t cell = gridCell(px, py, pz);
        return cell < 0 ? root : gridStart[static_cast<std::size_t>(cell)];
    }

    // ------------------------------------------------------------------------
    // Construction.
    // ------------------------------------------------------------------------

    // The finite neighbors of v, i.e. the vertices of its star.  This is the
    // walk of CGAL's finite_adjacent_vertices, with the same per-cell and
    // per-vertex marks, but it reuses the caller's buffers instead of
    // allocating on every call, which with many workers contends on the heap.
    // In the parallel build every cell and vertex touched here lies in the
    // region that the calling worker still holds locked.
    static void starNeighbors(const Delaunay& dt, Vertex_handle v, std::vector<Vertex_handle>& out,
                              std::vector<Cell_handle>& star) {
        out.clear();
        star.clear();
        if (dt.dimension() != 3) {
            dt.finite_adjacent_vertices(v, std::back_inserter(out));
            return;
        }
        const Cell_handle first = v->cell();
        first->tds_data().mark_in_conflict();
        star.push_back(first);
        for (std::size_t head = 0; head < star.size(); ++head) {
            const Cell_handle c = star[head];
            for (int i = 0; i < 4; ++i) {
                const Vertex_handle u = c->vertex(i);
                if (u == v) continue;
                const Cell_handle next = c->neighbor(i);
                if (next->tds_data().is_clear()) {
                    next->tds_data().mark_in_conflict();
                    star.push_back(next);
                }
                if (!u->visited_for_vertex_extractor) {
                    u->visited_for_vertex_extractor = true;
                    out.push_back(u);
                }
            }
        }
        for (const Cell_handle c : star) c->tds_data().clear();
        std::size_t kept = 0;
        for (const Vertex_handle u : out) {
            u->visited_for_vertex_extractor = false;
            if (!dt.is_infinite(u)) out[kept++] = u;
        }
        out.resize(kept);
    }

    // Serial build: insertion in the order sigma; the successors of a point
    // are the later points adjacent to it at their insertion.
    void buildSerial(const std::vector<Vec3>& pts, std::vector<int>&& sortedIds) {
        const std::size_t N = pts.size();
        ids = std::move(sortedIds);
        rank.resize(N);
        for (std::size_t i = 0; i < N; ++i) rank[i] = static_cast<uint32_t>(i);
        root = 0;
        std::vector<std::vector<uint32_t>> successors(N);
        {
            Delaunay dt;
            Vertex_handle hint;
            std::vector<Vertex_handle> vss;
            std::vector<Cell_handle> star;
            for (std::size_t i = 0; i < N; ++i) {
                hint = dt.insert(Delaunay::Point(pts[i].x(), pts[i].y(), pts[i].z()), hint);
                hint->info() = static_cast<unsigned>(i);
                starNeighbors(dt, hint, vss, star);
                for (Vertex_handle vv : vss) {
                    const uint32_t j = static_cast<uint32_t>(vv->info());
                    if (j < i) successors[j].push_back(static_cast<uint32_t>(i));
                }
            }
        }
        std::size_t total = 0;
        for (const auto& list : successors) total += list.size();
        if (total > std::numeric_limits<uint32_t>::max())
            throw std::length_error("dpnn DAG has too many edges");
        nodes.resize(N);
        children.resize(total);
        uint32_t offset = 0;
        for (std::size_t i = 0; i < N; ++i) {
            Node& n = nodes[i];
            n.x = pts[i].x(); n.y = pts[i].y(); n.z = pts[i].z();
            n.firstChild = offset;
            std::copy(successors[i].begin(), successors[i].end(), children.begin() + offset);
            offset += static_cast<uint32_t>(successors[i].size());
            n.endChild = offset;
        }
    }

    // Parallel build.  The points are inserted concurrently, one BRIO level of
    // sigma at a time: no point of a level is attempted before every point of
    // the earlier levels has committed, so the commit order pi never mixes two
    // levels; inside a level the commit order is free.  The successor edges
    // follow pi (a neighbor present at an insertion always committed earlier),
    // and every successor list is sorted by pi.  Keeping the levels preserves
    // the logarithmic expected query cost of the serial order.
    void buildParallel(const std::vector<Vec3>& pts, std::vector<int>&& sortedIds, int nthreads) {
        const std::size_t N = pts.size();
        ids = std::move(sortedIds);
        if (N == 0) return;

        struct Box { double lo[3], hi[3]; };
        const Box empty = {{1e30, 1e30, 1e30}, {-1e30, -1e30, -1e30}};
        const Box box = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, N, 16384), empty,
            [&](const tbb::blocked_range<std::size_t>& r, Box b) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const Vec3& q = pts[i];
                    b.lo[0] = std::min(b.lo[0], q.x()); b.hi[0] = std::max(b.hi[0], q.x());
                    b.lo[1] = std::min(b.lo[1], q.y()); b.hi[1] = std::max(b.hi[1], q.y());
                    b.lo[2] = std::min(b.lo[2], q.z()); b.hi[2] = std::max(b.hi[2], q.z());
                }
                return b;
            },
            [](Box a, const Box& b) {
                for (int d = 0; d < 3; ++d) {
                    a.lo[d] = std::min(a.lo[d], b.lo[d]);
                    a.hi[d] = std::max(a.hi[d], b.hi[d]);
                }
                return a;
            });
        const double xmin = box.lo[0], ymin = box.lo[1], zmin = box.lo[2];
        const double xmax = box.hi[0], ymax = box.hi[1], zmax = box.hi[2];
        const double diag = std::sqrt((xmax-xmin)*(xmax-xmin)+(ymax-ymin)*(ymax-ymin)+(zmax-zmin)*(zmax-zmin));
        const double pad = (diag > 0 ? diag : 1.0) * 1e-3 + 1e-9;

        // One lock word per vertex (hashed), at least 4N words.
        int lock_bits = 16;
        while (lock_bits < 26 && (std::size_t(1) << lock_bits) < 4 * N) ++lock_bits;
        VertexLocks locks(CGAL::Bbox_3(xmin-pad, ymin-pad, zmin-pad, xmax+pad, ymax+pad, zmax+pad),
                          lock_bits);
        Delaunay dt;
        dt.set_lock_data_structure(&locks);

        std::atomic<uint32_t> piCounter{0};
        std::atomic<uint64_t> order_violations{0};
        using EdgeVec = std::vector<uint64_t>;          // (parent pi << 32) | child pi
        tbb::enumerable_thread_specific<EdgeVec> tls_edges;
        std::vector<uint32_t> sigmaOfPi(N);
        const std::size_t edges_per_worker = (N / static_cast<std::size_t>(std::max(1, nthreads)) + 1) * 20;
        auto local_edges = [&]() -> EdgeVec& {
            EdgeVec& buf = tls_edges.local();
            if (buf.capacity() == 0) buf.reserve(edges_per_worker);
            return buf;
        };

        // The new vertex k becomes a successor of each of its neighbors.
        auto record_neighbors = [&](Vertex_handle v, uint32_t k, EdgeVec& buf) {
            thread_local std::vector<Cell_handle> star;
            thread_local std::vector<Vertex_handle> vss;
            starNeighbors(dt, v, vss, star);
            for (Vertex_handle vv : vss) {
                const uint32_t j = static_cast<uint32_t>(vv->info());
                if (j == k) continue;
                if (j > k) order_violations.fetch_add(1, std::memory_order_relaxed);
                buf.emplace_back((uint64_t(std::min(j, k)) << 32) | uint64_t(std::max(j, k)));
            }
        };

        // Point-location hints.  CGAL's parallel locate locks every cell it
        // walks through until the insertion finishes, so a walk that starts far
        // away both fails easily and blocks other workers along its path.
        // Every cell of a hint grid remembers one committed vertex inside it,
        // and an insertion starts from the vertex of its own cell.
        const int hint_res = std::min(300, std::max(50,
            static_cast<int>(1.5 * std::cbrt(static_cast<double>(N))) + std::max(0, nthreads - 8)));
        const double hint_scale[3] = {
            hint_res / ((xmax + pad) - (xmin - pad)),
            hint_res / ((ymax + pad) - (ymin - pad)),
            hint_res / ((zmax + pad) - (zmin - pad))};
        auto hint_cell = [&](const Vec3& q) -> std::size_t {
            auto axis = [&](double v, double lo, double scale) {
                return std::clamp(static_cast<int>((v - lo) * scale), 0, hint_res - 1);
            };
            const std::size_t ix = axis(q.x(), xmin - pad, hint_scale[0]);
            const std::size_t iy = axis(q.y(), ymin - pad, hint_scale[1]);
            const std::size_t iz = axis(q.z(), zmin - pad, hint_scale[2]);
            return (iz * hint_res + iy) * hint_res + ix;
        };
        // calloc'ed: untouched cells cost nothing, and 0 means "no vertex yet".
        std::unique_ptr<std::atomic<uint32_t>, decltype(&std::free)> hint_grid(
            static_cast<std::atomic<uint32_t>*>(std::calloc(
                std::size_t(hint_res) * hint_res * hint_res, sizeof(std::atomic<uint32_t>))),
            &std::free);
        if (!hint_grid) throw std::bad_alloc();
        std::vector<Vertex_handle> vertex_of_sigma(N);
        auto near_hint = [&](std::size_t i, Vertex_handle fallback) -> Vertex_handle {
            const uint32_t s = hint_grid.get()[hint_cell(pts[i])].load(std::memory_order_acquire);
            return s ? vertex_of_sigma[s - 1] : fallback;
        };

        auto commit = [&](Vertex_handle v, std::size_t sigma_i, EdgeVec& buf) {
            const uint32_t k = piCounter.fetch_add(1, std::memory_order_relaxed);
            v->info() = k;
            sigmaOfPi[k] = static_cast<uint32_t>(sigma_i);
            record_neighbors(v, k, buf);
            vertex_of_sigma[sigma_i] = v;
            hint_grid.get()[hint_cell(pts[sigma_i])].store(static_cast<uint32_t>(sigma_i + 1),
                                                           std::memory_order_release);
        };

        // A serial prefix seeds enough geometry that the first parallel
        // insertions do not all conflict in one small region.  CGAL's parallel
        // insertion also needs a full-dimensional triangulation.
        Vertex_handle hint;
        std::size_t num_seq = std::min<std::size_t>(N, std::max<std::size_t>(
            128, std::min<std::size_t>(4096, std::max<std::size_t>(
                64 * static_cast<std::size_t>(nthreads),
                static_cast<std::size_t>(std::sqrt(double(N)) * 0.5)))));
        {
            EdgeVec& buf = local_edges();
            for (std::size_t i = 0; i < num_seq; ++i) {
                hint = dt.insert(Delaunay::Point(pts[i].x(), pts[i].y(), pts[i].z()), hint);
                commit(hint, i, buf);
            }
            while (num_seq < N && dt.dimension() < 3) {
                hint = dt.insert(Delaunay::Point(pts[num_seq].x(), pts[num_seq].y(), pts[num_seq].z()), hint);
                commit(hint, num_seq, buf);
                ++num_seq;
            }
        }
        tbb::enumerable_thread_specific<Vertex_handle> tls_hint(hint);

        // One attempt to insert point i: it fails when a needed lock is held by
        // a thread of higher priority.
        auto try_insert_once = [&](std::size_t i, Vertex_handle& h, EdgeVec& buf) {
            const Delaunay::Point p(pts[i].x(), pts[i].y(), pts[i].z());
            bool success = false;
            h = near_hint(i, h);
            if (dt.try_lock_vertex(h)) {
                bool could_lock_zone = false;
                Vertex_handle nv = dt.insert(p, h, &could_lock_zone);
                if (could_lock_zone) {
                    commit(nv, i, buf);
                    h = nv;
                    success = true;
                }
            }
            dt.unlock_all_elements();
            return success;
        };
        // Retrying in a tight loop livelocks once the workers are
        // oversubscribed; back off after a few failures.
        auto insert_spinning = [&](std::size_t i, Vertex_handle& h, EdgeVec& buf) {
            for (unsigned attempt = 0; !try_insert_once(i, h, buf); ++attempt) {
                if (attempt >= 8) {
                    std::this_thread::yield();
                    attempt = 4;
                }
            }
        };

        // One level [first, last).  Workers take sigma-ordered blocks from a
        // counter, which keeps the Hilbert locality of the insertions.  A failed
        // insertion goes to the worker's own retry list and is retried between
        // blocks, so nobody waits for the last contended points.
        auto insert_level = [&](std::size_t first, std::size_t last, std::size_t bsize) {
            const std::size_t blocks = (last - first + bsize - 1) / bsize;
            std::atomic<std::size_t> next_block{0};
            std::atomic<std::size_t> inserted{0};
            const int workers = std::max(1, tbb::this_task_arena::max_concurrency());
            tbb::parallel_for(tbb::blocked_range<int>(0, workers, 1),
                [&](const tbb::blocked_range<int>&) {
                    Vertex_handle& h = tls_hint.local();
                    EdgeVec& buf = local_edges();
                    std::vector<std::pair<std::size_t, unsigned>> retry;   // (sigma, failures)
                    std::size_t done = 0;
                    auto retry_pass = [&] {
                        std::size_t out = 0;
                        for (std::size_t r = 0; r < retry.size(); ++r) {
                            auto [i, fails] = retry[r];
                            if (fails >= 64) {
                                insert_spinning(i, h, buf);
                                ++done;
                            } else if (try_insert_once(i, h, buf)) {
                                ++done;
                            } else {
                                retry[out++] = {i, fails + 1};
                            }
                        }
                        retry.resize(out);
                    };
                    for (;;) {
                        const std::size_t b = next_block.fetch_add(1, std::memory_order_relaxed);
                        if (b >= blocks) break;
                        const std::size_t lo = first + b * bsize;
                        const std::size_t hi = std::min(last, lo + bsize);
                        for (std::size_t i = lo; i < hi; ++i) {
                            if (try_insert_once(i, h, buf)) ++done;
                            else retry.push_back({i, 1u});
                        }
                        if (!retry.empty()) retry_pass();
                    }
                    while (!retry.empty()) {
                        retry_pass();
                        if (!retry.empty()) std::this_thread::yield();
                    }
                    inserted.fetch_add(done, std::memory_order_relaxed);
                }, tbb::simple_partitioner());
            if (inserted.load() != last - first)
                throw std::logic_error("parallel insertion left points uncommitted");
        };

        const std::size_t workers = static_cast<std::size_t>(std::max(1, nthreads));
        const std::size_t span = std::max(std::min<std::size_t>(1048576, std::max<std::size_t>(65536, N / 16)),
                                          8 * workers);
        const std::size_t block_size = std::max<std::size_t>(64, span / (4 * workers));
        const std::vector<std::size_t> levels = brio_level_bounds(N);
        for (std::size_t l = 0; l + 1 < levels.size(); ++l) {
            const std::size_t lo = std::max(levels[l], num_seq), hi = levels[l + 1];
            if (lo >= hi) continue;
            // A small level still reaches all workers: at most a quarter of
            // its per-worker share per block.
            const std::size_t bsize = std::min(block_size, std::max<std::size_t>(64, (hi - lo) / (4 * workers)));
            insert_level(lo, hi, bsize);
        }
        dt.set_lock_data_structure(nullptr);
        if (order_violations.load(std::memory_order_relaxed) != 0)
            throw std::logic_error("parallel Delaunay commit order invariant violated");

        // The triangulation is no longer needed; free it concurrently with the
        // assembly of the successor lists below.
        std::jthread dt_reaper([&dt] { dt.clear(); });

        // Successor lists by counting sort on the parent: count out-degrees,
        // prefix sum in sigma order, scatter, then sort each short list by pi.
        std::vector<std::pair<const uint64_t*, const uint64_t*>> spans;
        std::size_t edge_count = 0;
        for (const EdgeVec& buf : tls_edges) {
            edge_count += buf.size();
            constexpr std::size_t piece = std::size_t(1) << 16;
            for (std::size_t b = 0; b < buf.size(); b += piece)
                spans.emplace_back(buf.data() + b, buf.data() + std::min(buf.size(), b + piece));
        }
        if (edge_count > std::numeric_limits<uint32_t>::max())
            throw std::length_error("dpnn DAG has too many edges");
        std::unique_ptr<std::atomic<uint32_t>[]> cursor(new std::atomic<uint32_t>[N]);
        tbb::parallel_for(std::size_t(0), N, [&](std::size_t s) {
            cursor[s].store(0, std::memory_order_relaxed);
        });
        tbb::parallel_for(std::size_t(0), spans.size(), [&](std::size_t s) {
            for (const uint64_t* e = spans[s].first; e != spans[s].second; ++e)
                cursor[sigmaOfPi[static_cast<uint32_t>(*e >> 32)]].fetch_add(1, std::memory_order_relaxed);
        });
        std::vector<uint32_t> first_child(N + 1);
        {
            uint32_t offset = 0;
            for (std::size_t s = 0; s < N; ++s) {
                first_child[s] = offset;
                offset += cursor[s].load(std::memory_order_relaxed);
                cursor[s].store(first_child[s], std::memory_order_relaxed);
            }
            first_child[N] = offset;
        }
        children.resize(edge_count);
        tbb::parallel_for(std::size_t(0), spans.size(), [&](std::size_t s) {
            for (const uint64_t* e = spans[s].first; e != spans[s].second; ++e) {
                const uint32_t parent = sigmaOfPi[static_cast<uint32_t>(*e >> 32)];
                children[cursor[parent].fetch_add(1, std::memory_order_relaxed)] =
                    static_cast<uint32_t>(*e);          // child pi, for now
            }
        });
        cursor.reset();
        // Returning the edge buffers to the OS takes a few ms; do it off the
        // critical path, like the triangulation itself.
        std::jthread edge_reaper([bufs = std::move(tls_edges)]() mutable { bufs.clear(); });
        // Every adjacency is recorded once; a duplicate means a broken invariant.
        std::atomic<bool> has_dup{false};
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, N),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t s = r.begin(); s != r.end(); ++s) {
                    uint32_t* const b = children.data() + first_child[s];
                    uint32_t* const e = children.data() + first_child[s + 1];
                    std::sort(b, e);
                    for (uint32_t* c = b; c != e; ++c)
                        if (c != b && *c == c[-1]) has_dup.store(true, std::memory_order_relaxed);
                    for (uint32_t* c = b; c != e; ++c) *c = sigmaOfPi[*c];
                }
            });
        if (has_dup.load(std::memory_order_relaxed))
            throw std::logic_error("dpnn recorded a successor edge twice");

        rank.resize(N);
        tbb::parallel_for(std::size_t(0), N, [&](std::size_t p) {
            rank[sigmaOfPi[p]] = static_cast<uint32_t>(p);
        });
        root = sigmaOfPi[0];
        nodes.resize(N);
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, N),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t s = r.begin(); s != r.end(); ++s) {
                    Node& node = nodes[s];
                    node.x = pts[s].x();
                    node.y = pts[s].y();
                    node.z = pts[s].z();
                    node.firstChild = first_child[s];
                    node.endChild = first_child[s + 1];
                }
            });
    }
};

} // namespace

// ============================================================================
// Public API.
// ============================================================================
namespace dpnn {

static_assert(sizeof(Point) == 3 * sizeof(double), "Point must be three packed doubles");

struct NearestNeighbor::Impl {
    DpnnCore core;
    Impl(const std::vector<Vec3>& pts, int threads) : core(pts, threads) {}
};

NearestNeighbor::NearestNeighbor(const std::vector<Point>& points, int threads)
    : NearestNeighbor(reinterpret_cast<const double*>(points.data()), points.size(), threads) {}

NearestNeighbor::NearestNeighbor(const double* xyz, std::size_t num_points, int threads) {
    std::vector<Vec3> pts;
    pts.reserve(num_points);
    for (std::size_t i = 0; i < num_points; ++i) pts.emplace_back(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    impl_ = std::make_unique<Impl>(pts, threads);
}

NearestNeighbor::~NearestNeighbor() = default;
NearestNeighbor::NearestNeighbor(NearestNeighbor&&) noexcept = default;
NearestNeighbor& NearestNeighbor::operator=(NearestNeighbor&&) noexcept = default;

std::size_t NearestNeighbor::size() const { return impl_->core.size(); }

static void require_points(const NearestNeighbor& index) {
    if (index.size() == 0) throw std::logic_error("the dpnn index has no points");
}

std::uint32_t NearestNeighbor::query(const Point& p) const {
    require_points(*this);
    return impl_->core.nearest(Vec3(p[0], p[1], p[2]));
}

std::uint32_t NearestNeighbor::query(double x, double y, double z) const {
    require_points(*this);
    return impl_->core.nearest(Vec3(x, y, z));
}

std::vector<std::uint32_t> NearestNeighbor::query(const std::vector<Point>& queries,
                                                  bool parallel) const {
    std::vector<std::uint32_t> result(queries.size());
    query(queries, result.data(), parallel);
    return result;
}

void NearestNeighbor::query(const std::vector<Point>& queries, std::uint32_t* out,
                            bool parallel) const {
    if (!queries.empty()) require_points(*this);
    impl_->core.nearestBatch(reinterpret_cast<const double*>(queries.data()), queries.size(),
                             parallel, out);
}

std::vector<std::uint32_t> NearestNeighbor::query_knn(const Point& p, std::size_t k) const {
    if (k > size()) throw std::invalid_argument("k cannot exceed the number of indexed points");
    std::vector<std::uint32_t> result(k);
    if (k > 0) impl_->core.knnInto(Vec3(p[0], p[1], p[2]), k, result.data());
    return result;
}

std::vector<std::uint32_t> NearestNeighbor::query_knn(const std::vector<Point>& queries,
                                                      std::size_t k, bool parallel) const {
    std::vector<std::uint32_t> result(queries.size() * k);
    query_knn(queries, k, result.data(), parallel);
    return result;
}

void NearestNeighbor::query_knn(const std::vector<Point>& queries, std::size_t k,
                                std::uint32_t* out, bool parallel) const {
    if (k > size()) throw std::invalid_argument("k cannot exceed the number of indexed points");
    impl_->core.knnBatch(reinterpret_cast<const double*>(queries.data()), queries.size(), k,
                         parallel, out);
}

} // namespace dpnn
