// Builds an index, answers nearest-neighbor and k-NN queries, and checks a
// sample of the answers against brute force.
//
//   dpnn_example [POINTS.xyz | -] [NUM_QUERIES] [THREADS]
//
// POINTS.xyz holds one point per line ("x y z", further columns are
// ignored).  Without a file (or with "-"), 200,000 points are sampled on a
// torus.  NUM_QUERIES defaults to 1,000,000; THREADS = 0 (default) lets the
// library choose the number of build threads.  The
// queries are uniform in the bounding box of the points grown by half its
// size on every side.
#include <dpnn/dpnn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<dpnn::Point> read_xyz(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<dpnn::Point> points;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        dpnn::Point p;
        if (fields >> p[0] >> p[1] >> p[2]) points.push_back(p);
    }
    return points;
}

std::vector<dpnn::Point> torus(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(0.0, 2.0 * 3.141592653589793);
    std::vector<dpnn::Point> points(n);
    for (auto& p : points) {
        const double u = angle(rng), v = angle(rng);
        p = {(1.0 + 0.3 * std::cos(v)) * std::cos(u), (1.0 + 0.3 * std::cos(v)) * std::sin(u),
             0.3 * std::sin(v)};
    }
    return points;
}

double sq_dist(const dpnn::Point& a, const dpnn::Point& b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int run(int argc, char** argv) {
    std::mt19937_64 rng(1);
    const bool from_file = argc > 1 && std::string(argv[1]) != "-";
    const std::vector<dpnn::Point> points = from_file ? read_xyz(argv[1]) : torus(200000, rng);
    const std::size_t num_queries = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1000000;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 0;
    if (points.empty()) { std::fprintf(stderr, "no points\n"); return 1; }

    dpnn::Point lo = points[0], hi = points[0];
    for (const auto& p : points)
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); }
    std::vector<dpnn::Point> queries(num_queries);
    for (int a = 0; a < 3; ++a) {
        const double margin = 0.5 * (hi[a] - lo[a]);
        std::uniform_real_distribution<double> coordinate(lo[a] - margin, hi[a] + margin);
        for (auto& q : queries) q[a] = coordinate(rng);
    }

    auto t0 = std::chrono::steady_clock::now();
    const dpnn::NearestNeighbor index(points, threads);
    std::printf("points       %zu (%zu distinct)\n", points.size(), index.size());
    std::printf("build        %.1f ms\n", ms_since(t0));

    t0 = std::chrono::steady_clock::now();
    const std::vector<std::uint32_t> nearest = index.query(queries);
    const double t_nn = ms_since(t0);
    std::printf("1-NN batch   %.1f ms (%.3f us per query)\n", t_nn, 1e3 * t_nn / double(num_queries));

    const std::size_t k = 8;
    t0 = std::chrono::steady_clock::now();
    const std::vector<std::uint32_t> knn = index.query_knn(queries, k);
    const double t_knn = ms_since(t0);
    std::printf("%zu-NN batch   %.1f ms (%.3f us per query)\n", k, t_knn, 1e3 * t_knn / double(num_queries));

    // Brute-force check of a sample of the queries, by distance (ties may
    // return a different point at the same distance).
    std::size_t errors = 0;
    const std::size_t checked = std::min<std::size_t>(num_queries, 200);
    std::vector<double> d(points.size());
    for (std::size_t i = 0; i < checked; ++i) {
        const dpnn::Point& q = queries[i * (num_queries / checked)];
        const std::size_t qi = i * (num_queries / checked);
        for (std::size_t j = 0; j < points.size(); ++j) d[j] = sq_dist(points[j], q);
        std::vector<double> sorted = d;
        std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end());
        errors += d[nearest[qi]] != sorted[0];
        errors += d[index.query(q)] != sorted[0];
        for (std::size_t j = 0; j < k; ++j) errors += d[knn[qi * k + j]] != sorted[j];
    }
    std::printf("check        %zu queries against brute force: %s\n", checked,
                errors == 0 ? "all exact" : "MISMATCH");
    return errors == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
