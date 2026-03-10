/**
 * @file ant_simu_mpi2.cpp
 * @brief Phase 4B — MPI parallelization (method 2: map partitioning)
 *
 * @details Each MPI process owns a horizontal strip of the 513×513 grid.
 *          Only ants currently within a process's strip are managed locally.
 *          When an ant crosses a strip boundary, it is migrated to the
 *          neighboring process via MPI_Sendrecv.
 *
 *  Key differences from method 1 (ant_simu_mpi.cpp):
 *    - Each process stores only its local sub-map + 1 row of ghost cells
 *      on each side (halo exchange), NOT the full 513×513 grid.
 *    - Pheromone synchronization reduces to O(N) halo exchange instead
 *      of O(N²) Allreduce on the entire map.
 *    - Ant migration is handled via MPI point-to-point communication.
 *    - Evaporation is fully local — no Allgatherv needed.
 *
 *  Map partitioning:  1D row decomposition along the X-axis.
 *    - Process r owns rows [row_start_r, row_start_r + nrows_r)
 *    - Ghost cells: row (row_start - 1) from upper neighbor,
 *                   row (row_start + nrows) from lower neighbor.
 *    - Grid columns (Y-axis) are NOT partitioned (each process has all columns).
 *
 *  Memory layout (local pheromone):
 *    - (nrows + 2) × stride, where stride = dim + 2
 *    - Local row 0 = ghost from upper neighbor (or boundary)
 *    - Local rows 1..nrows = owned data
 *    - Local row nrows+1 = ghost from lower neighbor (or boundary)
 *
 *  Usage:  mpirun -np <P> ./ant_simu_mpi2.exe --headless -n 5000 [-t <T>]
 */
#include <mpi.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <omp.h>
#include "fractal_land.hpp"
#include "basic_types.hpp"
#include "rand_generator.hpp"

// ============================================================================
// Chronometry
// ============================================================================
using clock_hr = std::chrono::high_resolution_clock;
using dur_ms   = std::chrono::duration<double, std::milli>;

// ============================================================================
// Local pheromone map (sub-grid with ghost rows)
// ============================================================================
// Each element is {V1, V2} stored as two consecutive doubles.
// Memory: (local_nrows + 2) rows × stride columns, each cell = 2 doubles
// Row 0        = ghost (upper boundary)
// Rows 1..N    = owned rows
// Row N+1      = ghost (lower boundary)
//
// We maintain TWO buffers (map + buffer) like the original pheronome class,
// to support the double-buffering update() scheme.
struct local_pheromone {
    std::size_t dim;          // global grid dimension (513)
    std::size_t stride;       // dim + 2 (515)
    std::size_t local_nrows;  // number of owned rows
    int         global_row0;  // global row index of first owned row (0-based in [0, dim-1])
    double      alpha;
    double      beta;
    position_t  pos_food;
    position_t  pos_nest;

    // Total rows including ghosts = local_nrows + 2
    // Total size = (local_nrows + 2) * stride * 2 doubles
    std::vector<double> map_data;    // "current" readable map
    std::vector<double> buf_data;    // "buffer" writable by ants

    void init(std::size_t dim_, std::size_t nrows, int grow0,
              double alpha_, double beta_,
              const position_t& food, const position_t& nest)
    {
        dim        = dim_;
        stride     = dim + 2;
        local_nrows = nrows;
        global_row0 = grow0;
        alpha      = alpha_;
        beta       = beta_;
        pos_food   = food;
        pos_nest   = nest;

        std::size_t total_cells = (local_nrows + 2) * stride;
        map_data.assign(total_cells * 2, 0.0);
        buf_data.assign(total_cells * 2, 0.0);

        // Set boundary (ghost) cells to -1 on both buffers
        set_boundary();
        set_boundary_on(buf_data);

        // Set food/nest pheromone if they fall in this sub-grid
        set_special_cells(map_data);
        set_special_cells(buf_data);
    }

    // Convert global (i, j) to local index.  i is global row [0, dim-1].
    // Returns index into map_data/buf_data array (double index, multiply by 2 done).
    // local_row = i - global_row0 + 1  (the +1 accounts for the ghost row at top)
    inline std::size_t local_index(int gi, int gj) const {
        int lr = gi - global_row0 + 1;  // local row (1-based in local storage)
        return ((std::size_t)lr * stride + (std::size_t)(gj + 1)) * 2;
    }

    // Read pheromone at global (i, j) — from map_data
    inline double map_v(int gi, int gj, int pher_idx) const {
        return map_data[local_index(gi, gj) + pher_idx];
    }

    // Read pheromone at global (i, j) — from buf_data
    inline double buf_v(int gi, int gj, int pher_idx) const {
        return buf_data[local_index(gi, gj) + pher_idx];
    }

    // Write pheromone to buf_data at global (i, j)
    inline void buf_set(int gi, int gj, int pher_idx, double val) {
        buf_data[local_index(gi, gj) + pher_idx] = val;
    }

    // Check if global row i is owned by this process
    inline bool owns_row(int gi) const {
        return gi >= global_row0 && gi < global_row0 + (int)local_nrows;
    }

    // Check if global (i, j) is in the readable range (owned + ghost)
    inline bool can_read(int gi, int gj) const {
        return gi >= global_row0 - 1 && gi <= global_row0 + (int)local_nrows
            && gj >= 0 && gj < (int)dim;
    }

    // mark_pheronome: same logic as original, but uses local addressing
    void mark_pheronome(int gi, int gj) {
        // Read from map_data (the "current" readable map)
        auto read = [&](int i, int j, int idx) -> double {
            int lr = i - global_row0 + 1;
            std::size_t pos = ((std::size_t)lr * stride + (std::size_t)(j + 1)) * 2 + idx;
            double v = map_data[pos];
            return std::max(v, 0.0);
        };

        double v1_left   = read(gi - 1, gj, 0);
        double v2_left   = read(gi - 1, gj, 1);
        double v1_right  = read(gi + 1, gj, 0);
        double v2_right  = read(gi + 1, gj, 1);
        double v1_upper  = read(gi, gj - 1, 0);
        double v2_upper  = read(gi, gj - 1, 1);
        double v1_bottom = read(gi, gj + 1, 0);
        double v2_bottom = read(gi, gj + 1, 1);

        std::size_t base = local_index(gi, gj);
        buf_data[base + 0] = alpha * std::max({v1_left, v1_right, v1_upper, v1_bottom})
                            + (1 - alpha) * 0.25 * (v1_left + v1_right + v1_upper + v1_bottom);
        buf_data[base + 1] = alpha * std::max({v2_left, v2_right, v2_upper, v2_bottom})
                            + (1 - alpha) * 0.25 * (v2_left + v2_right + v2_upper + v2_bottom);
    }

    // Evaporation on owned rows only (buf_data)
    void do_evaporation() {
        for (std::size_t lr = 1; lr <= local_nrows; ++lr) {
            for (std::size_t j = 1; j <= dim; ++j) {
                std::size_t idx = (lr * stride + j) * 2;
                buf_data[idx + 0] *= beta;
                buf_data[idx + 1] *= beta;
            }
        }
    }

    // update(): swap map and buffer, then apply boundary conditions
    void update() {
        map_data.swap(buf_data);
        set_boundary();
        set_special_cells(map_data);
    }

    // Pointer to the start of local row `lr` in map_data (lr=0 is ghost top)
    double* map_row_ptr(std::size_t lr) {
        return map_data.data() + lr * stride * 2;
    }
    double* buf_row_ptr(std::size_t lr) {
        return buf_data.data() + lr * stride * 2;
    }

    std::size_t row_doubles() const { return stride * 2; }

private:
    void set_boundary_on(std::vector<double>& data) {
        std::size_t total_local_rows = local_nrows + 2;
        for (std::size_t lr = 0; lr < total_local_rows; ++lr) {
            // Left column ghost (j_local = 0)
            std::size_t idx_left = (lr * stride + 0) * 2;
            data[idx_left + 0] = -1.0; data[idx_left + 1] = -1.0;
            // Right column ghost (j_local = dim+1)
            std::size_t idx_right = (lr * stride + dim + 1) * 2;
            data[idx_right + 0] = -1.0; data[idx_right + 1] = -1.0;
        }
        // Top ghost row if this is the first process (global boundary)
        if (global_row0 == 0) {
            for (std::size_t j = 0; j < stride; ++j) {
                std::size_t idx = j * 2;
                data[idx + 0] = -1.0; data[idx + 1] = -1.0;
            }
        }
        // Bottom ghost row if this is the last process (global boundary)
        if (global_row0 + (int)local_nrows >= (int)dim) {
            std::size_t lr_bot = local_nrows + 1;
            for (std::size_t j = 0; j < stride; ++j) {
                std::size_t idx = (lr_bot * stride + j) * 2;
                data[idx + 0] = -1.0; data[idx + 1] = -1.0;
            }
        }
    }

    void set_boundary() {
        set_boundary_on(map_data);
    }

    void set_special_cells(std::vector<double>& data) {
        // Food: if pos_food.x is in our range
        int fi = pos_food.x;
        if (fi >= global_row0 && fi < global_row0 + (int)local_nrows) {
            int lr = fi - global_row0 + 1;
            std::size_t idx = ((std::size_t)lr * stride + (std::size_t)(pos_food.y + 1)) * 2;
            data[idx + 0] = 1.0;
        }
        // Nest: if pos_nest.x is in our range
        int ni = pos_nest.x;
        if (ni >= global_row0 && ni < global_row0 + (int)local_nrows) {
            int lr = ni - global_row0 + 1;
            std::size_t idx = ((std::size_t)lr * stride + (std::size_t)(pos_nest.y + 1)) * 2;
            data[idx + 1] = 1.0;
        }
    }
};

// ============================================================================
// SoA ant structure (local)
// ============================================================================
struct ants_local {
    std::vector<int>         px, py;    // global positions
    std::vector<int>         states;    // 0=unloaded, 1=loaded
    std::vector<std::size_t> seeds;
    std::size_t count = 0;

    void clear() { px.clear(); py.clear(); states.clear(); seeds.clear(); count = 0; }

    void push(int x, int y, int state, std::size_t seed) {
        px.push_back(x); py.push_back(y);
        states.push_back(state); seeds.push_back(seed);
        ++count;
    }

    void remove(std::size_t i) {
        // Swap with last and pop
        if (i < count - 1) {
            px[i] = px[count-1]; py[i] = py[count-1];
            states[i] = states[count-1]; seeds[i] = seeds[count-1];
        }
        px.pop_back(); py.pop_back();
        states.pop_back(); seeds.pop_back();
        --count;
    }
};

// ============================================================================
// Ant migration data (packed for MPI transfer)
// ============================================================================
// Each migrating ant is packed as 4 ints: px, py, state, seed(low32)
// (We truncate seed to 32 bits; the PRNG only uses low 32 bits anyway)
static const int ANT_PACK_SIZE = 4;

// ============================================================================
// Global parameters
// ============================================================================
static double s_eps = 0.;

// ============================================================================
// Advance local ants — returns food collected, fills migration lists
// ============================================================================
static std::size_t advance_local_ants(
    ants_local& ants, local_pheromone& phen,
    const fractal_land& land,
    const position_t& pos_food,
    const position_t& pos_nest,
    std::vector<int>& migrate_up,   // ants to send to rank-1
    std::vector<int>& migrate_down) // ants to send to rank+1
{
    std::size_t local_food = 0;
    migrate_up.clear();
    migrate_down.clear();

    // We'll iterate and potentially remove ants, so process carefully
    std::size_t i = 0;
    while (i < ants.count) {
        std::size_t& seed = ants.seeds[i];
        double consumed_time = 0.;
        bool migrated = false;

        while (consumed_time < 1. && !migrated) {
            int ind_pher = (ants.states[i] == 1) ? 1 : 0;
            double choix = rand_double(0., 1., seed);

            int ox = ants.px[i], oy = ants.py[i];
            int nx = ox, ny = oy;

            // Read pheromones from neighbors (must be readable = owned or ghost)
            double p_left   = phen.map_v(ox - 1, oy, ind_pher);
            double p_right  = phen.map_v(ox + 1, oy, ind_pher);
            double p_up     = phen.map_v(ox, oy - 1, ind_pher);
            double p_down   = phen.map_v(ox, oy + 1, ind_pher);

            double max_phen = std::max({p_left, p_right, p_up, p_down});

            if ((choix > s_eps) || (max_phen <= 0.)) {
                do {
                    nx = ox; ny = oy;
                    int d = rand_int32(1, 4, seed);
                    if (d == 1) nx -= 1;
                    if (d == 2) ny -= 1;
                    if (d == 3) nx += 1;
                    if (d == 4) ny += 1;
                } while (phen.map_v(nx, ny, ind_pher) == -1.);
            } else {
                if (p_left == max_phen)       nx = ox - 1;
                else if (p_right == max_phen) nx = ox + 1;
                else if (p_up == max_phen)    ny = oy - 1;
                else                          ny = oy + 1;
            }

            consumed_time += land(nx, ny);

            // Mark pheromone (write to buffer) — only if new pos is in owned region
            if (phen.owns_row(nx)) {
                phen.mark_pheronome(nx, ny);
            }

            ants.px[i] = nx;
            ants.py[i] = ny;

            // Check nest/food
            if (nx == pos_nest.x && ny == pos_nest.y) {
                if (ants.states[i] == 1) local_food += 1;
                ants.states[i] = 0;
            }
            if (nx == pos_food.x && ny == pos_food.y) {
                ants.states[i] = 1;
            }

            // Check if ant has left our domain — must stop inner loop immediately
            // because next iteration would read pheromones at (nx±1, ny±1) which
            // may be outside our local storage (2 rows beyond boundary = out of ghost)
            if (!phen.owns_row(nx)) {
                // Pack ant data for migration
                int pack[ANT_PACK_SIZE] = {
                    ants.px[i], ants.py[i], ants.states[i],
                    (int)(ants.seeds[i] & 0xFFFFFFFF)
                };
                if (nx < phen.global_row0) {
                    migrate_up.insert(migrate_up.end(), pack, pack + ANT_PACK_SIZE);
                } else {
                    migrate_down.insert(migrate_down.end(), pack, pack + ANT_PACK_SIZE);
                }
                ants.remove(i);
                migrated = true;
                // Don't increment i; remove() swaps with last
            }
        }
        if (!migrated) ++i;
    }
    return local_food;
}

// ============================================================================
// Halo exchange: swap border pheromone rows with neighbors
// ============================================================================
static void halo_exchange(local_pheromone& phen, int rank, int nprocs)
{
    const std::size_t row_bytes = phen.row_doubles() * sizeof(double);
    MPI_Status status;

    int upper = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int lower = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Exchange on map_data (after update, map_data is the readable one)
    // Send my top owned row to upper neighbor's bottom ghost
    // Recv upper neighbor's bottom owned row into my top ghost
    MPI_Sendrecv(
        phen.map_row_ptr(1), phen.row_doubles(), MPI_DOUBLE, upper, 0,
        phen.map_row_ptr(0), phen.row_doubles(), MPI_DOUBLE, upper, 1,
        MPI_COMM_WORLD, &status);

    // Send my bottom owned row to lower neighbor's top ghost
    // Recv lower neighbor's top owned row into my bottom ghost
    MPI_Sendrecv(
        phen.map_row_ptr(phen.local_nrows), phen.row_doubles(), MPI_DOUBLE, lower, 1,
        phen.map_row_ptr(phen.local_nrows + 1), phen.row_doubles(), MPI_DOUBLE, lower, 0,
        MPI_COMM_WORLD, &status);

    // Also exchange on buf_data (the buffer that ants write to)
    MPI_Sendrecv(
        phen.buf_row_ptr(1), phen.row_doubles(), MPI_DOUBLE, upper, 2,
        phen.buf_row_ptr(0), phen.row_doubles(), MPI_DOUBLE, upper, 3,
        MPI_COMM_WORLD, &status);

    MPI_Sendrecv(
        phen.buf_row_ptr(phen.local_nrows), phen.row_doubles(), MPI_DOUBLE, lower, 3,
        phen.buf_row_ptr(phen.local_nrows + 1), phen.row_doubles(), MPI_DOUBLE, lower, 2,
        MPI_COMM_WORLD, &status);
}

// ============================================================================
// Ant migration: exchange ants with neighbors
// ============================================================================
static void migrate_ants(ants_local& ants, local_pheromone& phen,
                          std::vector<int>& migrate_up,
                          std::vector<int>& migrate_down,
                          int rank, int nprocs)
{
    int upper = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int lower = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
    MPI_Status status;

    // First, exchange counts so each side knows how much to receive
    int send_up_count   = (int)migrate_up.size();
    int send_down_count = (int)migrate_down.size();
    int recv_from_up_count = 0, recv_from_down_count = 0;

    MPI_Sendrecv(&send_up_count,   1, MPI_INT, upper, 10,
                 &recv_from_down_count, 1, MPI_INT, lower, 10,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(&send_down_count, 1, MPI_INT, lower, 11,
                 &recv_from_up_count,   1, MPI_INT, upper, 11,
                 MPI_COMM_WORLD, &status);

    // Receive buffers
    std::vector<int> recv_from_up(recv_from_up_count);
    std::vector<int> recv_from_down(recv_from_down_count);

    // Exchange actual ant data
    MPI_Sendrecv(migrate_up.data(),   send_up_count,   MPI_INT, upper, 12,
                 recv_from_down.data(), recv_from_down_count, MPI_INT, lower, 12,
                 MPI_COMM_WORLD, &status);
    MPI_Sendrecv(migrate_down.data(), send_down_count, MPI_INT, lower, 13,
                 recv_from_up.data(),   recv_from_up_count,   MPI_INT, upper, 13,
                 MPI_COMM_WORLD, &status);

    // Unpack received ants
    auto unpack = [&](const std::vector<int>& buf) {
        for (std::size_t k = 0; k + ANT_PACK_SIZE - 1 < buf.size(); k += ANT_PACK_SIZE) {
            ants.push(buf[k], buf[k+1], buf[k+2], (std::size_t)(unsigned int)buf[k+3]);
        }
    };
    unpack(recv_from_up);
    unpack(recv_from_down);
}

// ============================================================================
// Boundary-row pheromone merge (MPI_MAX) for shared boundary rows
// ============================================================================
// When two processes share a boundary, the row that is the ghost for one
// process is the owned row of the other. After ants have marked pheromones
// near boundaries, the ghost rows may have stale values. But more critically,
// ants on adjacent processes may have marked the same physical row differently
// (each process only marks its owned rows). We handle this by:
// 1. Halo exchange (already done)
// 2. For the boundary rows, take element-wise MAX between the two copies
//    to ensure pheromone consistency (same semantics as method 1's Allreduce MAX)
// This function applies MAX merge on the ghost rows after halo exchange.
static void merge_boundary_pheromones(local_pheromone& phen, int rank, int nprocs)
{
    MPI_Status status;
    int upper = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int lower = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    std::size_t rd = phen.row_doubles();
    std::vector<double> tmp(rd);

    // Send my first owned row's buf_data to upper neighbor
    // Upper neighbor takes MAX with its last owned row
    MPI_Sendrecv(
        phen.buf_row_ptr(1), rd, MPI_DOUBLE, upper, 20,
        tmp.data(), rd, MPI_DOUBLE, lower, 20,
        MPI_COMM_WORLD, &status);

    if (lower != MPI_PROC_NULL) {
        double* my_last = phen.buf_row_ptr(phen.local_nrows);
        for (std::size_t k = 0; k < rd; ++k)
            my_last[k] = std::max(my_last[k], tmp[k]);
    }

    // Send my last owned row's buf_data to lower neighbor
    // Lower neighbor takes MAX with its first owned row
    MPI_Sendrecv(
        phen.buf_row_ptr(phen.local_nrows), rd, MPI_DOUBLE, lower, 21,
        tmp.data(), rd, MPI_DOUBLE, upper, 21,
        MPI_COMM_WORLD, &status);

    if (upper != MPI_PROC_NULL) {
        double* my_first = phen.buf_row_ptr(1);
        for (std::size_t k = 0; k < rd; ++k)
            my_first[k] = std::max(my_first[k], tmp[k]);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // --- CLI ---
    std::size_t max_iterations = 0;
    bool headless = false;
    int num_threads = 0;
    for (int a = 1; a < argc; ++a) {
        std::string arg(argv[a]);
        if ((arg == "--iterations" || arg == "-n") && a + 1 < argc)
            max_iterations = std::stoull(argv[++a]);
        if (arg == "--headless" || arg == "-H")
            headless = true;
        if ((arg == "--threads" || arg == "-t") && a + 1 < argc)
            num_threads = std::stoi(argv[++a]);
    }
    if (num_threads > 0) omp_set_num_threads(num_threads);

    // --- SDL init only on rank 0, non-headless ---
    if (rank == 0 && !headless) SDL_Init(SDL_INIT_VIDEO);

    // === Simulation parameters ===
    const int    total_ants = 5000;
    const double eps        = 0.8;
    const double alpha      = 0.7;
    const double beta       = 0.999;
    std::size_t  seed       = 2026;
    position_t   pos_nest{256, 256};
    position_t   pos_food{500, 500};
    s_eps = eps;

    // === Generate terrain (identical on all ranks — same seed) ===
    // ALL ranks need the full terrain because ants read land(nx, ny) for movement cost.
    // The terrain is read-only and only 513×513 doubles ≈ 2 MB. This is acceptable.
    auto t0_land = clock_hr::now();
    fractal_land land(8, 2, 1., 1024);
    auto t1_land = clock_hr::now();
    double time_land_ms = dur_ms(t1_land - t0_land).count();

    // Normalize terrain
    double max_val = 0., min_val = 0.;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j)
            land(i,j) = (land(i,j) - min_val) / delta;

    const std::size_t dim = land.dimensions(); // 513

    // === 1D row decomposition ===
    // Process r owns global rows [row_start_r, row_start_r + nrows_r)
    std::vector<int> all_row_starts(nprocs), all_nrows(nprocs);
    {
        int base_r = dim / nprocs;
        int rem_r  = dim % nprocs;
        int start  = 0;
        for (int r = 0; r < nprocs; ++r) {
            all_nrows[r]     = base_r + (r < rem_r ? 1 : 0);
            all_row_starts[r] = start;
            start += all_nrows[r];
        }
    }
    int my_row_start = all_row_starts[rank];
    int my_nrows     = all_nrows[rank];

    // === Local pheromone map ===
    local_pheromone phen;
    phen.init(dim, my_nrows, my_row_start, alpha, beta, pos_food, pos_nest);

    // === Initialize ALL ants on every rank (for deterministic seed consumption) ===
    // Then each rank keeps only the ants that start in its row range.
    std::vector<int> all_px(total_ants), all_py(total_ants);
    {
        std::size_t init_seed = seed;
        for (int i = 0; i < total_ants; ++i) {
            all_px[i] = rand_int32(0, land.dimensions() - 1, init_seed);
            all_py[i] = rand_int32(0, land.dimensions() - 1, init_seed);
        }
    }

    ants_local ants;
    for (int i = 0; i < total_ants; ++i) {
        if (phen.owns_row(all_px[i])) {
            ants.push(all_px[i], all_py[i], 0, 0);
        }
    }

    // Initial halo exchange
    halo_exchange(phen, rank, nprocs);

    // === For rendering: rank 0 needs global pheromone + all ant positions ===
    // We'll gather them if non-headless
    // (rendering code omitted for headless benchmark — see below)

    // === Timing accumulators ===
    double total_ants_ms   = 0., total_evap_ms  = 0., total_update_ms = 0.;
    double total_comm_ms   = 0., total_loop_ms = 0.;

    std::size_t local_food_total = 0;
    std::size_t global_food = 0;
    bool cont_loop = true;
    bool not_food_yet = true;
    std::size_t it = 0;

    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "=== Mesure des performances (MPI méthode 2 : partition) ===" << std::endl;
        std::cout << "Génération du terrain fractal : " << time_land_ms << " ms" << std::endl;
        std::cout << "Grille : " << dim << " x " << dim << std::endl;
        std::cout << "Nombre de fourmis total : " << total_ants << std::endl;
        std::cout << "Processus MPI : " << nprocs << std::endl;
        std::cout << "Threads OpenMP/proc : " << actual_threads << std::endl;
        std::cout << "Lignes locales (rank 0) : " << my_nrows
                  << " [" << my_row_start << ", " << my_row_start + my_nrows - 1 << "]" << std::endl;
        std::cout << "Fourmis initiales (rank 0) : " << ants.count << std::endl;
        if (max_iterations > 0)
            std::cout << "Mode benchmark : " << max_iterations << " itérations" << std::endl;
        std::cout << "Mode headless : " << (headless ? "oui" : "non") << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
    }

    // Migration buffers (pre-allocated, reused each iteration)
    std::vector<int> migrate_up, migrate_down;
    migrate_up.reserve(total_ants * ANT_PACK_SIZE / nprocs);
    migrate_down.reserve(total_ants * ANT_PACK_SIZE / nprocs);

    // ============================================================
    // Main loop
    // ============================================================
    SDL_Event event;
    while (cont_loop) {
        ++it;
        auto t_iter_start = clock_hr::now();

        // --- SDL events (rank 0) ---
        if (rank == 0 && !headless) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) cont_loop = false;
            }
        }

        // ---- Phase 1: Advance local ants ----
        auto t1 = clock_hr::now();
        std::size_t iter_food = advance_local_ants(
            ants, phen, land, pos_food, pos_nest, migrate_up, migrate_down);
        local_food_total += iter_food;
        auto t2 = clock_hr::now();
        double iter_ants_ms = dur_ms(t2 - t1).count();

        // ---- Phase 2: Communication ----
        auto t3 = clock_hr::now();

        // 2a. Migrate ants between neighbors
        migrate_ants(ants, phen, migrate_up, migrate_down, rank, nprocs);

        // 2b. Merge boundary pheromones (MAX on shared rows)
        merge_boundary_pheromones(phen, rank, nprocs);

        auto t4 = clock_hr::now();
        double iter_comm_ms = dur_ms(t4 - t3).count();

        // ---- Phase 3: Evaporation (local only) ----
        auto t5 = clock_hr::now();
        phen.do_evaporation();
        auto t6 = clock_hr::now();
        double iter_evap_ms = dur_ms(t6 - t5).count();

        // ---- Phase 4: Update pheromones + halo exchange ----
        auto t7 = clock_hr::now();
        phen.update();
        halo_exchange(phen, rank, nprocs);
        auto t8 = clock_hr::now();
        double iter_update_ms = dur_ms(t8 - t7).count();

        // ---- Food detection (headless) ----
        if (headless && not_food_yet) {
            std::size_t tmp_global = 0;
            MPI_Allreduce(&local_food_total, &tmp_global, 1,
                          MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
            global_food = tmp_global;
            if (tmp_global > 0) {
                if (rank == 0)
                    std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
                not_food_yet = false;
            }
        }

        auto t_iter_end = clock_hr::now();
        double iter_loop_ms = dur_ms(t_iter_end - t_iter_start).count();

        total_ants_ms   += iter_ants_ms;
        total_comm_ms   += iter_comm_ms;
        total_evap_ms   += iter_evap_ms;
        total_update_ms += iter_update_ms;
        total_loop_ms   += iter_loop_ms;

        if (max_iterations > 0 && it >= max_iterations) cont_loop = false;

        // Broadcast cont_loop from rank 0 (for SDL_QUIT)
        int cont_int = cont_loop ? 1 : 0;
        MPI_Bcast(&cont_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        cont_loop = (cont_int != 0);
    }

    // === Final food reduction ===
    MPI_Reduce(&local_food_total, &global_food, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // === Gather ant distribution statistics ===
    int final_local_ants = (int)ants.count;
    std::vector<int> all_ant_counts(nprocs);
    MPI_Gather(&final_local_ants, 1, MPI_INT, all_ant_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // === Performance report (rank 0) ===
    if (rank == 0) {
        std::cout << "\n=============================================" << std::endl;
        std::cout << "  RAPPORT DE PERFORMANCE (MPI méthode 2, " << nprocs << " processus";
        if (actual_threads > 1) std::cout << " × " << actual_threads << " threads";
        std::cout << ")" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Itérations effectuées      : " << it << std::endl;
        std::cout << "Nourriture collectée       : " << global_food << std::endl;
        std::cout << "Processus MPI              : " << nprocs << std::endl;
        std::cout << "Threads OpenMP/proc        : " << actual_threads << std::endl;
        std::cout << "Distribution finale fourmis: ";
        for (int r = 0; r < nprocs; ++r)
            std::cout << "R" << r << "=" << all_ant_counts[r] << " ";
        std::cout << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << std::setw(30) << std::left << "Composant"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Moy/iter (ms)"
                  << std::setw(10) << "%" << std::endl;
        std::cout << "---------------------------------------------" << std::endl;

        auto print_row = [&](const char* name, double total) {
            double avg = total / it;
            double pct = (total / total_loop_ms) * 100.0;
            std::cout << std::setw(30) << std::left << name
                      << std::setw(15) << total
                      << std::setw(15) << avg
                      << std::setw(10) << pct << std::endl;
        };

        print_row("Avancement fourmis",               total_ants_ms);
        print_row("Communication (migration+halo)",    total_comm_ms);
        print_row("Évaporation phéromones (locale)",   total_evap_ms);
        print_row("Mise à jour + halo exchange",       total_update_ms);
        std::cout << "---------------------------------------------" << std::endl;
        print_row("TOTAL boucle",                      total_loop_ms);
        std::cout << "=============================================" << std::endl;
    }

    if (rank == 0 && !headless) SDL_Quit();
    MPI_Finalize();
    return 0;
}
