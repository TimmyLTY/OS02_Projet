/**
 * @file ant_simu_mpi.cpp
 * @brief Phase 4A — MPI parallelization (method 1: full map replication)
 * @details Each process holds the complete environment (terrain + pheromone map)
 *          and controls a subset of ants. Pheromone synchronization uses
 *          MPI_Allreduce with MPI_MAX (as specified in Readme).
 *          Evaporation is distributed by row blocks + MPI_Allgatherv.
 *          Food count uses MPI_Reduce to rank 0.
 *          Optional OpenMP within each process via -t flag.
 *
 *  Usage:  mpirun -np <P> ./ant_simu_mpi.exe --headless -n 5000 [-t <T>]
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
#include <omp.h>
#include "fractal_land.hpp"
#include "pheronome.hpp"
#include "basic_types.hpp"
#include "rand_generator.hpp"
#include "renderer.hpp"
#include "window.hpp"

// ============================================================================
// Chronometry
// ============================================================================
using clock_hr = std::chrono::high_resolution_clock;
using dur_ms   = std::chrono::duration<double, std::milli>;

// ============================================================================
// SoA ant structure (local to this translation unit, no need for header)
// ============================================================================
struct ants_local {
    std::vector<int>         px, py;    // positions
    std::vector<int>         states;    // 0=unloaded, 1=loaded
    std::vector<std::size_t> seeds;
    std::size_t count = 0;
};

static double s_eps = 0.;

// ============================================================================
// Advance local ants (with optional OpenMP)
// ============================================================================
static std::size_t advance_local_ants(ants_local& ants, pheronome& phen,
                                       const fractal_land& land,
                                       const position_t& pos_food,
                                       const position_t& pos_nest)
{
    std::size_t local_food = 0;

    #pragma omp parallel for schedule(dynamic, 50) reduction(+:local_food)
    for (std::size_t i = 0; i < ants.count; ++i) {
        std::size_t& seed = ants.seeds[i];
        double consumed_time = 0.;

        while (consumed_time < 1.) {
            int ind_pher = (ants.states[i] == 1) ? 1 : 0;
            double choix = rand_double(0., 1., seed);

            int ox = ants.px[i], oy = ants.py[i];
            int nx = ox, ny = oy;

            double max_phen = std::max({
                phen(ox - 1, oy)[ind_pher],
                phen(ox + 1, oy)[ind_pher],
                phen(ox, oy - 1)[ind_pher],
                phen(ox, oy + 1)[ind_pher]
            });

            if ((choix > s_eps) || (max_phen <= 0.)) {
                do {
                    nx = ox; ny = oy;
                    int d = rand_int32(1, 4, seed);
                    if (d == 1) nx -= 1;
                    if (d == 2) ny -= 1;
                    if (d == 3) nx += 1;
                    if (d == 4) ny += 1;
                } while (phen(nx, ny)[ind_pher] == -1.);
            } else {
                if (phen(ox - 1, oy)[ind_pher] == max_phen)      nx = ox - 1;
                else if (phen(ox + 1, oy)[ind_pher] == max_phen) nx = ox + 1;
                else if (phen(ox, oy - 1)[ind_pher] == max_phen) ny = oy - 1;
                else                                              ny = oy + 1;
            }

            consumed_time += land(nx, ny);
            position_t new_pos{nx, ny};
            phen.mark_pheronome(new_pos);
            ants.px[i] = nx;
            ants.py[i] = ny;

            if (nx == pos_nest.x && ny == pos_nest.y) {
                if (ants.states[i] == 1) local_food += 1;
                ants.states[i] = 0;
            }
            if (nx == pos_food.x && ny == pos_food.y) {
                ants.states[i] = 1;
            }
        }
    }
    return local_food;
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

    // --- SDL init only on rank 0 ---
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

    // === Initialize ALL ants on every rank (for deterministic seed consumption) ===
    // Then each rank keeps only its slice.
    std::vector<int> all_px(total_ants), all_py(total_ants);
    {
        std::size_t init_seed = seed; // copy
        for (int i = 0; i < total_ants; ++i) {
            all_px[i] = rand_int32(0, land.dimensions() - 1, init_seed);
            all_py[i] = rand_int32(0, land.dimensions() - 1, init_seed);
        }
    }

    // Partition ants: rank r gets ants [start_ant, start_ant + local_count)
    int base = total_ants / nprocs;
    int remainder = total_ants % nprocs;
    int start_ant = rank * base + std::min(rank, remainder);
    int local_count = base + (rank < remainder ? 1 : 0);

    ants_local ants;
    ants.count = local_count;
    ants.px.assign(all_px.begin() + start_ant, all_px.begin() + start_ant + local_count);
    ants.py.assign(all_py.begin() + start_ant, all_py.begin() + start_ant + local_count);
    ants.states.assign(local_count, 0);
    ants.seeds.assign(local_count, 0); // Reproduce original bug (m_seed=0)

    // === Pheromone map (full, on each rank) ===
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    // Pheromone buffer size (stride × stride × 2 doubles)
    const std::size_t stride = land.dimensions() + 2;
    const std::size_t phen_buf_size = stride * stride; // number of pheronome_t elements

    // Temporary buffer for MPI_Allreduce receive
    // Each pheronome_t is std::array<double,2>, so total doubles = phen_buf_size * 2
    const std::size_t total_doubles = phen_buf_size * 2;
    std::vector<double> global_phen_buf(total_doubles);

    // Evaporation row distribution
    const std::size_t dim = land.dimensions();
    // Each rank handles rows [row_start, row_start + row_count) (1-indexed in pheromone grid)
    std::vector<int> row_counts(nprocs), row_displs(nprocs);
    {
        int base_r = dim / nprocs;
        int rem_r  = dim % nprocs;
        int disp = 0;
        for (int r = 0; r < nprocs; ++r) {
            row_counts[r] = base_r + (r < rem_r ? 1 : 0);
            row_displs[r] = disp;
            disp += row_counts[r];
        }
    }
    int my_row_start = row_displs[rank] + 1; // +1 because pheromone rows are 1-indexed
    int my_row_count = row_counts[rank];

    // For MPI_Allgatherv on evaporation: each rank sends my_row_count * stride * 2 doubles
    std::vector<int> evap_sendcounts(nprocs), evap_displs(nprocs);
    {
        int disp = 0;
        for (int r = 0; r < nprocs; ++r) {
            evap_sendcounts[r] = row_counts[r] * stride * 2;
            evap_displs[r] = disp;
            disp += evap_sendcounts[r];
        }
    }

    // --- Rendering (rank 0 only) ---
    // For rendering, rank 0 needs all ant positions → gather them
    std::vector<int> all_render_px, all_render_py;
    std::vector<int> ant_counts(nprocs), ant_displs_v(nprocs);
    if (!headless && rank == 0) {
        all_render_px.resize(total_ants);
        all_render_py.resize(total_ants);
    }
    // Pre-compute gather counts/displacements
    {
        int disp = 0;
        for (int r = 0; r < nprocs; ++r) {
            ant_counts[r] = base + (r < remainder ? 1 : 0);
            ant_displs_v[r] = disp;
            disp += ant_counts[r];
        }
    }

    Window* win_ptr = nullptr;
    Renderer* renderer_ptr = nullptr;
    if (!headless) {
        // All ranks participate in Gatherv for initial positions
        if (rank == 0) {
            MPI_Gatherv(ants.px.data(), local_count, MPI_INT,
                         all_render_px.data(), ant_counts.data(), ant_displs_v.data(), MPI_INT,
                         0, MPI_COMM_WORLD);
            MPI_Gatherv(ants.py.data(), local_count, MPI_INT,
                         all_render_py.data(), ant_counts.data(), ant_displs_v.data(), MPI_INT,
                         0, MPI_COMM_WORLD);
            win_ptr = new Window("Ant Simulation (MPI)", 2*land.dimensions()+10, land.dimensions()+266);
            renderer_ptr = new Renderer(land, phen, pos_nest, pos_food,
                                         all_render_px, all_render_py, total_ants);
        } else {
            MPI_Gatherv(ants.px.data(), local_count, MPI_INT,
                         nullptr, nullptr, nullptr, MPI_INT,
                         0, MPI_COMM_WORLD);
            MPI_Gatherv(ants.py.data(), local_count, MPI_INT,
                         nullptr, nullptr, nullptr, MPI_INT,
                         0, MPI_COMM_WORLD);
        }
    }

    // Temporary buffer for evaporation Allgatherv (allocated once, reused)
    const std::size_t interior_doubles = dim * stride * 2;
    std::vector<double> evap_tmp(interior_doubles);

    // === Timing accumulators ===
    double total_ants_ms   = 0., total_evap_ms  = 0., total_update_ms = 0.;
    double total_comm_ms   = 0., total_render_ms = 0., total_loop_ms = 0.;

    std::size_t local_food_total = 0;
    std::size_t global_food = 0;
    bool cont_loop = true;
    bool not_food_yet = true;
    std::size_t it = 0;

    // Actual thread count
    int actual_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "=== Mesure des performances (MPI) ===" << std::endl;
        std::cout << "Génération du terrain fractal : " << time_land_ms << " ms" << std::endl;
        std::cout << "Grille : " << dim << " x " << dim << std::endl;
        std::cout << "Nombre de fourmis total : " << total_ants << std::endl;
        std::cout << "Processus MPI : " << nprocs << std::endl;
        std::cout << "Threads OpenMP/proc : " << actual_threads << std::endl;
        std::cout << "Fourmis locales (rank 0) : " << local_count << std::endl;
        if (max_iterations > 0)
            std::cout << "Mode benchmark : " << max_iterations << " itérations" << std::endl;
        std::cout << "Mode headless : " << (headless ? "oui" : "non") << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
    }

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
        std::size_t iter_food = advance_local_ants(ants, phen, land, pos_food, pos_nest);
        local_food_total += iter_food;
        auto t2 = clock_hr::now();
        double iter_ants_ms = dur_ms(t2 - t1).count();

        // ---- Phase 2: Synchronize pheromones (MPI_Allreduce with MPI_MAX) ----
        //   Readme: "prendre la valeur la plus grande d'entre tous les processus"
        auto t3 = clock_hr::now();

        double* local_buf_ptr = phen.get_buffer_data();

        MPI_Allreduce(local_buf_ptr, global_phen_buf.data(),
                       total_doubles, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Write back the global max into local buffer
        std::memcpy(local_buf_ptr, global_phen_buf.data(), total_doubles * sizeof(double));

        auto t4 = clock_hr::now();
        double iter_comm_ms = dur_ms(t4 - t3).count();

        // ---- Phase 3: Evaporation (distributed by rows) ----
        auto t5 = clock_hr::now();
        // Each rank evaporates its own rows directly in the buffer
        for (int i = my_row_start; i < my_row_start + my_row_count; ++i) {
            for (std::size_t j = 1; j <= dim; ++j) {
                local_buf_ptr[(i * stride + j) * 2 + 0] *= beta;
                local_buf_ptr[(i * stride + j) * 2 + 1] *= beta;
            }
        }

        // Gather evaporated rows from all ranks
        // Each row has stride pheronome_t = stride*2 doubles
        // Rows 1..dim are contiguous in memory starting at offset 1*stride*2
        // Use pre-allocated evap_tmp to avoid send/recv overlap issues
        double* recv_base = local_buf_ptr + 1 * stride * 2;
        
        // Copy my evaporated rows to correct position in temp buffer
        double* my_send_ptr = local_buf_ptr + my_row_start * stride * 2;
        std::memcpy(evap_tmp.data() + evap_displs[rank], my_send_ptr,
                     evap_sendcounts[rank] * sizeof(double));

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                        evap_tmp.data(), evap_sendcounts.data(), evap_displs.data(),
                        MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy back to pheromone buffer
        std::memcpy(recv_base, evap_tmp.data(), interior_doubles * sizeof(double));

        auto t6 = clock_hr::now();
        double iter_evap_ms = dur_ms(t6 - t5).count();

        // ---- Phase 4: Update pheromones ----
        auto t7 = clock_hr::now();
        phen.update();
        auto t8 = clock_hr::now();
        double iter_update_ms = dur_ms(t8 - t7).count();

        // ---- Phase 5: Rendering (rank 0 only) ----
        double iter_render_ms = 0.;
        if (rank == 0 && !headless && win_ptr && renderer_ptr) {
            // Gather all ant positions to rank 0
            MPI_Gatherv(ants.px.data(), local_count, MPI_INT,
                         all_render_px.data(), ant_counts.data(), ant_displs_v.data(), MPI_INT,
                         0, MPI_COMM_WORLD);
            MPI_Gatherv(ants.py.data(), local_count, MPI_INT,
                         all_render_py.data(), ant_counts.data(), ant_displs_v.data(), MPI_INT,
                         0, MPI_COMM_WORLD);
            auto t_r0 = clock_hr::now();
            // Reduce food to rank 0 for display
            MPI_Reduce(&local_food_total, &global_food, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            renderer_ptr->display(*win_ptr, global_food);
            win_ptr->blit();
            auto t_r1 = clock_hr::now();
            iter_render_ms = dur_ms(t_r1 - t_r0).count();
        } else if (!headless) {
            // Other ranks participate in Gatherv + Reduce
            MPI_Gatherv(ants.px.data(), local_count, MPI_INT,
                         nullptr, nullptr, nullptr, MPI_INT,
                         0, MPI_COMM_WORLD);
            MPI_Gatherv(ants.py.data(), local_count, MPI_INT,
                         nullptr, nullptr, nullptr, MPI_INT,
                         0, MPI_COMM_WORLD);
            MPI_Reduce(&local_food_total, &global_food, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        // In headless mode, reduce food to detect first food arrival.
        // MPI_Allreduce so all ranks know when to stop checking.
        if (headless && not_food_yet) {
            std::size_t tmp_global = 0;
            MPI_Allreduce(&local_food_total, &tmp_global, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
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
        total_render_ms += iter_render_ms;
        total_loop_ms   += iter_loop_ms;

        // Non-headless: rank 0 detects first food from the MPI_Reduce in rendering block
        if (!headless && rank == 0 && not_food_yet && global_food > 0) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_yet = false;
        }

        if (max_iterations > 0 && it >= max_iterations) cont_loop = false;

        // Broadcast cont_loop from rank 0 (in case of SDL_QUIT)
        int cont_int = cont_loop ? 1 : 0;
        MPI_Bcast(&cont_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        cont_loop = (cont_int != 0);
    }

    // === Final food reduction ===
    MPI_Reduce(&local_food_total, &global_food, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // === Performance report (rank 0) ===
    if (rank == 0) {
        std::cout << "\n=============================================" << std::endl;
        std::cout << "  RAPPORT DE PERFORMANCE (MPI, " << nprocs << " processus";
        if (actual_threads > 1) std::cout << " × " << actual_threads << " threads";
        std::cout << ")" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Itérations effectuées : " << it << std::endl;
        std::cout << "Nourriture collectée  : " << global_food << std::endl;
        std::cout << "Processus MPI         : " << nprocs << std::endl;
        std::cout << "Threads OpenMP/proc   : " << actual_threads << std::endl;
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

        print_row("Avancement fourmis",         total_ants_ms);
        print_row("Communication phéromones",    total_comm_ms);
        print_row("Évaporation phéromones",      total_evap_ms);
        print_row("Mise à jour phéromones",      total_update_ms);
        print_row("Rendu graphique",             total_render_ms);
        std::cout << "---------------------------------------------" << std::endl;
        print_row("TOTAL boucle",                total_loop_ms);
        std::cout << "=============================================" << std::endl;
    }

    delete renderer_ptr;
    delete win_ptr;
    if (rank == 0 && !headless) SDL_Quit();
    MPI_Finalize();
    return 0;
}
