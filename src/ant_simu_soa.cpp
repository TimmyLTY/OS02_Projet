#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include "fractal_land.hpp"
#include "ant_soa.hpp"
#include "pheronome.hpp"
#include "renderer.hpp"
#include "window.hpp"
#include "rand_generator.hpp"

// ============================================================================
// Utilitaire de chronométrage
// ============================================================================
using clock_t_hr = std::chrono::high_resolution_clock;
using duration_ms = std::chrono::duration<double, std::milli>;

double ants_soa::s_eps = 0.; // Définition du membre statique

void advance_time_soa( const fractal_land& land, pheronome& phen,
                       const position_t& pos_nest, const position_t& pos_food,
                       ants_soa& ants, std::size_t& cpteur,
                       double& time_ants_ms, double& time_evap_ms, double& time_update_ms )
{
    auto t0 = clock_t_hr::now();
    advance_all_ants(ants, phen, land, pos_food, pos_nest, cpteur);
    auto t1 = clock_t_hr::now();
    phen.do_evaporation();
    auto t2 = clock_t_hr::now();
    phen.update();
    auto t3 = clock_t_hr::now();

    time_ants_ms   = duration_ms(t1 - t0).count();
    time_evap_ms   = duration_ms(t2 - t1).count();
    time_update_ms = duration_ms(t3 - t2).count();
}

int main(int nargs, char* argv[])
{
    // --- Paramètre : nombre max d'itérations (0 = infini / interactif) ---
    std::size_t max_iterations = 0;
    bool headless = false;
    for (int a = 1; a < nargs; ++a) {
        std::string arg(argv[a]);
        if (arg == "--iterations" || arg == "-n") {
            if (a + 1 < nargs) max_iterations = std::stoull(argv[++a]);
        }
        if (arg == "--headless" || arg == "-H") {
            headless = true;
        }
    }

    if (!headless) SDL_Init( SDL_INIT_VIDEO );

    std::size_t seed = 2026; // Graine pour la génération aléatoire ( reproductible )
    const int nb_ants = 5000; // Nombre de fourmis
    const double eps = 0.8;  // Coefficient d'exploration
    const double alpha=0.7; // Coefficient de chaos
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{256,256};
    // Location de la nourriture
    position_t pos_food{500,500};

    // === Mesure : génération du terrain fractal ===
    auto t_land_start = clock_t_hr::now();
    fractal_land land(8,2,1.,1024);
    auto t_land_end = clock_t_hr::now();
    double time_land_ms = duration_ms(t_land_end - t_land_start).count();

    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }

    // === Initialisation SoA des fourmis ===
    ants_soa::set_exploration_coef(eps);
    ants_soa ants;
    ants.init(nb_ants, land.dimensions(), seed);

    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    // --- Fenêtre SDL (seulement si pas headless) ---
    Window* win_ptr = nullptr;
    Renderer* renderer_ptr = nullptr;
    if (!headless) {
        win_ptr = new Window("Ant Simulation (SoA)", 2*land.dimensions()+10, land.dimensions()+266);
        renderer_ptr = new Renderer( land, phen, pos_nest, pos_food,
                                      ants.positions_x, ants.positions_y, ants.count );
    }

    size_t food_quantity = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    // === Accumulateurs de temps ===
    double total_ants_ms   = 0.0;
    double total_evap_ms   = 0.0;
    double total_update_ms = 0.0;
    double total_render_ms = 0.0;
    double total_loop_ms   = 0.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== Mesure des performances (vectorisation SoA) ===" << std::endl;
    std::cout << "Génération du terrain fractal : " << time_land_ms << " ms" << std::endl;
    std::cout << "Grille : " << land.dimensions() << " x " << land.dimensions() << std::endl;
    std::cout << "Nombre de fourmis : " << nb_ants << std::endl;
    if (max_iterations > 0)
        std::cout << "Mode benchmark : " << max_iterations << " itérations" << std::endl;
    std::cout << "Mode headless : " << (headless ? "oui" : "non") << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    while (cont_loop) {
        ++it;
        auto t_loop_start = clock_t_hr::now();

        if (!headless) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    cont_loop = false;
            }
        }

        double iter_ants_ms, iter_evap_ms, iter_update_ms;
        advance_time_soa( land, phen, pos_nest, pos_food, ants, food_quantity,
                          iter_ants_ms, iter_evap_ms, iter_update_ms );

        double iter_render_ms = 0.0;
        if (!headless && win_ptr && renderer_ptr) {
            auto t_render_start = clock_t_hr::now();
            renderer_ptr->display( *win_ptr, food_quantity );
            win_ptr->blit();
            auto t_render_end = clock_t_hr::now();
            iter_render_ms = duration_ms(t_render_end - t_render_start).count();
        }

        auto t_loop_end = clock_t_hr::now();
        double iter_loop_ms = duration_ms(t_loop_end - t_loop_start).count();

        total_ants_ms   += iter_ants_ms;
        total_evap_ms   += iter_evap_ms;
        total_update_ms += iter_update_ms;
        total_render_ms += iter_render_ms;
        total_loop_ms   += iter_loop_ms;

        if ( not_food_in_nest && food_quantity > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }

        // --- Arrêt après max_iterations si spécifié ---
        if (max_iterations > 0 && it >= max_iterations) {
            cont_loop = false;
        }
    }

    // === Rapport de performance ===
    std::cout << "\n=============================================" << std::endl;
    std::cout << "  RAPPORT DE PERFORMANCE (version SoA)" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Itérations effectuées : " << it << std::endl;
    std::cout << "Nourriture collectée  : " << food_quantity << std::endl;
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

    print_row("Avancement fourmis",   total_ants_ms);
    print_row("Évaporation phéromones", total_evap_ms);
    print_row("Mise à jour phéromones", total_update_ms);
    print_row("Rendu graphique",       total_render_ms);
    std::cout << "---------------------------------------------" << std::endl;
    print_row("TOTAL boucle",          total_loop_ms);
    std::cout << "=============================================" << std::endl;

    // --- Nettoyage ---
    delete renderer_ptr;
    delete win_ptr;
    if (!headless) SDL_Quit();
    return 0;
}
