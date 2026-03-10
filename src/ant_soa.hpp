#ifndef _ANT_SOA_HPP_
#define _ANT_SOA_HPP_
#include <vector>
#include <cstddef>
#include "pheronome.hpp"
#include "fractal_land.hpp"
#include "basic_types.hpp"
#include "rand_generator.hpp"

/**
 * @brief Structure of Arrays (SoA) pour les fourmis — version vectorisée
 * @details Remplace std::vector<ant> (AoS) par des tableaux séparés
 *          pour chaque attribut, améliorant la localité mémoire (cache).
 */
struct ants_soa {
    std::vector<int>         positions_x;  // Coordonnée x de chaque fourmi
    std::vector<int>         positions_y;  // Coordonnée y de chaque fourmi
    std::vector<int>         states;       // 0 = unloaded, 1 = loaded
    std::vector<std::size_t> seeds;        // Graine aléatoire par fourmi
    std::size_t              count;        // Nombre total de fourmis

    static double s_eps; // Coefficient d'exploration (partagé)

    /**
     * @brief Initialise N fourmis à des positions aléatoires
     * @note Dans le code original (AoS), le membre m_seed de chaque fourmi
     *       n'est PAS initialisé par le constructeur (bug). On reproduit le
     *       même comportement ici en initialisant seeds à 0 (valeur par
     *       défaut de la mémoire allouée par std::vector dans le code original).
     *       Cela garantit des résultats identiques à la version AoS.
     */
    void init(std::size_t nb_ants, unsigned long land_dim, std::size_t& seed) {
        count = nb_ants;
        positions_x.resize(nb_ants);
        positions_y.resize(nb_ants);
        states.resize(nb_ants, 0); // toutes non chargées
        seeds.resize(nb_ants, 0);  // reproduit le bug original (m_seed non initialisé → 0)
        for (std::size_t i = 0; i < nb_ants; ++i) {
            positions_x[i] = rand_int32(0, land_dim - 1, seed);
            positions_y[i] = rand_int32(0, land_dim - 1, seed);
            // Note : on ne fait PAS seeds[i] = seed; on laisse à 0
            // pour reproduire le comportement du code original (m_seed non initialisé)
        }
    }

    static void set_exploration_coef(double eps) { s_eps = eps; }
};

/**
 * @brief Avance toutes les fourmis d'un pas de temps (version SoA)
 */
inline void advance_all_ants(ants_soa& ants, pheronome& phen,
                              const fractal_land& land,
                              const position_t& pos_food,
                              const position_t& pos_nest,
                              std::size_t& cpteur_food)
{
    const double eps = ants_soa::s_eps;
    for (std::size_t i = 0; i < ants.count; ++i) {
        std::size_t& seed = ants.seeds[i];
        double consumed_time = 0.;

        while (consumed_time < 1.) {
            int ind_pher = (ants.states[i] == 1) ? 1 : 0;
            double choix = rand_double(0., 1., seed);

            int ox = ants.positions_x[i];
            int oy = ants.positions_y[i];
            int nx = ox, ny = oy;

            double max_phen = std::max({
                phen(ox - 1, oy)[ind_pher],
                phen(ox + 1, oy)[ind_pher],
                phen(ox, oy - 1)[ind_pher],
                phen(ox, oy + 1)[ind_pher]
            });

            if ((choix > eps) || (max_phen <= 0.)) {
                // Exploration aléatoire
                do {
                    nx = ox; ny = oy;
                    int d = rand_int32(1, 4, seed);
                    if (d == 1) nx -= 1;
                    if (d == 2) ny -= 1;
                    if (d == 3) nx += 1;
                    if (d == 4) ny += 1;
                } while (phen(nx, ny)[ind_pher] == -1.);
            } else {
                // Suivi du phéromone le plus fort
                if (phen(ox - 1, oy)[ind_pher] == max_phen)
                    nx = ox - 1;
                else if (phen(ox + 1, oy)[ind_pher] == max_phen)
                    nx = ox + 1;
                else if (phen(ox, oy - 1)[ind_pher] == max_phen)
                    ny = oy - 1;
                else
                    ny = oy + 1;
            }

            consumed_time += land(nx, ny);
            position_t new_pos{nx, ny};
            phen.mark_pheronome(new_pos);
            ants.positions_x[i] = nx;
            ants.positions_y[i] = ny;

            if (nx == pos_nest.x && ny == pos_nest.y) {
                if (ants.states[i] == 1) {
                    cpteur_food += 1;
                }
                ants.states[i] = 0;
            }
            if (nx == pos_food.x && ny == pos_food.y) {
                ants.states[i] = 1;
            }
        }
    }
}

#endif
