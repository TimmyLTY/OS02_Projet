#ifndef _ANT_SOA_OMP_HPP_
#define _ANT_SOA_OMP_HPP_
#include <omp.h>
#include "ant_soa.hpp"

/**
 * @brief Avance toutes les fourmis d'un pas de temps — version OpenMP
 * @details Parallélise la boucle sur les fourmis avec schedule(dynamic).
 *          Stratégie pour les conditions de course :
 *          - food_quantity : protégé par reduction(+:)
 *          - mark_pheronome() : NON protégé (méthode C — tolérance aux courses)
 *            Justification : probabilité de collision faible (~5%), algorithme 
 *            stochastique tolérant aux perturbations mineures.
 */
inline void advance_all_ants_omp(ants_soa& ants, pheronome& phen,
                                  const fractal_land& land,
                                  const position_t& pos_food,
                                  const position_t& pos_nest,
                                  std::size_t& cpteur_food)
{
    const double eps = ants_soa::s_eps;
    std::size_t local_food = 0;

    #pragma omp parallel for schedule(dynamic, 50) reduction(+:local_food)
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
                do {
                    nx = ox; ny = oy;
                    int d = rand_int32(1, 4, seed);
                    if (d == 1) nx -= 1;
                    if (d == 2) ny -= 1;
                    if (d == 3) nx += 1;
                    if (d == 4) ny += 1;
                } while (phen(nx, ny)[ind_pher] == -1.);
            } else {
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
            phen.mark_pheronome(new_pos);  // Race condition tolérée (méthode C)
            ants.positions_x[i] = nx;
            ants.positions_y[i] = ny;

            if (nx == pos_nest.x && ny == pos_nest.y) {
                if (ants.states[i] == 1) {
                    local_food += 1;
                }
                ants.states[i] = 0;
            }
            if (nx == pos_food.x && ny == pos_food.y) {
                ants.states[i] = 1;
            }
        }
    }
    cpteur_food += local_food;
}

#endif
