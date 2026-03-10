#pragma once
#include "fractal_land.hpp"
#ifndef USE_SOA
#include "ant.hpp"
#endif
#include "pheronome.hpp"
#include "window.hpp"

class Renderer
{
public:
#ifdef USE_SOA
    // Constructeur SoA : accepte des vecteurs de positions séparés
    Renderer(  const fractal_land& land, const pheronome& phen, 
               const position_t& pos_nest, const position_t& pos_food,
               const std::vector<int>& positions_x,
               const std::vector<int>& positions_y,
               std::size_t nb_ants );
#else
    Renderer(  const fractal_land& land, const pheronome& phen, 
               const position_t& pos_nest, const position_t& pos_food,
               const std::vector<ant>& ants );
#endif

    Renderer(const Renderer& ) = delete;
    ~Renderer();

    void display( Window& win, std::size_t const& compteur );
private:
    fractal_land const& m_ref_land;
    SDL_Texture* m_land{ nullptr }; 
    const pheronome& m_ref_phen;
    const position_t& m_pos_nest;
    const position_t& m_pos_food;
#ifdef USE_SOA
    const std::vector<int>& m_ref_positions_x;
    const std::vector<int>& m_ref_positions_y;
    std::size_t m_nb_ants;
#else
    const std::vector<ant>& m_ref_ants;
#endif
    std::vector<std::size_t> m_curve;    
};