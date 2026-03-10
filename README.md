# OS02 — Projet : Optimisation par Colonie de Fourmis sur Paysage Fractal

**ENSTA Paris — OS02 Systèmes Distribués et Parallélisme — Mars 2026**  
**Auteurs :** Tianyi LIANG · Tianze XIA · Zhe CHEN

---

## Présentation

Ce projet optimise un simulateur ACO (*Ant Colony Optimization*) sur une grille fractale 513×513 selon quatre approches de parallélisation :

| Approche | Meilleure config. | Accélération |
|---|---|---|
| Séquentiel (baseline) | — | 1.00× |
| Vectorisation SoA | — | 1.01× |
| **OpenMP** | **8 threads** | **5.43×** ← champion |
| MPI méthode 1 (réplication carte) | P=2 | 1.12× (échec) |
| MPI méthode 2 (partitionnement) | P=8 pur | 2.76× |
| MPI M2 hybride | P=6, T=2 | 3.28× |

> Référence séquentielle : **9051.20 ms** (5000 itérations, 5000 fourmis, headless)

---

## Structure du dépôt

```
OS02_Projet/
├── rapport_v1.tex          # Rapport LaTeX complet
├── src/                    # Sources C++ (aucun binaire)
│   ├── ant_simu.cpp        # Baseline séquentielle (AoS)
│   ├── ant_simu_soa.cpp    # Version SoA
│   ├── ant_simu_omp.cpp    # OpenMP (SoA + parallélisation)
│   ├── ant_simu_mpi.cpp    # MPI méthode 1 (réplication carte)
│   ├── ant_simu_mpi2.cpp   # MPI méthode 2 (partitionnement 1D)
│   ├── ant.cpp / ant.hpp   # Classe fourmi AoS
│   ├── ant_soa.hpp         # Structure SoA
│   ├── ant_soa_omp.hpp     # SoA + OpenMP
│   ├── pheronome.hpp       # Carte de phéromones
│   ├── fractal_land.*      # Terrain fractal
│   ├── renderer.*          # Rendu SDL2
│   ├── window.*            # Fenêtre SDL2
│   ├── basic_types.hpp     # Types de base
│   ├── rand_generator.hpp  # Générateur pseudo-aléatoire
│   ├── Makefile
│   ├── Make_linux.inc      # Config Linux (utilisée)
│   ├── Make_msys2.inc      # Config Windows/MSYS2
│   └── Make_osx.inc        # Config macOS
└── results/
    ├── baseline_3runs.txt  # Données brutes baseline
    ├── aos_3runs.txt       # Données brutes AoS
    ├── soa_3runs.txt       # Données brutes SoA
    ├── omp_scaling.txt     # Données brutes OpenMP
    ├── mpi_scaling.txt     # Données brutes MPI M1
    ├── mpi_omp_hybrid.txt  # Données brutes MPI M1 hybride
    ├── mpi2_scaling.txt    # Données brutes MPI M2
    ├── mpi2_omp_hybrid.txt # Données brutes MPI M2 hybride
    ├── plot_omp_scaling.py # Script → figures OpenMP
    ├── plot_mpi_scaling.py # Script → figures MPI M1
    ├── plot_mpi2_results.py# Script → figures MPI M2
    └── *.png               # Figures (11 graphiques)
```

---

## Compilation

**Environnement testé :** OrbStack Ubuntu 22.04 (aarch64, 12 cœurs ARM), GCC 11.4, OpenMPI, SDL2 2.0.20.

```bash
# Installer les dépendances
sudo apt install -y build-essential g++ libsdl2-dev openmpi-bin libopenmpi-dev

# Compiler toutes les cibles
cd src/
make clean && make all
```

Exécutables générés : `ant_simu.exe`, `ant_simu_soa.exe`, `ant_simu_omp.exe`, `ant_simu_mpi.exe`, `ant_simu_mpi2.exe`

---

## Exécution

Toutes les mesures utilisent `--headless` (sans rendu SDL) et `-n 5000` itérations.

```bash
# Baseline séquentielle
./ant_simu.exe --headless -n 5000

# SoA
./ant_simu_soa.exe --headless -n 5000

# OpenMP (8 threads — meilleure configuration)
./ant_simu_omp.exe --headless -n 5000 -t 8

# MPI méthode 1 (P processus)
mpirun -np 2 --oversubscribe ./ant_simu_mpi.exe --headless -n 5000 -t 1

# MPI méthode 2 pur (P=8 — meilleur pur)
mpirun -np 8 --oversubscribe ./ant_simu_mpi2.exe --headless -n 5000 -t 1

# MPI méthode 2 hybride (P=6, T=2 — meilleur global MPI)
mpirun -np 6 --oversubscribe ./ant_simu_mpi2.exe --headless -n 5000 -t 2
```

---

## Rapport

Le rapport complet est dans `rapport_v1.tex`. Compiler avec :

```bash
pdflatex rapport_v1.tex
pdflatex rapport_v1.tex   # 2e passe pour la table des matières
```

Ou importer directement sur [Overleaf](https://www.overleaf.com) avec le dossier `results/` pour les figures.

---

## Résultats clés

- **OpenMP 8T** : `1678.20 ms` — accélération `5.43×`, efficacité `67.80%`
- **MPI M1** : contre-productif dès P>2 (8.50 Mo/iter de communication collective)
- **MPI M2** : accélération super-linéaire des fourmis (`22.51×` à P=8) grâce à l'effet de cache (carte locale 0.55 Mo vs 4.20 Mo)
- **Hybride MPI M2 P=6, T=2** : `2779.40 ms` — accélération `3.28×`
- Déséquilibre de charge critique pour M2 : ratio max/min `40.88×` à P=12
