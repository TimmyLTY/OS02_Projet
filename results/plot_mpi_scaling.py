#!/usr/bin/env python3
"""
Phase 4A — MPI Scaling Visualization
=====================================
Generates 3 charts for the MPI performance analysis:
  1. mpi_scaling_chart.png  — Component breakdown vs. number of MPI processes
  2. mpi_vs_omp.png         — Speedup comparison: OpenMP / MPI / Hybrid
  3. mpi_comm_ratio.png     — Communication ratio vs. number of processes

Usage:
  cd /mnt/mac/Users/timmy/OS02/Cours_Ensta_2026/projet/results
  python3 plot_mpi_scaling.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data: MPI pure scaling (median of 3 runs, -t 1)
# ============================================================
mpi_procs   = np.array([1,      2,      4,       8,       12])
mpi_ants    = np.array([8540.1, 4607.4, 2634.0,  1717.8,  1713.6])
mpi_comm    = np.array([764.7,  3060.9, 5266.1,  16093.0, 67392.7])
mpi_evap    = np.array([1114.0, 1602.1, 3639.8,  8628.6,  32466.8])
mpi_update  = np.array([6.2,    8.7,    15.3,    14.1,    22.1])
mpi_total   = np.array([10423.5,9323.8, 11655.3, 27887.4, 112636.5])

# ============================================================
# Data: OpenMP scaling (median of 3 runs, from Phase 3)
# ============================================================
omp_threads = np.array([1,      2,      4,      8,      12])
omp_total   = np.array([9107.1, 4720.5, 2508.0, 1678.2, 3400.8])

# ============================================================
# Data: Hybrid MPI+OpenMP (median of 3 runs)
# ============================================================
hybrid_labels = ["P1T8", "P2T4", "P4T2", "P2T2", "P4T1"]
hybrid_cores  = np.array([8,      8,      8,      4,      4])
hybrid_total  = np.array([10434.5,9339.7, 10884.6,8964.5, 11090.4])

# Reference: sequential = OpenMP 1 thread
T_seq = 9107.1

# ============================================================
# Figure 1: MPI Component Breakdown (stacked bar + total line)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(mpi_procs))
width = 0.6

# Other time (total - ants - comm - evap - update)
mpi_other = mpi_total - mpi_ants - mpi_comm - mpi_evap - mpi_update
mpi_other = np.maximum(mpi_other, 0)  # safety

bars_ants   = ax1.bar(x, mpi_ants,   width, label="Fourmis (avancement)", color="#2ecc71")
bars_comm   = ax1.bar(x, mpi_comm,   width, bottom=mpi_ants, label="Communication (Allreduce)", color="#e74c3c")
bars_evap   = ax1.bar(x, mpi_evap,   width, bottom=mpi_ants+mpi_comm, label="Évaporation (Allgatherv)", color="#f39c12")
bars_update = ax1.bar(x, mpi_update, width, bottom=mpi_ants+mpi_comm+mpi_evap, label="Mise à jour", color="#3498db")

# Total line
ax1.plot(x, mpi_total, "ko--", linewidth=2, markersize=8, label="Total", zorder=5)

# Add total value labels above each bar
for i, total in enumerate(mpi_total):
    ax1.annotate(f"{total:.0f} ms",
                 xy=(x[i], total), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

ax1.set_xlabel("Nombre de processus MPI ($P$)", fontsize=12)
ax1.set_ylabel("Temps (ms)", fontsize=12)
ax1.set_title("Phase 4A — Décomposition du temps MPI (méthode 1 : réplication complète)", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels([str(p) for p in mpi_procs])
ax1.legend(loc="upper left", fontsize=10)
ax1.set_yscale("log")
ax1.set_ylim(bottom=1000, top=200000)
ax1.grid(axis="y", alpha=0.3)

fig1.tight_layout()
fig1.savefig("mpi_scaling_chart.png", dpi=150, bbox_inches="tight")
print("✅ mpi_scaling_chart.png saved")

# ============================================================
# Figure 2: Speedup Comparison — OpenMP vs MPI vs Hybrid
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Speedup = T_seq / T_config
omp_speedup = T_seq / omp_total
mpi_speedup = T_seq / mpi_total

# Hybrid: group by 8-core configs for fair comparison
# Only plot the 8-core hybrid configs alongside 8-core OpenMP and MPI
hybrid_8core_labels = ["P1T8", "P2T4", "P4T2"]
hybrid_8core_total  = np.array([10434.5, 9339.7, 10884.6])
hybrid_8core_speedup = T_seq / hybrid_8core_total

# OpenMP line
ax2.plot(omp_threads, omp_speedup, "s-", color="#2ecc71", linewidth=2.5,
         markersize=10, label="OpenMP pur", zorder=5)

# MPI line
ax2.plot(mpi_procs, mpi_speedup, "D-", color="#e74c3c", linewidth=2.5,
         markersize=10, label="MPI pur ($T=1$)", zorder=5)

# Ideal line
ideal = np.arange(1, 13)
ax2.plot(ideal, ideal, "k--", alpha=0.3, linewidth=1, label="Idéal")

# 1.0 reference line
ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

# Hybrid markers
hybrid_all_procs = [1, 2, 4, 2, 4]
hybrid_all_speedup = T_seq / hybrid_total
ax2.scatter(hybrid_cores, T_seq / hybrid_total, marker="^", s=120,
            color="#9b59b6", zorder=6, label="Hybride MPI+OMP")
for i, lbl in enumerate(hybrid_labels):
    ax2.annotate(lbl, xy=(hybrid_cores[i], (T_seq/hybrid_total)[i]),
                 xytext=(8, -5), textcoords="offset points", fontsize=8, color="#9b59b6")

# Annotations
ax2.annotate("5.43× (8T)", xy=(8, omp_speedup[3]), xytext=(9.5, 5.0),
             arrowprops=dict(arrowstyle="->", color="#2ecc71"),
             fontsize=10, fontweight="bold", color="#2ecc71")
ax2.annotate("0.09× (12P)", xy=(12, mpi_speedup[4]), xytext=(9, 0.5),
             arrowprops=dict(arrowstyle="->", color="#e74c3c"),
             fontsize=10, fontweight="bold", color="#e74c3c")

ax2.set_xlabel("Nombre de cœurs / processus", fontsize=12)
ax2.set_ylabel("Accélération $S = T_{seq} / T$", fontsize=12)
ax2.set_title("Phase 4A — Comparaison OpenMP vs MPI vs Hybride\n(référence : OpenMP 1T = 9107 ms)", fontsize=13)
ax2.set_xlim(0, 13)
ax2.set_ylim(-0.5, 7)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig("mpi_vs_omp.png", dpi=150, bbox_inches="tight")
print("✅ mpi_vs_omp.png saved")

# ============================================================
# Figure 3: Communication Ratio
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: stacked area of time components (% of total)
pct_ants   = mpi_ants / mpi_total * 100
pct_comm   = mpi_comm / mpi_total * 100
pct_evap   = mpi_evap / mpi_total * 100
pct_update = mpi_update / mpi_total * 100

ax3a.stackplot(mpi_procs, pct_ants, pct_comm, pct_evap, pct_update,
               labels=["Fourmis", "Communication (Allreduce)", "Évaporation (Allgatherv)", "Mise à jour"],
               colors=["#2ecc71", "#e74c3c", "#f39c12", "#3498db"],
               alpha=0.85)
ax3a.set_xlabel("Nombre de processus MPI ($P$)", fontsize=12)
ax3a.set_ylabel("% du temps total", fontsize=12)
ax3a.set_title("Décomposition relative du temps", fontsize=12)
ax3a.set_xlim(1, 12)
ax3a.set_ylim(0, 100)
ax3a.legend(loc="center right", fontsize=9)
ax3a.grid(axis="y", alpha=0.3)

# Right: communication-to-computation ratio
# Communication = comm + evap (both involve MPI collectives)
comm_total = mpi_comm + mpi_evap
calc_total = mpi_ants + mpi_update
ratio = comm_total / calc_total

ax3b.plot(mpi_procs, ratio, "o-", color="#e74c3c", linewidth=2.5, markersize=10)
for i, r in enumerate(ratio):
    ax3b.annotate(f"{r:.1f}×", xy=(mpi_procs[i], r), xytext=(0, 12),
                  textcoords="offset points", ha="center", fontsize=10, fontweight="bold")

ax3b.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Comm. = Calcul")
ax3b.set_xlabel("Nombre de processus MPI ($P$)", fontsize=12)
ax3b.set_ylabel("Ratio communication / calcul", fontsize=12)
ax3b.set_title("Évolution du ratio comm./calcul", fontsize=12)
ax3b.set_yscale("log")
ax3b.set_xlim(0, 13)
ax3b.legend(fontsize=10)
ax3b.grid(alpha=0.3)

fig3.suptitle("Phase 4A — Analyse de l'overhead de communication MPI", fontsize=14, y=1.02)
fig3.tight_layout()
fig3.savefig("mpi_comm_ratio.png", dpi=150, bbox_inches="tight")
print("✅ mpi_comm_ratio.png saved")

print("\n📊 All 3 charts generated successfully!")
print("   → mpi_scaling_chart.png")
print("   → mpi_vs_omp.png")
print("   → mpi_comm_ratio.png")
