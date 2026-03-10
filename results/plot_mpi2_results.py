#!/usr/bin/env python3
"""
Phase 4B Visualization — MPI Method 2 (Map Partitioning) Results
Generates 4 PNG figures for log.md:
  1. mpi2_speedup_comparison.png — Speedup: OpenMP vs MPI M1 vs MPI M2
  2. mpi2_time_breakdown.png    — Time breakdown by component (M2)
  3. mpi2_ant_distribution.png  — Ant distribution across processes
  4. mpi2_comm_comparison.png   — Communication time: M1 vs M2

Usage:
  cd /path/to/projet/results
  python3 plot_mpi2_results.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Data (medians of 3 runs, 5000 iterations)
# ============================================================

# --- Sequential baseline ---
T_SEQ = 9107.1  # SoA sequential (Phase 1 reference)

# --- OpenMP (Phase 3) ---
omp_threads = [1, 2, 4, 8, 12]
omp_total   = [9107.1, 4720.5, 2508.0, 1678.2, 3400.8]

# --- MPI Method 1 (Phase 4A) ---
m1_procs    = [1, 2, 4, 8, 12]
m1_total    = [10423.5, 9323.8, 11655.3, 27887.4, 112636.5]
m1_comm     = [764.7, 4663.0, 8905.9, 24721.6, 99859.5]

# --- MPI Method 2 (Phase 4B) — pure MPI ---
m2_procs    = [1, 2, 4, 8, 12]
m2_ants     = [8354.7, 3069.4, 1146.3, 371.2, 280.3]
m2_comm     = [2.0, 2345.4, 2123.8, 1991.7, 16774.9]
m2_evap     = [475.0, 222.4, 101.2, 65.6, 77.2]
m2_update   = [5.5, 74.3, 61.9, 458.4, 6999.8]
m2_total    = [8881.4, 5721.9, 3443.8, 3216.5, 39814.6]

# --- MPI M2 + OMP Hybrid ---
hybrid_labels = ['P1T8', 'P2T4', 'P2T6', 'P3T4', 'P4T2', 'P4T3', 'P6T2']
hybrid_cores  = [8, 8, 12, 12, 8, 12, 12]
hybrid_total  = [8874.6, 5553.6, 5563.8, 4153.0, 3316.9, 3317.1, 2779.4]

# --- Ant distribution (P=4 and P=8) ---
ant_dist_p4 = {'R0': 303, 'R1': 1342, 'R2': 1376, 'R3': 1979}
ant_dist_p8 = {'R0': 125, 'R1': 180, 'R2': 268, 'R3': 904,
               'R4': 850, 'R5': 613, 'R6': 625, 'R7': 1435}
ant_dist_p12 = {'R0': 33, 'R1': 148, 'R2': 131, 'R3': 145, 'R4': 372,
                'R5': 516, 'R6': 581, 'R7': 453, 'R8': 365, 'R9': 312,
                'R10': 595, 'R11': 1349}

plt.rcParams.update({
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# Figure 1: Speedup Comparison — OpenMP vs MPI M1 vs MPI M2
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Compute speedups (all relative to T_SEQ)
omp_speedup = [T_SEQ / t for t in omp_total]
m1_speedup  = [T_SEQ / t for t in m1_total]
m2_speedup  = [T_SEQ / t for t in m2_total]

# Ideal line
x_ideal = np.arange(1, 13)
ax.plot(x_ideal, x_ideal, 'k--', alpha=0.3, label='Idéal (linéaire)')

ax.plot(omp_threads, omp_speedup, 'g-o', linewidth=2, markersize=8, label='OpenMP', zorder=5)
ax.plot(m2_procs, m2_speedup, 'b-s', linewidth=2, markersize=8, label='MPI M2 (partitionnement)', zorder=5)
ax.plot(m1_procs, m1_speedup, 'r-^', linewidth=2, markersize=8, label='MPI M1 (réplication)', zorder=5)

# Best hybrid point
ax.plot(12, T_SEQ / 2779.4, 'mD', markersize=12, label='M2 Hybride P=6,T=2', zorder=6)

# Annotate key points
ax.annotate(f'OMP 8T\n{1678} ms\n5.43×', xy=(8, T_SEQ/1678), xytext=(9.5, 5.8),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='green'),
            color='green', fontweight='bold')
ax.annotate(f'M2 P=8\n{3217} ms\n2.83×', xy=(8, T_SEQ/3217), xytext=(10, 3.2),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='blue'),
            color='blue')
ax.annotate(f'M1 P=12\n0.08×', xy=(12, T_SEQ/112636.5), xytext=(10.5, 0.5),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='red'),
            color='red')

ax.set_xlabel('Nombre de cœurs / processus')
ax.set_ylabel('Accélération $S = T_{seq} / T_{par}$')
ax.set_title('Comparaison des accélérations — Phase 4B\n(OpenMP vs MPI M1 vs MPI M2, réf. séquentiel = 9107 ms)')
ax.set_xlim(0.5, 12.5)
ax.set_ylim(-0.5, 7)
ax.set_xticks([1, 2, 4, 6, 8, 12])
ax.legend(loc='upper left', fontsize=10)
ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle=':')

plt.tight_layout()
plt.savefig('mpi2_speedup_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ mpi2_speedup_comparison.png")

# ============================================================
# Figure 2: Time Breakdown — MPI Method 2
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(m2_procs))
width = 0.6

bars_ants   = ax.bar(x, m2_ants, width, label='Fourmis', color='#2196F3')
bars_comm   = ax.bar(x, m2_comm, width, bottom=m2_ants, label='Communication', color='#FF5722')
bars_evap   = ax.bar(x, m2_evap, width, bottom=[a+c for a,c in zip(m2_ants, m2_comm)],
                     label='Évaporation', color='#4CAF50')
bars_update = ax.bar(x, m2_update, width,
                     bottom=[a+c+e for a,c,e in zip(m2_ants, m2_comm, m2_evap)],
                     label='Mise à jour + halo', color='#FFC107')

# Add total time labels on top
for i, total in enumerate(m2_total):
    ax.text(i, total + 500, f'{total:.0f} ms', ha='center', fontsize=9, fontweight='bold')

# Add percentage labels for communication
for i in range(len(m2_procs)):
    comm_pct = m2_comm[i] / m2_total[i] * 100
    if comm_pct > 5:
        y_pos = m2_ants[i] + m2_comm[i] / 2
        ax.text(i, y_pos, f'{comm_pct:.0f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

ax.set_xlabel('Nombre de processus MPI')
ax.set_ylabel('Temps (ms)')
ax.set_title('Décomposition du temps — MPI Méthode 2\n(5000 itérations, médiane de 3 exécutions)')
ax.set_xticks(x)
ax.set_xticklabels([f'P={p}' for p in m2_procs])
ax.legend(loc='upper right', fontsize=10)

# Use log scale for y-axis since P=12 is much larger
ax.set_yscale('log')
ax.set_ylim(100, 80000)

plt.tight_layout()
plt.savefig('mpi2_time_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ mpi2_time_breakdown.png")

# ============================================================
# Figure 3: Ant Distribution across Processes
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (title, dist) in zip(axes, [
    ('P=4', ant_dist_p4),
    ('P=8', ant_dist_p8),
    ('P=12', ant_dist_p12),
]):
    ranks = list(dist.keys())
    counts = list(dist.values())
    colors = ['#FF5722' if c == max(counts) else '#2196F3' if c == min(counts) else '#90CAF9'
              for c in counts]
    bars = ax.bar(range(len(ranks)), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks, fontsize=8, rotation=45 if len(ranks) > 8 else 0)
    ax.set_xlabel('Rank MPI')
    ax.set_ylabel('Nombre de fourmis')
    ax.set_title(f'{title} processus\n(max/min = {max(counts)/max(min(counts),1):.1f}×)')

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(count), ha='center', va='bottom', fontsize=7)

    # Horizontal line for perfect balance
    perfect = 5000 / len(ranks)
    ax.axhline(y=perfect, color='green', linewidth=1, linestyle='--', alpha=0.7,
               label=f'Équilibre parfait ({perfect:.0f})')
    ax.legend(fontsize=8)

fig.suptitle('Distribution finale des fourmis par processus — MPI Méthode 2',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mpi2_ant_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ mpi2_ant_distribution.png")

# ============================================================
# Figure 4: Communication Time — Method 1 vs Method 2
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Communication time comparison
x = np.arange(len(m1_procs))
width = 0.35

# M2 comm = comm + update (both include synchronization)
m2_comm_total = [c + u for c, u in zip(m2_comm, m2_update)]

bars1 = ax1.bar(x - width/2, m1_comm, width, label='M1 (Allreduce + Allgatherv)',
                color='#FF5722', alpha=0.8)
bars2 = ax1.bar(x + width/2, m2_comm_total, width, label='M2 (halo + migration + sync)',
                color='#2196F3', alpha=0.8)

ax1.set_xlabel('Nombre de processus')
ax1.set_ylabel('Temps de communication (ms)')
ax1.set_title('Communication : Méthode 1 vs Méthode 2')
ax1.set_xticks(x)
ax1.set_xticklabels([f'P={p}' for p in m1_procs])
ax1.set_yscale('log')
ax1.legend(fontsize=9)

# Add ratio labels
for i in range(len(m1_procs)):
    if m2_comm_total[i] > 0 and m1_comm[i] > 10:
        ratio = m1_comm[i] / m2_comm_total[i]
        y_pos = max(m1_comm[i], m2_comm_total[i]) * 1.3
        ax1.text(i, y_pos, f'×{ratio:.1f}', ha='center', fontsize=9,
                 fontweight='bold', color='purple')

# Right: Total time comparison
bars3 = ax2.bar(x - width/2, m1_total, width, label='MPI M1 (total)',
                color='#FF5722', alpha=0.8)
bars4 = ax2.bar(x + width/2, m2_total, width, label='MPI M2 (total)',
                color='#2196F3', alpha=0.8)

# Add OpenMP 8T reference line
ax2.axhline(y=1678, color='green', linewidth=2, linestyle='--', alpha=0.7,
            label='OpenMP 8T (1678 ms)')

ax2.set_xlabel('Nombre de processus')
ax2.set_ylabel('Temps total (ms)')
ax2.set_title('Temps total : M1 vs M2 vs OpenMP')
ax2.set_xticks(x)
ax2.set_xticklabels([f'P={p}' for p in m1_procs])
ax2.set_yscale('log')
ax2.legend(fontsize=9)

# Add gain labels
for i in range(len(m1_procs)):
    if m2_total[i] > 0:
        gain = m1_total[i] / m2_total[i]
        y_pos = max(m1_total[i], m2_total[i]) * 1.3
        ax2.text(i, y_pos, f'M2 ×{gain:.1f}', ha='center', fontsize=9,
                 fontweight='bold', color='purple')

fig.suptitle('Comparaison des communications et performances — Phase 4B',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mpi2_comm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ mpi2_comm_comparison.png")

print("\n🎉 All 4 figures generated successfully!")
print("Files:")
print("  - mpi2_speedup_comparison.png")
print("  - mpi2_time_breakdown.png")
print("  - mpi2_ant_distribution.png")
print("  - mpi2_comm_comparison.png")
