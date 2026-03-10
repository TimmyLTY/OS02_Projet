#!/usr/bin/env python3
"""
Phase 3 — OpenMP Scaling Visualization
Generates performance plots from omp_scaling.txt results.
Output: results/omp_speedup.png, results/omp_breakdown.png, results/omp_efficiency.png
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Raw data from omp_scaling.txt (3 runs per thread count)
# ============================================================
data = {
    # threads: [(ants_ms, evap_ms, update_ms, total_ms), ...]
    1:  [(8357.162, 733.968, 4.292, 9095.585),
         (8367.566, 734.987, 4.351, 9107.062),
         (8369.873, 737.822, 4.379, 9112.251)],
    2:  [(4279.712, 396.729, 6.616, 4683.319),
         (4290.136, 422.575, 7.441, 4720.465),
         (4322.780, 414.836, 8.074, 4746.030)],
    4:  [(2236.713, 253.453, 9.558, 2500.266),
         (2231.322, 266.572, 9.469, 2507.980),
         (2249.613, 267.342, 9.696, 2527.244)],
    8:  [(1364.162, 299.007, 14.420, 1678.225),
         (1557.932, 397.663, 16.549, 1972.738),
         (1328.854, 264.308, 14.757, 1608.462)],
    12: [(2138.553, 1147.500, 20.399, 3307.100),
         (2190.142, 1187.398, 22.528, 3400.812),
         (2442.924, 1316.877, 24.081, 3784.480)],
}

# SoA sequential baseline (from Phase 2)
T_seq = 9003.0  # SoA median from Phase 2

# ============================================================
# Compute medians
# ============================================================
threads = sorted(data.keys())
medians = {}
for t in threads:
    runs = data[t]
    ants   = sorted([r[0] for r in runs])[1]  # median of 3
    evap   = sorted([r[1] for r in runs])[1]
    update = sorted([r[2] for r in runs])[1]
    total  = sorted([r[3] for r in runs])[1]
    medians[t] = (ants, evap, update, total)

# Reference T1 from OpenMP 1-thread
T1_omp = medians[1][3]

# Arrays for plotting
t_arr = np.array(threads)
total_arr = np.array([medians[t][3] for t in threads])
ants_arr  = np.array([medians[t][0] for t in threads])
evap_arr  = np.array([medians[t][1] for t in threads])
update_arr = np.array([medians[t][2] for t in threads])

# Speedup relative to OpenMP 1-thread
speedup_arr = T1_omp / total_arr
# Speedup per component
speedup_ants = medians[1][0] / ants_arr
speedup_evap = medians[1][1] / evap_arr

# Efficiency
efficiency_arr = speedup_arr / t_arr * 100

# Amdahl's law theoretical speedup
# f = fraction parallelizable ≈ (ants + evap) / total at 1 thread
f = (medians[1][0] + medians[1][1]) / medians[1][3]  # ~0.9995
t_cont = np.linspace(1, 12, 100)
amdahl = 1.0 / ((1 - f) + f / t_cont)

# ============================================================
# Figure 1: Speedup curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(t_cont, t_cont, 'k--', alpha=0.4, label='Idéal (linéaire)')
ax.plot(t_cont, amdahl, 'g-', alpha=0.6, linewidth=2, label=f'Amdahl (f={f:.4f})')
ax.plot(t_arr, speedup_arr, 'bo-', linewidth=2, markersize=8, label='Mesuré (total)')
ax.plot(t_arr, speedup_ants, 'rs--', linewidth=1.5, markersize=6, label='Fourmis seules')
ax.plot(t_arr, speedup_evap, 'g^--', linewidth=1.5, markersize=6, label='Évaporation seule')

# Annotate speedup values
for i, t in enumerate(threads):
    ax.annotate(f'{speedup_arr[i]:.2f}×',
                (t, speedup_arr[i]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=9, color='blue')

ax.set_xlabel('Nombre de threads', fontsize=12)
ax.set_ylabel('Accélération $S(p) = T_1 / T_p$', fontsize=12)
ax.set_title('Phase 3 — Passage à l\'échelle OpenMP\n(5000 fourmis, 5000 itérations, grille 513×513)', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.set_xticks(threads)
ax.set_xlim(0.5, 12.5)
ax.set_ylim(0, 13)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig('/Users/timmy/OS02/Cours_Ensta_2026/projet/results/omp_speedup.png', dpi=150)
print("Saved: omp_speedup.png")

# ============================================================
# Figure 2: Stacked bar chart — time breakdown
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

x = np.arange(len(threads))
width = 0.5

bars_ants = ax2.bar(x, ants_arr, width, label='Avancement fourmis', color='#2196F3')
bars_evap = ax2.bar(x, evap_arr, width, bottom=ants_arr, label='Évaporation phéromones', color='#FF9800')
bars_upd  = ax2.bar(x, update_arr, width, bottom=ants_arr + evap_arr, label='Mise à jour', color='#4CAF50')

# Add total time labels on top
for i, t in enumerate(threads):
    total = medians[t][3]
    ax2.text(i, total + 100, f'{total:.0f} ms',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Nombre de threads', fontsize=12)
ax2.set_ylabel('Temps (ms)', fontsize=12)
ax2.set_title('Phase 3 — Décomposition du temps par composant', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels([str(t) for t in threads])
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

fig2.tight_layout()
fig2.savefig('/Users/timmy/OS02/Cours_Ensta_2026/projet/results/omp_breakdown.png', dpi=150)
print("Saved: omp_breakdown.png")

# ============================================================
# Figure 3: Efficiency + anomaly at 12 threads
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Parallel efficiency
ax3a.bar(x, efficiency_arr, width=0.5, color=['#4CAF50' if e > 50 else '#F44336' for e in efficiency_arr])
ax3a.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Efficacité idéale')
ax3a.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='Seuil 50%')
for i, e in enumerate(efficiency_arr):
    ax3a.text(i, e + 2, f'{e:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax3a.set_xlabel('Threads', fontsize=12)
ax3a.set_ylabel('Efficacité parallèle $E(p)$ (%)', fontsize=12)
ax3a.set_title('Efficacité parallèle', fontsize=13)
ax3a.set_xticks(x)
ax3a.set_xticklabels([str(t) for t in threads])
ax3a.set_ylim(0, 120)
ax3a.legend(fontsize=9)
ax3a.grid(axis='y', alpha=0.3)

# Right: Evaporation time anomaly
evap_all = {t: sorted([r[1] for r in data[t]]) for t in threads}
evap_min = [evap_all[t][0] for t in threads]
evap_med = [evap_all[t][1] for t in threads]
evap_max = [evap_all[t][2] for t in threads]

ax3b.fill_between(threads, evap_min, evap_max, alpha=0.3, color='#FF9800', label='Min-Max')
ax3b.plot(threads, evap_med, 'o-', color='#FF9800', linewidth=2, markersize=8, label='Médiane')
ax3b.axhline(y=medians[1][1] / 12, color='green', linestyle='--', alpha=0.5,
             label=f'Idéal (734/{12}={medians[1][1]/12:.0f} ms)')
for i, t in enumerate(threads):
    ax3b.annotate(f'{evap_med[i]:.0f}', (t, evap_med[i]),
                  textcoords="offset points", xytext=(10, 5), fontsize=9)

ax3b.set_xlabel('Threads', fontsize=12)
ax3b.set_ylabel('Temps d\'évaporation (ms)', fontsize=12)
ax3b.set_title('Anomalie : temps d\'évaporation', fontsize=13)
ax3b.set_xticks(threads)
ax3b.legend(fontsize=9)
ax3b.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig('/Users/timmy/OS02/Cours_Ensta_2026/projet/results/omp_efficiency.png', dpi=150)
print("Saved: omp_efficiency.png")

# ============================================================
# Print summary table for log.md
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE (for log.md)")
print("=" * 80)
print(f"{'Threads':>7} | {'Ants (ms)':>10} | {'Evap (ms)':>10} | {'Update (ms)':>11} | {'Total (ms)':>10} | {'S(p)':>6} | {'E(p)':>7} | {'Food':>5}")
print("-" * 80)
for t in threads:
    m = medians[t]
    s = T1_omp / m[3]
    e = s / t * 100
    print(f"{t:>7} | {m[0]:>10.1f} | {m[1]:>10.1f} | {m[2]:>11.1f} | {m[3]:>10.1f} | {s:>5.2f}× | {e:>6.1f}% | {2440:>5}")

print(f"\nT1 (OpenMP 1-thread) = {T1_omp:.1f} ms")
print(f"f (parallélisable) = {f:.4f}")
print(f"Best speedup: {max(speedup_arr):.2f}× at {threads[np.argmax(speedup_arr)]} threads")
print(f"Optimal point: 8 threads → S={T1_omp/medians[8][3]:.2f}× (before regression)")
