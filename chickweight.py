# -*- coding: utf-8 -*-
"""
Chick Weight Analysis — Diet 1 vs Diet 2
Morgan Stanley Risk Analytics — Homework
"""

import pandas as pd
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import warnings
import matplotlib
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

df = pd.read_csv('chickweight.csv')

# Count number of weight observations per chicken id
obs_perchick = df.groupby('Chick')['Time'].count()

# Most chickens have time-steps: 0, 2, 4 .. 20, 21 → 12 observations
# We impose 12 observations as completeness criterion
complete_chicks = obs_perchick[obs_perchick >= 11].index
df_clean = df[df['Chick'].isin(complete_chicks)]

print(df_clean.groupby('Diet')['Chick'].nunique())
print("\n")

# ── SELECTION BIAS CHECK ──────────────────────────────────────────────────────

# Identify incomplete chicks and their diet distribution
incomplete_chicks = obs_perchick[obs_perchick < 12].index
df_incomplete = df[df['Chick'].isin(incomplete_chicks) & (df['Diet'] == 1)]

last_obs = df_incomplete.groupby('Chick').apply(
    lambda x: x.loc[x['Time'].idxmax(), ['Time', 'weight']]
).reset_index()
last_obs.columns = ['Chick', 'Time', 'weight']

print("Incomplete chicks — last observation:")
print(last_obs)

print("\nMean weight of complete diet 1 chicks at same time points:")
for _, row in last_obs.iterrows():
    t = row['Time']
    mean_complete = df_clean[
        (df_clean['Diet'] == 1) & (df_clean['Time'] == t)
    ]['weight'].mean()
    print(f"Chick {int(row['Chick'])}: last weight={row['weight']:.0f}g "
          f"at day {int(t)} | mean complete diet 1 at day {int(t)}: {mean_complete:.1f}g")

print("\n")

# ── REGRESSOR 1 — Absolute Final Weight ──────────────────────────────────────

# On clean dataset all chicks have Time==21 as last step
final_weight = df_clean[df_clean['Time'] == 21][['Chick', 'Diet', 'weight']].copy()
final_weight.columns = ['Chick', 'Diet', 'final_weight']

print("Results for 1st regressor: absolute final weight")
summary_fw = final_weight.groupby('Diet')['final_weight'].agg(
    ['mean', 'std', 'min', 'max', 'count']).round(2)
summary_fw.columns = ['Mean', 'Std', 'Min', 'Max', 'N']
print(summary_fw)
print("\n")

for diet in [2, 3, 4]:
    g1 = final_weight[final_weight['Diet'] == 1]['final_weight']
    g2 = final_weight[final_weight['Diet'] == diet]['final_weight']
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    print(f"Diet 1 vs Diet {diet}: t={t:.3f}, p={p:.4f}")

print("\n")

# ── REGRESSOR 2 — Global Growth Rate ─────────────────────────────────────────

# Computed as (w_final - w_initial) / (t_final - t_initial)
growth_rate = df_clean.groupby(['Chick', 'Diet']).apply(
    lambda x: (x['weight'].iloc[-1] - x['weight'].iloc[0]) /
              (x['Time'].iloc[-1] - x['Time'].iloc[0])
).reset_index()
growth_rate.columns = ['Chick', 'Diet', 'growth_rate']

summary_gr = growth_rate.groupby('Diet')['growth_rate'].agg(
    ['mean', 'std', 'min', 'max', 'count']).round(2)
summary_gr.columns = ['Mean', 'Std', 'Min', 'Max', 'N']

print("Results for 2nd regressor: average growth-rate over the whole time-span")
print(summary_gr)
print("\n")

for diet in [2, 3, 4]:
    g1 = growth_rate[growth_rate['Diet'] == 1]['growth_rate']
    g2 = growth_rate[growth_rate['Diet'] == diet]['growth_rate']
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    print(f"Diet 1 vs Diet {diet}: t={t:.3f}, p={p:.4f}")

print("\n")

# ── ASSUMPTIONS — Shapiro-Wilk and Levene ────────────────────────────────────

print("Shapiro-Wilk normality test for 1st regressor (absolute weight):")
for diet in [1, 2]:
    group = final_weight[final_weight['Diet'] == diet]['final_weight']
    stat, p = shapiro(group)
    print(f"Diet {diet}: W={stat:.3f}, p={p:.4f} {'normal' if p > 0.05 else 'non-normal'}")

print("\nLevene test for equal variances (final weight):")
groups = [final_weight[final_weight['Diet'] == d]['final_weight'] for d in [1, 2]]
stat, p = levene(*groups)
print(f"W={stat:.3f}, p={p:.4f} {'equal variances' if p > 0.05 else 'unequal variances'}")

print("\n")
print("Shapiro-Wilk normality test for 2nd regressor (global growth-rate):")
for diet in [1, 2]:
    group = growth_rate[growth_rate['Diet'] == diet]['growth_rate']
    stat, p = shapiro(group)
    print(f"Diet {diet}: W={stat:.3f}, p={p:.4f} {'normal' if p > 0.05 else 'non-normal'}")

print("\nLevene test for equal variances (growth rate):")
groups = [growth_rate[growth_rate['Diet'] == d]['growth_rate'] for d in [1, 2]]
stat, p = levene(*groups)
print(f"W={stat:.3f}, p={p:.4f} {'equal variances' if p > 0.05 else 'unequal variances'}")

print("\n")

# ── REGRESSOR 3 — Stepwise Growth Rate ───────────────────────────────────────

# Average of step-wise growth rates (delta_w / delta_t) per chick
# Removes linear growth assumption; differs from global rate when delta_t is non-uniform
# Note: collapses to global rate when all delta_t are equal

def avg_stepwise_rate(x):
    x = x.sort_values('Time')
    delta_w = x['weight'].diff().dropna()
    delta_t = x['Time'].diff().dropna()
    return (delta_w / delta_t).mean()

stepwise = df_clean.groupby(['Chick', 'Diet']).apply(
    avg_stepwise_rate).reset_index()
stepwise.columns = ['Chick', 'Diet', 'stepwise_rate']

summary_swr = stepwise.groupby('Diet')['stepwise_rate'].agg(
    ['mean', 'std', 'min', 'max', 'count']).round(2)
summary_swr.columns = ['Mean', 'Std', 'Min', 'Max', 'N']

print("Results for 3rd regressor: averaged step-wise rate")
print(summary_swr)
print("\n")

for diet in [2, 3, 4]:
    g1 = stepwise[stepwise['Diet'] == 1]['stepwise_rate']
    g2 = stepwise[stepwise['Diet'] == diet]['stepwise_rate']
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    print(f"Diet 1 vs Diet {diet}: t={t:.3f}, p={p:.4f}")

print("\n")

# ── ROBUSTNESS — Mann-Whitney U ───────────────────────────────────────────────

g1_gr = growth_rate[growth_rate['Diet'] == 1]['growth_rate']
g2_gr = growth_rate[growth_rate['Diet'] == 2]['growth_rate']
stat, p = mannwhitneyu(g1_gr, g2_gr, alternative='two-sided')
print(f"Mann-Whitney U (growth-rate): stat={stat:.1f}, p={p:.4f}")

g1_sw = stepwise[stepwise['Diet'] == 1]['stepwise_rate']
g2_sw = stepwise[stepwise['Diet'] == 2]['stepwise_rate']
stat, p = mannwhitneyu(g1_sw, g2_sw, alternative='two-sided')
print(f"Mann-Whitney U (stepwise-rate): stat={stat:.1f}, p={p:.4f}")

print("\n")

# ── ROBUSTNESS — Permutation Test ────────────────────────────────────────────

observed_diff = g2_gr.mean() - g1_gr.mean()

combined = pd.concat([
    growth_rate[growth_rate['Diet'].isin([1, 2])]['growth_rate']
]).values

n1 = len(g1_gr)
p_values_perm = []

rng = np.random.default_rng(42)
for _ in range(10000):
    perm = rng.permutation(combined)
    diff = perm[n1:].mean() - perm[:n1].mean()
    p_values_perm.append(diff)

p_perm = np.mean(np.abs(p_values_perm) >= np.abs(observed_diff))
print(f"Permutation test p-value: p = {p_perm:.4f}")

print("\n")

# ── LME — Complete Case (df_clean, diet 1 and 2 only) ────────────────────────

df_12 = df_clean[df_clean['Diet'].isin([1, 2])].copy()

# Random Intercept model (baseline — violated homoskedasticity assumption)
model_ri = mixedlm("weight ~ Time + C(Diet)", df_12, groups=df_12["Chick"])
result_ri = model_ri.fit()
print("LME Random Intercept — Complete Case:")
print(result_ri.summary())

# Random Slope model (corrects heteroskedasticity)
# Each chick has its own growth rate: u_i (intercept) + v_i * t (slope)
model_rs = mixedlm("weight ~ Time + C(Diet)",
                    df_12,
                    groups=df_12["Chick"],
                    re_formula="~Time")
result_rs = model_rs.fit()
print("LME Random Slope — Complete Case:")
print(result_rs.summary())

# Sqrt transformation + Random Slope (further heteroskedasticity correction)
df_12['sqrt_weight'] = np.sqrt(df_12['weight'])
model_sqrt_rs = mixedlm("sqrt_weight ~ Time + C(Diet)",
                          df_12,
                          groups=df_12["Chick"],
                          re_formula="~Time")
result_sqrt_rs = model_sqrt_rs.fit()
print("LME Random Slope + Sqrt — Complete Case:")
print(result_sqrt_rs.summary())

print("\n")

# ── LME — Available Case (full dataset, diet 1 and 2, includes incomplete chicks) ──

df_raw = df[df['Diet'].isin([1, 2])].copy()

model_rs_available = mixedlm("weight ~ Time + C(Diet)",
                               df_raw,
                               groups=df_raw["Chick"],
                               re_formula="~Time")
result_rs_available = model_rs_available.fit()
print("LME Random Slope — Available Case:")
print(result_rs_available.summary())

print("\n")

# ── PLOTS ─────────────────────────────────────────────────────────────────────

# Boxplot — growth rate by diet
fig, ax = plt.subplots(figsize=(8, 5))
data = [growth_rate[growth_rate['Diet'] == d]['growth_rate'].values for d in [1, 2, 3, 4]]
bp = ax.boxplot(data, tick_labels=['Diet 1', 'Diet 2', 'Diet 3', 'Diet 4'], patch_artist=True)
colors = ['steelblue', 'seagreen', 'firebrick', 'darkorange']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for i, d in enumerate([1, 2, 3, 4]):
    y = growth_rate[growth_rate['Diet'] == d]['growth_rate']
    ax.scatter([i + 1] * len(y), y, color='black', alpha=0.5, s=20, zorder=5)
ax.set_title('Growth rate by diet (g/day)', fontsize=13)
ax.set_ylabel('Growth rate (g/day)')
ax.set_xlabel('Diet')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('boxplot_growth.png', dpi=150)
plt.show()

# Growth trajectories — Diet 1 and Diet 2
for diet, color, title in [(1, 'steelblue', 'Diet 1'), (2, 'seagreen', 'Diet 2')]:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    chicks = growth_rate[growth_rate['Diet'] == diet]['Chick'].values[:4]
    for chick in chicks:
        data = df_clean[df_clean['Chick'] == chick]
        ax.plot(data['Time'], data['weight'], marker='o', label=f'Chick {chick}', alpha=0.7)
    ax.set_title(f'Growth trajectories — {title}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Weight (g)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'trajectories_diet{diet}.png', dpi=150)
    plt.show()

# QQ-plots — normality assessment (global growth rate and stepwise rate)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
regressors = [
    (growth_rate, 'growth_rate', 'Global Growth Rate'),
    (stepwise, 'stepwise_rate', 'Stepwise Rate')
]
for col_idx, (df_reg, col, label) in enumerate(regressors):
    for row_idx, diet in enumerate([1, 2]):
        data = df_reg[df_reg['Diet'] == diet][col]
        ax = axes[row_idx, col_idx]
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f'{label} — Diet {diet}')
        ax.grid(alpha=0.3)
plt.suptitle('QQ-plots: normality assessment', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('qqplots.png', dpi=150)
plt.show()

# Mixed effects assumptions — Residuals vs Fitted + QQ-plots
random_effects_ri = pd.Series(
    [result_ri.random_effects[i].values[0] for i in result_ri.random_effects])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(result_ri.fittedvalues, result_ri.resid, alpha=0.5, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted (Random Intercept)')
axes[0].grid(alpha=0.3)
stats.probplot(result_ri.resid, dist="norm", plot=axes[1])
axes[1].set_title('QQ-plot: Residuals')
axes[1].grid(alpha=0.3)
stats.probplot(random_effects_ri, dist="norm", plot=axes[2])
axes[2].set_title('QQ-plot: Random Effects $u_i$')
axes[2].grid(alpha=0.3)
plt.suptitle('Mixed Effects Model — Assumption Checks', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mixed_effects_assumptions.png', dpi=150)
plt.show()

# Heteroskedasticity comparison — three models
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
models_plot = [
    (result_ri, 'steelblue', 'Random Intercept'),
    (result_rs, 'seagreen', 'Random Slope'),
    (result_sqrt_rs, 'firebrick', 'Random Slope + Sqrt')
]
for ax, (result, color, title) in zip(axes, models_plot):
    ax.scatter(result.fittedvalues, result.resid, alpha=0.5, color=color)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    ax.grid(alpha=0.3)
plt.suptitle('Heteroskedasticity comparison — three models', fontsize=13)
plt.tight_layout()
plt.savefig('heteroskedasticity_3models.png', dpi=150)
plt.show()

# Permutation test — null distribution
plt.figure(figsize=(8, 5))
plt.hist(p_values_perm, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
plt.axvline(observed_diff, color='red', lw=2, label=f'Observed diff = {observed_diff:.2f}')
plt.axvline(-observed_diff, color='red', lw=2, linestyle='--', label=f'Mirror = {-observed_diff:.2f}')
plt.xlabel('Permuted mean difference (g/day)')
plt.ylabel('Count')
plt.title('Permutation test — Diet 1 vs Diet 2 (10000 permutations)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('permutation_dist.png', dpi=150)
plt.show()

# from statsmodels.genmod.generalized_estimating_equations import GEE
# from statsmodels.genmod.families import Gamma
import statsmodels.api as sm

# model_gee = GEE.from_formula("weight ~ Time + C(Diet)", 
#                               groups="Chick",
#                               data=df_12,
#                               family=Gamma())
# result_gee = model_gee.fit()
# print(result_gee.summary())

# # Plot 4 modelli
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# models_plot = [
#     (result_ri, 'steelblue', 'Random Intercept'),
#     (result_rs, 'seagreen', 'Random Slope'),
#     (result_sqrt_rs, 'firebrick', 'Random Slope + Sqrt'),
#     (result_gee, 'darkorange', 'GEE Gamma')
# ]

for ax, (result, color, title) in zip(axes, models_plot):
    ax.scatter(result.fittedvalues, result.resid,
               alpha=0.5, color=color)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    ax.grid(alpha=0.3)

plt.suptitle('Heteroskedasticity comparison — four models', fontsize=13)
plt.tight_layout()
plt.savefig('heteroskedasticity_4models.png', dpi=150)
plt.show()

from statsmodels.stats.diagnostic import het_breuschpagan

# Per ogni modello
for result, name in [(result_ri, 'Random Intercept'), 
                      (result_rs, 'Random Slope'),
                      (result_sqrt_rs, 'Random Slope + Sqrt')]:
    
    resid = result.resid
    fitted = result.fittedvalues
    exog = sm.add_constant(np.column_stack([fitted, fitted**2]))
    
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)  # usa exog, non ricostruirlo
    print(f"{name}: BP stat={bp_stat:.3f}, p={bp_p:.4f} "
          f"{'heteroskedastic' if bp_p < 0.05 else 'homoskedastic'}")