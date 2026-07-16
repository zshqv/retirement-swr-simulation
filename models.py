import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Load and clean data ----
df = pd.read_excel('chapt26.xlsx', sheet_name='Data', header=None, skiprows=8)
df_clean = df.iloc[:, [0, 1, 2, 5, 6, 15]]
df_clean.columns = ['Year', 'SP_Price', 'Dividend', 'Rate_Long10yr', 'CPI', 'Return_SP']
df_clean = df_clean[(df_clean['Year'] >= 1871) & (df_clean['Year'] <= 2012)]
df_clean = df_clean.reset_index(drop=True)
df_clean['Inflation'] = df_clean['CPI'].pct_change()
df_clean['Real_Return_SP'] = (1 + df_clean['Return_SP']) / (1 + df_clean['Inflation']) - 1
df_clean = df_clean.dropna().reset_index(drop=True)


# ---- Simulation engine ----
def simulate_path(returns, initial_balance=1_000_000, withdrawal_rate=0.04):
    """
    Applies each year's return, then subtracts a fixed withdrawal.
    Because returns here are REAL (already inflation-adjusted via the Fisher
    equation above), a constant nominal withdrawal in this space is
    equivalent to Bengen's inflation-adjusted 4% rule — there's no separate
    CPI adjustment needed on the withdrawal itself.
    """
    annual_withdrawal = initial_balance * withdrawal_rate
    balance = initial_balance
    balance_history = [balance]
    ruin_year = None
    for year, r in enumerate(returns, start=1):
        balance = balance * (1 + r)
        balance = balance - annual_withdrawal
        balance_history.append(balance)
        if balance <= 0 and ruin_year is None:
            ruin_year = year
            break
    return balance_history, ruin_year


# ---- Historical rolling-window backtest ----
def rolling_window_backtest(df, window_years=30, initial_balance=1_000_000, withdrawal_rate=0.04):
    results = []
    years = df['Year'].tolist()
    returns = df['Real_Return_SP'].tolist()
    for start_idx in range(len(years) - window_years + 1):
        start_year = years[start_idx]
        window_returns = returns[start_idx: start_idx + window_years]
        history, ruin = simulate_path(window_returns, initial_balance, withdrawal_rate)
        results.append({'start_year': start_year, 'ruin_year': ruin, 'survived': ruin is None,
                         'final_balance': history[-1]})
    return pd.DataFrame(results)


backtest = rolling_window_backtest(df_clean)


# ---- Model 1: naive i.i.d. normal ----
def simulate_iid_normal(mean_return, std_return, n_years=30, n_simulations=10_000,
                         initial_balance=1_000_000, withdrawal_rate=0.04, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for sim in range(n_simulations):
        random_returns = rng.normal(loc=mean_return, scale=std_return, size=n_years)
        history, ruin = simulate_path(random_returns, initial_balance, withdrawal_rate)
        results.append({'sim_id': sim, 'ruin_year': ruin, 'survived': ruin is None, 'final_balance': history[-1]})
    return pd.DataFrame(results)


mean_r = df_clean['Real_Return_SP'].mean()
std_r = df_clean['Real_Return_SP'].std()
N_SIMULATIONS = 10_000
iid_results = simulate_iid_normal(mean_r, std_r, n_years=30, n_simulations=N_SIMULATIONS)


# ---- Model 2: historical i.i.d. bootstrap ----
def simulate_bootstrap_iid(historical_returns, n_years=30, n_simulations=10_000,
                            initial_balance=1_000_000, withdrawal_rate=0.04, seed=42):
    rng = np.random.default_rng(seed)
    historical_returns = np.array(historical_returns)
    results = []
    for sim in range(n_simulations):
        random_returns = rng.choice(historical_returns, size=n_years, replace=True)
        history, ruin = simulate_path(random_returns, initial_balance, withdrawal_rate)
        results.append({'sim_id': sim, 'ruin_year': ruin, 'survived': ruin is None, 'final_balance': history[-1]})
    return pd.DataFrame(results)


bootstrap_results = simulate_bootstrap_iid(df_clean['Real_Return_SP'].values, n_years=30,
                                            n_simulations=N_SIMULATIONS)


# ---- Model 3: block bootstrap ----
def simulate_block_bootstrap(historical_returns, block_size=5, n_years=30, n_simulations=10_000,
                              initial_balance=1_000_000, withdrawal_rate=0.04, seed=42):
    rng = np.random.default_rng(seed)
    historical_returns = np.array(historical_returns)
    n_available = len(historical_returns)
    n_possible_starts = n_available - block_size + 1
    results = []
    for sim in range(n_simulations):
        sequence = []
        while len(sequence) < n_years:
            start_idx = rng.integers(0, n_possible_starts)
            block = historical_returns[start_idx: start_idx + block_size]
            sequence.extend(block)
        sequence = sequence[:n_years]
        history, ruin = simulate_path(sequence, initial_balance, withdrawal_rate)
        results.append({'sim_id': sim, 'ruin_year': ruin, 'survived': ruin is None, 'final_balance': history[-1]})
    return pd.DataFrame(results)


# Compute block bootstrap ONCE per block size (fixes earlier duplicate computation)
block_sizes = [3, 5, 10, 15, 20]
block_results = {
    bsize: simulate_block_bootstrap(df_clean['Real_Return_SP'].values, block_size=bsize,
                                     n_years=30, n_simulations=N_SIMULATIONS)
    for bsize in block_sizes
}

# ---- Print all headline numbers ----
print(f"Mean used: {mean_r:.4f}, Std used: {std_r:.4f}")
print(f"Historical rolling-window success rate:    {backtest['survived'].mean():.2%}  (n={len(backtest)})")
print(f"i.i.d. Normal success rate:                {iid_results['survived'].mean():.2%}  (n={N_SIMULATIONS})")
print(f"Historical i.i.d. Bootstrap success rate:  {bootstrap_results['survived'].mean():.2%}  (n={N_SIMULATIONS})")

print("\n--- Block bootstrap success rates by block size ---")
for bsize in block_sizes:
    print(f"Block size {bsize:2d}: success rate = {block_results[bsize]['survived'].mean():.2%}")


# =====================================================================
# FIGURE 1 — Success rates with 95% CI, consistent color grouping
# =====================================================================
def binomial_ci95(p, n):
    """95% CI half-width for a binomial proportion."""
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se


methods = ['Historical\n(ordered)', 'i.i.d. Normal', 'i.i.d. Bootstrap',
           'Block Bootstrap\n(5yr)', 'Block Bootstrap\n(10yr)', 'Block Bootstrap\n(20yr)']
rates = [
    backtest['survived'].mean(),
    iid_results['survived'].mean(),
    bootstrap_results['survived'].mean(),
    block_results[5]['survived'].mean(),
    block_results[10]['survived'].mean(),
    block_results[20]['survived'].mean(),
]
ns = [len(backtest), N_SIMULATIONS, N_SIMULATIONS, N_SIMULATIONS, N_SIMULATIONS, N_SIMULATIONS]
cis = [binomial_ci95(r, n) for r, n in zip(rates, ns)]

# Consistent grouping: Historical = gray, i.i.d. methods = orange, Block Bootstrap = green (shaded by block size)
colors = ['#333333', '#ff7f0e', '#ff7f0e', '#a8d5a2', '#5cb85c', '#2ca02c']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, [r * 100 for r in rates], yerr=[ci * 100 for ci in cis],
              color=colors, capsize=5, ecolor='black')
ax.set_ylabel('30-Year Portfolio Survival Rate (%)')
ax.set_title('Survival Rates Under the 4% Withdrawal Rule\n(100% Equity, 30-Year Retirement, Real Returns)')
ax.axhline(y=backtest['survived'].mean() * 100, color='black', linestyle='--', alpha=0.5,
           label='Historical ground truth')
for bar, rate, ci in zip(bars, rates, cis):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci * 100 + 0.8,
            f'{rate:.1%}', ha='center', fontweight='bold', fontsize=9)
ax.set_ylim(0, 90)
ax.legend()
ax.set_xlabel(f'Error bars show 95% CI. Historical n={len(backtest)} overlapping windows; '
              f'all simulated methods n={N_SIMULATIONS:,} paths.', fontsize=8, labelpad=10)
plt.tight_layout()
plt.savefig('chart1_success_rates.png', dpi=150)
plt.show()

# =====================================================================
# FIGURE 2 — ECDF instead of histogram (fixes historical-vs-10,000-sims
# sample size mismatch), legend outside plot, median markers
# =====================================================================
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


hist_fail_years = backtest[~backtest['survived']]['ruin_year'].values
iid_fail_years = iid_results[~iid_results['survived']]['ruin_year'].values

fig, ax = plt.subplots(figsize=(10, 6))

x_hist, y_hist = ecdf(hist_fail_years)
x_iid, y_iid = ecdf(iid_fail_years)

ax.step(x_hist, y_hist, where='post', color='#333333', linewidth=2.5,
        label=f'Historical (ordered), n={len(hist_fail_years)}')
ax.step(x_iid, y_iid, where='post', color='#d62728', linewidth=1.5, linestyle='--',
        label=f'i.i.d. Normal, n={len(iid_fail_years)}')

hist_median = np.median(hist_fail_years)
iid_median = np.median(iid_fail_years)
ax.axvline(hist_median, color='#333333', linestyle=':', alpha=0.6)
ax.axvline(iid_median, color='#d62728', linestyle=':', alpha=0.6)
ax.text(hist_median, 0.02, f'  median={hist_median:.0f}', color='#333333', fontsize=8)
ax.text(iid_median, 0.10, f'  median={iid_median:.0f}', color='#d62728', fontsize=8)

ax.set_xlabel('Year of Ruin (within 30-year retirement)')
ax.set_ylabel('Cumulative Proportion of Failures')
ax.set_title('Distribution of Portfolio Failure Timing\n(ECDF — comparable despite very different sample sizes)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig('chart2_ruin_timing.png', dpi=150, bbox_inches='tight')
plt.show()

# Note: with only ~n historical failures, small-sample noise still applies to
# the historical ECDF's shape — this is a display fix, not a fix for the
# underlying small-sample limitation, which is still called out in the paper.

# =====================================================================
# FIGURE 3 — Historical cohort outcomes: plain-language labels kept
# (per project's plain-language standard), correct annotations for
# THIS dataset's actual failure clusters (1899-1917, 1956-1976) —
# NOT dot-com/GFC, which fall outside the 1871-2012 usable window
# for a 30-year cohort start.
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 5))
colors_bar = ['#d62728' if not s else '#2ca02c' for s in backtest['survived']]
ax.bar(backtest['start_year'], [1] * len(backtest), color=colors_bar, width=1.0)
ax.set_yticks([])
ax.set_xlabel('Retirement Start Year')
ax.set_title('Which 30-Year Retirement Cohorts Survived? (Green = Succeeded, Red = Failed)')

overall_rate = backtest['survived'].mean()
ax.text(0.01, 0.92, f'Overall historical success rate: {overall_rate:.1%}',
        transform=ax.transAxes, fontsize=10, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

ax.annotate('Panic of 1907 /\nearly-1900s banking panics',
            xy=(1905, 1), xytext=(1905, 1.35),
            ha='center', fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('1970s stagflation',
            xy=(1966, 1), xytext=(1966, 1.35),
            ha='center', fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_ylim(0, 1.6)

plt.tight_layout()
plt.savefig('chart3_failure_timeline.png', dpi=150)
plt.show()

# =====================================================================
# SUMMARY STATISTICS TABLE
# (median ending wealth is conditional on survival — failed paths end
# at or below zero by definition, so blending them in would distort
# the "how much is left" question rather than answer it)
# =====================================================================
def summarize(name, res_df, n):
    survived = res_df[res_df['survived']]
    failed = res_df[~res_df['survived']]
    return {
        'Method': name,
        'N': n,
        'Success Rate': res_df['survived'].mean(),
        'Failure Rate': 1 - res_df['survived'].mean(),
        'Median Ending Wealth (survivors only)': survived['final_balance'].median() if len(survived) else np.nan,
        'Average Failure Year (failures only)': failed['ruin_year'].mean() if len(failed) else np.nan,
    }


summary_rows = [
    summarize('Historical rolling-window', backtest, len(backtest)),
    summarize('i.i.d. Normal', iid_results, N_SIMULATIONS),
    summarize('i.i.d. Bootstrap', bootstrap_results, N_SIMULATIONS),
]
for bsize in block_sizes:
    summary_rows.append(summarize(f'Block Bootstrap ({bsize}yr)', block_results[bsize], N_SIMULATIONS))

summary_df = pd.DataFrame(summary_rows)
summary_df['Success Rate'] = summary_df['Success Rate'].map('{:.2%}'.format)
summary_df['Failure Rate'] = summary_df['Failure Rate'].map('{:.2%}'.format)
summary_df['Median Ending Wealth (survivors only)'] = summary_df['Median Ending Wealth (survivors only)'].map(
    '${:,.0f}'.format)
summary_df['Average Failure Year'] = summary_df['Average Failure Year (failures only)'].map(
    lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')

print("\n--- Summary statistics table ---")
print(summary_df.to_string(index=False))
summary_df.to_csv('summary_statistics.csv', index=False)

print("\nAll charts saved to project folder. Summary table saved to summary_statistics.csv")

