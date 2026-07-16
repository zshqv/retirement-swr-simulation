import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. DATA LOADING + CLEANING
# =========================================================

df = pd.read_excel('chapt26.xlsx', sheet_name='Data', header=None, skiprows=8)

df_clean = df.iloc[:, [0, 1, 2, 5, 6, 15]]
df_clean.columns = ['Year', 'SP_Price', 'Dividend', 'Rate_Long10yr', 'CPI', 'Return_SP']

df_clean = df_clean[(df_clean['Year'] >= 1871) & (df_clean['Year'] <= 2012)]
df_clean = df_clean.reset_index(drop=True)

print(df_clean.shape)      # should print (142, 6)
print(df_clean.isna().sum())

# Real return = (1 + nominal) / (1 + inflation) - 1
df_clean['Inflation'] = df_clean['CPI'].pct_change()
df_clean['Real_Return_SP'] = (1 + df_clean['Return_SP']) / (1 + df_clean['Inflation']) - 1

# Drop first row — no prior-year CPI, so Inflation/Real_Return are NaN
df_clean = df_clean.dropna().reset_index(drop=True)

print(df_clean.shape)   # should now be (141, 8)
print(df_clean[['Year', 'Return_SP', 'Inflation', 'Real_Return_SP']].describe())


# =========================================================
# 2. CORE SIMULATION ENGINE
# =========================================================

def simulate_path(returns, initial_balance=1_000_000, withdrawal_rate=0.04):
    """
    Simulates one retirement path.
    returns: list/array of annual real returns, one per year of retirement
    Returns: (balance_history, ruin_year or None)
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


# =========================================================
# 3. METHOD 1: HISTORICAL ROLLING-WINDOW BACKTEST (ground truth)
# =========================================================

def rolling_window_backtest(df, window_years=30, initial_balance=1_000_000, withdrawal_rate=0.04):
    results = []
    years = df['Year'].tolist()
    returns = df['Real_Return_SP'].tolist()

    for start_idx in range(len(years) - window_years + 1):
        start_year = years[start_idx]
        window_returns = returns[start_idx: start_idx + window_years]

        history, ruin = simulate_path(window_returns, initial_balance, withdrawal_rate)
        results.append({
            'start_year': start_year,
            'ruin_year': ruin,
            'survived': ruin is None,
            'final_balance': history[-1]
        })

    return pd.DataFrame(results)


backtest = rolling_window_backtest(df_clean)
print(f"\nBacktest windows: {backtest.shape}")
print(f"Historical rolling-window success rate: {backtest['survived'].mean():.2%}")
print(f"Failed cohorts:\n{backtest[~backtest['survived']][['start_year', 'ruin_year']]}")


# =========================================================
# 4. METHOD 2: I.I.D. NORMAL SIMULATION
# =========================================================

def simulate_iid_normal(historical_returns, n_years=30, n_simulations=10_000,
                         initial_balance=1_000_000, withdrawal_rate=0.04, seed=42):
    rng = np.random.default_rng(seed)
    mean_r = np.mean(historical_returns)
    std_r = np.std(historical_returns)
    results = []
    for sim in range(n_simulations):
        random_returns = rng.normal(mean_r, std_r, size=n_years)
        history, ruin = simulate_path(random_returns, initial_balance, withdrawal_rate)
        results.append({'sim_id': sim, 'ruin_year': ruin, 'survived': ruin is None, 'final_balance': history[-1]})
    return pd.DataFrame(results)


mean_r = df_clean['Real_Return_SP'].mean()
std_r = df_clean['Real_Return_SP'].std()
iid_results = simulate_iid_normal(df_clean['Real_Return_SP'].values)
print(f"\nMean used: {mean_r:.4f}, Std used: {std_r:.4f}")
print(f"i.i.d. Normal success rate: {iid_results['survived'].mean():.2%}")


# =========================================================
# 5. METHOD 3: I.I.D. HISTORICAL BOOTSTRAP
# =========================================================

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


bootstrap_results = simulate_bootstrap_iid(df_clean['Real_Return_SP'].values)
print(f"i.i.d. historical bootstrap success rate: {bootstrap_results['survived'].mean():.2%}")


# =========================================================
# 6. METHOD 4: BLOCK BOOTSTRAP (block sizes 3, 5, 10, 15, 20)
# =========================================================

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


block_sizes = [3, 5, 10, 15, 20]
block_results = {
    bsize: simulate_block_bootstrap(df_clean['Real_Return_SP'].values, block_size=bsize)
    for bsize in block_sizes
}

print("\n--- Block bootstrap success rates by block size ---")
for bsize in block_sizes:
    print(f"Block size {bsize:2d}: success rate = {block_results[bsize]['survived'].mean():.2%}")


# =========================================================
# 7. HEADLINE SUMMARY (paste this whole block back to Claude)
# =========================================================

print("\n=== HEADLINE NUMBERS ===")
print(f"Backtest windows:                          {backtest.shape[0]}")
print(f"Historical rolling-window success rate:    {backtest['survived'].mean():.2%}")
print(f"i.i.d. Normal success rate:                {iid_results['survived'].mean():.2%}")
print(f"i.i.d. historical bootstrap success rate:  {bootstrap_results['survived'].mean():.2%}")
for bsize in block_sizes:
    print(f"Block bootstrap ({bsize:2d}yr) success rate:      {block_results[bsize]['survived'].mean():.2%}")


# =========================================================
# 8. CHARTS
# =========================================================

# ---- Chart 1: success rate comparison across all methods ----
chart1_methods = ['Historical\n(ordered)', 'i.i.d. Normal', 'i.i.d. Bootstrap'] + \
                  [f'Block\n({b}yr)' for b in block_sizes]
chart1_rates = [backtest['survived'].mean(), iid_results['survived'].mean(), bootstrap_results['survived'].mean()] + \
               [block_results[b]['survived'].mean() for b in block_sizes]
chart1_colors = ['#333333', '#d62728', '#d62728'] + ['#ff7f0e'] * (len(block_sizes) - 1) + ['#2ca02c']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(chart1_methods, [r * 100 for r in chart1_rates], color=chart1_colors)
ax.set_ylabel('30-Year Portfolio Survival Rate (%)')
ax.set_title('4% Rule Survival Rate: Historical Truth vs. Simulated Methods\n(100% Equity, 30-Year Retirement, Real Returns)')
ax.axhline(y=backtest['survived'].mean() * 100, color='black', linestyle='--', alpha=0.5, label='Historical ground truth')
for bar, rate in zip(bars, chart1_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{rate:.1%}', ha='center', fontweight='bold', fontsize=8)
ax.set_ylim(0, 85)
ax.legend()
plt.tight_layout()
plt.savefig('chart1_success_rates.png', dpi=150)
plt.show()

# ---- Chart 2: ruin-timing density histogram ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(backtest[~backtest['survived']]['ruin_year'], bins=15, alpha=0.6, label='Historical (ordered)', color='#333333')
ax.hist(iid_results[~iid_results['survived']]['ruin_year'], bins=15, alpha=0.6, label='i.i.d. Normal', color='#d62728')
ax.set_xlabel('Year of Ruin (within 30-year retirement)')
ax.set_ylabel('Count')
ax.set_title('When Does the Money Run Out? Historical vs. Naive Simulation')
ax.legend()
plt.tight_layout()
plt.savefig('chart2_ruin_timing.png', dpi=150)
plt.show()

# ---- Chart 3: failure-timeline strip ----
fig, ax = plt.subplots(figsize=(14, 4))
colors = ['#d62728' if not s else '#2ca02c' for s in backtest['survived']]
ax.bar(backtest['start_year'], [1] * len(backtest), color=colors, width=1.0)
ax.set_yticks([])
ax.set_xlabel('Retirement Start Year')
ax.set_title('Which 30-Year Retirement Cohorts Survived? (Green = Success, Red = Failure)')
plt.tight_layout()
plt.savefig('chart3_failure_timeline.png', dpi=150)
plt.show()