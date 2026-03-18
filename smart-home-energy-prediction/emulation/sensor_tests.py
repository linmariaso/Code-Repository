import matplotlib.pyplot as plt
# Summary Statistics
def test_summary_stats(df):
  """Print summary statistics and check against expectations."""
  print("=" * 60)
  print("TEST 1: Summary Statistics")
  print("=" * 60)

  print(f"\nDataset shape: {df.shape}")
  print(f"Date range: {df.index[0]} to {df.index[-1]}")
  print(f"Total records: {len(df)}")
  expected = days_to_generate * 24 * (60 // sample_freq)
  completeness = len(df) / expected * 100
  print(f"Completeness: {completeness:.1f}% "
        f"(expected {expected}, got {len(df)})")

  print("\n--- Temperature ranges (°C) ---")
  for room in rooms:
      col = f'temperature_{room}'
      print(f"  {room:15s}: "
            f"min={df[col].min():.1f}, "
            f"max={df[col].max():.1f}, "
            f"mean={df[col].mean():.1f}, "
            f"std={df[col].std():.1f}")

  print("\n--- Humidity ranges (%) ---")
  for room in rooms:
      col = f'humidity_{room}'
      print(f"  {room:15s}: "
            f"min={df[col].min():.1f}, "
            f"max={df[col].max():.1f}, "
            f"mean={df[col].mean():.1f}")

  print("\n--- Occupancy rates ---")
  for room in rooms:
      col = f'occupancy_{room}'
      rate = df[col].mean() * 100
      print(f"  {room:15s}: {rate:.1f}% occupied")

  print("\n--- Appliance power (W) ---")
  for appliance in appliances:
      col = f'power_{appliance}'
      print(f"  {appliance:20s}: "
            f"mean={df[col].mean():.1f}W, "
            f"max={df[col].max():.1f}W, "
            f"zeros={( df[col] < 1).sum() / len(df) * 100:.1f}%")

  print(f"\n--- Total power ---")
  print(f"  mean={df['total_power'].mean():.1f}W, "
        f"max={df['total_power'].max():.1f}W, "
        f"min={df['total_power'].min():.1f}W")

  return completeness

# Time Patterns
def test_time_patterns(df):
    """Plot 24-hour average profiles for each sensor type."""
    print("\n" + "=" * 60)
    print("TEST 2: Temporal Patterns (saving plots...)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('24-Hour Average Profiles', fontsize=14)

    # Temperature by room
    ax = axes[0, 0]
    for room in rooms:
        hourly = df.groupby('hour')[f'temperature_{room}'].mean()
        ax.plot(hourly.index, hourly.values, label=room, marker='o',
                markersize=3)
    ax.set_title('Temperature by Room')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('°C')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Humidity by room
    ax = axes[0, 1]
    for room in rooms:
        hourly = df.groupby('hour')[f'humidity_{room}'].mean()
        ax.plot(hourly.index, hourly.values, label=room, marker='o',
                markersize=3)
    ax.set_title('Humidity by Room')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('%')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Occupancy by room
    ax = axes[1, 0]
    for room in rooms:
        hourly = df.groupby('hour')[f'occupancy_{room}'].mean()
        ax.plot(hourly.index, hourly.values, label=room, marker='o',
                markersize=3)
    ax.set_title('Occupancy Probability by Room')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Proportion Occupied')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Total power
    ax = axes[1, 1]
    hourly_power = df.groupby('hour')['total_power'].mean()
    ax.bar(hourly_power.index, hourly_power.values, color='steelblue',
           alpha=0.7)
    ax.set_title('Average Total Power by Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Watts')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'temporal_profiles.png', dpi=150)
    plt.close()
    print("  Saved: temporal_profiles.png")

# Comparison Weekdays and Weekends
def test_weekday_weekend(df):
    """Compare weekday and weekend patterns."""
    print("\n" + "=" * 60)
    print("TEST 3: Weekday vs Weekend (saving plots...)")
    print("=" * 60)

    weekday = df[df['is_weekend'] == 0]
    weekend = df[df['is_weekend'] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Weekday vs Weekend Comparison', fontsize=13)

    # Occupancy count
    ax = axes[0]
    wd_occ = weekday.groupby('hour')['occupancy_count'].mean()
    we_occ = weekend.groupby('hour')['occupancy_count'].mean()
    ax.plot(wd_occ.index, wd_occ.values, label='Weekday', marker='o',
            markersize=3)
    ax.plot(we_occ.index, we_occ.values, label='Weekend', marker='s',
            markersize=3)
    ax.set_title('Occupancy Count')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Rooms Occupied')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total power
    ax = axes[1]
    wd_pow = weekday.groupby('hour')['total_power'].mean()
    we_pow = weekend.groupby('hour')['total_power'].mean()
    ax.plot(wd_pow.index, wd_pow.values, label='Weekday', marker='o',
            markersize=3)
    ax.plot(we_pow.index, we_pow.values, label='Weekend', marker='s',
            markersize=3)
    ax.set_title('Total Power')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Watts')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature (mean)
    ax = axes[2]
    wd_tmp = weekday.groupby('hour')['mean_temperature'].mean()
    we_tmp = weekend.groupby('hour')['mean_temperature'].mean()
    ax.plot(wd_tmp.index, wd_tmp.values, label='Weekday', marker='o',
            markersize=3)
    ax.plot(we_tmp.index, we_tmp.values, label='Weekend', marker='s',
            markersize=3)
    ax.set_title('Mean Temperature')
    ax.set_xlabel('Hour')
    ax.set_ylabel('°C')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'weekday_weekend.png', dpi=150)
    plt.close()
    print("  Saved: weekday_weekend.png")

# Correlation

def test_correlations(df):
    """Generate correlation heatmap for engineered features."""
    print("\n" + "=" * 60)
    print("TEST 5: Feature Correlations (saving plot...)")
    print("=" * 60)

    feature_cols = [
        'hour', 'day_of_week', 'is_weekend',
        'mean_temperature', 'mean_humidity',
        'occupancy_count', 'active_appliances', 'total_power'
    ]
    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_cols, fontsize=8)

    # Add correlation values
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5
                    else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'correlation_heatmap.png', dpi=150)
    plt.close()
    print("  Saved: correlation_heatmap.png")

    # Flag strong correlations with target
    print("\n  Correlations with total_power:")
    for col in feature_cols[:-1]:
        r = corr_matrix.loc[col, 'total_power']
        strength = "strong" if abs(r) > 0.5 else \
                   "moderate" if abs(r) > 0.3 else "weak"
        print(f"    {col:25s}: r={r:.3f} ({strength})")

# Time series

def test_full_timeseries(df):
    """Plot the full time series for key variables."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Time Series (saving plot...)")
    print("=" * 60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('2-Years Emulated Smart Home Data', fontsize=14)

    axes[0].plot(df.index, df['mean_temperature'], linewidth=0.5,
                 color='orangered')
    axes[0].set_ylabel('Mean Temp (°C)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, df['mean_humidity'], linewidth=0.5,
                 color='steelblue')
    axes[1].set_ylabel('Mean Humidity (%)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df.index, df['occupancy_count'], linewidth=0.5,
                 color='forestgreen')
    axes[2].set_ylabel('Rooms Occupied')
    axes[2].set_ylim(-0.5, 5.5)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df.index, df['total_power'], linewidth=0.5,
                 color='purple')
    axes[3].set_ylabel('Total Power (W)')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)

    # Add weekend shading
    for ax in axes:
        for day_offset in range(days_to_generate):
            day = df.index[0] + timedelta(days=day_offset)
            if day.weekday() >= 5:  # Weekend
                ax.axvspan(day, day + timedelta(days=1),
                           alpha=0.1, color='blue')

    plt.tight_layout()
    plt.savefig(f'full_timeseries.png', dpi=150)
    plt.close()
    print("  Saved: full_timeseries.png")

# Test appliance realism
def test_appliance_realism(df, appliance, expected_config):
  print(f"\n--- {appliance.replace('_', ' ').title()} "
          f"Realism Check ---")
  col = f'power_{appliance}'
  cfg = expected_config
  results = []
  # Max power
  max_p = df[col].max()
  ok = max_p <= cfg['max_watts']
  results.append(ok)
  status = "PASS" if ok else "FAIL"
  print(f"  Max reading: {max_p:.1f}W "
        f"(limit: {cfg['max_watts']}W) [{status}]")
    
  # Mean power
  mean_p = df[col].mean()
  lo, hi = cfg['mean_range']
  ok = lo < mean_p < hi
  results.append(ok)
  status = "PASS" if ok else "WARN"
  print(f"  Mean reading: {mean_p:.1f}W "
        f"(expected: {lo}-{hi}W) [{status}]")
    
  # Activations per day
  threshold = cfg['active_threshold']
  active = (df[col] > threshold).sum()
  days = len(df) / (24 * 12)
  per_day = active / days
  lo, hi = cfg['daily_activations']
  ok = lo < per_day < hi
  results.append(ok)
  status = "PASS" if ok else "WARN"
  print(f"  Activations/day: {per_day:.1f} "
        f"(expected: {lo}-{hi}) [{status}]")
    
  # Unoccupied activations
  unocc = df[df['occupancy_count'] == 0]
  unocc_active = (unocc[col] > threshold).sum()
  unocc_rate = unocc_active / len(unocc) * 100 if len(unocc) > 0 else 0
  ok = unocc_rate < 3
  results.append(ok)
  status = "PASS" if ok else "WARN"
  print(f"  Activations when unoccupied: {unocc_rate:.2f}% "
        f"(expect <3%) [{status}]")
    
  # Peak vs off-peak
  peak = df[df['hour'].isin(cfg['peak_hours'])]
  off = df[~df['hour'].isin(cfg['peak_hours'])]
  peak_mean = peak[col].mean()
  off_mean = off[col].mean() if len(off) > 0 else 0
  ok = peak_mean > off_mean * 1.5
  results.append(ok)
  status = "PASS" if ok else "WARN"
  print(f"  Peak mean: {peak_mean:.1f}W vs "
        f"off-peak: {off_mean:.1f}W [{status}]")
    
  # Zero readings
  zero_pct = (df[col] < 1).sum() / len(df) * 100
  lo, hi = cfg['zero_pct_range']
  ok = lo <= zero_pct <= hi
  results.append(ok)
  status = "PASS" if ok else "WARN"
  print(f"  Zero/standby: {zero_pct:.1f}% "
        f"(expected: {lo}-{hi}%) [{status}]")
    
  passed = sum(results)
  print(f"  Result: {passed}/{len(results)} checks passed")
  return passed, len(results)

appliance_expectations = {
    'kettle': {
        'max_watts': 2500,
        'mean_range': (3, 30),
        'daily_activations': (2, 10),
        'active_threshold': 100,
        'peak_hours': [7, 8, 10, 12, 15, 18],
        'zero_pct_range': (94, 100),
    },
    'microwave': {
        'max_watts': 1500,
        'mean_range': (1, 20),
        'daily_activations': (2, 8),
        'active_threshold': 50,
        'peak_hours': [7, 12, 18, 19],
        'zero_pct_range': (94, 100),
    },
    'washing_machine': {
        'max_watts': 2500,
        'mean_range': (1, 15),
        'daily_activations': (0.5, 5),
        'active_threshold': 100,
        'peak_hours': [9, 10, 11, 14],
        'zero_pct_range': (95, 100),
    },
    'dishwasher': {
        'max_watts': 2500,
        'mean_range': (2, 25),
        'daily_activations': (0.5, 4),
        'active_threshold': 200,
        'peak_hours': [20, 21],
        'zero_pct_range': (95, 100),
    },
    'television': {
        'max_watts': 250,
        'mean_range': (8, 50),
        'daily_activations': (15, 40),
        'active_threshold': 30,
        'peak_hours': [18, 19, 20, 21, 22],
        'zero_pct_range': (0, 5),
        'always_on': False,
    },
    'computer': {
        'max_watts': 500,
        'mean_range': (15, 80),
        'daily_activations': (25, 55),
        'active_threshold': 30,
        'peak_hours': [9, 10, 11, 13, 14, 15, 16],
        'zero_pct_range': (0, 5),
        'always_on': False,
    },
    'refrigerator': {
        'max_watts': 300,
        'mean_range': (50, 100),
        'daily_activations': (60, 120),  # Always on, cycling
        'active_threshold': 80,
        'peak_hours': list(range(24)),    # No peak — always on
        'zero_pct_range': (0, 1),         # Should never be zero
        'always_on': True,
    },
    'lighting': {
        'max_watts': 200,
        'mean_range': (5, 30),
        'daily_activations': (25, 55),
        'active_threshold': 15,
        'peak_hours': [6, 7, 17, 18, 19, 20, 21, 22],
        'zero_pct_range': (70, 92),
    },
}

def test_all_appliances(df):
    """Run realism checks for every appliance."""
    print("\n" + "=" * 60)
    print("APPLIANCE REALISM TESTS")
    print("=" * 60)
    
    total_passed = 0
    total_checks = 0
    
    for appliance, config in appliance_expectations.items():
        passed, checks = test_appliance_realism(df, appliance, config)
        total_passed += passed
        total_checks += checks
    
    print("\n" + "-" * 60)
    print(f"OVERALL: {total_passed}/{total_checks} checks passed")
    print("-" * 60)
    return total_passed, total_checks

