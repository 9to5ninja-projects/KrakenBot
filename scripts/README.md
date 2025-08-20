# KrakenBot Scripts

This directory contains helper scripts for running various KrakenBot features.

## Available Scripts

### `advanced_trading_strategy.ps1`

A comprehensive script that runs the full advanced trading strategy:
1. Finds optimal triangular arbitrage paths
2. Performs statistical analysis
3. Starts the monitor with optimal settings

```powershell
.\advanced_trading_strategy.ps1
```

### `optimize_triangles.ps1`

Runs the triangle optimizer to find the most profitable triangular arbitrage paths.

```powershell
.\optimize_triangles.ps1
```

### `run_statistical_analysis.ps1`

Performs statistical analysis on trading pairs to identify optimal entry points and generate trading recommendations.

```powershell
.\run_statistical_analysis.ps1
```

### `run_optimal_monitor.ps1`

Runs the monitor with optimal triangles and percentage-based profit threshold.

```powershell
.\run_optimal_monitor.ps1
```

## Usage

To run any of these scripts, open PowerShell and navigate to the KrakenBot directory, then execute the script:

```powershell
cd e:\KrakenBot
.\scripts\advanced_trading_strategy.ps1
```

Or right-click on the script in Windows Explorer and select "Run with PowerShell".

## Notes

- These scripts are designed for Windows PowerShell
- Make sure you have Python and all required dependencies installed
- The scripts will create necessary directories if they don't exist
- Results will be saved in the appropriate data directories