# Project instructions

Build a reproducible research engine. Keep observed option quotes separate from simulated hedging. All strategies use the same paths, option premium, funding assumptions, and costs. Preserve adverse results and rejected data. No live trading, historical performance, production deployment, or personal authorship claims.

Native Deribit inverse option prices are converted to contemporaneous USD equivalents using the reported index. Calibration uses expiry-specific reported forwards, zero discounting as an explicit convention, and European calls/puts. The hedge experiment is a separate hypothetical USD cash-settled contract on a zero-drift underlying, not an implementation of coin collateral or perpetual funding.

Keep source timestamps and hashes, parameter bounds, numerical tolerances, calibration failures, and source limitations visible. Keep repository content focused on research, implementation, usage and validation. Record Codex assistance accurately.
