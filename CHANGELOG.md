# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Support for FIT files without power data by automatically setting all power values to 0
- Proper handling of NaN power values in individual records by filling with zeros instead of interpolation

### Changed
- **BREAKING**: Migrated from fitparse to fitdecode library for improved thread safety and performance
- FIT file parsing now preserves records with missing power data instead of dropping them
- Power data gaps are filled with zeros rather than interpolated values to avoid creating artificial power readings
- Improved logging to indicate when power data is missing or being filled with zeros

### Technical Details
- **Migration**: Replaced fitparse with fitdecode for FIT file parsing - provides better thread safety
- **Dependencies**: Updated requirements.txt and pyproject.toml to use fitdecode>=0.10.0 instead of fitparse
- Modified `models/fit_file.py` to detect missing or all-NaN power data and create power column with zeros
- Updated `models/virtual_elevation.py` to gracefully handle missing power data instead of throwing errors
- Enhanced resampling process to fill NaN power values with zeros before and after interpolation
- Restructured FIT file parsing to use fitdecode's frame-based iteration instead of message filtering