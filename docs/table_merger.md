# Improved Table Merger

## Overview

The improved table merger is a module that enhances PDF table extraction by detecting and merging tables that span multiple pages. It provides configurable merge strategies and smart header matching to handle various PDF layouts.

## Key Features

- Multiple Merge Strategies: Choose between conservative, aggressive, or no merging
- Smart Header Detection: Intelligently matches table headers with slight variations
- Safe Merging: Ensures data integrity with validation during merging
- Configurable Thresholds: Adjust similarity requirements for different document types

## Merge Strategies

1. **Conservative** (default):
   - Requires high similarity between tables (80% threshold)
   - Only merges tables with nearly identical headers
   - Best for well-formatted documents

2. **Aggressive**:
   - Uses a lower similarity threshold (60%)
   - More likely to merge tables with variations in headers
   - Useful for documents with inconsistent formatting

3. **None**:
   - Disables table merging completely
   - Keeps all tables separate
   - Useful for debugging or when merging is not desired
