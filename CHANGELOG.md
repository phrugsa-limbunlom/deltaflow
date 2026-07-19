# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Initial scaffold: `core`, `interpolants` (`LinearInterpolant`), `samplers`
  (`FlowSampler`), `losses` (`FlowMatchingLoss`, `DeltaAlignmentLoss`),
  `models` (`MultiScaleProjector`, `EMA`), `datasets` (generic radiograph
  wrappers).
- Tiered examples (`00-foundations`, `10-sampling`, `20-training`,
  `90-showcase`).
- Unit tests for interpolants, samplers, losses, and models.
