# Skill Benchmark: token-saver-context

**Configurations**: with_skill vs without_skill
**Date**: 2026-03-18T03:26:13Z
**Evals**: 1, 2, 3 (1 run each per configuration)

## Summary

| Metric | With Skill | Without Skill | Delta |
|--------|------------|---------------|-------|
| Pass Rate | 22% | 11% | +0.11 |
| Time | 45.0s | 27.5s | +17.6s |
| Tokens | 1761 | 894 | +867 |

## Notes

- Runs were executed with tools disabled to isolate skill-trigger guidance from filesystem inspection.
- The skill improved project-specific command accuracy, but it still often fell back to broader repo guidance instead of the exact packaged scripts.
