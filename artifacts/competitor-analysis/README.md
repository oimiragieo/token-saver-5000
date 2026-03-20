# Competitor analysis references

The directories in this folder are pinned research references used for comparison work. They are
stored as Git submodules so the main repository does not vendor entire external codebases.

After cloning this repository, initialize them with:

```bash
git submodule update --init --recursive
```

These references are optional for normal runtime use. They support audits, benchmarking notes, and
product research under `docs/research/`.
