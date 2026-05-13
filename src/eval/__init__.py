"""Evaluation pipeline for generated motions.

Public entry point: ``src.eval.run.main`` / ``python -m src.eval.run <dir>``.

The eval is two-stage: ``eval_part.py`` writes generations into an experiment
folder, and this package loads them + computes FID / R-precision / Diversity /
contrastive metrics against TMR encoders, producing both paper-table output
and the detailed CSVs.
"""
