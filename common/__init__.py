"""Shared code for ARGUS: config schemas, constants, small utilities.

Imported by both the perception workers (L0/L1) and the scene reasoner (L2/L3),
so it must stay dependency-light (no torch, no cv2 at import time).
"""
