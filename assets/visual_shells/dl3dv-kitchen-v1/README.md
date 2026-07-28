# DL3DV kitchen visual shell v1

This directory contains the small, reviewable metadata shipped in Git. The
three finished Gaussian PLY layers are distributed as one versioned GitHub
Release bundle and are installed automatically by BiGym. No source image or
video is included.

Bundle:

- URL:
  `https://github.com/eust-w/bigym/releases/download/visual-shell-dl3dv-kitchen-v1/dl3dv-kitchen-shell-v1.tar.gz`
- bytes: `226124687`
- SHA-256:
  `bc72810aca750b3392e416ba696b81258218bbbcd54886cf1375da22e9b0dc26`

Profiles:

- `dl3dv-kitchen-mujoco`: clearest profile for cameras kept inside the captured
  path.
- `dl3dv-kitchen-bigym-h1`: alignment for the initial H1 head and wrist cameras.
  This profile is useful for integration testing, but its camera extrapolation
  is visibly softer and is marked diagnostic rather than collection-ready.

The shell is visual only. It is never added to MJCF and contributes zero MuJoCo
bodies, geoms, collisions, or contacts.

See [`doc/visual-shell.md`](../../../doc/visual-shell.md) for installation,
one-click loading, verification, custom MJCF alignment, and troubleshooting.
