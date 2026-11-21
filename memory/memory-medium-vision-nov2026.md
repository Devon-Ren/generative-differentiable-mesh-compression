---
name: memory-medium-vision-nov2026
description: 11/21 new-project north star (user): 3D scene as a memory medium replacing photos; one principle = scene is the boundary of the free space the camera is in (closed by construction, inside/outside unified, infinity via homogeneous inverse-depth); grows from a sphere; doc plan/memory-medium-2026-11-21.md
metadata:
  type: project
---

User 11/21: purpose = groundbreaking "3D memory" medium (replace 2D photos). Explicitly rejects patchworking others' representations (the old +++ stack: manifold + per-face texture + cards + displacement is NOT the direction). Start from a closed bounded shell (sphere); camera inside = indoor/outdoor, camera outside = object.

Formalised in plan/memory-medium-2026-11-21.md: requirements R1-R9 (casual capture, <=10 s first view, photo-sized, glTF/phone 60 fps, never broken, faithful near capture, progressive, walkable with collision, objects detachable). Principle: scene = boundary of the camera's free space Omega (normals into Omega) -> closed, no holes/floaters, collision = stay in Omega; object detach = pinch-off topology event. Infinity: vertex = (direction, inverse distance rho), rho=0 at infinity, GPUs rasterize homogeneous coords. Inside/outside from camera optical-axis convergence. Growth: stage 1 radial (360 depth panorama, works from 1 photo, texture = the photo), stage 2 unfold via visibility (space-carving signal), stage 3 rare topology surgery events. ByteMesh dD/dR law kept, dD weighted by viewing likelihood.

Found 11/21: P2/P2b "opaque surface -2..-5 dB" used 1 spp nvdiffrast without dr.antialias vs MeshSplatting 4x supersampling (render.py scaling=4) -> confounded, rerun E0 (~0.35 GPU-h) before trusting. Related 2026 work: MODL 2611.25414 (watertight from convex hull, objects, fixed genus), Meshtryoshka 2606.28622 (nested SDF shells, 4 h), DiffSoup 2603.27151 (15k textured triangles 24.76 Mip-360), TriSplat 2605.26115 (feed-forward soup).
Open decisions (doc section 7): quality metric, v1 input (1 photo vs video), scope (no fur/dynamics), first step E0 vs code.
