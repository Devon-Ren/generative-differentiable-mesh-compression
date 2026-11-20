"""make_fig_appendix_v1.py (2026-11-20): the appendix figures with the names used in the paper.
  fig_rd_db_obj_v2      = fig_rd_db_obj_v1 (make_figures_v13.py), legend 'MeshSplatting-cap + BM-Codec'
  fig_rd_ssim_lpips_v3  = fig_rd_ssim_lpips_v2 (make_figures_v11.py), same legend
  fig_codebook_sweep_v2 = fig_codebook_sweep_v1, legend title K + L
Data and drawing are unchanged; only labels and output names differ.  Run from the repository root."""
def run(path, reps):
    s = open(path).read()
    for a, b in reps:
        if s.count(a) < 1: raise SystemExit('%s: %r not found' % (path, a))
        s = s.replace(a, b)
    exec(compile(s, path, 'exec'), {'__name__': '__main__'})
run('paper/figures/make_figures_v13.py', [("'MeshSplatting with cap + BM-Codec'", "'MeshSplatting-cap + BM-Codec'"), ('fig_rd_db_obj_v1', 'fig_rd_db_obj_v2')])
run('paper/figures/make_figures_v11.py', [("'vertex cap + our codec'", "'MeshSplatting-cap + BM-Codec'"), ("'vertex cap + TFAN'", "'+ MPEG TFAN'"), ("'vertex cap + zstd'", "'+ zstd'"), ("'vertex cap + Draco'", "'+ Draco'"),
                                          ('fig_rd_ssim_lpips_v2', 'fig_rd_ssim_lpips_v3')])
run('paper/figures/make_fig_codebook_sweep_v1.py', [("title='$K + K_r$'", "title='$K + L$'"), ('fig_codebook_sweep_v1', 'fig_codebook_sweep_v2')])
