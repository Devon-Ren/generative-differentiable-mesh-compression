"""make_fig_appendix_v2.py (2026-11-20): the seven figures of the Appendix with one style.  Data and drawing are those of the scripts named below;
only the typeface, the font sizes, the colours of axes and background, and the output names change.
  typeface DejaVu Sans (as the figures of the paper) | axis labels, tick labels, legends, column headers 7.5 pt | panel titles 8 pt |
  text inside a plot (bar percentages, corner labels, 'better') 6.5 pt | background #fcfcfb, axes #dddcd6, labels #52514e (as the paper)
  fig_rd_all_v4 (make_fig_rd_all_v3)            fig_triangles_v7 (make_fig_wire_v6)          fig_qual_error_v3 (make_fig_qual_err_v2)
  fig_cap_instability_v2 (make_cap_fig_single_v1)   fig_streams_v2 (make_fig_streams_v1)     fig_codebook_sweep_v3 (make_fig_codebook_sweep_v1)
  fig_controller_traces_v3 (make_fig_traces_v1)
Run from the repository root."""
STYLE = "plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': '#dddcd6', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})"
def run(path, reps):
    s = open(path).read()
    for a, b, n in reps:
        if s.count(a) != n: raise SystemExit('%s: %r found %d times, expected %d' % (path, a, s.count(a), n))
        s = s.replace(a, b)
    print('==', path); exec(compile(s, path, 'exec'), {'__name__': '__main__'})
run('paper/figures/make_fig_rd_all_v3.py', [('fontsize=5.6', 'fontsize=6.5', 1), ('fontsize=7.2', 'fontsize=7.5', 1), ("'paper/figures/fig_rd_all_v3", "'paper/figures/fig_rd_all_v4", 2)])
run('paper/figures/make_fig_wire_v6.py', [("plt.rcParams.update({'font.size': 7,", "plt.rcParams.update({'font.size': 7.5,", 1), ('fontsize=5.6', 'fontsize=6.5', 1), ("va='bottom', fontsize=7, fontweight", "va='bottom', fontsize=7.5, fontweight", 1),
                                          ('labelsize=6.5', 'labelsize=7.5', 2), ("va='bottom', fontsize=6.5, color='#1f2430'", "va='bottom', fontsize=7.5, color='#1f2430'", 2), ("'paper/figures/fig_triangles_v6", "'paper/figures/fig_triangles_v7", 2)])
run('paper/figures/make_fig_qual_err_v2.py', [('import matplotlib.pyplot as plt\n', "import matplotlib.pyplot as plt\nplt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'text.color': '#0b0b0b'})\n", 1), ('fontsize=6,', 'fontsize=6.5,', 1),
                                              ("va='bottom', fontsize=7,", "va='bottom', fontsize=7.5,", 1), ("'paper/figures/fig_qual_error_v2", "'paper/figures/fig_qual_error_v3", 2)])
run('paper/figures/make_cap_fig_single_v1.py', [('fontsize=6.0', 'fontsize=6.5', 1), ('fontsize=6.3, color=MUTED', 'fontsize=6.5, color=MUTED', 1), ('frameon=False, fontsize=6.3', 'frameon=False, fontsize=7.5', 1),
                                                ("'paper/figures/fig_cap_instability.", "'paper/figures/fig_cap_instability_v2.", 2)])
run('paper/figures/make_fig_streams_v1.py', [("plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})", STYLE, 1),
                                             ("INK, MUTED, GRID = '#1f2430', '#5b6270', '#e3e6ea'", "INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dddcd6'", 1), ("ax.set_ylabel('file size (MB)')", "ax.set_ylabel('file size (MB)', color=MUTED)", 0),
                                             ('fontsize=5.6', 'fontsize=6.5', 1), ('if x > 0.05 * max(tot)', 'if x > 0.075 * max(tot)', 1), ("ncol=5, frameon=False, fontsize=6, handlelength=1.0, columnspacing=0.9", "ncol=3, frameon=False, fontsize=7.5, handlelength=1.0, columnspacing=1.2, labelspacing=0.3", 1),
                                             ('figsize=(3.4, 2.5)', 'figsize=(3.4, 2.8)', 1), ('rect=(0, 0, 1, 0.93)', 'rect=(0, 0, 1, 0.87)', 1), ("edgecolor='white'", "edgecolor='#fcfcfb'", 1), ("'paper/figures/fig_streams_v1", "'paper/figures/fig_streams_v2", 2)])
run('paper/figures/make_fig_codebook_sweep_v1.py', [('fontsize=6.2', 'fontsize=7.5', 1), ('title_fontsize=6.5', 'title_fontsize=7.5', 1), ("title='$K + K_r$'", "title='$K + L$'", 1), ('figsize=(3.35, 2.6)', 'figsize=(3.35, 3.0)', 1),
                                                    ("'paper/figures/fig_codebook_sweep_v1", "'paper/figures/fig_codebook_sweep_v3", 2)])
run('paper/figures/make_fig_traces_v1.py', [("plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})", STYLE, 1),
                                            ("INK, MUTED, GRID, BLUE, ORANGE = '#1f2430', '#5b6270', '#e3e6ea'", "INK, MUTED, GRID, BLUE, ORANGE = '#0b0b0b', '#52514e', '#dddcd6'", 1), ('fontsize=6.5', 'fontsize=7.5', 1),
                                            ("'paper/figures/fig_controller_traces_v2", "'paper/figures/fig_controller_traces_v3", 2)])
