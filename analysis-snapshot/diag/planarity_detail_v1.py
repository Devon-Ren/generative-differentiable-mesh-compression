"""planarity_detail_v1.py: print every (eps, theta) row + dihedral histogram from planarity_v1 json files."""
import glob, json, sys
d = sys.argv[1] if len(sys.argv) > 1 else '/scratch/zr2245/analysis/planarity-v1'
for p in sorted(glob.glob(d + '/*.json')):
    r = json.load(open(p)); f = r['facts']; h = f['dihedral_hist']
    print(f"{f['label']:24s} V {f['vertices']:>8d} F {f['faces']:>8d} | dihedral <2/<5/<10/<30 deg: {100*h['<2deg']:.0f}/{100*h['<5deg']:.0f}/{100*h['<10deg']:.0f}/{100*h['<30deg']:.0f} % | median edge {f['median_edge_length_pct']:.3f} % diag")
    for row in r['rows']:
        g, g10 = row['ge100'], row['ge10']
        cf = row['curved_rejected_faces'] / max(f['faces'], 1)
        print(f"    eps {row['eps_pct']:.2f} % theta {row['theta_deg']:>4.0f} | regions verified {row['verified_components']:>6d} curved-rejected {row['curved_rejected_components']:>5d} ({100*cf:.1f} % of faces) "
              f"| >=10 faces: vertices {100*g10['vertex_share']:.1f} % interior {100*g10['interior_vertex_share']:.1f} % | >=100: faces {100*g['face_share']:.1f} % area {100*g['area_share']:.1f} % vertices {100*g['vertex_share']:.1f} % interior {100*g['interior_vertex_share']:.1f} % polygons {g['face_reduction_x']:.1f}x")
