import matplotlib.pyplot as plt
import os

# Set Computer Modern font (standard LaTeX font)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

out_dir = os.path.abspath('app/static/img')
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# 1. Formula 1: sim(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
# ==============================================================================
fig1 = plt.figure(dpi=300)
f1_str = r'$sim(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$'
t1 = fig1.text(0.5, 0.5, f1_str, color='#ffffff', fontsize=44, ha='center', va='center')

# Compute tight bounding box of the text artist
fig1.canvas.draw()
renderer1 = fig1.canvas.get_renderer()
bbox1 = t1.get_window_extent(renderer=renderer1)
bbox_inches1 = bbox1.transformed(fig1.dpi_scale_trans.inverted()).padded(0.08)

fig1.savefig(os.path.join(out_dir, 'formula_cosine.svg'), bbox_inches=bbox_inches1, transparent=True)
fig1.savefig(os.path.join(out_dir, 'formula_cosine.png'), bbox_inches=bbox_inches1, transparent=True)
plt.close(fig1)

# ==============================================================================
# 2. Formula 2: v_i[u] = r_{ui} - \mu_u
# ==============================================================================
fig2 = plt.figure(dpi=300)
f2_str = r'$v_i[u] = r_{ui} - \mu_u$'
t2 = fig2.text(0.5, 0.5, f2_str, color='#ffffff', fontsize=44, ha='center', va='center')

# Compute tight bounding box of the text artist
fig2.canvas.draw()
renderer2 = fig2.canvas.get_renderer()
bbox2 = t2.get_window_extent(renderer=renderer2)
bbox_inches2 = bbox2.transformed(fig2.dpi_scale_trans.inverted()).padded(0.08)

fig2.savefig(os.path.join(out_dir, 'formula_mean_centering.svg'), bbox_inches=bbox_inches2, transparent=True)
fig2.savefig(os.path.join(out_dir, 'formula_mean_centering.png'), bbox_inches=bbox_inches2, transparent=True)
plt.close(fig2)

print("Successfully regenerated formula_cosine and formula_mean_centering with large font & tight bounding box!")
