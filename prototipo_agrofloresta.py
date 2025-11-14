# -*- coding: utf-8 -*-
"""
Protótipo simples de desenho e avaliação de agrofloresta
Terreno 33m x 30m com leve descaída na porção leste.
Ajuste os parâmetros nas seções indicadas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WIDTH_M = 33
HEIGHT_M = 30
CELL = 1.0

xs = np.arange(0, WIDTH_M, CELL)
ys = np.arange(0, HEIGHT_M, CELL)
X, Y = np.meshgrid(xs, ys)

# ---- Terreno ----
slope_start = 18.0
slope_grad = 0.02
Z = np.zeros_like(X, dtype=float)
sloped = X >= slope_start
Z[sloped] = -(X[sloped] - slope_start) * slope_grad
rng = np.random.default_rng(42)
Z += rng.normal(0, 0.01, size=Z.shape)

# ---- Espécies ----
species_catalog = {
    'banana': {'spacing': 3.0, 'crown_radius_mature': 2.0, 'height': 4.0,
               'shade_pref': (0.2, 0.6), 'water_pref': (0.5, 1.0), 'growth_rate': 'fast', 'n_fix': False, 'layer': 'pioneer'},
    'gliricidia': {'spacing': 6.0, 'crown_radius_mature': 3.0, 'height': 12.0,
                   'shade_pref': (0.0, 0.5), 'water_pref': (0.3, 0.8), 'growth_rate': 'fast', 'n_fix': True, 'layer': 'pioneer'},
    'inga': {'spacing': 8.0, 'crown_radius_mature': 3.5, 'height': 15.0,
             'shade_pref': (0.0, 0.6), 'water_pref': (0.4, 1.0), 'growth_rate': 'fast', 'n_fix': True, 'layer': 'pioneer'},
    'coffee': {'spacing': 2.0, 'crown_radius_mature': 1.2, 'height': 2.5,
               'shade_pref': (0.4, 0.8), 'water_pref': (0.4, 0.8), 'growth_rate': 'medium', 'n_fix': False, 'layer': 'understory'},
    'cassava': {'spacing': 1.5, 'crown_radius_mature': 1.0, 'height': 2.0,
                'shade_pref': (0.0, 0.5), 'water_pref': (0.3, 0.7), 'growth_rate': 'fast', 'n_fix': False, 'layer': 'shrub'},
    'pineapple': {'spacing': 1.0, 'crown_radius_mature': 0.5, 'height': 0.8,
                  'shade_pref': (0.1, 0.6), 'water_pref': (0.4, 0.9), 'growth_rate': 'medium', 'n_fix': False, 'layer': 'ground'},
    'arachis': {'spacing': 0.5, 'crown_radius_mature': 0.3, 'height': 0.1,
                'shade_pref': (0.2, 0.8), 'water_pref': (0.3, 0.9), 'growth_rate': 'fast', 'n_fix': True, 'layer': 'groundcover'},
    'vetiver_row': {'spacing': 0.5, 'crown_radius_mature': 0.3, 'height': 1.2,
                    'shade_pref': (0.0, 0.7), 'water_pref': (0.3, 0.9), 'growth_rate': 'fast', 'n_fix': False, 'layer': 'hedgerow'}
}

plan = []

def add_plant(species, x, y, age_years=1.0):
    plan.append({'species': species, 'x': float(x), 'y': float(y), 'age': float(age_years)})


def canopy_radius(species, age):
    s = species_catalog[species]
    r_max = s['crown_radius_mature']
    if s['growth_rate'] == 'fast':
        k = 1.4
    elif s['growth_rate'] == 'medium':
        k = 1.0
    else:
        k = 0.7
    return r_max * (1 - np.exp(-k * age / 3.0))


def shade_map_from_plan(plan):
    S = np.zeros_like(X, dtype=float)
    for p in plan:
        if p['species'] == 'vetiver_row':
            continue
        r = canopy_radius(p['species'], p['age'])
        if r <= 0.01:
            continue
        sigma = max(r / 2.0, 0.1)
        S += np.exp(-((X - p['x'])**2 + (Y - p['y'])**2) / (2 * sigma**2))
    S = np.clip(S / S.max() if S.max() > 0 else S, 0, 1)
    return S


def water_index(Z, hedgerow_xs):
    eastness = (X - X.min()) / (X.max() - X.min() + 1e-6)
    base = 0.2 + 0.6 * eastness
    if hedgerow_xs:
        H = np.zeros_like(base)
        for hx in hedgerow_xs:
            H = np.maximum(H, np.exp(-((X - hx)**2) / (2 * (1.0**2))))
        base = np.clip(base + 0.15 * H, 0, 1)
    z_norm = (Z.max() - Z) / (Z.max() - Z.min() + 1e-6)
    return np.clip(0.6 * base + 0.4 * z_norm, 0, 1)


def suitability_for(p, S, W):
    s = species_catalog[p['species']]
    xi = int(np.clip(round(p['x']), 0, X.shape[1]-1))
    yi = int(np.clip(round(p['y']), 0, X.shape[0]-1))
    shade = S[yi, xi]
    water = W[yi, xi]
    (sh_min, sh_max) = s['shade_pref']
    (wa_min, wa_max) = s['water_pref']
    def range_score(val, vmin, vmax):
        if vmin <= val <= vmax:
            return 1.0
        if val < vmin:
            return max(0.0, 1.0 - (vmin - val) / (vmin + 1e-6))
        return max(0.0, 1.0 - (val - vmax) / (1.0 - vmax + 1e-6))
    s_sh = range_score(shade, sh_min, sh_max)
    s_wa = range_score(water, wa_min, wa_max)
    spacing = s['spacing']
    penalty = 0.0
    for q in plan:
        if q is p: continue
        if species_catalog[q['species']]['layer'] == s['layer']:
            d = np.hypot(p['x'] - q['x'], p['y'] - q['y'])
            if d < spacing:
                penalty += (spacing - d) / spacing * 0.2
    penalty = np.clip(penalty, 0, 0.6)
    score = np.clip(0.5 * s_sh + 0.5 * s_wa - penalty, 0, 1)
    return score, shade, water

# ---- Layout padrão (ajuste livre) ----
hedgerow_positions_x = [21.0, 27.0]
for hx in hedgerow_positions_x:
    for y in np.arange(1.0, HEIGHT_M-1.0, 1.0):
        add_plant('vetiver_row', hx, y, age_years=1.0)

for x in [4.0, 12.0]:
    for y in np.arange(3.0, HEIGHT_M-3.0, 6.0):
        add_plant('gliricidia', x, y, age_years=2.0)

for x in np.arange(1.5, 16.0, 2.0):
    for y in np.arange(1.5, HEIGHT_M-0.5, 2.0):
        ok = True
        for p in plan:
            if p['species'] == 'gliricidia' and np.hypot(x - p['x'], y - p['y']) < 2.0:
                ok = False; break
        if ok:
            add_plant('cassava', x, y, age_years=1.0)

for x in np.arange(17.0, 21.5, 2.0):
    for y in np.arange(2.0, HEIGHT_M-1.0, 2.0):
        add_plant('coffee', x, y, age_years=1.5)

for x in [18.0, 20.0]:
    for y in np.arange(4.0, HEIGHT_M-2.0, 6.0):
        add_plant('banana', x, y, age_years=1.0)

for x in np.arange(22.5, WIDTH_M-0.5, 3.0):
    for y in np.arange(2.0, HEIGHT_M-1.0, 3.0):
        add_plant('banana', x, y, age_years=1.0)
        px, py = x + 0.8, y
        if px < WIDTH_M-0.5:
            add_plant('pineapple', px, py, age_years=0.5)

# ---- Cálculo ----
S = shade_map_from_plan([p for p in plan if p['species'] != 'vetiver_row'])
W = water_index(Z, hedgerow_positions_x)

rows = []
for p in plan:
    if p['species'] == 'vetiver_row':
        s, sh, wa = 1.0, S[int(round(p['y'])), int(round(p['x']))], W[int(round(p['y'])), int(round(p['x']))]
    else:
        s, sh, wa = suitability_for(p, S, W)
    rows.append({'species': p['species'], 'x': p['x'], 'y': p['y'], 'age_years': p['age'],
                 'suitability_score_0_1': s, 'local_shade_0_1': sh, 'local_water_0_1': wa})

df = pd.DataFrame(rows)

fig, axes = plt.subplots(2,2, figsize=(14,10), constrained_layout=True)
ax = axes[0,0]
im0 = ax.imshow(Z, origin='lower', extent=[0, WIDTH_M, 0, HEIGHT_M], cmap='terrain')
ax.set_title('Modelo de terreno (cotas relativas)')
for hx in hedgerow_positions_x:
    ax.plot([hx, hx], [0, HEIGHT_M], color='olive', linestyle='--', linewidth=2, label='Cerca viva / Vetiver')
ax.legend(loc='lower left')
fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04, label='Elevação relativa')

ax = axes[0,1]
im1 = ax.imshow(W, origin='lower', extent=[0, WIDTH_M, 0, HEIGHT_M], cmap='Blues')
ax.set_title('Índice de água (0..1)')
fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1,0]
im2 = ax.imshow(S, origin='lower', extent=[0, WIDTH_M, 0, HEIGHT_M], cmap='Greens')
ax.set_title('Índice de sombreamento (0..1)')
fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1,1]
ax.imshow(Z*0, origin='lower', extent=[0, WIDTH_M, 0, HEIGHT_M], alpha=0)
colors = {'banana': '#f1c232','gliricidia': '#6aa84f','inga': '#38761d','coffee': '#a64d79',
          'cassava': '#e69138','pineapple': '#ffd966','vetiver_row': '#274e13'}
for p in plan:
    s = df.loc[(df['x']==p['x']) & (df['y']==p['y']) & (df['species']==p['species']), 'suitability_score_0_1'].values[0]
    c = colors.get(p['species'], 'gray')
    size = {'gliricidia':55, 'inga':55, 'banana':45, 'coffee':25, 'cassava':18, 'pineapple':12, 'vetiver_row':5}.get(p['species'],20)
    ec = (1 - s, s, 0.0)
    ax.scatter(p['x'], p['y'], s=size, c=c, edgecolors=[ec], linewidths=1.0, alpha=0.9)
ax.set_title('Layout de plantas (borda = adequação)')
ax.set_xlim(0, WIDTH_M); ax.set_ylim(0, HEIGHT_M); ax.set_aspect('equal'); ax.grid(True, color='lightgray', linestyle=':', linewidth=0.5)

fig.savefig('plano_agrofloresta_prototipo.png', dpi=180)
df.to_csv('plano_agrofloresta_prototipo.csv', index=False)

print('Arquivos salvos: plano_agrofloresta_prototipo.png e plano_agrofloresta_prototipo.csv')
