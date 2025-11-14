# -*- coding: utf-8 -*-
"""
Protótipo V2 – Agrofloresta 33m x 30m (entrada incremental)
- Aceita arquivos de entrada opcionais:
  * entrada_elementos_existentes.csv (árvores/estruturas/casa)
  * entrada_lista_desejos_especies.csv (lista de espécies desejadas)
- Se os arquivos não existirem, roda com layout-base e imprime instruções.

Coordenadas: origem no canto SUDOESTE do terreno (x leste, y norte).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WIDTH_M = 33
HEIGHT_M = 30
CELL = 1.0

xs = np.arange(0, WIDTH_M, CELL)
ys = np.arange(0, HEIGHT_M, CELL)
X, Y = np.meshgrid(xs, ys)

# ------------- Terreno / Relevo -------------
slope_start = 18.0  # m a partir do oeste onde inicia a queda para leste
slope_grad = 0.02   # ~2% de declive
rng_seed = 42       # para reprodutibilidade do microrelevo

rng = np.random.default_rng(rng_seed)
Z = np.zeros_like(X, dtype=float)
sloped = X >= slope_start
Z[sloped] = -(X[sloped] - slope_start) * slope_grad
Z += rng.normal(0, 0.01, size=Z.shape)  # micro relevo

# ------------- Catálogo básico (padrões genéricos) -------------
# Observação: estes são padrões heurísticos e devem ser ajustados caso a caso.
# Camadas: canopy/pioneer/understory/shrub/ground/vine/annual/hedgerow
species_catalog = {
    'banana':      {'spacing': 3.0, 'crown_radius_mature': 2.0, 'height': 4.0,  'shade_pref': (0.2,0.6), 'water_pref': (0.5,1.0), 'growth_rate': 'fast',   'n_fix': False, 'layer':'pioneer'},
    'gliricidia':  {'spacing': 6.0, 'crown_radius_mature': 3.0, 'height': 12.0, 'shade_pref': (0.0,0.5), 'water_pref': (0.3,0.8), 'growth_rate': 'fast',   'n_fix': True,  'layer':'pioneer'},
    'inga':        {'spacing': 8.0, 'crown_radius_mature': 3.5, 'height': 15.0, 'shade_pref': (0.0,0.6), 'water_pref': (0.4,1.0), 'growth_rate': 'fast',   'n_fix': True,  'layer':'pioneer'},
    'coffee':      {'spacing': 2.0, 'crown_radius_mature': 1.2, 'height': 2.5,  'shade_pref': (0.4,0.8), 'water_pref': (0.4,0.8), 'growth_rate': 'medium', 'n_fix': False, 'layer':'understory'},
    'cassava':     {'spacing': 1.5, 'crown_radius_mature': 1.0, 'height': 2.0,  'shade_pref': (0.0,0.5), 'water_pref': (0.3,0.7), 'growth_rate': 'fast',   'n_fix': False, 'layer':'shrub'},
    'pineapple':   {'spacing': 1.0, 'crown_radius_mature': 0.5, 'height': 0.8,  'shade_pref': (0.1,0.6), 'water_pref': (0.4,0.9), 'growth_rate': 'medium', 'n_fix': False, 'layer':'ground'},
    'vetiver_row': {'spacing': 0.5, 'crown_radius_mature': 0.3, 'height': 1.2,  'shade_pref': (0.0,0.7), 'water_pref': (0.3,0.9), 'growth_rate': 'fast',   'n_fix': False, 'layer':'hedgerow'}
}

# ------------- Utilidades -------------

def add_plant(plan, species, x, y, age_years=1.0):
    plan.append({'species': species, 'x': float(x), 'y': float(y), 'age': float(age_years)})


def canopy_radius(species, age):
    s = species_catalog.get(species)
    if s is None:
        # fallback genérico
        r_max = 2.0; k = 1.0
    else:
        r_max = s['crown_radius_mature']
        k = {'fast':1.4,'medium':1.0,'slow':0.7}.get(s['growth_rate'],'medium')
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


def suitability_for(p, S, W, plan):
    sp = p['species']
    s = species_catalog.get(sp, {'spacing':2.0,'shade_pref':(0.0,1.0),'water_pref':(0.0,1.0),'layer':'shrub'})
    xi = int(np.clip(round(p['x']), 0, X.shape[1]-1))
    yi = int(np.clip(round(p['y']), 0, X.shape[0]-1))
    shade = S[yi, xi]
    water = W[yi, xi]
    sh_min, sh_max = s['shade_pref']
    wa_min, wa_max = s['water_pref']
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
        if species_catalog.get(q['species'], {'layer':s['layer']})['layer'] == s['layer']:
            d = np.hypot(p['x'] - q['x'], p['y'] - q['y'])
            if d < spacing:
                penalty += (spacing - d) / spacing * 0.2
    penalty = np.clip(penalty, 0, 0.6)
    score = np.clip(0.5 * s_sh + 0.5 * s_wa - penalty, 0, 1)
    return score, shade, water

# ------------- Leitura de entradas (opcional) -------------
features_path = 'entrada_elementos_existentes.csv'
wishlist_path = 'entrada_lista_desejos_especies.csv'

features_df = None
wishlist_df = None

if os.path.exists(features_path):
    try:
        features_df = pd.read_csv(features_path)
    except Exception as e:
        print('[Aviso] Não foi possível ler', features_path, e)

if os.path.exists(wishlist_path):
    try:
        wishlist_df = pd.read_csv(wishlist_path)
    except Exception as e:
        print('[Aviso] Não foi possível ler', wishlist_path, e)

# ------------- Plano inicial -------------
plan = []

# Cercas vivas (vetiver) – fixas (podem ser ajustadas)
hedgerow_positions_x = [21.0, 27.0]
for hx in hedgerow_positions_x:
    for y in np.arange(1.0, HEIGHT_M-1.0, 1.0):
        add_plant(plan, 'vetiver_row', hx, y, age_years=1.0)

# Macro zones base (somente se não houver wishlist para preencher automaticamente)
use_baseline_layout = wishlist_df is None

if use_baseline_layout:
    # Oeste: gliricidia + mandioca
    for x in [4.0, 12.0]:
        for y in np.arange(3.0, HEIGHT_M-3.0, 6.0):
            add_plant(plan, 'gliricidia', x, y, age_years=2.0)
    for x in np.arange(1.5, 16.0, 2.0):
        for y in np.arange(1.5, HEIGHT_M-0.5, 2.0):
            ok = True
            for p in plan:
                if p['species'] == 'gliricidia' and np.hypot(x - p['x'], y - p['y']) < 2.0:
                    ok = False; break
            if ok:
                add_plant(plan, 'cassava', x, y, age_years=1.0)
    # Meio: café + algumas bananas
    for x in np.arange(17.0, 21.5, 2.0):
        for y in np.arange(2.0, HEIGHT_M-1.0, 2.0):
            add_plant(plan, 'coffee', x, y, age_years=1.5)
    for x in [18.0, 20.0]:
        for y in np.arange(4.0, HEIGHT_M-2.0, 6.0):
            add_plant(plan, 'banana', x, y, age_years=1.0)
    # Leste: bananas + abacaxi
    for x in np.arange(22.5, WIDTH_M-0.5, 3.0):
        for y in np.arange(2.0, HEIGHT_M-1.0, 3.0):
            add_plant(plan, 'banana', x, y, age_years=1.0)
            px, py = x + 0.8, y
            if px < WIDTH_M-0.5:
                add_plant(plan, 'pineapple', px, py, age_years=0.5)

# ------------- Aplicação de elementos existentes -------------
# - Árvores adultas: adicionam sombreamento e áreas de exclusão por proximidade (raio opcional)
# - Estruturas e casa: áreas de exclusão (buffer = radius_m)
exclusion_mask = np.zeros_like(X, dtype=bool)
existing_plants = []

if features_df is not None:
    # sanitize columns
    cols = {c.lower():c for c in features_df.columns}
    def get(c):
        return features_df[cols.get(c,c)] if cols.get(c) in features_df.columns else None
    type_col = get('type'); spec_col = get('species'); x_col = get('x_m'); y_col = get('y_m')
    age_col = get('age_years'); rad_col = get('radius_m')

    for i in range(len(features_df)):
        t = type_col.iloc[i] if type_col is not None else ''
        sp = (spec_col.iloc[i] if spec_col is not None else '') or ''
        x0 = float(x_col.iloc[i]) if x_col is not None and not pd.isna(x_col.iloc[i]) else None
        y0 = float(y_col.iloc[i]) if y_col is not None and not pd.isna(y_col.iloc[i]) else None
        if x0 is None or y0 is None:
            continue
        rbuf = float(rad_col.iloc[i]) if rad_col is not None and not pd.isna(rad_col.iloc[i]) else 0.0
        if t == 'tree':
            age = float(age_col.iloc[i]) if age_col is not None and not pd.isna(age_col.iloc[i]) else 8.0
            add_plant(existing_plants, sp.lower(), x0, y0, age_years=age)
            if rbuf > 0:
                exclusion_mask |= ((X - x0)**2 + (Y - y0)**2) <= (rbuf**2)
        elif t in ('structure','house_footprint'):
            if rbuf > 0:
                exclusion_mask |= ((X - x0)**2 + (Y - y0)**2) <= (rbuf**2)

# ------------- Integração do wishlist (se existir) -------------
# Estratégia simples: tenta preencher em zonas candidatas, evitando exclusão e respeitando espaçamento mínimo.

def place_points_greedy(species_key, qty, spacing, candidate_mask):
    placed = []
    # varredura em grid 1m priorizando células com maior água ou menor sombra conforme espécie
    indices = np.argwhere(candidate_mask)
    # shuffle para diversidade
    rng = np.random.default_rng(123)
    rng.shuffle(indices)
    for yi, xi in indices:
        x0 = xs[xi]; y0 = ys[yi]
        ok = True
        for p in plan + placed + existing_plants:
            d = np.hypot(x0 - p['x'], y0 - p['y'])
            if d < spacing:
                ok = False; break
        if ok:
            placed.append({'species': species_key, 'x': float(x0), 'y': float(y0), 'age': 1.0})
            if len(placed) >= qty:
                break
    return placed

if wishlist_df is not None:
    # construir mapas primeiro
    # adicionar plantas existentes no cálculo de sombra
    temp_plan_for_shade = plan + existing_plants
    S = shade_map_from_plan([p for p in temp_plan_for_shade if p['species'] != 'vetiver_row'])
    W = water_index(Z, hedgerow_positions_x)

    # candidato geral: fora das exclusões
    base_candidates = ~exclusion_mask

    # Para cada linha do wishlist, tentar posicionar
    cols = {c.lower():c for c in wishlist_df.columns}
    def getc(c):
        return wishlist_df[cols.get(c,c)] if cols.get(c) in wishlist_df.columns else None

    sp_col = getc('species_common')
    qty_col = getc('qty_target')
    spc_col = getc('min_spacing_m')
    shmin_col = getc('shade_pref_0_1_min'); shmax_col = getc('shade_pref_0_1_max')
    wmin_col = getc('water_pref_0_1_min'); wmax_col = getc('water_pref_0_1_max')

    for i in range(len(wishlist_df)):
        name = str(sp_col.iloc[i]).strip().lower() if sp_col is not None else ''
        if not name or name == 'nan':
            continue
        qty = int(qty_col.iloc[i]) if qty_col is not None and not pd.isna(qty_col.iloc[i]) else 0
        spacing = float(spc_col.iloc[i]) if spc_col is not None and not pd.isna(spc_col.iloc[i]) else 2.0
        sh_min = float(shmin_col.iloc[i]) if shmin_col is not None and not pd.isna(shmin_col.iloc[i]) else 0.0
        sh_max = float(shmax_col.iloc[i]) if shmax_col is not None and not pd.isna(shmax_col.iloc[i]) else 1.0
        wa_min = float(wmin_col.iloc[i]) if wmin_col is not None and not pd.isna(wmin_col.iloc[i]) else 0.0
        wa_max = float(wmax_col.iloc[i]) if wmax_col is not None and not pd.isna(wmax_col.iloc[i]) else 1.0

        # criar entrada no catálogo se não existir
        if name not in species_catalog:
            species_catalog[name] = {
                'spacing': spacing,
                'crown_radius_mature': max(0.4, spacing/2.5),
                'height': 2.0,
                'shade_pref': (sh_min, sh_max),
                'water_pref': (wa_min, wa_max),
                'growth_rate': 'medium',
                'n_fix': False,
                'layer': 'shrub'
            }
        # máscara por preferência
        mask_shade = (S >= sh_min) & (S <= sh_max)
        mask_water = (W >= wa_min) & (W <= wa_max)
        candidate_mask = base_candidates & mask_shade & mask_water
        placed = place_points_greedy(name, qty, spacing, candidate_mask)
        plan.extend(placed)

    # incluir plantas existentes no plano para avaliação final
    plan = existing_plants + plan
else:
    # Sem wishlist: apenas adicionar plantas existentes para avaliação final
    plan = existing_plants + plan

# ------------- Cálculo de mapas e avaliação -------------
S = shade_map_from_plan([p for p in plan if p['species'] != 'vetiver_row'])
W = water_index(Z, hedgerow_positions_x)

rows = []
for p in plan:
    if p['species'] == 'vetiver_row':
        s, sh, wa = 1.0, S[int(round(p['y'])), int(round(p['x']))], W[int(round(p['y'])), int(round(p['x']))]
    else:
        s, sh, wa = suitability_for(p, S, W, plan)
    rows.append({'species': p['species'], 'x': p['x'], 'y': p['y'], 'age_years': p['age'],
                 'suitability_score_0_1': s, 'local_shade_0_1': sh, 'local_water_0_1': wa})

df = pd.DataFrame(rows)

# ------------- Visualização -------------
fig, axes = plt.subplots(2,2, figsize=(14,10), constrained_layout=True)
ax = axes[0,0]
im0 = ax.imshow(Z, origin='lower', extent=[0, WIDTH_M, 0, HEIGHT_M], cmap='terrain')
ax.set_title('Terreno (elevação relativa)')
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
colors = {'banana': '#f1c232','gliricidia': '#6aa84f','inga': '#38761d','coffee': '#a64d79','cassava': '#e69138','pineapple': '#ffd966','vetiver_row': '#274e13'}
for p in plan:
    s_val = df.loc[(df['x']==p['x']) & (df['y']==p['y']) & (df['species']==p['species']), 'suitability_score_0_1'].values[0]
    c = colors.get(p['species'], '#888888')
    size = {'gliricidia':55, 'inga':55, 'banana':45, 'coffee':25, 'cassava':18, 'pineapple':12, 'vetiver_row':5}.get(p['species'], 22)
    ec = (1 - s_val, s_val, 0.0)
    ax.scatter(p['x'], p['y'], s=size, c=c, edgecolors=[ec], linewidths=1.0, alpha=0.9)
ax.set_title('Layout de plantas (borda = adequação)')
ax.set_xlim(0, WIDTH_M); ax.set_ylim(0, HEIGHT_M); ax.set_aspect('equal'); ax.grid(True, color='lightgray', linestyle=':', linewidth=0.5)

fig.savefig('plano_agrofloresta_v2.png', dpi=180)
df.to_csv('plano_agrofloresta_v2.csv', index=False)

print('Arquivos salvos: plano_agrofloresta_v2.png e plano_agrofloresta_v2.csv')
# Fim
