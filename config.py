"""Parametrat e modelit. TË GJITHA vlerat me shenjën [SUPOZIM] duhet të zëvendësohen
me të dhëna zyrtare të KRU Hidromorava kur të jenë në dispozicion."""

# --- Rezervuari (Liqeni i Përlepnicës) ---------------------------------------
V_MAX = 4.2e6          # m3  vëllimi total (burim publik ~4.2 Mm3)
# Kurba V = Vmax*(h/Hmax)^m. Hmax = 30.3 m nga matjet zyrtare 2025 (maksimumi në dhjetor 2025).
H_MAX = 30.3           # m   thellësia në nivel të mbushjes (maksimumi i vëzhguar 2025)
M_SHAPE = 1.2          # eksponenti i kurbës  [SUPOZIM]
H_MIN_ALLOWED = 9.0    # m nga fundi – niveli minimal i lejuar (~0.98 Mm³)  [SUPOZIM]
H_CRISIS = 7.0         # m nga fundi – niveli i krizës 2014 (~0.72–0.96 Mm³)
H_SLUDGE = 5.0         # m nga fundi – fillon marrja e llumit (~0.49 Mm³)  [SUPOZIM]
A_LAKE_FULL = M_SHAPE * 4.2e6 / H_MAX   # m2 = dV/dh në mbushje (~0.17 km²)
K_EVAP = 1.05          # avullimi i ujit të hapur = K * ET0

# --- Kërkesa (KRU Hidromorava, "kapaciteti furnizimit", 2018) ----------------
DEMAND_ANNUAL = 3_303_350   # m3/vit  kapaciteti shfrytëzues nga liqeni (≈105 l/s)
SEASONAL = [0.92, 0.92, 0.95, 0.97, 1.02, 1.08, 1.12, 1.12, 1.05, 0.98, 0.94, 0.93]  # [SUPOZIM]
ECO_FLOW = 0.005       # m3/s prurja ekologjike poshtë digës  [SUPOZIM]

# --- Pellgu ujëmbledhës (modeli abcd, Thomas 1981) ---------------------------
ABCD = dict(a=0.98, b=250.0, c=0.35, d=0.12)   # parametra tipikë [SUPOZIM]
# Sipërfaqja efektive e pellgut kalibrohet automatikisht (shih model.calibrate)

# --- Periudhat ----------------------------------------------------------------
SIM_START, SIM_END = "2005-01", "2025-12"
CRISIS_WINDOW = ("2014-01", "2014-04")   # muajt kur raportohet niveli kritik
