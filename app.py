import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# -------------------------
# Konstanta fisika
# -------------------------
h = 6.62607015e-34  # J s
c = 299792458.0     # m/s
eV = 1.602176634e-19  # J per eV
kB = 1.380649e-23   # J/K

# -------------------------
# Util: energi, lambda, warna
# -------------------------
def E_n(n):
    """Energi tingkat n dalam eV (Bohr model, Hidrogen)."""
    return -13.6 / (n ** 2)

def deltaE_eV(n_i, n_f):
    return abs(E_n(n_f) - E_n(n_i))

def lambda_nm_from_deltaE(deltaE_eV):
    """panjang gelombang (nm) dari deltaE (eV)"""
    deltaJ = deltaE_eV * eV
    lam_m = (h * c) / deltaJ
    lam_nm = lam_m * 1e9
    return lam_nm

def wavelength_to_rgb(wavelength):
    """Approximate visible wavelength (nm) -> RGB tuple (0..1).
       Simple approximation for educational visualization."""
    wl = float(wavelength)
    if wl < 380 or wl > 780:
        return (0.5, 0.5, 0.5)
    # piecewise linear approx across visible
    if wl < 440:
        r = (440 - wl) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl < 510:
        r = 0.0
        g = 1.0
        b = (510 - wl) / (510 - 490)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl < 645:
        r = 1.0
        g = (645 - wl) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0
    # apply simple intensity falloff at edges (not critical)
    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl > 700:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    else:
        factor = 1.0
    return (r * factor, g * factor, b * factor)

# -------------------------
# UI: Pengaturan
# -------------------------
st.set_page_config(page_title="Atomic Emission Spectrum (Integrated)", layout="wide")
st.title("Atomic Emission Spectrum — Integrated Visualization")

st.write("""
Simulasi interaktif yang mengaitkan **tingkat energi (Bohr)**, **transisi elektron**, 
dan **garis spektrum emisi**. Pilih seri spektral, pilih mode tampilan, dan amati keterkaitan.
""")

# sidebar controls
st.sidebar.header("Pengaturan Tampilan")
series_choice = st.sidebar.selectbox("Pilih seri (nₙ → n_f)", ["Balmer (visible, n_f=2)",
                                                               "Lyman (UV, n_f=1)",
                                                               "Paschen (IR, n_f=3)"])
mode_choice = st.sidebar.radio("Mode tampilan", ["Single transition (pilih n_i)", "All transitions (dari n_i min ke n_i max)"])
show_intensity = st.sidebar.checkbox("Skala intensitas (Boltzmann)", value=False)
if show_intensity:
    T = st.sidebar.slider("Temperatur (K) untuk Boltzmann", 300, 10000, 3000, step=100)
else:
    T = None

# determine n_f
if "Balmer" in series_choice:
    n_f = 2
elif "Lyman" in series_choice:
    n_f = 1
else:
    n_f = 3

# single or all
if mode_choice.startswith("Single"):
    n_i = st.sidebar.selectbox("Tingkat awal n_i", [i for i in range(n_f+1, 11)], index=1)
    selected_transitions = [(n_i, n_f)]
else:
    n_min = st.sidebar.number_input("n_i minimum", min_value=n_f+1, max_value=20, value=n_f+1, step=1)
    n_max = st.sidebar.number_input("n_i maximum", min_value=n_min, max_value=30, value=n_f+5, step=1)
    n_min = int(n_min); n_max = int(n_max)
    selected_transitions = [(n, n_f) for n in range(n_min, n_max+1)]

# checkbox for showing series legend
show_legend = st.sidebar.checkbox("Tampilkan legenda seri & keterangan", value=True)

st.markdown("---")

# -------------------------
# Create data for transitions
# -------------------------
transitions = []
for (ni, nf) in selected_transitions:
    dE = deltaE_eV(ni, nf)
    lam = lambda_nm_from_deltaE(dE)
    color = wavelength_to_rgb(lam)
    if show_intensity:
        # population proportional to exp(-E_i/kT), E_i in J
        Ei_J = E_n(ni) * eV
        pop = np.exp(-Ei_J / (kB * T))
    else:
        pop = 1.0
    transitions.append({
        "n_i": ni, "n_f": nf, "ΔE(eV)": dE, "λ(nm)": lam, "color": color, "pop": pop
    })

# normalize intensity values for plotting heights
pops = np.array([t["pop"] for t in transitions])
if pops.size == 0:
    pops = np.array([1.0])
p_norm = pops / pops.max()

for i, t in enumerate(transitions):
    t["int_norm"] = float(p_norm[i])

# -------------------------
# Plot integrated figure
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# layout coordinates:
x_level_start = 0.05
x_level_end = 0.55
x_spec_start = 0.65
x_spec_end = 0.95

# energy levels to draw (n from 1 to max_n_display)
max_n = max([t["n_i"] for t in transitions] + [n_f, 7])
levels = list(range(1, max_n + 1))
y_levels = {n: E_n(n) for n in levels}

# draw energy levels on left
for n in levels:
    y = y_levels[n]
    ax.hlines(y, x_level_start, x_level_end, color="black", linewidth=1)
    ax.text(x_level_end + 0.01, y, f"n={n}", va="center", fontsize=9)

# vertical spacing for spectrum lines (stacked downwards)
spec_y_base = min(y_levels.values()) - 1.2  # base y for first spectrum line
spec_gap = 0.5
# compute x positions for each transition horizontally between level and spectrum
n_trans = len(transitions)
if n_trans == 1:
    xs = [ (x_level_start + x_level_end)/2.0 ]
else:
    xs = np.linspace(x_level_start + 0.05, x_level_end - 0.05, n_trans)

# draw transitions arrows and corresponding spectrum lines aligned vertically under arrow
for idx, t in enumerate(transitions):
    ni = t["n_i"]; nf = t["n_f"]
    Ei = y_levels[ni]; Ef = y_levels[nf]
    x_arrow = xs[idx]
    # arrow
    ax.annotate("", xy=(x_arrow, Ef), xytext=(x_arrow, Ei),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=t["color"], alpha=0.9 * (0.4 + 0.6 * t["int_norm"])))
    # label ΔE near mid arrow
    y_mid = (Ei + Ef) / 2.0
    ax.text(x_arrow + 0.02, y_mid, f"ΔE={t['ΔE(eV)']:.3f} eV", fontsize=8, va="center")
    # plot spectrum line directly below arrow (same x), stacked downward per index
    spec_y = spec_y_base - idx * spec_gap
    line_height = 0.9 * t["int_norm"]  # height proportional to normalized intensity
    ax.vlines(x_arrow, spec_y, spec_y + line_height, color=t["color"], linewidth=8, alpha=0.9 * (0.3 + 0.7 * t["int_norm"]))
    # annotate wavelength at the line
    ax.text(x_arrow + 0.02, spec_y + line_height/2.0, f"{t['λ(nm)']:.1f} nm", fontsize=8, va="center")
    # optionally show series name next to first group
    if idx == 0:
        series_label = "Balmer" if n_f == 2 else ("Lyman" if n_f == 1 else "Paschen")
        ax.text(x_arrow - 0.25, spec_y + line_height/2.0, f"{series_label} series (n→{n_f})", fontsize=9, color="gray")

# Add explanatory rulers: x axis for conceptual layout (no real scale)
ax.set_xlim(0.0, 1.0)
# set y limits to include energy levels and spectrum area
ax.set_ylim(spec_y_base - len(transitions)*spec_gap - 0.3, 0.5)
ax.set_xticks([])
ax.set_ylabel("Energi (eV), posisi visual")
ax.set_title("Tingkat Energi (atas) dan Garis Spektrum (sejajar dengan transisi yang menyebabkannya)")

# optional: add small legend
if show_legend:
    patches = []
    # color patch examples from visible ranges
    patches.append(mpatches.Patch(color=wavelength_to_rgb(656), label='~656 nm (red)'))
    patches.append(mpatches.Patch(color=wavelength_to_rgb(486), label='~486 nm (blue/green)'))
    patches.append(mpatches.Patch(color=wavelength_to_rgb(434), label='~434 nm (violet)'))
    ax.legend(handles=patches, loc='upper right')

st.pyplot(fig)

# -------------------------
# Dynamic explanatory text and table
# -------------------------
st.markdown("---")
st.header("Interpretasi dan Keterangan")
st.write("Setiap panah di bagian atas menunjukkan elektron yang turun dari tingkat energi nᵢ ke n_f. "
         "Garis berwarna tepat di bawah panah adalah 'jejak' foton yang dihasilkan — panjang gelombang (λ) ditulis di sampingnya.")
st.write("Catatan penting:")
st.write("- Posisi garis (warna/λ) **tidak berubah** dengan temperatur; warna bergantung pada ΔE (struktur energi atom).")
if show_intensity:
    st.write(f"- Mode intensitas aktif (Boltzmann) pada T = {T} K. Tinggi garis menunjukkan intensitas relatif (populasi tingkat awal).")
else:
    st.write("- Mode intensitas dinonaktifkan. Untuk melihat perubahan intensitas, aktifkan 'Skala intensitas' di sidebar.")

# show transitions table
df_table = pd.DataFrame(transitions)
df_table_display = df_table[["n_i", "n_f", "ΔE(eV)", "λ(nm)", "int_norm"]].rename(columns={"int_norm":"Intensitas(normalized)"})
st.subheader("Data Transisi")
st.dataframe(df_table_display.style.format({"ΔE(eV)":"{:.4f}", "λ(nm)":"{:.1f}", "Intensitas(normalized)":"{:.3f}"}), use_container_width=True)

# -------------------------
# Pedagogical Q&A short
# -------------------------
st.markdown("---")
st.subheader("Pertanyaan reflektif singkat (untuk mahasiswa)")
st.write("""
1. Jelaskan mengapa garis pada spektrum muncul pada posisi λ tertentu (hubungkan ke ΔE dan E_n).  
2. Apakah warna garis akan berubah jika temperatur dinaikkan? Mengapa/kenapa tidak?  
3. Jika kamu menampilkan Lyman series (n_f=1), mengapa sebagian garis berada di UV (di luar rentang tampak)?
""")
