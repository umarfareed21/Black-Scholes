import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Black–Scholes Option Pricing", layout="wide")
plt.style.use("dark_background")

# ------------------------------
# Black–Scholes Functions
# ------------------------------
def call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ------------------------------
# Sidebar (Branding + Inputs)
# ------------------------------
st.sidebar.markdown("## Mohammad Umar Fareed")
st.sidebar.caption("Quant Finance | Black–Scholes Model")
st.sidebar.markdown("---")

company = st.sidebar.text_input("Company / Stock Name", value="Apple Inc.")
S0 = st.sidebar.number_input("Stock Price (S₀)", value=150.0)
K = st.sidebar.number_input("Strike Price (K)", value=150.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=0.5)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma0 = st.sidebar.number_input("Volatility (σ)", value=0.2)

calculate = st.sidebar.button("Calculate")

# ------------------------------
# Title
# ------------------------------
st.title(f"Black–Scholes Option Pricing Dashboard — {company}")

 # ------------------------------
# Black–Scholes Formula (LaTeX)
# ------------------------------
st.markdown("##  Black–Scholes Formula")

st.latex(r"C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)")
st.latex(r"P = K e^{-rT} \Phi(-d_2) - S_0 \Phi(-d_1)")

st.latex(
        r"d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}"
        r"{\sigma \sqrt{T}}"
    )

st.latex(r"d_2 = d_1 - \sigma \sqrt{T}")


# ------------------------------
# Main Logic
# ------------------------------
if calculate:
    call_val = call_price(S0, K, T, r, sigma0)
    put_val = put_price(S0, K, T, r, sigma0)

    c1, c2 = st.columns(2)
    c1.success(f"Call Option Price: {call_val:.4f}")
    c2.error(f"Put Option Price: {put_val:.4f}")

    # ------------------------------
    # Explanation (Before Heatmaps)
    # ------------------------------
    st.markdown("##  Model Explanation")
    st.write(f"""
    This dashboard prices **European Call and Put Options** for **{company}**
    using the **Black–Scholes Option Pricing Model**.

    **Input Meaning:**
    - **Stock Price (S₀):** Current market price of {company}
    - **Strike Price (K):** Agreed exercise price
    - **Time to Maturity (T):** Remaining life of the option (in years)
    - **Risk-Free Rate (r):** Return on a risk-free investment
    - **Volatility (σ):** Expected variability in stock returns

    **Model Assumptions:**
    - No arbitrage
    - Constant interest rate and volatility
    - Lognormal stock price distribution
    """)

   
    # ------------------------------
    # Heatmap Short Note
    # ------------------------------
    st.markdown("##  Heatmap Insight")
    st.write("""
    A **heatmap** shows how option prices change when **two variables**
    vary simultaneously.

    - **X-axis:** Stock Price  
    - **Y-axis:** Volatility  
    - **Color Intensity:** Option Value  

    Darker shades indicate **higher option prices**.
    The **white dot** represents your exact input values.
    """)

    # ------------------------------
    # Heatmap Grid
    # ------------------------------
    S_range = np.linspace(S0 - 10, S0 + 10, 25)
    vol_range = np.linspace(0.10, 0.30, 25)

    call_matrix = np.zeros((len(vol_range), len(S_range)))
    put_matrix = np.zeros_like(call_matrix)

    for i, vol in enumerate(vol_range):
        for j, S in enumerate(S_range):
            call_matrix[i, j] = call_price(S, K, T, r, vol)
            put_matrix[i, j] = put_price(S, K, T, r, vol)

    x_idx = np.argmin(np.abs(S_range - S0))
    y_idx = np.argmin(np.abs(vol_range - sigma0))

    # ==============================
    # CALL HEATMAP 
    # ==============================
    fig_c, ax_c = plt.subplots(figsize=(9, 5))
    fig_c.patch.set_alpha(0)
    ax_c.set_facecolor("none")

    im_c = ax_c.imshow(call_matrix, cmap="Blues", origin="lower", aspect="auto")
    ax_c.set_title(f"Call Option Heatmap — {company}")
    ax_c.set_xlabel("Stock Price")
    ax_c.set_ylabel("Volatility")

    ax_c.set_xticks(np.linspace(0, len(S_range)-1, 6))
    ax_c.set_xticklabels(np.round(np.linspace(S_range[0], S_range[-1], 6), 1))
    ax_c.set_yticks(np.linspace(0, len(vol_range)-1, 6))
    ax_c.set_yticklabels(np.round(np.linspace(vol_range[0], vol_range[-1], 6), 2))

    cbar_c = fig_c.colorbar(im_c, ax=ax_c)
    cbar_c.set_label("Call Price")

    ax_c.scatter(x_idx, y_idx, s=70, c="white", edgecolors="black", zorder=5)
    ax_c.text(
        x_idx + 1,
        y_idx + 1,
        f"S: {S0:.2f}\nσ: {sigma0:.2f}\nCall: {call_val:.4f}",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.75, boxstyle="round")
    )

    call_file = "call_option_heatmap_transparent.png"
    plt.savefig(call_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig_c)

    st.image(call_file)
    st.download_button(" Download Call Heatmap (PNG)", open(call_file, "rb"), call_file)

    # ==============================
    # PUT HEATMAP 
    # ==============================
    fig_p, ax_p = plt.subplots(figsize=(9, 5))
    fig_p.patch.set_alpha(0)
    ax_p.set_facecolor("none")

    im_p = ax_p.imshow(put_matrix, cmap="Blues", origin="lower", aspect="auto")
    ax_p.set_title(f"Put Option Heatmap — {company}")
    ax_p.set_xlabel("Stock Price")
    ax_p.set_ylabel("Volatility")

    ax_p.set_xticks(np.linspace(0, len(S_range)-1, 6))
    ax_p.set_xticklabels(np.round(np.linspace(S_range[0], S_range[-1], 6), 1))
    ax_p.set_yticks(np.linspace(0, len(vol_range)-1, 6))
    ax_p.set_yticklabels(np.round(np.linspace(vol_range[0], vol_range[-1], 6), 2))

    cbar_p = fig_p.colorbar(im_p, ax=ax_p)
    cbar_p.set_label("Put Price")

    ax_p.scatter(x_idx, y_idx, s=70, c="white", edgecolors="black", zorder=5)
    ax_p.text(
        x_idx + 1,
        y_idx + 1,
        f"S: {S0:.2f}\nσ: {sigma0:.2f}\nPut: {put_val:.4f}",
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.75, boxstyle="round")
    )

    put_file = "put_option_heatmap_transparent.png"
    plt.savefig(put_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig_p)

    st.image(put_file)
    st.download_button(" Download Put Heatmap (PNG)", open(put_file, "rb"), put_file)

else:
    st.info(" Enter values and click **Calculate**")