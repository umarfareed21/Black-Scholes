# Black-Scholes-Model
# Black–Scholes Option Pricing Dashboard

An interactive **Streamlit web application** for pricing **European Call and Put options**
using the **Black–Scholes Option Pricing Model**.

This project demonstrates option valuation, sensitivity analysis, and professional
financial visualization using Python.

---

##  Features
- Manual input of option parameters
- Pricing of European Call & Put options
- Stock Price vs Volatility heatmaps
- Exact input point highlighted on heatmaps
- High-resolution transparent PNG downloads
- Clean, professional dashboard UI

---

##  Model Used
**Black–Scholes Option Pricing Model**

$$
C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
$$

$$
P = K e^{-rT} \Phi(-d_2) - S_0 \Phi(-d_1)
$$

Where:

$$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}
{\sigma\sqrt{T}},
\quad
d_2 = d_1 - \sigma\sqrt{T}
$$


---

##  Model Assumptions
- European options
- No arbitrage opportunities
- Constant risk-free rate and volatility
- Lognormally distributed stock prices
- No dividends

---

##  Tech Stack
- Python
- NumPy
- SciPy
- Matplotlib
- Streamlit

---

##  Project Structure
