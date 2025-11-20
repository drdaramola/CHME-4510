import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Heat Effects in PFR", layout="wide")

# --- SIDEBAR: CONTROL PANEL ---
st.sidebar.header("1. Reactor Conditions")
T0 = st.sidebar.slider("Inlet Temperature (K)", 300, 500, 350, help="Feed temperature entering the reactor")
U = st.sidebar.slider("Heat Transfer Coeff U (J/m²·s·K)", 0, 500, 150, help="0 = Adiabatic")
Ta = st.sidebar.slider("Coolant Temperature Ta (K)", 250, 500, 300)

st.sidebar.header("2. Reaction Parameters")
dH_rxn = st.sidebar.number_input("Heat of Reaction (J/mol)", value=-60000.0, step=1000.0, help="Negative for Exothermic")
Ea = st.sidebar.number_input("Activation Energy (J/mol)", value=70000.0)
Kc_300 = st.sidebar.number_input("Kc at 300K", value=100.0)

st.sidebar.header("3. Feed Properties")
FA0 = st.sidebar.number_input("Molar Flow Rate (mol/s)", value=1.0)
Cp = st.sidebar.number_input("Specific Heat (J/mol·K)", value=150.0)

# --- MODELING ---
# Constants
R = 8.314  # J/mol·K
k_ref = 0.1 # Rate constant at 300K
a = 4 / 0.1 # Area to Volume ratio (assuming diameter=0.1m for simplicity)

def pfr_model(V, y, T0, U, Ta, dH_rxn, Ea, FA0, Cp):
    X, T = y
    
    # Arrhenius Equation for k (Forward rate)
    k = k_ref * np.exp((Ea / R) * (1/300 - 1/T))
    
    # Van't Hoff Equation for Equilibrium Constant Kc
    # ln(K2/K1) = -dH/R * (1/T2 - 1/T1)
    Kc = Kc_300 * np.exp((-dH_rxn / R) * (1/T - 1/300))
    
    # Concentration (assuming constant pressure, liquid phase or negligible expansion)
    CA0 = 1.0 # mol/L placeholder
    CA = CA0 * (1 - X)
    CB = CA0 * X
    
    # Rate Law: -rA = k * (CA - CB/Kc)
    rate = k * (CA - CB / Kc)
    
    # 1. Mole Balance: dX/dV = -rA / FA0
    dX_dV = rate / FA0
    
    # 2. Energy Balance: dT/dV = (Q_gen - Q_removed) / (FA0 * Cp)
    # Q_gen = (-rA) * (-dH_rxn)
    # Q_removed = U * a * (T - Ta)
    Q_gen = rate * (-dH_rxn)
    Q_remove = U * a * (T - Ta)
    
    dT_dV = (Q_gen - Q_remove) / (FA0 * Cp)
    
    return [dX_dV, dT_dV]

# Run Simulation
V_span = (0, 50) # Reactor Volume 0 to 50 Liters
y0 = [0.0, T0] # Initial Conversion=0, Temp=T0

sol = solve_ivp(pfr_model, V_span, y0, args=(T0, U, Ta, dH_rxn, Ea, FA0, Cp), dense_output=True)
V_plot = np.linspace(0, 50, 100)
y_plot = sol.sol(V_plot)
X_profile = y_plot[0]
T_profile = y_plot[1]

# --- MAIN DASHBOARD ---
st.title("Heat Effects in Chemical Reactors")
st.markdown(f"""
**Instructor Mode:** This module demonstrates the competition between **Reaction Kinetics** (which wants high T) and **Thermodynamics** (which wants low T for exothermic reactions).
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reactor Profiles")
    
    # Dual Axis Plot
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Reactor Volume (L)')
    ax1.set_ylabel('Temperature (K)', color=color)
    ax1.plot(V_plot, T_profile, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Conversion (X)', color=color)
    ax2.plot(V_plot, X_profile, color=color, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    
    st.pyplot(fig)
    
    st.info(f"**Final Conversion:** {X_profile[-1]:.2%} | **Max Temp:** {np.max(T_profile):.1f} K")

with col2:
    st.subheader("The Equilibrium Limitation (X vs T)")
    
    # Calculate Equilibrium Curve
    T_range = np.linspace(300, 600, 100)
    Kc_range = Kc_300 * np.exp((-dH_rxn / R) * (1/T_range - 1/300))
    # For A <-> B, Xe = Kc / (1 + Kc)
    Xe_range = Kc_range / (1 + Kc_range)
    
    fig2, ax3 = plt.subplots()
    ax3.plot(T_range, Xe_range, 'g-', label='Equilibrium (Xe)')
    ax3.plot(T_profile, X_profile, 'r--', label='Actual Trajectory', linewidth=2)
    
    # Aesthetics
    ax3.set_xlabel("Temperature (K)")
    ax3.set_ylabel("Conversion (X)")
    ax3.set_title("Kinetic Trajectory vs. Equilibrium Limit")
    ax3.legend()
    ax3.grid(True)
    ax3.set_xlim(280, 600)
    ax3.set_ylim(0, 1)
    
    st.pyplot(fig2)

# --- EXPLANATION SECTION ---
st.divider()
st.subheader("Discussion Topics for Class")
st.markdown(f"""
1. **Adiabatic Operation:** Set `U = 0`. Notice how the temperature skyrockets? This pushes the reaction line to the right, hitting the equilibrium curve (Green line) early. This effectively "kills" the conversion.
2. **Interstage Cooling:** If you increase `U` (Heat Transfer), you bend the red trajectory back to the left. This keeps the reactor in a high-rate zone without hitting the equilibrium ceiling.
3. **Safety:** Look at the "Max Temp" readout. If you increase `T0` slightly, does the reactor run away? (Parametric Sensitivity).
""")