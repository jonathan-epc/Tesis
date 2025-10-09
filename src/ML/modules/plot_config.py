# src/ML/modules/plot_config.py

"""
Configuration file for plotting.
Contains all definitions for variable names, units, symbols, and text translations.
"""

UNITS = {
    "H": "m",
    "U": "m/s",
    "V": "m/s",
    "B": "m",
    "H*": "-",
    "U*": "-",
    "V*": "-",
    "H0": "m",
    "Q0": "m³/s",
    "n": r"$s \cdot m^{-1/3}$",
    "nut": r"$m^2/s$",
    "Hr": "-",
    "Fr": "-",
    "M": "-",
    "Re": "-",
    "B*": "-",
    "Ar": "-",
    "Vr": "-",
}

SYMBOLS = {
    "H": "$H$",
    "U": "$U$",
    "V": "$V$",
    "B": "$B$",
    "H*": "$H^*$",
    "U*": "$U^*$",
    "V*": "$V^*$",
    "H0": "$H_0$",
    "Q0": "$Q_0$",
    "n": "$n$",
    "nut": "$\\nu_t$",
    "Hr": "$H_r$",
    "Fr": "$F_r$",
    "M": "$M$",
    "Re": "$Re_t$",
    "B*": "$B^*$",
    "Ar": "$A_r$",
    "Vr": "$V_r$",
}

LONG_NAMES = {
    "H": {"en": "Water Depth", "es": "Profundidad del Agua"},
    "U": {"en": "X-Velocity", "es": "Velocidad en X"},
    "V": {"en": "Y-Velocity", "es": "Velocidad en Y"},
    "B": {"en": "Bed Geometry", "es": "Geometría del Lecho"},
    "H*": {"en": "Dimensionless Depth", "es": "Profundidad Adimensional"},
    "U*": {"en": "Dimensionless X-Velocity", "es": "Velocidad Adim. en X"},
    "V*": {"en": "Dimensionless Y-Velocity", "es": "Velocidad Adim. en Y"},
    "B*": {"en": "Dimensionless Bed", "es": "Lecho Adimensional"},
    "H0": {"en": "Initial Height", "es": "Altura Inicial"},
    "Q0": {"en": "Inflow", "es": "Caudal"},
    "n": {"en": "Manning's n", "es": "n de Manning"},
    "nut": {"en": "Turb. Viscosity", "es": "Visc. Turb."},
    "Hr": {"en": "Height Ratio", "es": "Relación de Altura"},
    "Fr": {"en": "Froude Number", "es": "Número de Froude"},
    "M": {"en": "Manning Param.", "es": "Parám. de Manning"},
    "Re": {"en": "Reynolds Number", "es": "Número de Reynolds"},
    "Ar": {"en": "Aspect Ratio", "es": "Relación de Aspecto"},
    "Vr": {"en": "Velocity Ratio", "es": "Relación de Velocidad"},
}

TRANSLATIONS = {
    "en": {
        "truth": "Ground Truth",
        "pred": "Prediction",
        "diff": "Difference",
        "identity": "Identity",
        "case": "Case",
        "error": "Error",
        "frequency": "Frequency",
        "count": "Point Density",
        "x_axis": "X Position",
        "y_axis": "Y Position",
        "scatter_title": "Prediction vs. Target Scatter Plots",
    },
    "es": {
        "truth": "Valor Real",
        "pred": "Predicción",
        "diff": "Diferencia",
        "identity": "Identidad",
        "case": "Caso",
        "error": "Error",
        "frequency": "Frecuencia",
        "count": "Densidad de Puntos",
        "x_axis": "Posición X",
        "y_axis": "Posición Y",
        "scatter_title": "Gráficos de Dispersión",
    },
}
