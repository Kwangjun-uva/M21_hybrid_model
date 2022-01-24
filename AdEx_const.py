t_ref = 2 * 10 ** (-3)   # ms
Cm = 281 * 10 ** (-12)   # pF
gL = 30 * 10 ** (-9)   # nS=
EL = -70.6 * 10 ** (-3)   # mV
VT = -50.4 * 10 ** (-3)   # mV
DeltaT = 2 * 10 ** (-3)   # mV

# Pick an physiological behaviour
# Regular spiking (as in the paper)
tauw = 144 * 10 ** (-3)   # ms
a = 4 * 10 ** (-9)   # nS
b = 0.0805 * 10 ** (-9)   # nA

# spike trace
x_reset = 550 * 10 ** (-12)
offset = 0 * 10 ** (-12)
I_reset = -1 * 10 ** (-12)   # pamp
tau_rise = 5 * 10 ** (-3)   # ms
tau_s = 50 * 10 ** (-3)  # ms

pamp = 10 ** -12