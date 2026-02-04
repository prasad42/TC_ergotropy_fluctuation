import matplotlib.pyplot as plt
from tc_erg_lib import *

N_arr = np.arange(2, 20, 2)
M = 50
ω0 = ω = 1.0
E = 5 * ω # Energy in the charger 
g_arr = np.round(np.arange(0.0, 1.0, 0.2),2)
tlist = np.arange(0, 10, 0.1)

N_arr = [2, 4]
M = 10
###############################################
# Plot Battery energy <Eb> as function of time 
# and pick time τ for which <Eb> is maximum
###############################################
for N in N_arr:
    j = N // 2
    for g in tqdm(g_arr):
        H = TC_fun(ω, ω0, j, M, g)
        Hb = Hb_fun(ω0, j, M)
        psi0 = coh_state_fun(E, M, j)
        result = qt.sesolve(H, psi0, tlist, e_ops=Hb)
        Eb_list = np.transpose(result.expect)

        plt.plot(tlist, Eb_list, label=f"g={g}")
    plt.title(f"N={N}")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\langle E_b\rangle$")
    plt.legend()
    plt.show()

    # Using these plots, select a time τ for which <Eb> is maximum