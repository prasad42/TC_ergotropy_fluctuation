import matplotlib.pyplot as plt
from tc_erg_lib import *

N_arr = np.arange(2, 20, 2)
M = 50
ω0 = ω = 1.0 
g_arr = np.round(np.arange(0.1, 1.0, 0.1),2)
tlist = np.arange(0, 20, 0.1)

N_arr = [2, 4, 6]
# g_arr = [0.1, 1.0, 2.0]
###############################################
# Plot Battery energy <Eb> as function of time 
# and pick time τ for which <Eb> is maximum
###############################################
τ_list = []
for N in N_arr:
    E = N * ω # Energy in the charger
    j = N // 2
    τ_list1 = []
    for g in tqdm(g_arr):
        H = TC_fun(ω, ω0, j, M, g)
        Hb = Hb_full_fun(ω0, j, M)
        ψ0 = fock_state_fun(E, M, ω, j)
        result = qt.sesolve(H, ψ0, tlist, e_ops=Hb)
        Eb_list = np.transpose(result.expect)
        plt.plot(tlist, Eb_list, label=f"g={g}")
        idx = np.argmax(Eb_list)
        τ = tlist[idx]
        τ_list1.append(τ)
    τ_list.append(τ_list1)
    plt.title(f"N={N}")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\langle E_b\rangle$")
    plt.legend()
    plt.show()

# Using these plots, select a time τ for which <Eb> is maximum
# for N_idx, N in enumerate(N_arr):
#     plt.plot(g_arr, τ_list[N_idx])
#     plt.xlabel(fr"Coupling $g$")
#     plt.ylabel(fr"$\tau$ at $\langle E_b^m\rangle$")
#     plt.show()

###############################################
# Using the τ above, run sesolve again till τ
# and find ρb(τ) = Tr_c[ρ(τ)]
# With ρb(τ) = Tr_c[ρ(τ)], use erg_fun(pnm_matrix)
# function to find ergotropy and erg_var_fun(pnm_matrix)
# to find variance in ergotropy
###############################################
print("Calculating Ergotropy")
erg_list = []
for N_idx, N in enumerate(N_arr):
    E = N * ω # Energy in the charger
    erg_list1 = []
    for g_idx, g in tqdm(enumerate(g_arr)):
        τ = τ_list[N_idx][g_idx]
        H = TC_fun(ω, ω0, j, M, g)
        Hb = Hb_fun(ω0, j, M)
        ψ0 = fock_state_fun(E, M, ω, j)
        result = qt.sesolve(H, ψ0, [0, τ])
        ψτ = result.states[-1]
        ρτ = qt.ket2dm(ψτ)
        ρbτ = qt.ptrace(ρτ, 1)
        erg = erg_fun(ρb0=ρbτ, Hb=Hb)
        Eb = (Hb*ρbτ).tr()
        erg_list1.append(erg/Eb)
    erg_list.append(erg_list1)

###############################################
# Plot erg as function of g for different N 
###############################################

for N_idx, N in enumerate(N_arr):
    plt.plot(g_arr, erg_list[N_idx], marker=".", label=f"N={N}")
    plt.xlabel(fr"Coupling $g$")
    plt.ylabel(fr"Ergotropy/Eb")
    plt.legend()
plt.show()