import matplotlib.pyplot as plt
from tc_erg_lib import *

N_arr = np.arange(2, 20, 2)
M = 50
ω0 = ω = 1.0 
N_arr = np.arange(2, 15, 2)
g = 1.0

tlist = np.arange(0.01, (2*np.pi), 0.1)

###############################################
# Plot Battery energy <Eb> as function of time 
# and pick time τ for which <Eb> is maximum
###############################################
τ_list = []
for N in N_arr:
    E = N * ω # Energy in the charger
    j = N // 2
    H = TC_fun(ω, ω0, j, M, g)
    Hb = Hb_full_fun(ω0, j, M)
    ψ0 = coh_state_fun(E, M, ω, j)
    result = qt.sesolve(H, ψ0, tlist, e_ops=Hb)
    Eb_list = np.transpose(result.expect).ravel()
    plt.title(f"g={g}")
    plt.plot(tlist, Eb_list/tlist, label=f"N={N}")
    idx = np.argmax(Eb_list/tlist)
    τ = tlist[idx]
    print(τ)
    τ_list.append(τ)
    plt.title(f"N={N}")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\langle E_b\rangle$")
    plt.legend()
plt.show()

###############################################
# Plot Battery Ergotropy as function of time 
# and pick time τ for which <Eb> is maximum
###############################################
# τ_list = []
# for N in N_arr:
    
#     E = N * ω # Energy in the charger
#     j = N // 2
#     H = TC_fun(ω, ω0, j, M, g)
#     Hb = Hb_fun(ω0, j, M)
#     ψ0 = coh_state_fun(E, M, ω, j)
#     result = qt.sesolve(H, ψ0, tlist)
#     ψ_list = result.states
#     erg_list = []
#     for idx, t in enumerate(tlist):
#         ρ = qt.ket2dm(ψ_list[idx])
#         ρb = qt.ptrace(ρ, 1)
#         erg = erg_fun(ρb, Hb)
#         erg_list.append(erg)
#     plt.title(f"g={g}")
#     plt.plot(tlist, erg_list/tlist, label=f"N={N}")
#     # idx = np.argmax(Eb_list)
#     # τ = tlist[idx]
#     # τ_list.append(τ)
#     plt.title(f"N={N}")
#     plt.xlabel(r"$\tau$")
#     plt.ylabel(r"$\langle \mathcal{E}\rangle$")
#     plt.legend()
# plt.show()

# Using these plots, select a time τ for which <Eb> is maximum
plt.plot(N_arr, τ_list)
plt.xlabel(fr"Coupling $g$")
plt.ylabel(fr"$\tau$ at $\langle E_b^m\rangle$")
plt.show()

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
    τ = τ_list[N_idx]
    H = TC_fun(ω, ω0, j, M, g)
    Hb = Hb_fun(ω0, j, M)
    ψ0 = coh_state_fun(E, M, ω, j)
    result = qt.sesolve(H, ψ0, [0, τ])
    ψτ = result.states[-1]
    ρτ = qt.ket2dm(ψτ)
    ρbτ = qt.ptrace(ρτ, 1)
    erg = erg_fun(ρbτ, Hb)
    Eb = (Hb*ρbτ).tr()
    erg_list.append(erg/Eb)

###############################################
# Plot erg as function of g for different N 
###############################################

plt.plot(N_arr, erg_list, marker=".")
plt.xlabel(fr"Spins $N$")
plt.ylabel(r"$\mathcal{E}/E_b$")
plt.ylim(-0.05,1.05)
plt.show()