import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'legend.fontsize': 11,
                     'axes.labelsize': 10,
                     'axes.titlesize': 12,
                     'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

t = 1.0
lam = 0.02
kappa = 1.7724
sigma = 1.0
sig2 = sigma**2 + 2.0*lam*t

mm = [10, 20, 30]
JJ = np.power(2, np.arange(11, 18))
NN = [10*int(np.sqrt(J/np.log(J))) for J in JJ]

DT_err = np.zeros([len(mm), len(JJ), 2])
DT_tms = np.zeros([len(mm), len(JJ)])
DT_u1p = np.zeros([len(mm), len(JJ), 100])
RT_err = np.zeros([len(mm), len(JJ), 2])
RT_tms = np.zeros([len(mm), len(JJ)])
RT_u1p = np.zeros([len(mm), len(JJ), 100])
for mi in range(len(mm)):
    DT_err[mi] = np.loadtxt("heat_data/heat_sv_DT_err_" + str(mm[mi]) + ".csv")
    DT_tms[mi] = np.loadtxt("heat_data/heat_sv_DT_tms_" + str(mm[mi]) + ".csv")
    DT_u1p[mi] = np.loadtxt("heat_data/heat_sv_DT_u1p_" + str(mm[mi]) + ".csv")
    RT_err[mi] = np.loadtxt("heat_data/heat_sv_RT_err_" + str(mm[mi]) + ".csv")
    RT_tms[mi] = np.loadtxt("heat_data/heat_sv_RT_tms_" + str(mm[mi]) + ".csv")
    RT_u1p[mi] = np.loadtxt("heat_data/heat_sv_RT_u1p_" + str(mm[mi]) + ".csv")
    
DN_err = np.zeros([len(mm), len(JJ), 2])
DN_tms = np.zeros([len(mm), len(JJ)])
DN_u1p = np.zeros([len(mm), len(JJ), 100])
RN_err = np.zeros([len(mm), len(JJ), 2])
RN_tms = np.zeros([len(mm), len(JJ)])
RN_u1p = np.zeros([len(mm), len(JJ), 100])
for mi in range(len(mm)):
    DN_err[mi] = np.loadtxt("heat_data/heat_sv_DN_err_" + str(mm[mi]) + ".csv")
    DN_tms[mi] = np.loadtxt("heat_data/heat_sv_DN_tms_" + str(mm[mi]) + ".csv")
    DN_u1p[mi] = np.loadtxt("heat_data/heat_sv_DN_u1p_" + str(mm[mi]) + ".csv")
    RN_err[mi] = np.loadtxt("heat_data/heat_sv_RN_err_" + str(mm[mi]) + ".csv")
    RN_tms[mi] = np.loadtxt("heat_data/heat_sv_RN_tms_" + str(mm[mi]) + ".csv")
    RN_u1p[mi] = np.loadtxt("heat_data/heat_sv_RN_u1p_" + str(mm[mi]) + ".csv")
    
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = cmap(np.arange(len(mm))/len(mm))

def N_of_J(J):
    return 8.0*np.sqrt(J/np.log(J))

def J_of_N(N):
    return np.exp(-scs.lambertw(-8.0**2/N**2).real)

# Error Plot
fig, ax = plt.subplots(figsize = (5.4, 4.0))
Jrate = np.power(np.log(JJ)/JJ, 0.25)
for mi in range(len(mm)):
    aRT = np.sum(Jrate*RT_err[mi, :, 1])/np.sum(np.square(Jrate))
    RT_fit = aRT*Jrate
    aRN = np.sum(Jrate*RN_err[mi, :, 1])/np.sum(np.square(Jrate))
    RN_fit = aRN*Jrate
    ax.plot(JJ, RT_err[mi, :, 1], color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None")
    ax.plot(JJ, RN_err[mi, :, 1], color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None")
    ax.plot(JJ, RT_fit, color = cols[mi], linestyle = (0, (1, 4)))
    ax.plot(JJ, RN_fit, color = cols[mi], linestyle = (0, (1, 4)))
     
ax.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{RT}_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (1, 4)), label = "$(\\ln(J)/J)^{1/4}$")

for mi in range(len(mm)):
    ax.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))
    
ax.legend(loc = "upper right", ncol = 2)
ax.set_xscale('log', base = 2)
plt.ylim([0.0, 0.3])
plt.xlabel("Number of samples $J$")
plt.ylabel("Empirical $L^2$-error on test set")
ax2 = ax.secondary_xaxis("top", functions = (N_of_J, J_of_N))
ax2.set_xscale('log', base = 2)
ax2.set_xlabel("Number of neurons $N = 10 \\sqrt{J/\\ln(J)}$")
plt.savefig("heat_plots/heat_sv_err.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

def ident(x):
    return x

# Function Plot
fig, ax = plt.subplots(figsize = (5.4, 4.0))
for mi in range(len(mm)):
    m = mm[mi]
    u = 0.4*np.ones([100, m])
    u[:, 0] = np.linspace(-1.0, 1.0, 100)
    y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(u), axis = -1)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
    plt.plot(u[:, 0], y, color = cols[mi], linestyle = "solid")
    
for mi in range(len(mm)):
    ax.plot(u[:, 0], DT_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 5, 1, 5)), marker = "+", markevery = slice(0, 500, 60))
    ax.plot(u[:, 0], RT_u1p[mi, -1], color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None", markevery = slice(15, 500, 60))
    ax.plot(u[:, 0], DN_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", markevery = slice(30, 500, 60))
    ax.plot(u[:, 0], RN_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", markevery = slice(45, 500, 60))

ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5)), marker = "+", label = "$\\mathcal{T}_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{RT}_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "None", label = " ")
ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", label = "$\\mathcal{N}^\\rho_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = "solid", label = "True")

for mi in range(len(mm)):
    ax.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))

ax.legend(loc = "upper right", ncol = 3)
plt.xlabel("$u_1$")
plt.ylabel("$f(1,(u_1,1/4,...,1/4))$")
plt.ylim([0.0, 2.0])
ax2 = ax.secondary_xaxis("top", functions = (ident, ident))
ax2.set_xlabel(" ")
plt.savefig("heat_plots/heat_sv_u1p.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Error and Time Table
txt = ""
for mi in range(len(mm)):
    txt = txt + "\multirow{8}{*}{$m = " + str(mm[mi]) + "$} "
    
    # DT
    txt = txt + "& \multirow{2}{*}{$\\mathcal{T}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(JJ)):
        txt = txt + "\\textbf{" + '{:.3f}'.format(DT_err[mi, Ni, 1]) + "} "
        if Ni < len(JJ)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
    
    for Ni in range(len(JJ)):
        txt = txt + "\\textit{" + '{:.2f}'.format(DT_tms[mi, Ni]) + "} s "
        if Ni < len(JJ)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    # RT
    txt = txt + "& \multirow{2}{*}{$\\mathcal{RT}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(JJ)):
        txt = txt + "\cellcolor{gray!40} \\textbf{" + '{:.3f}'.format(RT_err[mi, Ni, 1]) + "} "
        if Ni < len(JJ)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
            
    for Ni in range(len(JJ)):
        txt = txt + "\cellcolor{gray!40} \\textit{" + '{:.2f}'.format(RT_tms[mi, Ni]) + "} s "
        if Ni < len(JJ)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    # DN
    txt = txt + "& \multirow{2}{*}{$\\mathcal{N}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(JJ)):
        txt = txt + "\\textbf{" + '{:.3f}'.format(DN_err[mi, Ni, 1]) + "} "
        if Ni < len(JJ)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
    
    for Ni in range(len(JJ)):
        txt = txt + "\\textit{" + '{:.2f}'.format(DN_tms[mi, Ni]) + "} s "
        if Ni < len(JJ)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    # RN
    txt = txt + "& \multirow{2}{*}{$\\mathcal{RN}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(JJ)):
        txt = txt + "\cellcolor{gray!40} \\textbf{" + '{:.3f}'.format(RN_err[mi, Ni, 1]) + "} "
        if Ni < len(JJ)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
            
    for Ni in range(len(JJ)):
        txt = txt + "\cellcolor{gray!40} \\textit{" + '{:.2f}'.format(RN_tms[mi, Ni]) + "} s "
        if Ni < len(JJ)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + "\\hline \n"

text_file = open("heat_plots/heat_sv_err_tms.txt", "w")
n = text_file.write(txt[:-12])
text_file.close()