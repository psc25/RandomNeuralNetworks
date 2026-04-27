import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scs
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'legend.fontsize': 11,
                     'axes.labelsize': 10,
                     'axes.titlesize': 12,
                     'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

T = 1.0
lam = 0.02
kappa = 1.7724
sigma = 1.0
K = 20

tt = np.reshape(np.linspace(0.0, T, K+1), [-1, 1, 1])
sig2 = sigma**2 + 2.0*lam*tt

mm = [10, 20, 30]
JJ = np.power(2, np.arange(12, 19))
NN = [10*int(np.sqrt(J/np.log(J))) for J in JJ]
    
DF_err = np.zeros([len(mm), len(NN), 2])
DF_tms = np.zeros([len(mm), len(NN)])
DF_u1p = np.zeros([len(mm), len(NN), K+1, 100])
RF_err = np.zeros([len(mm), len(NN), 2])
RF_tms = np.zeros([len(mm), len(NN)])
RF_u1p = np.zeros([len(mm), len(NN), K+1, 100])
for mi in range(len(mm)):
    DF_err[mi] = np.loadtxt("heat_data/heat_usv_DF_err_" + str(mm[mi]) + ".csv")
    DF_tms[mi] = np.loadtxt("heat_data/heat_usv_DF_tms_" + str(mm[mi]) + ".csv")
    DF_u1p[mi] = np.reshape(np.loadtxt("heat_data/heat_usv_DF_u1p_" + str(mm[mi]) + ".csv"), [len(NN), K+1, 100])
    RF_err[mi] = np.loadtxt("heat_data/heat_usv_RF_err_" + str(mm[mi]) + ".csv")
    RF_tms[mi] = np.loadtxt("heat_data/heat_usv_RF_tms_" + str(mm[mi]) + ".csv")
    RF_u1p[mi] = np.reshape(np.loadtxt("heat_data/heat_usv_RF_u1p_" + str(mm[mi]) + ".csv"), [len(NN), K+1, 100])
    
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = cmap(np.arange(len(mm))/len(mm))

def N_of_J(J):
    return 10.0*np.sqrt(J/np.log(J))

def J_of_N(N):
    return np.exp(-scs.lambertw(-10.0**2/N**2).real)

# Error Plot
fig, ax = plt.subplots(figsize = (5.4, 4.0))
Jrate = np.power(np.log(JJ)/JJ, 0.25)
for mi in range(len(mm)):
    aDF = np.sum(Jrate*DF_err[mi, :, 1])/np.sum(np.square(Jrate))
    DF_fit = aDF*Jrate
    aRF = np.sum(Jrate*RF_err[mi, :, 1])/np.sum(np.square(Jrate))
    RF_fit = aRF*Jrate
    ax.plot(JJ, DF_err[mi, :, 1], color = cols[mi], linestyle = (0, (3, 5, 1, 5)), marker = "+")
    ax.plot(JJ, RF_err[mi, :, 1], color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None")
    ax.plot(JJ, DF_fit, color = cols[mi], linestyle = (0, (1, 4)))
    ax.plot(JJ, RF_fit, color = cols[mi], linestyle = (0, (1, 4)))
     
ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5)), marker = "+", markerfacecolor = "None", label = "$\\lbrace g \\rbrace_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{R}\\lbrace g \\rbrace_{\\mathbb{R}^m,1}$")
ax.plot(np.nan, np.nan, color = "black", linestyle = (0, (1, 4)), label = "$(\\ln(J)/J)^{1/4}$")

for mi in range(len(mm)):
    ax.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))
    
ax.legend(loc = "upper right", ncol = 2)
ax.set_xscale('log', base = 2)
plt.ylim([0.0, 0.4])
plt.xlabel("Number of samples $J$")
plt.ylabel("Empirical $L^2$-error on test set")
ax2 = ax.secondary_xaxis("top", functions = (N_of_J, J_of_N))
ax2.set_xscale('log', base = 2)
ax2.set_xlabel("Number of neurons $N = 10 \\sqrt{J/\\ln(J)}$")
plt.savefig("heat_plots/heat_usv_err.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Function Plots
for mi in range(len(mm)):
    m = mm[mi]
    fig = plt.figure(figsize = (5.4, 4.0))
    ax = fig.add_subplot(projection = '3d')
    u = 0.4*np.ones([1, 100, m])
    u1 = np.linspace(-1.0, 1.0, 100)
    u[0, :, 0] = u1
    y = 0.0001*m**6*np.power(kappa, m)*np.exp(-0.5*np.sum(np.square(u), axis = -1, keepdims = True)/sig2)/np.power(2.0*np.pi*sig2, m/2.0)
    uu_plot, tt_plot = np.meshgrid(u1, tt.flatten())
    ax.plot_surface(tt_plot, uu_plot, DF_u1p[mi, -1], cmap = plt.cm.Blues_r, alpha = 0.7, linewidth = 0, antialiased = False)
    ax.plot_surface(tt_plot, uu_plot, RF_u1p[mi, -1], cmap = plt.cm.Reds_r, alpha = 0.7, linewidth = 0, antialiased = False)
    ax.plot_wireframe(tt_plot, uu_plot, y[:, :, 0], rstride = 3, cstride = 4, color = "black", linewidth = 0.4)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$u_1$')
    ax.set_zlabel('$f(t,(u_1,0.4,...,0.4))$')
    ax.plot(np.nan, np.nan, np.nan, label = "True", marker = r"$\#$", linestyle = 'None', markerfacecolor = "black", markeredgecolor = "black")
    ax.plot(np.nan, np.nan, np.nan, label = " ", linestyle = 'None')
    ax.plot(np.nan, np.nan, np.nan, label = "$\\lbrace g \\rbrace_{\mathbb{R}^m,1}$", linestyle = 'None', marker = "s", markerfacecolor = plt.cm.Blues(0.9), markeredgecolor = plt.cm.Blues(0.9))
    ax.plot(np.nan, np.nan, np.nan, label = "$\\mathcal{R}\\lbrace g \\rbrace_{\mathbb{R}^m,1}$", linestyle = 'None', marker = "s", markerfacecolor = plt.cm.Reds(0.9), markeredgecolor = plt.cm.Reds(0.9))
    ax.legend(loc = "upper right", ncol = 2)
    plt.savefig("heat_plots/heat_usv_u1p_" + str(m) + ".png", dpi = 500)
    plt.show()
    plt.close(fig)

# Error and Time Table
txt = ""
for mi in range(len(mm)):
    txt = txt + "\multirow{4}{*}{$m = " + str(mm[mi]) + "$} "
            
    # DF
    txt = txt + "& \multirow{2}{*}{$\\lbrace g \\rbrace_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        txt = txt + "\\textbf{" + '{:.3f}'.format(DF_err[mi, Ni, 1]) + "} "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
    
    for Ni in range(len(NN)):
        txt = txt + "\\textit{" + '{:.2f}'.format(DF_tms[mi, Ni]) + "} s "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    # RF
    txt = txt + "& \multirow{2}{*}{$\\mathcal{R}\\lbrace g \\rbrace_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        txt = txt + "\cellcolor{gray!40} \\textbf{" + '{:.3f}'.format(RF_err[mi, Ni, 1]) + "} "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
            
    for Ni in range(len(NN)):
        txt = txt + "\cellcolor{gray!40} \\textit{" + '{:.2f}'.format(RF_tms[mi, Ni]) + "} s "
        if Ni < len(NN)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + "\\hline \n"

text_file = open("heat_plots/heat_usv_err_tms.txt", "w")
n = text_file.write(txt[:-12])
text_file.close()