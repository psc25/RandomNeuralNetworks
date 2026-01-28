import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'legend.fontsize': 11,
                     'axes.labelsize': 10,
                     'axes.titlesize': 12,
                     'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

T = 1.0
K = 20
K1 = 100
dt1 = T/K1

tt = np.reshape(np.linspace(0.0, T, K+1), [-1, 1, 1])

mm = [10, 20, 30]
NN = [16, 32, 64, 128, 256]
    
DF_err = np.zeros([len(mm), len(NN), 2])
DF_tms = np.zeros([len(mm), len(NN)])
DF_u1p = np.zeros([len(mm), len(NN), K+1, 100])
RF_err = np.zeros([len(mm), len(NN), 2])
RF_tms = np.zeros([len(mm), len(NN)])
RF_u1p = np.zeros([len(mm), len(NN), K+1, 100])
for mi in range(len(mm)):
    DF_err[mi] = np.loadtxt("fokker_data/fokker_usv_DF_err_" + str(mm[mi]) + ".csv")
    DF_tms[mi] = np.loadtxt("fokker_data/fokker_usv_DF_tms_" + str(mm[mi]) + ".csv")
    DF_u1p[mi] = np.reshape(np.loadtxt("fokker_data/fokker_usv_DF_u1p_" + str(mm[mi]) + ".csv"), [len(NN), K+1, 100])
    RF_err[mi] = np.loadtxt("fokker_data/fokker_usv_RF_err_" + str(mm[mi]) + ".csv")
    RF_tms[mi] = np.loadtxt("fokker_data/fokker_usv_RF_tms_" + str(mm[mi]) + ".csv")
    RF_u1p[mi] = np.reshape(np.loadtxt("fokker_data/fokker_usv_RF_u1p_" + str(mm[mi]) + ".csv"), [len(NN), K+1, 100])
    
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = cmap(np.arange(len(mm))/len(mm))

# Error Plot
fig = plt.figure(figsize = (5.4, 4.0))
for mi in range(len(mm)):
     plt.plot(NN, np.transpose(DF_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x")
     plt.plot(NN, np.transpose(RF_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None")
     
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", label = "$\\lbrace g \\rbrace_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{R}\\lbrace g \\rbrace_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))
    
plt.legend(loc = "upper right", ncol = 2)
plt.ylim([-0.001, 0.06])
plt.xlabel("Number of neurons $N$")
plt.ylabel("Empirical $L^2$-error on test set")
plt.savefig("fokker_plots/fokker_usv_err.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Function Plots
for mi in range(len(mm)):
    m = mm[mi]
    C1 = 0.1*np.expand_dims(np.eye(m, dtype = np.float32), 0)/np.sqrt(m)
    C2 = 0.1*np.expand_dims(np.eye(m, dtype = np.float32), 0)/np.sqrt(m)
    c2 = np.zeros([1, m], dtype = np.float32)/m
    D = 0.2*np.expand_dims(np.eye(m, dtype = np.float32), 0)/np.sqrt(m)
    mu = 0.1*np.zeros([K1+1, m], dtype = np.float32)
    Sigma = np.zeros([K1+1, m, m], dtype = np.float32)
    Sigma[0] = 0.5*np.eye(m, dtype = np.float32)/np.sqrt(m)
    for l in range(K1):
        mu[l+1] = mu[l] + (-np.matmul(C1[0], mu[l]) - c2)*dt1
        Sigma[l+1] = Sigma[l] + (-np.matmul(C1[0]+C2[0], Sigma[l]) - np.matmul(Sigma[l], C1[0]+C2[0]) + 2*D[0])*dt1
        
    mut = np.expand_dims(mu[::int(K1/K)], axis = 1)
    Sigmat = Sigma[::int(K1/K)]
    Sigmat1 = np.linalg.inv(Sigmat)
    
    u = 0.25*np.ones([1, 100, m])
    u1 = np.linspace(-0.4, 0.4, 100)
    u[0, :, 0] = u1
    y = np.exp(-0.5*np.sum(np.matmul(u-mut, Sigmat1)*(u-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.reshape(np.sqrt(np.linalg.det(Sigmat)), [-1, 1, 1])
    
    fig = plt.figure(figsize = (5.4, 4.0))
    ax = fig.add_subplot(projection = '3d')
    uu_plot, tt_plot = np.meshgrid(u1, tt.flatten())
    ax.plot_surface(tt_plot, uu_plot, DF_u1p[mi, -1], cmap = plt.cm.Blues_r, alpha = 0.7, linewidth = 0, antialiased = False)
    ax.plot_surface(tt_plot, uu_plot, RF_u1p[mi, -1], cmap = plt.cm.Reds_r, alpha = 0.7, linewidth = 0, antialiased = False)
    ax.plot_wireframe(tt_plot, uu_plot, y[:, :, 0], rstride = 3, cstride = 4, color = "black", linewidth = 0.4)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$u_1$')
    ax.set_zlabel('$f(t,(u_1,1/4,...,1/4))$')
    ax.plot(np.nan, np.nan, np.nan, label = "True", marker = r"$\#$", linestyle = 'None', markerfacecolor = "black", markeredgecolor = "black")
    ax.plot(np.nan, np.nan, np.nan, label = " ", linestyle = 'None')
    ax.plot(np.nan, np.nan, np.nan, label = "$\\lbrace g \\rbrace_{\mathbb{R}^m,1}$", linestyle = 'None', marker = "s", markerfacecolor = plt.cm.Blues(0.9), markeredgecolor = plt.cm.Blues(0.9))
    ax.plot(np.nan, np.nan, np.nan, label = "$\\mathcal{R}\\lbrace g \\rbrace_{\mathbb{R}^m,1}$", linestyle = 'None', marker = "s", markerfacecolor = plt.cm.Reds(0.9), markeredgecolor = plt.cm.Reds(0.9))
    ax.legend(loc = "upper right", ncol = 2)
    plt.savefig("fokker_plots/fokker_usv_u1p_" + str(m) + ".png", dpi = 500)
    plt.show()
    plt.close(fig)

# Error and Time Table
txt = ""
for mi in range(len(mm)):
    txt = txt + "\multirow{4}{*}{$m = " + str(mm[mi]) + "$} "
            
    # DF
    txt = txt + "& \multirow{2}{*}{$\\lbrace g \\rbrace_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        f1 = np.nanmean(DF_err[mi, Ni, 1])
        if f1 > 0:
            e1 = np.floor(np.log10(f1))
            r1 = f1/np.power(10, e1)
        else:
            e1 = 0
            r1 = 0
            
        txt = txt + "\\textbf{" + '{:.3f}'.format(r1) + " $\\cdot$ 10$^{\\text{" + '{:.0f}'.format(e1) + "}}$} "
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
        f1 = np.nanmean(RF_err[mi, Ni, 1])
        if f1 > 0:
            e1 = np.floor(np.log10(f1))
            r1 = f1/np.power(10, e1)
        else:
            e1 = 0
            r1 = 0
            
        txt = txt + "\cellcolor{gray!40} \\textbf{" + '{:.3f}'.format(r1) + " $\\cdot$ 10$^{\\text{" + '{:.0f}'.format(e1) + "}}$} "
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

text_file = open("fokker_plots/fokker_usv_err_tms.txt", "w")
n = text_file.write(txt[:-12])
text_file.close()