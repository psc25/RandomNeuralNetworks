import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'legend.fontsize': 10,
                     'axes.labelsize': 9,
                     'axes.titlesize': 12,
                     'xtick.labelsize': 8,
                     'ytick.labelsize': 8})

t = 1.0
lam = 4.0
sig = np.sqrt(2.0*lam*t)

mm = [10, 20, 30]
NNd = [10, 25, 50, 100]
NNr = [10, 25, 50, 100, 200, 400]

RT_det_err = np.zeros([len(mm), len(NNd), 2])
RT_det_tms = np.zeros([len(mm), len(NNd)])
RT_det_evl = np.zeros([len(mm), len(NNd)])
RT_det_u1p = np.zeros([len(mm), len(NNd), 500])
RT_rnd_err = np.zeros([len(mm), len(NNr), 2])
RT_rnd_tms = np.zeros([len(mm), len(NNr)])
RT_rnd_evl = np.zeros([len(mm), len(NNr)])
RT_rnd_u1p = np.zeros([len(mm), len(NNr), 500])
for mi in range(len(mm)):
    RT_det_err[mi] = np.loadtxt("heat_data/heat_RT_det_err_" + str(mm[mi]) + ".csv")
    RT_det_tms[mi] = np.loadtxt("heat_data/heat_RT_det_tms_" + str(mm[mi]) + ".csv")
    RT_det_evl[mi] = np.loadtxt("heat_data/heat_RT_det_evl_" + str(mm[mi]) + ".csv")
    RT_det_u1p[mi] = np.loadtxt("heat_data/heat_RT_det_u1p_" + str(mm[mi]) + ".csv")
    RT_rnd_err[mi] = np.loadtxt("heat_data/heat_RT_rnd_err_" + str(mm[mi]) + ".csv")
    RT_rnd_tms[mi] = np.loadtxt("heat_data/heat_RT_rnd_tms_" + str(mm[mi]) + ".csv")
    RT_rnd_evl[mi] = np.loadtxt("heat_data/heat_RT_rnd_evl_" + str(mm[mi]) + ".csv")
    RT_rnd_u1p[mi] = np.loadtxt("heat_data/heat_RT_rnd_u1p_" + str(mm[mi]) + ".csv")

RN_det_err = np.zeros([len(mm), len(NNd), 2])
RN_det_tms = np.zeros([len(mm), len(NNd)])
RN_det_evl = np.zeros([len(mm), len(NNd)])
RN_det_u1p = np.zeros([len(mm), len(NNd), 500])
RN_rnd_err = np.zeros([len(mm), len(NNr), 2])
RN_rnd_tms = np.zeros([len(mm), len(NNr)])
RN_rnd_evl = np.zeros([len(mm), len(NNr)])
RN_rnd_u1p = np.zeros([len(mm), len(NNr), 500])
for mi in range(len(mm)):
    RN_det_err[mi] = np.loadtxt("heat_data/heat_RN_det_err_" + str(mm[mi]) + ".csv")
    RN_det_tms[mi] = np.loadtxt("heat_data/heat_RN_det_tms_" + str(mm[mi]) + ".csv")
    RN_det_evl[mi] = np.loadtxt("heat_data/heat_RN_det_evl_" + str(mm[mi]) + ".csv")
    RN_det_u1p[mi] = np.loadtxt("heat_data/heat_RN_det_u1p_" + str(mm[mi]) + ".csv")
    RN_rnd_err[mi] = np.loadtxt("heat_data/heat_RN_rnd_err_" + str(mm[mi]) + ".csv")
    RN_rnd_tms[mi] = np.loadtxt("heat_data/heat_RN_rnd_tms_" + str(mm[mi]) + ".csv")
    RN_rnd_evl[mi] = np.loadtxt("heat_data/heat_RN_rnd_evl_" + str(mm[mi]) + ".csv")
    RN_rnd_u1p[mi] = np.loadtxt("heat_data/heat_RN_rnd_u1p_" + str(mm[mi]) + ".csv")
    
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = cmap(np.arange(len(mm))/len(mm))

# Error Plot
fig = plt.figure(figsize = (4.8, 3.4))
for mi in range(len(mm)):
     plt.plot(NNd, np.transpose(RT_det_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 5, 1, 5)), marker = "+")
     plt.plot(NNr, np.transpose(RT_rnd_err[mi, :, 1]), color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None")
     plt.plot(NNd, np.transpose(RN_det_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x")
     plt.plot(NNr, np.transpose(RN_rnd_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None")
     
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5)), marker = "+", label = "$\\mathcal{T}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{RT}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", label = "$\\mathcal{N}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))

plt.legend(loc = "upper right", ncol = 3)
plt.ylim([-0.002, 0.102])
plt.xlabel("Number of neurons $N$")
plt.ylabel("Empirical $L^2$-error on test set")
plt.savefig("heat_plots/heat_err.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Function Plot
fig = plt.figure(figsize = (6.4, 4.8))
for mi in range(len(mm)):
    m = mm[mi]
    u = 0.5*np.ones([500, m])
    u[:, 0] = np.linspace(-4.0, 4.0, 500)
    
    R = 4.0*np.power(m, 0.4)
    ncp = np.sum(np.square(u/sig), axis = -1, keepdims = True)
    y = scs.ncx2.cdf(np.square(R/sig), df = m, nc = ncp)
    plt.plot(u[:, 0], y, color = cols[mi], linestyle = "solid")
    
for mi in range(len(mm)):
    plt.plot(u[:, 0], RT_det_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 5, 1, 5)), marker = "+", markevery = slice(0, 500, 60))
    plt.plot(u[:, 0], RT_rnd_u1p[mi, -1], color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None", markevery = slice(15, 500, 60))
    plt.plot(u[:, 0], RN_det_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", markevery = slice(30, 500, 60))
    plt.plot(u[:, 0], RN_rnd_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", markevery = slice(45, 500, 60))

plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5)), marker = "+", label = "$\\mathcal{T}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{RT}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", label = "$\\mathcal{N}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "black", linestyle = "solid", label = "True")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "None", label = " ")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))

plt.legend(loc = "upper right", ncol = 4)
plt.xlabel("$u_1$")
plt.ylabel("$f(1,(u_1,0.5,...,0.5))$")
plt.ylim([0.398, 0.902])
plt.savefig("heat_plots/heat_u1p.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Time and Evaluation Table
txt = ""
for mi in range(len(mm)):
    txt = txt + "\multirow{8}{*}{$m = " + str(mm[mi]) + "$} "
    
    # Det Trigo
    txt = txt + "& \multirow{2}{*}{$\\mathcal{T}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NNr)):
        if Ni < len(NNd):
            txt = txt + "\\textit{" + '{:.2f}'.format(RT_det_tms[mi, Ni]) + "} & "
        elif Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + "& & "
    for Ni in range(len(NNr)):
        if Ni < len(NNd):
            if RT_det_evl[mi, Ni] > 0:
                e1 = np.floor(np.log10(RT_det_evl[mi, Ni]))
                r1 = RT_det_evl[mi, Ni]/np.power(10, e1)
            else:
                e1 = 0
                r1 = 0
        
            txt = txt + "$" + '{:.1f}'.format(r1) + "\\cdot 10^{" + '{:.0f}'.format(e1) + "}$ & "
        elif Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    # Rand Trigo  
    txt = txt + "& \cellcolor{Gray} \multirow{2}{*}{$\\mathcal{RT}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NNr)):
        txt = txt + "\cellcolor{Gray} \\textit{" + '{:.2f}'.format(RT_rnd_tms[mi, Ni]) + "} "
        if Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
    
    txt = txt + "& \cellcolor{Gray} & "    
    for Ni in range(len(NNr)):
        if RT_rnd_evl[mi, Ni] > 0:
            e1 = np.floor(np.log10(RT_rnd_evl[mi, Ni]))
            r1 = RT_rnd_evl[mi, Ni]/np.power(10, e1)
        else:
            e1 = 0
            r1 = 0
            
        txt = txt + "\cellcolor{Gray} $" + '{:.1f}'.format(r1) + "\\cdot 10^{" + '{:.0f}'.format(e1) + "}$ "
        if Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    # Det NN
    txt = txt + "& \multirow{2}{*}{$\\mathcal{N}^{\\tanh}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NNr)):
        if Ni < len(NNd):
            txt = txt + "\\textit{" + '{:.2f}'.format(RN_det_tms[mi, Ni]) + "} & "
        elif Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + "& & "
    for Ni in range(len(NNr)):
        if Ni < len(NNd):
            if RN_det_evl[mi, Ni] > 0:
                e1 = np.floor(np.log10(RN_det_evl[mi, Ni]))
                r1 = RN_det_evl[mi, Ni]/np.power(10, e1)
            else:
                e1 = 0
                r1 = 0
        
            txt = txt + "$" + '{:.1f}'.format(r1) + "\\cdot 10^{" + '{:.0f}'.format(e1) + "}$ & "
        elif Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    # Rand NN
    txt = txt + "& \cellcolor{Gray} \multirow{2}{*}{$\\mathcal{RN}^{\\tanh}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NNr)):
        txt = txt + "\cellcolor{Gray} \\textit{" + '{:.2f}'.format(RN_rnd_tms[mi, Ni]) + "} "
        if Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
    
    txt = txt + "& \cellcolor{Gray} & "    
    for Ni in range(len(NNr)):
        if RN_rnd_evl[mi, Ni] > 0:
            e1 = np.floor(np.log10(RN_rnd_evl[mi, Ni]))
            r1 = RN_rnd_evl[mi, Ni]/np.power(10, e1)
        else:
            e1 = 0
            r1 = 0
            
        txt = txt + "\cellcolor{Gray} $" + '{:.1f}'.format(r1) + "\\cdot 10^{" + '{:.0f}'.format(e1) + "}$ "
        if Ni < len(NNr)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + "\\hline \n"

text_file = open("heat_plots/heat_tms_evl.txt", "w")
n = text_file.write(txt[:-12])
text_file.close()