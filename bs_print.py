import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'font.size': 11})

T = 1.0
r = 0.01
sigma = 0.5
K = 82.0

mm = [10, 20, 30]
NNd = [10, 50, 100, 200]
NNr = [10, 50, 100, 200, 300, 400]

det_err = np.zeros([len(mm), len(NNd), 2])
det_tms = np.zeros([len(mm), len(NNd)])
det_evl = np.zeros([len(mm), len(NNd)])
det_u1p = np.zeros([len(mm), len(NNd), 500])
rnd_err = np.zeros([len(mm), len(NNr), 2])
rnd_tms = np.zeros([len(mm), len(NNr)])
rnd_evl = np.zeros([len(mm), len(NNr)])
rnd_u1p = np.zeros([len(mm), len(NNr), 500])
for mi in range(len(mm)):
    det_err[mi] = np.loadtxt("bs_data/bs_det_err_" + str(mm[mi]) + ".csv")
    det_tms[mi] = np.loadtxt("bs_data/bs_det_tms_" + str(mm[mi]) + ".csv")
    det_evl[mi] = np.loadtxt("bs_data/bs_det_evl_" + str(mm[mi]) + ".csv")
    det_u1p[mi] = np.loadtxt("bs_data/bs_det_u1p_" + str(mm[mi]) + ".csv")
    rnd_err[mi] = np.loadtxt("bs_data/bs_rnd_err_" + str(mm[mi]) + ".csv")
    rnd_tms[mi] = np.loadtxt("bs_data/bs_rnd_tms_" + str(mm[mi]) + ".csv")
    rnd_evl[mi] = np.loadtxt("bs_data/bs_rnd_evl_" + str(mm[mi]) + ".csv")
    rnd_u1p[mi] = np.loadtxt("bs_data/bs_rnd_u1p_" + str(mm[mi]) + ".csv")
    
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = cmap(np.arange(len(mm))/len(mm))

# Error Plot
fig = plt.figure()
for mi in range(len(mm)):
     plt.plot(NNd, np.transpose(det_err[mi, :, 1]), color = cols[mi], linestyle = "dotted", marker = "x")
     plt.plot(NNr, np.transpose(rnd_err[mi, :, 1]), color = cols[mi], linestyle = "dashed", marker = "o", markerfacecolor = "None")
     
plt.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "x", label = "$\\mathcal{NN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "dashed", marker = "o", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))

plt.legend(loc = "upper right", ncol = 2)
plt.xlabel("Number of neurons $N$")
plt.ylabel("Empirical $L^2$-error on test set")
plt.savefig("bs_plots/bs_err.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Function Plot
fig = plt.figure()
for mi in range(len(mm)):
    u = np.log(K)*np.ones([500, mm[mi]])
    u[:, 0] = np.linspace(4.0, 5.0, 500)
    S_avg = np.exp(np.mean(u, axis = 1, keepdims = True))
    mu1 = (r + sigma**2/2.0)*T
    sigma1 = sigma/np.sqrt(mm[mi])
    d1 = (np.log(S_avg/K) + mu1)/(sigma1*np.sqrt(T))
    d2 = d1 - sigma1*np.sqrt(T)
    y = S_avg*scs.norm.cdf(d1) - K*np.exp(-r*T)*scs.norm.cdf(d2)
    
    plt.plot(np.exp(u[:, 0]), y, color = cols[mi], linestyle = "solid")
    
for mi in range(len(mm)):
    plt.plot(np.exp(u[:, 0]), det_u1p[mi, -1], color = cols[mi], linestyle = "dotted", marker = "x", markevery = 50)
    plt.plot(np.exp(u[:, 0]), rnd_u1p[mi, -1], color = cols[mi], linestyle = "dashed", marker = "o", markerfacecolor = "None", markevery = 50)

plt.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "x", label = "$\\mathcal{NN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "dashed", marker = "o", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "black", linestyle = "solid", label = "True")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "None", label = " ")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))

plt.legend(loc = "upper right", ncol = 3)
plt.xlabel("$x_1 = \exp(u_1)$")
plt.ylabel("Price")
plt.ylim([0.0, 12.5])
plt.savefig("bs_plots/bs_u1p.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Time and Evaluation Table
txt = ""
for Ni in range(len(NNr)):
    txt = txt + "\multirow{2}{*}{$N = " + str(NNr[Ni]) + "$} & "
    for mi in range(len(mm)):
        if Ni < len(NNd):
            dettms = "\\textit{" + '{:.2f}'.format(det_tms[mi, Ni]) + "}"
        else:
            dettms = ""
            
        if mi+1 < len(mm):
            txt = txt + dettms + " & \cellcolor{Gray} \\textit{" + '{:.2f}'.format(rnd_tms[mi, Ni]) + "} & "
        else:
            txt = txt + dettms + " & \cellcolor{Gray} \\textit{" + '{:.2f}'.format(rnd_tms[mi, Ni]) + "} \\\ \n "
            
    txt = txt + " & "
    for mi in range(len(mm)):
        if Ni < len(NNd):
            if det_evl[mi, Ni] > 0:
                e1 = np.floor(np.log10(det_evl[mi, Ni]))
                r1 = det_evl[mi, Ni]/np.power(10, e1)
            else:
                e1 = 0
                r1 = 0
            
            txt = txt +  " $" + '{:.2f}'.format(r1) + " \\cdot 10^{" + '{:.0f}'.format(e1) + "}$ & "
        else:
            txt = txt + " & "
            
        if rnd_evl[mi, Ni] > 0:
            e2 = np.floor(np.log10(rnd_evl[mi, Ni]))
            r2 = rnd_evl[mi, Ni]/np.power(10, e2)
        else:
            e2 = 0
            r2 = 0
            
        if mi+1 < len(mm):
            txt = txt + "\cellcolor{Gray} $" + '{:.2f}'.format(r2) + " \\cdot 10^{" + '{:.0f}'.format(e2) + "}$ & "
        else:
            txt = txt + "\cellcolor{Gray} $" + '{:.2f}'.format(r2) + " \\cdot 10^{" + '{:.0f}'.format(e2) + "}$ \\\ \n"
            
    txt = txt + "\hline \n"

text_file = open("bs_plots/bs_tms_evl.txt", "w")
n = text_file.write(txt[:-13])
text_file.close()