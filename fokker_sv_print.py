import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'legend.fontsize': 11,
                     'axes.labelsize': 10,
                     'axes.titlesize': 12,
                     'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

t = 1.0
K1 = 100
tt = np.linspace(0.0, t, K1)
dt1 = t/K1

mm = [10, 20, 30]
NN = [16, 32, 64, 128, 256, 512, 1024]

DT_err = np.zeros([len(mm), len(NN), 2])
DT_tms = np.zeros([len(mm), len(NN)])
DT_u1p = np.zeros([len(mm), len(NN), 100])
RT_err = np.zeros([len(mm), len(NN), 2])
RT_tms = np.zeros([len(mm), len(NN)])
RT_u1p = np.zeros([len(mm), len(NN), 100])
for mi in range(len(mm)):
    DT_err[mi] = np.loadtxt("fokker_data/fokker_sv_DT_err_" + str(mm[mi]) + ".csv")
    DT_tms[mi] = np.loadtxt("fokker_data/fokker_sv_DT_tms_" + str(mm[mi]) + ".csv")
    DT_u1p[mi] = np.loadtxt("fokker_data/fokker_sv_DT_u1p_" + str(mm[mi]) + ".csv")
    RT_err[mi] = np.loadtxt("fokker_data/fokker_sv_RT_err_" + str(mm[mi]) + ".csv")
    RT_tms[mi] = np.loadtxt("fokker_data/fokker_sv_RT_tms_" + str(mm[mi]) + ".csv")
    RT_u1p[mi] = np.loadtxt("fokker_data/fokker_sv_RT_u1p_" + str(mm[mi]) + ".csv")
    
DN_err = np.zeros([len(mm), len(NN), 2])
DN_tms = np.zeros([len(mm), len(NN)])
DN_u1p = np.zeros([len(mm), len(NN), 100])
RN_err = np.zeros([len(mm), len(NN), 2])
RN_tms = np.zeros([len(mm), len(NN)])
RN_u1p = np.zeros([len(mm), len(NN), 100])
for mi in range(len(mm)):
    DN_err[mi] = np.loadtxt("fokker_data/fokker_sv_DN_err_" + str(mm[mi]) + ".csv")
    DN_tms[mi] = np.loadtxt("fokker_data/fokker_sv_DN_tms_" + str(mm[mi]) + ".csv")
    DN_u1p[mi] = np.loadtxt("fokker_data/fokker_sv_DN_u1p_" + str(mm[mi]) + ".csv")
    RN_err[mi] = np.loadtxt("fokker_data/fokker_sv_RN_err_" + str(mm[mi]) + ".csv")
    RN_tms[mi] = np.loadtxt("fokker_data/fokker_sv_RN_tms_" + str(mm[mi]) + ".csv")
    RN_u1p[mi] = np.loadtxt("fokker_data/fokker_sv_RN_u1p_" + str(mm[mi]) + ".csv")
    
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = cmap(np.arange(len(mm))/len(mm))

# Error Plot
fig = plt.figure(figsize = (5.4, 4.0))
for mi in range(len(mm)):
     plt.plot(NN, np.transpose(DT_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 5, 1, 5)), marker = "+")
     plt.plot(NN, np.transpose(RT_err[mi, :, 1]), color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None")
     plt.plot(NN, np.transpose(DN_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x")
     plt.plot(NN, np.transpose(RN_err[mi, :, 1]), color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None")
     
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5)), marker = "+", label = "$\\mathcal{T}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{RT}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", label = "$\\mathcal{N}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))
    
plt.legend(loc = "upper right", ncol = 3)
plt.ylim([0.0, 0.02])
plt.xlabel("Number of neurons $N$")
plt.ylabel("Empirical $L^2$-error on test set")
plt.savefig("fokker_plots/fokker_sv_err.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Function Plot
fig = plt.figure(figsize = (5.4, 4.0))
for mi in range(len(mm)):
    m = mm[mi]
    C1 = 0.1*np.eye(m, dtype = np.float32)/np.sqrt(m)
    C2 = 0.1*np.eye(m, dtype = np.float32)/np.sqrt(m)
    c2 = 0.1*np.ones([1, m], dtype = np.float32)/m
    D = 0.2*np.eye(m, dtype = np.float32)/np.sqrt(m)
    mu = np.zeros([K1+1, m], dtype = np.float32)
    Sigma = np.zeros([K1+1, m, m], dtype = np.float32)
    Sigma[0] = 0.5*np.eye(m, dtype = np.float32)/np.sqrt(m)
    for l in range(K1):
        mu[l+1] = mu[l] + (-np.matmul(C1, mu[l]) - c2)*dt1
        Sigma[l+1] = Sigma[l] + (-np.matmul(C1+C2, Sigma[l]) - np.matmul(Sigma[l], C1+C2) + 2*D)*dt1
        
    mut = mu[K1:]
    Sigmat = Sigma[K1]
    Sigmat1 = np.linalg.inv(Sigmat)
    
    u = 0.25*np.ones([100, m])
    u[:, 0] = np.linspace(-0.4, 0.4, 100)
    y = np.exp(-0.5*np.sum(np.matmul(u-mut, Sigmat1)*(u-mut), axis = -1, keepdims = True))/np.power(2.0*np.pi, m/2.0)/np.sqrt(np.linalg.det(Sigmat))
    plt.plot(u[:, 0], y, color = cols[mi], linestyle = "solid")
    
for mi in range(len(mm)):
    plt.plot(u[:, 0], DT_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 5, 1, 5)), marker = "+", markevery = slice(0, 500, 60))
    plt.plot(u[:, 0], RT_u1p[mi, -1], color = cols[mi], linestyle = "dotted", marker = "s", markerfacecolor = "None", markevery = slice(15, 500, 60))
    plt.plot(u[:, 0], DN_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", markevery = slice(30, 500, 60))
    plt.plot(u[:, 0], RN_u1p[mi, -1], color = cols[mi], linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", markevery = slice(45, 500, 60))

plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5)), marker = "+", label = "$\\mathcal{T}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "dotted", marker = "s", markerfacecolor = "None", label = "$\\mathcal{RT}_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "None", label = " ")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 5, 1, 5, 1, 5)), marker = "x", label = "$\\mathcal{N}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = (0, (3, 1, 1, 1)), marker = "D", markerfacecolor = "None", label = "$\\mathcal{RN}^\\rho_{\\mathbb{R}^m,1}$")
plt.plot(np.nan, np.nan, color = "black", linestyle = "solid", label = "True")

for mi in range(len(mm)):
    plt.plot(np.nan, np.nan, color = cols[mi], linestyle = "-", label = "m = " + str(mm[mi]))

plt.legend(loc = "upper right", ncol = 3)
plt.xlabel("$u_1$")
plt.ylabel("$f(1,(u_1,1/4,...,1/4))$")
plt.ylim([0.0, 0.05])
plt.savefig("fokker_plots/fokker_sv_u1p.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Error and Time Table
txt = ""
for mi in range(len(mm)):
    txt = txt + "\multirow{8}{*}{$m = " + str(mm[mi]) + "$} "
    
    # DT
    txt = txt + "& \multirow{2}{*}{$\\mathcal{T}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        txt = txt + "\\textbf{" + '{:.3f}'.format(DT_err[mi, Ni, 1]) + "} "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
    
    for Ni in range(len(NN)):
        txt = txt + "\\textit{" + '{:.2f}'.format(DT_tms[mi, Ni]) + "} s "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    # RT
    txt = txt + "& \multirow{2}{*}{$\\mathcal{RT}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        txt = txt + "\cellcolor{gray!40} \\textbf{" + '{:.3f}'.format(RT_err[mi, Ni, 1]) + "} "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
            
    for Ni in range(len(NN)):
        txt = txt + "\cellcolor{gray!40} \\textit{" + '{:.2f}'.format(RT_tms[mi, Ni]) + "} s "
        if Ni < len(NN)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    # DN
    txt = txt + "& \multirow{2}{*}{$\\mathcal{N}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        txt = txt + "\\textbf{" + '{:.3f}'.format(DN_err[mi, Ni, 1]) + "} "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
    
    for Ni in range(len(NN)):
        txt = txt + "\\textit{" + '{:.2f}'.format(DN_tms[mi, Ni]) + "} s "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    # RN
    txt = txt + "& \multirow{2}{*}{$\\mathcal{RN}_{\mathbb{R}^m,1}$} & "
    for Ni in range(len(NN)):
        txt = txt + "\cellcolor{gray!40} \\textbf{" + '{:.3f}'.format(RN_err[mi, Ni, 1]) + "} "
        if Ni < len(NN)-1:
            txt = txt + " & "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + " & & "
            
    for Ni in range(len(NN)):
        txt = txt + "\cellcolor{gray!40} \\textit{" + '{:.2f}'.format(RN_tms[mi, Ni]) + "} s "
        if Ni < len(NN)-1:
            txt = txt + "& "
        else:
            txt = txt + "\\\ \n"
            
    txt = txt + "\\hline \n"

text_file = open("fokker_plots/fokker_sv_err_tms.txt", "w")
n = text_file.write(txt[:-12])
text_file.close()