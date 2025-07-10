import tkinter as tk
from tkinter.font import Font
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename
from os import path
import pandas as pd
import numpy as np
from math import sqrt
import scipy.stats as s

def d2_d3_d2b(n_ite, n):
    A = np.random.normal(0, 1, (n_ite, n))
    r= np.zeros((n_ite))
    for i in range(n_ite):
        r[i] = (max(A[i,:])-min(A[i,:]))
    d2 = np.mean(r)
    d3 = np.std(r)
    return (d2, d3)

def ANOVA_Y(Y:list):
    Y = np.array(Y)
    
    r = Y.shape[0]
    p = Y.shape[1]
    a = Y.shape[2]

    Yij_ = np.sum(Y, axis=0)

    Y___ = np.sum(Y)
    Yi__ = np.sum(np.sum(Y, axis=0), axis=-1)
    Y_j_ = np.sum(np.sum(Y, axis=0), axis=0)
    Y__k = np.sum(np.sum(Y, axis=-1), axis=-1)
    Yij_ = np.sum(Y, axis=2)

    S_O = np.sum(np.power(Yi__, 2))/(a*r) - Y___**2/(a*r*p)
    S_S = np.sum(np.power(Y__k, 2))/(p*a) - Y___**2/(a*r*p)
    S_OS = np.sum(np.power(Yij_, 2))/a \
        - S_O - S_S - Y___**2/(a*r*p)
    SST = np.sum(np.power(Y, 2)) - Y___**2/(a*r*p)
    S_E = SST - S_O - S_S - S_OS

    deg_O = (p-1)
    deg_S = (r-1)
    deg_OS = (p-1)*(r-1)
    deg_E = p*r*(a-1)

    MS_O = S_O/deg_O # MSA
    MS_S = S_S/deg_S # MSB
    MS_OS = S_OS/deg_OS
    MS_E = S_E/deg_E

    F_O = MS_O/MS_E
    F_S = MS_S/MS_E
    F_OS = MS_OS/MS_E

    F_theo_O = s.f.ppf(q=1-0.05, dfn=deg_O, dfd=deg_E)
    F_theo_S = s.f.ppf(q=1-0.05, dfn=deg_S, dfd=deg_E)
    F_theo_OS = s.f.ppf(q=1-0.05, dfn=deg_OS, dfd=deg_E)

    if F_O < F_theo_O:
        S_E += S_O
        deg_E += deg_O
        S_O = 0
    if F_S < F_theo_S:
        S_E += S_S
        deg_E += deg_S
        S_S = 0
    if F_OS < F_theo_OS:
        S_E += S_OS
        deg_E += deg_OS
        S_OS = 0

    MS_O = S_O/deg_O # MSA
    MS_S = S_S/deg_S # MSB
    MS_OS = S_OS/deg_OS
    MS_E = S_E/deg_E

    sigma_S_2 = (MS_S-MS_OS)/(p*a)
    if sigma_S_2 < 0:
        sigma_S_2 = 0
    sigma_O_2 = (MS_O-MS_OS)/(a*r)
    if sigma_O_2 < 0:
        sigma_O_2 = 0
    sigma_OS_2 = (MS_OS-MS_E)/(a)
    if sigma_OS_2 < 0:
        sigma_OS_2 = 0
    sigma_E_2 = MS_E
    if sigma_E_2 < 0:
        sigma_E_2 = 0

    return [sqrt(sigma_E_2), sqrt(sigma_O_2), sqrt(sigma_OS_2), sqrt(sigma_S_2)] 

class class_fn:
    def __init__(self):
        # Variables
        self._L_donnee = []

        self._fn = tk.Tk()

        # Variable
        self._var_gauge = tk.StringVar(self._fn, value=0)
        self._var_nb_it = tk.IntVar(self._fn, value = 10_000_000)
        self._var_USL = tk.DoubleVar(self._fn, value = 1)
        self._var_LSL = tk.DoubleVar(self._fn, value = 0)

        self._police =  Font(family = "Time", size = 9)
        self._police_g =  Font(family = "Time", size = 13)
        self._fn.title('Non parametric ANOVA')

        label_donnees = tk.LabelFrame(self._fn, text="Parameter:")
        label_donnees.grid(row=0, column=0)
        self._parametre(label_donnees)

        label_donnees = tk.LabelFrame(self._fn, text="Data:")
        label_donnees.grid(row=1, column=0)
        self._xy_scroll_label(label_donnees)

        label_donnees = tk.LabelFrame(self._fn, text="")
        label_donnees.grid(row=2, column=0)
        self._parametre_gauge(label_donnees)        
        
        self._fn.mainloop()

    def _xy_scroll_label(self, frame, lign:int=0, col:int=0):
        #frame.grid_propagate(False)
        vscrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        vscrollbar.grid(row=lign, column=col+1, sticky='ns')

        hscrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        hscrollbar.grid(row=lign+1, column=col, sticky='we')

        self._canvas = tk.Canvas(frame, width=500, height=200,
                           yscrollcommand=vscrollbar.set,
                           xscrollcommand=hscrollbar.set)
        self._canvas.grid(row=lign, column=col)

        vscrollbar.config(command=self._canvas.yview)
        hscrollbar.config(command=self._canvas.xview)

        self._canvas.xview_moveto(0)
        self._canvas.yview_moveto(0)

        self._frame = tk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self._frame,
                                           anchor=tk.NW)

    def _parametre(self, frame, lign:int=0, col:int=0):
        self._var_op = tk.IntVar(self._fn, value =3)
        self._var_piec = tk.IntVar(self._fn, value =10)
        self._var_app = tk.IntVar(self._fn, value =3)
        
        cdr = tk.Frame(frame)
        cdr.grid(row=lign, column = col)
        lb = tk.Label(cdr, text = "Number of operators: ")

        lb.grid(row=0, column=0 )
        sp = tk.Spinbox(cdr, from_=1, to=99, width=2,
                         textvariable = self._var_op)
        sp.grid(row=0, column=1)

        cdr = tk.Frame(frame)
        cdr.grid(row=lign, column = col+1)

        lb = tk.Label(cdr, text = "Number of measures per series: ")
        lb.grid(row=0, column=0 ) 
        sp = tk.Spinbox(cdr, from_=1, to=99, width=2,
                         textvariable = self._var_app)
        sp.grid(row=0, column=1)

        cdr = tk.Frame(frame)
        cdr.grid(row=lign, column = col+2)

        lb = tk.Label(cdr, text = "Number of series: ")
        lb.grid(row=0, column=0 ) 
        sp = tk.Spinbox(cdr, from_=1, to=99, width=2,
                         textvariable = self._var_piec)
        sp.grid(row=0, column=1)

        bout = tk.Button(frame, text="Open",
                          command=self._command_ouvrir)
        bout.grid(row=lign, column=col+3)

    def _parametre_gauge(self, frame, lign:int=0, col:int=0):
        r1 = tk.Radiobutton(frame, text='Bootstrap', value=0, variable=self._var_gauge)
        r1.grid(row=lign, column=col)
        
        r1 = tk.Radiobutton(frame, text='Range method', value=1, variable=self._var_gauge)
        r1.grid(row=lign, column=col+1)

        lab = tk.Label(frame, text="Iteration")
        lab.grid(row=lign, column=col+2)

        lab = tk.Entry(frame, width=12,
                         textvariable = self._var_nb_it)
        lab.grid(row=lign, column=col+3)

        lab = tk.Label(frame, text="USL")
        lab.grid(row=lign, column=col+4)

        lab = tk.Entry(frame, width=12,
                         textvariable = self._var_USL)
        lab.grid(row=lign, column=col+5)

        lab = tk.Label(frame, text="LSL")
        lab.grid(row=lign, column=col+6)

        lab = tk.Entry(frame, width=12,
                         textvariable = self._var_LSL)
        lab.grid(row=lign, column=col+7)

        bout = tk.Button(frame, text="Calculation", command=self._calcul_RR)
        bout.grid(row=lign+1, column=col, columnspan=8)

    def _calcul_RR(self):
        if int(self._var_gauge.get()) == 0:
            self._bootstrap()
        else :
            self._range_method()

    def _range_method(self):
        n = len(self._L_donnee)
        if n > 0:
            o = len(self._L_donnee[0])
            if o > 0 :
                p = len(self._L_donnee[0][0])
                if p > 0 :
                    tab = np.array([[[float(self._L_donnee[i][j][k].get()) \
                                    for k in range(len(self._L_donnee[0][0]))] \
                                    for j in range(len(self._L_donnee[0]))] \
                                        for i in range(len(self._L_donnee))]).reshape(\
                                            np.array(self._L_donnee).shape)

                    R_p = np.mean(np.max(tab, axis=0)-np.min(tab, axis=0))
                    R_R = np.max(np.mean(np.mean(tab,axis=0),axis=-1))-np.min(np.mean(np.mean(tab,axis=0),axis=-1))
                    R_r = np.mean(np.max(tab, axis=-1)-np.min(tab, axis=-1))

                    (d_2_r, d_3_r) = d2_d3_d2b(int(self._var_nb_it.get()), p)
                    d_2_s_r = np.sqrt(d_2_r**2 + (d_3_r**2)/(o*n))
                    D_4_r = (1+3*(d_3_r/d_2_r))                    

                    (d_2_R, d_3_R) = d2_d3_d2b(int(self._var_nb_it.get()), o)
                    d_2_s_R = np.sqrt(d_2_R**2 + d_3_R**2)

                    (d_2_p, d_3_p) = d2_d3_d2b(int(self._var_nb_it.get()), n)
                    d_2_s_p = np.sqrt(d_2_p**2 + d_3_p**2)

                    cond = False
                    for elmt in np.mean(np.max(tab, axis=-1)-np.min(tab, axis=-1), axis=0):
                        if elmt > D_4_r*R_r:
                            cond = True
                            break
                    if cond:
                        showerror("Error", "Problem: the subgroup range exceeed the upper range limit")

                    nb_arrondi = int(0.5*(len(str(int(self._var_nb_it.get())))-1))
                    fn = tk.Toplevel(self._fn)
                    fn.title("Result")
                    L_sigma_r = [R_r/d_2_r, R_r/d_2_s_r]

                    text = "Repeatability (σr): \n- Rr/d2: "+str(round(L_sigma_r[0], nb_arrondi)) \
                            +"\n- Rr/d2* : "+str(round(L_sigma_r[1], nb_arrondi))
                    tk.Label(fn, text = text).grid(row=0, column=0)

                    if (R_R/d_2_s_R)**2 > (L_sigma_r[0]**2)/(n*p) and \
                            (R_R/d_2_s_R)**2 > (L_sigma_r[1]**2)/(n*p):
                        L_sigma_R = [R_R/d_2_R, R_R/d_2_s_R, 
                                    np.sqrt((R_R/d_2_s_R)**2 - (L_sigma_r[0]**2)/(n*p)),
                                    np.sqrt((R_R/d_2_s_R)**2 - (L_sigma_r[1]**2)/(n*p))]
                        text = "Reproducibility (σR): \n- RR/d2: "+str(round(L_sigma_R[0], nb_arrondi))\
                                + "\n- RR/d2* : "+str(round(L_sigma_R[1], nb_arrondi))\
                                + "\n- √(RR/d2*)² - σr²/np: "+str(round(L_sigma_R[2], nb_arrondi)) \
                                + "\n- √(RR/d2*)² - σr*²/np: "+str(round(L_sigma_R[3],  nb_arrondi))
                        tk.Label(fn, text = text).grid(row=0, column=1) 
                    else:
                        L_sigma_R = [R_R/d_2_R, R_R/d_2_s_R, R_R/d_2_s_R, R_R/d_2_s_R]
                        text = "Reproducibility (σR): \n- RR/d2: "+str(round(L_sigma_R[0], nb_arrondi))\
                                + "\n- RR/d2*: "+str(round(L_sigma_R[1], nb_arrondi))
                        tk.Label(fn, text = text).grid(row=0, column=1) 

                    if (R_p/d_2_s_p)**2 > (L_sigma_r[0]**2)/(n*o) and\
                        (R_p/d_2_s_p)**2 > (L_sigma_r[1]**2)/(n*o):
                        L_sigma_p = [R_p/d_2_p, R_p/d_2_s_p, 
                                    np.sqrt((R_p/d_2_s_p)**2 - (L_sigma_r[0]**2)/(n*o)),
                                    np.sqrt((R_p/d_2_s_p)**2 - (L_sigma_r[1]**2)/(n*o))]

                        text = "Product (σp): \n- Rp/d2: "+str(round(L_sigma_p[0], nb_arrondi))\
                                + "\n- Rp/d2*: "+str(round(L_sigma_p[1], nb_arrondi))\
                                + "\n- √(Rp/d2*)² - σr²/no: "+str(round(L_sigma_p[2], nb_arrondi))\
                                + "\n- √(Rp/d2*)² - σr*²/no: "+str(round(L_sigma_p[3], nb_arrondi))
                        tk.Label(fn, text = text).grid(row=0, column=2) 
                    else:
                        L_sigma_p = [R_p/d_2_p, R_p/d_2_s_p, R_p/d_2_s_p, R_p/d_2_s_p]

                        text = "Product (σp): \n- Rp/d2: "+str(round(L_sigma_p[0], nb_arrondi))\
                                + "\n- Rp/d2*: "+str(round(L_sigma_p[1], nb_arrondi))
                        tk.Label(fn, text = text).grid(row=0, column=2) 

                    sigma_RR_2 = L_sigma_r[1]**2 + L_sigma_R[3]**2
                    sigma_Var_2 = sigma_RR_2 + L_sigma_p[3]**2
                    
                    text = "Repeatability and Reproducibility (σrR): "+str(round(np.sqrt(sigma_RR_2), nb_arrondi))\
                            +"\nTotal standard deviation (σvar): "+str(round(np.sqrt(sigma_Var_2), nb_arrondi))

                    L_prop = [(L_sigma_r[1]**2)/sigma_Var_2,
                            (L_sigma_R[3]**2)/sigma_Var_2,
                            sigma_RR_2/sigma_Var_2,
                            (L_sigma_p[1]**2)/sigma_Var_2]
                    Att = 1- np.sqrt(L_prop[3])
                    text += "\n\nRepeatability Proportion of Total Variation (%): "+str(round(100*L_prop[0], 2))\
                            +"\nReproducibility Proportion of Total Variation (%): "+str(round(100*L_prop[1], 2))\
                            +"\nCombined Repeatability and Reproducibility Proportion of Total Variation (%): "+str(round(100*L_prop[2], 2))\
                            +"\nThat proportion of the Total Variance that is consumed by Product Variation (%): "+str(round(100*L_prop[3], 2))\
                            +"\n\nIntraclass Correlation Coefficient: "+str(round(L_prop[3], nb_arrondi+2))\
                            +"\nProduction process signals will be attenuated by: "+str(round(100*Att, nb_arrondi))+" %"

                    C = (float(self._var_USL.get())-float(self._var_LSL.get()))/(6*L_sigma_r[1])
                    L_C = [C*np.sqrt(1-.8), C*np.sqrt(1-.5), C*np.sqrt(1-.2)] 

                    text += "\n\nCan track process improvement up to Cp80: "+str(round(L_C[0], nb_arrondi))+ " while a First Class Monitor."\
                            "\nCan track process improvement up to Cp50: "+str(round(L_C[1], nb_arrondi))+ " while a Second Class Monitor."\
                            "\nCan track process improvement up to Cp20: "+str(round(L_C[2], nb_arrondi))+ " while a Third Class Monitor."
                    PE = 0.675*np.sqrt(L_sigma_r[1]**2)

                    text += "\n\nThe Probable Error of a single measurement is: "+str(round(PE, nb_arrondi))\
                            +"\nThe Smallest Effective Measurement Increment is: "+str(round(0.2*PE, nb_arrondi))\
                            +"\nThe Largest Effective Measurement Increment is: "+str(round(2*PE, nb_arrondi))\
                            +"\nThe Specifications Limits are "+str(round(self._var_LSL.get(), nb_arrondi))+" and "+str(round(self._var_USL.get(), nb_arrondi))\
                            +"\nThe Watershed Specifications are "+str(round(float(self._var_LSL.get())-0.2*PE, nb_arrondi))+" and "\
                            +str(round(float(self._var_USL.get())+0.2*PE, nb_arrondi))\
                            +"\n96% Manufacturing Specifications are thus "+str(round(float(self._var_LSL.get())+2*PE, nb_arrondi))+" to "\
                            +str(round(float(self._var_USL.get())-2*PE, nb_arrondi))
                    tk.Label(fn, text = text, justify="left").grid(row=1, column=0, columnspan=3) 
                    fn.mainloop()

    def _bootstrap(self):
        n = len(self._L_donnee)
        if n > 0:
            o = len(self._L_donnee[0])
            if o > 0 :
                p = len(self._L_donnee[0][0])
                if p > 0 :
                    Y = np.array([[[float(self._L_donnee[i][j][k].get()) \
                                    for k in range(len(self._L_donnee[0][0]))] \
                                    for j in range(len(self._L_donnee[0]))] \
                                        for i in range(len(self._L_donnee))]).reshape(\
                                            np.array(self._L_donnee).shape)

                    [sigma_r, sigma_R, sigma_OS, sigma_S] = ANOVA_Y(Y)

                    L_sigma_r = []
                    L_sigma_R = []
                    L_sigma_OS = []
                    L_sigma_S = []
                    Y = np.array(Y)
                    r = Y.shape[0] # k
                    p = Y.shape[1] # i
                    a = Y.shape[2] # j
                    
                    res = np.zeros(Y.shape)
                    for r in range(Y.shape[0]):
                        for i in range(Y.shape[1]):
                            res[r,i,:] = Y[r,i,:]-np.mean(Y[r,i,:])

                    nb = 1000
                    r = Y.shape[0] # k
                    p = Y.shape[1] # i
                    a = Y.shape[2] # j

                    for it in range(nb): 
                        Y_theo =  np.random.permutation(res.reshape(-1)).reshape(Y.shape)
                        for r in range(Y.shape[0]):
                            for i in range(Y.shape[1]):
                                Y_theo[r,i,:] += np.mean(Y[r,i,:])
                        L = ANOVA_Y(Y_theo)
                        L_sigma_r.append(L[0]**2)
                        L_sigma_R.append(L[1]**2)
                        L_sigma_OS.append(L[2]**2)
                        L_sigma_S.append(L[3]**2)

                    alpha = 0.05

                    L_sigma_r = np.sort(L_sigma_r)
                    L_sigma_R = np.sort(L_sigma_R)
                    L_sigma_OS = np.sort(L_sigma_OS)
                    L_sigma_S = np.sort(L_sigma_S)    
                    
                    nb_min = int(nb*alpha/(2*100))
                    nb_max = int(nb*(100-alpha/2)/100)
                    print(sigma_r, sqrt(L_sigma_r[nb_min]), sqrt(L_sigma_r[nb_max]))
                    print(sigma_R, sqrt(L_sigma_R[nb_min]), sqrt(L_sigma_R[nb_max]))
                    print(sigma_OS, sqrt(L_sigma_OS[nb_min]), sqrt(L_sigma_OS[nb_max]))
                    print(sigma_S, sqrt(L_sigma_S[nb_min]), sqrt(L_sigma_S[nb_max]))
                    nb_arrondi = len(str(Y[0,0,0]))-2
                    text = "Repeatability: \n- σr: "+str(round(sigma_r, nb_arrondi)) \
                            +"\n- IC(5%): "+str(round(sqrt(L_sigma_r[nb_min]), nb_arrondi)) \
                            +" - "+str(round(sqrt(L_sigma_r[nb_max]), nb_arrondi))
                    text += "\nReproducibility: \n- σR: "+str(round(sigma_R, nb_arrondi)) \
                            +"\n- IC(5%): "+str(round(sqrt(L_sigma_R[nb_min]), nb_arrondi)) \
                            +" - "+str(round(sqrt(L_sigma_R[nb_max]), nb_arrondi))
                    text += "\nVariations between operators and series: \n- σOS: "+str(round(sigma_OS, nb_arrondi)) \
                            +"\n- IC(5%): "+str(round(sqrt(L_sigma_OS[nb_min]), nb_arrondi)) \
                            +" - "+str(round(sqrt(L_sigma_OS[nb_max]), nb_arrondi))
                    text += "\nVariability between measurement series: \n- σS: "+str(round(sigma_S, nb_arrondi)) \
                            +"\n- IC(5%): "+str(round(sqrt(L_sigma_S[nb_min]), nb_arrondi)) \
                            +" - "+str(round(sqrt(L_sigma_S[nb_max]), nb_arrondi))
                    fn = tk.Toplevel(self._fn)
                    fn.title("Result")
                    tk.Label(fn, text = text, justify="left").grid(row=0, column=0) 
                    fn.mainloop()

    def _command_ouvrir(self):
        try:
            lien = askopenfilename(title="Open excel", initialdir=path.expanduser("~/Documents"),
                                filetypes = [("Excel files", ".xlsx")])
            tab = pd.read_excel(lien, index_col= 0)
            tab = tab.to_numpy()[1:]
            for child in self._frame.winfo_children():
                child.destroy()

            self._L_donnee = []

            for j in range(1,self._var_op.get()+1):
                    cdr = tk.Label(self._frame, text="Operator "+str(j), borderwidth=1, 
                                   relief=tk.RIDGE)
                    cdr.grid(row = 0, column=1+(j-1)*self._var_app.get(), columnspan=self._var_op.get()+1,
                             sticky="nesw")

                    for k in range(self._var_app.get()):
                        cdr = tk.Label(self._frame, text="Measure "+str(k+1),
                                       borderwidth=1, relief=tk.RIDGE)
                        cdr.grid(row = 2, column=1+k+(j-1)*self._var_app.get(),
                                 sticky="nesw")
            
            cdr = tk.Label(self._frame, text="Measurement series",
                           borderwidth=1, relief=tk.RIDGE)
            cdr.grid(row = 0, column=0, rowspan=3, sticky="nesw")
            for i in range(self._var_piec.get()):
                cdr = tk.Label(self._frame, text=str(i+1),
                               borderwidth=1, relief=tk.RIDGE)
                cdr.grid(row = 3+i, column=0, sticky="nesw")
            
            for i in range(self._var_piec.get()):
                X = []
                for j in range(self._var_op.get()):
                    Y = []
                    for k in range(self._var_app.get()):
                        en = tk.Entry(self._frame, width = 10, bg="#F0F0ED", relief=tk.RIDGE)
                        en.insert(0, tab[i, j*self._var_app.get()+k])
                        en.grid(row=i+3, column=j*self._var_app.get()+k+1,
                                sticky="nesw")
                        Y.append(en)
                    X.append(Y)
                self._L_donnee.append(X)
            
            self._frame.update_idletasks()
            self._canvas.config(scrollregion=self._canvas.bbox("all"))
        except:
            showerror(title="Problem", message="Opening problem")

if '__main__' == __name__:
    fn = class_fn()