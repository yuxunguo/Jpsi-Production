from Evolution import Evo_WilsonCoef_SG, M_jpsi
import numpy as np
import csv

NF = 4

mulst = np.linspace(0.65,2,60) * M_jpsi/2

W_ref = Evo_WilsonCoef_SG(M_jpsi/2,NF)[1]

Wlst = np.array([Evo_WilsonCoef_SG(mu,NF)/W_ref for mu in mulst ])
    
with open("Output/WilsonCoef.csv","w",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(np.column_stack((mulst, np.real(Wlst))))