from Evolution import Evo_WilsonCoef_SG, M_jpsi, AlphaS
import numpy as np
import csv

NF = 4

#print(AlphaS(2,NF,M_jpsi/2))
'''
mulst = np.linspace(0.65,2,60) * M_jpsi/2

W_ref = Evo_WilsonCoef_SG(M_jpsi/2,NF,p_order=3)[1]

Wp1lst = np.array([Evo_WilsonCoef_SG(mu,NF,p_order=1)/W_ref for mu in mulst ])

Wp2lst = np.array([Evo_WilsonCoef_SG(mu,NF,p_order=2)/W_ref for mu in mulst ])

Wp3lst = np.array([Evo_WilsonCoef_SG(mu,NF,p_order=3)/W_ref for mu in mulst ])
    
with open("Output/WilsonCoef.csv","w",newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(np.column_stack((mulst, np.real(Wp1lst), np.real(Wp2lst), np.real(Wp3lst))))
'''

Ag0lat = 0.4828775568674785
MAglat = 1.6427055429755593
Cg0lat = -0.46439056070121026
MCglat = 0.84695582681868

Aq0lat = 0.49392591129324553
MAqlat = 1.9336471336475611
Cq0lat = -0.26396541362213805
MCqlat = 1.1685635047886935

Ag0p1 = 0.48195072463915617
MAgp1 = 1.639561790773252
Cg0p1 = -0.42167023316958085
MCgp1 = 0.8850152916228669

Aq0p1 = 0.4938627712393706
MAqp1 = 1.9337917769506476
Cq0p1 = -0.2643097005548671
MCqp1 = 1.1675689414245247

Ag0p2 = 0.49973940241429887
MAgp2 = 1.5758960355173612
Cg0p2 = -0.3241864209521915
MCgp2 = 0.9911402004620952

Aq0p2 = 0.5013517221248852
MAqp2 =  1.887711752148423
Cq0p2 = -0.20118762318524377
MCqp2 = 1.3737723492513645