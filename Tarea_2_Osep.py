import gurobipy as gp
from gurobipy import *
import numpy as np
import pandas as pan
from numpy import diag, ones, pi, zeros, arange,array, ix_, r_, flatnonzero as find
from numpy.linalg import solve, inv
from scipy.sparse import csr_matrix as sparse
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate

import case39 as mpc
sep = mpc.case39()

t0 = time.time() # Tiempo inicial formulación

# SEP parameters

ng = len(sep['gen'])  
nh = len(sep['demand']) 
nb = len(sep['bus'])
Sb = sep["baseMVA"]
Dda = sep['demand']
SL = sep['SL'][0]
Pmaxg = sep['units'][:,3]
#Parámetros generadores
ngen=[1,2,3,4,5,6,7,8,9,10]
RUg = sep["units"][:,11]
RDg = sep["units"][:,11]
SUg = sep["units"][:,4]
SDg = sep["units"][:,4]
P_g0 = np.array(sep["p02006"])
B_g0 = np.array(sep["p02006"]).astype("bool").astype(int)
Pmax = sep['units'][:,3]
Pmin = sep["units"][:,4]   
CUg = sep["units"][:,8] 

Gen = sep['units']
pos_g = sep["pos_g"]
Cg = sep["Cg"]

#seleccion de criterio
#Para realizar el UC sin la red de transmision se coloca solo Uc=1
Uc=0
#Para incorporar la red de transmision se coloca solo Tras=1
tras=0
#Para incorporar la red de transmision y los generadores virtuales se coloca solo virtual=1
virtual=0
#Para realizar el TS se coloca solo ts=1
switching=0
#Para realizar el TS con perdidas se coloca solo tsplusloss=1
tsplusloss=0
#Para incorporar las perdidas se coloca solo loss=1
loss=0
#Para incorporar la complementariedad se coloca loss=1 y complementariedad=1
complementariedad=0
#Para incorporar la adyacencia se coloca loss=1, complementariedad=1 y adyacencia=1
adyacencia=0
#Para incorporar la energia renovable se coloca renovable=1 y almacenamiento=1 si se desea incorporar las baterias
renovable=1
almacenamiento=1
#Antes de limitar la linea se configura los limites de las lineas en 1000, para ello se coloca prueba=1
prueba=0
if prueba:
    for i in range(len(sep['branch'])):
        sep['branch'][i,5]=1000
#Limitación de Flujos
#Con la variable caso se elige la limitacion de la linea 14-15
caso=2
if caso==1:
    sep['branch'][23,5] = 100
elif caso==2:
    sep['branch'][23,5] = 75
elif caso==3:
    sep['branch'][23,5] = 50

#Parámetros Transmisión
nl = len(sep["branch"]) 
xl = sep["branch"][:,3]
from_b = (sep['branch'][:,0]-1).astype(int)
to_b = (sep['branch'][:,1]-1).astype(int)
A = sep["S"]
I = r_[range(nl), range(nl)]
A_f=array(sparse((r_[ones(nl), zeros(nl)], (I, r_[from_b, to_b])), (nl, nb)).todense())
A_t =array(sparse((r_[zeros(nl), -ones(nl)], (I, r_[from_b, to_b])), (nl, nb)).todense())
FM = sep["branch"][:,5]
b = diag(1/xl)
Bf = sep['Bf']

#perdidas
rl=sep["branch"][:,2]
zprim = zeros(nl).astype(complex)    
for l in range(nl):
    zprim[l] = (complex(rl[l],xl[l]))
yprim=1/zprim
bloss=diag(-yprim.imag)
G=sep['G']
B=sep['B']
L=9
C_loss=np.ones(nl)*1000
#Costo perdidas

#Estimacion generacion eolica
#indawind=np.array([31,32]) #posicion de los generadores eolicos (barras 32 y 33)
#nw=len(indawind)
#Cwind=np.array(sparse((ones(len(indawind)), (indawind,range(len(indawind)))), (len(sep['bus']), len(indawind))).todense())
#wind_max=np.array([[107,107,106,103,101,99,88,87,82,62,34,21,13,9,7,7,8,9,12,21,34,44,45,41],[107,107,106,103,101,99,88,87,82,62,34,21,13,9,7,7,8,9,12,21,34,44,45,41]])
indawind=np.array([31]) #posicion de los generadores eolicos (barras 32 y 33)
nw=len(indawind)
Cwind=np.array(sparse((ones(len(indawind)), (indawind,range(len(indawind)))), (len(sep['bus']), len(indawind))).todense())
wind_max=np.array([[54,51,49,48,45,43,37,35,32,23,13,10,7,4,3,3,3,4,5,8,16,21,21,21]])*1.2
#Parametros baterias
n_charging=0.95
n_discharging=0.95
battery_energy_max=10
Carga_max=0.2
Descarga_max=0.2
degradation_cost=1
battery_cost=2
e_inicial=2
#ens
ind1=sep['bus'][:,0]
for i in range(len(sep['bus'])):
    for j in range(len(sep['gen'])):
        if sep['bus'][:,0][i] == sep['gen'][:,0][j]:
            ind1=np.delete(ind1,np.where(ind1 == sep['gen'][:,0][j]))
        else:
            continue
ind1=ind1-1 #indice de barras sin generacion\n",
indaux=[]
for i in range(len(ind1)): 
    for j in range(len(sep['bus'])): 
        if ind1[i]+1 == sep['bus'][j,0] and sep['bus'][j,2] > 0:
            indaux.append(j)
        else:
            continue
lista_ens=indaux
n_ens=len(indaux)
indaux=np.array(indaux) #indice barras sin generacion y con demanda",
Cens=np.array(sparse((ones(len(indaux)), (indaux,range(len(indaux)))), (len(sep['bus']), len(indaux))).todense())
CENS=np.ones(len(indaux))*500
# Trabajo Previo a TS 
if switching:
#Líneas no candidatas a TS
    pl_nots = find(sep['branch'][:,17] == 0)    # posicion de lineas no candidatas a ts
    nl_nots = len(pl_nots)                      # numero de lineas no candidatas a ts

#Lineas candidatas a TS
    pl_ts = find(sep['branch'][:,17] == 1)      # posicion de Lineas candidatas a switching
    nl_ts = len(pl_ts)                          # cantidad de lineas candidatas a switching

    b_nots = np.delete(np.delete(b,pl_ts,axis=1),pl_ts,axis=0)   #matriz b sin lineas de ts
    b_ts = np.delete(np.delete(b,pl_nots,axis=1),pl_nots,axis=0) #matriz b con lineas de ts

    FM_lnots = np.delete(FM,pl_ts,axis=0)       # flujo de lineas existentes
    FM_lts = np.delete(FM,pl_nots,axis=0)       #flujos de lineas candidatas a ts 

    from_b_nots = np.delete(from_b,pl_ts,axis=0) # from lineas existentes
    to_b_nots = np.delete(to_b,pl_ts,axis=0)     # to lineas existentes

    from_ts=sep['branch'][pl_ts,0]
    to_ts=sep['branch'][pl_ts,1]

    M = 1000000
    costo_ts = 50
if tsplusloss:
#Líneas no candidatas a TS
    pl_nots = find(sep['branch'][:,17] == 0)    # posicion de lineas no candidatas a ts
    nl_nots = len(pl_nots)                      # numero de lineas no candidatas a ts

#Lineas candidatas a TS
    pl_ts = find(sep['branch'][:,17] == 1)      # posicion de Lineas candidatas a switching
    nl_ts = len(pl_ts)                          # cantidad de lineas candidatas a switching

    bloss_nots = np.delete(np.delete(bloss,pl_ts,axis=1),pl_ts,axis=0)   #matriz b sin lineas de ts
    bloss_ts = np.delete(np.delete(bloss,pl_nots,axis=1),pl_nots,axis=0) #matriz b con lineas de ts

    FM_lnots = np.delete(FM,pl_ts,axis=0)       # flujo de lineas existentes
    FM_lts = np.delete(FM,pl_nots,axis=0)       #flujos de lineas candidatas a ts 

    from_b_nots = np.delete(from_b,pl_ts,axis=0) # from lineas existentes
    to_b_nots = np.delete(to_b,pl_ts,axis=0)     # to lineas existentes

    from_ts=sep['branch'][pl_ts,0]
    to_ts=sep['branch'][pl_ts,1]

    M = 1000000
    costo_ts = 50
# Initializing model
m = Model('UC_tarea_1')
m.setParam('OutputFlag', False)
m.setParam('DualReductions',0)
m.Params.MIPGap = 1e-6


# VARIABLE DEFINITIONS

p_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='P_gt')                              # variable de generación para cada generador y para cada hora
b_gt = m.addMVar((ng,nh), vtype=GRB.BINARY, name='b_gt')                                        # variable binaria que indica estado de encendido/apagado de generador
pbar_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pbar_gt')                        # potencia de reserva de cada generador en cada hora
CU = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='CU')                                  # costos de encendido
if tras or virtual or renovable:
    d = m.addMVar((nb,nh), vtype=GRB.CONTINUOUS, ub=pi, lb=-pi, name="delta")                       # angulos de cada barra en cada hora
    f = m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flujo")   # flujos de cada linea en cada hora
if renovable:
    p_wind=m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, name='P_wind') 
    p_cw=m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="vertimiento")
if almacenamiento:
    p_charging= m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, name='P_charging')
    p_discharging= m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, name='P_discharging')
    e_battery = m.addMVar((nw,nh), vtype=GRB.CONTINUOUS,lb=0, name='b_battery')
    b_charging=m.addMVar((nw,nh), vtype=GRB.BINARY, name='b_charging')
    b_discharging=m.addMVar((nw,nh), vtype=GRB.BINARY, name='b_discharging')
if switching:
    d = m.addMVar((nb,nh), vtype=GRB.CONTINUOUS, ub=pi, lb=-pi, name="delta")                       # angulos de cada barra en cada hora
    f = m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flujo")   # flujos de cada linea en cada hora 
    s_ts = m.addMVar((nl_ts, nh), vtype=GRB.BINARY, name='s_ts')                                    # variable binaria de TS
if virtual:
    pens = m.addMVar((len(indaux),nh), vtype=GRB.CONTINUOUS, lb=0, name='P_ens')                    # variable de generacion virtual
if loss:
    d = m.addMVar((nb,nh), vtype=GRB.CONTINUOUS, ub=pi, lb=-pi, name="delta")                       # angulos de cada barra en cada hora
    f = m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flujo")   # flujos de cada linea en cada hora
    f_p=m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="f_p")                 # flujo positivo en cada linea en cada hora
    f_n=m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="f_n")                 # flujo negativo en cada linea en cada hora
    d_f=m.addMVar((nl,L,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="d_f")               # Delta flujo 
    p_loss=m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="p_loss")           # Perdidas en cada hora
    if complementariedad:
        n_l = m.addMVar((nl,nh), vtype=GRB.BINARY, name='n_l')                                      # variable binaria complentaridad
    if adyacencia and complementariedad:
        n_a = m.addMVar((nl,L,nh), vtype=GRB.BINARY, name='n_a')                                    # variable binaria adyacencia
if tsplusloss:
    d = m.addMVar((nb,nh), vtype=GRB.CONTINUOUS, ub=pi, lb=-pi, name="delta")                       # angulos de cada barra en cada hora
    f = m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flujo")   # flujos de cada linea en cada hora  
    s_ts = m.addMVar((nl_ts, nh), vtype=GRB.BINARY, name='s_ts')                                    # variable binaria de TS
    f_p=m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="f_p")                 # flujo positivo en cada linea en cada hora
    f_n=m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="f_n")                 # flujo negativo en cada linea en cada hora
    d_f=m.addMVar((nl,L,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="d_f")               # Delta flujo 
    p_loss=m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="p_loss")           # Perdidas en cada hora       

# OPTIMIZATION PROBLEM
f_obj = 0 # OF

Cop = 0
Cup = 0
Cts = 0
Ce = 0
Closs=0
Cbattery=0
Cdegradation=0
total_loss=0
for h in range(nh):
    Cop += p_gt[:,h]*Sb @ np.diag(sep["units"][:,2]) @ p_gt[:,h]*Sb + sep["units"][:,1] @ p_gt[:,h]*Sb + sep["units"][:,0] @ b_gt[:,h] 
    Cup += CU[:,h].sum()
    if switching:
        Cts += costo_ts * (1 - s_ts[:,h]).sum()                 # Cuando la línea sale de servicio s_ts=0, se aplica el costo del TS
    if virtual:
        Ce +=CENS@pens[:,h]*Sb
    if loss:
        total_loss += p_loss[:,h].sum() 
        Closs +=C_loss@p_loss[:,h]*Sb
    if tsplusloss:
        Cts += costo_ts * (1 - s_ts[:,h]).sum()
        total_loss += p_loss[:,h].sum() 
    if almacenamiento:
        Cdegradation +=degradation_cost*Sb*(p_charging[:,h]+p_discharging[:,h]).sum()
        Cbattery +=battery_cost*Sb*(e_battery[:,h].sum())




f_obj = Cop + Cup + Cts + Ce + Closs + Cdegradation

m.setObjective(f_obj, GRB.MINIMIZE)
m.getObjective()

#Creación de restricciones
if switching or tsplusloss:
    index_sinlts = np.delete(np.arange(0,nl),pl_ts, axis=0)
#Restriccion 

for h in range(nh):
    #Balance Nodal y Reserva   
    if tras or switching:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h] - Dda_bus/Sb, name="LCK") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        m.addConstr(d[SL,h] == 0, name="SL") #Angulo de referencia
    if virtual:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h] +Cens@pens[:,h]- Dda_bus/Sb, name="LCK") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        m.addConstr(-pens[:,h] >= -500/Sb, name='P_maxens')
        m.addConstr(d[SL,h] == 0, name="SL") #Angulo de referencia
    if renovable and almacenamiento==0:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h]+Cwind@p_wind[:,h]-Cwind@p_cw[:,h]- Dda_bus/Sb, name="LCK") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        m.addConstr( -p_wind[:,h]>=-wind_max[:,h]/Sb, name="Renovable")
        m.addConstr( p_wind[:,h]+p_cw[:,h]==wind_max[:,h]/Sb, name="Renovable")
        m.addConstr(d[SL,h] == 0, name="SL") #Angulo de referencia
    elif renovable and almacenamiento:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h]+Cwind@p_wind[:,h]-Cwind@p_cw[:,h]-Cwind@p_charging[:,h]+Cwind@p_discharging[:,h]- Dda_bus/Sb, name="LCK") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        m.addConstr( -p_wind[:,h]>=-wind_max[:,h]/Sb, name="Renovable")
        m.addConstr( p_wind[:,h]+p_cw[:,h]==wind_max[:,h]/Sb, name="Renovable")
        m.addConstr(d[SL,h] == 0, name="SL") #Angulo de referencia   
        if h==0:
            #m.addConstr( e_battery[:,h]==p_charging[:,h]*n_charging-p_discharging[:,h]/n_discharging, name="Estado en la hora 0")
            m.addConstr( e_battery[:,h]==e_inicial/Sb+p_charging[:,h]*n_charging-p_discharging[:,h]/n_discharging, name="Estado en la hora n")  
        else:
            m.addConstr( e_battery[:,h]==e_battery[:,h-1]+p_charging[:,h]*n_charging-p_discharging[:,h]/n_discharging, name="Estado en la hora n")
        m.addConstr(-p_charging[:,h]*n_charging>=-(Carga_max/Sb)*b_charging[:,h], name="Limite carga")
        m.addConstr(-p_discharging[:,h]/n_discharging>=-(Descarga_max/Sb)*b_discharging[:,h], name="Limite descarga")
        m.addConstr(-(b_charging[:,h]+b_discharging[:,h])>=-1, name="carga o descarga")
        m.addConstr(-e_battery[:,h]>=-battery_energy_max/Sb, name="Limite estado bateria")
        m.addConstr( e_inicial/Sb==e_battery[:,23], name="incio/final")
    if loss or tsplusloss:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T@f[:,h]== Cg@p_gt[:,h]-(Dda_bus/Sb+0.5*abs(A.T)@p_loss[:,h]), name="LCK") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        m.addConstr(d[SL,h] == 0, name="SL") #Angulo de referencia
    if Uc:
        m.addConstr(p_gt[:,h].sum() == Dda[h]/Sb,name="LCK")                       # Ecuación (8) Carrión - Arroyo
        m.addConstr(pbar_gt[:,h].sum() >= Dda[h]/Sb*1.1,name='Reserva')            # Ecuación (7) Carrión - Arroyo  


    #Pmin y Pmax de P y P_disp
    m.addConstr( p_gt[:,h] >= np.diag(Pmin/Sb) @ b_gt[:,h], name="Pmin" )
    m.addConstr( -p_gt[:,h] >= -pbar_gt[:,h] , name="Pmax")
    m.addConstr( pbar_gt[:,h] >= 0, name="Pmin_r")
    m.addConstr( -pbar_gt[:,h] >= -(np.diag(Pmax) @ b_gt[:,h])/Sb , name="Pmax_r")

    #Costos de encendido
    if h==0:
        m.addConstr( CU[:,h] >= np.diag(CUg)@(b_gt[:,h]-B_g0) )
    else:
        m.addConstr( CU[:,h] >= np.diag(CUg)@(b_gt[:,h]-b_gt[:,h-1]) )

    #Restricciones sistema de transmisión
    if tras or virtual or renovable:
        m.addConstr(f[:,h]== b@d[from_b,h] - b@d[to_b,h])
        m.addConstr(-f[:,h] >= -FM/Sb, name = 'fp')
        m.addConstr(f[:,h] >= -FM/Sb, name = 'fn')
    if loss:
        m.addConstr(f[:,h]== bloss@d[from_b,h] - bloss@d[to_b,h]) #flujo considerando perdidas
        m.addConstr(-f[:,h]-0.5*p_loss[:,h] >= -FM/Sb, name = 'f_p_loss')
        m.addConstr(f[:,h]-0.5*p_loss[:,h] >= -FM/Sb, name = 'f_n_loss')
        m.addConstr(f[:,h]==f_p[:,h]-f_n[:,h], name='Flujo_total')       
        if complementariedad:
            m.addConstr(-f_p[:,h]>=-n_l[:,h]*FM/Sb, name='fp')   #flujo positvo-restriccion de complementaridad
            m.addConstr(-f_n[:,h]>=(1-n_l[:,h])*(-FM/Sb), name='fn') #flujo nefativo-restriccion de complementaridad
        else:
            m.addConstr(-f_p[:,h]>=-FM/Sb, name='fp')   #flujo positvo-restriccion 
            m.addConstr(-f_n[:,h]>=-FM/Sb, name='fn') #flujo nefativo-restriccion 
        if adyacencia and complementariedad:
            for l in range(L): 
                if l==0:
                    m.addConstr(-d_f[:,l,h]>=-FM/(Sb*L), name='d_f_Res_max_A_l')
                    m.addConstr(d_f[:,l,h]>=n_a[:,l,h]*(FM/(Sb*L)), name='d_f_Res_min_A_l')
                elif l==L-1:
                    m.addConstr(-d_f[:,l,h]>=-n_a[:,l-1,h]*FM/(Sb*L), name='d_f_Res_max_A_L')
                    m.addConstr(d_f[:,l,h]>=0, name='d_f_Res_min_A_L')
                else:
                    m.addConstr(-d_f[:,l,h]>=-n_a[:,l-1,h]*(FM/(Sb*L)), name='d_f_Res_max_A_L-1')
                    m.addConstr(d_f[:,l,h]>=n_a[:,l,h]*(FM/(Sb*L)), name='d_f_Res_min_A_L-1')
        else:
            for l in range(L): 
                m.addConstr(-d_f[:,l,h]>=-FM/(Sb*L), name='d_f_Res_max')
        m.addConstr(d_f[:,:,h].sum(1)==f_p[:,h]+f_n[:,h], name='d_f_Res')
        for i in range(nl):
            kl = np.zeros(L)
            for l in range(L):
                kl[l] = (2*(l+1)-1)*(FM[i]/Sb)/L# 
            m.addConstr(p_loss[i,h] == (G[i]/(B[i]**2)*((kl*d_f[i,:,h]).sum())), name='ploss_Res')
             
    if switching:
        m.addConstr( f[index_sinlts,h] == b_nots@d[from_b_nots,h] - b_nots@d[to_b_nots,h])
        m.addConstr( f[index_sinlts,h] <= FM_lnots/Sb, name="flujo_max1")
        m.addConstr( f[index_sinlts,h] >= -FM_lnots/Sb, name="flujo_max2")
        for index_ts, ts in enumerate(pl_ts):  
            m.addConstr(f[ts,h] - (b[ts,ts] * d[from_b[ts],h] - b[ts,ts] * d[to_b[ts],h]) >= -(1- s_ts[index_ts,h])* M   , name = 'fe_ts_p'+'_' +str(ts))
            m.addConstr(-f[ts,h] + (b[ts,ts] * d[from_b[ts],h] - b[ts,ts] * d[to_b[ts],h]) >= -(1- s_ts[index_ts,h])* M  , name = 'fe_ts_n'+'_' +str(ts))
            m.addConstr(-f[ts,h] >= -FM[ts]/Sb * s_ts[index_ts,h] , name = 'fe_p_ts'+'_' +str(ts))
            m.addConstr(f[ts,h] >= -FM[ts]/Sb * s_ts[index_ts,h], name = 'fe_n_ts'+'_' +str(ts))
    if tsplusloss:
        m.addConstr(f[index_sinlts,h]== bloss_nots@d[from_b_nots,h] - bloss_nots@d[from_b_nots,h]) #flujo considerando perdidas
        m.addConstr( -f[index_sinlts,h]-0.5*p_loss[index_sinlts,h] >=  -FM_lnots/Sb, name="flujo_max1")
        m.addConstr( f[index_sinlts,h] -0.5*p_loss[index_sinlts,h] >= -FM_lnots/Sb, name="flujo_max2")        
        m.addConstr(f[index_sinlts,h]==f_p[index_sinlts,h]-f_n[index_sinlts,h], name='Flujo_total')
        for index_ts, ts in enumerate(pl_ts):  
            m.addConstr(f[ts,h] - (bloss[ts,ts] * d[from_b[ts],h] - bloss[ts,ts] * d[to_b[ts],h]) >= -(1- s_ts[index_ts,h])* M   , name = 'fe_ts_p'+'_' +str(ts))
            m.addConstr(-f[ts,h] + (bloss[ts,ts] * d[from_b[ts],h] - bloss[ts,ts] * d[to_b[ts],h]) >= -(1- s_ts[index_ts,h])* M  , name = 'fe_ts_n'+'_' +str(ts))
            m.addConstr(-f[ts,h]-0.5*p_loss[ts,h]>= -FM[ts]/Sb * s_ts[index_ts,h] , name = 'fe_p_ts'+'_' +str(ts))
            m.addConstr(f[ts,h]-0.5*p_loss[ts,h]>= -FM[ts]/Sb * s_ts[index_ts,h], name = 'fe_n_ts'+'_' +str(ts))
            m.addConstr(f[ts,h]==(f_p[ts,h]-f_n[ts,h])*s_ts[index_ts,h], name='Flujo_total') 
            for l in range(L):
                m.addConstr(-d_f[ts,l,h]>=-(FM[ts]/(Sb*L))*s_ts[index_ts,h], name='d_f_Res_max')   
            m.addConstr(d_f[ts,:,h].sum()==(f_p[ts,h]+f_n[ts,h]), name='d_f_Res') 
            klts = np.zeros(L)
            for l in range(L):
                klts[l] = (2*(l+1)-1)*(FM[ts]/Sb)/L# 
            m.addConstr(p_loss[ts,h] == ((G[ts]/(B[ts]**2)*((klts*d_f[ts,:,h]).sum()))*s_ts[index_ts,h]), name='ploss_Res')
        for l in range(L): 
            m.addConstr(-d_f[index_sinlts,l,h]>=-FM_lnots/(Sb*L), name='d_f_Res_max')
        m.addConstr(d_f[index_sinlts,:,h].sum(1)==f_p[index_sinlts,h]+f_n[index_sinlts,h], name='d_f_Res')
        for i in range(nl_nots):
            kl = np.zeros(L)
            for l in range(L):
                kl[l] = (2*(l+1)-1)*(FM[i]/Sb)/L# 
            m.addConstr(p_loss[i,h] == (G[i]/(B[i]**2)*((kl*d_f[i,:,h]).sum())), name='ploss_Res')

    #Rampas 
    if h==0:
        m.addConstr( -pbar_gt[:,h] >= - (P_g0 + np.diag(RUg) @ B_g0 + np.diag(SUg)@(b_gt[:,h]-B_g0) + Pmax - np.diag(Pmax)@b_gt[:,h] )/Sb, name="CA_eq11" )
        m.addConstr( -( P_g0/Sb - p_gt[:,h] ) >= -( np.diag(RDg)@b_gt[:,h] + np.diag(SDg)@(B_g0-b_gt[:,h]) + Pmax - np.diag(Pmax)@B_g0 )/Sb, name="CA_eq13" )

    else:
        m.addConstr( -pbar_gt[:,h] >= - p_gt[:,h-1] - ( np.diag(RUg) @ b_gt[:,h-1] + np.diag(SUg)@(b_gt[:,h]-b_gt[:,h-1]) + Pmax - np.diag(Pmax)@b_gt[:,h] )/Sb, name="CA_eq11" )
        m.addConstr( -( p_gt[:,h-1] - p_gt[:,h] ) >= -( np.diag(RDg)@b_gt[:,h] + np.diag(SDg)@(b_gt[:,h-1]-b_gt[:,h]) + Pmax - np.diag(Pmax)@b_gt[:,h-1] )/Sb, name="CA_eq13" )

    for h in range(nh-1):
        m.addConstr( -pbar_gt[:,h] >= -( np.diag(Pmax)@b_gt[:,h+1] + np.diag(SDg)@( b_gt[:,h]- b_gt[:,h+1] ) )/Sb, name="CA_eq12" ) 

t1 = time.time() # Tiempo final formulación

# SOLVER & INFO
t2 = time.time() #Tiempo inicial solver
m.optimize()
t3 = time.time() #Tiempo final solver

fixed=m.fixed()
fixed.optimize()


status = m.Status
if status == GRB.Status.OPTIMAL:
    if switching:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cts = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Cts.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    elif virtual:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cens = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Ce.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    elif renovable and almacenamiento==0:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))           
        for h in range(nh):
            for w in range(p_wind.getAttr('x').shape[0]):
                print("P_wind[%d,%d] = %.3f"%(w+1,h+1,p_wind.X[w,h]*Sb))  
            for w in range(p_cw.getAttr('x').shape[0]):
                print("P_cw[%d,%d] = %.3f"%(w+1,h+1,p_cw.X[w,h]*Sb))
    elif renovable and almacenamiento:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cbattery = %.2f ($) + Cdegradation = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Cbattery.getValue(),Cdegradation.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))           
        for h in range(nh):
            for w in range(p_wind.getAttr('x').shape[0]):
                print("P_wind[%d,%d] = %.3f"%(w+1,h+1,p_wind.X[w,h]*Sb))  
            for w in range(p_cw.getAttr('x').shape[0]):
                print("P_cw[%d,%d] = %.3f"%(w+1,h+1,p_cw.X[w,h]*Sb)) 
            for w in range(e_battery.getAttr('x').shape[0]):
                print("P_Battery[%d,%d] = %.3f"%(w+1,h+1,e_battery.X[w,h]*Sb))          
    elif loss:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Closs = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Closs.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))       
        print ('Total P_loss = %.2f [MW]'%(total_loss.getValue()*Sb))
        #for h in range(nh):
        #    print("P_loss[%d] = %.3f"% (h+1,p_loss[:,h].sum().getValue()*Sb))
        for h in range(nh):
            for l in range(f_p.getAttr('x').shape[0]):
                if f_p.X[l,h]!=0 and f_n.X[l,h]!=0:
                    print("f_p[%d,%d,%d] = %.3f // f_n[%d,%d,%d] = %.3f"%(from_b[l]+1,to_b[l]+1,h+1,f_p.X[l,h]*Sb,from_b[l]+1,to_b[l]+1,h+1,f_n.X[l,h]*Sb))            
            for i in range(d_f.getAttr('x').shape[0]):
                if d_f.X[i,1,h]*Sb>d_f.X[i,0,h]*Sb:
                    print("d_f[%d,%d,%d] = %.3f [l=1]    -     %.3f [l=2] "%(from_b[i]+1,to_b[i]+1,h+1,d_f.X[i,0,h]*Sb,d_f.X[i,1,h]*Sb))          
    else:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) ' % (m.objVal,Cop.getValue(),Cup.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    if virtual:
        df_name=[]
        df_value=[]
        df_h=1
        contador=1
        for v in fixed.getConstrs():
            if abs(v.pi) > 1e-2:
                if 'LCK' in v.ConstrName:
                    print('%s = %g ($/MWh)' % (v.ConstrName,v.pi))
                    df_name.append(v.ConstrName+str(df_h))
                    df_value.append(v.pi/Sb)
                    contador+=1
                    if contador==nb+1:
                        df_h+=1
                        contador=1
        df_test=pan.DataFrame([df_name,df_value])
        df_test.T.to_csv(r"C:\Users\pocke\OneDrive\Desktop\Universidad\semestre 7-1\OSEP\tarea 2\resultados.csv", index=False)
    if switching:
        vX = zeros([nh,ng]); pX = zeros([nh,ng]); ptX = zeros([nh,ng]); CUX = zeros([nh,ng]); s_tsX = zeros([nh,nl_ts])
        for t in range(nh):
            vX[t] = b_gt.x[:,t]
            pX[t] = p_gt.x[:,t]
            ptX[t] = pbar_gt.x[:,t]
            CUX[t] = CU.x[:,t]
            s_tsX[t] = s_ts.x[:,t]
        print("Cantidad de valores iguales a cero:",np.count_nonzero(s_tsX.T == 0))
        filas, columnas = np.nonzero(s_tsX==0)
    else:
        vX = zeros([nh,ng]); pX = zeros([nh,ng]); ptX = zeros([nh,ng]); CUX = zeros([nh,ng])
        for t in range(nh):
            vX[t] = b_gt.x[:,t]
            pX[t] = p_gt.x[:,t]
            ptX[t] = pbar_gt.x[:,t]
            CUX[t] = CU.x[:,t]  
    print('=> Formulation time: %.4f (s)'% (t1-t0))
    print('=> Solution time: %.4f (s)' % (t3-t2))
    print('=> Solver time: %.4f (s)' % (m.Runtime))
    if switching: 
        fig = plt.figure(figsize=(7, 7), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(s_ts.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(nh)])
        ax.set_xticklabels([(k+1) for k in range(nh)])
        ax.set_yticks( [k for k in range(nl_ts)]   )
        ax.set_yticklabels([str('%.f-%.f' %(from_ts[g],to_ts[g])) for g in range(nl_ts)])
        ax.set_ylabel('Switching (1/0)')
        ax.set_xlabel('Hora (h)')
        for g in range(nl_ts):
            for h in range(nh):
                ax.text( h, g, np.around(s_ts.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        plt.show()  
elif status == GRB.Status.INF_OR_UNBD or \
   status == GRB.Status.INFEASIBLE  or \
   status == GRB.Status.UNBOUNDED:
   print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)
    # sys.exit(1)   #1

#tabla 1

fig = plt.figure(figsize=(7, 10), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
ax = plt.subplot(gs[0, 0])
sPlot = ax.imshow(p_gt.x.T*(Sb/Pmaxg), cmap=plt.cm.jet, alpha=0.75)
ax.set_xticks([k for k in range(ng)])
ax.set_xticklabels([str(k+1) for k in range(ng)])
ax.set_yticks([k for k in range(nh)])
ax.set_yticklabels([str(k+1) for k in range(nh)])
ax.set_ylabel('Tiempo (h)')
ax.set_xlabel('Generadores')
for g in range(ng):
    for h in range(nh):
        ax.text(g, h, np.around(p_gt.x[g,h].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=5)
ax = plt.subplot(gs[0, 1])
fig.colorbar(sPlot, cax=ax, extend='both')
ax.set_ylabel('Cargabilidad (%)')
plt.show()


if tras or virtual or switching or loss:
    fig = plt.figure(figsize=(7, 10), dpi=60)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
    ax = plt.subplot(gs[0, 0])
    sPlot = ax.imshow(f.x, cmap=plt.cm.jet, alpha=0.75)
    ax.set_xticks([k for k in range(nh)])
    ax.set_xticklabels([(k+1) for k in range(nh)])
    ax.set_yticks( [k for k in range(nl)]   )
    ax.set_yticklabels([str('%.f-%.f' %(from_b[g]+1,to_b[g]+1)) for g in range(nl)])
    ax.set_ylabel('Flujos (MW)')
    ax.set_xlabel('Hora (h)')
    for g in range(nl):
        for h in range(nh):
            ax.text( h, g, np.around(f.x.T[h,g].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=4)
    ax = plt.subplot(gs[0, 1])
    fig.colorbar(sPlot, cax=ax, extend='both')
    ax.set_ylabel('Cargabilidad (%)')
    plt.savefig('flujo_lineas.pdf')
    plt.show()        
if virtual:
    fig = plt.figure(figsize=(7, 10), dpi=100)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
    ax = plt.subplot(gs[0, 0])
    sPlot = ax.imshow(pens.x.T*(Sb/500), cmap=plt.cm.jet, alpha=0.75)
    ax.set_xticks([k for k in range(n_ens)])
    ax.set_xticklabels([str(lista_ens[k]+1) for k in range(n_ens)])
    ax.set_yticks([k for k in range(nh)])
    ax.set_yticklabels([str(k+1) for k in range(nh)])
    ax.set_ylabel('Tiempo (h)')
    ax.set_xlabel('Barras generadores virtuales')
    for g in range(n_ens): 
        for h in range(nh):
            ax.text(g, h, np.around(pens.x[g,h].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=5)
    plt.show()

