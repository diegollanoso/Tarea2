from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from case39 import case39

sep = case39()
Sb=sep['baseMVA']

# UC - SIN TRANSMISIÓN

# SEP parameters
t0 = time.time()

# Generation data
ng = len(sep['gen'])
P_g0 = np.array(sep["p02006"])  # Potencia en la hora 0 de los generadores
b_g0 = np.array(sep["p02006"]).astype("bool").astype(int)   # Estado inicial de generadores

CUg = sep["units"][:,8]  # Costos de encendido
Gen = sep['units']       # Unidades de generación
Pmax = sep['units'][:,3]# Potencia máxima de generadores
Pmin = sep['units'][:,4]
#pos_g = sep["pos_g"]     # Posición generador 
#Cg = sep["Cg"]           # Matriz de conexión de generadores

#Rampas de generadores
RUg = sep["units"][:,11]    
RDg = sep["units"][:,11]
SUg = sep["units"][:,4]
SDg = sep["units"][:,4]

# Load demand
Dda = sep['demand']
nh = len(Dda)
#nb = len(sep['bus'])
#Load_bus = sep['bus'][:,2]

# Modelación
m = Model('UC-ED')  # se crea el modelo
m.setParam('OutputFlag', False) # off Gurobi messages

#Variables en PU
p_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pg') # variable de generación para cada generador y para cada hora
b_gt = m.addMVar((ng,nh), vtype=GRB.BINARY, name='n_G') # variable binaria que indica estado de encendido/apagado de generador.
pbar_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pbar_gt') # potencia de reserva de cada generador en cada hora
C_on = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='CU')  # costos de encendido

# Optimization Function
f_obj = 0 # OF
Cop = 0
Cup = 0

for h in range(nh):     # Ciclo para cada hora
    # Costos por uso
    # Costo = (MW de cada gen)^2 * (Costo cuadratico)            +     (Costos lineales) * (MW de cada gen) + (Costos Fijos) * (ON_OFF de cada gen)
    Cop += p_gt[:,h]*Sb @ np.diag(sep["units"][:,2]) @ p_gt[:,h]*Sb + sep["units"][:,1] @ p_gt[:,h]*Sb + sep["units"][:,0] @ b_gt[:,h] 
    # Costos por encender unidades
    Cup += C_on[:,h].sum()

f_obj = Cop + Cup

m.setObjective(f_obj, GRB.MINIMIZE)
m.getObjective()

for h in range(nh):     # Ciclo para cada hora
    #Balance Nodal
    # (Suma de MW gens) = (Dda)
    m.addConstr(p_gt[:,h].sum() == Dda[h]/Sb,name='Balance')        # Ecuación (8) Carrión - Arroyo
    # (Reserva total de gens) >= (110% Dda)
    m.addConstr(pbar_gt[:,h].sum() >= Dda[h]/Sb*1.1,name='Reserva')  # Ecuación (7) Carrión - Arroyo  

    # Pmin y Pmax de P y P_disp
    m.addConstr( p_gt[:,h] >= np.diag(Pmin/Sb) @ b_gt[:,h], name="Pmin" )   # Ecuación (9) Carrión - Arroyo  
    m.addConstr( -p_gt[:,h] >= -pbar_gt[:,h] , name="Pmax")            # Ecuación (9) Carrión - Arroyo  
    m.addConstr( pbar_gt[:,h] >= 0, name="Pmin_r")                     # Ecuación (10) Carrión - Arroyo  
    m.addConstr( -pbar_gt[:,h] >= -(np.diag(Pmax) @ b_gt[:,h])/Sb , name="Pmax_r")  # Ecuación (10) Carrión - Arroyo  

    #Costos de encendido
    if h==0:
        m.addConstr( C_on[:,h] >= np.diag(CUg)@(b_gt[:,h]-b_g0) )       # Ecuación (3) Carrión - Arroyo  
    else:
        m.addConstr( C_on[:,h] >= np.diag(CUg)@(b_gt[:,h]-b_gt[:,h-1]) )    # Ecuación (3) Carrión - Arroyo  

#Rampas 
    #Hora = 1
    if h==0:
        # (P_bar en hora 1) <= (MW en hora 0) + (Rampa Up * Gen ON en hora 0)  +  (Start Up * (Gen ON en hora 1 - hora 0)) + (Pmax) - (Pmax * Gen On en hora 1)
        m.addConstr( -pbar_gt[:,h] >= - (P_g0 + np.diag(RUg) @ b_g0 + np.diag(SUg)@(b_gt[:,h]-b_g0) + Pmax - np.diag(Pmax)@b_gt[:,h] )/Sb, name="CA_eq11" )
        # (MW en hora 0 - MW en hora 1) <= (Rampa Down * Gen ON en hora 1)  +  (Shut Down * (Gen ON en hora 0 - hora 1)) + (Pmax) - (Pmax * Gen On en hora 0)
        m.addConstr( -( P_g0/Sb - p_gt[:,h] ) >= -( np.diag(RDg)@b_gt[:,h] + np.diag(SDg)@(b_g0-b_gt[:,h]) + Pmax - np.diag(Pmax)@b_g0 )/Sb, name="CA_eq13" )

    else:
        # (P_bar en hora h) <= (MW en hora h-1) + (Rampa Up * Gen ON en hora h-1)  +  (Start Up * (Gen ON en hora h - hora h-1)) + (Pmax) - (Pmax * Gen On en hora h)
        m.addConstr( -pbar_gt[:,h] >= - p_gt[:,h-1] - ( np.diag(RUg) @ b_gt[:,h-1] + np.diag(SUg)@(b_gt[:,h]-b_gt[:,h-1]) + Pmax - np.diag(Pmax)@b_gt[:,h] )/Sb, name="CA_eq11" )
        # (MW en hora h-1 - MW en hora h) <= (Rampa Down * Gen ON en hora h)  +  (Shut Down * (Gen ON en hora h-1 - hora h)) + (Pmax) - (Pmax * Gen On en hora h-1)
        m.addConstr( -( p_gt[:,h-1] - p_gt[:,h] ) >= -( np.diag(RDg)@b_gt[:,h] + np.diag(SDg)@(b_gt[:,h-1]-b_gt[:,h]) + Pmax - np.diag(Pmax)@b_gt[:,h-1] )/Sb, name="CA_eq13" )

for h in range(nh-1):
    # (P_bar en hora h) <= (Pmax * Gen On en hora h+1) + (Shut Down * (Gen On en hora h - hora h+1)
    m.addConstr( -pbar_gt[:,h] >= -( np.diag(Pmax)@b_gt[:,h+1] + np.diag(SDg)@( b_gt[:,h]- b_gt[:,h+1] ) )/Sb, name="CA_eq12" ) 

t1 = time.time() # Tiempo final formulación

# SOLVER & INFO
t2 = time.time() #Tiempo inicial solver
m.optimize()
t3 = time.time() #Tiempo final solver

status = m.Status
if status == GRB.Status.OPTIMAL:
    print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) ' % (m.objVal,Cop.getValue(),Cup.getValue()))
    print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    print('=> Formulation time: %.4f (s)'% (t1-t0))
    print('=> Solution time: %.4f (s)' % (t3-t2))
    print('=> Solver time: %.4f (s)' % (m.Runtime))
elif status == GRB.Status.INF_OR_UNBD or \
   status == GRB.Status.INFEASIBLE  or \
   status == GRB.Status.UNBOUNDED:
   print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)

fig = plt.figure(figsize=(7, 10), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
ax = plt.subplot(gs[0, 0])
sPlot = ax.imshow(p_gt.x.T*(Sb/Pmax), cmap=plt.cm.jet, alpha=0.75)
ax.set_xticks([k for k in range(ng)])
ax.set_xticklabels([str(k+1) for k in range(ng)])
ax.set_yticks([k for k in range(nh)])
ax.set_ylabel('Tiempo (h)')
ax.set_xlabel('Generadores')
for g in range(ng):
    for h in range(nh):
        ax.text(g, h, np.around(p_gt.x[g,h].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=5)
ax = plt.subplot(gs[0, 1])
fig.colorbar(sPlot, cax=ax, extend='both')
ax.set_ylabel('Cargabilidad (%)')
plt.show()