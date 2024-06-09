from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import csr_matrix as sparse
import time

from case39 import case39

sep = case39()
Sb=sep['baseMVA']

# UC - TRANSMISIÓN

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
pos_g = sep["pos_g"]     # Posición generador 
Cg = sep["Cg"]           # Matriz de conexión de generadores

#Rampas de generadores
RUg = sep["units"][:,11]    
RDg = sep["units"][:,11]
SUg = sep["units"][:,4]
SDg = sep["units"][:,4]

# Load demand
Dda = sep['demand']
nh = len(Dda)
nb = len(sep['bus'])
Load_bus = sep['bus'][:,2]

# Transmission data
nl = len(sep['branch']) # number of transmission elements
SF = sep['SF'] # shift-factors
FM = sep['branch'][:,5] # thermal limit
FM[23] = 75
from_b = (sep['branch'][:,0]-1).astype(int)
to_b = (sep['branch'][:,1]-1).astype(int)

# ERV - wind
indawind =np.array([31]) # Posición de las barras 32 y 33 en array
nw=len(indawind)
wind_max=1.2*np.array([[54,51,49,48,45,43,37,35,32,23,13,10,7,4,3,3,3,4,5,8,16,21,21,21]])
vert_cost = 50
#Parametros baterias
almacenamiento = True 
n_charging = 0.95
n_discharging = 0.95
battery_initial = 1
battery_max = 50
carga_max = 7.5
descarga_max = 3.5
degradation_cost = 5 
battery_cost = 10




## Generador virtual
#ind1=sep['bus'][:,0]
#for i in range(len(sep['bus'])):
#    for j in range(len(sep['gen'])):
#        if sep['bus'][:,0][i] == sep['gen'][:,0][j]:
#            ind1=np.delete(ind1,np.where(ind1 == sep['gen'][:,0][j]))
#        else:
#            continue
#ind1=ind1-1 #indice de barras sin generacion\n",
#indaux=[]
#for i in range(len(ind1)): 
#    for j in range(len(sep['bus'])): 
#        if ind1[i]+1 == sep['bus'][j,0] and sep['bus'][j,2] > 0:
#            indaux.append(j)
#        else:
#            continue
#indaux=np.array(indaux) #indice barras sin generacion y con demanda",
#Cens=np.array(sparse((np.ones(len(indaux)), (indaux,range(len(indaux)))), (len(sep['bus']), len(indaux))).todense())
#CENS=np.ones(len(indaux))*500

# Modelación
m = Model('NCUC')  # se crea el modelo
m.setParam('OutputFlag', False) # off Gurobi messages
#m.setParam('DualReductions', 0)

#Variables en PU
p_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pg') # variable de generación para cada generador y para cada hora
b_gt = m.addMVar((ng,nh), vtype=GRB.BINARY, name='n_G') # variable binaria que indica estado de encendido/apagado de generador.
pbar_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pbar_gt') # potencia de reserva de cada generador en cada hora
C_on = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='CU')  # costos de encendido
f = m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='f')   # Flujo por cada línea
p_w = m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='p_w')   # Potencia de gen eolica
pw_cu = m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='pw_cu')   # Flujo por cada línea
#p_ens = m.addMVar((len(indaux),nh), vtype=GRB.CONTINUOUS, lb=0, name='P_ens')    # variable de generacion virtual
if almacenamiento:
    p_charging= m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='P_charging')
    p_discharging= m.addMVar((nw,nh), vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='P_discharging')
    e_battery = m.addMVar((nw,nh), vtype=GRB.CONTINUOUS,lb=0, ub=GRB.INFINITY, name='b_battery')
    b_charging=m.addMVar((nw,nh), vtype=GRB.BINARY, name='b_charging')
    b_discharging=m.addMVar((nw,nh), vtype=GRB.BINARY, name='b_discharging')



# Optimization Function
f_obj = 0 # OF
Cop = 0
Cup = 0
Ce = 0 
Cbattery = 0
Cdegradation = 0
Cvertimiento = 0

for h in range(nh):     # Ciclo para cada hora
    # Costos por uso
    # Costo = (MW de cada gen)^2 * (Costo cuadratico)            +     (Costos lineales) * (MW de cada gen) + (Costos Fijos) * (ON_OFF de cada gen)
    Cop += p_gt[:,h]*Sb @ np.diag(sep["units"][:,2]) @ p_gt[:,h]*Sb + sep["units"][:,1] @ p_gt[:,h]*Sb + sep["units"][:,0] @ b_gt[:,h] 
    # Costos por encender unidades
    Cup += C_on[:,h].sum()
    # Costos generador Virtual
    #Ce += CENS @ p_ens[:,h]*Sb
    Cvertimiento += vert_cost*Sb*(pw_cu[:,h].sum())
    if almacenamiento:
        Cdegradation +=degradation_cost*Sb*(p_charging[:,h]+p_discharging[:,h]).sum()
        Cbattery +=battery_cost*Sb*(e_battery[:,h].sum())

f_obj = Cop + Cup + Ce + Cdegradation + Cbattery + Cvertimiento

m.setObjective(f_obj, GRB.MINIMIZE)
m.getObjective()

for h in range(nh):     # Ciclo para cada hora
    #Balance Nodal
    # (Suma de MW gens) = (Dda)
    dda_bus = Dda[h] * Load_bus
    if not almacenamiento:
        m.addConstr(p_gt[:,h].sum() + p_w[:,h].sum() - pw_cu[:,h].sum() == Dda[h]/Sb, name = 'Balance')
    if almacenamiento:
        m.addConstr(p_gt[:,h].sum() + p_w[:,h].sum() - pw_cu[:,h].sum() - p_charging[:,h].sum() + p_discharging[:,h].sum() == Dda[h]/Sb, name = 'Balance')
    #m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h] +Cens@pens[:,h]- Dda_bus/Sb, name="Balance nodal") 
    # (Reserva total de gens) >= (110% Dda)
    m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
    # Limitación de Gen Virtual
    #m.addConstr(-p_ens[:,h] >= -500/Sb, name='P_maxens')
    m.addConstr(p_w[:,h] + pw_cu[:,h] == wind_max[:,h]/Sb, name='BalanceViento')


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

    # Sistema de transmisión
    if not almacenamiento:
        m.addConstr(f[:,h]== SF[:,pos_g] @ p_gt[:,h] + SF[:,indawind] @ (p_w[:,h] - pw_cu[:,h])- SF @ dda_bus/Sb)
    if almacenamiento:
        m.addConstr(f[:,h]== SF[:,pos_g] @ p_gt[:,h] + SF[:,indawind] @ (p_w[:,h] - pw_cu[:,h]) + SF[:,indawind] @ (-p_charging[:,h] + p_discharging[:,h])- SF @ dda_bus/Sb)
    m.addConstr(-f[:,h] >= -FM/Sb, name = 'fp')
    m.addConstr(f[:,h] >= -FM/Sb, name = 'fn')   

    #Almacenamiento
    if almacenamiento:
        if h==0:
            m.addConstr( e_battery[:,h]==battery_initial/Sb+p_charging[:,h]*n_charging-p_discharging[:,h]/n_discharging, name="Estado en la hora 0")
        else:
            m.addConstr(e_battery[:,h]==e_battery[:,h-1]+p_charging[:,h]*n_charging-p_discharging[:,h]/n_discharging, name="Estado en la hora n")
        m.addConstr(-p_charging[:,h]*n_charging>=-(carga_max/Sb)*b_charging[:,h], name="Limite carga")
        m.addConstr(-p_discharging[:,h]/n_discharging>=-(descarga_max/Sb)*b_discharging[:,h], name="Limite descarga")
        m.addConstr(-(b_charging[:,h]+b_discharging[:,h])>=-1, name="carga o descarga")
        m.addConstr(-pw_cu[:,h]>=-wind_max[:,h]+p_w[:,h]-p_charging[:,h], name="Almacenar vertimiento")
        m.addConstr( battery_initial/Sb==e_battery[:,23], name="incio/final") 


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
m.write('NCUC.lp')

# SOLVER & INFO
t2 = time.time() #Tiempo inicial solver
m.optimize()
t3 = time.time() #Tiempo final solver

fixed = m.fixed()
fixed.optimize

status = m.Status
if status == GRB.Status.OPTIMAL:
    print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cvert = %.2f ($) + Cbatt = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Cvertimiento.getValue(),Cdegradation.getValue()+Cbattery.getValue()))
    print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    for h in range(nh):
        for w in range(p_w.getAttr('x').shape[0]):
            print("P_wind[%d,%d] = %.3f" % (w+1,h+1,p_w.X[w,h]*Sb))
        for w in range(pw_cu.getAttr('x').shape[0]):
            print("P_cw[%d,%d] = %.3f" % (w+1,h+1,pw_cu.X[w,h]*Sb))
        for b in range(e_battery.getAttr('x').shape[0]):
            print("E_batt[%d,%d] = %.3f" % (b+1,h+1,e_battery.X[b,h]*Sb))
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


fig = plt.figure(figsize=(7, 10), dpi=70)
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
print('Finish!')