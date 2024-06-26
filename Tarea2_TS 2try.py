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
sep['branch'][23,5] = 75
FM = sep['branch'][:,5] # thermal limit
from_b = (sep['branch'][:,0]-1).astype(int)
to_b = (sep['branch'][:,1]-1).astype(int)
A = sep['S']

# Líneas no candidtas a TS
pl_nots = np.flatnonzero(sep['branch'][:,17] == 0)    # posicion de lineas no candidatas a ts
nl_nots = len(pl_nots)                      # numero de lineas no candidatas a ts

# Lineas candidatas a TS
pl_ts = np.flatnonzero(sep['branch'][:,17] == 1)      # posicion de Lineas candidatas a switching
nl_ts = len(pl_ts)                          # cantidad de lineas candidatas a switching
index_sinlts = np.delete(np.arange(0,nl),pl_ts, axis=0)
from_ts=sep['branch'][pl_ts,0]
to_ts=sep['branch'][pl_ts,1]


FM_lnots = np.delete(FM,pl_ts,axis=0)       # flujo de lineas existentes
FM_lts = np.delete(FM,pl_nots,axis=0)       #flujos de lineas candidatas a ts 

M = 1000000
costo_ts = 10

# Generador virtual
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
m = Model('NCTS')  # se crea el modelo
m.setParam('OutputFlag', True) # off Gurobi messages
m.setParam('DualReductions', 0)
m.Params.MIPGap = 1e-6


#Variables en PU
p_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pg') # variable de generación para cada generador y para cada hora
b_gt = m.addMVar((ng,nh), vtype=GRB.BINARY, name='n_G') # variable binaria que indica estado de encendido/apagado de generador.
pbar_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pbar_gt') # potencia de reserva de cada generador en cada hora
C_on = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='CU')  # costos de encendido
f = m.addMVar((nl_ts,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flujo")   #flujos de cada linea en cada hora
s_ts = m.addMVar((nl_ts, nh), vtype=GRB.BINARY, name='s_ts')                                    # variable binaria de TS
#p_ens = m.addMVar((len(indaux),nh), vtype=GRB.CONTINUOUS, lb=0, name='P_ens')    # variable de generacion virtual


# Optimization Function
f_obj = 0 # OF
Cop = 0
Cup = 0
Ce = 0 
Cts = 0

for h in range(nh):     # Ciclo para cada hora
    # Costos por uso
    # Costo = (MW de cada gen)^2 * (Costo cuadratico)            +     (Costos lineales) * (MW de cada gen) + (Costos Fijos) * (ON_OFF de cada gen)
    Cop += p_gt[:,h]*Sb @ np.diag(sep["units"][:,2]) @ p_gt[:,h]*Sb + sep["units"][:,1] @ p_gt[:,h]*Sb + sep["units"][:,0] @ b_gt[:,h] 
    # Costos por encender unidades
    Cup += C_on[:,h].sum()
    # Costo Transmission Switching
    Cts += costo_ts * (1-s_ts[:,h]).sum()
    # Costos generador Virtual
    #Ce += CENS @ p_ens[:,h]*Sb
    

f_obj = Cop + Cup + Ce + Cts

m.setObjective(f_obj, GRB.MINIMIZE)
m.getObjective()

#Creación de restricciones


for h in range(nh):     # Ciclo para cada hora
    #Balance Nodal
    # (Suma de MW gens) = (Dda)
    dda_bus = Dda[h] * Load_bus
    m.addConstr(p_gt[:,h].sum()  == Dda[h]/Sb, name = 'Balance')
    #m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h] +Cens@pens[:,h]- Dda_bus/Sb, name="Balance nodal") 
    # (Reserva total de gens) >= (110% Dda)
    m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
    # Limitación de Gen Virtual
    #m.addConstr(-p_ens[:,h] >= -500/Sb, name='P_maxens')


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
    #Restricciones sistema de transmisión lineas no candidatas    
    fe = SF[pl_nots,:][:,pos_g] @ p_gt[:,h] - SF[pl_nots,:]@dda_bus/Sb
    fv = (SF[pl_nots,:] @ A[pl_ts,:].T) @ f[:,h]
    m.addConstr(-(fe+fv) >= -FM_lnots/Sb, name = 'fe_p')
    m.addConstr(fe+fv >= -FM_lnots/Sb, name = 'fe_n')

    #Restricciones sistema de transmisión lineas candidatas
    f1 = SF[pl_ts,:][:,pos_g] @ p_gt[:,h] - SF[pl_ts,:]@dda_bus/Sb
    f2 = f[:,h] - (SF[pl_ts,:]@A[pl_ts,:].T) @ f[:,h] 

    m.addConstr(f1-f2 <= np.diag(FM_lts)/Sb @ s_ts[:,h], name = 'fs1_p') # 1
    m.addConstr(f1-f2 >= -np.diag(FM_lts)/Sb @ s_ts[:,h], name = 'fs1_n')
    
    m.addConstr(f[:,h] <= M*(1 - s_ts[:,h]), name = 'fs2_p') # 2
    m.addConstr(f[:,h] >= -M*(1 - s_ts[:,h]), name = 'fs2_n') # 2

    #m.addConstr(f[:,h] <= M*(np.ones(nl_ts)).T - np.diag(M*(np.ones(nl_ts))) @ s_ts[:,h], name = 'fs2_p') # 2
    #m.addConstr(f[:,h] >= -M*(np.ones(nl_ts)).T + np.diag(M*(np.ones(nl_ts))) @ s_ts[:,h], name = 'fs2_n') 


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
m.write('NCTS.lp')

# SOLVER & INFO
t2 = time.time() #Tiempo inicial solver
m.optimize()
t3 = time.time() #Tiempo final solver

status = m.Status
if status == GRB.Status.OPTIMAL:
    print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cts = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Cts.getValue()))
    print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    print('=> Formulation time: %.4f (s)'% (t1-t0))
    print('=> Solution time: %.4f (s)' % (t3-t2))
    print('=> Solver time: %.4f (s)' % (m.Runtime))
elif status == GRB.Status.INF_OR_UNBD or \
   status == GRB.Status.INFEASIBLE  or \
   status == GRB.Status.UNBOUNDED:
   print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)

if True:
    fig = plt.figure(figsize=(7, 10), dpi=150)
    gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
    ax = plt.subplot(gs[0, 0])
    sPlot = ax.imshow(s_ts.x, cmap=plt.cm.jet, alpha=0.75)
    ax.set_xticks([k for k in range(nh)])
    ax.set_xticklabels([(k+1) for k in range(nh)])
    ax.set_yticks( [k for k in range(nl_ts)]   )
    ax.set_yticklabels([str('%.f-%.f' %(from_ts[g],to_ts[g])) for g in range(nl_ts)])
    ax.set_ylabel('Switching (1/0) (MW)')
    ax.set_xlabel('Hora (h)')
    for g in range(nl_ts):
        for h in range(nh):
            ax.text( h, g, np.around(s_ts.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
    plt.savefig('flujo_lineas_ts_on-off.pdf')
    plt.show()

if True:
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

if True:
    f_nots = np.zeros((nl_nots,nh))
    for h in range(nh): 
        fe = SF[pl_nots,:][:,pos_g] @ p_gt[:,h].x - SF[pl_nots,:]@dda_bus/Sb
        fv = (SF[pl_nots,:] @ A[pl_ts,:].T) @ f[:,h].x
        variable=fe+fv
        f_nots[:,h] = (variable)*Sb
    from_b_nots = from_b[pl_nots]
    to_b_nots = to_b[pl_nots]

    fig = plt.figure(figsize=(7, 10), dpi=70)
    gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
    ax = plt.subplot(gs[0, 0])
    sPlot = ax.imshow(f_nots, cmap=plt.cm.jet, alpha=0.75)
    ax.set_xticks([k for k in range(nh)])
    ax.set_xticklabels([(k+1) for k in range(nh)])
    ax.set_yticks( [k for k in range(nl_nots)])
    ax.set_yticklabels([str('%.f-%.f' %(from_b_nots[g]+1,to_b_nots[g]+1)) for g in range(nl_nots)])
    ax.set_ylabel('Flujos (MW)')
    ax.set_xlabel('Hora (h)')
    for h in range(nh):
        fe = SF[pl_nots,:][:,pos_g] @ p_gt[:,h].x - SF[pl_nots,:]@dda_bus/Sb
        fv = (SF[pl_nots,:] @ A[pl_ts,:].T) @ f[:,h].x
        variable=fe+fv
        for l in range(nl_nots):
            ax.text( h, l, np.around(variable.T[l].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=4)
    ax = plt.subplot(gs[0, 1])
    fig.colorbar(sPlot, cax=ax, extend='both')
    ax.set_ylabel('Cargabilidad (%)')
    plt.savefig('flujo_lineas_nots.pdf')
    plt.show()


resta = np.zeros((nl_ts,nh))
for h in range(nh):
    dda_bus = Dda[h] * Load_bus
    f1 = SF[pl_ts,:][:,pos_g] @ p_gt.x[:,h] - SF[pl_ts,:]@dda_bus/Sb
    f2 = f.x[:,h] - (SF[pl_ts,:]@A[pl_ts,:].T) @ f.x[:,h] 
    resta[:,h] = f1-f2


from_b_ts = from_b[pl_ts]
to_b_ts = to_b[pl_ts]

fig = plt.figure(figsize=(7, 10), dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
ax = plt.subplot(gs[0, 0])
sPlot = ax.imshow(resta, cmap=plt.cm.jet, alpha=0.75)
ax.set_xticks([k for k in range(nh)])
ax.set_xticklabels([(k+1) for k in range(nh)])
ax.set_yticks( [k for k in range(nl_ts)])
ax.set_yticklabels([str('%.f-%.f' %(from_b_ts[g]+1,to_b_ts[g]+1)) for g in range(nl_ts)])
ax.set_ylabel('Flujos (MW)')
ax.set_xlabel('Hora (h)')
for h in range(nh):
    for l in range(nl_ts):
        ax.text( h, l, np.around(resta[l,h]*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=5)
#ax = plt.subplot(gs[0, 1])
#fig.colorbar(sPlot, cax=ax, extend='both')
#ax.set_ylabel('Cargabilidad (%)')
plt.savefig('flujo_lineas_ts.pdf')
plt.show()




#fig = plt.figure(figsize=(7, 10), dpi=70)
#gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
#ax = plt.subplot(gs[0, 0])
#sPlot = ax.imshow(f.x, cmap=plt.cm.jet, alpha=0.75)
#ax.set_xticks([k for k in range(nh)])
#ax.set_xticklabels([(k+1) for k in range(nh)])
#ax.set_yticks( [k for k in range(nl_nots+nl_ts)])
#ax.set_yticklabels([str('%.f-%.f' %(from_b[g]+1,to_b[g]+1)) for g in range(nl_nots+nl_ts)])
#ax.set_ylabel('Flujos (MW)')
#ax.set_xlabel('Hora (h)')
#all_f = np.zeros((nl_ts + nl_nots,nh))
#for h in range(nh): 
#    fe = SF[pl_nots,:][:,pos_g] @ p_gt[:,h].x - SF[pl_nots,:]@dda_bus/Sb
#    fv = (SF[pl_nots,:] @ A[pl_ts,:].T) @ f[:,h].x
#    variable=fe+fv
#    for g in range(nl_nots):
#        ax.text( h, g, np.around(variable.T[g].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=4)
#    all_f[pl_nots,h] = (fe + fv)*Sb
#    ax = plt.subplot(gs[0, 1])
#fig.colorbar(sPlot, cax=ax, extend='both')
#ax.set_ylabel('Cargabilidad (%)')
#plt.savefig('flujo_lineas.pdf')
#plt.show()