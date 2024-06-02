import gurobipy as gp
from gurobipy import *
import numpy as np
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
tras=1
virtual=0
ts=0
caso=0
loss=0
#Antes de limitar
prueba=0
if prueba:
    for i in range(len(sep['branch'])):
        sep['branch'][i,5]=1000

#Limitación de Flujos
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
L=6

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
indaux=np.array(indaux) #indice barras sin generacion y con demanda",
Cens=np.array(sparse((ones(len(indaux)), (indaux,range(len(indaux)))), (len(sep['bus']), len(indaux))).todense())
CENS=np.ones(len(indaux))*500


# Trabajo Previo a TS 
if ts:
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

    M = 1000000
    costo_ts = 10

# Initializing model
m = Model('UC_tarea_1')
m.setParam('OutputFlag', False)
m.Params.MIPGap = 1e-6
#m.setParam(GRB.Param.PoolSolutions, 10) # Limit how many solutions to collect
#m.setParam(GRB.Param.PoolSearchMode, 2) # systematic search for the k-best solutions

# VARIABLE DEFINITIONS

p_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='P_gt')                              # variable de generación para cada generador y para cada hora
b_gt = m.addMVar((ng,nh), vtype=GRB.BINARY, name='b_gt')                                        # variable binaria que indica estado de encendido/apagado de generador
pbar_gt = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='Pbar_gt')                        # potencia de reserva de cada generador en cada hora
CU = m.addMVar((ng,nh), vtype=GRB.CONTINUOUS, lb=0, name='CU')                                  # costos de encendido
if tras or virtual:
    d = m.addMVar((nb,nh), vtype=GRB.CONTINUOUS, ub=pi, lb=-pi, name="delta")                       # angulos de cada barra en cada hora
    f = m.addMVar((nl,nh), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="flujo")   # flujos de cada linea en cada hora
if ts:
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
    n_l = m.addMVar((nl,nh), vtype=GRB.BINARY, name='n_l')                                          # variable binaria complentaridad
# OPTIMIZATION PROBLEM
f_obj = 0 # OF

Cop = 0
Cup = 0
Cts = 0
Ce = 0

for h in range(nh):
    Cop += p_gt[:,h]*Sb @ np.diag(sep["units"][:,2]) @ p_gt[:,h]*Sb + sep["units"][:,1] @ p_gt[:,h]*Sb + sep["units"][:,0] @ b_gt[:,h] 
    Cup += CU[:,h].sum()
    if ts:
        Cts += costo_ts * (1 - s_ts[:,h]).sum()                 # Cuando la línea sale de servicio s_ts=0, se aplica el costo del TS
    if virtual:
        Ce +=CENS@pens[:,h]*Sb

f_obj = Cop + Cup + Cts + Ce

m.setObjective(f_obj, GRB.MINIMIZE)
m.getObjective()

#Creación de restricciones
if ts:
    index_sinlts = np.delete(np.arange(0,nl),pl_ts, axis=0)
#Restriccion 

for h in range(nh):
    #Balance Nodal y Reserva   
    if tras:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h] - Dda_bus/Sb, name="Balance nodal") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
    
        #Angulo de referencia
        m.addConstr(d[SL,h] == 0, name="SL")
    if virtual:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T @ f[:,h] == Cg@p_gt[:,h] +Cens@pens[:,h]- Dda_bus/Sb, name="Balance nodal") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        m.addConstr(-pens[:,h] >= -500/Sb, name='P_maxens')
        #Angulo de referencia
        m.addConstr(d[SL,h] == 0, name="SL")
    if loss:
        Dda_bus = Dda[h] * sep["bus"][:,2]
        m.addConstr( A.T@f[:,h]== Cg@p_gt[:,h]-(Dda_bus/Sb+0.5*abs(A.T)@p_loss[:,h]), name="Balance nodal") 
        m.addConstr( pbar_gt[:,h].sum() >= 1.1*Dda[h]/Sb, name="Reserva")
        #Angulo de referencia
        m.addConstr(d[SL,h] == 0, name="SL")
    else:
        m.addConstr(p_gt[:,h].sum() == Dda[h]/Sb,name='Bal')                       # Ecuación (8) Carrión - Arroyo
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
    if tras or virtual:
        m.addConstr(f[:,h]== b@d[from_b,h] - b@d[to_b,h])
        m.addConstr(-f[:,h] >= -FM/Sb, name = 'fp')
        m.addConstr(f[:,h] >= -FM/Sb, name = 'fn')
    if loss:
        m.addConstr(f[:,h]== bloss@d[from_b,h] - bloss@d[to_b,h]) #flujo
        m.addConstr(-f[:,h]-0.5*p_loss[:,h] >= -FM/Sb, name = 'fp')
        m.addConstr(f[:,h]-0.5*p_loss[:,h] >= -FM/Sb, name = 'fn')
        m.addConstr(f[:,h]==f_p[:,h]-f_n[:,h], name='Flujo_total') 
        m.addConstr(-f_p[:,h]>=-n_l[:,h]*FM/Sb)   #flujo positvo-restriccion de complementaridad
        m.addConstr(-f_n[:,h]>=(1-n_l[:,h])*(-FM/Sb)) #flujo nefativo-restriccion de complementaridad
        for l in range(L): 
            m.addConstr(-d_f[:,l,h]>=-FM/(Sb*L))
        for i in range(nl):
            m.addConstr(d_f[i,:,h].sum()==f_p[:,h]+f_n[:,h])
        for i in range(nl):
            kl = np.zeros(L)
            for l in range(L):
                kl[l] = (2*(l+1)-1)*(FM[i]/Sb)/L# 
            m.addConstr(p_loss[i,h] == (G[i]/(B[i]**2)*((kl*d_f[i,:,h]).sum())))
             
    if ts:
        m.addConstr( f[index_sinlts,h] == b_nots@d[from_b_nots,h] - b_nots@d[to_b_nots,h] )
        m.addConstr( f[index_sinlts,h] <= FM_lnots/Sb, name="flujo_max1")
        m.addConstr( f[index_sinlts,h] >= -FM_lnots/Sb, name="flujo_max2")
        for index_ts, ts in enumerate(pl_ts):  
            m.addConstr(f[ts,h] - (b[ts,ts] * d[from_b[ts],h] - b[ts,ts] * d[to_b[ts],h]) <= (1- s_ts[index_ts,h])* M   , name = 'fe_ts_p'+'_' +str(ts))
            m.addConstr(-f[ts,h] + (b[ts,ts] * d[from_b[ts],h] - b[ts,ts] * d[to_b[ts],h]) <= (1- s_ts[index_ts,h])* M  , name = 'fe_ts_n'+'_' +str(ts))
            m.addConstr(f[ts,h] <= FM[ts]/Sb * s_ts[index_ts,h] , name = 'fe_p_ts'+'_' +str(ts))
            m.addConstr(f[ts,h] >= -FM[ts]/Sb * s_ts[index_ts,h], name = 'fe_n_ts'+'_' +str(ts))


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

if False:
    m.write('UC_tarea_0.lp')      

status = m.Status
if status == GRB.Status.OPTIMAL:
    if ts:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cts = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Cts.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    elif virtual:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cens = %.2f ($)' % (m.objVal,Cop.getValue(),Cup.getValue(),Ce.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    else:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) ' % (m.objVal,Cop.getValue(),Cup.getValue()))
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (m.NumVars,m.NumConstrs,m.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))      
    # print ('Decision variables (Pg):')          
    
    # for h in range(nh):
    #     for l in range(p_gt.getAttr('x').shape[0]):
    #         print("p[%d,%d] = %.3f"%(l,h,p_gt.X[l,h]))

    # for h in range(nh):
    #     for l in range(b_gt.getAttr('x').shape[0]):
    #         print("n[%d,%d] = %.3f"%(l,h,b_gt.getAttr('x')[l,h]))   

    # print ('Lagrange multipliers:')            
    # for v in fixed.getConstrs():
    #     if v.pi > 1e-2:
    #         print('%s = %g ($/MWh)' % (v.ConstrName,v.pi))
    #if tras:
            #print ('Power flows:') 
            #for h in range(nh): 
            #    for l in range(nl):
            #        print('f[%.0f-%.0f][%.0f] = %.3f (MW)' % (sep['branch'][l,0], sep['branch'][l,1],h+1, f.x[l,h]))
    if ts:
        for h in range(nh):
            print('Lines candidates of Transmission Switching activates in hour %.0f'%h)
            for l in range(nl_ts):
                if s_ts.xn[l,h] < 0.5:
                    print('n[%.0f-%.0f] = %.3f' % (sep['branch'][pl_ts[l],0], sep['branch'][pl_ts[l],1], 1-s_ts.xn[l,h]))
                    print ('Power generation solution in hour %.0f (Pg): Load = %.0f' % (h,Dda[h]))
                    for g in range(ng):
                        print('p[%.0f,%.0f] = %.3f (MW)' % (g,h, Sb*p_gt.xn[g,h]))
        vX = zeros([nh,ng]); pX = zeros([nh,ng]); ptX = zeros([nh,ng]); CUX = zeros([nh,ng]); s_tsX = zeros([nh,nl_ts])
        for t in range(nh):
            vX[t] = b_gt.x[:,t]
            pX[t] = p_gt.x[:,t]
            ptX[t] = pbar_gt.x[:,t]
            CUX[t] = CU.x[:,t]
            s_tsX[t] = s_ts.x[:,t]
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
ax.set_ylabel('Tiempo (h)')
ax.set_xlabel('Generadores')
for g in range(ng):
    for h in range(nh):
        ax.text(g, h, np.around(p_gt.x[g,h].T*Sb,1).astype(int), color='black', ha='center', va='center', fontsize=5)
ax = plt.subplot(gs[0, 1])
fig.colorbar(sPlot, cax=ax, extend='both')
ax.set_ylabel('Cargabilidad (%)')
plt.show()

#tabla 2
#import matplotlib.pyplot as plt
#import numpy as np
#
#num_h, num_gen = p_gt.X.T.shape
#color_gen = np.random.rand(num_gen, 3) 
#fig, ax = plt.subplots(figsize=(10, 6))
#
#for i in range(num_gen):
#    ax.bar(range(num_h), p_gt.X.T[:, i]*Sb, bottom=np.sum(p_gt.X.T[:, :i]*Sb, axis=1), color=color_gen[i], label=f'Generador {i+1}')
#
#ax.set_xlabel('Hora')
#ax.set_ylabel('Potencia Generada')
#ax.set_title('Potencia Generada por Generador y Hora')
#ax.set_xticks(range(num_h))
#ax.set_xticklabels([f'H{h+1}' for h in range(num_h)])
## ax.legend()
#ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
## ax.legend(loc='upper right', bbox_to_anchor=(1, 0.5))
# Mostrar el gráfico
#plt.show()
if tras or virtual:
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
    #fig = plt.figure(figsize=(7, 10), dpi=150)
    #gs = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0)
    #ax = plt.subplot(gs[0, 0])
    #sPlot = ax.imshow(b_gt.x, cmap=plt.cm.jet, alpha=0.75)
    #ax.set_xticks([k for k in range(nh)])
    #ax.set_xticklabels([(k+1) for k in range(nh)])
    #ax.set_yticks( [k for k in range(ng)]   )
    #ax.set_yticklabels([str('%.f' %(ngen[g])) for g in range(ng)])
    # ax.set_ylabel('Switching (1/0) (MW)')
    #ax.set_ylabel('Estados (1/0)')
    #ax.set_xlabel('Hora (h)')
    #for g in range(ng):
    #    for h in range(nh):
    #        ax.text( h, g, np.around(b_gt.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
    #ax = plt.subplot(gs[0, 1])
    # fig.colorbar(sPlot, cax=ax, extend='both')
    #plt.savefig('Estados.pdf')
    #plt.show() 