from gurobipy import *
from numpy import savetxt, ones, zeros, diag, dot as mult, flatnonzero as find, pi
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.sparse import csr_matrix as sparse
from numpy import fmax, ones, zeros, arange, ix_, r_, flatnonzero as find
from numpy.linalg import solve, inv
t00 = time.time() #formulation time   

import case39 as mpc
sep = mpc.case39()

ramps = 1 
transmission=0
virtual=1
Switching=0
#Para SCUC con el sistema de transmision se coloca ramps y transmission igual a 1
#Para generadores virtuales con el sistema de transmision se coloca ramps y virtual igual a 1
#Para TS con el sistema de transmision se coloca ramps y switching igual a 1
# Load data
nb = len(sep['bus'])
load_hourly = sep['demand']; 
# dem_base = sep['bus'][:,2]
# total_dem = dem_base.sum()

T = len(load_hourly); # Ventana horaria
ngen=[1,2,3,4,5,6,7,8,9,10]

#Generation data
Gen = sep['units'] # generation data
ng = len(Gen) # Número de generadores
nh = len(sep['demand'])
pos_g = sep['pos_g'] # Ubicación de los generadores
R = 0.1 # 10% de reserva.
a = Gen[:,2] 
b = Gen[:,1] 
c = Gen[:,0] 
CU = Gen[:,8] 
p0 = sep['p02006'] # Potencia inicial generadores.
p02 = sep['pp02006']
PM = Gen[:,3] # Pmax
Pm = Gen[:,4] # Pmin
SU = SD = Pm # Rampas on/off = Pmin
RU = RD = Gen[:,11]  # Rampas up/down
UT = Gen[:,5] # t minimo encendido
DT = Gen[:,6] # t mínimo apagado
V0 = sep['V0']# Estado inicial unidades.
U0 = sep['U0'] # Cantidad de tiempo que lleva encendida unidad 
S0 = sep['S0'] # Cantidad de tiempo que lleva apagada la unidad.
G = sep['G'] # Number of periods unit must be initially ON due to its minimum up time constraint
L = sep['L'] # Number of periods unit j must be initially OFF due to its minimum down time constraint.
Cg = np.array(sparse((ones(ng), (sep['gen'][:,0]-1, range(ng))), (nb, ng)).todense())
# Transmission data
nl = len(sep['branch']) # Cantidad de líneas
FM = sep['branch'][:,5] # Límite térmico
SF = sep['SF'] # shift-factors
fr = sep['branch'][:,0]
to = sep['branch'][:,1]
Sb = sep['baseMVA'] 
M = 5 * pi * Sb
PTDF = sep['PTDF'] # shift-factors
I_PTDF = sep['I_PTDF'] # I - PTDF matrix
#Transmission switching
#Lineas in
pos_le=find(sep['branch'][:,17]==0)
nl_e=len(pos_le)
#Lineas out
pos_ls=find(sep['branch'][:,17]==1)
nl_s=len(pos_ls)
CTS=0*ones(nl_s) #Si quieres agregar algun costo para el switchin cambia el cero por otro numero
#flujos in 
fr_i=sep['branch'][pos_le,0]
to_i=sep['branch'][pos_le,1]
#flujos out
fr_ls=sep['branch'][pos_ls,0]
to_ls=sep['branch'][pos_ls,1]



# Initializing m
model = Model('UC')
model.Params.MIPGap = 1e-6
model.Params.OutputFlag = 0
Pd=[]
for h in range(nh):
    Dda_total=sep['demand'][h] #Demanda total hora i
    Dda_barra=np.dot(Dda_total,sep['bus'][:,2]) #Demanda en cada barra.
    Pd.append(Dda_barra)
Pd=np.transpose(np.array(Pd)) #Demanda en cada hora para cada barra.
# Generadores virtuales\n",
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
# VARIABLE DEFINITIONS
p   = model.addMVar((ng, T), vtype=GRB.CONTINUOUS, lb=0, name='p') # pot
p_t = model.addMVar((ng, T), vtype=GRB.CONTINUOUS, lb=0, name= 'P_d') # pot disp.
v   = model.addMVar((ng, T), vtype=GRB.BINARY, name='v') # Variable binaria UC
cu  = model.addMVar((ng, T), vtype=GRB.CONTINUOUS, lb=0, name='c_up') # Costo encendido
f   = model.addMVar((nl, T), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='f') # flujo por lineas.
#Varible ens
if virtual:
    pens = model.addMVar((len(indaux),nh), vtype=GRB.CONTINUOUS, lb=0, name='P_ens')
#TS variable
if Switching:
    f_e = model.addMVar((nl_e, T), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='fe') # flujo lineas no candidatas
    x_s = model.addMVar((nl_s, T), vtype=GRB.BINARY, name='ts')  # Variable binaria de TS
    f_s = model.addMVar((nl_s, T), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='fs') # flujo líneas candidatas
# OPTIMIZATION PROBLEM
# Start-up costs
CUp = cu[:,:].sum()
# Operational costs
COp = 0; 
for t in range(T):
    COp += p[:,t] @ diag(a) @ p[:,t] + b @ p[:,t] + c @ v[:,t]
Ce = 0
if virtual:
    for t in range(T):
        Ce +=CENS@pens[:,t]
CTs = 0
# TS costs
if Switching:
    if nl_s > 0:
        CTs = 0; 
        for t in range(T):
           CTs += (sum(CTS) - CTS @ x_s[:,t])
    else:
        CTs = 0
#Función Objetivo:
model.setObjective(COp + CUp + Ce + CTs, GRB.MINIMIZE)


# s.t
#Linea 14-15
if Switching:
    s=0
    for i in range(len(sep['branch'])):
        if sep['branch'][:,0][s]==14 and sep['branch'][:,1][s]==15:
            FM[s]=75
        s+=1
if virtual: #Aqui modificas el flujo de la linea para los generadores virtuales
    s=0
    for i in range(len(sep['branch'])):
        if sep['branch'][:,0][s]==14 and sep['branch'][:,1][s]==15:
            FM[s]=50
        s+=1

CU_aux = diag(CU)
for t in range(T):
    if t == 0:
        aux = CU_aux @ v[:,0] - CU * V0
    else:
        aux = CU_aux @ v[:,t] - CU_aux @ v[:,t-1]        
    model.addConstr(cu[:,t] >= aux, name = 'Cup')   

# Balance
if virtual:
    for t in range(T):
        model.addConstr(p[:,t].sum()+pens[:,t].sum()== load_hourly[t], name = 'Bal') 
else:
    for t in range(T):
        model.addConstr(p[:,t].sum()== load_hourly[t], name = 'Bal') 

# Reserva
for t in range(T):
    model.addConstr(p_t[:,t].sum() >= (1+R) * load_hourly[t], name = 'Res') #Potencia máxima disponible contenmpla 10% de reserva.
# p_min
Pm_aux = diag(Pm)
for t in range(T):
    model.addConstr(p[:,t] >= Pm_aux @ v[:,t], name = 'p_m') 

# Pmax
PM_aux = diag(PM)
for t in range(T):
    model.addConstr(p[:,t] <= p_t[:,t], name = 'p1_M') 
    model.addConstr(p_t[:,t] <= PM_aux @ v[:,t], name = 'p2_M')     
        

#rampas
if ramps:             
    SU_aux = diag(SU); RU_aux = diag(RU); SD_aux = diag(SD); RD_aux = diag(RD);
    # ramp-up and startup ramp rates 
    for t in range(T):
        if t == 0: #rhs
            aux = p0 + RU * V0 + SU_aux @ v[:,t] - SU * V0 + PM - PM_aux @ v[:,t]
        else:
            aux = p[:,t-1] + RU_aux @ v[:,t-1] + SU_aux @ v[:,t] - SU_aux @ v[:,t-1] + PM - PM_aux @ v[:,t]
        model.addConstr(p_t[:,t] <= aux, name ='SUp_ramp')        
    # shutdown ramp rates 
    for t in range(T-1):
        model.addConstr(p_t[:,t] <= PM_aux @ v[:,t+1] + SD_aux @ v[:,t] - SD_aux @ v[:,t+1], name = 'SDown_ramp')           
    # ramp-down limits
    for t in range(T):
        if t == 0: #rhs
            aux1 = p0 - p[:,t] 
            aux2 = RD_aux @ v[:,t] + SD * V0 - SD_aux @ v[:,t] + PM - PM * V0
        else:
            aux1 = p[:,t-1] - p[:,t] 
            aux2 = RD_aux @ v[:,t] + SD_aux @ v[:,t-1] - SD_aux @ v[:,t] + PM - PM_aux @ v[:,t-1]
        model.addConstr(aux1 <= aux2, name ='Down_ramp')    
       
        
#           
# Transmission power flows 
if virtual:
    for t in range(T):
        model.addConstr(-SF@Cg@p[:,t]+SF@Pd[:,t]==f[:,t],name='flux')
        model.addConstr(-SF@Cg@p[:,t]-SF@Cens@pens[:,t]+SF@Pd[:,t]>=-FM,name='fluxmax')
        model.addConstr(SF@Cg@p[:,t]+SF@Cens@pens[:,t]-SF@Pd[:,t]>=-FM,name='fluxmax')
        model.addConstr(-pens[:,t] >= -500, name='P_maxens')
if transmission:    
    SF_p = SF[:,pos_g];
    for t in range(T):
        load_bus = load_hourly[t] * sep['bus'][:,2] 
        faux = SF_p @ p[:,t] - mult(SF,load_bus)  
        model.addConstr(f[:,t] == faux, name = 'fe') 
        model.addConstr(f[:,t] <= FM, name = 'fe_p') 
        model.addConstr(f[:,t] >= -FM, name = 'fe_n') 
if Switching: 
    # Lineas existentes o no candidatas
    SFe=SF[pos_le,:];  SFe_p = SFe[:,pos_g]; PTDFe = PTDF[pos_le,:]; FMe_aux = FM[pos_le]; 
    for t in range(T):
        load_bus = load_hourly[t] * sep['bus'][:,2]
        fe_aux = SFe_p @ p[:,t] - mult(SFe,load_bus)
        fv_aux = PTDFe[:,pos_ls] @ f_s[:,t]
        model.addConstr(f_e[:,t] == fe_aux + fv_aux, name = 'fe')
        model.addConstr(f_e[:,t] <= FMe_aux, name = 'fe_p')
        model.addConstr(f_e[:,t] >= -FMe_aux, name = 'fe_n')                  

    # Transmission switching
    SFs=SF[pos_ls,:]; SFs_p = SFs[:,pos_g]; PTDFs=PTDF[pos_ls,:]; I_PTDFs=I_PTDF[pos_ls,:]; FMs_aux = diag(FM[pos_ls])
    for t in range(T):
        load_bus = load_hourly[t] * sep['bus'][:,2]
        fe_aux = SFs_p @ p[:,t] - mult(SFs,load_bus) #SF*Pneta
        fv_aux = I_PTDFs[:,pos_ls] @ f_s[:,t]   # (I - SF*A.T)*f_ts
        model.addConstr(fe_aux - fv_aux <= FMs_aux @ x_s[:,t], name = 'fs1_p') # 1
        model.addConstr(fe_aux - fv_aux >= -FMs_aux @ x_s[:,t], name = 'fs1_n')
        model.addConstr(f_s[:,t] <= (1-x_s[:,t]) * M, name = 'fs2_p') # 2
        model.addConstr(f_s[:,t] >= -(1-x_s[:,t]) * M, name = 'fs2_n') 

t11 = time.time() #formulation time

if 1: # write LP
    model.write('Osep.lp'); 
    
# SOLVER & INFO
model.optimize()
status = model.Status
if ramps: 
    if status == GRB.Status.OPTIMAL:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($)' % (model.objVal,model.objVal-CUp.getValue(),CUp.getValue()))       
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (model.NumVars,model.NumConstrs,model.DNumNZs))
        print('Pg=%.2f')
        #for h in range(nh):
        #    for l in range(p.getAttr('x').shape[0]):
        #        print("p[%d,%d] = %.3f"%(l,h,p.X[l,h]))
        vX = zeros([T,ng]); pX = zeros([T,ng]); ptX = zeros([T,ng]); CUX = zeros([T,ng])
        for t in range(T):
            vX[t] = v.x[:,t]
            pX[t] = p.x[:,t]
            ptX[t] = p_t.x[:,t]
            CUX[t] = cu.x[:,t]
        if 0:
            savetxt('UC.txt', vX, delimiter=",", fmt='%d')
            savetxt('UC_p.txt', pX, delimiter="     ", fmt='%.0f') #csv
            savetxt('UC_pt.txt', ptX, delimiter="     ", fmt='%.0f')
            savetxt('UC_CU.txt', CUX, delimiter=",", fmt='%d')
        print('=> Formulation time: %.4f (s)'% (t11-t00))
        print('=> Solver time: %.4f (s)' % (model.Runtime))  
    elif status == GRB.Status.INF_OR_UNBD or \
       status == GRB.Status.INFEASIBLE  or \
       status == GRB.Status.UNBOUNDED:
       print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)
       model.computeIIS() 
       model.write("MILP_UC2013.ilp")    
       
       # fr = sep['branch'][g,0]
       # to = sep['branch'][g,1]
    # f --> f.x
    if 1:    
        fig = plt.figure(figsize=(14, 22), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(p.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(ng)]   )
        ax.set_yticklabels([str('%.f' %(ngen[g])) for g in range(ng)])
        ax.set_ylabel('Potencias (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(ng):
            for h in range(T):
                ax.text( h, g, np.around(p.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        plt.savefig('Potencias.pdf')
        plt.show()  
if transmission: 
    if status == GRB.Status.OPTIMAL:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($)' % (model.objVal,model.objVal-CUp.getValue(),CUp.getValue()))       
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (model.NumVars,model.NumConstrs,model.DNumNZs))
        print('Pg=%.2f')
        for h in range(nh):
            for l in range(p.getAttr('x').shape[0]):
                print("p[%d,%d] = %.3f"%(l,h,p.X[l,h]))
        vX = zeros([T,ng]); pX = zeros([T,ng]); ptX = zeros([T,ng]); CUX = zeros([T,ng])
        for t in range(T):
            vX[t] = v.x[:,t]
            pX[t] = p.x[:,t]
            ptX[t] = p_t.x[:,t]
            CUX[t] = cu.x[:,t]
        if 0:
            savetxt('UC.txt', vX, delimiter=",", fmt='%d')
            savetxt('UC_p.txt', pX, delimiter="     ", fmt='%.0f') #csv
            savetxt('UC_pt.txt', ptX, delimiter="     ", fmt='%.0f')
            savetxt('UC_CU.txt', CUX, delimiter=",", fmt='%d')
        print('=> Formulation time: %.4f (s)'% (t11-t00))
        print('=> Solver time: %.4f (s)' % (model.Runtime))  
    elif status == GRB.Status.INF_OR_UNBD or \
       status == GRB.Status.INFEASIBLE  or \
       status == GRB.Status.UNBOUNDED:
       print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)
       model.computeIIS() 
       model.write("MILP_UC2013.ilp")    
       
       # fr = sep['branch'][g,0]
       # to = sep['branch'][g,1]
    # f --> f.x
    if 1:    
        fig = plt.figure(figsize=(14, 22), dpi=30)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(f.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(nl)]   )
        ax.set_yticklabels([str('%.f-%.f' %(fr[g],to[g])) for g in range(nl)])
        ax.set_ylabel('Flujos (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(nl):
            for h in range(T):
                ax.text( h, g, np.around(f.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        ax = plt.subplot(gs[0, 1])
        fig.colorbar(sPlot, cax=ax, extend='both')
        ax.set_ylabel('Cargabilidad (%)')
        plt.savefig('flujo_lineas.pdf')
        plt.show()
    if 1:    
        fig = plt.figure(figsize=(14, 22), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(p.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(ng)]   )
        ax.set_yticklabels([str('%.f' %(ngen[g])) for g in range(ng)])
        ax.set_ylabel('Potencias (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(ng):
            for h in range(T):
                ax.text( h, g, np.around(p.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        plt.savefig('Potencias.pdf')
        plt.show()  
if virtual: 
    if status == GRB.Status.OPTIMAL:
        print ('Cost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + Cens = %.2f ($)' % (model.objVal,model.objVal-CUp.getValue()-Ce.getValue(),CUp.getValue(),Ce.getValue()))   
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (model.NumVars,model.NumConstrs,model.DNumNZs)) 
        vX = zeros([T,ng]); pX = zeros([T,ng]); ptX = zeros([T,ng]); CUX = zeros([T,ng])
        for t in range(T):
            vX[t] = v.x[:,t]
            pX[t] = p.x[:,t]
            ptX[t] = p_t.x[:,t]
            CUX[t] = cu.x[:,t]
        if 0:
            savetxt('UC.txt', vX, delimiter=",", fmt='%d')
            savetxt('UC_p.txt', pX, delimiter="     ", fmt='%.0f') #csv
            savetxt('UC_pt.txt', ptX, delimiter="     ", fmt='%.0f')
            savetxt('UC_CU.txt', CUX, delimiter=",", fmt='%d')
        print('=> Formulation time: %.4f (s)'% (t11-t00))
        print('=> Solver time: %.4f (s)' % (model.Runtime))  
    elif status == GRB.Status.INF_OR_UNBD or \
       status == GRB.Status.INFEASIBLE  or \
       status == GRB.Status.UNBOUNDED:
       print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)
       model.computeIIS() 
       model.write("MILP_UC2013.ilp")    
       
       # fr = sep['branch'][g,0]
       # to = sep['branch'][g,1]
    # f --> f.x
    if 1:    
        fig = plt.figure(figsize=(14, 22), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(f.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(nl)]   )
        ax.set_yticklabels([str('%.f-%.f' %(fr[g],to[g])) for g in range(nl)])
        ax.set_ylabel('Flujos (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(nl):
            for h in range(T):
                ax.text( h, g, np.around(f.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        ax = plt.subplot(gs[0, 1])
        fig.colorbar(sPlot, cax=ax, extend='both')
        ax.set_ylabel('Cargabilidad (%)')
        plt.savefig('flujo_lineas.pdf')
        plt.show()        
    if 1:
         fig = plt.figure(figsize=(7*ng, 10*ng), dpi=150)
         gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
         ax = plt.subplot(gs[0, 0])
         sPlot = ax.imshow(v.x, cmap=plt.cm.jet, alpha=0.75)
         ax.set_xticks([k for k in range(T)])
         ax.set_xticklabels([(k+1) for k in range(T)])
         ax.set_yticks( [k for k in range(ng)]   )
         ax.set_yticklabels([str('%.f' %(ngen[g])) for g in range(ng)])
         # ax.set_ylabel('Switching (1/0) (MW)')
         ax.set_ylabel('Estados (1/0)')
         ax.set_xlabel('Hora (h)')
         for g in range(ng):
             for h in range(T):
                 ax.text( h, g, np.around(v.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
         #ax = plt.subplot(gs[0, 1])
         # fig.colorbar(sPlot, cax=ax, extend='both')
         plt.savefig('Estados.pdf')
         plt.show()       
    if 1:    
        fig = plt.figure(figsize=(14, 22), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(p.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(ng)]   )
        ax.set_yticklabels([str('%.f' %(ngen[g])) for g in range(ng)])
        ax.set_ylabel('Potencias (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(ng):
            for h in range(T):
                ax.text( h, g, np.around(p.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        plt.savefig('Potencias.pdf')
        plt.show()  

if Switching:
    if status == GRB.Status.OPTIMAL:
        if nl_s > 0:
            print ('\nCost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($) + CTs = %.2f ($)' % (model.objVal,COp.getValue(),CUp.getValue(),CTs.getValue()))   
            for l in range(nl_s): 
                print('Lin %d-%d / ON =>  %d hours' % (sep['branch'][pos_ls[l],0],sep['branch'][pos_ls[l],1], int(round(sum(x_s.x[l,:])))))
            if 0:                
                for l in range(nl_s): 
                    for t in range(T):                
                        if x_s.x[l,t] > 0:
                            print('Lin %d-%d / ON => %d hours' % (sep['branch'][pos_ls[l],0],sep['branch'][pos_ls[l],1],x_s.x[l,t]))
        else:
            print ('\nCost = %.2f ($) => Cop = %.2f ($) + Cup = %.2f ($)' % (model.objVal,COp.getValue(),CUp.getValue()))   
        print('num_Vars =  %d / num_Const =  %d / num_NonZeros =  %d' % (model.NumVars,model.NumConstrs,model.DNumNZs)) #print('num_Vars =  %d / num_Const =  %d' % (len(m.getVars()), len(m.getConstrs())))
        #PF = zeros([S,T,nl-1]);
        #for s in range(S):
          # for t in range(T):
            #   PF[s,t,:] = (abs(np.divide(f_e.x[:,t,s],FMe_aux)))*100
        # vX = zeros([T,ng]); pX = zeros([T,ng]); ptX = zeros([T,ng]); CUX = zeros([T,ng])
        # for t in range(T):
        #     vX[t] = v.x[:,t]
        #     pX[t] = p.x[:,t]
        #     ptX[t] = p_t.x[:,t]
        #     CUX[t] = cu.x[:,t]
        print('=> Formulation time: %.4f (s)'% (t11-t00))
        print('=> Solver time: %.4f (s)' % (model.Runtime))  
      
    
    elif status == GRB.Status.INF_OR_UNBD or \
       status == GRB.Status.INFEASIBLE  or \
       status == GRB.Status.UNBOUNDED:
       print('The model cannot be solved because it is infeasible or unbounded => status "%d"' % status)
       model.computeIIS() # compute an Irreducible Inconsistent Subsystem (IIS)
       model.write("MILP_UC.ilp")    
       
    
       # fr = sep['branch'][g,0]
       # to = sep['branch'][g,1]
       # f --> f.x
    if Switching:    
        fig = plt.figure(figsize=(14, 22), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(f_e.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(nl_e)]   )
        ax.set_yticklabels([str('%.f-%.f' %(fr[g],to[g])) for g in range(nl_e)])
        ax.set_ylabel('Flujo lineas existentes (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(nl_e):
            for h in range(T):
                ax.text( h, g, np.around(f_e.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        ax = plt.subplot(gs[0, 1])
        fig.colorbar(sPlot, cax=ax, extend='both')
        ax.set_ylabel('Cargabilidad (%)')
        plt.savefig('flujo_lineas.pdf')
        plt.show()
        
        
        # fig = plt.figure(figsize=(15*nl_s, 22*nl_s), dpi=300)
        # gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        # ax = plt.subplot(gs[0, 0])
        # sPlot = ax.imshow(f_s.x, cmap=plt.cm.jet, alpha=0.75)
        # ax.set_xticks([k for k in range(T)])
        # ax.set_xticklabels([(k+1) for k in range(T)])
        # ax.set_yticks( [k for k in range(nl_s)] )
        # ax.set_yticklabels([str('%.f-%.f' %(fr_ls[g],to_ls[g])) for g in range(nl_s)])
        # ax.set_ylabel('Flujo lineas candidatas (MW)')
        # ax.set_xlabel('Hora (h)')
        # for g in range(nl_s):
        #     for h in range(T):
        #         ax.text( h, g, np.around(f_s.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        # # ax = plt.subplot(gs[0, 1])
        # # fig.colorbar(sPlot, cax=ax, extend='both')
        # # ax.set_ylabel('Cargabilidad (%)')
        # plt.savefig('flujo_linea_switching.pdf')
        # plt.show()   
        
    if True:
         fig = plt.figure(figsize=(7*nl_s, 10*nl_s), dpi=150)
         gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
         ax = plt.subplot(gs[0, 0])
         sPlot = ax.imshow(x_s.x, cmap=plt.cm.jet, alpha=0.75)
         ax.set_xticks([k for k in range(T)])
         ax.set_xticklabels([(k+1) for k in range(T)])
         ax.set_yticks( [k for k in range(nl_s)]   )
         ax.set_yticklabels([str('%.f-%.f' %(fr_ls[g],to_ls[g])) for g in range(nl_s)])
         # ax.set_ylabel('Switching (1/0) (MW)')
         ax.set_xlabel('Hora (h)')
         for g in range(nl_s):
             for h in range(T):
                 ax.text( h, g, np.around(x_s.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
         #ax = plt.subplot(gs[0, 1])
         # fig.colorbar(sPlot, cax=ax, extend='both')
         plt.savefig('lineas_switching.pdf')
         plt.show()   
    if 1:    
        fig = plt.figure(figsize=(14, 22), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[75,1], wspace=0)
        ax = plt.subplot(gs[0, 0])
        sPlot = ax.imshow(p.x, cmap=plt.cm.jet, alpha=0.75)
        ax.set_xticks([k for k in range(T)])
        ax.set_xticklabels([(k+1) for k in range(T)])
        ax.set_yticks( [k for k in range(ng)]   )
        ax.set_yticklabels([str('%.f' %(ngen[g])) for g in range(ng)])
        ax.set_ylabel('Potencias (MW)')
        ax.set_xlabel('Hora (h)')
        for g in range(ng):
            for h in range(T):
                ax.text( h, g, np.around(p.x.T[h,g],1).astype(int), color='black', ha='center', va='center', fontsize=12)
        plt.savefig('Potencias.pdf')
        plt.show()  
