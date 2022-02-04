''' Created on 6/01/2022

* This module captures all the metabolic engineering functions considered for the current research:

These models are based on the literature.
Models:

   -- FBA -- my_FBA(infeas(optional),S,LB,UB,rxn,met,biomas,chemical,KO(optional))
                    return fluxes
   -- FVA -- my_FVA(infeas(optional),minprod,,S,LB,UB,rxn,met,biomas,chemical,KO(optional)))
                    return fluxes
   -- ReacKnock -- my_reacknock(infeas(optional),BM(optional),minprod,S,LB,UB,rxn,met,biomas,chemical,KO,k)
                    return del_strat, vs, s


And, the algorithms for the bi-level case proposed in the current research

Algorithms:

   -- algorithm -- my_algorithm(infeas(optional),S,LB,UB,rxn,met,biomas,chemical,KO,k,minprod)
   -- algorithm2 -- my_algorithm(infeas(optional),S,LB,UB,rxn,met,biomas,chemical,KO,k,minprod)
   -- algorithm3 -- my_algorithm(infeas(optional),S,LB,UB,rxn,met,biomas,chemical,KO,k,minprod)
                    return del_strat, vs, s

The difference between this algorithms is in the way the cuts are introduced

Author: @j-alex-vindel

'''

import gurobipy as gp
import math as ma
from numpy import inf
from gurobipy import GRB
from itertools import combinations
import numpy as np

def my_FBA(infeas=1e-6,**Params):
    '''
    my_FBA(infeas=1e-6(optional),S=S,LB=LB,UB=UB,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO(optional))
        return fluxes
    '''
    if len(Params.keys()) < 6:
        return f'Insuficient Parameters. my_FBA(infeas(optional), S, LB, UB, rxn, met, biomas, chemical,KO(optional))'

    S,LB,UB,rxn,met = Params['S'], Params['LB'], Params['UB'], Params['rxn'], Params['met']
    biomas,chemical = Params['biomas'], Params['chemical']

    M = [i for i in range(len(Params['rxn']))]
    N = [i for i in range(len(Params['met']))]

    if 'KO' not in Params:
        y = [1 for i in M]
        print('* Full Metabolic Network *')
    elif 'KO' in Params:
        y = [0 if i in [Params['rxn'].index(r) for r in Params['KO']] else 1 for i in M]
        print('Knockout Index:',[Params['rxn'].index(r) for r in Params['KO']],['y_%s'%(i) for i in Params['KO']],sep=' -> ')

    print('**** Solving FBA ****')
    print('Current Infeasibility:',infeas,sep=' -> ')
    print(f'# Variables (reactions in the network): {len(M)}')
    o = gp.Model()

    v = o.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')

    o.setObjective(1*v[biomas],GRB.MAXIMIZE)

    o.addConstrs((gp.quicksum(S[i,j] *v[j] for j in M) == 0 for i in N), name='Stoichiometry')

    o.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='lower_bound')
    o.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='upper_bound')

    o.addConstrs((LB[j] <= v[j] for j in M), name='lb')
    o.addConstrs((v[j] <= UB[j] for j in M), name='ub')

    o.Params.OptimalityTol = infeas
    o.Params.IntFeasTol = infeas
    o.Params.FeasibilityTol = infeas
    o.params.LogToConsole = False
    o.optimize()

    if o.status == GRB.OPTIMAL:
        fluxes = [o.getVarByName('v[%d]'%j).x for j in M]
        # print('*'*3,'VALUES','*'*3)
        # print('Biomass:',fluxes[biomas],sep=' -> ')
        # print('Chemical:',fluxes[chemical],sep=' -> ')

    if o.status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        fluxes = ['~' for i in M]
        # print('INFEASIBLE or UNBOUNDED')
    print('*** FINISHED ***')
    return fluxes

def my_FVA(infeas=1e-6,**Params):
    '''
    my_FVA(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO(optional))
        return fluxes
    '''
    if len(Params.keys()) < 8:
        return f'Insuficient Parameters. my_FBA(infeas(optional), minprod, S, LB, UB, rxn, met, biomas, chemical,KO(optional))'

    S,LB,UB,rxn,met = Params['S'], Params['LB'], Params['UB'], Params['rxn'], Params['met']
    biomas,chemical,minprod = Params['biomas'], Params['chemical'], Params['minprod']

    M = [i for i in range(len(Params['rxn']))]
    N = [i for i in range(len(Params['met']))]

    if 'KO' not in Params:
        y = [1 for i in M]
        print('* Full Metabolic Network *')
    elif 'KO' in Params:
        y = [0 if i in [Params['rxn'].index(r) for r in Params['KO']] else 1 for i in M]
        print('Knockout Index:',[Params['rxn'].index(r) for r in Params['KO']],['y_%s'%(i) for i in Params['KO']],sep=' -> ')

    print('Current Infeasibility:',infeas,sep=' -> ')
    print('**** Solving FVA ****')
    print(f'# Variables (reactions in the network): {len(M)}')
    o = gp.Model()

    v = o.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')

    o.setObjective(1*v[chemical],GRB.MAXIMIZE)

    o.addConstrs((gp.quicksum(S[i,j] *v[j] for j in M
    ) == 0 for i in N), name='Stoichiometry')

    o.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='lower_bound')
    o.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='upper_bound')

    o.addConstrs((LB[j] <= v[j] for j in M), name='lb')
    o.addConstrs((v[j] <= UB[j] for j in M), name='ub')

    o.addConstr(v[biomas] >= minprod, name='target')

    o.Params.OptimalityTol = infeas
    o.Params.IntFeasTol = infeas
    o.Params.FeasibilityTol = infeas
    o.params.LogToConsole = False
    o.optimize()

    if o.status == GRB.OPTIMAL:
        fluxes = [o.getVarByName('v[%d]'%j).x for j in M]
        # print('*'*3,'VALUES','*'*3)
        # print('Biomass:',fluxes[biomas],sep=' -> ')
        # print('Chemical:',fluxes[chemical],sep=' -> ')

    if o.status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        fluxes = ['~' for i in M]
        # print('INFEASIBLE or UNBOUNDED')
    print('*** FINISHED ***')
    return fluxes

def my_reacknock(infeas=1e-6,BM=1000,**Params):
    '''
    my_reacknock(infeas=1e-6(optional),BM=1000(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    if len(Params.keys()) < 10:
        return f'Insuficient Parameters. my_FBA(infeas(optional), BM(optional), minprod, S, LB, UB, rxn, met, biomas, chemical, KO, k)'

    S,LB,UB,rxn,met,KO = Params['S'], Params['LB'], Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'], Params['chemical'], Params['minprod'], Params['k']
    M = [i for i in range(len(Params['rxn']))]
    N = [i for i in range(len(Params['met']))]

    print('**** Solving ReacKnock ****')
    print(f'# Variables (reactions in the network): {len(M)}')
    print('Current Infeasibility:',infeas,sep=' -> ')
    print('KO set: ',len(KO), ' reactions')

    lb = LB.copy()
    lb[biomas] = minprod

    m = gp.Model()

    # Variables

    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    #  Dual Variables

    l = m.addVars(N,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='lambda')
    a1 = m.addVars(M,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='alpha1')
    b1 = m.addVars(M,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='beta1')
    a2 = m.addVars(M,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='alpha1')
    b2 = m.addVars(M,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='beta1')
    a = m.addVars(M,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='alpha')
    b = m.addVars(M,lb=0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='beta')

    m.setObjective((1*v[chemical]),GRB.MAXIMIZE)

    # Knapsack Constraints

    m.addConstrs((y[j] == 1 for j in M if j not in KO), name='y_essentials')

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    # Stoichiometric Constraints

    m.addConstrs((gp.quicksum(S[i,j] * v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    # Dual objective

    m.addConstr((v[biomas] == (sum(a1[j]*UB[j] - b1[j]*lb[j] for j in M)
     + sum(a2[j]*UB[j] - b2[j]*lb[j] for j in M))),name='dual-objective')

    # Dual constraints

    m.addConstrs((gp.quicksum(S.transpose()[i,j]*l[j] for j in N)
              - b[i]
              + a[i] - b2[i] + a2[i]
               == 0 for i in M if i !=biomas)
             ,name='S_dual')

    m.addConstr((gp.quicksum(S.transpose()[biomas,j]*l[j] for j in N)
            - b[biomas]
            + a[biomas]
            - b2[biomas] + a2[biomas] == 1), name='Sdual_t')

    # linearization

    m.addConstrs((a1[j] <= BM*y[j] for j in M),name='l1_a1')

    m.addConstrs((a1[j] >= -BM*y[j] for j in M),name='l2_a1')

    m.addConstrs((a1[j] <= a[j] + BM*(1-y[j]) for j in M),name='l3_a1')

    m.addConstrs((a1[j] >= a[j] - BM*(1-y[j]) for j in M),name='l4_a1')

    m.addConstrs((b1[j] <= BM*y[j] for j in M),name='l1_b1')

    m.addConstrs((b1[j] >= -BM*y[j] for j in M),name='l2_b1')

    m.addConstrs((b1[j] <= b[j] + BM*(1-y[j]) for j in M),name='l3_b1')

    m.addConstrs((b1[j] >= b[j] - BM*(1-y[j]) for j in M),name='l4_b1')

    # Bounds

    m.addConstrs((lb[j]*y[j] <= v[j] for j in M), name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M), name='UB')

    m.addConstrs((lb[j] <= v[j] for j in M),name='lb')
    m.addConstrs((v[j] <= UB[j] for j in M),name='ub')

    # m.setParam(GRB.Param.PoolSolutions,10)
    # m.setParam(GRB.Param.PoolSearchMode,2)
    # m.setParam(GRB.Param.PoolGap,.1)
    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    m.Params.NodefileStart = 0.5
    m.optimize()

    # nsolutions = m.SolCount

    s = m.Runtime
    del_strat = []
    if m.status == GRB.OPTIMAL:
        chem = m.getObjective().getValue()
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vs = [m.getVarByName('v[%d]'%j).x for j in M]
        print('*'*4,'SOLUTION','*'*4)
        print('Time (s):',s,sep=' -> ')
        print('Chemical Overproduction:',chem,sep=' -> ')
        print('Biomass production:',vs[biomas],sep=' -> ')
        print('**** Deletion Strategy: ****')
        for i in M:
            if ys[i] < .5:
                print('*'*2,i,rxn[i],sep=' -> ')
                del_strat.append(rxn[i])
        # print('*'*3,'Pool Solutions:','*'*3)
        # for e in range(nsolutions):
        #     m.setParam(GRB.Param.SolutionNumber,e)
        #     print('Succinate Overproduction:','%g'%m.PoolObjVal,sep=' -> ')
        #     print('Biomass Production:',v[biomas].Xn,sep=' -> ')
        #     if e <= nsolutions:
        #         m.setParam(GRB.Param.SolutionNumber,e)
        #         for i in M:
        #             if y[i].XN < .5:
        #                 print('**Knockout Strategy:', rxn[i],sep=' -> ')

    if m.status in (GRB.INFEASIBLE,GRB.INF_OR_UNBD,GRB.UNBOUNDED):
        print('Model status: *** INFEASIBLE or UNBOUNDED ***')
        ys = ['$' for i in M]
        vs = ['~' for i in M]
        print('Chemical:',vs[chemical],sep=' ^ ')
        print('Biomass:',vs[biomas],sep=' ^ ')
        del_strat = 'all'

    print('*'*4,' FINISHED!!! ','*'*4)

    return  del_strat, vs, s

def my_algorithm(infeas=1e-6,**Params):
    '''
    my_algorithm(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global max_value
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm ******')

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    max_value = []
    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.Params.NodefileStart = 0
        # imodel.Params.Threads = 4
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):

        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO]
            ki = list(combinations(knockset_inner,2))
            # print('****Knockset Len****',len(ki))
            if len(knockset) !=2:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')
            max_value.append(round(model._vij[biomas],5))
            #  print(max_value)
            print(f'Knockset: {knockset}')
            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')

            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        # print(f'**** Lazy constrain {i}: {rxn[comb[0]]}-{rxn[comb[1]]}*****; inner values: {round(model._vij[comb[0]],3)},{round(model._vij[comb[1]],3)}; outer values: {round(model._voj[comb[0]],3)},{round(model._voj[comb[1]],3)}')
                        model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] +
                                (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                                model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[knockset[0]]),'+',str(model._varsy[knockset[1]]),' >=1' )
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.Params.NodefileStart = 0
    # m.Params.Threads = 4
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i] < .5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']



    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat, vouter, s


def my_algorithm_n(infeas=1e-6,**Params): #Uses the max value from the inner model
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global max_value
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm ******')

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    max_value = []

    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.Params.NodefileStart = 0
        # imodel.Params.Threads = 4
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):

        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO]
            ki = list(combinations(knockset_inner,2))
            # print('****Knockset Len****',len(ki))
            if len(knockset) !=2:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')
            max_value.append(round(model._vij[biomas],5))
            print(max_value)
            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')

            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        # print(f'**** Lazy constrain {i}: {comb}*****')
                        model.cbLazy(round(max(max_value),5) <= model._vars[biomas] +
                                (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                                model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[knockset[0]]),'+',str(model._varsy[knockset[1]]),' >=1' )
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.Params.NodefileStart = 0
    # m.Params.Threads = 4
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i] < .5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']



    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat, vouter, s

def my_algorithm_d(infeas=1e-6,**Params): #uses the connectivity degree from the Stoichiometric matrix
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global con_d
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print("**** Solving New Algorithm with new rules for the y's NO connectivity degree ******")

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]

    def knocklist(kvi,kvo):
            k1 = [(kvi[i],kvi[i+1]) for i in range(0,len(kvi),2)] #generates a list [(y1,y2)]
            k2 = [(i,j) for i in kvi for j in kvo if i!=j] #generates a list of pair combinations [(kvi,kvo)]
            for i in k2:
                if i not in k1:
                    k1.append(i)
            return k1
    def connectivity_degree(S):
            S_count = list(np.count_nonzero(S,axis=0))
            con_d = {}
            for i, v in enumerate(S_count):
                con_d[i] = v
            return con_d

    con_d = connectivity_degree(S)

    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.Params.NodefileStart = 0
        # imodel.Params.Threads = 4
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):

        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            ky = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            kv = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO and abs(model._voj[i]) > 1e-6]
            ki = knocklist(ky,kv)
            if len(ky) !=k:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')

            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            #print(f'Index for the combinations {kv}')
            print(f'Lazy constraints to add: {len(ki)}')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')

            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        # if (con_d[comb[0]] + con_d[comb[1]]) >= 3:
                        #print(f'**** Lazy constrain {i+1}: {comb}*****')
                        model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] +
                                (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                                model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[knockset[0]]),'+',str(model._varsy[knockset[1]]),' >=1' )
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.Params.NodefileStart = 0
    # m.Params.Threads = 4
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i] < .5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']



    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat, vouter, s


def my_algorithm_d2(infeas=1e-6,**Params): #uses the connectivity degree from the Stoichiometric matrix
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global max_value,con_d
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm with connectivity degree ******')

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    max_value = []

    def connectivity_degree(S):
            S_count = list(np.count_nonzero(S,axis=0))
            con_d = {}
            for i, v in enumerate(S_count):
                con_d[i] = v
            return con_d

    con_d = connectivity_degree(S)

    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.Params.NodefileStart = 0
        # imodel.Params.Threads = 4
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):

        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO]
            ki = list(combinations(knockset_inner,2))
            # print('****Knockset Len****',len(ki))
            if len(knockset) !=2:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')
            max_value.append(round(model._vij[biomas],5))
            print(max_value)
            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')

            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        if (con_d[comb[0]] + con_d[comb[1]]) >= 3:
                            #print(f'**** Lazy constrain {i}: {comb}*****')
                            model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] +
                                (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                                model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[knockset[0]]),'+',str(model._varsy[knockset[1]]),' >=1' )
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.Params.NodefileStart = 0
    # m.Params.Threads = 4
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i] < .5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']



    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat, vouter, s

def my_algorithm_d3(infeas=1e-6,**Params): #uses the connectivity degree from the Stoichiometric matrix
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global max_value,con_d,max_value_chemical
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm ******')

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    max_value_innerbiomas = []
    max_value_chemical = []
    def connectivity_degree(S):
            S_count = list(np.count_nonzero(S,axis=0))
            con_d = {}
            for i, v in enumerate(S_count):
                con_d[i] = v
            return con_d

    con_d = connectivity_degree(S)

    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.Params.NodefileStart = 0
        # imodel.Params.Threads = 4
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):

        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO]
            ki = list(combinations(knockset_inner,2))
            # print('****Knockset Len****',len(ki))
            if len(knockset) !=2:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')
            max_value_innerbiomas.append(round(model._vij[biomas],5))
            max_value_chemical.append(round(model._voj[chemical],5))
            print(max_value)
            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')

            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        if (con_d[comb[0]] + con_d[comb[1]]) >= 3:
                            #print(f'**** Lazy constrain {i}: {comb}*****')
                            model.cbLazy(max(max_value_innerbiomas) <= model._vars[biomas] +
                                (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                                model._varsy[comb[1]]))
                            model.cbLazy(max(max_value_chemical) <= model._vars[chemical] +
                                (ma.ceil(model._vij[biomas]*10)/10)* (model._varsy[comb[0]] + model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[knockset[0]]),'+',str(model._varsy[knockset[1]]),' >=1' )
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.Params.NodefileStart = 0
    # m.Params.Threads = 4
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i] < .5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']



    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat, vouter, s


def my_algorithm_d4(infeas=1e-6,**Params): #uses the connectivity degree from the Stoichiometric matrix
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global max_value,con_d,max_value_chemical,max_value_innerbiomas
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm ******')

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    max_value_innerbiomas = []
    max_value_chemical = []
    def connectivity_degree(S):
            S_count = list(np.count_nonzero(S,axis=0))
            con_d = {}
            for i, v in enumerate(S_count):
                con_d[i] = v
            return con_d

    con_d = connectivity_degree(S)

    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.Params.NodefileStart = 0
        # imodel.Params.Threads = 4
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):

        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO]
            ki = list(combinations(knockset_inner,2))
            # print('****Knockset Len****',len(ki))
            if len(knockset) !=2:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')
            max_value_innerbiomas.append(round(model._vij[biomas],5))
            max_value_chemical.append(round(model._voj[chemical],5))
            print(max_value)
            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')

            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        # if (con_d[comb[0]] + con_d[comb[1]]) >= 3:
                            #print(f'**** Lazy constrain {i}: {comb}*****')
                        model.cbLazy(max(max_value_innerbiomas) <= model._vars[biomas] +
                            (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                            model._varsy[comb[1]]))
                        model.cbLazy(max(max_value_chemical) <= model._vars[chemical] +
                            (ma.ceil(model._vij[biomas]*10)/10)* (model._varsy[comb[0]] + model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[knockset[0]]),'+',str(model._varsy[knockset[1]]),' >=1' )
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.Params.NodefileStart = 0
    # m.Params.Threads = 4
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i] < .5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']



    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat, vouter, s



def my_algorithm2(infeas=1e-6,**Params):
    '''
    my_algorithm2(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm 2 Different Cuts ******')
    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]

    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):
        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6]
            ki = list(combinations(knockset_inner,2))  #[(i,j),(i+n,j+n)]
            # if len(knockset) !=2:
            #     # print('Error knocking out')
            #     return
                #print('***','Begin Lazy Constraints','***')
            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:

                if model._vij[biomas] != 2000:
                    for i,comb in enumerate(ki):
                        model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] + (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] + model._varsy[comb[1]]))

                        model.cbLazy(model._vars[chemical] <= model._vij[chemical] + UB[chemical]*(model._varsy[comb[0]] + model._varsy[comb[1]]))
                else:
                    model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    m.Params.NodefileStart = 0.5
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i]<.5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']


    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat,vouter,s

def my_algorithm3(infeas=1e-6,**Params):
    '''
    my_algorithm3(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print('**** Solving New Algorithm 3 Different Cuts ******')
    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    def inner(imodel,yoj):
        global vinner
        # print('Printing lenght of arguments',len(imodel.getVars()), len(M),sep=' -> ')
        imodel.setAttr('LB',imodel.getVars(), [LB[j]*yoj[j] for j in M])
        imodel.setAttr('UB',imodel.getVars(), [UB[j]*yoj[j] for j in M])

        # print('Optimizing Inner Problem...')
        imodel.Params.OptimalityTol = infeas
        imodel.Params.IntFeasTol = infeas
        imodel.Params.FeasibilityTol = infeas
        # imodel.params.LogToConsole = False
        imodel.optimize()

        if imodel.status == GRB.OPTIMAL:
            vinner = [imodel.getVarByName('vi[%d]'%j).x for j in M]
        elif imodel.status in (GRB.INFEASIBLE, GRB.UNBOUNDED,GRB.INF_OR_UNBD):
            vinner = [2000 if i == biomas else yoj[i] for i in M]
        return vinner

    def lazyctr(model, where):
        if where == GRB.Callback.MIPSOL:
            # print('** Begin Lazyctr callback (MIPSOL) ***')
            model._voj = model.cbGetSolution(model._vars)
            model._yoj = model.cbGetSolution(model._varsy)
            model._m   = model._voj[chemical]

            keys = model._vars.keys()
            model._vij = inner(model._inner,model._yoj)

            knockset = [i for i,y in enumerate(model._yoj) if model._yoj[i] < 1e-6]
            knockset_inner = [i for i,y in enumerate(model._vij) if abs(model._vij[i]) < 1e-6 and i in KO]
            # print(len(knockset_inner))
            ki = list(combinations(knockset_inner,2))  #[(i,j),(i+n,j+n)]
            # if len(knockset) !=2:
            #     # print('Error knocking out')
            #     return
                #print('***','Begin Lazy Constraints','***')
            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:

                for i, comb in enumerate(ki):
                    if model._vij[biomas] != 2000:
                        model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] + (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] + model._varsy[comb[1]]))

                        model.cbLazy(model._vars[chemical] <= model._vij[chemical] + UB[chemical]*(model._varsy[comb[0]] + model._varsy[comb[1]]))
                    else:
                        model.cbLazy(model._varsy[comb[0]] + model.varsy[comb[1]] >= 1)


                # if model._vij[biomas] != 2000:
                #     for i,comb in enumerate(ki):
                #         model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] + (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] + model._varsy[comb[1]]))
                #
                #         model.cbLazy(model._vars[chemical] <= model._vij[chemical] + UB[chemical]*(model._varsy[comb[0]] + model._varsy[comb[1]]))
                # else:
                #     model.cbLazy(model._varsy[knockset[0]] + model._varsy[knockset[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))
            # print('*** Deletion Strategy ***')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                model._vij = inner(model._inner,model._ryoj)

                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
            # print('*** Set Solution Passed ***', model.cbUseSolution())

    m = gp.Model()
    v = m.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='v')
    y = m.addVars(M,vtype=GRB.BINARY,name='y')

    m.setObjective(1*v[chemical],GRB.MAXIMIZE)

    m.addConstrs((gp.quicksum(S[i,j]*v[j] for j in M) == 0 for i in N),name='Stoichiometry')

    m.addConstrs((y[j] == 1 for j in M if j not in KO))

    m.addConstr(sum(1-y[j] for j in KO) == k, name='knapsack')

    m.addConstr(v[biomas] >= minprod, name='target')

    m.addConstrs((LB[j]*y[j] <= v[j] for j in M),name='LB')
    m.addConstrs((v[j] <= UB[j]*y[j] for j in M),name='UB')

    m._vars = v
    m._varsy = y
    m.Params.lazyConstraints = 1

    imodel = gp.Model()
    vi = imodel.addVars(M,lb=-GRB.INFINITY,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='vi')
    imodel.params.LogToConsole = 0
    imodel.setObjective(2000*vi[biomas] + vi[chemical], GRB.MAXIMIZE)

    imodel.addConstrs((gp.quicksum(S[i,j]*vi[j] for j in M) == 0 for i in N),name='S2')
    imodel.addConstr(vi[biomas] >= minprod, name='target2')

    imodel.update()
    m._inner = imodel.copy()
    m._innerv = vi

    m.Params.OptimalityTol = infeas
    m.Params.IntFeasTol = infeas
    m.Params.FeasibilityTol = infeas
    # m.params.LogToConsole = False
    m.Params.NodefileStart = 0.5
    m.optimize(lazyctr)
    # m.setParam(GRB.Param.PoolSolutions, 10)
    # m.setParam(GRB.Param.PoolSearchMode, 2)
    # m.setParam(GRB.Param.PoolGap, 0.01)
    s = m.Runtime
    # nsolutions = m.SolCount

    if m.status == GRB.OPTIMAL:
        ys = [m.getVarByName('y[%d]'%j).x for j in M]
        vouter = [m.getVarByName('v[%d]'%j).x for j in M]
        del_strat = [rxn[i] for i in M if ys[i]<.5]
    elif m.status in (GRB.INFEASIBLE,GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ys = ['all' for i in M]
        vouter = ['~' for i in M]
        del_strat = ['all']


    print('*** Best Solution ***')
    print('Biomass outer v:',vouter[biomas],sep=' -> ')
    print('Biomass inner v:',vinner[biomas],sep=' -> ')

    print('Chemical Overproduction:',vouter[chemical],sep=' -> ')
    print('Deletion Strategy:',[rxn[i] for i in M if ys[i]<.5],sep=' -> ')

    # print('******  ******')
    # print('Alternate Solutions:')
    # for e in range(1,3):
    #     m.setParam(GRB.Param.SolutionNumber,e)
    #     print('Solution #:',e,sep=' -> ')
    #     print('Chemical Overproduction:','%g'%m.PoolObjVal,sep=' - >')
    #     print('Biomas Production:', v[biomas].Xn)
    #     print('Alternate Deletion Strategy:')
    #     if e <= 3:
    #         m.setParam(GRB.Param.SolutionNumber,e)
    #         print('Knockout Strategy:',[rxn[i] for i in M if y[i].Xn < .5],sep=' -> ')
    # print('*****  ********')
    print('Time in seconds: %d'%s,'Time in minutes: %d'%(s/60),sep=' -> ')

    return del_strat,vouter,s
