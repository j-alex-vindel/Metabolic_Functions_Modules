''' Created on 02/02/2022

* This module runs the bilevel algorithm and keeps track of all the improvements in the code:

- mya_o: my algorithm original model
- mya_d: improved way to introduced the number of lazy cuts based on the ys conditions
- mya_d1: prints the MIPNODE Status
Author: @j-alex-vindel

'''

import gurobipy as gp
import math as ma
from numpy import inf
from gurobipy import GRB
from itertools import combinations
import numpy as np
import random

def mya_o(infeas=1e-6,**Params):
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
    print(f'# of Variables (reactions in the network): {len(M)}')
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


def mya_d(infeas=1e-6,**Params): #
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k)
        return delstrat, fluxes, time
    '''
    global con_d
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k = Params['biomas'],Params['chemical'],Params['minprod'],Params['k']

    print("**** Solving  ******")

    print('Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    print(f'# Variables (reactions in the network): {len(M)}')
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
                print('Error knocking out')
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
                    print(f'Here the lazy is:', str(model._varsy[ky[0]]),'+',str(model._varsy[ky[1]]),' >=1' )
                    model.cbLazy(model._varsy[ky[0]] + model._varsy[ky[1]] >= 1)

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

def mya_d1(infeas=1e-6,**Params): #uses the connectivity degree from the Stoichiometric matrix
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k,model=name)
        return delstrat, fluxes, time
    '''
    global con_d
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k, mname = Params['biomas'],Params['chemical'],Params['minprod'],Params['k'],Params['model']

    print('-> Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    print(f'-> # Variables (reactions in the network): {len(M)}')
    print(f'-> Metabolic Network: {mname}')
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
            ## try to get the samples from the ki list to control the number of lazy cuts to add
            print('** MIPSOL **')
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
                    print(f'Here the lazy is:', str(model._varsy[ky[0]]),'+',str(model._varsy[ky[1]]),' >=1' )
                    model.cbLazy(model._varsy[ky[0]] + model._varsy[ky[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')
            else:
                print(f'Same Biomas values; Inner value: {round(model._vij[biomas],4)}, Outer Value {round(model._voj[biomas],4)}')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            print(f'** MIPNODE **')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))

            print(f'* If Condition {sum(model._ryoj.values())} == {len(model._ryoj) - 2}')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                print(f'Condition satisfied!')
                print('Deletion Strategy:',[rxn[i] for i in M if model._ryoj[i]<.5],[i for i in M if model._ryoj[i]<.5],sep=' -> ')
                model._vij = inner(model._inner,model._ryoj) #call to solve the inner problem again using the rounded y values
                print(f' Values after recalculating the v values -> Biomas : {round(model._vij[biomas],4)}; Chemical: {round(model._vij[chemical],4)}')
                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
                objval = model.cbUseSolution()

                print(f'*** Set Solution Passed *** ')

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

def mya_d2(infeas=1e-6,**Params): #uses the connectivity degree from the Stoichiometric matrix
    '''
    my_algorithm_n(infeas=1e-6(optional),S=S,LB=LB,UB=UB,minprod=minprod,rxn=rxn,met=met,biomas=biomas,chemical=chemical,KO=KO,k=k,model=name)
        return delstrat, fluxes, time
    '''
    if len(Params.keys()) < 10 :
        return f'Insuficient Parameters. infeas(Optional), S, LB, UB, rxn, met, KO, biomas, chemical, k, minprod)'

    S,LB,UB,rxn,met,KO = Params['S'],Params['LB'],Params['UB'], Params['rxn'], Params['met'], Params['KO']
    biomas, chemical, minprod, k, mname = Params['biomas'],Params['chemical'],Params['minprod'],Params['k'],Params['model']

    print('-> Current Infeasibility:',infeas,sep=' -> ')
    M = [i for i in range(len(rxn))]
    N = [i for i in range(len(met))]
    print(f'-> # Variables (reactions in the network): {len(M)}')
    print(f'-> Metabolic Network: {mname}')

    def knocklist(kvi,kvo):
            k1 = [(kvi[i],kvi[i+1]) for i in range(0,len(kvi),2)] #generates a list [(y1,y2)]
            k2 = [(i,j) for i in kvi for j in kvo if i!=j] #generates a list of pair combinations [(kvi,kvo)]
            for i in k2:
                if i not in k1:
                    k1.append(i)
            return k1
    def ki_r(k1):
        if len(k1) == 1 or len(k1)<10:
            return k1
        elif len(k1) >10:
            g = random.sample(k1,5)
            h = [k1[0]]
            for pair in g:
                if pair not in h:
                    h.append(pair)
            return h
    # def connectivity_degree(S):
    #         S_count = list(np.count_nonzero(S,axis=0))
    #         con_d = {}
    #         for i, v in enumerate(S_count):
    #             con_d[i] = v
    #         return con_d
    #
    # con_d = connectivity_degree(S)

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
            kir = ki_r(ki)
            if len(ky) !=k:
                # print('Error knocking out')
                return
                #print('***','Begin Lazy Constraints','***')
            ## try to get the samples from the ki list to control the number of lazy cuts to add
            print('** MIPSOL **')
            print('Deletion Strategy:',[rxn[i] for i in M if model._yoj[i]<.5],[i for i in M if model._yoj[i]<.5],sep=' -> ')
            #print(f'Index for the combinations {kv}')
            print(f'Lazy constraints to add: {len(ki)}')
            print('Inner Biomas Value: ',model._vij[biomas],sep=' -> ')
            print('Outer Biomas Value: ',model._voj[biomas],sep=' -> ')
            print('Inner Chem Value: ',model._vij[chemical],sep=' -> ')
            print('Outer Chem Value: ',model._voj[chemical],sep=' -> ')
            print(f'Best Objective Bound found so far: {model.cbGet(GRB.Callback.MIPSOL_OBJBND)}')
            print(f'Current explored node count: {model.cbGet(GRB.Callback.MIPSOL_NODCNT)}')
 
            if abs(model._vij[biomas] - model._voj[biomas]) >= 1e-6:
                if model._vij[biomas] != 2000:
                    for index,comb in enumerate(kir):
                        # if (con_d[comb[0]] + con_d[comb[1]]) >= 3:
                        print(f'**** Lazy constrain {index+1}: {comb}*****')
                        model.cbLazy(round(model._vij[biomas],7) <= model._vars[biomas] +
                                (ma.ceil(model._vij[biomas]*10)/10) * (model._varsy[comb[0]] +
                                model._varsy[comb[1]]))


                else:
                    print(f'Here the lazy is:', str(model._varsy[ky[0]]),'+',str(model._varsy[ky[1]]),' >=1' )
                    model.cbLazy(model._varsy[ky[0]] + model._varsy[ky[1]] >= 1)

                # print('*** ENd Lazy Constraints ***')
            else:
                print(f'Same Biomas values; Inner value: {round(model._vij[biomas],4)}, Outer Value {round(model._voj[biomas],4)}')

        elif where == GRB.Callback.MIPNODE:
            #print('*** Begin Lazy CTR Callback (MIPNODE) ***')
            print(f'** MIPNODE **')
            model._ryoj = model.cbGetNodeRel(model._varsy)
            for i,y in enumerate(model._ryoj):
                if model._ryoj[y] >= 0.8:
                    model._ryoj[y] = 1.0
                elif model._ryoj[y] <= 0.2:
                     model._ryoj[y] = 0.0
                else:
                    model._ryoj[y] = 1.0
            #print('Rounded Solution', sum(model._ryoj.values()))

            print(f'* If Condition {sum(model._ryoj.values())} == {len(model._ryoj) - 2}')
            if sum(model._ryoj.values()) == len(model._ryoj)-2:
                print(f'Condition satisfied!')
                print('Deletion Strategy:',[rxn[i] for i in M if model._ryoj[i]<.5],[i for i in M if model._ryoj[i]<.5],sep=' -> ')
                model._vij = inner(model._inner,model._ryoj) #call to solve the inner problem again using the rounded y values
                print(f' Values after recalculating the v values -> Biomas : {round(model._vij[biomas],4)}; Chemical: {round(model._vij[chemical],4)}')
                model.cbSetSolution(model._vars, model._vij)
                model.cbSetSolution(model._varsy, model._ryoj)
                objval = model.cbUseSolution()

                print(f'*** Set Solution Passed *** ')

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
