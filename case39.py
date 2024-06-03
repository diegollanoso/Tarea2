# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 39 bus New England system.
"""
    
from numpy import insert, int64, array, matrix, diag, zeros, ones, arange, ix_, r_, flatnonzero as find, real, imag, random, concatenate
from scipy.sparse import csr_matrix as sparse, identity as sparseI
from numpy.linalg import inv

def case39():
    """Power flow data for 39 bus New England system.
    Please see L{caseformat} for details on the case file format.

    Data taken from [1] with the following modifications/additions:

        - renumbered gen buses consecutively (as in [2] and [4])
        - added C{Pmin = 0} for all gens
        - added C{Qmin}, C{Qmax} for gens at 31 & 39 (copied from gen at 35)
        - added C{Vg} based on C{V} in bus data (missing for bus 39)
        - added C{Vg, Pg, Pd, Qd} at bus 39 from [2] (same in [4])
        - added C{Pmax} at bus 39: C{Pmax = Pg + 100}
        - added line flow limits and area data from [4]
        - added voltage limits, C{Vmax = 1.06, Vmin = 0.94}
        - added identical quadratic generator costs
        - increased C{Pmax} for gen at bus 34 from 308 to 508
          (assumed typo in [1], makes initial solved case feasible)
        - re-solved power flow

    Notes:
        - Bus 39, its generator and 2 connecting lines were added
          (by authors of [1]) to represent the interconnection with
          the rest of the eastern interconnect, and did not include
          C{Vg, Pg, Qg, Pd, Qd, Pmin, Pmax, Qmin} or C{Qmax}.
        - As the swing bus, bus 31 did not include and Q limits.
        - The voltages, etc in [1] appear to be quite close to the
          power flow solution of the case before adding bus 39 with
          it's generator and connecting branches, though the solution
          is not exact.
        - Explicit voltage setpoints for gen buses are not given, so
          they are taken from the bus data, however this results in two
          binding Q limits at buses 34 & 37, so the corresponding
          voltages have probably deviated from their original setpoints.
        - The generator locations and types are as follows:
            - 1   30      hydro
            - 2   31      nuke01
            - 3   32      nuke02
            - 4   33      fossil02
            - 5   34      fossil01
            - 6   35      nuke03
            - 7   36      fossil04
            - 8   37      nuke04
            - 9   38      nuke05
            - 10  39      interconnection to rest of US/Canada

    This is a solved power flow case, but it includes the following
    violations:
        - C{Pmax} violated at bus 31: C{Pg = 677.87, Pmax = 646}
        - C{Qmin} violated at bus 37: C{Qg = -1.37,  Qmin = 0}

    References:

    [1] G. W. Bills, et.al., I{"On-Line Stability Analysis Study"}
    RP90-1 Report for the Edison Electric Institute, October 12, 1970,
    pp. 1-20 - 1-35.
    prepared by
      - E. M. Gulachenski - New England Electric System
      - J. M. Undrill     - General Electric Co.
    "...generally representative of the New England 345 KV system, but is
    not an exact or complete model of any past, present or projected
    configuration of the actual New England 345 KV system."

    [2] M. A. Pai, I{Energy Function Analysis for Power System Stability},
    Kluwer Academic Publishers, Boston, 1989.
    (references [3] as source of data)

    [3] Athay, T.; Podmore, R.; Virmani, S., I{"A Practical Method for the
    Direct Analysis of Transient Stability,"} IEEE Transactions on Power
    Apparatus and Systems , vol.PAS-98, no.2, pp.573-584, March 1979.
    U{http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4113518&isnumber=4113486}
    (references [1] as source of data)

    [4] Data included with TC Calculator at
    U{http://www.pserc.cornell.edu/tcc/} for 39-bus system.

    @return: Power flow data for 39 bus New England system.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 1, 97.6, 44.2, 0, 0, 2, 1.0393836, -13.536602, 345, 1, 1.06, 0.94],
        [2, 1, 0, 0, 0, 0, 2, 1.0484941, -9.7852666, 345, 1, 1.06, 0.94],
        [3, 1, 322, 2.4, 0, 0, 2, 1.0307077, -12.276384, 345, 1, 1.06, 0.94],
        [4, 1, 500, 184, 0, 0, 1, 1.00446, -12.626734, 345, 1, 1.06, 0.94],
        [5, 1, 0, 0, 0, 0, 1, 1.0060063, -11.192339, 345, 1, 1.06, 0.94],
        [6, 1, 0, 0, 0, 0, 1, 1.0082256, -10.40833, 345, 1, 1.06, 0.94],
        [7, 1, 233.8, 84, 0, 0, 1, 0.99839728, -12.755626, 345, 1, 1.06, 0.94],
        [8, 1, 522, 176.6, 0, 0, 1, 0.99787232, -13.335844, 345, 1, 1.06, 0.94],
        [9, 1, 6.5, -66.6, 0, 0, 1, 1.038332, -14.178442, 345, 1, 1.06, 0.94],
        [10, 1, 0, 0, 0, 0, 1, 1.0178431, -8.170875, 345, 1, 1.06, 0.94],
        [11, 1, 0, 0, 0, 0, 1, 1.0133858, -8.9369663, 345, 1, 1.06, 0.94],
        [12, 1, 8.53, 88, 0, 0, 1, 1.000815, -8.9988236, 345, 1, 1.06, 0.94],
        [13, 1, 0, 0, 0, 0, 1, 1.014923, -8.9299272, 345, 1, 1.06, 0.94],
        [14, 1, 0, 0, 0, 0, 1, 1.012319, -10.715295, 345, 1, 1.06, 0.94],
        [15, 1, 320, 153, 0, 0, 3, 1.0161854, -11.345399, 345, 1, 1.06, 0.94],
        [16, 1, 329, 32.3, 0, 0, 3, 1.0325203, -10.033348, 345, 1, 1.06, 0.94],
        [17, 1, 0, 0, 0, 0, 2, 1.0342365, -11.116436, 345, 1, 1.06, 0.94],
        [18, 1, 158, 30, 0, 0, 2, 1.0315726, -11.986168, 345, 1, 1.06, 0.94],
        [19, 1, 0, 0, 0, 0, 3, 1.0501068, -5.4100729, 345, 1, 1.06, 0.94],
        [20, 1, 680, 103, 0, 0, 3, 0.99101054, -6.8211783, 345, 1, 1.06, 0.94],
        [21, 1, 274, 115, 0, 0, 3, 1.0323192, -7.6287461, 345, 1, 1.06, 0.94],
        [22, 1, 0, 0, 0, 0, 3, 1.0501427, -3.1831199, 345, 1, 1.06, 0.94],
        [23, 1, 247.5, 84.6, 0, 0, 3, 1.0451451, -3.3812763, 345, 1, 1.06, 0.94],
        [24, 1, 308.6, -92.2, 0, 0, 3, 1.038001, -9.9137585, 345, 1, 1.06, 0.94],
        [25, 1, 224, 47.2, 0, 0, 2, 1.0576827, -8.3692354, 345, 1, 1.06, 0.94],
        [26, 1, 139, 17, 0, 0, 2, 1.0525613, -9.4387696, 345, 1, 1.06, 0.94],
        [27, 1, 281, 75.5, 0, 0, 2, 1.0383449, -11.362152, 345, 1, 1.06, 0.94],
        [28, 1, 206, 27.6, 0, 0, 3, 1.0503737, -5.9283592, 345, 1, 1.06, 0.94],
        [29, 1, 283.5, 26.9, 0, 0, 3, 1.0501149, -3.1698741, 345, 1, 1.06, 0.94],
        [30, 2, 0, 0, 0, 0, 2, 1.0499, -7.3704746, 345, 1, 1.06, 0.94],
        [31, 3, 9.2, 4.6, 0, 0, 1, 0.982, 0, 345, 1, 1.06, 0.94],
        [32, 2, 0, 0, 0, 0, 1, 0.9841, -0.1884374, 345, 1, 1.06, 0.94],
        [33, 2, 0, 0, 0, 0, 3, 0.9972, -0.19317445, 345, 1, 1.06, 0.94],
        [34, 2, 0, 0, 0, 0, 3, 1.0123, -1.631119, 345, 1, 1.06, 0.94],
        [35, 2, 0, 0, 0, 0, 3, 1.0494, 1.7765069, 345, 1, 1.06, 0.94],
        [36, 2, 0, 0, 0, 0, 3, 1.0636, 4.4684374, 345, 1, 1.06, 0.94],
        [37, 2, 0, 0, 0, 0, 2, 1.0275, -1.5828988, 345, 1, 1.06, 0.94],
        [38, 2, 0, 0, 0, 0, 3, 1.0265, 3.8928177, 345, 1, 1.06, 0.94],
        [39, 2, 1104, 250, 0, 0, 1, 1.03, -14.535256, 345, 1, 1.06, 0.94]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [30, 250, 161.762, 400, 140, 1.0499, 100, 1, 1040, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [31, 677.871, 221.574, 300, -100, 0.982, 100, 1, 646, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [32, 650, 206.965, 300, 150, 0.9841, 100, 1, 725, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [33, 632, 108.293, 250, 0, 0.9972, 100, 1, 652, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [34, 508, 166.688, 167, 0, 1.0123, 100, 1, 508, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [35, 650, 210.661, 300, -100, 1.0494, 100, 1, 687, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [36, 560, 100.165, 240, 0, 1.0636, 100, 1, 580, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [37, 540, -1.36945, 250, 0, 1.0275, 100, 1, 564, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [38, 830, 21.7327, 300, -150, 1.0265, 100, 1, 865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [39, 1000, 78.4674, 300, -100, 1.03, 100, 1, 1100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([#                                                   exist nw SC TS 
       # 0  1        2    3        4    5    6    7  8  9 10    11   12       13 14  15 16 17 
        [1, 2, 0.0035, 0.0411, 0.6987, 600, 600, 600, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [1, 39, 0.001, 0.025, 0.75, 1000, 1000, 1000, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [2, 3, 0.0013, 0.0151, 0.2572, 500, 500, 500, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [2, 25, 0.007, 0.0086, 0.146, 500, 500, 500, 0, 0, 1,       -360, 360, 1, 1,  0, 0, 0],
        [2, 30, 0, 0.0181, 0, 900, 900, 2500, 1.025, 0, 1,          -360, 360, 1, 1,  0, 0, 0],
        [3, 4, 0.0013, 0.0213, 0.2214, 500, 500, 500, 0, 0, 1,      -360, 360, 1, 1,  0, 1, 1],
        [3, 18, 0.0011, 0.0133, 0.2138, 500, 500, 500, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 1],
        [4, 5, 0.0008, 0.0128, 0.1342, 600, 600, 600, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [4, 14, 0.0008, 0.0129, 0.1382, 500, 500, 500, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 1],
        [5, 6, 0.0002, 0.0026, 0.0434, 1200, 1200, 1200, 0, 0, 1,   -360, 360, 1, 1,  0, 0, 0],
        [5, 8, 0.0008, 0.0112, 0.1476, 900, 900, 900, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [6, 7, 0.0006, 0.0092, 0.113, 900, 900, 900, 0, 0, 1,       -360, 360, 1, 1,  0, 0, 0],
        [6, 11, 0.0007, 0.0082, 0.1389, 480, 480, 480, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [6, 31, 0, 0.025, 0, 1800, 1800, 1800, 1.07, 0, 1,          -360, 360, 1, 1,  0, 0, 0],
        [7, 8, 0.0004, 0.0046, 0.078, 900, 900, 900, 0, 0, 1,       -360, 360, 1, 1,  0, 0, 0],
        [8, 9, 0.0023, 0.0363, 0.3804, 900, 900, 900, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [9, 39, 0.001, 0.025, 1.2, 900, 900, 900, 0, 0, 1,          -360, 360, 1, 1,  0, 0, 0],
        [10, 11, 0.0004, 0.0043, 0.0729, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [10, 13, 0.0004, 0.0043, 0.0729, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [10, 32, 0, 0.02, 0, 900, 900, 2500, 1.07, 0, 1,            -360, 360, 1, 1,  0, 0, 0],
        [12, 11, 0.0016, 0.0435, 0, 500, 500, 500, 1.006, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [12, 13, 0.0016, 0.0435, 0, 500, 500, 500, 1.006, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [13, 14, 0.0009, 0.0101, 0.1723, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [14, 15, 0.0018, 0.0217, 0.366, 600, 600, 600, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 1], # candidata TS 1
        [15, 16, 0.0009, 0.0094, 0.171, 600, 600, 600, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 1],
        [16, 17, 0.0007, 0.0089, 0.1342, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 1],
        [16, 19, 0.0016, 0.0195, 0.304, 600, 600, 2500, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [16, 21, 0.0008, 0.0135, 0.2548, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [16, 24, 0.0003, 0.0059, 0.068, 600, 600, 600, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [17, 18, 0.0007, 0.0082, 0.1319, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [17, 27, 0.0013, 0.0173, 0.3216, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [19, 20, 0.0007, 0.0138, 0, 900, 900, 2500, 1.06, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [19, 33, 0.0007, 0.0142, 0, 900, 900, 2500, 1.07, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [20, 34, 0.0009, 0.018, 0, 900, 900, 2500, 1.009, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [21, 22, 0.0008, 0.014, 0.2565, 900, 900, 900, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [22, 23, 0.0006, 0.0096, 0.1846, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [22, 35, 0, 0.0143, 0, 900, 900, 2500, 1.025, 0, 1,         -360, 360, 1, 1,  0, 0, 0],
        [23, 24, 0.0022, 0.035, 0.361, 600, 600, 600, 0, 0, 1,      -360, 360, 1, 1,  0, 0, 0],
        [23, 36, 0.0005, 0.0272, 0, 900, 900, 2500, 1, 0, 1,        -360, 360, 1, 1,  0, 0, 0],
        [25, 26, 0.0032, 0.0323, 0.531, 600, 600, 600, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [25, 37, 0.0006, 0.0232, 0, 900, 900, 2500, 1.025, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [26, 27, 0.0014, 0.0147, 0.2396, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [26, 28, 0.0043, 0.0474, 0.7802, 600, 600, 600, 0, 0, 1,    -360, 360, 1, 1,  0, 0, 0],
        [26, 29, 0.0057, 0.0625, 1.029, 600, 600, 600, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [28, 29, 0.0014, 0.0151, 0.249, 600, 600, 600, 0, 0, 1,     -360, 360, 1, 1,  0, 0, 0],
        [29, 38, 0.0008, 0.0156, 0, 1200, 1200, 2500, 1.025, 0, 1,  -360, 360, 1, 1,  0, 0, 0]
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2]
    ])
    
    
    
    # load of the customers
    deterministic = 1
    if deterministic:
        ppc["demand"] = array([700, 750, 850, 950, 1000, 1100, 1150, 1200, 1300, 1400, 1450, 1500, 1400, 1300, 1200, 1050, 1000, 1100, 1200, 1400, 1300, 1100, 900, 800])
        # ppc["demand"] = array([586, 633, 724, 824, 867, 954, 1006, 1067, 1108, 1193, 1246, 1278, 1224, 1225, 1139, 973, 950, 1069, 1067, 1371, 1127, 1036, 902, 784]);
    else:
        ppc["demand"] = array([
            [586,591,586,591,586,591,591,591,591,586,586,586,586,586,586],
            [633,639,628,633,639,639,628,633,639,633,633,628,628,633,628],
            [724,731,718,718,718,711,718,724,718,731,731,724,718,731,731],
            [824,795,817,795,817,802,809,795,809,795,809,802,809,824,802],
            [867,837,837,845,845,867,875,852,860,845,875,837,860,845,860],
            [954,937,945,970,937,970,921,929,945,945,954,937,921,937,954],
            [1006,1006,980,997,963,997,1014,1006,980,971,971,1006,1023,971,980],
            [1067,1040,1040,1013,1013,1067,1049,1031,1076,1076,1067,1031,1058,1022,1004],
            [1108,1117,1117,1176,1156,1108,1156,1166,1098,1088,1108,1127,1137,1098,1137],
            [1193,1235,1203,1235,1172,1214,1203,1235,1277,1277,1193,1256,1203,1182,1245],
            [1246,1290,1279,1279,1290,1290,1257,1279,1235,1312,1225,1268,1301,1257,1225],
            [1278,1391,1323,1289,1357,1312,1289,1301,1323,1278,1323,1346,1379,1391,1323],
            [1224,1308,1235,1266,1266,1182,1224,1308,1287,1298,1266,1214,1182,1214,1245],
            [1225,1117,1147,1166,1176,1176,1108,1127,1166,1108,1156,1176,1098,1166,1108],
            [1139,1058,1049,1013,1139,1130,1040,1112,1013,1013,1094,1139,1067,1103,1094],
            [973,887,942,910,958,918,934,887,997,902,879,887,918,973,879],
            [950,860,882,845,942,845,935,957,882,942,965,890,957,845,882],
            [1069,921,1011,1036,1036,978,1044,1044,1044,1069,1011,1069,1036,921,995],
            [1067,1022,1040,1094,1031,1022,1094,1085,1112,1139,1148,1049,1031,1094,1094],
            [1371,1361,1224,1371,1371,1224,1224,1214,1382,1350,1182,1382,1224,1319,1319],
            [1127,1156,1264,1225,1186,1244,1088,1283,1156,1254,1127,1156,1166,1117,1254],
            [1036,1044,1003,1061,945,1077,1020,987,954,1077,1102,987,1028,995,1077],
            [902,868,902,875,753,909,882,767,895,841,767,794,767,834,821],
            [784,760,700,760,736,814,802,796,772,814,778,682,730,808,724]]);        
    # Power generation data
             # c b a p_max p_tin  Min_up  Min_do  H_cost C_cost C_time I_state Ramp-up and -down limit 
    ppc["units"] = array([
            [1000, 16.19, 0.00048, 455, 150, 8, 8, 4500,  9000, 5,  8, 225],   # Gen1
            [ 970, 17.26, 0.00031, 455, 150, 8, 8, 5000, 10000, 5,  8, 225],   # Gen2
            [ 700,  16.6, 0.002  , 130,  20, 5, 5,  550,  1100, 4, -5, 50],   # Gen3
            [ 680,  16.5, 0.00211, 130,  20, 5, 5,  560,  1120, 4, -5, 50],   # Gen4
            [ 450,  19.7, 0.00398, 162,  25, 6, 6,  900,  1800, 4, -6, 60],   # Gen5
            [ 370, 22.26, 0.00712,  80,  20, 3, 3,  170,   340, 2, -3, 60],   # Gen6
            [ 480, 27.74, 0.00079,  85,  25, 3, 3,  260,   520, 2, -3, 60],   # Gen7
            [ 660, 25.92, 0.00413,  55,  10, 1, 1,   30,    60, 0, -1, 135],   # Gen8
            [ 665, 27.27, 0.00222,  55,  10, 1, 1,   30,    60, 0, -1, 135],   # Gen9
            [ 670, 27.79, 0.00173,  55,  10, 1, 1,   30,    60, 0, -1, 135]]) # Gen10       

    # SEP 
    ng = len(ppc['gen'])
    nb = len(ppc['bus'])
    nl = len(ppc['branch'])
    T = len(ppc['demand'])

    # Generation
    ppc['p02006'] = array([455, 245, 0, 0, 0, 0, 0, 0, 0, 0]) # Initial power unit generation 2006        
    # ppc["p02006'] = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # Initial power unit generation 2006        
    ppc['p02013'] = array([455-150, 245-150, 0, 0, 0, 0, 0, 0, 0, 0]) # Initial power unit generation 2013
    pos_g = (ppc['gen'][:,0]-1).astype(int) # gen location (astype... for integer positions in the SF matrix)
    ppc['pos_g'] = pos_g
    Cg = sparse((ones(ng), (pos_g, range(ng))), (nb, ng)) #conection gen matrix
    ppc['Cg'] = Cg

    """ Extrae las condiciones iniciales de los generadores. """
    V0 = ppc['p02006'].astype('bool').astype(int)
    ppc['V0'] = V0
    
    """ Cantidad de horas que la unidad j ha permanecido en servicio al comienzo del horizonte de planificacion t=0. """
    U0 = zeros(ng,int)
    aux = find(ppc['units'][:,10]>0)
    U0[aux] = ppc['units'][aux,10]
    ppc["U0"] = U0

    """ Cantidad de horas que la unidad j ha permanecido fuera de servicio al comienzo del horizonte de planificacion t=0. """
    S0 = zeros(ng,int);
    aux = find(ppc['units'][:,10]<0)
    S0[aux] = abs(ppc['units'][aux,10])
    ppc["S0"] = S0
    
    """ Cantidad de horas que la unidad j debe permanecer en servicio desde el comienzo del horizonte de planificación t>=1. """
    UT = ppc["units"][:,5] # minimum up time
    G = zeros(ng,int);
    for g in range(ng):
        G[g] = min(T,int((UT[g] - U0[g]) * V0[g]))    
    ppc["G"] = G
    
    """ Cantidad de horas que la unidad j debe permanecer fuera de servicio desde el horizonte de planificación t>=1. """
    DT = ppc["units"][:,6] # minimum down time
    L = zeros(ng,int)
    for g in range(ng):
        L[g] = min(T,int((DT[g] - S0[g])*(1-V0[g])))
    ppc["L"] = L
    
    
    # Load
    aux = sum(ppc["bus"][:,2])
    for b in range(nb):
        if ppc["bus"][b,2] < 0:
            ppc["bus"][b,2] = 0
        if ppc["bus"][b,2] > 0:  
            ppc["bus"][b,2] = ppc["bus"][b,2] / aux

    # load uncertainty            
    ppc['load_esc'] = array([1])
    # ppc['load_esc'] = array([1.0, 0.98, 0.95, 0.93, 0.91])
    # ppc['load_esc'] = array([1, 1, 1, 1, 1])
    # ppc['load_esc'] = array([1.05, 1.02, 1.01, 1.0, 0.98, 0.97, 0.95, 0.93, 0.91, 0.90])
    
    # Slack bus
    slack_bus=find(ppc['bus'][:,1]==3)
    ppc['SL'] = slack_bus
    
    # Transmission modeling
    b = 1 / ppc['branch'][:,3]
    f = ppc['branch'][:, 0]-1
    t = ppc['branch'][:, 1]-1
    if max(f) > nb: #sort SEPs
        aux = ppc['bus'][:,0]
        for i in range(ng):
            pos = find(aux == ppc['units'][i,0])
            ppc['units'][i,0] = pos + 1
        for i in range(nl): 
            pos = find(aux == ppc['branch'][i,0]) #from
            ppc['branch'][i,0] = pos + 1
            pos = find(aux == ppc['branch'][i,1]) #to
            ppc['branch'][i,1] = pos + 1
        ppc['bus'][:,0] = range(1, nb+1)
        b = 1 / ppc['branch'][:,3]
        f = ppc['branch'][:, 0]-1
        t = ppc['branch'][:, 1]-1   

    I = r_[range(nl), range(nl)]
    S = sparse((r_[ones(nl), -ones(nl)], (I, r_[f, t])), (nl, nb))
    ppc['S'] = array(S.todense())
    
    Bf = sparse((r_[b, -b], (I, r_[f, t])), (nl,nb))
    Bbus = S.T * Bf
    ppc['Bbus'] = array(Bbus.todense())    
    buses = arange(1, nb)
    noslack = find(arange(nb) != slack_bus)
    SF = zeros((nl, nb))
    SF[:,noslack] = Bf[:, noslack].todense()*inv(Bbus[ix_(noslack, noslack)].todense())
    PTDF = SF * S.T
    I_PTDF = array(sparseI(nl) - PTDF)
    # LODF = PTDF * sum(diag(1/(ones(nl)-diag(PTDF)))) # LODF = SF * S.T * sum(diag(1/(ones(nl)-diag(SF * S.T))))
    # LODF = PTDF * diag(diag(1/(ones(nl)-diag(PTDF))))
    ppc['SF'] = SF
    ppc['PTDF'] = PTDF
    ppc['I_PTDF'] = I_PTDF        
    # ppc['LODF'] = LODF            
    
    # Transmission N-1 modeling    
    pos_lo = find(ppc['branch'][:,13]==0) 
    SF_post = []; PTDF_post = []; I_PTDF_post = [];
    for l in range(len(pos_lo)):
        SF_out = SF + array(matrix(LODF[:,pos_lo[l]]).T * matrix(SF[pos_lo[l]]))
        SF_post.insert(l, SF_out)
        PTDF_out = SF_out * S.T
        PTDF_post.insert(l, PTDF_out)
        I_PTDF_out = array(sparseI(nl) - PTDF_out)
        I_PTDF_post.insert(l, I_PTDF_out)           
    ppc['SF_post'] = SF_post
    ppc['PTDF_post'] = PTDF_post
    ppc['I_PTDF_post'] = I_PTDF_post
    
    # Transmission losses
    R = ppc['branch'][:,2]
    X = ppc['branch'][:,3]
    yprim = zeros(nl).astype(complex)    
    for l in range(nl):
        yprim[l] = 1/(complex(R[l],X[l]))        
    ppc['G'] = real(yprim)
    ppc['B'] = imag(yprim)
    
    # Modeling Z = R + j X
    BfR = sparse((r_[imag(yprim), -imag(yprim)], (I, r_[f, t])), (nl,nb))
    ppc['BfR'] = array(BfR.todense()) 
    BbusR = S.T * BfR
    ppc['BbusR'] = array(BbusR.todense())    
    SFR = zeros((nl, nb))
    SFR[:,noslack] = BfR[:, noslack].todense()*inv(BbusR[ix_(noslack, noslack)].todense())    
    ppc['SFR'] = SFR

    
    return ppc