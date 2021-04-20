"""
3 population admixture model for tracts

First, Nat and Eur populations admix in first generation, followed by Afr
admixture.  If time  is not integer, the migration is divided between
neighboring times proportional to the non-integer time fraction. We assume
Afr ancestry still replaces migrants from Nat and Eur after the mixture of
Eur and Nat if they arrive at the same generation (strict time inequality
of migration).

This is adapted from the PUR example in the tracts repo.
"""

import numpy as np


def MXL_3pop(*params):
    """
    Nat (population 0) and Eur (population 1) admix first, followed by
    introduction of Afr (population 2).

    Parameters are (prop0, T0, prop2, T2), where prop0 is the initial proportion
    of Nat (1-prop0) is initial proportion of Eur. prop2 is the proportion of
    Afr (replacing 0 and 1). T0 is the initial admixture time of 0 and 1, and T2
    is the admixture time of 2.

    We enforce that ceil(T2) <= floor(T1).

    Bounds on all parameters of [0, 1].

    The two times are measured in units of 100 generations, because some
    python optimizers work better when all parameters have the same scale.
    """
    (prop0, T0, prop2, T2) = params[0]

    T0 = 100 * T0
    T2 = 100 * T2

    # some sanity checks
    if T2 > T0 or T2 < 0 or T0 < 0:
        # This will be caught by "outofbounds" function. Return empty matrix
        gen = int(np.ceil(max(T0, 0))) + 1
        mig = numpy.zeros((gen + 1, 3))
        return mig

    # How many generations we'll need to accomodate all the migrations.
    gen = int(np.ceil(T0)) + 1

    # How far off the continuous time is from its discrete optimization
    timefrac = gen - T0 - 1

    # Build an empty matrix with enough generations to handle all migrations
    # Row i is i generations back in time (0 is current gen, 1 is one gen back, etc)
    mig = np.zeros((gen + 1, 3))

    # replace a fraction at first and second generation to ensure a continuous
    # model
    prop1 = 1 - prop0
    mig[-1, :] = np.array([prop0, prop1, 0])

    interEu = prop0 * timefrac
    interNat = prop1 * timefrac
    mig[-2, :] = np.array([interEu, interNat, 0])

    # Which integer generation to add the migrants from pop 2
    gen2 = int(np.ceil(T2)) + 1
    timefrac2 = gen2 - T2 - 1

    # we want the total proportion replaced  by 2 to be prop2. We therefore add
    # a fraction f at generation gen2-1, and (prop2-f)/(1-f) at generation gen.
    mig[gen2 - 1, 2] = timefrac2 * prop2
    mig[gen2, 2] = (prop2 - timefrac2 * prop2) / (1 - np.sum(mig[gen2 - 1]))

    if np.sum(mig[gen2]) > 1:
        mtot = np.sum(mig[gen2])
        mig[gen2][0] -= prop0 * (mtot - 1)
        mig[gen2][1] -= prop1 * (mtot - 1)

    return mig


def outofbounds_3pop(*params):
    """
    Constraint function evaluating below zero when constraints not satisfied
    
    Check that migration proportions are between 0 and 1, etc
    """
    ret = 1
    (prop0, T0, prop2, T2) = params[0]

    # all proportions are between 0 and 1
    ret = min(1 - prop0, 1 - prop2)
    ret = min(ret, prop0, prop2)

    # Pedestrian way of testing for all possible issues
    func = MXL_3pop
    mig = func(params[0])
    totmig = mig.sum(axis=1)

    if prop0 > 1 or prop2 > 1:
        print("Pulse greater than 1")
    if prop0 < 0 or prop2 < 0:
        print("Pulse less than 0")

    ret = min(ret, -abs(totmig[-1] - 1) + 1e-8)

    # recent generations (0 and 1) have no migrants
    ret = min(ret, -totmig[0], -totmig[1])

    # all migration sums between 0 and 1
    ret = min(ret, min(1 - totmig), min(totmig))

    # time constraints
    ret = min(ret, T0 - T2)
    ret = min(ret, T0)
    ret = min(ret, T2)
    return ret


# We don't have to calculate all the tract length distributions to have the
# global ancestry proportion right. Here we define a function that
# automatically adjusts the migration rates to have the proper global ancestry
# proportion, saving a lot of optimization time!

# We first define a function that calculates the final proportions of ancestry
# based on the migration matrix


def propfrommig(mig):
    curr = mig[-1, :]
    for row in mig[-2::-1, :]:
        curr = curr * (1 - np.sum(row)) + row
    return curr


# Then, assuming the global ancestry fractions are known, we define a function
# to get the full migration matrix knowing those global fractions and the given
# T0 and T2.


def MXL_3pop_fix(Ts, fracs):
    """
    fracs = [frac_NAT, frac_EUR, frac_AFR]
    """
    T0, T2 = Ts

    def fun(props):
        prop0, prop2 = props
        return propfrommig(MXL_3pop((prop0, T0, prop2, T2)))[0:2] - fracs[0:2]

    (prop0, prop2) = scipy.optimize.fsolve(fun, (0.2, 0.2))
    return MXL_3pop((prop0, T0, prop2, T2))


# And now for the out of bounds function for the fixed global ancestry case


def outofbounds_3pop_fix(params, fracs):
    ret = 1

    (T0, T2) = params
    if T0 > 1:
        return 1 - T0

    def fun(props):
        prop0, prop2 = props
        return propfrommig(MXL_3pop((prop0, T0, prop2, T2)))[0:2] - fracs[0:2]

    (prop0, prop2) = scipy.optimize.fsolve(fun, (0.2, 0.2))
    return outofbounds_3pop((prop0, T0, prop2, T2))
