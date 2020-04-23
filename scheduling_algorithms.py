# this library/python file contains
# (1) all the three basic online algorithms for the problem of energy minimization via speed scaling
# (2) an implementation of the offline optimal algorithm
# (3) the implementation of LAS algorithm

from scheduling_functions import *
import numpy as np
import sys
import copy
from random import sample, randint, seed
from math import isclose, ceil, floor
from decimal import *
from fractions import *


def BKP_alg(J, dt, alpha):
    # input: receives an instance J which is represented as a dictionary:
    #                                                           key--> job id
    #                                                           value--> (job weight, release time, deadline) as a tuple
    #       dt the time step to approximate the speed integral
    # Needs to import euler constant from math
    # output: energy consumption of BKP

    keys = sorted(J.keys())
    min_time = J[keys[0]][1]
    max_time = J[keys[-1]][2]
    energy = 0
    t = min_time
    while t < max_time:
        density = densest_interval_BKP(J, t, dt)
        energy += dt * ((e * density) ** alpha)
        t += dt
    return energy


def OptimalOnline(_J):
    # input: receives an instance J which is represented as a dictionary:
    #                                                           key--> job id
    #                                                           value--> (job weight, release time, deadline) as a tuple
    # output: speed_list where speeds[t] denotes the speed of Optimal Online for the interval [t,t+1]

    all_idxs = sorted(_J.keys())
    _, start, _ = _J[all_idxs[0]]
    _, _, end = _J[all_idxs[-1]]

    # we will change the release times/deadlines to simulate that they arrive one by one
    J_sim = {}
    for idx in all_idxs:
        w, r, d = _J[idx]
        T = d - r
        J_sim[idx] = (w, 0, T)

    speeds_OO = []
    J = {}
    for t in range(start, end):
        if t < all_idxs[-1]:
            wt, rt, dt = J_sim[t + 1]
            J[t] = (wt, rt, dt)

        # here we will handle the case where there is no job left and no jobs will arrive in the future
        if (t >= all_idxs[-1]) and (not J):
            return speeds_OO

        # (1) we will compute the optimal instance of this
        idxs = sorted(J.keys())
        J_sol = compute_optimal_instance(copy.deepcopy(J))
        # solution
        speed_list = compute_speed_per_integer_time(J_sol)

        # here we handle the case where the first job has zero weight
        if len(speed_list)== 0:
            speed  = 0
        else:
            speed = speed_list[0]

        # since we refer to one time unit
        work = speed * 1
        # now we find how much wwork should we delete from the instance
        speeds_OO.append(speed)

        # here we find which jobs are run during the interval (t,t+1)
        tota_weight = 0
        jobs_involved = []
        for idx in idxs:
            weight_to_add, _, _ = J[idx]
            tota_weight += weight_to_add
            if tota_weight >= work:
                jobs_involved.append(idx)
                break
            jobs_involved.append(idx)
        # here we diminish the remaining work of the jobs involved
        for job in jobs_involved:
            job_weight, release_time, deadline = J[job]

            if job_weight > work:
                del J[job]
                J[job] = (job_weight - work, release_time, deadline)
                if job_weight == work:
                    del J[job]
                work = 0
            elif job_weight == work:
                del J[job]
                work = 0
            else:
                del J[job]
                work -= job_weight

        # here I fix my instance in order to start from 0 and release times and deadlines should be diminished by one
        job_keys = sorted(J.keys())
        for job_key in job_keys:
            w, r, d = J[job_key]
            if r == 0:
                del J[job_key]
                J[job_key] = (w, r, d - 1)
            else:
                print("we have an error")



        # I delete my previous solution
        del J_sol
    return speeds_OO


def Avg_rate(J):
    # input: receives an instance J which is represented as a dictionary:
    #                                                           key--> job id
    #                                                           value--> (job weight, release time, deadline) as a tuple
    # output: the speed_list dictionary of this algorithm
    ids = sorted(J.keys())
    _, _start, _ = J[ids[0]]
    _, _, _end = J[ids[-1]]
    start = _start
    end = _end
    speed_list_dictionary = {}
    speed_list_dictionary[(start, end)] = Fraction(0, 1)
    for id in ids:
        w_id, r, d = J[id]
        speed = Fraction(w_id, d - r)
        r_id = Fraction(r, 1)
        d_id = Fraction(d, 1)
        interval = (r_id, d_id)
        add_speed(speed_list_dictionary, speed, interval)
    return speed_list_dictionary


def Optimal_Alg(J):
    # input: receives an instance J which is represented as a dictionary:
    #                                                           key--> job id
    #                                                           value--> (job weight, release time, deadline) as a tuple
    # output: the speed_list dictionary of this algorithm, thus the optimal speed function
    #                           key --> interval as a tuple (r,d)
    #                           value--> speed in the aforementioned interval
    #         J_sol the solution instance as a dictionary with:
    #                                           key--> the id of the job
    #                                           value--> (weight, speed, start processing, end processing)

    J_sol = compute_optimal_instance(copy.deepcopy(J))
    speed_list_dictionary = compute_speed(J_sol)
    return speed_list_dictionary, J_sol


def Alg_with_Predictions(_J_prediction, _J_true, epsilon):
    # we first need to change the instance by doing the epsilon bucketing
    # we need to do the epsilon bucketing only when 1/epsilon is smaller than T
    # we extend our algorithm to values of epsilon > 1/3. In this case we do not do the bucketing at all

    w, r, d = _J_prediction[1]
    T = d - r
    if epsilon * T > 1 and epsilon < Fraction(1, 3):
        J_prediction = rounding_instance(_J_prediction, epsilon)
        J_true = rounding_instance(_J_true, epsilon)
    else:
        J_prediction = _J_prediction
        J_true = _J_true
    # we will do the capping with the epsilon**2
    speed_list_optimal, J_prediction_opt = Optimal_Alg(J_prediction)

    # now we will do the capping of the speed and we will change the weight of the jon appropriately
    # that means:
    #           if s[id] > w_id / ((d-r)*epsilon**2) where r= arrival time and d = deadline
    #               then: s[id] --> w_id/((d-r)*epsilon**2)
    #                     w_id --> s[id]*(end_processing - start_processing)
    ids = sorted(J_prediction_opt.keys())
    for id in ids:
        w_id, speed_id, start_processing, end_processing = J_prediction_opt[id]
        _, r, d = J_prediction[id]
        average_speed = Fraction(w_id, d - r)
        if speed_id > Fraction(average_speed, epsilon ** 2):
            # cap the speed
            speed_id = Fraction(average_speed, epsilon ** 2)
            # update the weight for which we are solving the instance
            w_id = speed_id * (end_processing - start_processing)
            del J_prediction_opt[id]
            J_prediction_opt[id] = (w_id, speed_id, start_processing, end_processing)
    # now we compute the capped speed list
    speed_list_dictionary = compute_speed(J_prediction_opt)

    # from now on, we simulate how the online algorithm with predictions adjust to misspredictions:
    for id in ids:
        w_true, _r_true, _d_true = J_true[id]
        w_online, speed_online, start_processing, end_processing = J_prediction_opt[id]
        r_true = Fraction(_r_true, 1)
        d_true = Fraction(_d_true, 1)
        # if we overpredicted then we scale down the speed in the processing interval of the optimal solution of the predicted instance
        if w_true <= w_online:
            work_to_remove = w_online - w_true
            try:
                speed_to_remove = Fraction(work_to_remove, end_processing - start_processing)
            except ZeroDivisionError as err:
                print("I have azero division", err)
                print("we are talking about job =", id)
                print("start processing", start_processing)
                print("end processing", end_processing)
                print("speed to remove", speed_to_remove)
                print("the true weight of this job is:", w_true)
            interval = (start_processing, end_processing)
            add_speed(speed_list_dictionary, -speed_to_remove, interval)
        else:
            # this is the case we underestimate, or we capped the job...
            # no matter the reason we need to augment the speed in order to produce a feasible solution
            w_to_add = w_true - w_online

            speed_to_add = Fraction(w_to_add, d_true - r_true)
            interval = (r_true, d_true)
            add_speed(speed_list_dictionary, speed_to_add, interval)
        # now the speed list stores the speed dictionary
        # which is finally ran by the online+prediction algorithm
    return speed_list_dictionary
