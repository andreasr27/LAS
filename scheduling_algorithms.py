# this library/python file contains
# (1) all the three basic online algorithms for the problem of energy minimization via speed scaling
# (2) an implementation of the offline optimal algorithm
# (3) the implementation of LAS algorithm
from fractions import Fraction
from typing import Dict, Any, Tuple, Union

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


def LAS(_J_prediction, _J_true, _epsilon, dt, alpha):
    # input: (1)receives the prediction and real instances as dictionaries:
    #                                                       key --> job id
    #                                                       value--> (job weight, release time, deadline) as a tuple
    #        (2) the robustness parameter _epsilon
    #        (3) dt --> the time granularity of the output list
    #        (4) alpha --> the convexity parameter of the problem
    # output: a speed list which is the actual speed that our processor runs
    #         element i of the output speed represents time = i*dt
    
    
    
    # find epsilon such that ((1+epsilon)/(1 - epsilon))^alpha = 1 + _epsilon
    epsilon = scale_down_epsilon(_epsilon, alpha , 0.01)
    
    w, r, d = _J_prediction[1]
    T = d-r
    T_prime = Fraction(1-epsilon, 1)*T
    J_true = copy.deepcopy(_J_true)
    
    #preprocess the prediction so as to have T_prime = (1-epsilon)*T
    J_prediction = {}
    ids = sorted(_J_prediction.keys())
    for id in ids:
        w_id, start, end = _J_prediction[id]
        end_prime = start + T_prime
        J_prediction[id] = (w_id, start, end_prime)    
      
    #Optimal_Alg returns a speed list and a solution instance 
    _, J_prediction_opt = Optimal_Alg(J_prediction)

    speed_not_robust = {}
    speed_to_smear_out = {}
    ids = sorted(J_prediction_opt.keys())
    for id in ids:
        w_pred, speed_pred, start_processing, end_processing = J_prediction_opt[id]
        w_true, start, end = J_true[id]
        _, start_prime, end_prime = J_prediction[id]
        # just to check if everything is ok
        if start_prime != start:
            print("Houston we have a problem")
             
        processing_interval = (start_processing, end_processing)
        feasible_interval = (start_prime, end_prime)
        if w_true <= w_pred:
            speed_to_finish_between_start_and_end = Fraction(w_true, end_processing-start_processing)
            speed_not_robust[processing_interval] = speed_to_finish_between_start_and_end
        else:
            speed_not_robust[processing_interval] = speed_pred
            w_exceeding = w_true - w_pred
            speed_to_smear_out[feasible_interval] = Fraction(w_exceeding, end_prime-start_prime)
        
    
    # at this point we will use speed_to_smear out in order to 
    # augment the speed in the speed_not_robust dictionary and
    # produce a feasible schedule
    
    intervals = speed_to_smear_out.keys()
    for interval in intervals:
        speed_to_add = speed_to_smear_out[interval]
        add_speed(speed_not_robust, speed_to_add, interval)
    

    
    #now it remains to robustify the speed through convolution
    
    # we turn our dictionary to a list with granularity dt
    # mul_factor is a parameter iwhich is set more than 1 in the noise robust version of LAS
    speed_to_make_robust = speed_dictionary_to_list(speed_not_robust, dt, mul_factor = 1)
    
    # we use the robustify function to perform the convolution
    speed_robust = robustify(speed_to_make_robust, epsilon, T, dt)
    
    return speed_robust






def Alg_with_Predictions(_J_prediction, _J_true, epsilon):
    return 0
#     # we first need to change the instance by doing the epsilon bucketing
#     # we need to do the epsilon bucketing only when 1/epsilon is smaller than T
#     # we extend our algorithm to values of epsilon > 1/3. In this case we do not do the bucketing at all

#     w, r, d = _J_prediction[1]
#     T = d - r
#     if epsilon * T > 1 and epsilon < Fraction(1, 3):
#         J_prediction = rounding_instance(_J_prediction, epsilon)
#         J_true = rounding_instance(_J_true, epsilon)
#     else:
#         J_prediction = _J_prediction
#         J_true = _J_true

#     # we will do the capping with the epsilon**2

#     #Optimal_Alg returns a speed list and the a solution instance
#     _, J_prediction_opt = Optimal_Alg(J_prediction)

#     #we will only need from now on the speed list solution in a dictionary form

#     speed_guideline = compute_speed(J_prediction_opt)

#     # we will proceed by doing the smoothing of the speed_guideline
#     # which was initialized to be the optimal solution of the J_prediction

#     ids = sorted(J_prediction_opt.keys())
#     for id in ids:
#         w_id, speed_id, start_processing, end_processing = J_prediction_opt[id]
#         _, r, d = J_prediction[id]
#         cappped_speed = Fraction(w_id, (d - r)*epsilon**2)

#         if speed_id > cappped_speed:
#             # cap the speed
#             interval = (start_processing, end_processing)
#             add_speed(speed_guideline, cappped_speed - speed_id, interval)

#             # compute how much work we should smear out
#             w_cap = (speed_id - cappped_speed)*(end_processing - start_processing)
#             smear_out_speed = Fraction(w_cap, d-r)
#             interval = (r,d)
#             add_speed(speed_guideline, smear_out_speed, interval)


#     # from now on, we simulate how the online algorithm with predictions adjust to misspredictions:
#     for id in ids:
#         w_true, _r_true, _d_true = J_true[id]
#         w_online, speed_online, _start_processing, _end_processing = J_prediction_opt[id]

#         #be sure that times are of Fraction type
#         r_true = Fraction(_r_true, 1)
#         d_true = Fraction(_d_true, 1)
#         start_processing = Fraction(_start_processing,1)
#         end_processing = Fraction(_end_processing, 1)

#         # we precompute the two intervals in which we may change the speed
#         whole_interval = (r_true, d_true)
#         processing_interval = (start_processing, end_processing)

#         # we precompute the lenght T(same of every job) and T_proc which is the lenght of the interval
#         # in which the job is processed in the optimal speed schedule for the prediction
#         T = d_true - r_true
#         T_proc = end_processing-start_processing
#         too_high = (T_proc < T*epsilon**2)

#         # if we overpredicted then we scale down the speed by a factor w_true/w_online
#         # since the scale down of the speed is referred to the parallel execution of the
#         # job setting, we should be careful how we do this scaling (here we update the cumulative speed)




#         if w_true < w_online and too_high:
#             alpha_tilde = 1 - Fraction(T_proc, T * epsilon**2)
#             alpha = Fraction(1, epsilon**2) + alpha_tilde

#             #first we add speed to the whole interval
#             speed_to_add = Fraction(w_true - w_online, T)*alpha_tilde
#             add_speed(speed_guideline, speed_to_add, whole_interval)

#             #we subsequently add speed to the processing interval
#             speed_to_add = Fraction(w_true - w_online, T) * Fraction(1,epsilon**2)
#             add_speed(speed_guideline, speed_to_add, processing_interval)

#         elif w_true < w_online and (not too_high):

#             #we add speed only in the processing interval since the capping did not happen
#             speed_to_add = Fraction(w_true - w_online, T_proc)
#             add_speed(speed_guideline, speed_to_add, processing_interval)
#         else:
#             # this is the case we underestimate, or we capped the job...
#             # no matter the reason we need to augment the speed in order to produce a feasible solution
#             speed_to_add = Fraction(w_true - w_online, T)
#             add_speed(speed_guideline, speed_to_add, whole_interval)
#         # now the speed list stores the speed dictionary
#         # which is finally ran by the online+prediction algorithm
#     return speed_guideline
