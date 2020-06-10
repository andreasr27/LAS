import numpy as np
import sys
import copy
from random import sample, randint, seed
from math import isclose, ceil, floor, e
from decimal import *
from fractions import *


def find_arrivals(J):
    # input: takes a dictionary which represents the instance
    #                               key --> job id
    #                               value --> [release time , deadline]
    # output: output a list of all the release times
    arrivals = []
    for key in J.keys():
        _, a, _ = J[key]
        arrivals.append(a)
    arrivals = list(set(arrivals))
    return arrivals


def find_deadlines(J):
    # input: takes a dictionary which represents the instance
    #                               key --> job id
    #                               value --> [release time , deadline]
    # output: output a list of all the deadlines
    deadlines = []
    for key in J.keys():
        _, _, d = J[key]
        deadlines.append(d)
    deadlines = list(set(deadlines))
    return deadlines

def find_weight(J, r, d):
    # input: an instance J which is represented as a dictionary
    #        an interval [r,d]
    # output: returns (1) the total weight of the jobs (completely)contained in this interval
    #                 (2) the ids of the jobs completely contained in this interval
    total_weight = 0
    #the ids of the jobs which are contained in the interval [r,d]
    ids =  []
    for key in J.keys():
        w, rk, dk = J[key]
        if rk >= r and dk <= d:
            total_weight += w
            ids.append(key)
    return total_weight, ids

def adjust_instance(J,r,d):
    #input: (1) an instance J represented by a dictionary
    #       (2) an interval [r,d]
    # output: change the instance J erasing the interval [r,d] (and modifying all he jobs that interfere with this interval)

    # as a first step we will eliminate from the dictionary all the jobs which are included in the interval [r,d]
    for key in list(J.keys()):
        _,rk,dk = J[key]
        if rk>=r and dk<=d:
            del J[key]
    # now we need to delete the interval [r,d] of the instance
    for key in J.keys():
        wk,rk,dk = J[key]
        # case 1: interval which completely containes the critical interval
        if rk<=r and dk>=d:
            dk = dk - (d-r)
        # case 2: the interval ends into the ctitical interval
        elif rk<=r and dk>=r and dk<=d:
            dk = r
        # case 3: the interval starts into the critical interval
        elif rk>=r and rk<=d and dk>=d:
            rk = r
            dk = dk -(d-r)
        # case 4: the interval starts and ends outdside the critical interval but on the right ---> thus, we have to move starts and end to the left
        elif  rk>=d and dk>=d:
            rk = rk - (d-r)
            dk = dk - (d-r)

        # now we will update the interval's release time and deadline
        J[key] = (wk, rk, dk)


def compute_energy_integer_speed_list(speed_list, alpha):
    # input: alpha--> exponent of the energy function
    #       a list of floats which represent the speed of the processor at any point in time
    #       speed_list[t] = speed of the processor at time t
    # output: outputs the energy used by the processor
    energy = sum([(speed**alpha) for speed in speed_list])
    return energy

def compute_energy(speed_list, alpha):
    # input: (1) alpha---> exponent of the energy function
    #        (2) a speed function represented by a dictionary:
    #                                               key--> interval [r,d] represented in teh form of a tuple (r,d)
    #                                               value--> the speed in the interval [r,d]
    # output: returns the energy consumed
    energy = sum([  (interval[1]-interval[0])*(speed_list[interval])**alpha  for interval in speed_list.keys() ])
    return energy


def get_speed(speed_dictionary, t):
    speed = 0
    intervals = speed_dictionary.keys()
    for interval in intervals:
        start, end = interval
        if (t>=start) and (t<=end):
            speed = speed_dictionary[interval]
            break
    return speed


def compute_speed(J_sol):
    # input: solution instance --> dictionary key --> id
    #                                        value--> (weight, speed, start processing, end processing)
    # output: a dictionary with key --> the interval in which we are reffering to as a tuple (start, end])
    #                           value--> the speed at this time interval
    speed_lst = {}
    for id in J_sol.keys():
        _, speed, start, end = J_sol[id]
        speed_lst[(start, end)] = speed
    return speed_lst

def intersecting_intervals(speed_list, interval):
    # input: speed_list is a dictionary with key --> interval as a tuple (start,end)
    #                                       value--> speed at this particular interval
    #       interval: a particular interval that we will subsequently update
    # output: we will modify the speed_list dictionary in order to erase any partial coverings of the input interval
    interval_start = interval[0]
    interval_end  = interval[1]
    #check sanity of the input
    if interval_start >= interval_end:
        print("there is a problem in the input of the function update_intervals")
        exit(-1)

    intervals_to_update_list = []
    for interval_to_update in speed_list.keys():
        start = interval_to_update[0]
        end   = interval_to_update[1]
        intersects = not((start >= interval_end) or (end <= interval_start))
        if intersects:
            intervals_to_update_list.append(interval_to_update)
    #now the intervals_to_update list is full with the info of the intervals we need to update
    return intervals_to_update_list

def modify_speedlist(speed_list, intersectings, interval):

    start_interval = interval[0]
    end_interval   = interval[1]

    for intersecting in intersectings:
        speed = speed_list[intersecting]
        start_intersecting = intersecting[0]
        end_intersecting = intersecting[1]
        #case 1: the interval which we update completely contains the intersecting interval
        if start_interval <= start_intersecting and end_interval >= end_intersecting:
            #in this case we conitnue without doing somehting
            continue
        #case 2: the interval which we update is completely contained in the intersecting interval
        elif start_interval >= start_intersecting and end_interval <= end_intersecting:
            interval1=(start_intersecting, start_interval)
            interval2=(start_interval, end_interval)
            interval3=(end_interval, end_intersecting)
            del speed_list[intersecting]
            if start_intersecting < start_interval:
                speed_list[interval1] = speed
            if start_interval < end_interval:
                speed_list[interval2] = speed
            if end_interval < end_intersecting:
                speed_list[interval3] = speed
        #case 3 the interval left intersects with the intersecting interval
        elif start_interval < start_intersecting and end_interval > start_intersecting and end_interval < end_intersecting:
            interval1 = (start_intersecting, end_interval)
            interval2 = (end_interval, end_intersecting)
            del speed_list[intersecting]
            if start_intersecting < end_interval:
                speed_list[interval1] = speed
            if end_interval < end_intersecting:
                speed_list[interval2] = speed
        #case 4 the interval right intersects with the intersecting interval
        elif start_intersecting < start_interval and start_interval < end_intersecting and end_intersecting < end_interval:
            interval1 = (start_intersecting, start_interval)
            interval2 = (start_interval, end_intersecting)
            del speed_list[intersecting]
            if start_intersecting < start_interval:
                speed_list[interval1] = speed
            if start_interval < end_intersecting:
                speed_list[interval2] = speed

def add_speed(speed_list, speed, interval):
    # input: (1) speed_list is a dictionary with key --> interval as a tuple (start,end)
    #                                       value--> speed at this particular interval
    #        (2) speed: the amount of speed that we want to add
    #        (3) interval: the interval in which we want to increase the speed
    # output: update the speed list dictionary to add the speed in the appropriate interval

    start_of_update = interval[0]
    end_of_update   = interval[1]
    # we first get all the intersecting intervals
    intersecting_intervals_list = intersecting_intervals(speed_list, interval)

    # now we have to modify the speed list dictionary in order to add the new interval
    modify_speedlist(speed_list, intersecting_intervals_list, interval)

    # after modify speed_list it must be the case that all intervals in the speed_list are either contained in the interval that we want to update or are completely disjoint
    speed_list_keys = sorted(speed_list.keys(), key=lambda x: x[0])
    for interval_to_update in speed_list_keys:
        start = interval_to_update[0]
        end  = interval_to_update[1]
        if start_of_update <= start and end <= end_of_update:
            #here we need to do the update
            speed_to_increase = speed_list[interval_to_update]
            del speed_list[interval_to_update]
            new_speed = speed_to_increase + speed
            speed_list[interval_to_update] = new_speed
        elif end_of_update <= start or start_of_update >= end:
            #that means I do not need to do something
            continue
        else:
            print("I have a problem")
            print("start of update = ", start_of_update, "---", "end of update = ", end_of_update)
            print("start = ", start, "---", "end = ", end)
            exit(-1)



def scale_speed(speed_list, mul_factor, interval):
    # input: (1) speed_list is a dictionary with key --> interval as a tuple (start,end)
    #                                       value--> speed at this particular interval
    #        (2) mul_factor is a number that we want to multilply the speed with 
    #        (3) interval: the interval in which we want to increase the speed
    # output: update the speed list dictionary to add the speed in the appropriate interval

    start_of_update = interval[0]
    end_of_update   = interval[1]
    # we first get all the intersecting intervals
    intersecting_intervals_list = intersecting_intervals(speed_list, interval)

    # now we have to modify the speed list dictionary in order to add the new interval
    modify_speedlist(speed_list, intersecting_intervals_list, interval)

    # after modify speed_list it must be the case that all intervals in the speed_list are either contained in the interval that we want to update or are completely disjoint
    speed_list_keys = sorted(speed_list.keys(), key=lambda x: x[0])
    for interval_to_update in speed_list_keys:
        start = interval_to_update[0]
        end  = interval_to_update[1]
        if start_of_update <= start and end <= end_of_update:
            #here we need to do the update
            speed_to_increase = speed_list[interval_to_update]
            del speed_list[interval_to_update]
            new_speed = speed_to_increase * mul_factor
            speed_list[interval_to_update] = new_speed
        elif end_of_update <= start or start_of_update >= end:
            #that means I do not need to do something
            continue
        else:
            print("I have a problem")
            print("start of update = ", start_of_update, "---", "end of update = ", end_of_update)
            print("start = ", start, "---", "end = ", end)
            exit(-1)



def rounding_instance(J, epsilon):
    #we assume that d-r = T for every job
    w, r, d = J[1]
    T = d-r
    #now we need to do the rounding
    ids = sorted(J.keys())
    intervals = []
    for id in ids:
        w, r, d = J[id]
        r_new = epsilon * T * ceil(float(r)/(epsilon*T))
        d_new = r_new + (1-epsilon)*T
        del J[id]
        J[id] = (w, r_new, d_new)
        interval = (r_new, d_new)
        intervals.append(interval)

    intervals = sorted(list(set(intervals)), key=lambda x: x[0])
    J_new = {}
    for interval in intervals:
        J_new[interval] = 0
    for id in ids:
        w, r, d = J[id]
        interval = (r, d)
        J_new[interval] += w
    del J
    J = {}
    for id in range(0, len(intervals)):
        interval = intervals[id]
        r, d = interval
        w = J_new[interval]
        J[id + 1] = (w, r, d)


    return J


def print_speed_list(speed_list):
    speed_list_keys = sorted(speed_list.keys(), key=lambda x: x[0])
    for interval in speed_list_keys:
        start = interval[0]
        end = interval[1]
        speed = speed_list[interval]
        print(start, "---", end, "--->speed= ", speed)




def compute_speed_per_integer_time(J_sol):
    #input: solution instance --> dictionary key --> id
    #                                        value--> (weight, speed, start processing, end processing)
    #output: a list of speed at any point in time (the length of the list is the same as
    ids = sorted(J_sol.keys())

    # here we deal with the case where the last job is of zero weight and we

    _, speed, start_of_time, _ = J_sol[ids[0]]
    _, _, _, end_of_time = J_sol[ids[-1]]
    start_of_time = int(start_of_time)
    end_of_time = int(end_of_time)



    start = start_of_time
    speed_list = [Fraction(1,1)]*int(end_of_time-start_of_time)
    for id in ids:
        w_id, s_id, start_id, end_id = J_sol[id]
        floored_start_of_interval = floor(start_id)
        ceiled_end_of_interval = ceil(end_id)
        speed_list[floored_start_of_interval: ceiled_end_of_interval] = [s_id]*(ceiled_end_of_interval-floored_start_of_interval)
    #this is a sanity check
    if len(speed_list) != (end_of_time-start_of_time):
        print("the lenghts of the speed list and the time horizon do not match")
        print("the lenght = ", len(speed_list))
        print("the time horizon is set to be = ", end_of_time-start_of_time)
    return speed_list


def is_Sol_correct(J_sol, J):
    # input: J a list of jobs in the form (weight, release time, deadline)
    #       J_sol the solution of this instance in the form (weight, speed, start processing, end processing)
    # (1) feasibility of the solution
    # (2) optimality conditions--> forall jobs, any time outside their processing time but inside their release-deadline
    #                              the speed of the processor is higher (>=)

    # (1) check feasibility
    ids = sorted(J.keys())
    speed_list = compute_speed_per_integer_time(J_sol)
    prev_processing_end = 0.0
    for id in ids:
        w, r, d = J[id]
        w_sol, speed_sol, start, end = J_sol[id]
        validity = (w_sol == w) and (r <=start) and (end <=d) and (start < end) and speed_sol==Fraction(w,(end-start)) and (start>=prev_processing_end)
        if not validity:
            print("Houston, we have a problem with job = ", id)
            print(J[id])
            print(J_sol[id])
            print("or maybe the processing times")
            print("start = ", start, "should be more than previous processing ends = ", prev_processing_end)
            exit()
        prev_processing_end = end
    # (2) check the optimality conditions
        # for any t in [r, d ] speed[t] >= speed_sol
        r = int(r)
        d = int(d)

        optimality = all([(speed_list[t] >= speed_sol) for t in range(r,d)])
        if not optimality:
            print("Houston, your instance is feasible but not optimal")
            print(J_sol[id])
            print(r, d)
            print(optimality)
            exit()


    print("congratulations, your algorithm is great")
    return 0




def find_densest_interval(J):
     # given a Job instance, the function finds the densest interval
     # Input: a dictionary J where: (1) key --> job id
     #                              (2) value --> (job_weight, release time, deadline)
     # Output: (1) the start-end of the densest interval
     #         (2) the jobs\' ids contained in the denset interval
     #         (3) the density of the densest interval

    # step1: finding all the distinct arrival times and deadlines
    arrivals = find_arrivals(J)
    deadlines = find_deadlines(J)

    # step2: trying all the different pairs of arrival - deadline and compute the maximum density
    maximum_density=Fraction(0,100)
    best_ids = []
    best_arrival =0
    best_deadline = 0
    for arrival in arrivals:
        for deadline in deadlines:
            if arrival == deadline:
                continue
            total_weight, ids = find_weight(J, arrival, deadline)

            density = Fraction(total_weight, deadline-arrival)
            if density>=maximum_density:
                best_ids = ids
                maximum_density=density
                best_arrival=arrival
                best_deadline=deadline
    return maximum_density, best_ids, best_arrival, best_deadline



def compute_optimal_instance(J):
    # input: a dictionary of entries of the form key-->id
    #                                             value: weight, release time, deadline
    # output: a dictionary where all the information of the optimal schedule are stored (remember that there is not nested job
    #         key ---> id (the same as in the inout)
    #         value ---> weight, speed in the optimal schedule, starting to process, finish to process
    J_save = copy.deepcopy(J)
    J_sol = {}
    for id in J.keys():
        w, r, d = J[id]
        r_id = Fraction(r,1)
        d_id = Fraction(d,1)
        speed_id = Fraction(0,1)
        J_sol[id] = (w, speed_id, r_id, d_id)
    while J:
        speed, ids, r, d = find_densest_interval(J)
        for id in ids:
            speed_id = speed
            w_id, _, r_id, d_id  = J_sol[id]
            J_sol[id] = (w_id, speed_id, r_id, d_id)
        adjust_instance(J, r, d)

    # now we need to adjust the release time , deadline to the actual processing times
    ids = sorted(J_sol.keys())
    previous_deadline = Fraction(0,1)
    for id in ids:
        w_id, speed_id, r_id, d_id = J_sol[id]
        start = previous_deadline
        if w_id == 0 and speed_id == 0:
            end = start
        else:
            end = Fraction(w_id,speed_id) + start
        previous_deadline = end
        J_sol[id] = (w_id, speed_id, start, end)
    return J_sol


def print_solution(J_sol):
    # prints the solution instance J_sol in a nice way
    ids = sorted(J_sol.keys())
    for id in ids:
        w_id, speed_id, r_id, d_id = J_sol[id]
        srt_to_print = str(id) + "/weight="+str(w_id) + "/speed=" + str(float(speed_id))+"/start="+str(float(r_id))+ "/end="+str(float(d_id))
        print(srt_to_print)

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
    speed_list_dictionary[(start, end)] = Fraction(0,1)
    for id in ids:
        w_id, r, d  = J[id]
        speed = Fraction(w_id, d-r)
        r_id = Fraction(r,1)
        d_id = Fraction(d,1)
        interval = (r_id, d_id)
        add_speed(speed_list_dictionary, speed, interval)
    return speed_list_dictionary

def Avg_rate_integer_speedlist(J):
    # the same as before but now we return a speed list which represents the speed during integer time
    # mainly for debugging purposes

    ids = sorted(J.keys())
    _, start, _ = J[ids[0]]
    _, _, end   = J[ids[-1]]
    start = int(start)
    end = int(end)

    speed_list = [Fraction(0,1)]*(end - start)
    for id in ids:
        w_id, r, d = J[id]
        r = int(r)
        d = int(d)
        speed = Fraction(w_id, d-r)
        for t in range(r,d):
            speed_list[t]+=speed
    return speed_list




def densest_interval_BKP(J, t, dt):
    # input: a job instance J
    #       time t (for which we ask the densest interval around)
    #       granularity dt

    # computes density of a densest interval for BKP
    keys = sorted(J.keys())
    min_time = J[keys[0]][1]
    max_time = J[keys[-1]][2]
    if t<min_time: return 0
    if t>max_time: return 0
    time_arg = t+dt
    max_density = 0
    bound = (e/(e-1))*float(t)-float(min_time)/(e-1)
    bound = max(bound, float(max_time))
    while float(time_arg)<2*bound:
        a = e*t-((e-1)*time_arg)
        b = time_arg
        total_weight_released = 0
        i=keys[0]
        while i<=keys[-1]:
            if ((J[i][1]<=t) and (J[i][1]>=a) and (J[i][2]<=b)):
                total_weight_released+=J[i][0]
            i+=1
        density = total_weight_released/(e*(time_arg-t))
        if(density>max_density):
            max_density=density
        time_arg+=dt
    return max_density






def robustify(speed_to_make_robust, epsilon, T, dt):
    # input: (1) speed_to_make_robust is the speed list we want to robustify
    #        (2) epsilon is the robustness parameter
    #        (3) T is the uniform deadline in our instances
    #        (4) dt is the granularity of our output
    # output: speed which is the output speed list that is actually ran by our algorithm
    
    dim = int(float(epsilon*T)/float(dt)) +1
    mask_val = 1.0/float(dim)
    mask = [mask_val]*dim
    lenght_of_the_solution = len(speed_to_make_robust)
    speed_to_make_robust = np.array(speed_to_make_robust)
    mask = np.array(mask)
    speed = np.convolve(mask, speed_to_make_robust)
    
    return speed




def scale_down_epsilon(epsilon, alpha , error):
    # input: epsilon that we wish to have (1+epsilon) consistency
    #        alpha is the convexity parameter of the problem
    #        error is how
    
    
    # since we want our algorithm to be (1+epsilon) consistent
    # we need ((1+new_epsilon)/(1-new_epsilon) )**alpha = (1 + epsilon)
    search_granularity = 1000000
    epsilons = np.linspace(0,0.9,search_granularity)
    for new_epsilon in epsilons:
        diff = ((1+ new_epsilon)/ (1 - new_epsilon) )**alpha - (1+epsilon)
        if abs(diff) < error:
            break
    new_epsilon = Fraction.from_float(new_epsilon).limit_denominator()
    return new_epsilon




def speed_dictionary_to_list(speed_dict, dt, mul_factor):
    # input: speed_dict which is the speed function in a dictionary form
    #        dt which is teh desired granularity of the output list
    #        mul_factor is used if we want to multiply every speed by a multiplicative factor
    # output: s which is the speed in a list form
    
    
    intervals = list(speed_dict.keys())
    intervals.sort(key=lambda x: x[1])
    
    # multiply with mul_factor the speed at any moment if needed
    for interval in intervals:        
        old_speed = speed_dict[interval]
        new_speed = old_speed*mul_factor
        del speed_dict[interval]
        speed_dict[interval] = new_speed
        
    #now we will turn the dictionary to a speed list with a granularity of dt
    s = []
    t = 0
    for interval in intervals:
        start, end = interval
        speed = speed_dict[interval]
        while t<end:
            s.append(speed)
            t+=dt
    return s





