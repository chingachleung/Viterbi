states = ['Noun', 'Verb', 'Other']
N = len(states)
observations = "time flies like an arrow".split()
T = len(observations)

init_prob = {"Noun":1/2, "Verb":0, "Other":1/2}
transition_probability = {"Noun":{"Noun":1/3, "Verb":1/3,"Other":0,"End":1/3},
                          "Verb":{"Noun":1/3,"Verb":0,"Other":1/3,"End":1/3},
                          "Other":{"Noun":1/2,"Verb":0,"Other":1/2,"End":0}} # change other-noun from 1/2 to zeoro

emission_probability = {"Noun":{"an":0,"arrow":1/5,"bear":2/5,"flies":1/5,"like":0,"to":0,"time":1/5},
                        "Verb":{"an":0,"arrow":0,"bear":1/5,"flies":2/5,"like":1/5,"to":0,"time":1/5},
                       "Other":{"an":2/5,"arrow":0,"bear":0,"flies":0,"like":1/5,"to":2/5,"time":0}}

import numpy as np 
viterbi = np.zeros((N,T))

#initialization (first observation)
for s in range(N):
    viterbi[s,0] = init_prob[states[s]] * emission_probability[states[s]][observations[0]]
    
#recusion
from collections import defaultdict
backpointer_dict = {}
defaultdict(list)
for o in range(1,T): #looping from the 2nd to the last word
    connected = False 
    for cur_s in range(N): #looping over each state
        #temp_val = 0
        for pre_s in range (N):
            cur_emission_prob = emission_probability[states[cur_s]][observations[o]] # constant for the smallest loop
            pre_path_prob = viterbi[pre_s,o-1]
            transition_prob  = transition_probability[states[pre_s]][states[cur_s]]
            if (pre_path_prob == 0 or transition_prob == 0):
                print("{} {} gives no additional prob to {} {}:".format(pre_s,o-1,cur_s,o))
            else:
                temp_val = cur_emission_prob * pre_path_prob *transition_prob
                if temp_val > viterbi[cur_s,o]:
                    viterbi[cur_s,o] = round(temp_val,8)
                    backpointer_dict[cur_s,o] = pre_s,o-1
                    print("updated prob of {} {}:".format(cur_s,o),viterbi[cur_s,o] )
                    connected = True 
    if connected == False:
        print("no valid parse available")
        break
     
max_index = (np.argmax(viterbi[:,-1]), 4) # find the higest prob index in the last observation 
max_value = np.amax(viterbi[:,-1]) # find the higest value in the last column
parse = [max_index]

for i in range(T-1) :
    parse.append(backpointer_dict[max_index])
    max_index = backpointer_dict[max_index]
 
parse.reverse() #reverse the order of the list 

for item in parse:
    print(observations[item[1]], states[item[0]], "->", end=" ")

print('\n')    
print("with a probability of {}".format(max_value))    