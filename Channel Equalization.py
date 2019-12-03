import regex as re
import numpy as np
import random as rd



####################    defining parameters     ###########
n = None
l = None
variance = None
w = []
train = None
test = None



##############   reading config.txt file   ################
file = open("config.txt", "r")
temp_read_file = 1
str_read_file = []
for line in file:
    str_read_file = line.split()
    if temp_read_file == 1:
        n = int(str_read_file[0])
        l = int(str_read_file[1])
        temp_read_file  = 2
    elif temp_read_file == 2:
        for i in range(len(str_read_file)):
            w.append(float(str_read_file[i]))
        temp_read_file = 3
    elif temp_read_file == 3:
        variance = float(str_read_file[0])
file.close()



##############   reading train.txt file   ################
file = open("train.txt", "r")
train = file.readline()
file.close()




##############   reading test.txt file   ################
file = open("test.txt", "r")
test = file.readline()
file.close()




###################      defination of state     #####################
class State(object):
    def __init__(self, name=None, mean=None, var=None):
        self.name = name
        self.mean = mean
        self.var = var


##############    decleration of states and prior probability      #######################
states = []
prior_probability = []
total_num_of_bits = len(train)

##################     creating states     #######################
for i in range(2**n):
    states.append(State(bin(i)[2:].zfill(n)))


################ calculating prior probability  #################
def occurrences(text, sub):
    return len(re.findall('(?={0})'.format(re.escape(sub)), text))

zero_padded_train = train.zfill(total_num_of_bits + n -1)
for i in range(2**n):
    str_prior_prob = states[i].name
    prior_probability.append(((occurrences(zero_padded_train, str_prior_prob) * 1.0 )/ total_num_of_bits))
print "prior probabilities : "
print prior_probability
print "\n"



############### calculating transition probabilities   ###############
transition_probability = np.zeros((2**n, 2**n))
for i in range(2**n):
    for j in range(2**n):
        str_trans_prob = states[i].name + states[j].name
        if occurrences(zero_padded_train, states[i].name) == 0:
            transition_probability[i][j] = 0
        else:
            transition_probability[i][j] = (occurrences(zero_padded_train, str_trans_prob) * 1.0) / occurrences(zero_padded_train, states[i].name)
print "transition probabity : ( given that i index, probability y index )"

def show_transition_prob():
    for i in range(2**n):
        for j in range(2**n):
            print "{0:.3f}".format(transition_probability[i][j]),
        print
show_transition_prob()



#####   finding mean and variance for each state    #####
x_k = []
temp_x_k_arr = np.array(w)
reversed_w = temp_x_k_arr[::-1]
for i in range(len(train)):
    temp_x_k = 0
    for j in range(len(reversed_w)):
        temp_x_k += reversed_w[j]*float(zero_padded_train[i+j])
    x_k.append(temp_x_k)


def find_indexes(seq, sub):
    return [i for i in range(0, len(seq), 1) if seq[i:len(sub)+i] == sub]

#rd.seed(11)
for i in range(len(states)):
    arr_index = find_indexes(train, states[i].name)
    sum_x_k = 0
    list_x_ks = []
    for j in range(len(arr_index)):
        list_x_ks.append(x_k[arr_index[j]] + rd.random())
        sum_x_k += list_x_ks[j]
    if len(arr_index) == 0:
        states[i].mean = 0
        states[i].var = 0
    else:
        states[i].mean = sum_x_k / len(arr_index)
        states[i].var = np.var(list_x_ks)
    print "mean and variance for ",
    print states[i].name,
    print " : ",
    print states[i].mean,
    print " , ",
    print states[i].var



print "\n\ntesting starts : \n"



##################   end of training and start of testing   ####################

#################         calculating x_k for test         #############################
x_k_test = []
zero_padded_test = test.zfill(len(test) + n -1)
for i in range(len(test)):
    temp_x_k_test = 0
    for j in range(len(w)):
        temp_x_k_test += w[j]*float(zero_padded_test[i+j])
    x_k_test.append(temp_x_k_test)


######################     calculating received signal    ################################
def name_map_possible_states(given_name):
    temp_s0 = '0'
    temp_s1 = '1'
    temp_str1 = given_name[1:n] + temp_s0
    temp_str2 = given_name[1:n] + temp_s1
    return name_to_indices(temp_str1), name_to_indices(temp_str2)


def name_to_indices(given_name):
    for i in range(2**n):
        if given_name==states[i].name:
            return i


class SeqFinder(object):
    def __init__(self, parent=None, value=None):
        self.parent = parent
        self.value = value



test_seq_finder = [ [ 0 for y in range( len(x_k_test) ) ] for x in range( 2**n ) ]
for y in range(1, len(x_k_test)):
    for x in range(len(states)):
        index_1, index_2 = name_map_possible_states(states[x].name)
        if y == 1:
            likelihood_index_1 = prior_probability[index_1] * transition_probability[index_1][x] * transition_probability[index_1][index_1] * transition_probability[index_1][x]
            likelihood_index_2 = prior_probability[index_2] *transition_probability[index_2][x] * transition_probability[index_1][index_2] * transition_probability[index_1][x]
            if likelihood_index_1 > likelihood_index_2:
                seq = SeqFinder(states[index_1].name,likelihood_index_1)
            else:
                seq = SeqFinder(states[index_2].name, likelihood_index_2)
            test_seq_finder[x][y] = seq
        else:
            likelihood_index_1 = test_seq_finder[index_1][y-1].value * transition_probability[index_1][index_1] * transition_probability[index_1][x]
            likelihood_index_2 =test_seq_finder[index_2][y-1].value * transition_probability[index_1][index_2] * transition_probability[index_1][x]
            if likelihood_index_1 > likelihood_index_2:
                seq = SeqFinder(states[index_1].name, likelihood_index_1)
            else:
                seq = SeqFinder(states[index_2].name, likelihood_index_2)
            test_seq_finder[x][y] = seq

max = -9999
index = None
for x in range(len(states)):
    if test_seq_finder[x][len(test_seq_finder[0]) - 1].value > max:
        max = test_seq_finder[x][len(test_seq_finder[0]) - 1].value
        index = x


##################### finding received signal sequence  ###################
reversed_predicted_signal = bin(index)[2:].zfill(n)
for i in range(len(test_seq_finder[0]) - 1, 0, -1):
    reversed_predicted_signal += test_seq_finder[index][i].parent
    index = int(test_seq_finder[index][i].parent, 2)

#print reversed_predicted_signal

predicted_signal = ''
for i in range(len(reversed_predicted_signal)-1, -1, -1):
    predicted_signal += reversed_predicted_signal[i]

answer = ""
temp_var = 1
for i in range(len(predicted_signal)):
    if temp_var == n:
        answer += predicted_signal[i]
        temp_var = 1
    else:
        temp_var +=1

#print answer


###########################       calculate accuracy         ########################
count = 0
for i in range(len(answer)):
    if answer[i] == test[i]:
        count += 1

print 'Accuracy : ',
print "{0:.2f}".format((count * 1.0)/len(test)),
print '%'