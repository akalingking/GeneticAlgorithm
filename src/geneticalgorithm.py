# The MIT License (MIT)
# 
# Copyright (c) 2016 Ariel Kalingking <akalingking@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import pandas as pd
import numpy as np
import itertools
import neuralnetwork as N
import dataset
import futures
from multiprocessing import cpu_count
from pip.commands.search import print_results
from sklearn import metrics

rng = np.random.RandomState(5000)

chromosomes = dict(
        network           = [{'input':9,'hidden_min':2,'hidden_max':4,'hidden_sizes':[18,24,36,54,81],'output':1}],
        connection_rate   = [1.0,.9,.8,.7,.6],
        learning_rate     = [.001,.002,.003,.004,.005,.006,.007,.008,.009,.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.2,.3,.4,.5,.6],
        learning_momentum = [.001,.002,.003,.004,.005,.006,.007,.008,.009,.01,.02,.03,.04,.05,.06,.07,.08,.09,.1,.2,.3],
        initial_weight    = [.70,.71,.72,.73,.74,.75,.76,.77,.78,.79,.8,.81,.82,.83,.84],
        hidden_activation = [N.SIGMOID,N.SIGMOID_SYMMETRIC,N.SIGMOID_STEPWISE, N.SIGMOID_SYMMETRIC_STEPWISE],
        output_activation = [N.SIGMOID,N.SIGMOID_SYMMETRIC,N.SIGMOID_STEPWISE, N.SIGMOID_SYMMETRIC_STEPWISE],
        training_algorithm = [N.TRAIN_INCREMENTAL, N.TRAIN_BATCH, N.TRAIN_QUICKPROP, N.TRAIN_RPROP, N.TRAIN_SARPROP],
    )

chromosomes_fixed = dict(
                         desired_error     = 0.0001,
                         epoch             = 100,
                         show              = 0,
                         )

def get_random_network(params):
    network_ = [params['input'],]
    n_hidden = rng.randint(params['hidden_min'],params['hidden_max']+1)
    for i in xrange(n_hidden):
        hidden_size_index = rng.randint(0, len(params['hidden_sizes']))
        hidden_size = params['hidden_sizes'][hidden_size_index]
        network_.append(hidden_size)
    network_.append(params['output'])
    return network_
    
    
def individual():
    keys = chromosomes.keys()
    individual_ = []
    for i,key in enumerate(keys):
        if key == 'network':
            network_ = get_random_network(chromosomes['network'][0])
            individual_.append(network_)
        else:
            index = 0 if len(chromosomes[key]) == 1 else rng.randint(0, len(chromosomes[key]), 1) 
            individual_.append(chromosomes[key][index])
    
    return individual_


def get_random_individual(population, low=None, high=None):
    assert (len(population) > 1)
    individual = None
    if low is None and high is not None:
        assert(False)
    if low is None and high is None:
        individual = population[rng.randint(0, len(population), 1)]
    elif low is not None and high is None:
        individual = population[rng.randint(low, len(population), 1)]
    elif low is not None and high is not None:
        individual = population[rng.randint(low, high, 1)]
    else:
        assert(False)
    assert(individual is not None)
    return individual


def mutate_individual(individual):
    assert (len(chromosomes) > 1)
    # randomize how many dna
    n_traits = rng.randint(0, len(chromosomes), 1)
    
    trait_index_set = set()
    while (len(trait_index_set) < n_traits):
        trait_index_set.add(rng.randint(0, len(chromosomes)))
    trait_indices =  list(trait_index_set)
    
    keys = chromosomes.keys()
    for trait_index in trait_indices:
        key = keys[trait_index]
        if (key=='network'):
            network_ = get_random_network(chromosomes['network'][0])
            individual[trait_index] = network_
        else:
            trait_value_index = 0 if len(chromosomes[key]) == 1 else rng.randint(0, len(chromosomes[key]), 1)
            individual[trait_index] = chromosomes[key][trait_value_index]
    #print 'mutate_individual: %d traits  %s' % (n_traits, individual)
    return individual
     

def population(size=100):
    population_ = []
    while len(population_) < size:
        individual_ = individual()
        if individual_ not in population_:
            population_.append(individual_)
        else:
            print 'population: individual already in population.'
        
    return population_


def fitness(model, individual, x_train, y_train, x_valid, y_valid, target):
    
    dict_ = dict(zip(chromosomes.keys(), individual))
    dict_.update(chromosomes_fixed)
    
    model.set_params(**dict_)
    model.fit(x_train, y_train)
    #score = model.score(x_train, y_train)
    #score = model.score(x_valid, y_valid)
    y_pred = model.predict_proba(x_valid)[:,1]
#     print "y_pred:", y_pred.shape
#     print "y_valid:", y_valid.shape
    assert (y_pred.shape[0] == y_valid.shape[0])
    score = metrics.log_loss(y_valid, y_pred, normalize=True)
    print 'fitness: score=%f for %s' % (score, individual)
    
    return abs(score - target)


def fitness_(model, index, individual, x_train, y_train, x_valid, y_target, target):
    score = fitness(model, individual, x_train, y_train, x_valid, y_target, target)
    
    dict_ = dict(zip(chromosomes.keys(), individual))
    dict_.update(chromosomes_fixed)
    
    return (index, score)


def print_grade(generation, results, pop):
    data = [(results[i],ind) for i,ind in enumerate(pop)]
    data.sort()
    columns=['result']
    columns.extend(chromosomes.keys())
    data_ = []
    for i, item in enumerate(data):
        row = []
        row.append(item[0])
        row.extend(item[1])
        data_.append(row)
    f = pd.DataFrame(data=data_,columns=columns)
    f.to_csv('../result/result_'+ str(np.mean(results)).replace('.', '_') + '_' + str(generation) + '.csv')
    
    
def grade(model, generation, pop, x_train, y_train, x_valid, y_valid, target, do_parallel=False):
    results = np.zeros(len(pop))
    print 'grade:', chromosomes.keys()
    if do_parallel:
        # pickle is having error during multi process.
        with futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            for i, individual in enumerate(pop):
                f = executor.submit(fitness_, model, i, individual, x_train, y_train, x_valid, y_valid, target)
                index, score = f.result()
                results[index] = score
    else:
        for i, individual in enumerate(pop):
            results[i] = fitness(model, individual, x_train, y_train, x_valid, y_valid, target)
            
    mean_ = np.mean(results)
    
    print_grade(generation, results, pop)
    
    return mean_
        
    

def evolve(model, pop, x_train, y_train, x_valid, y_valid, target, 
           retain=0.2, random_select=0.05, mutate=0.01):
    graded = [(fitness(model, ind, x_train, y_train, x_valid, y_valid, target),ind) for ind in pop]
    graded = [ind[1] for ind in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    
    # randomly add other individuals to promote genetic diversity
    random_individual = []
    for individual_ in graded[retain_length:]:
        # Select only from graded[retain_length:]
        if random_select > rng.rand():
            #individual_ = graded[rng.randint(retain_length, len(graded)-1, 1)]
            individual_ = get_random_individual(graded, retain_length)
            if individual_ not in random_individual and individual not in parents:
                random_individual.append(individual_)
            else:
                print 'evolve: Random individual already present %d=%d from %d.' % (retain_length, len(random_individual), len(graded))
    
    print 'evolve: Extending parent of %d random individuals.' % len(random_individual)
    parents.extend(random_individual)
            
            
    # Mutate chromosomes of some individuals
    for i in xrange(len(parents)):
        if mutate > rng.rand():
            parents[i] = mutate_individual(parents[i])
    
    # Crossover to create children
    max_search_for_mate_cnt = 10
    search_for_mate_cnt = 0 
    parents_len = len(parents)
    desired_len = len(pop) - parents_len
    children = []
    while (len(children) < desired_len):
        assert (parents_len > 1)
        male = rng.randint(0, parents_len, 1)
        female = rng.randint(0, parents_len, 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            # make sure we have enough permutations
            if child not in children and child not in parents:
                children.append(child)
            else:
                print 'evolve: Child already in population parents=%d children=%d desired=%d' % \
                    (parents_len, len(children), desired_len)
        else:
            search_for_mate_cnt += 1
            if search_for_mate_cnt == max_search_for_mate_cnt:
                while (True):
                    #random_individual_ = graded[rng.randint(retain_length, len(graded)-1, 1)]
                    selection_method = 1 if rng.rand() > 0.5 else 0
                    if (selection_method == 0):
                        random_individual_ = get_random_individual(graded, low=retain_length)
                        if random_individual_ not in parents and random_individual not in children:
                            print 'evolve: Adding new parent by random select.'
                            parents.append(random_individual_)
                            parents_len = len(parents)
                            desired_len = len(pop) - parents_len
                            search_for_mate_cnt = 0
                            break;
                    elif (selection_method == 1):
                        # Mutate individual from the high grade population
                        random_parent = get_random_individual(graded, low=0, high=retain_length-1);
                        random_parent = mutate_individual(random_parent)
                        if random_parent not in parents and random_parent not in children:
                            print 'evolve: Adding new parent by mutation.'
                            parents.append(random_parent)
                            parents_len = len(parents)
                            desired_len = len(pop) - parents_len
                            search_for_mate_cnt = 0
                            break
                    else:
                        assert (False)
                        
                    print 'evolve: Failed to add parent, retrying..'                        
                        
            print 'evolve: Got same gender on crossover, retry search for male and female...'
    
    parents.extend(children)
    # Make sure we keep teh sam size
    assert len(parents) == len(pop)
    return parents


def find_best_individual(model, pop, x_train, y_train, x_valid, y_valid, target):
    grades = [(fitness(model, ind, x_train, y_train, x_valid, y_valid, target), ind) for ind in pop]
    return sorted(grades)


            
def loaddataset():
    params =    {   
            'TrainFile'             : '../data/train.csv',
            'TestFile'              : '../data/test.csv',
            'TrainSize'             : 0.9
        }

    df = dataset.load_train_data(params)
    train_data = df.values
    
    # Start in the PClass column, we will not be using the passengerid
    X_train = train_data[:,2:]
    Y_train = train_data[:,0].astype(int)
    
    # Partition training data
    trainSize = int(params['TrainSize'] * np.size(Y_train))
    x_train, x_valid = X_train[:trainSize, :], X_train[trainSize:,:]
    y_train, y_valid = Y_train[:trainSize], Y_train[trainSize:]
    
    return [x_train, y_train, x_valid, y_valid]





def main():
    max_retention_rate = 0.5
    min_retention_rate = 0.3
    retention_rate_delta = 0.1
    random_select = 0.001#05
    mutate = .001
    
    #target = 1.0 # accuracy
    target = 0.0 # rmse
    n_generation = 30
    n_population = 30
    
    model = N.NeuralNetwork()
    x_train, y_train, x_valid, y_valid  = loaddataset()
    
    p = population(n_population)
    
    
    current_retention_rate = max_retention_rate
    best_grade = np.inf
    previous_grade = np.inf
    previous_retention_rate = max_retention_rate
    #fitness_history = [grade(model, p, x_train, y_train, x_valid, y_valid, target), ]
    fitness_history = []
    for i in xrange(n_generation):
        print 'Running generation %d...' % i
        
        p = evolve(model, p, x_train, y_train, x_valid, y_valid, target, 
            retain=current_retention_rate,
            random_select=random_select,
            mutate=mutate)
        
        grade_ = grade(model, i, p, x_train, y_train, x_valid, y_valid, target)
        
        print 'Generation=%d grade=%f' % (i, grade_)
        fitness_history.append(grade_)
        
        if grade_ > previous_grade:
            current_retention_rate *= (1.0 - retention_rate_delta)
            if current_retention_rate < min_retention_rate:
                current_retention_rate = min_retention_rate
            else:
                print 'evolve: grade=%f > previous grade=%f, reducing retention from %f to %f' % \
                    (grade_, previous_grade, previous_retention_rate, current_retention_rate)
        else:
            current_retention_rate *= (1.0 + retention_rate_delta)
            if current_retention_rate > max_retention_rate:
                current_retention_rate = max_retention_rate
            else:
                print 'evolve: grade=%f < previous grade=%f, increasing retention from %f to %f' % \
                    (grade_, previous_grade, previous_retention_rate, current_retention_rate)
                
        previous_retention_rate = current_retention_rate
        previous_grade = grade_
        if grade_ <  best_grade: best_grade = grade_
    
    for i in fitness_history: print i
    
    print 'Best grade %f' % best_grade    
    best = find_best_individual(model, p, x_train, y_train, x_valid, y_valid, target)
    
    # Reports
    dict_ = dict(zip(chromosomes.keys(), best[0][1]))
    dict_.update(chromosomes_fixed)
    model.set_params(**dict_)
    print 'Best Model:', model
    print '%s' % chromosomes.keys() 
    for item in best: print item



if __name__=='__main__':
    main()
