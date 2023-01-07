import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import numpy as np
ST = []
PD = []
SD = []
last_fitness = 0
best_ST = []
best_PD = []
best_SD = []


def inspect(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format
          (fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    #idx=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[2])
    idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[2]
    print("Change     = {change}".format
          (change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    print(ga_instance.population)
    ST.append([agent[0] for agent in ga_instance.population])
    PD.append([agent[1] for agent in ga_instance.population])
    SD.append([agent[2] for agent in ga_instance.population])
    best_ST.append(ga_instance.population[idx][0])
    best_PD.append(ga_instance.population[idx][1])
    best_SD.append(ga_instance.population[idx][2])
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def plot_hyper_parameters():
    generation_axis = [i for i in range(0, 40)]
    generation_axis_best = [i for i in range(0, 40)]

    generation_axis = np.repeat(generation_axis, 10)
    print(generation_axis)
    print(ST)

    fig = plt.figure(100)
    plt.scatter(generation_axis, ST)
    plt.scatter(generation_axis_best, best_ST, color="red")
    plt.title("ST - Hyper-parameter")
    plt.xlabel("hyper-iteration")
    plt.ylabel("value")
    plt.show()

    fig = plt.figure(200)
    plt.scatter(generation_axis, PD)
    plt.scatter(generation_axis_best, best_PD, color="red")
    plt.title("PD - Hyper-parameter")
    plt.xlabel("hyper-iteration")
    plt.ylabel("value")
    plt.show()
    # plt.scatter(generation_axis, SD)

    fig = plt.figure(300)
    plt.scatter(generation_axis, SD)
    plt.scatter(generation_axis_best, best_SD, color="red")
    plt.title("SD - Hyper-parameter")
    plt.xlabel("hyper-iteration")
    plt.ylabel("value")
    plt.show()
