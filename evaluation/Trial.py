import json

import matplotlib

import matplotlib.pyplot as plt


class Trial:
    nr_iterations = 200
    nr_hypergenerations = 40

    def __init__(self, problem, heuristics, nr_executions=100):
        self.problem = problem
        self.heuristics = heuristics
        self.nr_executions = nr_executions

    def obtain_rank(self):
        results = {heuristic.__class__.__name__: [] for heuristic in self.heuristics}

        for heuristic in self.heuristics:
            for execution in range(self.nr_executions):
                if 'HyperHeuristic' in heuristic.__class__.__name__:
                    best_fitness = heuristic.optimize(objective_func=self.problem['objective_func'],
                                                      dimensions=self.problem['dimensions'],
                                                      bounds=self.problem['bounds'],
                                                      nr_hypergenerations=self.nr_hypergenerations).pop()
                else:
                    histogram, best_position, best_fitness = heuristic.optimize(
                        objective_func=self.problem['objective_func'],
                        dimensions=self.problem['dimensions'],
                        bounds=self.problem['bounds'],
                        nr_iterations=self.nr_iterations)
                results[heuristic.__class__.__name__].append(best_fitness)

        self.write_best_to_file(results)
        ranks = {key: self.assign_rank(results[key], self.problem['global_value']) for key in results}
        return ranks

    def obtain_rank_verbose(self):
        ranks = {}
        results = {}
        for heuristic in self.heuristics:
            for execution in range(self.nr_executions):
                if 'HyperHeuristic' in heuristic.__class__.__name__:
                    histogram = heuristic.optimize(objective_func=self.problem['objective_func'],
                                                   dimensions=self.problem['dimensions'],
                                                   bounds=self.problem['bounds'],
                                                   nr_hypergenerations=self.nr_hypergenerations)
                else:
                    histogram, best_position, best_fitness = heuristic.optimize(
                        objective_func=self.problem['objective_func'],
                        dimensions=self.problem['dimensions'],
                        bounds=self.problem['bounds'],
                        nr_iterations=self.nr_iterations)
                results[execution] = histogram
            self.write_verbose_histogram_to_file(results, heuristic)
            last_results = [results[key][-1] for key in results]
            ranks[heuristic.__class__.__name__] = self.assign_rank(last_results, self.problem['global_value'])
        return ranks

    def obtain_histogram(self):
        results = {}
        for heuristic in self.heuristics:
            for execution in range(self.nr_executions):
                if 'HyperHeuristic' in heuristic.__class__.__name__:
                    histogram = heuristic.optimize(objective_func=self.problem['objective_func'],
                                                   dimensions=self.problem['dimensions'],
                                                   bounds=self.problem['bounds'],
                                                   nr_hypergenerations=self.nr_hypergenerations)
                else:
                    histogram, best_position, best_fitness = heuristic.optimize(
                        objective_func=self.problem['objective_func'],
                        dimensions=self.problem['dimensions'],
                        bounds=self.problem['bounds'],
                        nr_iterations=self.nr_iterations)
                results[heuristic.__class__.__name__] = histogram
        self.write_histogram_to_file(results)
        return results

    def write_best_to_file(self, results):
        with open(self.problem["objective_func"].__name__+'.json', 'w') as outfile:
            json.dump(results, outfile)

    def write_histogram_to_file(self, results):
        with open(self.problem["objective_func"].__name__ + '_histogram.json', 'w') as outfile:
            json.dump(results, outfile)

    def write_verbose_histogram_to_file(self, results, heuristic):
        with open(self.problem["objective_func"].__name__ + '_' + heuristic.__class__.__name__ + '_histogram.json',
                  'w') as outfile:
            json.dump(results, outfile)

    def obtain_rank_from_file(self):
        with open(self.problem["objective_func"].__name__+'.json') as json_file:
            results = json.load(json_file)
        ranks = {key: self.assign_rank(results[key], self.problem['global_value']) for key in results}
        return ranks

    def obtain_rank_verbose_from_file(self):
        ranks = {}
        for heuristic in self.heuristics:
            with open(self.problem[
                          "objective_func"].__name__ + '_' + heuristic.__class__.__name__ + '_histogram.json') as json_file:
                results = json.load(json_file)
                last_results = [results[key][-1] for key in results]
                ranks[heuristic.__class__.__name__] = self.assign_rank(last_results, self.problem['global_value'])
        return ranks

    def obtain_median_rank_verbose_from_file(self):
        ranks = {}
        for heuristic in self.heuristics:
            with open(self.problem[
                          "objective_func"].__name__ + '_' + heuristic.__class__.__name__ + '_histogram.json') as json_file:
                results = json.load(json_file)
                if 'time' in results:
                    del results['time']
                last_results = [ results[key][-1] for key in results]
                ranks[heuristic.__class__.__name__] = self.assign_median_rank(last_results, self.problem['global_value'])
        return ranks

    def plot_convergence(self):
        colors = ['black', 'blue', 'brown', 'green', 'purple', 'cyan', 'magenta', 'yellow']
        for idx, heuristic in enumerate(self.heuristics):
            with open(self.problem['objective_func'].__name__ + '_' + heuristic.__class__.__name__ + '_histogram.json') as json_file:
                results = json.load(json_file)
            avg_histogram = []
            if 'HyperHeuristic' not in heuristic.__class__.__name__:
                for i in range(0, self.nr_iterations):
                    s = 0
                    if i % 5 == 0:
                        for e in range(0, self.nr_executions):
                            s += results[str(e)][i]
                        avg_histogram.append(s / self.nr_executions)
                color = colors[idx % 8]
            else:
                for i in range(0, self.nr_hypergenerations):
                    s = 0
                    for e in range(0, self.nr_executions):
                        s += results[str(e)][i]
                    avg_histogram.append(s / self.nr_executions)
                color = 'red'
            matplotlib.pyplot.plot(avg_histogram, linewidth=1, color=color,
                               label=heuristic.__class__.__name__.replace('HyperHeuristic', '').replace('Heuristic', ''))

        matplotlib.pyplot.title(self.problem['name'])
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()

    def plot_box(self, verbose=True):
        data = []
        if verbose is True:
            for heuristic in self.heuristics:
                with open(self.problem[
                              'objective_func'].__name__ + '_' + heuristic.__class__.__name__ + '_histogram.json') as json_file:
                    results = json.load(json_file)
                    best = []
                for key in results:
                    best.append(results[key][-1])
                data.append(best)
        else:
            with open(self.problem["objective_func"].__name__) as json_file:
                results = json.load(json_file)
            for heuristic in self.heuristics:
                data.append(results[heuristic.__class__.__name__])

        fig = matplotlib.pyplot.figure()
        fig1 = fig.add_subplot(1, 1, 1)
        fig1.boxplot(data, showfliers=False)

        indexes = [idx for idx in range(1, len(self.heuristics) + 1)]
        names = [heuristic.__class__.__name__.replace('HyperHeuristic', '').replace('Heuristic', '') for heuristic in
                 self.heuristics]

        plt.xticks(indexes, names)
        # show plot
        plt.title(self.problem["name"])
        plt.show()

    def assign_rank(self, values, global_value):
        final_rank = 0
        denominator = abs(global_value) if global_value != 0 else 1
        for value in values:
            rank = (value - global_value) / denominator
            final_rank += rank
        return final_rank / len(values)

    def assign_median_rank(self, values, global_value):
        ranks = []
        denominator = abs(global_value) if global_value != 0 else 1
        for value in values:
            rank = (value - global_value) / denominator
            ranks.append(rank)
        ranks.sort()
        return ranks[(len(ranks) + 1)//2]