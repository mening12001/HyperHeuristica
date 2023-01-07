

class Exchanger:

    def __init__(self, set_size, nr_iterations):
        self.set_size = set_size
        self.solutions = {k: [] for k in range(nr_iterations)}
        self.flag = [False for _ in range(nr_iterations)]

    async def collect(self, solution, iteration):
        self.solutions[iteration].append(solution)
        if len(self.solutions[iteration]) == self.set_size:
            self.flag[iteration] = True

    async def obtain(self, iteration):
        while not self.flag[iteration]:
            pass
        return self.solutions[iteration]

    def refresh(self):
        self.solutions = {}
