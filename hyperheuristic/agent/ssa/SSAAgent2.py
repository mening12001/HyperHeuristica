from hyperheuristic.agent.ssa.SSAAgent import SSAAgent


class SSAAgent2(SSAAgent):
    solutions = None
    solutions_state = None
    window_iteration = None

    def solve(self, mode='sequential'):
        self.solutions = SSAAgent.solve(self)
        self.solutions_state = self.solutions[self.epoch - 1]
        self.window_iteration = self.epoch
        return self.solutions

    def extend(self):
        SSAAgent.init_initial_positions(self, self.solutions_state)
        self.epoch = 1
        self.solutions_state = SSAAgent.solve(self)[0]
        self.window_iteration = self.window_iteration + 1
        self.solutions[self.window_iteration - 1] = self.solutions_state
        return self.solutions

