class ActualSolution:

    value = 0
    velocity = 0
    position = ()
    pbest_cost = 0
    id = 0
    def __init__(self, id, value, velocity, position, pbest_cost, pbest_pos):
        self.value = value
        self.id = id
        self.velocity = velocity
        self.position = position
        self.pbest_cost = pbest_cost
        self.pbest_pos = pbest_pos

    def show(self):
        print(self.id)
        print(self.value)
        print(self.velocity)
        print(self.position)
        print(self.pbest_cost)