import abc


class Metric(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute(self, ensemble_particles_history, maximize=True):
        pass
