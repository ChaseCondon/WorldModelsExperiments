import abc

class Agent(abc.ABC):

    @abc.abstractmethod
    def learn(self, state, next_state, ):
        pass

    @abc.abstractmethod
    def get_action(self, state):
        pass