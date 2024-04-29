#from enoppy.paper_based.ihaoavoa_2022 import TensionCompressionSpringProblem

from evaluation.functions.ihaoavoa_2022 import TensionCompressionSpringProblem, WeldedBeamProblem, \
    CantileverBeamProblem, SpeedReducerProblem, RollingElementBearingProblem


def spring_func(solutions):
    func = TensionCompressionSpringProblem()
    result = []
    for sol in solutions:
        a = func.evaluate(sol)
        #result.append(func.evaluate(sol)[0])
        result.append(a[0])
    return result

def welded_func(solutions):
    func = WeldedBeamProblem()
    result = []
    for sol in solutions:
        a = func.evaluate(sol)[0]
        #result.append(func.evaluate(sol)[0])
        result.append(a)
    return result

def beam_func(solutions):
    func = CantileverBeamProblem()
    result = []
    for sol in solutions:
        a = func.evaluate(sol)[0]
        #result.append(func.evaluate(sol)[0])
        result.append(a)
    return result

def speed_func(solutions):
    func = SpeedReducerProblem()
    result = []
    for sol in solutions:
        a = func.evaluate(sol)
        #result.append(func.evaluate(sol)[0])
        result.append(a[0])
    return result

def rolling_func(solutions):
    func = RollingElementBearingProblem()
    result = []
    for sol in solutions:
        a = func.evaluate(sol)
        #result.append(func.evaluate(sol)[0])
        result.append(a[0])
    return result



def spring_actual(solution):
    func = TensionCompressionSpringProblem()
    a, b = func.evaluate(solution)
    return b[0]




