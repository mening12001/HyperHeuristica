import numpy as np
import math

# Global variable
directivity = 0


def antennafunccircular(x1, null, phi_desired, distance):
    global directivity

    pi = math.pi
    dim = len(x1)
    y = 0
    temp, num_null = null.shape
    num1 = 300
    phi = np.linspace(0, 360, num1)
    phizero = 0
    yax = np.zeros(num1)

    # Calculate array factor values at each angle phi
    for i in range(num1):
        yax[i] = array_factorcir(x1, (pi / 180) * phi[i], phi_desired, distance, dim)

    maxi = np.max(yax)
    phi_ref = np.argmax(yax)

    # Calculate maximum gain and corresponding angle
    phizero = phi[phi_ref]

    sidelobes = []
    sllphi = []
    count = 0

    # Check for sidelobes in the array factor values
    if yax[0] > yax[num1 - 1] and yax[0] > yax[1]:
        count += 1
        sidelobes.append(yax[0])
        sllphi.append(phi[0])

    if yax[num1 - 1] > yax[0] and yax[num1 - 1] > yax[num1 - 2]:
        count += 1
        sidelobes.append(yax[num1 - 1])
        sllphi.append(phi[num1 - 1])

    for i in range(1, num1 - 1):
        if yax[i] > yax[i + 1] and yax[i] > yax[i - 1]:
            count += 1
            sidelobes.append(yax[i])
            sllphi.append(phi[i])
    print(sidelobes)
    # Sort sidelobe levels in descending order
    sidelobes = np.sort(sidelobes)[::-1]

    upper_bound = 180
    lower_bound = 180
    y = sidelobes[1] / maxi
    sllreturn = 20 * np.log10(y)

    # Calculate upper and lower bounds for beamwidth
    for i in range(1, int(num1 / 2)):
        if phi_ref + i > num1 - 1:
            upper_bound = 180
            break
        tem = yax[phi_ref + i]
        if yax[phi_ref + i] < yax[phi_ref + i - 1] and yax[phi_ref + i] < yax[phi_ref + i + 1]:
            upper_bound = phi[phi_ref + i] - phi[phi_ref]
            break

    for i in range(1, int(num1 / 2)):
        if phi_ref - i < 2:
            lower_bound = 180
            break
        tem = yax[phi_ref - i]
        if yax[phi_ref - i] < yax[phi_ref - i - 1] and yax[phi_ref - i] < yax[phi_ref - i + 1]:
            lower_bound = phi[phi_ref] - phi[phi_ref - i]
            break

    bwfn = upper_bound + lower_bound
    y1 = 0

    # Calculate objective function for null control
    for i in range(num_null):
        y1 += array_factorcir(x1, null[0][i], phi_desired, distance, dim) / maxi

    y2 = 0
    uavg = trapezoidalcir(x1, 0, 2 * pi, 50, phi_desired, distance, dim)
    y2 = abs(2 * pi * maxi * maxi / uavg)
    directivity = 10 * np.log10(y2)
    y3 = abs(phizero - phi_desired)

    if y3 < 5:
        y3 = 0

    y = 0

    if bwfn > 80:
        y += abs(bwfn - 80)

    y = sllreturn + y + y1 + y3

    return y, sllreturn, bwfn


def array_factorcir(x1, phi, phi_desired, distance, dim):
    pi = math.pi
    y = 0
    y1 = 0

    for i1 in range(int(dim / 2)):
        delphi = 2 * pi * i1 / dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired * pi / 180 - delphi)
        shi = shi * dim * distance
        y += x1[i1] * np.cos(shi + x1[int(dim / 2) + i1] * pi / 180)

    for i1 in range(int(dim / 2), dim):
        delphi = 2 * pi * i1 / dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired * pi / 180 - delphi)
        shi = shi * dim * distance
        y += x1[i1 - int(dim / 2)] * np.cos(shi - x1[i1] * pi / 180)

    for i1 in range(int(dim / 2)):
        delphi = 2 * pi * i1 / dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired * pi / 180 - delphi)
        shi = shi * dim * distance
        y1 += x1[i1] * np.sin(shi + x1[int(dim / 2) + i1] * pi / 180)

    for i1 in range(int(dim / 2), dim):
        delphi = 2 * pi * i1 / dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired * pi / 180 - delphi)
        shi = shi * dim * distance
        y1 += x1[i1 - int(dim / 2)] * np.sin(shi - x1[i1] * pi / 180)

    y = y * y + y1 * y1
    y = np.sqrt(y)

    return y


def trapezoidalcir(x2, upper, lower, N1, phi_desired, distance, dim):
    pi = math.pi
    h = (upper - lower) / N1
    x1 = lower
    y = np.abs(np.real(np.power(array_factorcir(x2, lower, phi_desired, distance, dim), 2) * np.sin(lower - pi / 2)))

    for i in range(2, N1 + 1):
        x1 += h
        y = np.append(y, np.abs(
            np.real(np.power(array_factorcir(x2, x1, phi_desired, distance, dim), 2) * np.sin(x1 - pi / 2))))

    s = 0

    for i in range(1, N1 + 1):
        if i == 1 or i == N1 + 1:
            s += y[i]
        else:
            s += 2 * y[i]

            q = (h / 2) * s

            return q