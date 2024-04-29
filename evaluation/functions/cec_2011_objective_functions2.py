import numpy as np
import math
from scipy.integrate import trapz

def antennafunccircular(x1, null, phi_desired, distance):
    pi = 3.141592654
    dim = len(x1)
    y = 0

    num_null = len(null)
    num1 = 300
    phi = np.linspace(0, 360, num1)
    phizero = 0
    yax = np.zeros(num1)
    yax[0] = array_factorcir(x1, (pi / 180) * phi[0], phi_desired, distance, dim)
    maxi = yax[0]
    phi_ref = 0

    for i in range(1, num1):
        yax[i] = array_factorcir(x1, (pi / 180) * phi[i], phi_desired, distance, dim)
        if maxi < yax[i]:
            maxi = yax[i]
            phizero = phi[i]
            phi_ref = i

    maxtem = 0
    count = 0
    sidelobes = np.zeros(num1)
    sllphi = np.zeros(num1)

    if yax[0] > yax[num1 - 1] and yax[0] > yax[1]:
        count += 1
        sidelobes[count] = yax[0]
        sllphi[count] = phi[0]

    if yax[num1 - 1] > yax[0] and yax[num1 - 1] > yax[num1 - 2]:
        count += 1
        sidelobes[count] = yax[num1 - 1]
        sllphi[count] = phi[num1 - 1]

    for i in range(1, num1 - 1):
        if yax[i] > yax[i + 1] and yax[i] > yax[i - 1]:
            count += 1
            sidelobes[count] = yax[i]
            sllphi[count] = phi[i]

    sidelobes = np.sort(sidelobes)[::-1]
    upper_bound = 180
    lower_bound = 180
    y = sidelobes[1] / maxi
    sllreturn = 20 * np.log10(y)

    for i in range(1, num1 // 2):
        if phi_ref + i > num1 - 1:
            upper_bound = 180
            break
        tem = yax[phi_ref + i]
        if yax[phi_ref + i] < yax[phi_ref + i - 1] and yax[phi_ref + i] < yax[phi_ref + i + 1]:
            upper_bound = phi[phi_ref + i] - phi[phi_ref]
            break

    for i in range(1, num1 // 2):
        if phi_ref - i < 2:
            lower_bound = 180
            break
        tem = yax[phi_ref - i]
        if yax[phi_ref - i] < yax[phi_ref - i - 1] and yax[phi_ref - i] < yax[phi_ref - i + 1]:
            lower_bound = phi[phi_ref] - phi[phi_ref - i]
            break

    bwfn = upper_bound + lower_bound
    y1 = 0

    for i in range(num_null):
        y1 += array_factorcir(x1, null[i], phi_desired, distance, dim) / maxi

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

import numpy as np

def array_factorcir(x1, phi, phi_desired, distance, dim):
    pi = 3.141592654
    y = 0
    y1 = 0

    # Convert single values to arrays
    phi = np.atleast_1d(phi)
    delphi = 2 * pi * np.arange(dim // 2) / dim
    phi_desired_array = np.full_like(phi, phi_desired * (pi / 180))

    # Compute array factors for each element of phi
    for phi_val in phi:
        shi = np.cos(phi_val - delphi) - np.cos(phi_desired_array - delphi)
        shi *= dim * distance

        y += np.sum(x1[:dim // 2] * np.cos(shi + x1[dim // 2:] * (pi / 180)))
        y1 += np.sum(x1[:dim // 2] * np.sin(shi + x1[dim // 2:] * (pi / 180)))

    y = y * y + y1 * y1
    y = np.sqrt(y)

    return y

def trapezoidalcir(x2, upper, lower, N1, phi_desired, distance, dim):
    pi = 3.141592654
    h = (upper - lower) / N1
    x1 = lower
    y = np.zeros(N1 + 1)

    for i in range(1, N1 + 2):
        x1 += h
        y[i - 1] = abs(math.pow(array_factorcir(x2, x1, phi_desired, distance, dim), 2) * math.sin(x1 - pi / 2))

    s = trapz(y, dx=h)
    q = (h / 2) * s

    return q
