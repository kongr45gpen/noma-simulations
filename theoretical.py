import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

τ = 1  # time correlation coefficient
t = 1  # number of frs wrong
expectedRates = [
    0.585,  # delay-critical user u
    2  # delay-tolerant user v
]
zeta = 2  # path loss exponent
a = [
    0.7,  # power for user u
    0.3  # power for user v
]
d = [
    2,  # distance of user u
    1  # distance of user v
]

# Calculate theoretical values
SNR = np.linspace(0, 40, 100)
#SNR = 10
σ = 10 ** (- SNR / 20)
ρ = 1 / σ ** 2
λ = 1 / (1 + np.power(d, zeta))
r = np.power(2, expectedRates) - 1
φ = np.max([expectedRates[0] / (a[0] - a[1] * r[0]), r[1] / a[1]])

# OMA
#PoutTheoretical = [
#    1 - np.exp(- 2 * (2 ** (2 * expectedRates[0]) - 1) / ((2 - τ ** (2 * t)) * ρ * λ[0])),
#    1 - 2 * np.exp(- (2 ** (2 * expectedRates[1]) - 1) / ρ / λ[1]) + np.exp(- 2 * (2 ** (2 * expectedRates[1]) - 1) / ((2 - τ ** (2 * t)) * ρ * λ[1]))
#]

# NOMA - Outdated CSI
PoutTheoretical = [
    1 - np.exp(- 2 * r[0] * σ ** 2 / (2 - τ ** (2 * t)) / (a[0] - a[1] * r[0]) / λ[0]),
    1 - 2 * np.exp(- φ * σ ** 2 / λ[1]) + np.exp(- 2 * φ * σ ** 2 / (2 - τ ** (2 * t)) / λ[1])
]

# NOMA - Statistic CSI
#PoutTheoretical = [
#    1 - np.exp(- φ * σ ** 2 / λ[0]),
#    1 - np.exp(- r[0] * σ ** 2 / λ[1])
#]

print(PoutTheoretical)


# Draw the plots
fig, ax = plt.subplots()  # initializes plotting

# Plot the two functions in a logarithmic fashion
ax.semilogy(SNR, PoutTheoretical[0], label='User $u$')
ax.semilogy(SNR, PoutTheoretical[1], label='User $v$')

# Set up grid, legend, axes
ax.grid()
ax.legend()
plt.xlabel('SNR (dB)')
plt.ylabel('Outage Probability')

plt.show()
