import numpy as np
import matplotlib.pyplot as plt

τ = 0.862  # time correlation coefficient
t = 3  # number of frs wrong
zeta = 2  # path loss exponent
a = [
    0.7,  # power for user u
    0.3  # power for user v
]
d = [
    2,  # distance of user u
    1  # distance of user v
]

expectedRates = np.linspace(0, 2, 1000)
PoutTheoreticalNOMA = [[], []]
PoutTheoreticalOMA = [[], []]

fig, ax = plt.subplots()  # initializes plotting

for rate in expectedRates:
    λ = 1 / (1 + np.power(d, zeta))
    r = np.power(2, rate) - 1
    φ = np.max([r / (a[0] - a[1] * r), r / a[1]])
    SNR = 35
    σ = 10 ** (- SNR / 20)
    ρ = 1 / σ ** 2

    PoutTheoreticalNOMA[0].append(
        1 - np.exp(- 2 * r * σ ** 2 / (2 - τ ** (2 * t)) / (a[0] - a[1] * r) / λ[0]),
    )
    PoutTheoreticalNOMA[1].append(
        1 - 2 * np.exp(- φ * σ ** 2 / λ[1]) + np.exp(- 2 * φ * σ ** 2 / (2 - τ ** (2 * t)) / λ[1])
    )
    PoutTheoreticalOMA[0].append(
        1 - np.exp(- 2 * (2 ** (2 * rate) - 1) / ((2 - τ ** (2 * t)) * ρ * λ[0])),
    )
    PoutTheoreticalOMA[1].append(
        1 - 2 * np.exp(- (2 ** (2 * rate) - 1) / ρ / λ[1]) + np.exp(
            - 2 * (2 ** (2 * rate) - 1) / ((2 - τ ** (2 * t)) * ρ * λ[1]))
    )

ax.semilogy(expectedRates, np.clip(PoutTheoreticalNOMA[0], 0, 1), label='NOMA, User $u$, τ = {}'.format(τ))
ax.semilogy(expectedRates, np.clip(PoutTheoreticalNOMA[1], 0, 1), label='NOMA, User $v$, τ = {}'.format(τ))
ax.semilogy(expectedRates, np.clip(PoutTheoreticalOMA[0], 0, 1), '--', label='OMA, User $u$, τ = {}'.format(τ))
ax.semilogy(expectedRates, np.clip(PoutTheoreticalOMA[1], 0, 1), '--', label='OMA, User $v$, τ = {}'.format(τ))

ax.grid()
ax.legend()
plt.xlabel('Target rate (bps/Hz)')
plt.ylabel('Outage Probability')

plt.show()
