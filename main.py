import logging
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

τ = 0.6  # time correlation coefficient
t = 3  # number of frs wrong
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

SNR = 10  # SNR in decibels
totalFrames = 20000
successfulFrames = np.array([0, 0])

# TODO: Use a convergence metric?

x = []
for frame in tqdm(range(totalFrames)):
    # Step 1: Calculate g of channel for each user
    h = np.random.normal(0, np.sqrt(2) / 2, (2, 2)) @ [1, 1j]
    g = np.divide(h, np.sqrt(1 + np.power(d, zeta)))

    # Sort channel gains
    if np.abs(h[0]) ** 2 > np.abs(h[1]) ** 2:
        h[1], h[0] = h[0], h[1]
        g = np.divide(h, np.sqrt(1 + np.power(d, zeta)))

    # Step 1.5: Apply time-correlated small fading coefficient
    ω = np.random.normal(0, np.sqrt(2) / 2, (2, 2)) @ [1, 1j]
    h = τ ** t * h + np.sqrt(1 - τ ** (2 * t)) * ω
    g = np.divide(h, np.sqrt(1 + np.power(d, zeta)))

    # Step 2: Calculate AWGN
    σ = 10 ** (- SNR / 20)
    n = np.random.normal(0, σ, 2)

    # Step 3: Calculate reception SNR
    # Note: We are using the actual noise value n, instead of just the SNR-based σ
    SNRuu = a[0] * np.abs(g[0])**2 / (a[1] * np.abs(g[0])**2 + σ**2)
    SNRvv = a[1] * np.abs(g[1])**2 / σ**2
    SNRvu = a[0] * np.abs(g[1])**2 / (a[1] * np.abs(g[1])**2 + σ**2)

    receiverRates = [
        np.log2(1 + SNRuu),
        np.log2(1 + SNRvv),
        np.log2(1 + SNRvu)
    ]

    successfulFrames += [
        receiverRates[0] >= expectedRates[0],
        receiverRates[1] >= expectedRates[1] and receiverRates[2] >= expectedRates[0]
    ]

    # End the simulation early if we have found 1000 frames
    #if np.all(np.greater_equal(frame - successfulFrames, 1000)):
    #    break

logging.debug('Completed after {} iterations'.format(frame))
logging.info('Packet success: {} / {}'.format(successfulFrames, totalFrames))
logging.info('Outage rate u: {:.2e}'.format(1 - successfulFrames[0] / totalFrames))
logging.info('Outage rate v: {:.2e}'.format(1 - successfulFrames[1] / totalFrames))

# Calculate theoretical values
σ = 10 ** (- SNR / 20)
ρ = 1 / σ ** 2
λ = 1 / (1 + np.power(d, zeta))
r = np.power(2, expectedRates) - 1
φ = np.max([r[0] / (a[0] - a[1] * r[0]), r[1] / a[1]])

# NOMA - Outdated CSI
PoutTheoretical = [
    1 - np.exp(- 2 * r[0] * σ ** 2 / (2 - τ ** (2 * t)) / (a[0] - a[1] * r[0]) / λ[0]),
    1 - 2 * np.exp(- φ * σ ** 2 / λ[1]) + np.exp(- 2 * φ * σ ** 2 / (2 - τ ** (2 * t)) / λ[1])
]

logging.info('Theor. Outage rate u: {:.2e}'.format(PoutTheoretical[0]))
logging.info('Theor. Outage rate v: {:.2e}'.format(PoutTheoretical[1]))
