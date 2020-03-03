import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

tau = 1
expectedRates = [
    0.585,  # delay-critical user u
    2  # delay-tolerant user v
]
zeta = 2  # path loss exponent
au = 0.7  # power for user u
av = 0.3  # power for user v
d = [
    2,  # distance of user u
    1  # distance of user v
]

SNR = 10  # SNR in decibels
totalFrames = 100000
successfulFrames = np.array([0, 0])

for frame in tqdm(range(totalFrames)):
    # Step 1: Calculate g of channel for each user
    h = np.random.normal(0, 1, (2, 2)) @ [1, 1j]
    g = h / (1 + np.sqrt(np.power(d, zeta)))

    # Step 2: Calculate AWGN
    σ = 10 ** (- SNR / 20)
    n = np.random.normal(0, σ)

    # Step 3: Calculate reception SNR
    SNRvv = av * np.abs(g[1])**2 / σ**2
    SNRuu = au * np.abs(g[0])**2 / (au * np.abs(g[0])**2 + σ**2)

    # Step 4: Calculate reception rates
    receiverRates = [
        np.log2(1 + SNRuu),
        np.log2(1 + SNRvv)
    ]

    successfulFrames += np.greater_equal(receiverRates, expectedRates)

    # End the simulation early if we have found 1000 frames
    if np.all(np.greater_equal(frame - successfulFrames, 1000)):
        break

logging.debug('Completed after {} iterations'.format(frame))
logging.info('Packet success: {} / {}'.format(successfulFrames, totalFrames))
logging.info('Outage rate u: {:.2e}'.format(1 - successfulFrames[0] / totalFrames))
logging.info('Outage rate v: {:.2e}'.format(1 - successfulFrames[1] / totalFrames))

