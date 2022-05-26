"""
An Implementation of a Continuous Cellular Automaton known as "Smooth Life", which
was originally developed by Stephan Rafler.

The original research paper by Stephan Rafler: https://arxiv.org/pdf/1111.1567.pdf

This implementation was adapted from an opengl shaders implementation by user "chronos"
on shadertoy.com. https://www.shadertoy.com/view/XtdSDn#
"""

import numpy as np
import numexpr as ne
import imageio as io
from tqdm import trange


OUTPUT = 'smoothlife.mp4'
FPS = 30
W = 1920 // 2
H = 1080 // 2
GENERATIONS = FPS * 60 * 2

DT = 0.30
OUTER_RADIUS = 10
INNER_RADIUS = 3

B1 = 0.257
B2 = 0.336
D1 = 0.365
D2 = 0.549

ALPHA_N = 0.028
ALPHA_M = 0.147


def sigmoid(x, a, b, buffer):
    ne.evaluate('1 / (1 + exp(-(x - a) * 4 / b))', out=buffer)


def smoothlife(dx, dy, buff1, buff2, buff3):
    # Calculate the next generation of the cellular automaton.
    sigmoid(dx, B1, ALPHA_N, buff1)
    sigmoid(dx, B2, ALPHA_N, buff2)
    ne.evaluate('buff1 * (1 - buff2)', out=buff1)

    sigmoid(dx, D1, ALPHA_N, buff2)
    sigmoid(dx, D2, ALPHA_N, buff3)
    ne.evaluate('buff2 * (1 - buff3)', out=buff2)

    sigmoid(dy, 0.5, ALPHA_M, buff3)
    ne.evaluate('buff1 * (1 - buff3) + buff2 * buff3', out=buff3)


def create_kernels():
    # Create the kernels to be used in the convolution to calculate
    # the next generation of the cellular automaton. Done via fft convolution.
    kernelx = np.zeros((H, W), dtype=np.complex128)
    kernely = np.zeros((H, W), dtype=np.complex128)
    for y in range(-OUTER_RADIUS, OUTER_RADIUS + 1):
        for x in range(-OUTER_RADIUS, OUTER_RADIUS + 1):
            dist = np.sqrt(x * x + y * y)
            inner_dist = np.clip(dist - INNER_RADIUS + 0.5, 0, 1)
            kernelx[y % H, x % W] = inner_dist * (1 - np.clip(dist - OUTER_RADIUS + 0.5, 0, 1))
            kernely[y % H, x % W] = 1 - inner_dist

    return np.fft.fft2(kernelx), np.fft.fft2(kernely)


def main():
    outer_area = np.pi * OUTER_RADIUS * OUTER_RADIUS
    inner_area = np.pi * INNER_RADIUS * INNER_RADIUS
    outer_area -= inner_area

    np.random.seed(None)
    current = np.random.rand(H, W).astype(np.complex128)
    next = np.empty((H, W), dtype=np.complex128)

    kernel_cx, kernel_cy = create_kernels()

    buffer1 = np.empty((H, W), dtype=np.float64)
    buffer2 = np.empty((H, W), dtype=np.float64)
    buffer3 = np.empty((H, W), dtype=np.float64)

    cx = np.empty_like(current)
    cy = np.empty_like(current)

    with io.save(OUTPUT, fps=FPS) as writer:
        frame_buffer = np.empty((H, W), dtype=np.uint8)
        for _ in trange(GENERATIONS):
            current_fft = np.fft.fft2(current)

            ne.evaluate('current_fft * kernel_cx', out=cx)
            ne.evaluate('current_fft * kernel_cy', out=cy)

            cxr = np.fft.ifft2(cx).real
            cyr = np.fft.ifft2(cy).real

            ne.evaluate('cxr / outer_area', out=cxr)
            ne.evaluate('cyr / inner_area', out=cyr)

            dt = DT
            smoothlife(cxr, cyr, buffer1, buffer2, buffer3)
            ne.evaluate('current + dt * (2 * buffer3 - 1)', out=next)
            np.clip(next, 0, 1, out=next)

            current, next = next, current

            ne.evaluate('current.real * 255', out=frame_buffer, casting='unsafe')

            writer.append_data(frame_buffer)


if __name__ == '__main__':
    main()
