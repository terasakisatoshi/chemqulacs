import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def parametric_fn(f):
    def estimator(o, s, p):
        t = s + p
        return f(o, t)

    return estimator


def add(x, y):
    return x + y


def wrapper(f, o_s_p_triplet):
    o, s, p = o_s_p_triplet
    estimator = parametric_fn(f)
    return estimator(o, s, p)


def main() -> None:
    assert parametric_fn(add)(1, 2, 3) == 1 + 2 + 3 == 6
    assert wrapper(add, (1, 2, 3)) == 6

    parametric_estimator = partial(wrapper, add)
    assert parametric_estimator((1, 2, 3)) == 1 + 2 + 3 == 6

    samples = [
        (random.random(), random.random(), random.random()) for _ in range(10000)
    ]
    expected = [parametric_estimator((x, y, z)) for (x, y, z) in samples]

    with ProcessPoolExecutor() as executor:
        for (r, e) in zip(executor.map(parametric_estimator, samples), expected):
            assert r == e
    print("Done")


if __name__ == "__main__":
    main()
