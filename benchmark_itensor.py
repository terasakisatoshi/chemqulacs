import concurrent
from multiprocessing import get_context
import sys
import time
from importlib import metadata

from pyscf import gto, scf
from quri_parts.algo.optimizer import Adam
from quri_parts.itensor.load_itensor import ensure_itensor_loaded

from chemqulacs.vqe import vqemcscf
from chemqulacs.vqe.vqeci import Ansatz, ITensorBackend, QulacsBackend
import numpy as np
from matplotlib import pyplot as plt

ensure_itensor_loaded()  # ITensorBackend を使う場合はグローバル空間でこの関数を実行する必要がある．


def main(npartitions=1, executor=concurrent.futures.ProcessPoolExecutor()):
    print("=== START ===")
    """
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto3g")
    ncas = 3
    nelecs = 4
    """

    # https://github.com/pyscf/pyscf.github.io/blob/master/examples/mcscf/00-simple_casci.py
    mol = gto.M(atom="O 0 0 0; O 0 0 1.2", basis="ccpvdz", spin=2)
    ncas = 6
    nelecs = 8

    mf = scf.RHF(mol)
    mf.run()

    mc_vqe = vqemcscf.VQECASCI(
        mf,
        ncas=ncas,
        nelecas=nelecs,
        optimizer=Adam(ftol=1e-3),
        backend=ITensorBackend(),  # ここを QulacsBackend に変更しても良い
        # backend=QulacsBackend(),  # ここを ITensorBackend に変更しても良い
        ansatz=Ansatz.HardwareEfficient,
        layers=4,
        is_init_random=False,
        seed=10,
        npartitions=npartitions,
        executor=executor,
    )

    t1 = time.time()
    print(f"{mc_vqe.run()}")
    t2 = time.time()
    elapsed = t2 - t1
    print(f"{npartitions=} partitions")
    print(f"elapsed time = {elapsed}", "[sec]")
    print("=== END ===")
    print()
    return elapsed


if __name__ == "__main__":
    print(sys.version)
    assert metadata.version("quri_parts_itensor") == "0.15.1"
    with concurrent.futures.ProcessPoolExecutor(mp_context=get_context("spawn")) as executor:
        ttfx = {}
        etimes = {i: [] for i in [None, 1, 2, 4, 6]}
        for npartitions in [None, 1, 2, 4, 6]:
            # 初回実行のオーバヘッドを避ける
            ttfx[npartitions] = main(npartitions=npartitions, executor=executor)
            for i in range(5):
                etimes[npartitions].append(
                    main(npartitions=npartitions, executor=executor)
                )

    s = [ttfx[k] for k in [None, 1, 2, 4, 6]]
    fig, ax = plt.subplots()
    ax.plot(s)
    fig.savefig("ttfx_itensor.png")

    xs = []
    ys = []
    for npartitions in [None, 1, 2, 4, 6]:
        x = npartitions
        if npartitions is None:
            x = 0
        y = np.mean(etimes[npartitions])
        xs.append(x)
        ys.append(y)

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    fig.savefig("benchmark_itensor.png")
