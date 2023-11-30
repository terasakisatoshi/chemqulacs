import sys
from importlib import metadata
import time

from pyscf import gto, scf
from chemqulacs.vqe.vqeci import Ansatz
from chemqulacs.vqe import vqemcscf
from quri_parts.algo.optimizer import Adam
#from chemqulacs.vqe.vqeci import QulacsBackend

from quri_parts.itensor.load_itensor import ensure_itensor_loaded
from chemqulacs.vqe.vqeci import ITensorBackend
ensure_itensor_loaded() # ITensorBackend を使う場合はグローバル空間でこの関数を実行する必要がある．

def main(npartitions=1):
    print("=== START ===")
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto3g")
    #mol = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz'), ncas = 6, nelecs = 8
    mf = scf.RHF(mol)
    mf.run()
    mc_vqe = vqemcscf.VQECASCI(
        mf,
        ncas=3,
        nelecas=4,
        optimizer=Adam(ftol=1e-3),
        backend=ITensorBackend(), # ここを QulacsBackend に変更しても良い
        ansatz=Ansatz.HardwareEfficient,
        layers=4,
        is_init_random=False,
        seed=10,
        npartitions=npartitions,
    )

    t1 = time.time()
    print(mc_vqe.run())
    t2 = time.time()
    elapsed = t2 - t1
    print(f"{npartitions=} partitions")
    print(f"elapsed time = {elapsed}", "[sec]")
    print("=== END ===")
    print()


if __name__ == "__main__":
    print(sys.version)
    assert metadata.version("quri_parts_itensor") == "0.15.1"
    #main(npartitions=None)
    #main(npartitions=None)

    #main(npartitions=1)
    #main(npartitions=1)
    main(npartitions=2)
    main(npartitions=2)
    main(npartitions=4)
    main(npartitions=4)
    main(npartitions=8)
