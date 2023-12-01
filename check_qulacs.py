from pyscf import gto, mcscf, scf

from chemqulacs.util import utils
from chemqulacs.vqe import vqemcscf
from chemqulacs.vqe.vqeci import Ansatz
from quri_parts.algo.optimizer import Adam
from chemqulacs.vqe.vqeci import QulacsBackend

def test_vqecasci():
    print("=== START ===")
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto3g")
    # mol = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz'), ncas = 6, nelecs = 8
    mf = scf.RHF(mol)
    mf.run()
    mc_vqe = vqemcscf.VQECASCI(
        mf,
        ncas=6,
        nelecas=4,
        optimizer=Adam(ftol=1e-3),
        # backend=ITensorBackend(),  # ここを QulacsBackend に変更しても良い
        backend=QulacsBackend(),  # ここを QulacsBackend に変更しても良い
        ansatz=Ansatz.HardwareEfficient,
        layers=4,
        is_init_random=False,
        seed=10,
    )
    return mc_vqe.e_tot


def test_vqecasci_npartititions():
    print("=== START ===")
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto3g")
    # mol = gto.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis = 'ccpvdz'), ncas = 6, nelecs = 8
    mf = scf.RHF(mol)
    mf.run()
    mc_vqe = vqemcscf.VQECASCI(
        mf,
        ncas=6,
        nelecas=4,
        optimizer=Adam(ftol=1e-3),
        # backend=ITensorBackend(),  # ここを QulacsBackend に変更しても良い
        backend=QulacsBackend(),  # ここを QulacsBackend に変更しても良い
        ansatz=Ansatz.HardwareEfficient,
        layers=4,
        is_init_random=False,
        seed=10,
        npartitions=4
    )
    return mc_vqe.e_tot

if __name__ == "__main__":
    a = test_vqecasci()
    b = test_vqecasci_npartititions()
    assert utils.almost_equal(a, b)
    print("OK")