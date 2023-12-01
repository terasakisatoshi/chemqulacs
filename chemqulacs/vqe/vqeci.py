# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type:ignore

import importlib
import concurrent.futures
from multiprocessing import get_context
from enum import Enum, auto
from itertools import product
from typing import Any, Mapping, Optional, Sequence, List

import numpy as np
from braket.aws import AwsDevice
from openfermion.ops import FermionOperator, InteractionOperator
from quri_parts.core.operator import Operator
from quri_parts.core.utils.concurrent import execute_concurrently
from openfermion.transforms import get_fermion_operator
from pyscf import ao2mo
from qiskit import IBMQ
from quri_parts.algo.ansatz import HardwareEfficient, SymmetryPreserving
from quri_parts.algo.optimizer import Adam, OptimizerStatus
from quri_parts.braket.backend import BraketSamplingBackend
from quri_parts.chem.ansatz import (
    AllSinglesDoubles,
    GateFabric,
    ParticleConservingU1,
    ParticleConservingU2,
)
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
)
from quri_parts.core.estimator.gradient import (
    create_parameter_shift_gradient_estimator,
)
from quri_parts.core.estimator.sampling import create_sampling_estimator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling import (
    create_concurrent_sampler_from_sampling_backend,
)
from quri_parts.core.sampling.shots_allocator import (
    create_weighted_random_shots_allocator,
)
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
)
from quri_parts.openfermion.ansatz import KUpCCGSD, TrotterUCCSD
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapping,
    jordan_wigner,
)
from quri_parts.qiskit.backend import QiskitSamplingBackend
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)

from chemqulacs.vqe.rdm import get_1rdm, get_2rdm

class Backend:
    """Base class represents backend"""

    pass


class QulacsBackend(Backend):
    """Backend class for Qulacs"""

    pass


class ITensorBackend(Backend):
    """Backend class for ITensor"""

    pass



class AWSBackend(Backend):
    """
    Backend class for AWS Braket

    Args:
        arn (str):
            ARN which specifies the device to use
        qubit_mapping (Optional[Mapping[[int, int]]]):
            Qubit mappings between the qubit indices used in the code and the qubit indices
            used in the backend.
    """

    def __init__(
        self, arn: str, qubit_mapping: Optional[Mapping[int, int]] = None, **run_kwargs
    ):
        device = AwsDevice(arn)
        sampling_backend = BraketSamplingBackend(
            device, qubit_mapping=qubit_mapping, **run_kwargs
        )
        self.sampler = create_concurrent_sampler_from_sampling_backend(sampling_backend)


class QiskitBackend(Backend):
    """
    Backend class for Qiskit

    Args:
        backend_name (str):
            Name of the qiskit backend
        hub (str)
        group (str)
        project (str)
        qubit_mapping (Optional[Mapping[[int, int]]]):
            Qubit mappings from the qubit indices used in the code to the backend qubit indices
    """

    def __init__(
        self,
        backend_name: str,
        hub: str = "ibm-q",
        group: str = "open",
        project: str = "main",
        qubit_mapping: Optional[Mapping[int, int]] = None,
        **run_kwargs,
    ):
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub, group, project)
        device = provider.get_backend(backend_name)
        sampling_backend = QiskitSamplingBackend(
            device, qubit_mapping=qubit_mapping, **run_kwargs
        )
        self.sampler = create_concurrent_sampler_from_sampling_backend(sampling_backend)


class Ansatz(Enum):
    """An enum representing an ansatz for VQE"""

    HardwareEfficient = auto()
    SymmetryPreserving = auto()
    AllSinglesDoubles = auto()
    ParticleConservingU1 = auto()
    ParticleConservingU2 = auto()
    GateFabric = auto()
    UCCSD = auto()
    KUpCCGSD = auto()


def _get_active_hamiltonian(h1, h2, norb, ecore):
    n_orbitals = norb
    n_spin_orbitals = 2 * n_orbitals

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_spin_orbitals, n_spin_orbitals))
    two_body_coefficients = np.zeros(
        (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals)
    )
    # Set MO one and two electron-integrals
    # according to OpenFermion conventions
    one_body_integrals = h1
    h2_ = ao2mo.restore(
        1, h2.copy(), n_orbitals
    )  # no permutation see two_body_integrals of _pyscf_molecular_data.py
    two_body_integrals = np.asarray(h2_.transpose(0, 2, 3, 1), order="C")

    # Taken from OpenFermion
    # Loop through integrals.
    for p in range(n_spin_orbitals // 2):
        for q in range(n_spin_orbitals // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_spin_orbitals // 2):
                for s in range(n_spin_orbitals // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    two_body_coefficients[
                        2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                    ] = (two_body_integrals[p, q, r, s] / 2.0)

    # Get Hamiltonian in OpenFermion format
    active_hamiltonian = InteractionOperator(
        ecore, one_body_coefficients, two_body_coefficients
    )
    return active_hamiltonian

def in_partition(sites: tuple, rank: int, npartitions: int) -> bool:
    """
    Determine whether a given set of 'sites' belongs to a specific rank or not.
    This Python implementation is borrowled from ITensorChemistry.jl package
    https://github.com/ITensor/ITensorParallel.jl/blob/main/src/partition/partition_sum_split.jl

    Args:
        sites (tuple): A tuple representing the sites to be checked.
        rank (int): The rank or index of the partition being considered.
        npartitions (int): The total number of partitions.

    Returns:
        bool: True if the 'sites' belong to the specified partition, False otherwise.

    Raises:
        ValueError: If the input 'npartitions' is not a positive integer.
        ValueError: If the input 'sites' are not in a valid format.

    Note:
        The function checks the 'sites' tuple based on its length:
        - If len(sites) == 0, it handles the case of a constant term.
        - If len(sites) == 2, it checks the sites for a pair of factors.
        - If len(sites) == 4, it checks the sites for a quadruplet of factors.

        The criteria for partitioning are applied according to the OpenFermion library's conventions.

    Example:
        To check if a pair of factors ((0, 1), (2, 0)) belongs to partition 1 out of 3:
        >>> rank = 1
        >>> npartitions = 3
        >>> is_in_partition = in_partition(((0, 1), (2, 0)), rank, npartitions)
    """
    if npartitions < 1:
        raise ValueError("npartitions must be a positive integer")
    if len(sites) == 0:
        # constant term
        return rank % npartitions == 0
    if len(sites) == 2:
        if isinstance(sites[0], int):
            # https://github.com/quantumlib/OpenFermion/blob/85e533edb234fdccc3433b03c9b3d723f9a2a88c/src/openfermion/ops/operators/symbolic_operator.py#L231-L232
            # `sites` should be a single factor e.g., (0, 0)
            (i, _) = sites
            return rank == i % npartitions
        else:
            # `sites` should be a pair of factors e.g., ((0, 0), (3, 1))
            (i, _), (j, _) = sites
            # Make sure `i ≥ j`
            if j > i:
                i, j = j, i
            assert i >= j
            return rank == ((i - 1) * i // 2 + j) % npartitions
    if len(sites) == 4:
        # `sites` should be a quadruplet of factors e.g., ((0, 1), (0, 1), (2, 0), (2, 0))
        (i, _), (j, _), (k, _), (l, _) = sites
        i, j, k, l = sorted((i, j, k, l))
        assert i <= j <= k <= l
        dummy = 0
        if j == k:
            sites = (j, dummy)
            return in_partition(sites, rank, npartitions)
        else:
            sites = ((i, dummy), (j, dummy))
            return in_partition(sites, rank, npartitions)
    raise ValueError(f"Invalid input {sites=}")


def partition(h: FermionOperator, npartitions: int) -> List[FermionOperator]:
    """
    Partition a FermionOperator into multiple FermionOperators based on a specified criteria
    proposed by Zhai and Chan in https://arxiv.org/abs/2103.09976.

    Args:
        h (FermionOperator): The input FermionOperator to be partitioned.
        npartitions (int): The number of partitions to create.

    Returns:
        List[FermionOperator]: A list of FermionOperator objects, where each object
            represents a partition of the input FermionOperator.

    Note:
        The partitioning is performed based on the `in_partition` function, which is expected
        to determine whether a term in the input FermionOperator belongs to a specific partition
        or not. The terms are divided among the partitions, and each partition is represented by
        a FermionOperator object in the returned list.

    Example:
        To partition a FermionOperator `h` into 3 partitions:
        >>> partitions = partition(h, 3)
    """
    fermion_operators: List[FermionOperator] = [
        FermionOperator() for _ in range(npartitions)
    ]
    for rank in range(npartitions):
        for sites in h.terms.keys():
            if in_partition(sites, rank, npartitions):
                c = h.terms[sites]
                fermion_operators[rank] += FermionOperator(sites, c)
    return fermion_operators

def _create_concurrent_estimators(
    backend: Backend, shots_per_iter: int
) -> tuple[
    ConcurrentQuantumEstimator[CircuitQuantumState],
    ConcurrentParametricQuantumEstimator[ParametricCircuitQuantumState],
]:
    if isinstance(backend, QulacsBackend):
        estimator = create_qulacs_vector_concurrent_estimator()
        parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
        return estimator, parametric_estimator
    elif isinstance(backend, ITensorBackend):
        from quri_parts.itensor.estimator import (
            create_itensor_mps_concurrent_estimator,
            create_itensor_mps_concurrent_parametric_estimator,
        )

        estimator = create_itensor_mps_concurrent_estimator()
        parametric_estimator = create_itensor_mps_concurrent_parametric_estimator()
        return estimator, parametric_estimator
    elif isinstance(backend, (AWSBackend, QiskitBackend)):
        allocator = create_weighted_random_shots_allocator()
        sampling_estimator = create_sampling_estimator(
            shots_per_iter,
            backend.sampler,
            bitwise_commuting_pauli_measurement,
            allocator,
        )

        def concurrent_estimator(
            operator: Sequence[Estimatable],
            states: Sequence[ParametricCircuitQuantumState],
        ) -> Sequence[Estimate]:
            if len(operator) == 1:
                return [sampling_estimator(operator[0], state) for state in states]
            elif len(states) == 1:
                return [sampling_estimator(op, states[0]) for op in operator]
            else:
                return [
                    sampling_estimator(op, state) for op, state in zip(operator, states)
                ]

        estimator = concurrent_estimator

        def concurrent_parametric_estimator(
            operator: Estimatable,
            state: ParametricCircuitQuantumState,
            params: Sequence[Sequence[float]],
        ) -> Sequence[Estimate[complex]]:
            states = [state.bind_parameters(param) for param in params]
            return estimator([operator], states)

        parametric_estimator = concurrent_parametric_estimator
        return estimator, parametric_estimator
    else:
        raise ValueError("Invalid backend")


def _create_ansatz(
    ansatz: Ansatz,
    fermion_qubit_mapping: OpenFermionQubitMapping,
    n_sorbs: int,
    n_electrons: int,
    layers: int,
    k: int,
    trotter_number: int,
    include_pi: bool,
    use_singles: bool,
    delta_sz: int,
    singlet_excitation: bool,
) -> LinearMappedUnboundParametricQuantumCircuit:
    n_qubits = fermion_qubit_mapping.n_qubits_required(n_sorbs)
    if ansatz == Ansatz.HardwareEfficient:
        return HardwareEfficient(n_qubits, layers)
    elif ansatz == Ansatz.SymmetryPreserving:
        return SymmetryPreserving(n_qubits, layers)
    elif ansatz == Ansatz.AllSinglesDoubles:
        return AllSinglesDoubles(n_qubits, n_electrons)
    elif ansatz == Ansatz.ParticleConservingU1:
        return ParticleConservingU1(n_qubits, layers)
    elif ansatz == Ansatz.ParticleConservingU2:
        return ParticleConservingU2(n_qubits, layers)
    elif ansatz == Ansatz.GateFabric:
        return GateFabric(n_qubits, layers, include_pi)
    elif ansatz == Ansatz.UCCSD:
        return TrotterUCCSD(
            n_sorbs,
            n_electrons,
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
            delta_sz,
            singlet_excitation,
        )
    elif ansatz == Ansatz.KUpCCGSD:
        return KUpCCGSD(
            n_sorbs,
            n_electrons,
            k,
            fermion_qubit_mapping,
            trotter_number,
            delta_sz,
            singlet_excitation,
        )


def vqe(init_params, cost_fn, grad_fn, optimizer):
    opt_state = optimizer.get_init_state(init_params)
    while True:
        opt_state = optimizer.step(opt_state, cost_fn, grad_fn)
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state


# ======================
class VQECI(object):
    """
    VQECI
        Args:
            mol:
                SCF or Mole to define the problem size.
            fermion_qubit_mapping (quri_parts.openfermion.transforms.OpenFermionQubitMapping):
                Mapping from :class:`FermionOperator` or :class:`InteractionOperator` to :class:`Operator`
            optimizer
            backend (Backend)
            shots_per_iter (int)
            ansatz: ansatz used for VQE
            layers (int):
                Layers of gate operations. Used for ``HardwareEfficient``, ``SymmetryPreserving``, ``ParticleConservingU1``, ``ParticleConservingU2``, and ``GateFabric``.
            k (int):
                Number of repetitions of excitation gates. Used for ``KUpCCGSD``.
            trotter_number (int):
                Number of trotter decomposition. Used for ``UCCSD`` and ``kUpCCGSD``.
            include_pi (bool):
                If ``True``, the optional constant gate is inserted. Used for ``GateFabric``.
            use_singles: (bool):
                If ``True``, single-excitation gates are applied. Used for ``UCCSD``.
            delta_sz (int):
                Changes of spin in the excitation. Used for ``KUpCCGSD``.
            singlet_excitation (bool):
                If ``True``, the ansatz will be spin symmetric. Used for ``UCCSD`` and
                ``KUpCCGSD``.
            is_init_random (bool):
                If ``False``, initial parameters are initialized to 0s, else, initialized randomly.
            seeed (int):
                Random seed.
        Return:
            None
    """

    def __init__(
        self,
        mol=None,
        fermion_qubit_mapping=jordan_wigner,
        optimizer=Adam(),
        backend: Backend = QulacsBackend(),
        shots_per_iter: int = 10000,
        ansatz: Ansatz = Ansatz.ParticleConservingU1,
        layers: int = 2,
        k: int = 1,
        trotter_number: int = 1,
        include_pi: bool = False,
        use_singles: bool = True,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
        is_init_random: bool = False,
        seed: int = 0,
    ):
        self.mol = mol
        self.fermion_qubit_mapping = fermion_qubit_mapping
        self.opt_param = None  # to be used to store the optimal parameter for the VQE
        self.initial_state = None
        self.opt_state = None
        self.n_qubit: int = None
        self.n_orbitals: int = None
        self.ansatz: Ansatz = ansatz
        self.optimizer = optimizer
        self.backend: Backend = backend
        self.n_electron: int = None
        self.layers: int = layers
        self.k: int = k
        self.trotter_number: int = trotter_number
        self.include_pi: bool = include_pi
        self.use_singles: bool = use_singles
        self.delta_sz: int = delta_sz
        self.singlet_excitation: bool = singlet_excitation
        self.is_init_random: bool = is_init_random
        self.seed: int = seed
        self.e = 0

        self.estimator, self.parametric_estimator = _create_concurrent_estimators(
            self.backend, shots_per_iter
        )

    # =======================================================================================
    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        self.n_orbitals = norb
        self.n_qubit = self.fermion_qubit_mapping.n_qubits_required(2 * self.n_orbitals)
        self.n_electron = nelec[0] + nelec[1]

        # Get the active space Hamiltonian
        active_hamiltonian = _get_active_hamiltonian(h1, h2, norb, ecore)
        # Convert the Hamiltonian using `self.fermion_qubit_mapping`
        self.fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        op_mapper = self.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=2 * self.n_orbitals,
            n_fermions=self.n_electron,
        )
        qubit_hamiltonian = op_mapper(
            self.fermionic_hamiltonian,
        )
        # Set initial Quantum State
        occ_indices = list(range(self.n_electron))
        state_mapper = self.fermion_qubit_mapping.get_state_mapper(
            2 * self.n_orbitals, self.n_electron
        )
        self.initial_state = state_mapper(occ_indices)

        # Set given ansatz
        ansatz = _create_ansatz(
            self.ansatz,
            self.fermion_qubit_mapping,
            2 * self.n_orbitals,
            self.n_electron,
            self.layers,
            self.k,
            self.trotter_number,
            self.include_pi,
            self.use_singles,
            self.delta_sz,
            self.singlet_excitation,
        )
        # Create parametric state
        param_circuit = LinearMappedUnboundParametricQuantumCircuit(self.n_qubit)
        param_circuit.extend(self.initial_state.circuit)
        param_circuit.extend(ansatz)
        param_state = ParametricCircuitQuantumState(self.n_qubit, param_circuit)

        gradient_estimator = create_parameter_shift_gradient_estimator(
            self.parametric_estimator
        )

        def cost_fn(params):
            estimate = self.parametric_estimator(
                qubit_hamiltonian, param_state, [params]
            )[0]
            return estimate.value.real

        def grad_fn(params):
            estimate = gradient_estimator(qubit_hamiltonian, param_state, params)
            return np.asarray([g.real for g in estimate.values])

        print("----VQE-----")

        if self.is_init_random:
            np.random.seed(self.seed)
            init_params = np.random.random(size=param_circuit.parameter_count)
        else:
            init_params = [0.0] * param_circuit.parameter_count

        result = vqe(init_params, cost_fn, grad_fn, self.optimizer)

        self.opt_param = result.params

        # Store optimal state
        self.opt_state = param_state.bind_parameters(result.params)

        # Get energy
        self.e = result.cost
        return self.e, None

    # ======================
    def make_rdm1(self, _, norb, nelec, link_index=None, **kwargs):
        nelec = sum(nelec)
        dm1 = self._one_rdm(self.opt_state, norb, nelec)
        return dm1

    # ======================
    def make_rdm12(self, _, norb, nelec, link_index=None, **kwargs):
        nelec = sum(nelec)
        dm2 = self._two_rdm(self.opt_state, norb, nelec)
        return self._one_rdm(self.opt_state, norb, nelec), dm2

    # ======================
    def spin_square(self, civec, norb, nelec):
        return 0, 1

    # ======================
    def _one_rdm(self, state, norb, nelec):
        vqe_one_rdm = np.zeros((norb, norb))
        # get 1 rdm
        spin_dependent_rdm = np.real(
            get_1rdm(
                state,
                self.fermion_qubit_mapping,
                self.estimator,
                nelec,
            )
        )
        # transform it to spatial rdm
        vqe_one_rdm += spin_dependent_rdm[::2, ::2] + spin_dependent_rdm[1::2, 1::2]
        self.my_one_rdm = vqe_one_rdm
        return vqe_one_rdm

    # ======================
    def _dm2_elem(self, i, j, k, m, state, norb, nelec):
        op_mapper = self.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=2 * norb,
            n_fermions=nelec,
        )
        qubit_hamiltonian = op_mapper(
            FermionOperator(((i, 1), (j, 1), (k, 0), (m, 0))),
        )
        two_rdm_real = self.estimator([qubit_hamiltonian], [state])[0].value.real
        #
        # pyscf use real spin-free RDM (i.e. RDM in spatial orbitals)
        #
        return two_rdm_real

    # ======================
    def _two_rdm(self, state, norb, nelec):
        vqe_two_rdm = np.zeros((norb, norb, norb, norb))
        dm2aa = np.zeros_like(vqe_two_rdm)
        dm2ab = np.zeros_like(vqe_two_rdm)
        dm2bb = np.zeros_like(vqe_two_rdm)

        # generate 2 rdm
        spin_dependent_rdm = np.real(
            get_2rdm(
                state,
                self.fermion_qubit_mapping,
                self.estimator,
                nelec,
            )
        )

        # convert it into spatial
        for i, j, k, l in product(range(norb), range(norb), range(norb), range(norb)):
            ia = 2 * i
            ja = 2 * j
            ka = 2 * k
            la = 2 * l
            ib = 2 * i + 1
            jb = 2 * j + 1
            kb = 2 * k + 1
            lb = 2 * l + 1
            # aa
            dm2aa[i, j, k, l] = spin_dependent_rdm[ia, ja, ka, la]
            # bb
            dm2bb[i, j, k, l] = spin_dependent_rdm[ib, jb, kb, lb]
            #
            dm2ab[i, j, k, l] = spin_dependent_rdm[ia, jb, kb, la]
        self.my_two_rdm = (
            dm2aa.transpose(0, 3, 1, 2)
            + dm2bb.transpose(0, 3, 1, 2)
            + dm2ab.transpose(0, 3, 1, 2)
            + (dm2ab.transpose(0, 3, 1, 2)).transpose(2, 3, 0, 1)
        )
        return self.my_two_rdm

    # ======================
    def make_dm2(self, _, norb, nelec, link_index=None, **kwargs):
        dm2 = np.zeros((norb, norb, norb, norb))
        for i, j, k, l in product(range(norb), range(norb), range(norb), range(norb)):
            ia = 2 * i
            ja = 2 * j
            ka = 2 * k
            la = 2 * l
            ib = 2 * i + 1
            jb = 2 * j + 1
            kb = 2 * k + 1
            lb = 2 * l + 1
            # aa
            dm2[i, j, k, l] = (
                self._dm2_elem(ia, ja, ka, la, self.opt_state, norb, nelec)
                + self._dm2_elem(ib, jb, kb, lb, self.opt_state, norb, nelec)
                + self._dm2_elem(ia, ja, kb, lb, self.opt_state, norb, nelec)
                + self._dm2_elem(ib, jb, ka, la, self.opt_state, norb, nelec)
            )
        return dm2

def _sequential_cost_fn(estimator_s_p_triplet, hs):
    parametric_estimator, s, p = estimator_s_p_triplet
    return [parametric_estimator(h, s, p) for h in hs]

from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
def _sequential_grad_fn(estimator_s_p_triplet, hs):
    parametric_estimator, state, params = estimator_s_p_triplet
    return [
        parameter_shift_gradient_estimates(
            h, state, params, parametric_estimator
        ) for h in hs
    ]

class ParallelVQECI(VQECI):
    def __init__(self, *args: Any, npartitions:int=1, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.npartitions: int = npartitions
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

    def get_estimator(self):
        if isinstance(self.backend, QulacsBackend):
            from quri_parts.qulacs.estimator import create_qulacs_vector_estimator
            estimator = create_qulacs_vector_estimator()
        if isinstance(self.backend, ITensorBackend):
            estimator = importlib.import_module("quri_parts.itensor.estimator")._estimate
        return estimator
    def get_parametric_estimator(self):
        if isinstance(self.backend, QulacsBackend):
            from quri_parts.qulacs.estimator import create_qulacs_vector_parametric_estimator
            estimator = create_qulacs_vector_parametric_estimator()
        if isinstance(self.backend, ITensorBackend):
            from quri_parts.itensor.estimator import create_itensor_mps_parametric_estimator
            estimator = create_itensor_mps_parametric_estimator
        return estimator

    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        self.n_orbitals = norb
        self.n_qubit = self.fermion_qubit_mapping.n_qubits_required(2 * self.n_orbitals)
        self.n_electron = nelec[0] + nelec[1]

        # Get the active space Hamiltonian
        active_hamiltonian = _get_active_hamiltonian(h1, h2, norb, ecore)
        # Convert the Hamiltonian using `self.fermion_qubit_mapping`
        self.fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        op_mapper = self.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=2 * self.n_orbitals,
            n_fermions=self.n_electron,
        )

        qubit_hamiltonians: List[Operator] = []

        for f_h in partition(self.fermionic_hamiltonian, npartitions=self.npartitions):
            if f_h.terms:
                qubit_hamiltonians.append(op_mapper(f_h))
            else:
                continue

        # Set initial Quantum State
        occ_indices = list(range(self.n_electron))
        state_mapper = self.fermion_qubit_mapping.get_state_mapper(
            2 * self.n_orbitals, self.n_electron
        )
        self.initial_state = state_mapper(occ_indices)

        # Set given ansatz
        ansatz = _create_ansatz(
            self.ansatz,
            self.fermion_qubit_mapping,
            2 * self.n_orbitals,
            self.n_electron,
            self.layers,
            self.k,
            self.trotter_number,
            self.include_pi,
            self.use_singles,
            self.delta_sz,
            self.singlet_excitation,
        )
        # Create parametric state
        param_circuit = LinearMappedUnboundParametricQuantumCircuit(self.n_qubit)
        param_circuit.extend(self.initial_state.circuit)
        param_circuit.extend(ansatz)
        param_state = ParametricCircuitQuantumState(self.n_qubit, param_circuit)

        gradient_estimator = create_parameter_shift_gradient_estimator(
            self.parametric_estimator
        )

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
        def cost_fn(params):
            estimator = self.get_parametric_estimator()
            #s = param_state.bind_parameters(params)
            result = execute_concurrently(
                _sequential_cost_fn,
                common_input=(estimator, param_state, params),
                individual_inputs=qubit_hamiltonians,
                executor=executor,
                concurrency=2,
            )
            r = sum(r.value.real for r in result)
            return r

        """
        def grad_fn(params):
            with concurrent.futures.ProcessPoolExecutor(mp_context=get_context("spawn")) as executor:
                # Start the load operations and mark each future with its URL
                result = execute_concurrently(
                    gradient_estimator,
                    common_input=(param_state, params),
                    individual_inputs=qubit_hamiltonians,
                    executor=executor,
                )
            gs = np.sum([g.real for g in r.values] for r in result)
            return gs
        """

        """
        def grad_fn(params):
            gs = 0.0
            for qubit_hamiltonian in qubit_hamiltonians:
                estimate = gradient_estimator(qubit_hamiltonian, param_state, params)
                gs += np.asarray([g.real for g in estimate.values])
            return gs
        """

        qubit_hamiltonian = op_mapper(self.fermionic_hamiltonian)
        def grad_fn(params):
            estimate = gradient_estimator(qubit_hamiltonian, param_state, params)
            return np.asarray([g.real for g in estimate.values])

        print("----VQE-----")

        if self.is_init_random:
            np.random.seed(self.seed)
            init_params = np.random.random(size=param_circuit.parameter_count)
        else:
            init_params = [0.0] * param_circuit.parameter_count

        result = vqe(init_params, cost_fn, grad_fn, self.optimizer)

        self.opt_param = result.params

        # Store optimal state
        self.opt_state = param_state.bind_parameters(result.params)

        # Get energy
        self.e = result.cost
        return self.e, None