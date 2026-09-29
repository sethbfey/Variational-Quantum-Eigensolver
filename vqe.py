"""
Command-lines Args:
    uv run vqe.py
        --qubits <num of qubits>
        --layers <num of layers>
        --circuit "<circuit>"
        --omega <qubit 1> ... <qubit n>
        --J <Jx> <Jy> <Jz>
        --restarts <num of optimizer starting angles to try>
        --seed <for reproducibility>
"""

# IMPORTS
import argparse
import numpy as np
from scipy.optimize import minimize


# COST FUNCTION PARAMS
ALPHA = 100
BETA = 100 
GAMMA = 0.10
DELTA = 0.50
ETA = 0.10
ZETA = 5
NOISE_PER_LAYER = 0.05


# CONSTRUCTING QUANTUM GATES
# pauli gates
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# native device controls
GENERATOR = {
    "rx": X / 2,
    "ry": Y / 2,
    "rz": Z / 2,
    "ex": (np.kron(X, X) + np.kron(Y, Y)) / 2,
    "zz": np.kron(Z, Z) / 2,
}

def gate_matrix(control, theta):
    # contructs the resource set
    # uses spectral theorem to calculate the available device gates
    # e^{-i \theta A} for hermitian A
    eigenvalues, eigenvectors = np.linalg.eigh(GENERATOR[control])
    return eigenvectors @ np.diag(np.exp(-1j * theta * eigenvalues)) @ eigenvectors.conj().T


# DEFINING THE HAMILTONIAN
def operator_on(num_qubits, matrix_on_qubit):
    # kronecker product on qubit q with identity everywhere else
    # e.g. operator_on(4, {2: Z}) -> I2 x I2 x Z x I2
    full_operator = np.eye(1, dtype=complex)
    for qubit in range(num_qubits):
        small_matrix = matrix_on_qubit.get(qubit, I2)
        full_operator = np.kron(full_operator, small_matrix)
    return full_operator

def build_hamiltonian(num_qubits, omegas, couplings):
    Jx, Jy, Jz = couplings
    dimension = 2**num_qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    # single-qubit terms
    for qubit in range(num_qubits):
        hamiltonian += omegas[qubit] / 2 * operator_on(num_qubits, {qubit: Z})

    # coupling terms between qubit and qubit+1
    for qubit in range(num_qubits - 1):
        couple = qubit + 1
        hamiltonian += Jx * operator_on(num_qubits, {qubit: X, couple: X})
        hamiltonian += Jy * operator_on(num_qubits, {qubit: Y, couple: Y})
        hamiltonian += Jz * operator_on(num_qubits, {qubit: Z, couple: Z})

    return hamiltonian


# PARSING THE CIRCUIT STRING
def split_outside_parenthesis(text_to_split, delimiter):
    # separates text by delimeter, ignoring inside delimeters
    pieces = []
    current = ""
    depth = 0

    for character in text_to_split:
        if character == "(":
            depth += 1
            current += character
        elif character == ")":
            depth -= 1
            current += character
        elif character == delimiter and depth == 0:
            pieces.append(current.strip())
            current = ""
        else:
            current += character

    pieces.append(current.strip())
    return [piece for piece in pieces if piece]

def parse_circuit(circuit_text):
    # turns the given circuit text into a list of gates
    # where each gate is (control, qubits, angle_code, letters)
    gates = []

    for gate_text in split_outside_parenthesis(circuit_text, " "):
        control, _, arguments_text = gate_text.partition("(")
        control = control.lower()
        qubit_text, angle_text = split_outside_parenthesis(arguments_text[:-1], ",")
        qubits = tuple(int(qubit) - 1 for qubit in qubit_text.strip(" ()").split(","))

        angle_code = compile(angle_text.strip(), "<angle>", "eval")
        letters = [name for name in angle_code.co_names if name != "pi"]
        gates.append((control, qubits, angle_code, letters))
        
    return gates

def repeat_layers(gates, num_layers):
    # stacks the circuit num_layers times
    # and each layer gets a copy of the angle to optimize
    steps = []
    parameters = []

    for layer in range(1, num_layers + 1):
        for control, qubits, angle_code, letters in gates:
            letter_to_parameter = {}
            for letter in letters:
                parameter = letter if num_layers == 1 else f"{letter}{layer}"
                letter_to_parameter[letter] = parameter
                if parameter not in parameters:
                    parameters.append(parameter)
            steps.append((control, qubits, angle_code, letter_to_parameter))

    return steps, parameters

def angle_of(step, parameter_values):
    # returns the angle of one gate in the layered circuit
    _, _, angle_code, letter_to_parameter = step
    letter_values = {letter: parameter_values[parameter] for letter, parameter in letter_to_parameter.items()}
    return float(eval(angle_code, {"__builtins__": {}, "pi": np.pi}, letter_values))


# SIMULATION
def apply_gate(state, matrix, qubits):
    # multiplies the gate matrix onto the given qubits of the state
    num_gate_qubits = len(qubits)
    gate_tensor = matrix.reshape([2] * (2 * num_gate_qubits))  # e.g. 4x4 -> 2x2x2x2
    output_axes = list(range(num_gate_qubits))
    input_axes = list(range(num_gate_qubits, 2 * num_gate_qubits))
    state = np.tensordot(gate_tensor, state, axes=(input_axes, qubits))
    return np.moveaxis(state, output_axes, qubits)

def run_circuit(steps, parameter_values, num_qubits):
    # starts in |00...0> and applies every step in order
    state = np.zeros([2] * num_qubits, dtype=complex)
    state[(0,) * num_qubits] = 1.0

    for step in steps:
        control, qubits, _, _ = step
        matrix = gate_matrix(control, angle_of(step, parameter_values))
        state = apply_gate(state, matrix, list(qubits))

    return state

def energy_and_gradient(parameter_vector, steps, parameters, num_qubits, hamiltonian):
    # returns the energy <psi|H|psi> and its gradient, for the optimizer
    parameter_values = dict(zip(parameters, parameter_vector))
    psi = run_circuit(steps, parameter_values, num_qubits)
    h_psi = (hamiltonian @ psi.reshape(-1)).reshape(psi.shape)
    energy = float(np.real(np.vdot(psi, h_psi)))

    # propogate backwards through the circuit one gate at a time
    state_back = psi
    h_psi_back = h_psi
    gradient = {parameter: 0.0 for parameter in parameters}

    for step in reversed(steps):
        control, qubits, _, letter_to_parameter = step
        U = gate_matrix(control, angle_of(step, parameter_values))
        U_inverse = U.conj().T
        state_back = apply_gate(state_back, U_inverse, list(qubits))

        if letter_to_parameter:
            dU_state = apply_gate(state_back, -1j * GENERATOR[control] @ U, list(qubits))
            gate_slope = 2 * np.real(np.vdot(h_psi_back, dU_state))

            for parameter in letter_to_parameter.values():
                nudged_up = dict(parameter_values)
                nudged_up[parameter] += 1e-6
                nudged_down = dict(parameter_values)
                nudged_down[parameter] -= 1e-6
                angle_slope = (angle_of(step, nudged_up) - angle_of(step, nudged_down)) / 2e-6
                gradient[parameter] += gate_slope * angle_slope

        h_psi_back = apply_gate(h_psi_back, U_inverse, list(qubits))

    return energy, np.array([gradient[parameter] for parameter in parameters])


# COST FUNCTION
def circuit_depth(steps, num_qubits):
    # number of time steps, if gates on different qubits run at the same time
    time_on_qubit = [0] * num_qubits

    for _, qubits, _, _ in steps:
        gate_time = max(time_on_qubit[qubit] for qubit in qubits) + 1
        for qubit in qubits:
            time_on_qubit[qubit] = gate_time

    return max(time_on_qubit)


# MAIN PROGRAM
def main():
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--omega", type=float, nargs="+")
    parser.add_argument("--J", type=float, nargs=3, default=[0.3, 0.1, 0.2])
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=202680)
    args = parser.parse_args()

    num_qubits = args.qubits

    # default alternating zeeman splittings for now
    if args.omega is None:
        omegas = [1.0 if qubit % 2 == 0 else 0.8 for qubit in range(num_qubits)]
    elif len(args.omega) == 1:
        omegas = args.omega * num_qubits
    else:
        omegas = args.omega

    # answer key for eigenvalues and eigenvectors of the hamiltonian
    hamiltonian = build_hamiltonian(num_qubits, omegas, args.J)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    ground_energy = eigenvalues[0]
    ground_states = eigenvectors[:, eigenvalues < ground_energy + 1e-9]

    # build the layered circuit
    steps, parameters = repeat_layers(parse_circuit(args.circuit), args.layers)

    # optimize from many random angles
    rng = np.random.default_rng(args.seed)
    best_vector = np.array([])

    if parameters:
        results = []
        for _ in range(args.restarts):
            start = rng.uniform(0, 2 * np.pi, len(parameters))
            results.append(
                minimize(
                    energy_and_gradient,
                    start,
                    args=(steps, parameters, num_qubits, hamiltonian),
                    jac=True,
                    method="BFGS"
                )
            )

        best_vector = min(results, key=lambda result: result.fun).x
    best_values = dict(zip(parameters, best_vector))

    # final state and energy
    psi = run_circuit(steps, best_values, num_qubits).reshape(-1)
    energy = float(np.real(np.vdot(psi, hamiltonian @ psi)))

    # cost function pieces
    energy_error = abs(energy - ground_energy)
    infidelity = 1 - float(np.sum(np.abs(ground_states.conj().T @ psi) ** 2))
    depth = circuit_depth(steps, num_qubits)
    num_entagling_gates = sum(1 for _, qubits, _, _ in steps if len(qubits) == 2)
    num_parameters = len(parameters)
    noise_risk = NOISE_PER_LAYER * args.layers
    cost = (
        ALPHA * energy_error + BETA * infidelity + GAMMA * depth
        + DELTA * num_entagling_gates + ETA * num_parameters + ZETA * noise_risk
    )

    # print report
    print(f"Cost = {cost:.4f}")
    print(f"Reached Ground Energy: {'YES' if energy_error < 1e-6 else 'NO'}"
          f"\n\tCircuit Energy E = {energy:.5f}"
          f"\n\tTrue Ground Energy = {ground_energy:.5f}")
    
    if parameters:
        print("Optimized Angles:")
        for parameter, value in best_values.items():
            print(f"\t{parameter} = {value:.5f}")

    print("Hamiltonian Energies:")
    for index, eigenvalue in enumerate(eigenvalues):
        print(f"\tE_{index} = {eigenvalue:.5f}")


if __name__ == "__main__":
    main()