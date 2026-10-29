# Stonkfly integration notes

Reviewed https://github.com/nftechie/stonkfly at commit
`78ef3e05ab0fa086032098558d893667068944a0` on September 10, 2026.
License: MIT, copyright 2026 nftechie and DOOMFLY contributors.

## Applied in Flytrader

- Reproducibility records inspired by its source locks and checkpoint manifests:
  replay reports now record SHA-256 fingerprints of both inputs and model/CLI
  source, Python/NumPy/SciPy versions, and the random seed. Inputs are checked
  again after evaluation; changes prevent writing a report. Reports are replaced
  atomically. This records identity; it does not verify against upstream locks.
- Explicit distinction between retained biological connectivity, modeled
  dynamics, and demonstrated financial results in the GUI.
- Full retained connectivity using the same explicit inclusion policy: assigned
  neuronal superclasses, excluding glia and unresolved objects. A new original
  Arrow-to-CSR compiler reproduced 166,700 neurons, 25,582,938 directed connections
  and 124,177,617 contacts from the downloaded release. The full graph runs through
  Flytrader's reservoir; display sampling does not reduce the computation.

These implementations are original Flytrader code. No Stonkfly code has been
vendored or executed. All trades remain simulated.

## Other architecture, not yet integrated

Upstream array locks have not been adopted. Flytrader records its own source and
compiled-file fingerprints, and validates the sparse structure when loading it.

Its spiking simulator, retinal chart encoding, reinforcement signals, native C++
kernel and plasticity checkpoints form a different model from Flytrader's
feature-driven reservoir. Importing those requires a separate backend, upstream
MIT notices, numerical tests, and fresh performance measurements. Their presence
is not proof of trading skill. Exchange account access and live orders remain
outside the paper-only implementation.

Upstream references:
- https://github.com/nftechie/stonkfly/blob/78ef3e05ab0fa086032098558d893667068944a0/docs/model.md
- https://github.com/nftechie/stonkfly/blob/78ef3e05ab0fa086032098558d893667068944a0/stonkfly/neural/connectome.py
- https://github.com/nftechie/stonkfly/blob/78ef3e05ab0fa086032098558d893667068944a0/stonkfly/neural/brain.py
