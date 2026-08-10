# Methodology: Training and Experiment Protocol

This document outlines the training methodology and experimental protocols used to optimize the Self-Healing Neural Cellular Automata (NCA). To achieve stable morphogenesis and self-healing, the model is trained end-to-end using a specialized environment that exposes the update rule to varying stages of cellular development[cite: 19].

---

## 1. Backpropagation Through Time (BPTT)

The NCA is optimized by simulating the cellular grid forward in time and comparing the resulting visible tissue fields to the target image[cite: 19]. Because the same neural update rule is applied iteratively, we treat the sequence of updates as a recurrent neural network unrolled over a spatial grid.

### Rollout Execution
During a single training step, a batch of cellular states is unrolled for a configured number of iterations (e.g., 64 steps). The computational graph is retained across these steps[cite: 19]. 

### Objective Function
At the end of the rollout, we extract the visible channels (Epidermis, Dermis, Vasculature) and compute the Mean Squared Error (MSE) against the normalized RGB target image:

$$ \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (S_{visible}^{(T)} - Y_{target})^2 $$

Where $T$ is the number of rollout steps. The gradients are computed by backpropagating through the entire temporal chain of local cellular interactions, updating the shared $1 \times 1$ convolutional parameters.

---

## 2. Sample Pool Lifecycle

Training a model strictly from a single, initial seed state at every step leads to catastrophic forgetting; the model learns how to grow the structure but immediately forgets how to maintain it once fully grown. To ensure long-term stability, we utilize a **Sample Pool**[cite: 19].

### Pool Mechanics
1. **Initialization:** A fixed-size pool (e.g., 128 states) is initialized with identical, single-cell seed states[cite: 19].
2. **Sampling:** For each training step, a batch (e.g., 8 states) is randomly sampled from the pool[cite: 19].
3. **Reseeding:** To ensure the model never forgets how to begin growth from scratch, a configured fraction of the sampled batch (typically $0.25$ or 25%) is forcefully overwritten with fresh seed states[cite: 19].
4. **Replacement:** After the BPTT rollout completes, the newly evolved states are detached from the autograd graph and placed back into the pool at their original indices[cite: 19].

By training on partially evolved states, the NCA learns continuous homeostasis—it encounters structures at intermediate and fully developed stages and learns to stabilize them toward the target morphology.

---

## 3. Damage Injection and Self-Healing

Biological systems do not inherently know how to heal; they rely on robust, decentralized signaling to adapt to structural loss. We enforce this behavior in the NCA through explicit, reproducible damage simulations[cite: 19].

During the pool sampling phase, if damage is enabled, a random subset of the sampled batch undergoes a destructive operation. We apply geometric zero-masks (e.g., circles, squares, or half-grid removals) that instantly destroy all state data (both visible and hidden channels) in the affected region[cite: 19]. 

Because these severely damaged states are fed directly into the BPTT rollout, the neural update rule is penalized if it fails to reconstruct the missing tissue. This forces the latent channels to learn robust geometric coordination and regenerative signaling to minimize the MSE loss at the end of the rollout[cite: 19].

---

## 4. Standardized Evaluation Benchmarks

To systematically prove the NCA's capabilities, all models are subjected to a unified, three-phase evaluation pipeline[cite: 19]:

1. **Morphogenesis (Growth):** The model must grow the target structure from a single seed within a standard temporal window (e.g., 128 steps), evaluated via final MSE[cite: 19].
2. **Structural Persistence:** A successfully grown state is rolled forward without gradients for an extended horizon (e.g., 1,000 steps)[cite: 19]. We calculate the mathematical variance and structural drift to ensure the tissue does not degrade, oscillate, or explode into noise.
3. **Regeneration (Healing):** A fully grown tissue is subjected to a severe geometric wound[cite: 19]. We record the immediate Post-Damage MSE, run a recovery rollout, and calculate the Recovery Efficiency based on the model's ability to restore the Pre-Damage baseline.