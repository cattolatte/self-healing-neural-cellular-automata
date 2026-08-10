# Experimental Results and Benchmarks

This document showcases the qualitative and quantitative evaluations of the trained Self-Healing Neural Cellular Automata (NCA). The visualizations and metrics below demonstrate the model's ability to learn morphogenesis, maintain structural persistence, and regenerate tissue after geometric damage.

---

## 1. Training Diagnostics

The model is optimized using Backpropagation Through Time (BPTT) with a Mean Squared Error (MSE) reconstruction loss. The loss curve tracks the convergence of the shared 1x1 convolutional update rule over the training epochs.

*(The training script automatically exports this figure to the outputs directory.)*

![Training Reconstruction Loss](../outputs/training_loss_curve.png)

> **Note:** A successful training run will show a rapid initial drop in MSE, followed by a long tail of refinement as the model learns to stabilize the hidden communication channels.

---

## 2. Morphogenesis (Growth)

The primary objective is for the NCA to grow the target tissue pattern starting from a single, identically initialized central seed cell. The animation below illustrates the unrolled evaluation rollout from step 0 to the configured evaluation horizon.

![Growth Animation](../outputs/growth_animation.gif)

*Alternatively, view the high-quality MP4 render: [growth_animation.mp4](../outputs/growth_animation.mp4)*

### Quantitative Growth Metrics
*   **Target Pattern:** (e.g., Smiley, Lizard, or custom RGB grid)
*   **Final Growth MSE:** *(Refer to `outputs/evaluation_metrics.json`)*

---

## 3. Structural Persistence

A biological system must not only grow its adult form but maintain it indefinitely. The persistence benchmark tests whether the learned cellular update rule remains stable over an extended timeline (e.g., 1,000+ steps) without external intervention or gradient updates.

*   **Drift:** Measures the absolute change in MSE from the fully grown state to the end of the extended rollout.
*   **Variance:** Tracks structural oscillation.

If the internal latent signals are poorly stabilized during training, the visual pattern will degrade into noise or "explode" exponentially. Stable models maintain a strict variance near zero. 

*(Check `outputs/evaluation_metrics.json` for the boolean `is_stable` flag and quantitative variance/drift metrics.)*

---

## 4. Self-Healing and Regeneration

The most advanced capability of the decentralized NCA is autonomous damage recovery. In this scenario, a fully grown structure is subjected to a severe, randomized geometric wipe (e.g., a circular void or removal of the entire right half of the grid). All visible and hidden state data in the affected region is strictly zeroed out.

### Regeneration Rollout
The surviving cells must utilize their hidden signaling channels to detect the boundary loss, propagate repair signals, and rebuild the missing morphology perfectly.

![Healing Animation](../outputs/healing_animation.gif)

*Alternatively, view the high-quality MP4 render: [healing_animation.mp4](../outputs/healing_animation.mp4)*

### Recovery Diagnostics
The bar chart below maps the Mean Squared Error against the ground truth target across the three phases of the injury simulation:
1.  **Pre-Damage:** The baseline MSE of the intact structure.
2.  **Wounded:** The immediate spike in MSE instantly following the geometric cut.
3.  **Recovered:** The final MSE after the NCA is allowed to run its recovery rollout.

![Tissue Regeneration Evaluation](../outputs/healing_evaluation_bar_chart.png)

> **Success Criterion:** A high *Recovery Efficiency* percentage indicates that the NCA successfully eliminated the error introduced by the wound, restoring the tissue to near Pre-Damage quality.