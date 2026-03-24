# OT-scGHSOM

**OT-scGHSOM** is a framework that integrates hierarchical clustering with **Optimal Transport (OT)** to analyze the flow of cell populations across different time points.

The method is built on top of **scGHSOM hierarchical clustering**, and is designed to quantify **cluster-to-cluster transitions** by estimating the proportion of cells that move from one cluster to another over time.

By combining hierarchical structure with optimal transport, OT-scGHSOM allows users to examine **population dynamics at the cluster level**. This provides an intuitive way to understand how cellular populations evolve, expand, contract, or transition between states in longitudinal datasets.

Using this tool, users can visualize the **flow proportions between clusters at different time points**, helping to reveal potential biological trajectories or population shifts.

---

## Example Usage

To run the visualization using the example dataset, execute the following command:

```bash
python optimal_transport/visualize/ot_map.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --source=day0 --target=day7

---
