### Practical game-plan for **600 labelled + 2 400 unlabelled** samples

| Stage                                                           | Goal                                                                     | How to do it (tools that fit MorphoFeatures code-base)                                                                                                                                         | Why this order?                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **1. Unsupervised (self-supervised) pre-training**              | Make the encoder learn *all* morphology variation, regardless of labels. | *Exactly* the pipeline you’ve just studied—contrastive + auto-encoder on all 3 000 samples. (`train_shape.py`, `train_texture.py` in the repo).                                                | The 2 400 extra samples carry structure the labels don’t cover. Self-supervised pre-training lets you use it “for free”. |
| **2. Freeze encoder → train a light classifier on the 600**     | Turn 480-D vectors into class logits.                                    | Simple **logistic-regression** or **1-layer MLP** on top of frozen embeddings. (In repo: `analysis/clustering/cell_types.ipynb` does this.)                                                    | Gives you a quick baseline and tells you whether morphology alone is enough.                                             |
| **3. *Optional fine-tune* with a mixed loss (semi-supervised)** | Let labelled examples nudge embeddings into nicer class islands.         | Replace pure NT-Xent with **Supervised-Contrastive loss** (Khosla 2020) or add a cross-entropy head:<br>`L = L_contrast(all) + β·CE(frozen or shared head on 600)`.<br>Keep β small (0.1–0.3). | When labelled fraction ≥ 10 – 20 %, a tiny supervised term often lifts accuracy by 2–5 pp without overfitting.           |
| **4. Self-training / pseudo-labels**                            | Exploit high-confidence predictions to expand the train set.             | • Run the step-2 classifier on unlabelled data.<br>• Pick predictions with softmax > 0.9 as **pseudo-labels**.<br>• Re-train classifier (or step-3 fine-tune) on 600 + pseudo samples.         | Works well when classes form tight clusters in the embedding—common with MorphoFeatures.                                 |
| **5. Label-propagation sanity check**                           | See if k-NN voting already matches ground truth.                         | Build a 30-NN graph in the 480-D space; propagate 600 labels by majority vote; measure accuracy.                                                                                               | If accuracy is already high, a heavier supervised step may be unnecessary.                                               |

---

#### Why *not* train the encoder from scratch with the 600 labels only?

* You’d throw away 80 % of the data.
* DeepGCN/UNet branches have \~1 M parameters each—easy to over-fit 600 samples.
* Self-supervised pre-training is fast (hours) and gives a reusable backbone for any future labels.

#### When to **bake labels into the contrastive loss**?

* If classes you care about differ subtly (e.g. two neuron sub-types with similar shape), a **Supervised-Contrastive (SupCon)** variant during fine-tune often sharpens boundaries.
* Keep the unsupervised term too; otherwise embeddings collapse onto just the labelled structure.



### Rule of thumb

> **Learn the representation first, then let labels do the lightest possible shaping.**
> Only escalate to semi-supervised fine-tuning if the frozen-encoder baseline under-performs.

With 20 % coverage (600/3 000) you’ll usually get:

* **Frozen embedding + logistic head** → solid performance (often >90 % of fully supervised ceiling).
* **SupCon fine-tune** → +2-5 pp if classes are hard.

That mirrors what Zinchenko & Hugger observed: they trained the encoder without any labels, then a tiny logistic regressor on just \~60 labelled cells per class hit 96 % accuracy. That’s generally the most bang-for-buck path.
