"""
PULSE Multi-Agent Uncertainty Estimator
"""

import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, required=True, default='HumanEval')
argparser.add_argument('--model', type=str, required=True, default='Qwen/Qwen3-4B-Instruct-2507')
argparser.add_argument('--device', type=str, default="0")
argparser.add_argument('--agent_num', type=int, default=5)
argparser.add_argument('--random_split', type=int, default=42)
argparser.add_argument('--noise', type=float, default=0.001)

args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from utils import get_AUROC, get_gradients, get_kernel, get_cholesky
import numpy as np
from sklearn.preprocessing import StandardScaler


class PULSE:
    """
    Multi-agent PULSE with (agent_num + 1) agent streams.

    fit():
        For each agent stream k ∈ {0, …, agent_num}:
          • Build kernel  K_k  from training gradients
          • Cholesky-factorise  L_k  of (K_k + λI)  for variance solves

    predict(test_grads):
        For each agent i:
          1.  v_i^(0) = κ(x_i, x_i) − k_{i,*}^T L_i^{-T} L_i^{-1} k_{i,*}   (Eq. 9)
          2.  For every other agent j ≠ i, apply one belief-update step:         (Eq. 10)
                v_i ← v_i * (1 − κ(x_i, m_j)^2 / ((1+λ)*v_i + κ(m_j, m_j)))
          3.  Accumulate  Δ_i = 0.5 * log(v_i^(0) / v_i^(final))

        Return:
          G_total = Σ_i Δ_i                                                      (Eq. 14)
    """

    def __init__(self, noise=1e-2, device="cuda", jitter=1e-6, agent_num=2):
        self.noise     = noise
        self.device    = device
        self.jitter    = jitter
        self.agent_num = agent_num
        self.num_streams = agent_num + 1   # total number of agent streams

    # ------------------------------------------------------------------
    # fit:  Cholesky factorisations for each agent stream
    # ------------------------------------------------------------------

    def fit(self, train_input, y_train, valid_input=None, y_valid=None,
            test_input=None, y_test=None):
        """
        Args
        ----
        train_input : list of (agent_num + 1) lists, each of length N,
                      where train_input[k][i] is the gradient vector for
                      agent stream k, training sample i.
        y_train     : (N,) tensor of binary correctness labels.
        """
        if len(train_input) != self.num_streams:
            raise ValueError(
                f"Expected {self.num_streams} agent streams, got {len(train_input)}"
            )

        self.X_train = []        # list of [N, d] tensors, one per stream
        self.L       = []        # Cholesky factors

        for k in range(self.num_streams):
            X_k = torch.tensor(train_input[k], dtype=torch.float32).to(self.device)
            self.X_train.append(X_k)

            K_k = get_kernel(train_input[k], train_input[k])
            L_k = get_cholesky(K_k, self.noise, self.jitter).to(self.device)
            self.L.append(L_k)

        print(f"[PULSE] fit complete  —  {self.num_streams} agent streams, "
              f"N={len(train_input[0])} training samples\n")

    # ------------------------------------------------------------------
    # predict:  Eq. 9 → Eq. 10 → Eq. 14
    # ------------------------------------------------------------------

    def predict(self, test_grads):
        """
        Args
        ----
        test_grads : list of (agent_num + 1) gradient vectors for one
                     test sample, i.e. test_grads[k] is the gradient
                     vector for agent stream k.

        Returns
        -------
        G_total : float
            System-level uncertainty score.  Higher → more confident → likely correct.
        """
        # Build per-stream test tensors  [1, d]
        x = []
        for k in range(self.num_streams):
            t = torch.tensor(test_grads[k], dtype=torch.float32).to(self.device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            x.append(t)

        g_total = 0.0

        with torch.no_grad():
            for i in range(self.num_streams):
                # ---- Step 1: v_i^(0) via Eq. 9 ----
                # κ(x_i, x_i)
                kappa_xi_xi = (x[i] * x[i]).sum()

                # k_{i,*} = K(x_test_i, X_train_i)   [1, N]
                k_star = x[i] @ self.X_train[i].T

                # (K_i + λI)^{-1} k_star^T   via Cholesky
                v_solved = torch.cholesky_solve(k_star.T, self.L[i])   # [N, 1]
                v_i = kappa_xi_xi - (k_star @ v_solved).squeeze()       # scalar
                v_i = v_i.clamp(min=1e-10)

                v_i_init = v_i.clone()

                # ---- Step 2: belief update from every other agent j ≠ i  (Eq. 10) ----
                for j in range(self.num_streams):
                    if j == i:
                        continue

                    # κ(x_i, m_j) : cross-similarity, agent i's output vs agent j's message
                    kappa_xi_mj = (x[i] @ x[j].T).squeeze()
                    # κ(m_j, m_j) : self-similarity of agent j's message
                    kappa_mj_mj = (x[j] * x[j]).sum()

                    denom       = (1.0 + self.noise) * v_i + kappa_mj_mj + 1e-10
                    contraction = (kappa_xi_mj ** 2) / denom
                    contraction = contraction.clamp(max=1.0 - 1e-6)

                    v_i = v_i * (1.0 - contraction)
                    v_i = v_i.clamp(min=1e-10)

                # ---- Step 3: accumulate log-ratio for agent i  (part of Eq. 14) ----
                g_total += 0.5 * torch.log(v_i_init / v_i).item()

        return g_total


# ═══════════════════════════════════════════════════════════════════════════
# Main  —  data loading, gradient computation, evaluation
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dataset    = args.dataset
    model_name = args.model
    agent_num  = args.agent_num

    if model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        embed_dim = 2048
    elif model_name == "meta-llama/Llama-3.1-70B-Instruct":
        embed_dim = 8192
    else:
        embed_dim = 5120

    # ----------------------------------------------------------------
    # Load raw results
    # ----------------------------------------------------------------
    filename = f"outputs/runs/{dataset}/{model_name}/{agent_num}react_not_learn_prompt_not_learn_demo/records.jsonl"
    print("Reading the results from " + filename)

    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lines.append(json.loads(line))

    results = []
    for line in tqdm(lines):
        tmp_dict = {}
        tmp_dict['is_solved'] = line['Solved']

        tmp = []
        for i in range(len(line['record'])):
            tmp.append(line['record'][i]['input'])
        tmp_dict['implementation_input'] = tmp

        tmp = []
        for i in range(len(line['record'])):
            tmp.append(line['record'][i]['output'])
        tmp_dict['responses'] = tmp

        results.append(tmp_dict)

    num_sample = len(results)
    sample_indices = [i for i in range(num_sample)]
    print("Total number of samples: ", num_sample,
          "positive samples: ", sum([int(result['is_solved']) for result in results]),
          "negative samples: ", num_sample - sum([int(result['is_solved']) for result in results]))

    # ----------------------------------------------------------------
    # Train / valid / test split
    # ----------------------------------------------------------------
    X_train, test_index, y_train, y_test = train_test_split(
        sample_indices, [int(result['is_solved']) for result in results],
        test_size=0.5, random_state=args.random_split,
        stratify=[int(result['is_solved']) for result in results]
    )
    print("Total number of test samples: ", len(y_test),
          "Test positive samples: ", sum(y_test),
          "Test negative samples: ", len(y_test) - sum(y_test))

    train_index, valid_index, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2, random_state=args.random_split,
        stratify=y_train
    )
    print("Total number of train samples: ", len(y_train),
          "Train positive samples: ", sum(y_train),
          "Train negative samples: ", len(y_train) - sum(y_train))
    print("Total number of valid samples: ", len(y_val),
          "Valid positive samples: ", sum(y_val),
          "Valid negative samples: ", len(y_val) - sum(y_val))

    # ---- Build sample tuples  [response_0, input_0, response_1, input_1, …] ----
    def _build_samples(indices):
        texts     = [results[i]['implementation_input'] for i in indices]
        responses = [results[i]['responses']            for i in indices]
        samples = []
        for i in range(len(responses)):
            tmp = []
            for j in range(agent_num + 1):
                tmp.append(responses[i][j])
                tmp.append(texts[i][j])
            samples.append(tmp)
        return samples

    train_samples = _build_samples(train_index)
    valid_samples = _build_samples(valid_index)
    test_samples  = _build_samples(test_index)

    orig_valid_index = valid_index

    y_train = torch.tensor([results[i]['is_solved'] for i in train_index], dtype=torch.float32)
    y_valid = torch.tensor([results[i]['is_solved'] for i in valid_index], dtype=torch.float32)
    y_test  = torch.tensor([results[i]['is_solved'] for i in test_index],  dtype=torch.float32)

    # ----------------------------------------------------------------
    # Gradient computation  (lazy model loading, caching to .npy)
    # ----------------------------------------------------------------
    _llm = {"m": None, "t": None, "e": None}

    def _load_or_compute_grads(filepath, samples, description):
        """Load cached gradients or compute and cache them."""
        if os.path.exists(filepath):
            grads = np.load(filepath).tolist()
            if len(grads) != len(samples):
                return grads
            return grads

        # Lazy-load LLM
        if _llm["m"] is None:
            _llm["t"] = AutoTokenizer.from_pretrained(model_name)
            _llm["t"].pad_token = _llm["t"].eos_token
            _llm["m"] = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", dtype=torch.float16
            )
            _llm["m"].eval()
            _llm["e"] = _llm["m"].get_input_embeddings().weight

        grads = []
        for i in tqdm(range(len(samples)), desc=description):
            tmp = []
            for j in range(agent_num + 1):
                tmp.append(get_gradients(
                    [samples[i][2 * j], samples[i][2 * j + 1]],
                    _llm["m"], _llm["t"], _llm["e"]
                ))
            grads.append(tmp)

        parent_dir = os.path.dirname(filepath)
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(filepath):
            os.remove(filepath)
        np.save(filepath, grads)
        return grads

    grad_base = (
        f"{filename.split('/')[0]}/{filename.split('/')[1]}/{filename.split('/')[2]}"
        f"/Grads/{str(args.random_split)}/{filename.split('/')[3]}"
        f"/{filename.split('/')[4]}/{filename.split('/')[5]}"
    )

    train_responses_grads = _load_or_compute_grads(
        f"{grad_base}-train_grad.npy", train_samples, "train grads"
    )
    valid_responses_grads = _load_or_compute_grads(
        f"{grad_base}-valid_grad.npy", valid_samples, "valid grads"
    )
    test_responses_grads = _load_or_compute_grads(
        f"{grad_base}-test_grad.npy", test_samples, "test grads"
    )

    # Release GPU memory from LLM
    del _llm
    torch.cuda.empty_cache()

    num_streams = agent_num + 1

    # Collect per-stream training matrices  [N, d]
    train_matrices = []
    for j in range(num_streams):
        mat = np.array([train_responses_grads[i][j]
                        for i in range(len(train_responses_grads))])
        train_matrices.append(mat)

    # Fit one scaler per agent stream on the FULL training set
    scalers = []
    for j in range(num_streams):
        sc = StandardScaler()
        sc.fit(train_matrices[j])
        scalers.append(sc)

    # Transform all splits
    def _normalise_grads(grads_list):
        normalised = []
        for i in range(len(grads_list)):
            tmp = []
            for j in range(num_streams):
                vec = np.array(grads_list[i][j]).reshape(1, -1)
                tmp.append(scalers[j].transform(vec).tolist()[0])
            normalised.append(tmp)
        return normalised

    train_responses_grads = _normalise_grads(train_responses_grads)
    valid_responses_grads = _normalise_grads(valid_responses_grads)
    test_responses_grads  = _normalise_grads(test_responses_grads)

    # ----------------------------------------------------------------
    # Reshape to  train_input[agent_k] = [sample_0, sample_1, …]
    # ----------------------------------------------------------------
    train_input = []
    for k in range(num_streams):
        train_input.append([sample[k] for sample in train_responses_grads])

    model_gp = PULSE(
        noise=args.noise,
        jitter=1e-5,
        agent_num=agent_num,
    )
    model_gp.fit(train_input, y_train)

    # ----------------------------------------------------------------
    # Evaluate — validation
    # ----------------------------------------------------------------
    preds_valid = []
    for grad in valid_responses_grads:
        pred = model_gp.predict(grad)
        preds_valid.append(pred)
    auroc_valid = get_AUROC(y_valid.tolist(), preds_valid)
    print(f"Valid AUROC: {auroc_valid:.4f}")

    # ----------------------------------------------------------------
    # Evaluate — test
    # ----------------------------------------------------------------
    preds_test = []
    for grad in test_responses_grads:
        pred = model_gp.predict(grad)
        preds_test.append(pred)
    auroc_test = get_AUROC(y_test.tolist(), preds_test)
    print(f"Test  AUROC: {auroc_test:.4f}")

    # ----------------------------------------------------------------
    # Save predictions
    # ----------------------------------------------------------------
    preds = []
    for i, (pred, label) in enumerate(zip(preds_test, y_test.tolist())):
        preds.append({
            "index": i + 1,
            "label": label,
            "mean": pred,
            "variance": pred,
        })

    out_file = (
        f"{filename.split('/')[0]}/{filename.split('/')[1]}/{filename.split('/')[2]}"
        f"/PULSE/{str(args.random_split)}/{filename.split('/')[3]}"
        f"/{filename.split('/')[4]}/{filename.split('/')[5]}.jsonl"
    )
    parent_dir = os.path.dirname(out_file)
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(out_file):
        os.remove(out_file)
    with open(out_file, "w", encoding="utf-8") as file:
        for pred in preds:
            file.write(json.dumps(pred) + "\n")

    print(f"Predictions saved to {out_file}")
