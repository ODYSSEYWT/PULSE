import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, required=True, default='HumanEval')
argparser.add_argument('--model', type=str, required=True, default='Qwen/Qwen3-4B-Instruct-2507')
argparser.add_argument('--device', type=str, default="0")
argparser.add_argument('--agent_num', type=int, default=5)
argparser.add_argument('--random_split', type=int, default=42)
argparser.add_argument('--epochs', type=int, default=200)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--mixer_hidden_dim', type=int, default=256)
argparser.add_argument('--mixer_layers', type=int, default=3)
argparser.add_argument('--noise', type=float, default=0.001)

args = argparser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from utils import get_AUROC, get_prediction, get_gradients, get_kernel, get_cholesky, get_prediction_grads
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler


class GPResidualCorrector(nn.Module):
    """
    Takes GP predictions + kernel features and learns weights for each prediction
    """
    def __init__(self, num_train_samples, hidden_dim=128, agent_num=2, use_softmax=True):
        super().__init__()
        self.agent_num = agent_num
        self.use_softmax = use_softmax
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_train_samples * (agent_num + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Predict weights for each GP prediction
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim + agent_num + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, agent_num + 2)  # One weight per GP prediction
        )
    
    def forward(self, k_list, pred_list):
        """
        Learn weights to multiply with GP predictions
        Returns weights tensor
        """
        # Extract features from kernels
        kernel_features = torch.cat(k_list, dim=-1)
        features = self.feature_extractor(kernel_features)
        
        # Combine with GP predictions
        gp_preds = torch.stack(pred_list, dim=-1)
        combined = torch.cat([features, gp_preds], dim=-1)
        
        # Predict weights for each GP prediction
        raw_weights = self.weight_head(combined)

        if self.use_softmax:
            # Use softmax for interpretable, normalized weights (excluding bias)
            weights = torch.softmax(raw_weights[:-1], dim=-1)
            bias = raw_weights[-1]
            
            return torch.cat([weights, bias.unsqueeze(0)])
        else:
            return raw_weights


class PULSE:
    def __init__(self, noise=1e-2, device="cuda", jitter=1e-6, mixer_hidden_dim=256, mixer_layers=1, agent_num=2, label_smoothing=0.1, weight_decay=1e-4):
        self.noise = noise
        self.device = device
        self.jitter = jitter
        self.mixer = None
        self.mixer_optimizer = None
        self.mixer_hidden_dim = mixer_hidden_dim
        self.mixer_layers = mixer_layers
        self.agent_num = agent_num
        self.label_smoothing = label_smoothing  # Prevents overconfidence
        self.best_mixer_state = None
        self.best_valid_auroc = -1.0
        self.weight_decay = weight_decay

    def fit(self, train_input_grad, y_train, valid_input, y_valid, test_input, y_test, epochs=100, lr=0.001, scheduler_patience=10, min_lr=1e-6):
        if len(train_input_grad) != (self.agent_num + 1):
            print("Shape mismatch for training")
        
        self.train_responses_grad = train_input_grad
        self.y_train = y_train.to(self.device).float()  # Ensure float for BCE

        self.valid_responses_grad = valid_input
        self.y_valid = y_valid

        self.test_responses_grad = test_input
        self.y_test = y_test

        K = []
        for i in range(self.agent_num + 1):
            K.append(get_kernel(self.train_responses_grad[i], self.train_responses_grad[i]))

        self.K = K

        L = []
        for i in range(self.agent_num + 1):
            L.append(get_cholesky(self.K[i], self.noise, self.jitter).to(self.device))

        self.L = L

        alpha = []
        for i in range(self.agent_num + 1):
            tmp = torch.linalg.solve_triangular(self.L[i], self.y_train.reshape(-1, 1), upper=False)
            alpha.append(torch.linalg.solve_triangular(self.L[i].T, tmp, upper=True).to(self.device))

        self.alpha = alpha

        self.train_mixer(num_epochs=epochs, lr=lr, scheduler_patience=scheduler_patience, min_lr=min_lr)
    
    def train_mixer(self, num_epochs=100, lr=1e-3, scheduler_patience=10, min_lr=1e-6):
        N = len(self.train_responses_grad[0])
        
        if self.mixer is None:
            self.mixer = GPResidualCorrector(
                num_train_samples=N,
                hidden_dim=self.mixer_hidden_dim,
                agent_num=self.agent_num
            ).to(self.device)
            self.mixer_optimizer = optim.AdamW(
                self.mixer.parameters(), 
                lr=lr, 
                weight_decay=self.weight_decay
            )
            # Add learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.mixer_optimizer, 
                mode='max',  # maximize AUROC
                factor=0.5, 
                patience=scheduler_patience,
                min_lr=min_lr,
            )
        
        K = []
        for i in range(self.agent_num + 1):
            K.append(self.K[i].to(self.device))
        
        y_target = self.y_train.to(self.device).squeeze()
        
        # Apply label smoothing: 0 -> epsilon, 1 -> 1-epsilon
        if self.label_smoothing > 0:
            y_target_smoothed = y_target * (1 - self.label_smoothing) + self.label_smoothing / 2
        else:
            y_target_smoothed = y_target
        
        alpha = []
        for i in range(self.agent_num + 1):
            alpha.append(self.alpha[i].squeeze())

        
        with torch.no_grad():
            gp_preds_train = []
            for i in range(N):
                pred_sum = sum([(K[j][i] * alpha[j]).sum() for j in range(self.agent_num + 1)])
                gp_preds_train.append(torch.sigmoid(pred_sum / (self.agent_num + 1)).item())
            gp_baseline_auroc = get_AUROC(y_target.cpu().tolist(), gp_preds_train)
        
        # Use BCEWithLogitsLoss for numerical stability
        # It combines sigmoid + BCE in one operation
        pos_count = y_target.sum()
        neg_count = len(y_target) - pos_count
        pos_weight = neg_count / (pos_count + 1e-8)  # Handle class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # For early stopping
        patience = num_epochs
        patience_counter = 0
        
        print(f"\n[Training Residual Corrector] Epochs: {num_epochs}, LR: {lr}, Label Smoothing: {self.label_smoothing}")
        print(f"Early stopping patience: {patience}")
        
        for epoch in range(num_epochs):
            self.mixer.train()
            total_loss = 0
            indices = torch.randperm(N)
            
            for idx in indices:
                i = idx.item()
                k_i = []
                pred_i = []
                # Get base GP predictions
                for j in range(self.agent_num + 1):
                    k_i.append(K[j][i])
                    pred_i.append((k_i[j] * alpha[j]).sum())
                
                # Learn weights (returns weights for each prediction)
                weights = self.mixer(k_i, pred_i).squeeze()
                
                # Element-wise multiply weights with predictions, then sum
                pred_tensor = torch.stack(pred_i)
                y_pred_logits = (weights[:-1] * pred_tensor).sum() + weights[-1]
                
                # BCEWithLogitsLoss expects logits (before sigmoid)
                loss = criterion(y_pred_logits, y_target_smoothed[i])
                
                self.mixer_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), max_norm=1.0)
                self.mixer_optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / N
                
                # Print some stats
                with torch.no_grad():
                    tmp = [k_tmp @ alpha_tmp for k_tmp, alpha_tmp in zip(K, alpha)]
                    gp_baseline_logits = (sum(tmp) / (self.agent_num + 1)).squeeze()
                    baseline_loss = criterion(gp_baseline_logits, y_target_smoothed).item()

                    preds_valid = []
                    for grad in self.valid_responses_grad:
                        pred = self.predict(grad)
                        preds_valid.append(pred)
                    auroc_valid = get_AUROC(self.y_valid.tolist(), preds_valid)

                    preds_test = []
                    for grad in self.test_responses_grad:
                        pred = self.predict(grad)
                        preds_test.append(pred)
                    auroc_test = get_AUROC(self.y_test.tolist(), preds_test)

                    print(f"Epoch: [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, GP Loss: {baseline_loss:.4f}, GP AUROC: {gp_baseline_auroc:.4f}, Valid AUROC: {auroc_valid:.4f}, Test AUROC: {auroc_test:.4f}")

                    self.scheduler.step(auroc_valid)
                    
                    # Save best model based on validation AUROC
                    if auroc_valid >= self.best_valid_auroc:
                        self.best_valid_auroc = auroc_valid
                        self.best_mixer_state = {
                            'epoch': epoch + 1,
                            'state_dict': self.mixer.state_dict(),
                            'optimizer_state_dict': self.mixer_optimizer.state_dict(),
                            'auroc': auroc_valid,
                            'loss': avg_loss
                        }
                        print(f"  *** New best model! AUROC: {auroc_valid:.4f} ***")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= patience // 5:  # Check every 5 epochs, so divide patience
                        print(f"\nEarly stopping triggered. No improvement for {patience} epochs.")
                        print(f"Best AUROC: {self.best_valid_auroc:.4f} at epoch {self.best_mixer_state['epoch']}")
                        break
        
        # Restore best model
        if self.best_mixer_state is not None:
            print(f"\n[Restoring Best Model]")
            print(f"  Best epoch: {self.best_mixer_state['epoch']}")
            print(f"  Best AUROC: {self.best_mixer_state['auroc']:.4f}")
            print(f"  Best Loss: {self.best_mixer_state['loss']:.6f}")
            self.mixer.load_state_dict(self.best_mixer_state['state_dict'])
        
        print("[Training Complete]\n")
    
    def predict(self, test_intput_grad, return_probs=True):
        """
        Args:
            test_intput_grad: Input gradients for prediction
            return_probs: If True, return probabilities (after sigmoid)
                         If False, return logits
        """
        K_star = []
        for i in range(self.agent_num + 1):
            K_star.append(get_kernel(test_intput_grad[i], self.train_responses_grad[i]).to(self.device))
        
        if K_star[0].ndim == 1:
            for i in range(self.agent_num + 1):
                K_star[i] = K_star[i].unsqueeze(0)

        alpha = []
        for i in range(self.agent_num + 1):
            alpha.append(self.alpha[i].squeeze())
        
        self.mixer.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(K_star[0].shape[0]):
                k_i = []
                pred_i = []
                for j in range(self.agent_num + 1):
                    k_i.append(K_star[j][i])
                    pred_i.append((k_i[j] * alpha[j]).sum())
                
                # Learned weights
                weights = self.mixer(k_i, pred_i).squeeze()
                
                # Element-wise multiply and sum to get logits
                pred_tensor = torch.stack(pred_i)
                y_pred_logits = (weights[:-1] * pred_tensor).sum() + weights[-1]
                
                # Convert to probability if requested
                if return_probs:
                    y_pred = torch.sigmoid(y_pred_logits)
                else:
                    y_pred = y_pred_logits
                    
                predictions.append(y_pred)
        
        predictions = torch.stack(predictions)
        return predictions.tolist()[0] if len(predictions) == 1 else predictions.tolist()
    
    def save_best_model(self, filepath):
        """Save the best mixer model to disk"""
        if self.best_mixer_state is not None:
            torch.save(self.best_mixer_state, filepath)
            print(f"Best model saved to {filepath}")
        else:
            print("No best model to save. Train the model first.")
    
    def load_best_model(self, filepath):
        """Load a saved mixer model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Initialize mixer if not already done
        if self.mixer is None:
            N = len(self.train_responses_grad[0])
            self.mixer = GPResidualCorrector(
                num_train_samples=N,
                hidden_dim=self.mixer_hidden_dim,
                agent_num=self.agent_num
            ).to(self.device)
        
        self.mixer.load_state_dict(checkpoint['state_dict'])
        self.best_valid_auroc = checkpoint['auroc']
        self.best_mixer_state = checkpoint
        
        print(f"Model loaded from {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  AUROC: {checkpoint['auroc']:.4f}")
        print(f"  Loss: {checkpoint['loss']:.6f}")


if __name__ == "__main__":
    dataset = args.dataset
    model_name = args.model
    agent_num = args.agent_num

    if model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        embed_dim = 2048
    elif model_name == "meta-llama/Llama-3.1-70B-Instruct":
        embed_dim = 8192
    else:
        embed_dim = 5120

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
    print("Total number of samples: ", num_sample, "positive samples: ", sum([int(result['is_solved']) for result in results]), "negative samples: ", num_sample - sum([int(result['is_solved']) for result in results]))

    X_train, test_index, y_train, y_test = train_test_split(sample_indices, [int(result['is_solved']) for result in results], test_size=0.5, random_state=args.random_split, stratify=[int(result['is_solved']) for result in results])
    print("Total number of test samples: ", len(y_test), "Test positive samples: ", sum(y_test), "Test negative samples: ", len(y_test) - sum(y_test))
    
    train_index, valid_index, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=args.random_split, stratify=y_train)
    print("Total number of train samples: ", len(y_train), "Train positive samples: ", sum(y_train), "Train negative samples: ", len(y_train) - sum(y_train))
    print("Total number of valid samples: ", len(y_val), "Valid positive samples: ", sum(y_val), "Valid negative samples: ", len(y_val) - sum(y_val))
    
    train_texts = [results[i]['implementation_input'] for i in train_index]
    train_responses = [results[i]['responses'] for i in train_index]
    y_train = torch.tensor([results[i]['is_solved'] for i in train_index], dtype=torch.float32)
    train_samples = []
    for i in range(len(train_responses)):
        tmp = []
        for j in range(agent_num + 1):
            tmp.append(train_responses[i][j])
            tmp.append(train_texts[i][j])
        train_samples.append(tmp)

    orig_valid_index = valid_index

    valid_texts = [results[i]['implementation_input'] for i in valid_index]
    valid_responses = [results[i]['responses'] for i in valid_index]
    y_valid = torch.tensor([results[i]['is_solved'] for i in valid_index], dtype=torch.float32)
    valid_samples = []
    for i in range(len(valid_responses)):
        tmp = []
        for j in range(agent_num + 1):
            tmp.append(valid_responses[i][j])
            tmp.append(valid_texts[i][j])
        valid_samples.append(tmp)

    test_texts = [results[i]['implementation_input'] for i in test_index]
    test_responses = [results[i]['responses'] for i in test_index]
    y_test = torch.tensor([results[i]['is_solved'] for i in test_index], dtype=torch.float32)
    test_samples = []
    for i in range(len(test_responses)):
        tmp = []
        for j in range(agent_num + 1):
            tmp.append(test_responses[i][j])
            tmp.append(test_texts[i][j])
        test_samples.append(tmp)

    model, tokenizer, embedding_weights = None, None, None
    
    filename1 = f"{filename.split('/')[0]}/{filename.split('/')[1]}/{filename.split('/')[2]}/Grads/{str(args.random_split)}/{filename.split('/')[3]}/{filename.split('/')[4]}/{filename.split('/')[5]}-train_grad.npy"
    if os.path.exists(filename1):
        train_responses_grads = np.load(filename1).tolist()

        if len(train_responses_grads) != len(train_samples):
            train_responses_grads_orig = train_responses_grads
            train_responses_grads = []
            for i in range(len(train_samples)):
                index = train_index[i]
                index_orig = orig_train_index.index(index)
                train_responses_grads.append(train_responses_grads_orig[index_orig])
    else:
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.float16)
            model.eval()
            embedding_weights = model.get_input_embeddings().weight

        train_responses_grads = []
        for i in range(len(train_samples)):
            tmp = []
            for j in range(agent_num + 1):
                tmp.append(get_gradients([train_samples[i][2 * j], train_samples[i][2 * j + 1]], model, tokenizer, embedding_weights))
            train_responses_grads.append(tmp)

        parent_dir = os.path.dirname(filename1)
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(filename1):
            os.remove(filename1)
        np.save(filename1, train_responses_grads)

    train_responses_grads_normalized = []
    response_scaler_list = [StandardScaler()] * (len(train_responses_grads[0]))
    for i in range(len(train_responses_grads)):
        for j in range(len(response_scaler_list)):
            response_scaler_list[j].fit(np.array(train_responses_grads[i][j]).reshape(1, -1))
    
    train_responses_grads_normalized = []
    for i in range(len(train_responses_grads)):
        tmp = []
        for j in range(len(response_scaler_list)):
            tmp.append(response_scaler_list[j].transform(np.array(train_responses_grads[i][j]).reshape(1, -1)).tolist()[0])
        train_responses_grads_normalized.append(tmp)

    train_responses_grads = train_responses_grads_normalized

    filename1 = f"{filename.split('/')[0]}/{filename.split('/')[1]}/{filename.split('/')[2]}/Grads/{str(args.random_split)}/{filename.split('/')[3]}/{filename.split('/')[4]}/{filename.split('/')[5]}-valid_grad.npy"
    if os.path.exists(filename1):
        valid_responses_grads = np.load(filename1).tolist()

        if len(valid_responses_grads) != len(valid_samples):
            valid_responses_grads_orig = valid_responses_grads
            valid_responses_grads = []
            for i in range(len(valid_samples)):
                index = valid_index[i]
                index_orig = orig_valid_index.index(index)
                valid_responses_grads.append(valid_responses_grads_orig[index_orig])
    else:
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.float16)
            model.eval()
            embedding_weights = model.get_input_embeddings().weight

        valid_responses_grads = []
        for i in range(len(valid_samples)):
            tmp = []
            for j in range(agent_num + 1):
                tmp.append(get_gradients([valid_samples[i][2 * j], valid_samples[i][2 * j + 1]], model, tokenizer, embedding_weights))
            valid_responses_grads.append(tmp)

        parent_dir = os.path.dirname(filename1)
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(filename1):
            os.remove(filename1)
        np.save(filename1, valid_responses_grads)

    valid_responses_grads_normalized = []
    for i in range(len(valid_responses_grads)):
        tmp = []
        for j in range(len(response_scaler_list)):
            tmp.append(response_scaler_list[j].transform(np.array(valid_responses_grads[i][j]).reshape(1, -1)).tolist()[0])
        valid_responses_grads_normalized.append(tmp)

    valid_responses_grads = valid_responses_grads_normalized
    
    filename1 = f"{filename.split('/')[0]}/{filename.split('/')[1]}/{filename.split('/')[2]}/Grads/{str(args.random_split)}/{filename.split('/')[3]}/{filename.split('/')[4]}/{filename.split('/')[5]}-test_grad.npy"
    if os.path.exists(filename1):
        test_responses_grads = np.load(filename1).tolist()
    else:
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.float16)
            model.eval()
            embedding_weights = model.get_input_embeddings().weight

        test_responses_grads = []
        for i in range(len(test_samples)):
            tmp = []
            for j in range(agent_num + 1):
                tmp.append(get_gradients([test_samples[i][2 * j], test_samples[i][2 * j + 1]], model, tokenizer, embedding_weights))
            test_responses_grads.append(tmp)

        parent_dir = os.path.dirname(filename1)
        os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(filename1):
            os.remove(filename1)
        np.save(filename1, test_responses_grads)

    test_responses_grads_normalized = []
    for i in range(len(test_responses_grads)):
        tmp = []
        for j in range(len(response_scaler_list)):
            tmp.append(response_scaler_list[j].transform(np.array(test_responses_grads[i][j]).reshape(1, -1)).tolist()[0])
        test_responses_grads_normalized.append(tmp)

    test_responses_grads = test_responses_grads_normalized

    del model, tokenizer, embedding_weights
    torch.cuda.empty_cache()

    train_input = []
    for i in range(agent_num + 1):
        train_input.append([sample[i] for sample in train_responses_grads])

    model_gp = PULSE(noise=args.noise, jitter=1e-5, mixer_hidden_dim=args.mixer_hidden_dim, mixer_layers=args.mixer_layers, agent_num=agent_num)
    model_gp.fit(train_input, y_train, valid_responses_grads, y_valid, test_responses_grads, y_test, epochs=args.epochs, lr=args.lr)

    preds_test = []
    for grad in test_responses_grads:
        pred = model_gp.predict(grad, return_probs=True)
        preds_test.append(pred)
    
    auroc_test = get_AUROC(y_test.tolist(), preds_test)

    print(f"Test AUROC: {auroc_test}")

    preds = []
    i = 0
    for x_test_response, y_test in zip(preds_test, y_test.tolist()):
        pred = x_test_response
        i += 1
        preds.append({"index": i, "label": y_test, "mean": pred, "variance": pred})

    filename = f"{filename.split('/')[0]}/{filename.split('/')[1]}/{filename.split('/')[2]}/PULSE/{str(args.random_split)}/{filename.split('/')[3]}/{filename.split('/')[4]}/{filename.split('/')[5]}.jsonl"
    
    parent_dir = os.path.dirname(filename)
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(filename):
        os.remove(filename)
    
    with open(filename, "w", encoding="utf-8") as file:
        for pred in preds:
            file.write(json.dumps(pred) + "\n")


# python main_run.py --model Qwen/Qwen2.5-Coder-32B-Instruct --dataset HumanEval --agent_num 2 --device 0,1 --random_split 43 --epochs 200 --lr 0.001 --mixer_hidden_dim 256 --noise 0.001
