import torch
import numpy as np
import glob
import heapq

def _get_topk_activations(feat_idx, k, latent_vector_files):
    #TODO: change to min-heap (sorting is expensive)
    # First loop - get top k activations across all files
    top_k = []
    for latent_file in latent_vector_files:
        loaded_vectors = torch.load(latent_file, weights_only=True)
        values, ind = torch.topk(loaded_vectors[:, feat_idx], k)
        
        # Add all values from this file
        for idx, val in zip(ind, values):
            top_k.append((idx.item(), val.item(), latent_file))
        
        # Keep only top k overall after processing each file
        top_k.sort(key=lambda x: x[1], reverse=True)
        if len(top_k) > k:
            top_k = top_k[:k]

    minibatch_size = loaded_vectors.shape[0]

    return top_k, minibatch_size

def _get_topk_context(feat_idx, k, top_k, context_len, minibatch_size, act_id_dataset, tokenizer, verbose=True):
    # Second loop - get context for top k activations
    if verbose:
        print(f"Top {k} activations for feature index {feat_idx} across all files:")
    
    examples_dict = {}
    for i, (idx, val, latent_fname) in enumerate(top_k):
        # Determine the corresponding last3 file
        batch_number = int(latent_fname.split('_batch_')[1].split('_')[0])
        minibatch_number = int(latent_fname.split('minibatch_')[1].split('.')[0])
        idx = idx + minibatch_number * minibatch_size # adjusting the index to a global offset
        last3_fname = act_id_dataset.replace('*', f"{batch_number:04d}")

        # Load the last3 file
        last3_data = np.load(last3_fname)

        # Get sentence index and token index for target token
        sent_idx, tok_idx, token = last3_data[idx]  # sent_idx, token_idx, token

        # Get context window indices, staying within the same sentence
        start_idx = max(0, idx - context_len)
        end_idx = min(len(last3_data), idx + context_len + 1)

        # Adjust window to stay within the same sentence
        context_window = last3_data[start_idx:end_idx]
        context_window = context_window[context_window[:, 0] == sent_idx]  # Match sent_idx

        # Extract tokens and activations
        lt = torch.load(latent_fname, weights_only=True)
        min_tk_i = context_window[0, 1]
        max_tk_i = context_window[-1, 1]
        context = lt[(lt[:, -2] >= min_tk_i) & (lt[:, -2] <= max_tk_i) & (lt[:, -3] == sent_idx)]
        context_activations = context[:, feat_idx]

        # Decode tokens
        context_tokens = context_window[:, 2].tolist()
        text = tokenizer.decode(context_tokens)
        decoded_tokens = tokenizer.convert_ids_to_tokens(context_tokens)
        target_token_txt = tokenizer.decode([int(token)])

        # Print details
        clean_tokens_list = [token.replace('Ġ', ' ') for token in decoded_tokens]
        np_activations = context_activations.cpu().numpy()
        # text = ''.join(clean_tokens_list)

        if verbose:
            print(f"File: {latent_fname}")
            print(f"Activation value: {val}") # {val:.4f}")
            print(f"Target token: '{target_token_txt}'")# (index {rel_idx})")
            text1 = text.replace('\n', ' ')
            print(f"Context: {text1}")
            print("")
        
        examples_dict[i] = {
            "file": latent_fname,
            "target_token": target_token_txt,
            "target_token_value": val,
            "context": text,
            "token_list": clean_tokens_list,
            "activations": np_activations.tolist()
        }
    
    return examples_dict

def _get_topk_activations_list(feat_idx_list, k, latent_vector_files):
    # Create a separate min-heap for each feature index.
    heaps = {feat_idx: [] for feat_idx in feat_idx_list}
    minibatch_size = None

    for latent_file in latent_vector_files:
        loaded_vectors = torch.load(latent_file, weights_only=True)
        if minibatch_size is None:
            minibatch_size = loaded_vectors.shape[0]
        for feat_idx in feat_idx_list:
            # Get top k activations for this feature in the current file.
            values, indices = torch.topk(loaded_vectors[:, feat_idx], k)
            for idx, val in zip(indices, values):
                entry = (val.item(), idx.item(), latent_file)
                heap = heaps[feat_idx]
                if len(heap) < k:
                    heapq.heappush(heap, entry)
                else:
                    if entry[0] > heap[0][0]:
                        heapq.heapreplace(heap, entry)
    # For each feature, sort the heap (largest first)
    topk_dict = {}
    for feat_idx in feat_idx_list:
        topk_dict[feat_idx] = sorted(heaps[feat_idx], key=lambda x: x[0], reverse=True)
    return topk_dict, minibatch_size

def _get_topk_context_list(feat_idx_list, k, topk_dict, context_len, minibatch_size, act_id_dataset, tokenizer, verbose=True):
    # Organize results as:
    # top_examples[feat_idx][ordered_ex_ix] = {file, target_token, target_token_value, context, token_list, activations}
    examples_dict = {feat_idx: {} for feat_idx in feat_idx_list}
    
    # Build a mapping from latent filename to all top-k entries from all features.
    # For each feature, we also record the order (which corresponds to the sorted order from topk_dict).
    file_to_entries = {}
    for feat_idx in feat_idx_list:
        for order, (activation, local_idx, latent_fname) in enumerate(topk_dict[feat_idx]):
            file_to_entries.setdefault(latent_fname, []).append((feat_idx, activation, local_idx, order))
    
    # Process each unique latent file once.
    for latent_fname, entries in file_to_entries.items():
        # Parse batch and minibatch numbers from the latent filename.
        # Example filename format: "latent_vectors_batch_XXXX_minibatch_YYY.pt"
        batch_number = int(latent_fname.split('_batch_')[1].split('_')[0])
        minibatch_number = int(latent_fname.split('minibatch_')[1].split('.')[0])
        last3_fname = act_id_dataset.replace('*', f"{batch_number:04d}")
        
        # Load the latent file and its corresponding last3 file (only once).
        lt = torch.load(latent_fname, weights_only=True)
        last3_data = np.load(last3_fname)
        
        for feat_idx, activation, local_idx, order in entries:
            # Adjust to global index.
            global_idx = local_idx + minibatch_number * minibatch_size
            
            # Retrieve sentence and token info.
            sent_idx, tok_idx, token = last3_data[global_idx]
            start_idx = max(0, global_idx - context_len)
            end_idx = min(len(last3_data), global_idx + context_len + 1)
            context_window = last3_data[start_idx:end_idx]
            # Keep only tokens from the same sentence.
            context_window = context_window[context_window[:, 0] == sent_idx]
            
            # Extract context activations from the latent file.
            min_tk_i = context_window[0, 1]
            max_tk_i = context_window[-1, 1]
            mask = (lt[:, -2] >= min_tk_i) & (lt[:, -2] <= max_tk_i) & (lt[:, -3] == sent_idx)
            context = lt[mask]
            context_activations = context[:, feat_idx]
            
            # Decode tokens.
            context_tokens = context_window[:, 2].tolist()
            text = tokenizer.decode(context_tokens)
            decoded_tokens = tokenizer.convert_ids_to_tokens(context_tokens)
            target_token_txt = tokenizer.decode([int(token)])
            clean_tokens_list = [t.replace('Ġ', ' ') for t in decoded_tokens]
            np_activations = context_activations.cpu().numpy()
            
            if verbose:
                print(f"Feature: {feat_idx} | Order: {order} | File: {latent_fname}")
                print(f"Activation value: {activation}")
                print(f"Target token: '{target_token_txt}'")
                t = text.replace('\n', ' ')
                print(f"Context: {t}")
                print("")
            
            examples_dict[feat_idx][order] = {
                "file": latent_fname,
                "target_token": target_token_txt,
                "target_token_value": activation,
                "context": text,
                "token_list": clean_tokens_list,
                "activations": np_activations.tolist()
            }
    return examples_dict

def top_activations(feat_idx_list, k, context_len, latent_dataset_path, dataset_slice, act_id_dataset, tokenizer, verbose=True):
    a, b = dataset_slice
    latent_vector_files = sorted(glob.glob(latent_dataset_path))[a:b]
    topk_dict, minibatch_size = _get_topk_activations_list(feat_idx_list, k, latent_vector_files)
    top_examples = _get_topk_context_list(feat_idx_list, k, topk_dict, context_len, minibatch_size, act_id_dataset, tokenizer, verbose)
    return top_examples



def extract_features(positive_prompts, negative_prompts, tokenizer, llm_model, sae_model):
    prompts = positive_prompts + negative_prompts

    activation_cache = []
    def save_activations_hook(module, input, output):
        # input is a tuple; input[0] is the tensor we need
        activation_cache.append(input[0].cpu().detach().numpy())

    # Register hook on the 16th layer
    layer_index = 15  # Zero-based index; 15 corresponds to the 16th layer
    hook_handle = llm_model.model.layers[layer_index].register_forward_hook(save_activations_hook)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenize sentences
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=35, # can equal to the length (in tokens) of the longest sentence
                    # limit suggested due to VRAM constraints
        padding="max_length",
    ).to(device)

    # Forward pass
    with torch.no_grad():
        llm_model(**inputs)

    activations = np.array(activation_cache)
    # print(f"Activations shape: {activations.shape}") # (1, num_sent, seq_len, 3072)

    # Reshape activations to (num_sent*seq_len, 3072)
    activations = activations.squeeze(0)  # Remove batch dimension
    num_sent, seq_len, hidden_dim = activations.shape
    activations = activations.reshape(-1, hidden_dim)

    # Build index tensor for sentence separation
    sent_idx_tensor = torch.tensor([[i] * seq_len for i in range(num_sent)]).flatten()

    # Get attention mask and remove padding tokens
    attention_mask = inputs['attention_mask'].view(-1).cpu().numpy()
    activations = activations[attention_mask == 1]
    sent_idx_tensor = sent_idx_tensor[attention_mask == 1]

    activations = torch.tensor(activations, dtype=torch.float32)  # Convert to torch tensor

    # Get latent vectors for sentences
    with torch.no_grad():
        _, encoded = sae_model(activations)
        latent_vector = encoded.cpu().numpy()

    # Remove hook to prevent side effects
    hook_handle.remove()
    
    return latent_vector, sent_idx_tensor


def prompt_search_mean_local(positive_prompts, negative_prompts, top_k, tokenizer, llm_model, sae_model, verbose=True):
    latent_vector, sent_idx_tensor = extract_features(positive_prompts, negative_prompts, tokenizer, llm_model, sae_model)

    # Positive prompts
    pos_ix = np.where(sent_idx_tensor < len(positive_prompts))[0]
    pos_latent = latent_vector[pos_ix]
    # Apply min-max normalization to latent vectors
    latent_vector_normalized = (pos_latent - np.min(pos_latent, axis=0)) / (np.max(pos_latent, axis=0) - np.min(pos_latent, axis=0) + 1e-8)
    # Get mean activation for each feature across tokens
    mean_activation = np.mean(latent_vector_normalized, axis=0)

    # Negative prompts
    if len(negative_prompts) > 0:
        neg_ix = np.where(sent_idx_tensor >= len(positive_prompts))[0]
        neg_latent = latent_vector[neg_ix]
        # Apply min-max normalization to latent vectors
        neg_latent_normalized = (neg_latent - np.min(neg_latent, axis=0)) / (np.max(neg_latent, axis=0) - np.min(neg_latent, axis=0) + 1e-8)
        # Get mean activation for each feature across tokens
        neg_mean_activation = np.mean(neg_latent_normalized, axis=0)
        # Subtract negative mean activation
        mean_activation -= neg_mean_activation

    # Get top k features and their mean activation values
    top_k = 15
    top_k_indices = np.argsort(mean_activation)[-top_k:][::-1]
    top_k_values = mean_activation[top_k_indices]

    if verbose:
        print(f"Top {top_k} features and their mean activation values:")
        for idx, val in zip(top_k_indices, top_k_values):
            print(f"Feature {idx}: {val:.4f}")
    
    return top_k_indices, top_k_values

def prompt_search_rank(positive_prompts, negative_prompts, topN, tokenizer, llm_model, sae_model, verbose=True):
    
    latent_vector, sent_idx_tensor = extract_features(positive_prompts, negative_prompts, tokenizer, llm_model, sae_model)

    pos_ix = np.where(sent_idx_tensor < len(positive_prompts))[0]
    pos_latent = latent_vector[pos_ix]
    neg_ix = np.where(sent_idx_tensor >= len(positive_prompts))[0]
    neg_latent = latent_vector[neg_ix]

    # 1. Choose how large the "top slice" is. 
    #    For example, top_frac = 0.05 means top 5% of features for each token.
    #    Or use top_k = 100 to pick top 100 features per token.
    top_frac = 0.05 # at 65k features, this is 3250 features
    num_features = pos_latent.shape[1]  # same as neg_latent.shape[1]

    top_k = int(num_features * top_frac)  # number of features to pick per token
    top_k = max(top_k, 1)                # ensure at least 1
    if topN > top_k:
        top_k = topN # obey user's request

    # 2. Count how often each feature appears in that top slice for positive vs. negative.

    pos_count = np.zeros(num_features, dtype=np.int64)
    for row in pos_latent:
        # row shape: (num_features,)
        # get indices of top_k features in that row
        # argsort returns ascending, so we take last top_k
        top_indices = np.argpartition(row, -top_k)[-top_k:]
        # increment counts
        pos_count[top_indices] += 1

    neg_count = np.zeros(num_features, dtype=np.int64)
    for row in neg_latent:
        top_indices = np.argpartition(row, -top_k)[-top_k:]
        neg_count[top_indices] += 1

    # 3. Convert counts to frequencies (how often a feature is in top_k%).
    pos_freq = pos_count / pos_latent.shape[0]
    neg_freq = neg_count / neg_latent.shape[0] if len(negative_prompts) > 0 else neg_count / 1.0

    # 4. Score = difference or ratio, e.g. difference:
    score = pos_freq - neg_freq

    # 5. Sort by score descending
    sorted_features = np.argsort(score)[::-1]

    if verbose:
        # topN = 15
        print(f"Top {topN} features and their scores:")
        for i in range(topN):
            f_idx = sorted_features[i]
            print(f"Feature {f_idx} => pos_freq={pos_freq[f_idx]:.4f}, neg_freq={neg_freq[f_idx]:.4f}, score={score[f_idx]:.4f}")
    
    return sorted_features[:topN], score[sorted_features[:topN]]


def influence(feat_idx, multiplier, start_sent, num_tok, tokenizer, llm_model, sae_model, verbose=True):
    # Generate artificial latent vector, pass through SAE decoder, and boost it

    hidden_dim = sae_model.encoder.out_features
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Boosted activations setup
    artificial_latent_vector = np.zeros(hidden_dim)
    artificial_latent_vector[feat_idx] = 1

    # if not 0 print
    if verbose:
        if multiplier != 0:
            print(f"{multiplier} boost on feature {feat_idx}")
        else:
            print("No boost")

    # Decode to get activations
    with torch.no_grad():
        artificial_latent_vector_tensor = torch.tensor(artificial_latent_vector, dtype=torch.float32).unsqueeze(0)
        reconstructed_activations = sae_model.decoder(artificial_latent_vector_tensor)
        boosted_activations = reconstructed_activations.to(torch.float16) * multiplier

    # Hook function to add boosted activations to all tokens
    def influence_hook(module, input, output):
        if isinstance(output, tuple):  # Handle tuple output
            boosted_output = output[0] + boosted_activations.to(device)
            return (boosted_output,) + output[1:]
        else:
            return output + boosted_activations.to(device)

    layer_index = 15  # Target layer for injection

    try:
        # Register the hook
        hook_handle = llm_model.model.layers[layer_index].register_forward_hook(influence_hook)

        # Text generation loop
        inputs = tokenizer(start_sent, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = inputs["input_ids"]

        generated_ids = input_ids.clone()

        for _ in range(num_tok):
            with torch.no_grad():
                outputs = llm_model(input_ids=generated_ids)

                # Maximum likelihood sampling
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0) 

                # Stop generation if EOS or padding token is predicted
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                # Append the predicted token ID
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # Decode the generated sequence
        generated_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    finally:
        # Remove the hook to prevent side effects
        hook_handle.remove()
    
    if verbose:
        print(f"Generated text with influence: '{generated_text}'")
    return generated_text


def cosine_similarity_search(feat_idx, top_k, latent_vector_files, verbose=True):
    # Dictionary to accumulate global similarity scores for each feature
    global_similarities = {}

    # Process each chunk
    for latent_file in latent_vector_files:
        loaded_vectors = torch.load(latent_file, weights_only=True)[:,:-3]  # Load current chunk

        # Get the column corresponding to the target feature index
        target_feature = loaded_vectors[:, feat_idx]

        # Compute similarity with all other features (columns)
        similarities = torch.nn.functional.cosine_similarity(
            target_feature.unsqueeze(0),  # Make it 2D for broadcasting
            loaded_vectors.T,  # Transpose for feature-wise comparison
            dim=1
        )

        # Accumulate similarity scores globally
        for i, similarity in enumerate(similarities):
            if i not in global_similarities:
                global_similarities[i] = 0.0
            global_similarities[i] += similarity.item()

    # Identify the top-k features globally
    top_k_features = heapq.nlargest(top_k, global_similarities.items(), key=lambda x: x[1])

    if verbose:
        # Display results
        for idx, agg_similarity in top_k_features[:25]:
            print(f"Feature {idx} has aggregated similarity {agg_similarity:.4f}")

    return top_k_features  # Returns a list of (feature_index, aggregated_similarity) tuples
