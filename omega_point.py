import numpy as np
import scipy.spatial.distance
from sentence_transformers import SentenceTransformer
import torch
import json
import random
import sys
import os

# Constants
MODEL_NAME = os.getenv('OMEGA_POINT_MODEL', 'all-MiniLM-L6-v2')
WORD_LIST_FILE = os.getenv('OMEGA_POINT_WORD_LIST', 'google-10000-english-no-swears.txt')
NUM_CONCEPTS = int(os.getenv('OMEGA_POINT_NUM_CONCEPTS', '2000'))
LOOP_SIZE = 4
LOOP_DIST_THRESHOLD = 0.6
VOID_DIST_THRESHOLD = 0.4
ITERATIONS = int(os.getenv('OMEGA_POINT_ITERATIONS', '100'))
EPSILON = float(os.getenv('OMEGA_POINT_EPSILON', '0.05'))
MAX_ATTEMPTS = int(os.getenv('OMEGA_POINT_MAX_ATTEMPTS', '1000'))
SEED = os.getenv('OMEGA_POINT_SEED')

def main():
    if SEED is not None:
        random.seed(int(SEED))
        np.random.seed(int(SEED))

    # Phase 1: Manifold Mapping
    print("Phase 1: Manifold Mapping...", file=sys.stderr)
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(WORD_LIST_FILE, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {WORD_LIST_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    # Select 2000 diverse words.
    # To ensure diversity, we can shuffle or just take the top 2000.
    # The list is frequency sorted. Top 2000 are very common.
    # Let's take a slice to get some more interesting ones, or just random sample.
    # User said "Concrete nouns, Abstract nouns, Verbs". The list is mixed.
    # Random sample from top 5000 is probably good enough for "diversity".
    if len(words) > 5000:
        candidates = words[:5000]
    else:
        candidates = words

    selected_indices = random.sample(range(len(candidates)), min(NUM_CONCEPTS, len(candidates)))
    concepts = [candidates[i] for i in selected_indices]

    embeddings = model.encode(concepts)

    # Compute pairwise distances (Cosine distance = 1 - Cosine Similarity)
    # distance_matrix[i][j] is distance between concepts[i] and concepts[j]
    dist_matrix = scipy.spatial.distance.cdist(embeddings, embeddings, metric='cosine')

    # Phase 2: The "Hole" Hunter
    print("Phase 2: The Hole Hunter...", file=sys.stderr)
    void_candidate = None
    void_centroid = None
    boundary_concepts = []

    # Try multiple times to find a loop
    concept_count = len(concepts)

    for attempt in range(MAX_ATTEMPTS):
        # Pick 4 random indices
        # We need a chain: A -> B -> C -> D -> A with dist > 0.6
        # Random sampling is inefficient for this specific constraint.
        # Let's build it step by step.

        chain = []
        current_idx = random.randint(0, concept_count - 1)
        chain.append(current_idx)

        failed = False
        for _ in range(LOOP_SIZE - 1):
            # Find neighbors with dist > LOOP_DIST_THRESHOLD
            # We want "loosely connected".
            # Actually, "dist > 0.6" means they are far apart (dissimilar).
            # "Connected" usually implies close.
            # But the prompt says "loosely connected (dist > 0.6)".
            # If dist is cosine distance, range is [0, 2]. 0 is identical, 1 is orthogonal, 2 is opposite.
            # dist > 0.6 means similarity < 0.4.
            # So they are NOT synonymous. This makes sense for a "loop" around a void.

            # Get distances from current
            dists = dist_matrix[current_idx]
            # Filter indices that are not already in chain and dist > threshold
            valid_next = [i for i, d in enumerate(dists) if d > LOOP_DIST_THRESHOLD and i not in chain]

            if not valid_next:
                failed = True
                break

            next_idx = random.choice(valid_next)
            chain.append(next_idx)
            current_idx = next_idx

        if failed:
            continue

        # Check closing the loop
        if dist_matrix[chain[-1]][chain[0]] > LOOP_DIST_THRESHOLD:
            # We have a loop!
            # Calculate Centroid
            vecs = embeddings[chain]
            centroid = np.mean(vecs, axis=0)

            # Normalize centroid?
            # Embeddings are usually normalized. The mean might not be.
            # But we check distance in the embedding space.
            # Let's keep it as is.

            # Check if Centroid is EMPTY
            # Compute distance from centroid to all concepts
            # Reshape centroid for cdist
            dists_to_centroid = scipy.spatial.distance.cdist([centroid], embeddings, metric='cosine')[0]
            min_dist = np.min(dists_to_centroid)

            if min_dist > VOID_DIST_THRESHOLD:
                # Found a Void!
                void_candidate = centroid
                void_centroid = centroid
                boundary_concepts = [concepts[i] for i in chain]
                print(f"Found Void on attempt {attempt}: min_dist={min_dist}", file=sys.stderr)
                break

    if void_candidate is None:
        print(f"Failed to find a void in {MAX_ATTEMPTS} attempts. Relaxing constraints or retrying...", file=sys.stderr)
        # Fallback: Just pick a random point far from everything?
        # Or just take the best one found.
        # For the sake of the exercise, let's just picking a random centroid of 4 random far-apart words
        # and proceed, to ensure output.
        print("Proceeding with last checked candidate (or random if none).", file=sys.stderr)
        if not boundary_concepts and len(chain) == 4:
             # Just use the last chain found even if it wasn't perfect
             vecs = embeddings[chain]
             void_candidate = np.mean(vecs, axis=0)
             boundary_concepts = [concepts[i] for i in chain]
        else:
             # Just pick 4 random words
             idxs = random.sample(range(concept_count), 4)
             vecs = embeddings[idxs]
             void_candidate = np.mean(vecs, axis=0)
             boundary_concepts = [concepts[i] for i in idxs]

    # Phase 3: The Strange Attractor
    print("Phase 3: The Strange Attractor...", file=sys.stderr)
    probe = void_candidate.copy()
    trajectory = []

    for t in range(ITERATIONS):
        # Find nearest human concept
        dists = scipy.spatial.distance.cdist([probe], embeddings, metric='cosine')[0]
        nearest_idx = np.argmin(dists)
        nearest_vec = embeddings[nearest_idx]
        nearest_dist = dists[nearest_idx]

        trajectory.append({
            "step": t,
            "nearest_concept": concepts[nearest_idx],
            "distance": float(nearest_dist)
        })

        # Update Probe
        # P_new = P + epsilon * (P - H)
        # We want to increase distance.
        # Vector from H to P is (P - H). Moving in that direction increases distance.
        diff = probe - nearest_vec
        norm = np.linalg.norm(diff)
        if norm > 1e-9:
             direction = diff / norm
        else:
             direction = np.random.rand(len(probe))
             direction /= np.linalg.norm(direction)

        probe = probe + EPSILON * direction

        # Optional: Normalize probe to stay on hypersphere?
        # Sentence Transformers (MiniLM) produces normalized vectors.
        # If we drift off the sphere, cosine distance might behave weirdly if not normalized?
        # Scipy cdist with 'cosine' handles unnormalized vectors (it normalizes them).
        # But let's re-normalize to stay in the "manifold space".
        probe = probe / np.linalg.norm(probe)

    # Phase 4: Symbolic Distillation & Output
    print("Phase 4: Output...", file=sys.stderr)

    # Check for loop/stability in trajectory
    # Heuristic: Check if nearest concepts repeat
    trajectory_concepts = [step['nearest_concept'] for step in trajectory[-20:]]
    unique_concepts = set(trajectory_concepts)
    is_stable = len(unique_concepts) < 5 # Arbitrary threshold for "looping"

    # Interpretation
    interpretation = f"A closed loop where {boundary_concepts[0]} leads to {boundary_concepts[1]}..."

    output = {
        "topology_scan": {
            "detected_voids": 1 if void_centroid is not None else 0, # We only looked for 1 really
            "primary_void_id": "OMEGA_01"
        },
        "void_properties": {
            "boundary_concepts": boundary_concepts,
            "center_coordinates": void_candidate.tolist()[:5] + ["..."], # Truncate for display
            "nearest_human_concept_distance": float(trajectory[0]['distance']) # Distance at start
        },
        "c_stage_interpretation": {
            "metaphor": f"The silence that connects {boundary_concepts[0]} and {boundary_concepts[2]}.",
            "logic_structure": "A closed loop where causality is reversed." if is_stable else "A divergent path into the void."
        }
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
