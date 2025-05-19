import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree

import networkx as nx
import os
import pandas as pd
import gc
import argparse

# check if Cuda is initialized before the script
torch.cuda.init()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA initialized. Using device: {torch.cuda.get_device_name(0)}")

# ============= CONFIGURATION =============
START_BATCH = 0

CHECKPOINT_DIR = "/home/arikan/ml_cuda/gnn/checkpoints" 

DATA_DIRS = [
    "/home/arikan/ml_cuda/inputs/edited/batch_01/processed_files",
    "/home/arikan/ml_cuda/inputs/edited/batch_02",
    "/home/arikan/ml_cuda/inputs/edited/batch_03/processed_files",
    "/home/arikan/ml_cuda/inputs/edited/batch_04/processed_files",
    "/home/arikan/ml_cuda/inputs/edited/batch_06",
]

CUDA_TIMES_FILE = "/home/arikan/ml_cuda/timings_only.csv"

VALIDATION_NAMES = [
    "bio-SC-TS-edited.edges",
    "bio-SC-LC-edited.edges",
    "bio-SC-HT-edited.edges",
    "bio-SC-GT-edited.edges",
    "bn-fly-drosophila_medulla_1-edited.edges",
    "bn-macaque-rhesus_cerebral-cortex_1-edited.edges",
    "ca-IMDB-edited.edges",
    "com-orkut.ungraph-edited.txt",
    "email-Enron-edited.txt",
    "roadNet-CA-edited.txt",
    "ca-MathSciNet_sym0.mtx",
    "ca-netscience_sym0.mtx",
    "luxembourg_osm_sym0.mtx",
    "road_central_sym0.mtx",
    "germany_osm_sym0.mtx",
    "graph500-scale19-ef16_adj-edited.edges",
    "rec-amazon-ratings-edited.edges",
    "rec-dating-edited.edges",
    "yahoo-msg-edited.txt",
    "musae_facebook_edges-edited.txt",
    "sc-nasasrb_sym0.mtx",
    "web-baidu-baike-edited.edges",
    "web-Stanford_sym0.mtx",
]

# MAX_NODES_THRESHOLD = 1000000  # Skip graphs with more than 1 million nodes
MAX_EDGES_THRESHOLD = 40000000  # Skip graphs with more than 40-50 million edges

class GNNCUDAPredictor(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNNCUDAPredictor, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 5)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

def load_edge_list(file_path):
    """Load edge list from a text file and convert to networkx graph."""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        is_mtx = file_path.endswith('.mtx')
        header_read = False
        
        for line in f:
            # Skip comments
            if line.startswith('%') or line.startswith('#'):
                continue
            if not line.strip():
                continue
                
            # For MTX files, skip the header line (contains dimensions)
            if is_mtx and not header_read:
                header_read = True
                continue
                
            # Parse edge data
            parts = line.strip().split()
            if len(parts) >= 2:  # Ensure at least 2 values exist
                source, target = map(int, parts[:2])  # Take only first two values
                G.add_edge(source, target)
    
    return G

def load_from_directory(data_dir, cuda_times_file, validation_names=None, for_validation=False):
    """Load datasets from a directory, optionally filtering for validation or training."""
    
    df = pd.read_csv(cuda_times_file)
    graphs = {}
    
    print(f"\nProcessing directory: {data_dir}")
    
    try:
        all_items = os.listdir(data_dir)
        graph_files = [f for f in all_items 
                      if os.path.isfile(os.path.join(data_dir, f)) and 
                      (f.endswith('.edges') or f.endswith('.txt') or f.endswith('.mtx'))]
        
        for filename in graph_files:
            # Skip if we're loading training data and this is a validation dataset
            if not for_validation and validation_names and filename in validation_names:
                continue
            # Skip if we're loading validation data and this isn't a validation dataset
            if for_validation and validation_names and filename not in validation_names:
                continue
                
            try:
                file_path = os.path.join(data_dir, filename)
                print(f"Loading {filename}...")
                
                G = load_edge_list(file_path)
                edge_count = G.number_of_edges() 
                print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                
                # Skip if graph exceeds size thresholds
                if edge_count > MAX_EDGES_THRESHOLD:
                    print(f"Skipping {filename}: Graph exceeds size threshold ({edge_count} edges)")
                    continue
                
                graphs[filename] = G
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                
    except Exception as e:
        print(f"Error accessing directory {data_dir}: {str(e)}")
    
    # Create cuda_times dictionary
    cuda_times = {}
    block_sizes = [64, 128, 256, 512, 1024]
    
    for _, row in df.iterrows():
        dataset_name = row['Dataset']
        if dataset_name not in graphs:
            continue
        for block_size in block_sizes:
            time_col = f'Time_{block_size}'
            if time_col in row:
                cuda_times[(dataset_name, block_size)] = float(row[time_col])
    
    return graphs, cuda_times

def convert_nx_to_pyg(G, label_idx=None):
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.
    
    Args:
        G (networkx.Graph): Input graph
        label_idx (int, optional): Class label index
        
    Returns:
        torch_geometric.data.Data: PyG Data object
    """
    # Create edge index
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    
    # Create node features (using degree)
    deg = degree(edge_index[0], num_nodes=G.number_of_nodes())
    x = deg.view(-1, 1).float()  # Ensure float type for features
    
    # Create labels tensor if provided
    if label_idx is not None:
        y = torch.tensor([label_idx], dtype=torch.long)
    else:
        y = None
    
    return Data(x=x, edge_index=edge_index, y=y)

def save_training_state(model, optimizer, epoch, batch_id, save_dir=CHECKPOINT_DIR):
    """Save model and optimizer state."""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch_id': batch_id,
    }
    torch.save(state, os.path.join(save_dir, f'checkpoint_may15_batch_{batch_id}.pt'))

def load_training_state(model, optimizer, batch_id, save_dir=CHECKPOINT_DIR):
    """Load saved model and optimizer state."""
    # For batch 0, use a fresh model
    if batch_id == 0:
        print(f"Training batch 0. Starting with a fresh model.")
        return model, optimizer, 0
    
    # For batch 1 or higher, load the previous batch's checkpoint
    prev_batch = batch_id - 1
    checkpoint_path = os.path.join(save_dir, f'checkpoint_may15_batch_{prev_batch}.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from batch {prev_batch}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded successfully.")
        return model, optimizer, start_epoch
    else:
        print(f"No checkpoint found for batch {prev_batch}. Starting with a fresh model.")
    
    return model, optimizer, 0

def clear_memory():
    """Clear CUDA cache and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_batch_size_for_graph(graph_size):
    """Determine batch size based on graph size"""
    if graph_size > 5000000:  # > 5M nodes
        return 8
    elif graph_size > 1000000:  # > 1M nodes
        return 16  # Very small batch size for extremely large graphs
    elif graph_size > 500000:  # > 500K nodes
        return 32
    elif graph_size > 100000:  # > 100K nodes
        return 64
    else:   #  100K > nodes
        return 128
    # else:
    #     return 1  # Larger batch size for small graphs

def train_model_with_neighbor_sampling(model, data_dirs, cuda_times_file, validation_names, 
                                      num_epochs=5, start_batch=0):
    """Train model using NeighborLoader with all neighbors for maximum information."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Changed enumeration to handle both multi-batch and single-batch cases
    for i, data_dir in enumerate(data_dirs):
        batch_id = start_batch + i  # Use this instead of the enumeration's batch_id
        print(f"\nProcessing batch {batch_id}: {data_dir}")
        
        # Load model state and data
        model, optimizer, start_epoch = load_training_state(model, optimizer, batch_id)
        model = model.to(device)
        graphs, cuda_times = load_from_directory(data_dir, cuda_times_file, validation_names)
        print(f"Loaded {len(graphs)} graphs from batch {batch_id}")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            model.train()
            total_loss, total = 0, 0
            
            # Process graphs one by one
            for graph_idx, (graph_name, G) in enumerate(graphs.items()):
                try:
                    print(f"Processing graph {graph_idx+1}/{len(graphs)}: {graph_name}")
                    
                    # Get optimal block size (ground truth label)
                    block_sizes = [64, 128, 256, 512, 1024]
                    times = [cuda_times[(graph_name, size)] for size in block_sizes]
                    optimal_size_idx = times.index(min(times))
                    
                    # Convert graph to PyG format
                    data = convert_nx_to_pyg(G, optimal_size_idx)
                    batch_size = get_batch_size_for_graph(G.number_of_nodes())
                    
                    # Try to sample all neighbors, fall back only if absolutely necessary
                    try:
                        loader = NeighborLoader(
                            data, 
                            num_neighbors=[-1, -1, -1],  # Sample ALL neighbors at each hop
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=4
                        )
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"  Memory error with full sampling for {graph_name}, skipping...")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            print(f"  Error with {graph_name}: {e}")
                            continue
                    
                    # Mini-batch training on this graph
                    graph_loss, graph_batches = 0, 0
                    
                    for batch in loader:
                        try:
                            batch = batch.to(device)
                            optimizer.zero_grad()
                            
                            out = model(batch.x, batch.edge_index, batch.batch)
                            target = torch.tensor([optimal_size_idx], dtype=torch.long).to(device)
                            
                            loss = criterion(out, target)
                            loss.backward()
                            optimizer.step()
                            
                            graph_loss += loss.item()
                            graph_batches += 1
                            
                            del batch, out, loss, target
                            torch.cuda.empty_cache()
                            
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                torch.cuda.empty_cache()
                                continue
                            else:
                                break
                    
                    # Update metrics
                    if graph_batches > 0:
                        total_loss += graph_loss / graph_batches
                        total += 1
                        print(f"  Completed {graph_batches} batches for {graph_name}")
                    else:
                        print(f"  No successful batches for {graph_name}")
                    
                    del data, loader
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing graph {graph_name}: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            # Print epoch results and save checkpoint
            epoch_loss = total_loss / total if total > 0 else 0
            print(f'Batch {batch_id}, Epoch {epoch+1}/{num_epochs}, '
                  f'Loss: {epoch_loss:.4f}, Processed graphs: {total}/{len(graphs)}')
            save_training_state(model, optimizer, epoch + 1, batch_id)
        
        # Clean up batch memory
        del graphs, cuda_times
        torch.cuda.empty_cache()
        gc.collect()
        
    return model

def evaluate_model_with_neighbor_sampling(model, data_dirs, cuda_times_file, validation_names):
    """Evaluate model using NeighborLoader with all neighbors for validation graphs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = {}
    correct = 0
    total = 0
    
    # Load validation graphs from each directory
    for data_dir in data_dirs:
        graphs, cuda_times = load_from_directory(data_dir, cuda_times_file, 
                                              validation_names, for_validation=True)
        
        if not graphs:
            continue
            
        # Process one validation graph at a time
        for graph_name, G in graphs.items():
            try:
                print(f"Evaluating {graph_name}...")
                
                # Get ground truth label
                block_sizes = [64, 128, 256, 512, 1024]
                times = [cuda_times[(graph_name, size)] for size in block_sizes]
                optimal_size_idx = times.index(min(times))
                
                with torch.no_grad():
                    # Convert graph to PyG format
                    data = convert_nx_to_pyg(G, optimal_size_idx)
                    batch_size = get_batch_size_for_graph(G.number_of_nodes())
                    
                    # Create NeighborLoader with all neighbors
                    try:
                        loader = NeighborLoader(
                            data,
                            num_neighbors=[-1, -1, -1],  # Sample all neighbors
                            batch_size=batch_size,
                            shuffle=False  # No need to shuffle for evaluation
                        )
                        
                        # Collect predictions from multiple batches
                        batch_predictions = []
                        max_batches = 30  # Limit number of batches for very large graphs
                        
                        for batch_idx, batch in enumerate(loader):
                            if batch_idx >= max_batches:
                                break
                                
                            try:
                                batch = batch.to(device)
                                out = model(batch.x, batch.edge_index, batch.batch)
                                pred = out.argmax(dim=1).item()
                                batch_predictions.append(pred)
                                
                                del batch, out
                                torch.cuda.empty_cache()
                                
                            except RuntimeError:
                                torch.cuda.empty_cache()
                                continue
                        
                        # Use majority voting for final prediction
                        if batch_predictions:
                            from collections import Counter
                            counter = Counter(batch_predictions)
                            pred = counter.most_common(1)[0][0]
                            
                            # Check if prediction matches ground truth
                            if pred == optimal_size_idx:
                                correct += 1
                            total += 1
                            
                            # Store prediction
                            predicted_block_size = block_sizes[pred]
                            actual_block_size = block_sizes[optimal_size_idx]
                            all_predictions[graph_name] = {
                                'predicted': predicted_block_size,
                                'actual': actual_block_size
                            }
                        else:
                            print(f"  No successful predictions for {graph_name}")
                            
                    except Exception as e:
                        print(f"  Error processing {graph_name}: {e}")
                        torch.cuda.empty_cache()
                        continue
                        
                    # Clean up
                    del data, loader
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error evaluating {graph_name}: {str(e)}")
                torch.cuda.empty_cache()
                continue
    
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"\nValidation Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Print individual predictions
    print("\nPredictions for each validation dataset:")
    for name, result in all_predictions.items():
        print(f"{name}:")
        print(f"  Predicted block size: {result['predicted']}")
        print(f"  Actual block size: {result['actual']}")
    
    return accuracy, all_predictions

if __name__ == "__main__":
    
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Train or Evaluate GNN CUDA Predictor')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True,
                       help='Mode to run: train or evaluate')
    parser.add_argument('--batch', type=int, default=-1, 
                       help='Batch to train (-1 for all batches, 0-N for specific batch)')
    args = parser.parse_args()

    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current GPU device:", torch.cuda.current_device())
        print("GPU device name:", torch.cuda.get_device_name(0))
    
    # Initialize model
    num_node_features = 1
    model = GNNCUDAPredictor(num_node_features, hidden_channels=128)
    
    if args.mode == 'train':
        if args.batch >= 0:
            # Train single batch
            print(f"\nTraining batch {args.batch} only")
            if args.batch >= len(DATA_DIRS):
                print(f"Error: Batch {args.batch} is out of range. Max batch is {len(DATA_DIRS)-1}")
                exit(1)
                
            single_batch_dirs = [DATA_DIRS[args.batch]]
            model = train_model_with_neighbor_sampling(
                model=model,
                data_dirs=single_batch_dirs,
                cuda_times_file=CUDA_TIMES_FILE,
                validation_names=VALIDATION_NAMES,
                num_epochs=5,
                start_batch=args.batch  
            )
            print(f"\nCompleted training batch {args.batch}")
            
        else:
            # Train all batches
            print(f"\nTraining all batches starting from {START_BATCH}")
            model = train_model_with_neighbor_sampling(
                model=model,
                data_dirs=DATA_DIRS,
                cuda_times_file=CUDA_TIMES_FILE,
                validation_names=VALIDATION_NAMES,
                num_epochs=5,
                start_batch=START_BATCH
            )
            print("\nCompleted training all batches")
    
    elif args.mode == 'evaluate':
        # Load the latest checkpoint
        latest_batch = len(DATA_DIRS) - 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model, _, _ = load_training_state(model, optimizer, latest_batch)
        
        print("\nRunning evaluation...")
        accuracy, predictions = evaluate_model_with_neighbor_sampling(
            model=model,
            data_dirs=DATA_DIRS,
            cuda_times_file=CUDA_TIMES_FILE,
            validation_names=VALIDATION_NAMES
        )