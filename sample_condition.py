import argparse
import math
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem

# FlowMol imports
from propmolflow.models.flowmol import FlowMol
from propmolflow.analysis.metrics import SampleAnalyzer
from propmolflow.analysis.propmolflow_metrics import compute_all_standard_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate molecules with PropMolFlow.")
    
    # Model checkpoint
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    
    # Sampling parameters
    parser.add_argument("--n_mols", type=int, default=1000,
                        help="Number of molecules to sample")
    parser.add_argument("--max_batch_size", type=int, default=128,
                        help="Maximum batch size for sampling")
    parser.add_argument("--n_atoms_per_mol", type=int, default=None,
                        help="Number of atoms per molecule (if fixed)")
    parser.add_argument("--n_timesteps", type=int, default=100,
                        help="Number of timesteps for sampling")
    
    # Trajectory and metrics options
    parser.add_argument("--xt_traj", action="store_true",
                        help="Store x_t trajectory")
    parser.add_argument("--ep_traj", action="store_true",
                        help="Store episode trajectory")
    
    # Stochasticity and thresholds
    parser.add_argument("--stochasticity", type=float, default=None,
                        help="Stochasticity parameter")
    parser.add_argument("--hc_thresh", type=float, default=None,
                        help="High confidence threshold")
    
    # Property conditioning
    parser.add_argument("--properties_for_sampling", type=float, default=None,
                        help="Property value for conditioning")
    parser.add_argument("--training_mode", action="store_true",
                        help="Use training mode")
    parser.add_argument("--property_name", type=str, default=None,
                        help="Property name")
    parser.add_argument("--normalization_file_path", type=str, default=None,
                        help="Path to property normalization file")
    parser.add_argument("--properties_handle_method", type=str, default="concatenate_sum",
                        help="Method to handle properties")
    parser.add_argument("--multilple_values_file", type=str, default=None,
                        help="Path to numpy file containing multiple property values")
    parser.add_argument("--number_of_atoms", type=str, default=None,
                        help="Path to numpy file containing number of atoms in the molecule")

    # Output
    parser.add_argument("--output_file", type=str, default="sampled_molecules.sdf",
                        help="Output SDF file path")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze generated molecules")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (or -1 for CPU)")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_checkpoint}")
    model = FlowMol.load_from_checkpoint(args.model_checkpoint)
    model.to(device)
    model.eval()
    
    # Load multiple property values if specified
    multilple_values_to_one_property, number_of_atoms = None, None
    if args.multilple_values_file is not None:
        print(f"Loading multiple property values from {args.multilple_values_file}")
        multilple_values_to_one_property = np.load(args.multilple_values_file).tolist()[:args.n_mols]
    if args.number_of_atoms is not None:
        print(f"Loading number of atoms from {args.number_of_atoms}")
        number_of_atoms = np.load(args.number_of_atoms).tolist()[:args.n_mols]
    
    n_batches = math.ceil(args.n_mols / args.max_batch_size)
    molecules = []

    print(f"Sampling {args.n_mols} molecules in {n_batches} batches\n")
    for _ in range(n_batches):
        # print(f"Batch {batch_idx+1}/{n_batches}")
        n_mols_needed = args.n_mols - len(molecules)
        batch_size = min(n_mols_needed, args.max_batch_size)
        
        # Extract batch property values if using multiple values
        batch_property, batch_no_of_atoms = None, None
        if multilple_values_to_one_property is not None:
            batch_property = multilple_values_to_one_property[len(molecules): len(molecules) + batch_size]
        if number_of_atoms is not None:
            batch_no_of_atoms = number_of_atoms[len(molecules): len(molecules) + batch_size]

        # Sample molecules
        batch_molecules = model.sample_random_sizes(
            batch_size,
            device=device,
            n_timesteps=args.n_timesteps,
            xt_traj=args.xt_traj,
            ep_traj=args.ep_traj,
            stochasticity=args.stochasticity,
            high_confidence_threshold=args.hc_thresh,
            properties_for_sampling=args.properties_for_sampling,
            training_mode=args.training_mode,
            property_name=args.property_name,
            normalization_file_path=args.normalization_file_path,
            properties_handle_method=args.properties_handle_method,
            multilple_values_to_one_property=batch_property,
            number_of_atoms=batch_no_of_atoms,
        )
        molecules.extend(batch_molecules)

    # Analyze molecules if requested
    if args.analyze:
        analyzer = SampleAnalyzer()
        analysis_results = analyzer.analyze(molecules, energy_div=True, functional_validity=True)
        print("FlowMol original metrics:")
        print(" Analysis results:")
        for metric, value in analysis_results.items():
            print(f"  {metric}: {value}")
    
    # Write molecules to SDF file
    sdf_writer = Chem.SDWriter(args.output_file)
    sdf_writer.SetKekulize(False)
    valid_count = 0
    for mol in molecules:
        rdkit_mol = mol.rdkit_mol
        if rdkit_mol is not None:
            sdf_writer.write(rdkit_mol)
            valid_count += 1
    sdf_writer.close()
    print(f"Successfully wrote {valid_count} valid molecules to {args.output_file}\n")

    # New metrics
    new_metrics = compute_all_standard_metrics(args.output_file, n_mols=len(molecules))
    print("PropMolFlow Metrics:\n")
    for metric, value in new_metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()