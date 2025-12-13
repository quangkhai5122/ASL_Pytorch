"""
Sign Language Sentence to Video Generator
Sử dụng WLASL_Mediapipe làm từ điển
"""
import argparse
import os
import sys

# Ensure scripts can be imported
sys.path.insert(0, os.getcwd())

from scripts.sign_dictionary import SignDictionary
from scripts.motion_synthesizer import MotionSynthesizer
from scripts.sign_visualizer import SignVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Sign Language Sentence-to-Video Generator using WLASL_Mediapipe"
    )
    parser.add_argument(
        "--sentence", 
        type=str, 
        required=True, 
        help="Space-separated list of glosses (e.g., 'hello how you')"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.gif", 
        help="Output file path (.gif or .mp4)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/WLASL_Mediapipe",
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--transition_frames",
        type=int,
        default=10,
        help="Number of transition frames between words"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--list_glosses",
        action="store_true",
        help="List all available glosses and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize dictionary
    print(f"Loading dictionary from {args.data_dir}...")
    dictionary = SignDictionary(data_dir=args.data_dir)
    
    # List glosses if requested
    if args.list_glosses:
        glosses = dictionary.get_available_glosses()
        print(f"\nAvailable glosses ({len(glosses)}):")
        # Print in columns
        cols = 5
        for i in range(0, len(glosses), cols):
            row = glosses[i:i+cols]
            print("  " + "  ".join(f"{g:15}" for g in row))
        return
    
    # Parse sentence
    glosses = args.sentence.lower().split()
    print(f"\nProcessing glosses: {glosses}")
    
    # Check which glosses are available
    available = [g for g in glosses if dictionary.has_gloss(g)]
    missing = [g for g in glosses if not dictionary.has_gloss(g)]
    
    if missing:
        print(f"Warning: Missing glosses will be skipped: {missing}")
    
    if not available:
        print("Error: No valid glosses found!")
        return
    
    # Initialize synthesizer
    synthesizer = MotionSynthesizer(
        dictionary, 
        transition_frames=args.transition_frames,
        context_frames=3
    )
    
    # Synthesize motion sequence
    print("\nSynthesizing motion sequence...")
    sequence = synthesizer.synthesize_phrase(available)
    
    if sequence is None:
        print("Failed to generate sequence!")
        return
    
    print(f"Generated sequence: {sequence.shape[0]} frames")
    
    # Visualize and save
    visualizer = SignVisualizer(fps=args.fps)
    visualizer.create_animation(sequence, args.output)
    
    print(f"\nVideo saved to: {args.output}")


if __name__ == "__main__":
    main()
