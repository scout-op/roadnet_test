# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx


@TRANSFORMS.register_module()
class ParallelSeqTransform(BaseTransform):
    """Transform RoadNet sequences into parallel subsequences for SAR-RNTR.
    
    This transform decomposes the original RoadNet sequence into multiple
    parallel subsequences, each starting from a keypoint (intersection, endpoint, etc.).
    
    Args:
        max_parallel_seqs (int): Maximum number of parallel subsequences.
        max_seq_len (int): Maximum length of each subsequence.
        keypoint_strategy (str): Strategy for keypoint selection.
            Options: 'intersection', 'endpoint', 'random', 'grid'.
        min_seq_len (int): Minimum length for a valid subsequence.
        noise_ratio (float): Ratio of noise subsequences to add.
    """
    
    def __init__(self,
                 max_parallel_seqs: int = 16,
                 max_seq_len: int = 128,
                 keypoint_strategy: str = 'intersection',
                 min_seq_len: int = 3,
                 noise_ratio: float = 0.1):
        
        self.max_parallel_seqs = max_parallel_seqs
        self.max_seq_len = max_seq_len
        self.keypoint_strategy = keypoint_strategy
        self.min_seq_len = min_seq_len
        self.noise_ratio = noise_ratio
        
        # Token definitions (consistent with AR-RNTR)
        self.box_range = 200
        self.category_start = 200
        self.connect_start = 250
        self.coeff_start = 350
        self.no_known = 575
        self.start = 574
        self.end = 573
        self.noise_connect = 572
        self.noise_label = 571
        self.noise_coeff = 570
    
    def extract_keypoints_from_sequence(self, sequence: List[int]) -> List[int]:
        """Extract keypoints from a RoadNet sequence.
        
        Args:
            sequence (List[int]): Original RoadNet sequence.
            
        Returns:
            List[int]: Indices of keypoints in the sequence.
        """
        keypoints = []
        
        if self.keypoint_strategy == 'intersection':
            # Find intersection points based on vertex categories
            for i, token in enumerate(sequence):
                if self.category_start <= token < self.connect_start:
                    # Check if it's an intersection type (simplified logic)
                    vertex_type = token - self.category_start
                    if vertex_type in [0, 2]:  # Ancestor or Offshoot
                        keypoints.append(i)
        
        elif self.keypoint_strategy == 'endpoint':
            # Find endpoints (start and end of sequences)
            keypoints = [0]
            if len(sequence) > 1:
                keypoints.append(len(sequence) - 1)
        
        elif self.keypoint_strategy == 'grid':
            # Uniform grid sampling
            step = max(1, len(sequence) // self.max_parallel_seqs)
            keypoints = list(range(0, len(sequence), step))
        
        elif self.keypoint_strategy == 'random':
            # Random sampling
            num_keypoints = min(self.max_parallel_seqs, len(sequence) // self.min_seq_len)
            if num_keypoints > 0:
                keypoints = sorted(np.random.choice(
                    len(sequence), size=num_keypoints, replace=False))
        
        # Ensure we have at least one keypoint
        if not keypoints:
            keypoints = [0]
        
        # Limit to max_parallel_seqs
        keypoints = keypoints[:self.max_parallel_seqs]
        
        return keypoints
    
    def build_subsequences(self, sequence: List[int], keypoints: List[int]) -> Tuple[List[List[int]], List[bool]]:
        """Build parallel subsequences from keypoints.
        
        Args:
            sequence (List[int]): Original sequence.
            keypoints (List[int]): Keypoint indices.
            
        Returns:
            Tuple[List[List[int]], List[bool]]: (subsequences, validity_mask)
        """
        subsequences = []
        validity_mask = []
        
        for i, keypoint_idx in enumerate(keypoints):
            # Determine subsequence boundaries
            start_idx = keypoint_idx
            if i + 1 < len(keypoints):
                end_idx = keypoints[i + 1]
            else:
                end_idx = len(sequence)
            
            # Extract subsequence
            subseq = sequence[start_idx:end_idx]
            
            # Check validity
            is_valid = len(subseq) >= self.min_seq_len
            
            # Truncate if too long
            if len(subseq) > self.max_seq_len:
                subseq = subseq[:self.max_seq_len]
            
            subsequences.append(subseq)
            validity_mask.append(is_valid)
        
        return subsequences, validity_mask
    
    def pad_subsequences(self, subsequences: List[List[int]], validity_mask: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Pad subsequences to uniform length and add noise sequences.
        
        Args:
            subsequences (List[List[int]]): List of subsequences.
            validity_mask (List[bool]): Validity mask for subsequences.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (padded_sequences, sequence_masks)
        """
        # Initialize output arrays
        padded_seqs = np.full((self.max_parallel_seqs, self.max_seq_len), 
                             self.no_known, dtype=np.int64)
        seq_masks = np.zeros((self.max_parallel_seqs, self.max_seq_len), dtype=np.float32)
        
        # Fill valid subsequences
        num_valid = 0
        for i, (subseq, is_valid) in enumerate(zip(subsequences, validity_mask)):
            if i >= self.max_parallel_seqs:
                break
                
            if is_valid and len(subseq) > 0:
                # Fill sequence
                seq_len = min(len(subseq), self.max_seq_len)
                padded_seqs[num_valid, :seq_len] = subseq[:seq_len]
                seq_masks[num_valid, :seq_len] = 1.0
                num_valid += 1
        
        # Add noise sequences
        num_noise = int(self.noise_ratio * self.max_parallel_seqs)
        for i in range(num_valid, min(num_valid + num_noise, self.max_parallel_seqs)):
            # Generate random noise sequence
            noise_len = np.random.randint(self.min_seq_len, self.max_seq_len // 2)
            
            # Random coordinates
            noise_coords = np.random.randint(0, self.box_range, size=noise_len * 2)
            # Noise category
            noise_categories = np.full(noise_len, self.noise_label)
            # Noise connections
            noise_connections = np.full(noise_len, self.noise_connect)
            # Noise coefficients
            noise_coeffs = np.random.randint(self.coeff_start, self.coeff_start + 220, size=noise_len * 2)
            
            # Interleave tokens (simplified pattern)
            for j in range(noise_len):
                if j * 6 + 5 < self.max_seq_len:
                    padded_seqs[i, j*6:(j+1)*6] = [
                        noise_coords[j*2], noise_coords[j*2+1],
                        noise_categories[j], noise_connections[j],
                        noise_coeffs[j*2], noise_coeffs[j*2+1]
                    ]
            
            # Don't set mask for noise sequences (they should be ignored in loss)
        
        return padded_seqs, seq_masks
    
    def transform(self, results: Dict) -> Dict:
        """Transform function.
        
        Args:
            results (Dict): Result dict from loading pipeline.
            
        Returns:
            Dict: Updated result dict with parallel sequences.
        """
        # Extract original sequence from results
        # This assumes the sequence is stored in results['roadnet_sequence']
        if 'roadnet_sequence' not in results:
            # If no sequence available, create dummy data
            sequence = [self.start] + [self.no_known] * 10 + [self.end]
        else:
            sequence = results['roadnet_sequence']
        
        # Convert to list if tensor
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        elif isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        
        # Extract keypoints
        keypoints = self.extract_keypoints_from_sequence(sequence)
        
        # Build subsequences
        subsequences, validity_mask = self.build_subsequences(sequence, keypoints)
        
        # Pad subsequences
        padded_seqs, seq_masks = self.pad_subsequences(subsequences, validity_mask)
        
        # Create input and target sequences
        # For training: input is shifted version of target (teacher forcing)
        parallel_seqs_tgt = padded_seqs.copy()
        parallel_seqs_in = np.roll(padded_seqs, 1, axis=1)
        parallel_seqs_in[:, 0] = self.start  # Start token for each subsequence
        
        # Store in results
        results['parallel_seqs_in'] = torch.from_numpy(parallel_seqs_in)
        results['parallel_seqs_tgt'] = torch.from_numpy(parallel_seqs_tgt)
        results['parallel_seqs_mask'] = torch.from_numpy(seq_masks)
        results['keypoints'] = keypoints
        results['num_valid_seqs'] = sum(validity_mask)
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_parallel_seqs={self.max_parallel_seqs}, '
        repr_str += f'max_seq_len={self.max_seq_len}, '
        repr_str += f'keypoint_strategy={self.keypoint_strategy}, '
        repr_str += f'min_seq_len={self.min_seq_len}, '
        repr_str += f'noise_ratio={self.noise_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RoadNetSequenceParser(BaseTransform):
    """Parse centerline annotations into RoadNet sequences.
    
    This transform converts the centerline annotations from the dataset
    into the token-based RoadNet sequence format.
    """
    
    def __init__(self,
                 pc_range: List[float],
                 discretization_resolution: float = 0.5,
                 max_sequence_length: int = 1000):
        
        self.pc_range = pc_range
        self.discretization_resolution = discretization_resolution
        self.max_sequence_length = max_sequence_length
        
        # Token definitions
        self.box_range = 200
        self.category_start = 200
        self.connect_start = 250
        self.coeff_start = 350
    
    def discretize_coordinate(self, coord: float, coord_range: Tuple[float, float]) -> int:
        """Discretize a coordinate value to token space.
        
        Args:
            coord (float): Coordinate value.
            coord_range (Tuple[float, float]): (min_val, max_val) of coordinate range.
            
        Returns:
            int: Discretized token value.
        """
        min_val, max_val = coord_range
        normalized = (coord - min_val) / (max_val - min_val)
        discretized = int(normalized * (self.box_range - 1))
        return max(0, min(discretized, self.box_range - 1))
    
    def parse_centerlines_to_sequence(self, centerlines: List[Dict]) -> List[int]:
        """Parse centerline annotations to RoadNet sequence.
        
        Args:
            centerlines (List[Dict]): List of centerline annotations.
            
        Returns:
            List[int]: RoadNet sequence tokens.
        """
        sequence = []
        
        x_range = (self.pc_range[0], self.pc_range[3])
        y_range = (self.pc_range[1], self.pc_range[4])
        
        for i, centerline in enumerate(centerlines):
            # Extract centerline points
            if 'pts' in centerline:
                pts = centerline['pts']
            elif 'points' in centerline:
                pts = centerline['points']
            else:
                continue
            
            # Process each point in the centerline
            for j, pt in enumerate(pts):
                x, y = pt[0], pt[1]
                
                # Discretize coordinates
                x_token = self.discretize_coordinate(x, x_range)
                y_token = self.discretize_coordinate(y, y_range)
                
                # Determine vertex category (simplified)
                if j == 0:
                    category = 0  # Ancestor
                elif j == len(pts) - 1:
                    category = 1  # Lineal (end)
                else:
                    category = 1  # Lineal (middle)
                
                category_token = self.category_start + category
                
                # Connection info (simplified)
                if j < len(pts) - 1:
                    connect_token = self.connect_start + j + 1  # Connect to next
                else:
                    connect_token = self.connect_start  # No connection
                
                # Bezier coefficients (simplified - use same coordinates)
                coeff_x_token = self.coeff_start + (x_token % 110)
                coeff_y_token = self.coeff_start + 110 + (y_token % 110)
                
                # Add 6-token sequence for this vertex
                sequence.extend([
                    x_token, y_token, category_token,
                    connect_token, coeff_x_token, coeff_y_token
                ])
        
        # Truncate if too long
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
        
        return sequence
    
    def transform(self, results: Dict) -> Dict:
        """Transform function.
        
        Args:
            results (Dict): Result dict from loading pipeline.
            
        Returns:
            Dict: Updated result dict with RoadNet sequence.
        """
        # Extract centerline annotations
        centerlines = results.get('gt_labels_3d', [])
        if not isinstance(centerlines, list):
            centerlines = []
        
        # Parse to sequence
        sequence = self.parse_centerlines_to_sequence(centerlines)
        
        # Store in results
        results['roadnet_sequence'] = sequence
        
        return results
