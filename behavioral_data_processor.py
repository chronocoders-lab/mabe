import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import CORE_BODY_PARTS, VERBOSE

class BehavioralDataProcessor:
    
    def __init__(self, data_path='/kaggle/input/mabe-mouse-behavior-detection/'):
        self.data_path = Path(data_path)
        self.train_df = None
        self.test_df = None
        self.vocabulary_df = None
        
        self.behavior_hierarchy = {
            'social': ['approach', 'attack', 'mount', 'sniff'],
            'locomotion': ['chase', 'avoid'],
            'individual': ['rear']
        }
        
        self.behavior_priority = {
            'attack': 7, 'mount': 6, 'sniff': 5, 'approach': 4,
            'chase': 3, 'avoid': 2, 'rear': 1
        }
        
        if VERBOSE:
            print('[INIT] Initialized BehavioralDataProcessor with temporal annotation support')
    
    def create_frame_level_labels(self, annotations: pd.DataFrame, 
                                  video_frames: np.ndarray, 
                                  video_metadata: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        n_frames = len(video_frames)
        frame_labels = ['no_behavior'] * n_frames
        agent_ids = ['none'] * n_frames
        target_ids = ['none'] * n_frames
        behavior_scores = np.zeros(n_frames)
        
        if annotations is None or len(annotations) == 0:
            return np.array(frame_labels), np.array(agent_ids), np.array(target_ids)
        
        for _, annotation in annotations.iterrows():
            start_frame = int(annotation.get('start_frame', 0))
            stop_frame = int(annotation.get('stop_frame', start_frame + 30))
            action = annotation.get('action', 'unknown')
            agent_id = annotation.get('agent_id', 'mouse1')
            target_id = annotation.get('target_id', 'mouse2')
            
            if action not in self.behavior_priority:
                continue
            
            frame_to_index = {frame: idx for idx, frame in enumerate(video_frames)}
            
            for frame_num in range(start_frame, stop_frame):
                if frame_num in frame_to_index:
                    i = frame_to_index[frame_num]
                    if 0 <= i < n_frames:
                        current_priority = self.behavior_priority.get(frame_labels[i], 0)
                        new_priority = self.behavior_priority[action]
                        
                        if new_priority > current_priority:
                            frame_labels[i] = action
                            agent_ids[i] = agent_id
                            target_ids[i] = target_id
                            behavior_scores[i] = new_priority
        
        return np.array(frame_labels), np.array(agent_ids), np.array(target_ids)
    
    def create_behavioral_windows(self, tracking_df: pd.DataFrame, 
                                  annotations: pd.DataFrame,
                                  window_sizes: Dict[str, int] = None) -> List[Dict]:
        
        if window_sizes is None:
            window_sizes = {
                'approach': 45, 'attack': 15, 'chase': 90,
                'mount': 30, 'sniff': 20, 'avoid': 60, 'rear': 35
            }
        
        behavioral_windows = []
        
        if annotations is None or len(annotations) == 0:
            return behavioral_windows
        
        unique_frames = tracking_df['video_frame'].unique()
        
        for _, annotation in annotations.iterrows():
            action = annotation.get('action', 'unknown')
            start_frame = int(annotation.get('start_frame', 0))
            stop_frame = int(annotation.get('stop_frame', start_frame + 30))
            
            if action not in window_sizes:
                continue
            
            window_size = window_sizes[action]
            behavior_center = (start_frame + stop_frame) // 2
            
            window_start = max(unique_frames.min(), behavior_center - window_size // 2)
            window_end = min(unique_frames.max(), behavior_center + window_size // 2)
            
            window_mask = (tracking_df['video_frame'] >= window_start) & \
                         (tracking_df['video_frame'] <= window_end)
            window_tracking = tracking_df[window_mask].copy()
            
            if len(window_tracking) == 0:
                continue
            
            window_frames = window_tracking['video_frame'].unique()
            frame_labels, agent_ids, target_ids = self.create_frame_level_labels(
                annotations, window_frames, {}
            )
            
            behavioral_windows.append({
                'action': action,
                'agent_id': annotation.get('agent_id', 'mouse1'),
                'target_id': annotation.get('target_id', 'mouse2'),
                'start_frame': start_frame,
                'stop_frame': stop_frame,
                'window_start': window_start,
                'window_end': window_end,
                'tracking_data': window_tracking,
                'frame_labels': frame_labels,
                'agent_ids': agent_ids,
                'target_ids': target_ids
            })
        
        return behavioral_windows
    
    def validate_behavioral_data(self, tracking_df: pd.DataFrame, 
                                 annotations: pd.DataFrame,
                                 video_metadata: Dict) -> Tuple[bool, List[str]]:
        
        issues = []
        
        if tracking_df is None or len(tracking_df) == 0:
            issues.append('[ERROR] Empty tracking data')
            return False, issues
        
        required_cols = ['video_frame', 'mouse_id', 'bodypart', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in tracking_df.columns]
        if missing_cols:
            issues.append(f'[ERROR] Missing tracking columns: {missing_cols}')
        
        if annotations is not None and len(annotations) > 0:
            ann_required = ['agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
            ann_missing = [col for col in ann_required if col not in annotations.columns]
            if ann_missing:
                issues.append(f'[WARN] Missing annotation columns: {ann_missing}')
            
            if 'start_frame' in annotations.columns and 'stop_frame' in annotations.columns:
                invalid_temporal = annotations['start_frame'] >= annotations['stop_frame']
                if invalid_temporal.any():
                    issues.append(f'[ERROR] {invalid_temporal.sum()} annotations have invalid temporal ranges')
                
                tracking_start = tracking_df['video_frame'].min()
                tracking_end = tracking_df['video_frame'].max()
                
                out_of_range = (annotations['start_frame'] < tracking_start) | \
                              (annotations['stop_frame'] > tracking_end)
                if out_of_range.any():
                    issues.append(f'[WARN] {out_of_range.sum()} annotations outside tracking range')
            
            if 'action' in annotations.columns:
                valid_behaviors = set(self.behavior_priority.keys())
                invalid_behaviors = set(annotations['action'].unique()) - valid_behaviors
                if invalid_behaviors:
                    issues.append(f'[WARN] Unknown behaviors: {invalid_behaviors}')
        
        if 'mouse_id' in tracking_df.columns:
            mice = set(tracking_df['mouse_id'].unique())
            if annotations is not None and len(annotations) > 0:
                if 'agent_id' in annotations.columns:
                    annotation_mice = set(annotations['agent_id'].unique())
                    if 'target_id' in annotations.columns:
                        annotation_mice.update(annotations['target_id'].unique())
                    
                    missing_mice = annotation_mice - mice
                    if missing_mice:
                        issues.append(f'[WARN] Annotated mice not in tracking: {missing_mice}')
        
        if 'bodypart' in tracking_df.columns:
            available_parts = set(tracking_df['bodypart'].unique())
            missing_core_parts = set(CORE_BODY_PARTS) - available_parts
            if missing_core_parts:
                issues.append(f'[WARN] Missing core body parts: {missing_core_parts}')
        
        is_valid = len([issue for issue in issues if issue.startswith('[ERROR]')]) == 0
        
        if VERBOSE and issues:
            for issue in issues:
                print(f'  {issue}')
        
        return is_valid, issues
    
    def create_competition_labels(self, frame_labels: np.ndarray,
                                  agent_ids: np.ndarray,
                                  target_ids: np.ndarray,
                                  video_frames: np.ndarray) -> pd.DataFrame:
        
        competition_labels = []
        current_behavior = None
        current_agent = None
        current_target = None
        behavior_start = None
        
        for i, (frame, label, agent, target) in enumerate(zip(video_frames, frame_labels, agent_ids, target_ids)):
            if label == 'no_behavior':
                if current_behavior is not None:
                    competition_labels.append({
                        'agent_id': current_agent,
                        'target_id': current_target,
                        'action': current_behavior,
                        'start_frame': behavior_start,
                        'stop_frame': video_frames[i-1] if i > 0 else frame
                    })
                    current_behavior = None
            else:
                if (label != current_behavior or agent != current_agent or target != current_target):
                    if current_behavior is not None:
                        competition_labels.append({
                            'agent_id': current_agent,
                            'target_id': current_target,
                            'action': current_behavior,
                            'start_frame': behavior_start,
                            'stop_frame': video_frames[i-1] if i > 0 else frame
                        })
                    
                    current_behavior = label
                    current_agent = agent
                    current_target = target
                    behavior_start = frame
        
        if current_behavior is not None:
            competition_labels.append({
                'agent_id': current_agent,
                'target_id': current_target,
                'action': current_behavior,
                'start_frame': behavior_start,
                'stop_frame': video_frames[-1]
            })
        
        return pd.DataFrame(competition_labels)
    
    def process_video_with_behaviors(self, video_id: int, is_train: bool = True) -> Dict:
        
        try:
            from data.data_processor import MABeDataProcessor
            base_processor = MABeDataProcessor(str(self.data_path))
            
            tracking_data = base_processor.load_tracking_data(video_id, is_train)
            annotations = base_processor.load_annotations(video_id, is_train) if is_train else None
            video_metadata = base_processor.get_video_metadata(video_id, is_train)
            
            if tracking_data is None:
                return {'success': False, 'error': 'No tracking data'}
            
            is_valid, issues = self.validate_behavioral_data(tracking_data, annotations, video_metadata)
            if not is_valid:
                return {'success': False, 'error': f'Validation failed: {issues}'}
            
            tracking_data = base_processor.preprocess_coordinates(tracking_data, video_metadata)
            
            unique_frames = tracking_data['video_frame'].unique()
            frame_labels, agent_ids, target_ids = self.create_frame_level_labels(
                annotations, unique_frames, video_metadata
            )
            
            behavioral_windows = self.create_behavioral_windows(
                tracking_data, annotations
            ) if annotations is not None else []
            
            competition_labels = self.create_competition_labels(
                frame_labels, agent_ids, target_ids, unique_frames
            )
            
            result = {
                'success': True,
                'video_id': video_id,
                'tracking_data': tracking_data,
                'annotations': annotations,
                'video_metadata': video_metadata,
                'frame_labels': frame_labels,
                'agent_ids': agent_ids,
                'target_ids': target_ids,
                'behavioral_windows': behavioral_windows,
                'competition_labels': competition_labels,
                'validation_issues': issues
            }
            
            if VERBOSE:
                n_behaviors = len([l for l in frame_labels if l != 'no_behavior'])
                print(f'  [OK] Video {video_id}: {len(frame_labels)} frames, {n_behaviors} behavior frames, '
                      f'{len(behavioral_windows)} behavior windows')
            
            return result
            
        except Exception as e:
            if VERBOSE:
                print(f'  [ERROR] Processing video {video_id}: {str(e)}')
            return {'success': False, 'error': str(e)}
    
    def get_behavior_statistics(self, processed_videos: List[Dict]) -> Dict:
        
        behavior_counts = {}
        total_frames = 0
        behavior_frames = 0
        
        for video_data in processed_videos:
            if not video_data.get('success', False):
                continue
            
            frame_labels = video_data.get('frame_labels', [])
            total_frames += len(frame_labels)
            
            for label in frame_labels:
                if label != 'no_behavior':
                    behavior_frames += 1
                    behavior_counts[label] = behavior_counts.get(label, 0) + 1
        
        return {
            'total_frames': total_frames,
            'behavior_frames': behavior_frames,
            'behavior_coverage': behavior_frames / max(1, total_frames),
            'behavior_counts': behavior_counts,
            'unique_behaviors': len(behavior_counts)
        }