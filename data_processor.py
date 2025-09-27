import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import random
import pickle
import hashlib
import json
import gc
import psutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
warnings.filterwarnings('ignore')

from config import CORE_BODY_PARTS, VERBOSE

class MABeDataProcessor:
    
    def __init__(self, data_path='/kaggle/input/mabe-mouse-behavior-detection/', cache_dir='cache', n_workers=None):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.n_workers = n_workers or min(8, mp.cpu_count())
        self.memory_limit_gb = 4
        self.chunk_size = 10000
        
        self.enable_cache = True
        self.cache_ttl = 3600 * 24
        
        self.train_df = None
        self.test_df = None
        self.vocabulary_df = None
        
        self.load_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if VERBOSE:
            print(f'[INIT] High-Performance MABeDataProcessor with {self.n_workers} workers')
            print(f'[INIT] Cache directory: {self.cache_dir}')
    
    def _get_cache_key(self, *args):
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file):
        if not cache_file.exists():
            return False
        
        file_age = time.time() - cache_file.stat().st_mtime
        return file_age <= self.cache_ttl
    
    def _save_to_cache(self, key, data, metadata=None):
        if not self.enable_cache:
            return
        
        try:
            cache_file = self.cache_dir / f'{key}.pkl'
            cache_meta_file = self.cache_dir / f'{key}_meta.json'
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            meta_info = {
                'timestamp': time.time(),
                'shape': getattr(data, 'shape', None),
                'type': type(data).__name__,
                'metadata': metadata or {}
            }
            
            with open(cache_meta_file, 'w') as f:
                json.dump(meta_info, f)
                
        except Exception as e:
            if VERBOSE:
                print(f'[CACHE] Failed to save {key}: {str(e)}')
    
    def _load_from_cache(self, key):
        if not self.enable_cache:
            return None
        
        try:
            cache_file = self.cache_dir / f'{key}.pkl'
            
            if not self._is_cache_valid(cache_file):
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.cache_hits += 1
            if VERBOSE:
                print(f'[CACHE HIT] Loaded {key}')
            return data
            
        except Exception:
            self.cache_misses += 1
            return None
    
    @staticmethod
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def cleanup_memory():
        gc.collect()
    
    def load_data(self):
        cache_key = self._get_cache_key('metadata', str(self.data_path))
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            self.train_df, self.test_df, self.vocabulary_df = cached_data
            if VERBOSE:
                print('[CACHE] Loaded metadata from cache')
            return
        
        train_metadata_path = self.data_path / 'train.csv'
        if not train_metadata_path.exists():
            raise FileNotFoundError(f'Training metadata file not found: {train_metadata_path}')
        
        self.train_df = pd.read_csv(train_metadata_path)
        self.train_df = self._optimize_dataframe_memory(self.train_df)
        if VERBOSE:
            print(f'[OK] Loaded train metadata: {self.train_df.shape}')
        
        test_metadata_path = self.data_path / 'test.csv'
        if not test_metadata_path.exists():
            raise FileNotFoundError(f'Test metadata file not found: {test_metadata_path}')
        
        self.test_df = pd.read_csv(test_metadata_path)
        self.test_df = self._optimize_dataframe_memory(self.test_df)
        if VERBOSE:
            print(f'[OK] Loaded test metadata: {self.test_df.shape}')
        
        vocab_path = self.data_path / 'vocabulary.csv'
        if vocab_path.exists():
            self.vocabulary_df = pd.read_csv(vocab_path)
            self.vocabulary_df = self._optimize_dataframe_memory(self.vocabulary_df)
            if VERBOSE:
                print(f'[OK] Loaded vocabulary: {self.vocabulary_df.shape}')
        
        cache_data = (self.train_df, self.test_df, self.vocabulary_df)
        self._save_to_cache(cache_key, cache_data)
    
    def _optimize_dataframe_memory(self, df):
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                num_unique = df[col].nunique()
                num_total = len(df[col])
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
        
        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        if VERBOSE:
            reduction = (start_memory - end_memory) / start_memory * 100
            print(f'[MEMORY] Reduced by {reduction:.1f}% ({start_memory:.1f}MB -> {end_memory:.1f}MB)')
        
        return df
    
    def parallel_load_multiple_videos(self, video_ids, is_train=True, load_annotations=True):
        if VERBOSE:
            print(f'[PARALLEL] Loading {len(video_ids)} videos with {self.n_workers} workers')
        
        results = {}
        
        def load_single_video(video_id):
            try:
                start_time = time.time()
                
                tracking_data = self.load_tracking_data(video_id, is_train)
                if tracking_data is None:
                    return video_id, None
                
                annotations = None
                if load_annotations and is_train:
                    annotations = self.load_annotations(video_id, is_train)
                
                video_metadata = self.get_video_metadata(video_id, is_train)
                
                tracking_data = self.preprocess_coordinates(tracking_data, video_metadata)
                
                load_time = time.time() - start_time
                
                return video_id, {
                    'tracking_data': tracking_data,
                    'annotations': annotations,
                    'video_metadata': video_metadata,
                    'load_time': load_time
                }
                
            except Exception as e:
                if VERBOSE:
                    print(f'[ERROR] Failed to load video {video_id}: {str(e)}')
                return video_id, None
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_video = {executor.submit(load_single_video, vid): vid for vid in video_ids}
            
            for future in as_completed(future_to_video):
                video_id, result = future.result()
                if result is not None:
                    results[video_id] = result
                    
                current_memory = self.get_memory_usage()
                if current_memory > self.memory_limit_gb * 1024:
                    if VERBOSE:
                        print(f'[MEMORY] Usage high ({current_memory:.1f}MB), cleaning up')
                    self.cleanup_memory()
        
        if VERBOSE:
            successful = len(results)
            total_time = sum(r['load_time'] for r in results.values())
            avg_time = total_time / max(1, successful)
            print(f'[PARALLEL] Loaded {successful}/{len(video_ids)} videos successfully')
            print(f'[PARALLEL] Average load time: {avg_time:.2f}s')
        
        return results
    
    def load_tracking_data_fast(self, video_id, is_train=True):
        cache_key = self._get_cache_key('tracking', video_id, is_train)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        if is_train:
            video_metadata = self.get_video_metadata(video_id, is_train=True)
            lab_id = video_metadata.get('lab_id')
            if not lab_id:
                raise ValueError(f'No lab_id found for training video {video_id}')
            
            tracking_path_parquet = self.data_path / 'train_tracking' / lab_id / f'{video_id}.parquet'
            tracking_path_csv = self.data_path / 'train_tracking' / lab_id / f'{video_id}.csv'
        else:
            tracking_path_parquet = self.data_path / 'test_tracking' / f'{video_id}.parquet'
            tracking_path_csv = self.data_path / 'test_tracking' / f'{video_id}.csv'
        
        tracking_path = None
        if tracking_path_parquet.exists():
            tracking_path = tracking_path_parquet
        elif tracking_path_csv.exists():
            tracking_path = tracking_path_csv
        
        if not tracking_path or not tracking_path.exists():
            raise FileNotFoundError(f'Tracking data file not found for video {video_id}')
        
        if tracking_path.suffix == '.parquet':
            tracking_df = pd.read_parquet(tracking_path)
        else:
            file_size = tracking_path.stat().st_size
            if file_size > 50 * 1024 * 1024:
                chunks = []
                for chunk in pd.read_csv(tracking_path, chunksize=self.chunk_size):
                    chunk = self._optimize_dataframe_memory(chunk)
                    chunks.append(chunk)
                tracking_df = pd.concat(chunks, ignore_index=True)
            else:
                tracking_df = pd.read_csv(tracking_path)
        
        tracking_df = self._optimize_dataframe_memory(tracking_df)
        
        self._save_to_cache(cache_key, tracking_df, {'video_id': video_id, 'shape': tracking_df.shape})
        
        if VERBOSE:
            print(f'[OK] Loaded tracking data for video {video_id}: {tracking_df.shape}')
        return tracking_df
    
    def load_tracking_data(self, video_id, is_train=True):
        return self.load_tracking_data_fast(video_id, is_train)
    
    def batch_preprocess_coordinates(self, video_data_dict):
        if VERBOSE:
            print(f'[BATCH] Preprocessing coordinates for {len(video_data_dict)} videos')
        
        processed_videos = {}
        
        for video_id, video_info in video_data_dict.items():
            try:
                tracking_data = video_info['tracking_data']
                video_metadata = video_info['video_metadata']
                
                processed_tracking = self.preprocess_coordinates(tracking_data, video_metadata)
                
                processed_videos[video_id] = {
                    **video_info,
                    'tracking_data': processed_tracking
                }
                
            except Exception as e:
                if VERBOSE:
                    print(f'[ERROR] Failed to preprocess video {video_id}: {str(e)}')
                continue
        
        return processed_videos
    
    def load_annotations(self, video_id, is_train=True):
        cache_key = self._get_cache_key('annotations', video_id, is_train)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        if not is_train:
            return None
        
        video_metadata = self.get_video_metadata(video_id, is_train=True)
        lab_id = video_metadata.get('lab_id')
        
        if not lab_id:
            raise ValueError(f'No lab_id found for training video {video_id}')
        
        annotations_path_parquet = self.data_path / 'train_annotation' / lab_id / f'{video_id}.parquet'
        annotations_path_csv = self.data_path / 'train_annotation' / lab_id / f'{video_id}.csv'
        
        annotations_path = None
        if annotations_path_parquet.exists():
            annotations_path = annotations_path_parquet
        elif annotations_path_csv.exists():
            annotations_path = annotations_path_csv
        
        if not annotations_path or not annotations_path.exists():
            raise FileNotFoundError(f'Annotation file not found for video {video_id}')
        
        if annotations_path.suffix == '.parquet':
            annotations_df = pd.read_parquet(annotations_path)
        else:
            annotations_df = pd.read_csv(annotations_path)
        
        annotations_df = self._optimize_dataframe_memory(annotations_df)
        self._save_to_cache(cache_key, annotations_df)
        
        if VERBOSE:
            print(f'[OK] Loaded annotations for video {video_id}: {annotations_df.shape}')
        return annotations_df
    
    @lru_cache(maxsize=128)
    def get_video_metadata(self, video_id, is_train=True):
        if is_train and self.train_df is not None:
            video_meta = self.train_df[self.train_df['video_id'] == video_id]
        elif not is_train and self.test_df is not None:
            video_meta = self.test_df[self.test_df['video_id'] == video_id]
        else:
            raise ValueError(f'No metadata found for video {video_id}')
        
        if len(video_meta) == 0:
            raise ValueError(f'Video {video_id} not found in metadata')
        
        return video_meta.iloc[0].to_dict()
    
    def validate_tracking_data(self, tracking_df, video_metadata):
        if tracking_df is None or len(tracking_df) == 0:
            return False, '[ERROR] Empty tracking data'
        
        issues = []
        
        required_cols = ['video_frame', 'mouse_id', 'bodypart']
        missing_cols = [col for col in required_cols if col not in tracking_df.columns]
        if missing_cols:
            issues.append(f'[ERROR] Missing columns: {missing_cols}')
        
        coord_cols = []
        if 'x' in tracking_df.columns and 'y' in tracking_df.columns:
            coord_cols = ['x', 'y']
        elif 'x_norm' in tracking_df.columns and 'y_norm' in tracking_df.columns:
            coord_cols = ['x_norm', 'y_norm']
        
        if not coord_cols:
            issues.append('[ERROR] No coordinate columns found (need x,y or x_norm,y_norm)')
        
        if 'bodypart' in tracking_df.columns:
            mice = tracking_df['mouse_id'].unique() if 'mouse_id' in tracking_df.columns else []
            if len(mice) < 2:
                issues.append(f'[WARN] Expected at least 2 mice, found: {len(mice)}')
        
        if coord_cols:
            coord_range = tracking_df[coord_cols].max().max() - tracking_df[coord_cols].min().min()
            if coord_range < 10:
                issues.append('[WARN] Coordinates seem to have very small range')
        
        is_valid = len([issue for issue in issues if issue.startswith('[ERROR]')]) == 0
        return is_valid, issues
    
    def preprocess_coordinates(self, tracking_df, video_metadata):
        if tracking_df is None or len(tracking_df) == 0:
            return tracking_df
        
        processed_df = tracking_df.copy()
        
        pix_per_cm = video_metadata.get('pix_per_cm_approx', 15.0)
        arena_width_cm = video_metadata.get('arena_width_cm', 45.0)
        arena_height_cm = video_metadata.get('arena_height_cm', 35.0)
        video_width_pix = video_metadata.get('video_width_pix', 1024)
        video_height_pix = video_metadata.get('video_height_pix', 768)
        
        if 'x_norm' in processed_df.columns and 'y_norm' in processed_df.columns:
            processed_df['x_norm'] = np.clip(processed_df['x_norm'].values, 0, 1)
            processed_df['y_norm'] = np.clip(processed_df['y_norm'].values, 0, 1)
        elif 'x' in processed_df.columns and 'y' in processed_df.columns:
            if pix_per_cm > 0 and arena_width_cm > 0 and arena_height_cm > 0:
                processed_df['x_cm'] = processed_df['x'].values / pix_per_cm
                processed_df['y_cm'] = processed_df['y'].values / pix_per_cm
                
                processed_df['x_norm'] = processed_df['x_cm'].values / arena_width_cm
                processed_df['y_norm'] = processed_df['y_cm'].values / arena_height_cm
            else:
                processed_df['x_norm'] = processed_df['x'].values / video_width_pix
                processed_df['y_norm'] = processed_df['y'].values / video_height_pix
            
            processed_df['x_norm'] = np.clip(processed_df['x_norm'].values, 0, 1)
            processed_df['y_norm'] = np.clip(processed_df['y_norm'].values, 0, 1)
        else:
            raise ValueError('No valid coordinate columns found')
        
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        return processed_df
    
    def get_sample_videos(self, n_videos=20, strategy='balanced'):
        if self.train_df is None:
            self.load_data()
        
        if strategy == 'balanced':
            sample_videos = []
            if 'lab_id' in self.train_df.columns:
                for lab in self.train_df['lab_id'].unique():
                    lab_videos = self.train_df[self.train_df['lab_id'] == lab]['video_id'].tolist()
                    n_lab_samples = min(len(lab_videos), max(1, n_videos // len(self.train_df['lab_id'].unique())))
                    sample_videos.extend(random.sample(lab_videos, n_lab_samples))
            else:
                sample_videos = self.train_df['video_id'].head(n_videos).tolist()
        else:
            sample_videos = self.train_df['video_id'].sample(min(n_videos, len(self.train_df))).tolist()
        
        return sample_videos
    
    def get_train_video_ids(self):
        if self.train_df is None:
            self.load_data()
        return self.train_df['video_id'].tolist()
    
    def get_test_video_ids(self):
        if self.test_df is None:
            self.load_data()
        return self.test_df['video_id'].tolist()
    
    def get_lab_info(self):
        if self.train_df is None:
            self.load_data()
        
        lab_info = {}
        if 'lab_id' in self.train_df.columns:
            for lab_id in self.train_df['lab_id'].unique():
                lab_videos = self.train_df[self.train_df['lab_id'] == lab_id]
                lab_info[lab_id] = {
                    'n_videos': len(lab_videos),
                    'video_ids': lab_videos['video_id'].tolist()
                }
        
        return lab_info
    
    def get_lab_behaviors(self, lab_id):
        if self.train_df is None:
            self.load_data()
        
        lab_videos = self.train_df[self.train_df['lab_id'] == lab_id]
        if len(lab_videos) > 0:
            behaviors_str = lab_videos['behaviors_labeled'].iloc[0]
            if pd.notna(behaviors_str):
                import ast
                try:
                    return ast.literal_eval(behaviors_str)
                except Exception:
                    return behaviors_str.split(',')
        
        return ['approach', 'attack', 'avoid', 'chase', 'mount', 'sniff', 'rear']
    
    def get_lab_body_parts(self, lab_id):
        if self.train_df is None:
            self.load_data()
        
        lab_videos = self.train_df[self.train_df['lab_id'] == lab_id]
        if len(lab_videos) > 0:
            parts_str = lab_videos['body_parts_tracked'].iloc[0]
            if pd.notna(parts_str):
                import ast
                try:
                    return ast.literal_eval(parts_str)
                except Exception:
                    return parts_str.split(',')
        
        return CORE_BODY_PARTS
    
    def clear_cache(self):
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            if VERBOSE:
                print('[CACHE] Cleared all cached data')
    
    def get_cache_stats(self):
        cache_files = list(self.cache_dir.glob('*.pkl'))
        total_size = sum(f.stat().st_size for f in cache_files) / 1024**2
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'total_files': len(cache_files),
            'total_size_mb': total_size
        }
    
    def benchmark_loading_speed(self, n_videos=5):
        if VERBOSE:
            print(f'[BENCHMARK] Testing loading speed with {n_videos} videos')
        
        sample_videos = self.get_sample_videos(n_videos)
        
        start_time = time.time()
        results = self.parallel_load_multiple_videos(sample_videos)
        parallel_time = time.time() - start_time
        
        start_time = time.time()
        sequential_results = {}
        for video_id in sample_videos:
            tracking_data = self.load_tracking_data(video_id)
            if tracking_data is not None:
                sequential_results[video_id] = tracking_data
        sequential_time = time.time() - start_time
        
        speedup = sequential_time / max(0.001, parallel_time)
        
        if VERBOSE:
            print(f'[BENCHMARK] Parallel: {parallel_time:.2f}s, Sequential: {sequential_time:.2f}s')
            print(f'[BENCHMARK] Speedup: {speedup:.2f}x')
            print(f'[BENCHMARK] Cache stats: {self.get_cache_stats()}')
        
        return {
            'parallel_time': parallel_time,
            'sequential_time': sequential_time,
            'speedup': speedup,
            'videos_processed': len(results),
            'cache_stats': self.get_cache_stats()
        }