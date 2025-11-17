import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

class FilePredictionSystem:
    """ System for making prediction on files and comparing with actual values"""
    def __init__(self,
                    model,
                    id_to_label: Dict = None,
                    label_to_id: Dict = None,
                    full_label_map: Dict = None,
                    target_column: str = 'X_AST Uniclass',
                    text_column: str = 'merged_text',
                    feature_columns: List[str] = None,
                    columns_to_merge: List[str] = None,
                    columns_to_drop: List[str] = None,
                    merge_separator: str = ' ',
                    batch_size: Optional[int] = 32, 
                    auto_merge: bool = True,
                    model_type: str = 'auto'):
        
        self.model = model
        self.id_to_label = id_to_label or {}
        self.label_to_id = label_to_id or {}
        self.full_label_map = full_label_map or {}
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.text_column = text_column
        self.columns_to_merge = columns_to_merge
        self.columns_to_drop = columns_to_drop or []
        self.merge_separator = merge_separator
        self.auto_merge = auto_merge
        self.batch_size = batch_size
        
        # Detect model type
        self.model_type = self._detect_model_type(model, model_type)
        print(f"✓ Detected model type: {self.model_type}")
        
        # For sklearn pipelines with FeatureMerger, feature_columns should be columns_to_merge
        if self.model_type == 'sklearn' and hasattr(model, 'named_steps'):
            if 'feature_merger' in model.named_steps or 'FeatureMerger' in str(type(model.steps[0][1])):
                print("✓ Pipeline includes FeatureMerger - will pass raw DataFrame")
                self.needs_raw_df = True
            else:
                self.needs_raw_df = False
        else:
            self.needs_raw_df = False
        
        self.has_proba = hasattr(model, 'predict_proba')

    def _detect_model_type(self, model, model_type: str) -> str:
        """Detect if model is SetFit, sklearn, or other"""
        if model_type != 'auto':
            return model_type
        
        model_class_name = model.__class__.__name__
        model_module = model.__class__.__module__
        
        if 'setfit' in model_module.lower() or model_class_name == 'SetFitModel':
            return 'setfit'
        elif hasattr(model, 'predict') and hasattr(model, 'fit'):
            return 'sklearn'
        else:
            return 'unknown'
    
    def load_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load csv or excel file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading file: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns)")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that the dataframe has required columns"""
        if self.needs_raw_df and self.columns_to_merge:
            # Check if all required columns exist
            missing_cols = [col for col in self.columns_to_merge if col not in df.columns]
            is_valid = len(missing_cols) == 0
            
            if not is_valid:
                print(f" Warning: Missing columns for FeatureMerger: {missing_cols}")
            
            return is_valid, missing_cols
        
        return True, []
    
    def _predict_sklearn(self, X, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using sklearn-style model"""
        all_predictions = []
        all_probabilities = []
        
        print(f"Making sklearn predictions on {len(X)} samples...")
        
        # Process in single batch since pipeline handles batching internally
        try:
            predictions = self.model.predict(X)
            all_predictions.extend(predictions)
            
            if self.has_proba:
                probabilities = self.model.predict_proba(X)
                all_probabilities.extend(probabilities)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise
                
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities) if all_probabilities else np.array([])
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions, probabilities
    
    def predict_file(self,
                     file_path: str,
                     has_actual_labels: bool= True,
                     top_n_predictions: int = 1,
                     batch_size: int = 32,
                     **load_kwargs) -> pd.DataFrame:
        """Make predictions on entire file"""
        
        df = self.load_file(file_path, **load_kwargs)
        original_df = df.copy()
        
        # Validate data
        is_valid, missing_cols = self.validate_data(df)
        
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"\nMaking predictions on {len(df)} samples ....")
        
        # For sklearn models with FeatureMerger, pass the raw DataFrame
        # The pipeline will handle column selection and merging internally
        if self.model_type == 'sklearn' and self.needs_raw_df:
            X = df[self.columns_to_merge]  # Only pass required columns
            predictions, probabilities = self._predict_sklearn(X, batch_size)
        else:
            raise ValueError(f"Unsupported configuration")
        
        print(" Predictions complete")
        
        # Create result frame
        results_df = self._create_results_dataframe(
            original_df,
            predictions,
            probabilities,
            top_n_predictions,
            has_actual_labels
        )
        
        return results_df
    
    def _create_results_dataframe(self,
                                  original_df: pd.DataFrame,
                                  predictions: np.ndarray,
                                  probabilities: np.ndarray,
                                  top_n: int,
                                  has_actual_labels: bool = True) -> pd.DataFrame:
        """Create comprehensive results dataframe"""
        
        results_df = original_df.copy()
        
        # Add predictions
        results_df['Prediction_Label'] = [
            self.id_to_label.get(pred_id, f"Unknown_{pred_id}") for pred_id in predictions.astype(int)
        ]
            
        # Add confidence scores
        if len(probabilities) > 0:
            results_df['Prediction_Confidence'] = probabilities.max(axis = 1)
            results_df['Confidence_Percentage'] = results_df['Prediction_Confidence'].apply(lambda x: f"{x*100:.2f}%")

            def get_confidence_level(conf):
                if conf >=0.85:
                    return "High Confidence"
                elif conf>=0.7:
                    return "Review some"
                else:
                    return"Must Review"
            
            results_df['Confidence_Level'] = results_df['Prediction_Confidence'].apply(get_confidence_level)

        
        # Add actual labels and comparison
        if has_actual_labels and self.target_column in original_df.columns:
            actual_values = original_df[self.target_column]
            results_df['Actual_Value'] = actual_values.astype(str)
            
            actual_ids = []
            for val in actual_values:
                if val in self.label_to_id:
                    actual_ids.append(self.label_to_id[val])
                elif isinstance(val, (int, np.integer)):
                    actual_ids.append(int(val))
                else:
                    actual_ids.append(np.nan)
                    
            results_df['Actual_ID'] = actual_ids
            results_df['Actual_Label'] = [
                self.id_to_label.get(aid, str(aid)) if not pd.isna(aid) else 'NaN/Missing'
                for aid in actual_ids
            ]
            
            results_df['Prediction_Correct'] = (results_df['Predictioned_ID'] == results_df['Actual_ID'])
            results_df['Prediction_Status'] = results_df.apply(
                lambda row: "✓ Correct" if row['Prediction_Correct'] else "x Wrong",
                axis = 1
            )
        
        # Reorder columns
        important_cols = []
        for col in ['Prediction_Label', 'Confidence_Percentage', 'Confidence_Level', 
                    'Prediction_Confidence', 'Actual_Label', 'Prediction_Correct', 'Prediction_Status']:
            if col in results_df.columns:
                important_cols.append(col)
        
        for n in range(1, top_n+1):
            for col in [f"Top_{n}_Label", f"Top_{n}_Probability_%"]:
                if col in results_df.columns:
                    important_cols.append(col)
        
        remaining_cols = [col for col in results_df.columns if col not in important_cols]
        final_column_order = important_cols + remaining_cols
        results_df = results_df[final_column_order]
        
        return results_df
    
    def export_results(self, results_df: pd.DataFrame, output_path: Path, 
                      include_metrics: bool=True, include_visualization: bool = True, 
                      create_summary_sheet: bool = True):
        """Export result with comprehensive analysis"""
        print(f"\nExporting results to: {output_path}")
        
        file_ext = output_path.suffix.lower()
        
        if file_ext =='.csv':
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to CSV: {output_path}")
            
        elif file_ext in ['.xlsx', '.xls']:    
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Predictions', index=False)
            print(f"Results saved to Excel: {output_path}")
        else:
            output_path = output_path.with_suffix('.csv')
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to CSV: {output_path}")
        
        print("Export complete")
        return output_path
    
    @staticmethod
    def predict_from_file(input_file: str,
                          model,
                          output_file: str = None,
                          id_to_label: Dict = None,
                          label_to_id: Dict = None,
                          full_label_map: Dict = None,
                          feature_columns: List[str] = None,
                          target_column: str ='X_AST Uniclass',
                          text_column: str = 'merged_text',
                          columns_to_merge: List[str] = None,
                          columns_to_drop: List[str] = None,
                          merge_separator: str = ' ',
                          auto_merge: bool = True,
                          has_actual_labels: bool = True,
                          top_n: int = 1,
                          create_visualization: bool = True,
                          model_type: str = 'auto'):
        """Quick function to predict from file and export results"""
        
        pred_system = FilePredictionSystem(
            model=model,
            id_to_label=id_to_label,
            label_to_id=label_to_id,
            full_label_map=full_label_map,
            target_column=target_column,
            text_column=text_column,
            feature_columns=feature_columns,
            columns_to_merge=columns_to_merge,
            columns_to_drop=columns_to_drop,
            merge_separator=merge_separator,
            auto_merge=auto_merge,
            model_type=model_type
        )
        
        results_df = pred_system.predict_file(
            file_path=input_file, 
            has_actual_labels=has_actual_labels, 
            top_n_predictions=top_n
        )
        
        if output_file is None:
            input_path = Path(input_file)
            output_file_path = input_path.parent / f"{input_path.stem}_predictions.xlsx"
        else:
            output_file_path = Path(output_file)
            
        output_path = pred_system.export_results(
            results_df=results_df, 
            output_path=output_file_path, 
            include_metrics=has_actual_labels, 
            include_visualization=create_visualization, 
            create_summary_sheet=has_actual_labels
        )
        
        if has_actual_labels and 'Prediction_Correct' in results_df.columns:
            accuracy = results_df['Prediction_Correct'].mean()
            print(f"\n{'-'*40}")
            print(f"PREDICTION SUMMARY")
            print(f"Total Samples: {len(results_df)}")
            print(f"Correct Predictions: {results_df['Prediction_Correct'].sum()}")
            print(f"Accuracy: {accuracy*100:.2f}%")
            if 'Prediction_Confidence' in results_df.columns:
                print(f"Avg Confidence: {results_df['Prediction_Confidence'].mean():.2f}")
            print(f"{'-'*40}")
        
        return results_df