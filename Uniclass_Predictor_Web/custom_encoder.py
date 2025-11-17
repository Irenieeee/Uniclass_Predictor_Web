from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None  # Initialize as None

    def fit(self, X, y=None):
        # Load model during fit
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        # CRITICAL FIX: Ensure model is loaded before transform
        if self.model is None:
            print("⚠️  Model not loaded in transform, loading now...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        
        texts = [str(x) for x in X]
        
        try:
            # Use the encode method with proper parameters
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            print(f"❌ Encoding error: {e}")
            raise RuntimeError(f"SentenceTransformer encoding failed: {e}")
    
    def __getstate__(self):
        """Custom serialization - don't save the model object"""
        state = self.__dict__.copy()
        state['model'] = None  # Don't pickle the model
        return state
    
    def __setstate__(self, state):
        """Custom deserialization - reload model when unpickling"""
        self.__dict__.update(state)
        # Model will be loaded lazily in transform if needed