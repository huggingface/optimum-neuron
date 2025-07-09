from transformers.pipelines.base import GenericTensor, Pipeline

from optimum.utils import is_sentence_transformers_available


if is_sentence_transformers_available():
    from optimum.exporters.tasks import TasksManager


def is_sentence_transformer_model(model: str, token: str = None, revision: str = None):
    """Checks if the model is a sentence transformer model based on provided model id"""
    try:
        _library_name = TasksManager.infer_library_from_model(model, token=token, revision=revision)
        return _library_name == "sentence_transformers"
    except ValueError:
        return False


class FeatureExtractionPipeline(Pipeline):
    """
    Sentence Transformers compatible Feature extraction pipeline uses no model head.
    This pipeline extracts the sentence embeddings from the sentence transformers, which can be used
    in embedding-based tasks like clustering and search. The pipeline is based on the `transformers` library.
    And automatically used instead of the `transformers` library's pipeline when the model is a sentence transformer model.

    Example:

    ```python
    >>> from optimum.neuron import pipeline

    >>> extractor = pipeline(model="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction", export=True, batch_size=2, sequence_length=128)
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, dimension] representing the input string.
    torch.Size([1, 384])
    ```
    """

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, **tokenize_kwargs) -> dict[str, GenericTensor]:
        model_inputs = self.tokenizer(inputs, return_tensors=self.framework, **tokenize_kwargs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, _model_outputs, return_tensors=False):
        # Needed change for sentence transformers.
        # Check if the model outputs sentence embeddings or not.
        if hasattr(_model_outputs, "sentence_embedding"):
            model_outputs = _model_outputs.sentence_embedding
        else:
            model_outputs = _model_outputs
        # [0] is the first available tensor, logits or last_hidden_state.
        if return_tensors:
            return model_outputs[0]
        if self.framework == "pt":
            return model_outputs[0].tolist()

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `list[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)
