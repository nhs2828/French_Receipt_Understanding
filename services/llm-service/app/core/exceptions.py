"""
Custom pipeline exception classes to ensure consistent JSON responses from the error handler
and contextual logging (identifying which stage failed: input/segmentation/ocr).
"""


class PipelineError(Exception):
    """Base class for errors in extraction pipeline."""
    error_code: str = "PIPELINE_ERROR"
    status_code: int = 500

    def __init__(self, message: str, stage: str | None = None):
        self.message = message
        self.stage = stage
        super().__init__(message)


class InvalidImageError(PipelineError):
    error_code = "INVALID_IMAGE"
    status_code = 400


class FileTooLargeError(PipelineError):
    error_code = "FILE_TOO_LARGE"
    status_code = 413


class UnsupportedMediaTypeError(PipelineError):
    error_code = "UNSUPPORTED_MEDIA_TYPE"
    status_code = 415


class InferenceTimeoutError(PipelineError):
    error_code = "INFERENCE_TIMEOUT"
    status_code = 504


class NoDocumentDetectedError(PipelineError):
    """Segmentation does not detect any polygon."""
    error_code = "NO_DOCUMENT_DETECTED"
    status_code = 422


class SegmentationError(PipelineError):
    error_code = "SEGMENTATION_ERROR"
    status_code = 500


class OCRError(PipelineError):
    error_code = "OCR_ERROR"
    status_code = 500


class ModelNotLoadedError(PipelineError):
    error_code = "MODEL_NOT_LOADED"
    status_code = 503


class ModelSourceNotFoundError(PipelineError):
    """Could not find model code"""
    error_code = "MODEL_SOURCE_NOT_FOUND"
    status_code = 503

class ServiceBusyError(PipelineError):
    """Queue is full - refuse request."""
    error_code = "SERVICE_BUSY"
    status_code = 503
