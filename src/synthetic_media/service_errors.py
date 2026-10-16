"""Safe public messages for failures from shared inference services."""

class SharedServiceError(RuntimeError):
    def __init__(self, message, code='shared_service_failed', status=502):
        super().__init__(message)
        self.code = code
        self.status = status


def service_error(detail):
    text = str(detail).lower()
    if any(term in text for term in ('zerogpu runs limit', 'gpu quota', 'quota exceeded', 'exceeded your zerogpu')):
        return SharedServiceError(
            'The shared GPU service quota is exhausted. If you have not signed in locally, '
            'run .venv/bin/hf auth login from the project folder using your free Hugging Face account. '
            'If your account quota is also exhausted, wait for the provider to reset it. '
            'No image was generated and no automatic retry was made.',
            'shared_quota_exhausted', 429)
    if 'timed out' in text or 'timeout' in text:
        return SharedServiceError('The shared service timed out. No completed image was received. '
                                  'Check the local shared-service.log before trying again.', 'shared_timeout', 504)
    return SharedServiceError('The shared service could not complete this request. '
                              'Technical details are saved in the job’s local shared-service.log.')


def public_manifest(data):
    """Also hide tracebacks stored by older versions of the application."""
    data = dict(data)
    error = data.get('error')
    if error and ('Traceback' in error or 'Upstream processor failed' in error):
        safe = service_error(error)
        data['error'] = str(safe)
        data['error_code'] = safe.code
    return data
