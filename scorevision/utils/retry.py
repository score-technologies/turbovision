from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from botocore.exceptions import ClientError, EndpointConnectionError

# Retry for these network failure modes
retry_network = retry(
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
    retry=retry_if_exception_type((ClientError, EndpointConnectionError)),
)
