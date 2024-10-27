# daphne_config.py
timeout_seconds = 29  # Slightly less than Daphne's 30-second default
graceful_shutdown_timeout = 10  # Seconds to wait for graceful shutdown

application = app  # Your FastAPI app instance

# Configure Daphne settings
daphne_settings = {
    'websocket_timeout': timeout_seconds,
    'http_timeout': timeout_seconds,
    'proxy_timeout': timeout_seconds,
    'graceful_shutdown_timeout': graceful_shutdown_timeout,
}