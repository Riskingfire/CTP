from ctprotocol import CTProtocolConfig, CTProtocolStream, URLSource

url = "https://example.com/data.jsonl"

config = CTProtocolConfig(
    cache_dir=".ctp_cache",
    ahead_seconds=60,
    max_cache_mb=1024,
    delete_consumed=True
)

source = URLSource(url)
stream = CTProtocolStream(source, config)

try:
    for chunk in stream.stream():
        print("Received", len(chunk), "bytes")
finally:
    stream.cleanup()
