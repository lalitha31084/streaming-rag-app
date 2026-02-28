import time
import redis

print("Worker started...", flush=True)

r = redis.Redis(host="redis", port=6379)

while True:
    try:
        print("Worker alive...", flush=True)
        time.sleep(5)
    except Exception as e:
        print("Worker error:", e, flush=True)
        time.sleep(5)