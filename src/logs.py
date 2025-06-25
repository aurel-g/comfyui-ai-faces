import time 

def error(message, start):
    return {
        "status": "error",
        "message": message,
        "processing_time": round(time.time() - start, 3)
    }

def ok(message, start):
    return {
        "status": "ok",
        "message": message,
        "processing_time": round(time.time() - start, 3)
    }