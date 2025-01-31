# Function to log when a fall is detected by saving the current time.
def log_fall():
    with open("falls.log", "a") as f:
        f.write("Fall detected at frame\n")
