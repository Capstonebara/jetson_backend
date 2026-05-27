import asyncio
import websockets
import json
from datetime import datetime
import random

# WebSocket server URL
WEBSOCKET_URL = "wss://api.fptuaiclub.me/logs/"
# WEBSOCKET_URL = "ws://localhost:5500/logs/"


async def send_log():
    async def send_ping(websocket):
        while True:
            try:
                await websocket.send("ping")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Ping failed: {e}")
                break

    async def connect_websocket(device_id):
        for attempt in range(5):
            try:
                websocket = await websockets.connect(f"{WEBSOCKET_URL}{device_id}")
                print(f"Connected to server as {WEBSOCKET_URL}{device_id}")
                return websocket
            except Exception as e:
                print(f"Connection failed (attempt {attempt + 1}): {e}")
                await asyncio.sleep(5)
        return None

    async def send_image(websocket, image_path):
        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
            await websocket.send("image")
            await websocket.send(image_data)
            print(f"Image sent: {image_path}")
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
        except Exception as e:
            print(f"Failed to send image: {e}")

    def parse_txt_to_json(txt_path):
        try:
            with open(txt_path, "r") as file:
                line = file.readline().strip()
                if not line:
                    raise ValueError("Text file is empty.")

                parts = line.split()
                if len(parts) != 4:
                    raise ValueError("Line must have exactly 4 fields.")

                user_id = int(parts[0])
                device_id = int(parts[1])
                raw_timestamp = parts[2]  # Format: H:M:S.YYYY
                log_type = parts[3]

                # Convert timestamp to Unix time
                dt = datetime.strptime(raw_timestamp, "%H:%M:%S.%Y")
                unix_timestamp = int(dt.timestamp())

                log_data = {
                    "id": user_id,
                    "device_id": device_id,
                    "timestamp": unix_timestamp,
                    "type": log_type
                }

                return log_data

        except Exception as e:
            print(f"Failed to parse .txt file: {e}")

    # Read first line to extract device_id for WebSocket path
    initial_log = parse_txt_to_json("/home/jetson/FaceRecognitionSystem/jetson/backend/data/data.txt")
    if initial_log is None:
        print("Cannot proceed without a valid log entry.")
        return

    websocket = await connect_websocket(initial_log["device_id"])
    if websocket is None:
        print("Failed to connect to server after multiple attempts.")
        return

    asyncio.ensure_future(send_ping(websocket))

    while True:
        try:
            log_data = parse_txt_to_json("/home/jetson/FaceRecognitionSystem/jetson/backend/data/data.txt")
            if log_data:
                await websocket.send(json.dumps(log_data))
                print("Log sent:", log_data)

            image_path = f"/home/jetson/FaceRecognitionSystem/jetson/backend/data/saved_image.png"
            await send_image(websocket, image_path)

            await asyncio.sleep(10)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection lost: {e}. Reconnecting...")
            websocket = await connect_websocket(log_data["device_id"])
        except KeyboardInterrupt:
            print("Connection closed by user")
            if websocket:
                await websocket.close()
            break


        # Run the WebSocket client
while True:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_log())
