import argparse
import concurrent.futures
import math
import platform
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc

MAX_OPENCV_INDEX = 60

def parse_camera_id(val):
    try:
        return int(val)
    except ValueError:
        return val

def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    safe_camera_id = str(camera_index).replace(":", "_").replace("/", "_").replace("?", "_")
    path = images_dir / f"camera_{safe_camera_id}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)
    print(f"✅ Saved: {path}")

def save_images_from_cameras(
    images_dir: Path,
    camera_ids: list | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    if camera_ids is None or len(camera_ids) == 0:
        print("No camera IDs provided.")
        return

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        config = OpenCVCameraConfig(camera_index=cam_idx, fps=fps, width=width, height=height, mock=mock)
        camera = OpenCVCamera(config)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.capture_width}, height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()
            for camera in cameras:
                image = camera.read() if fps is None else camera.async_read()
                if image is None:
                    print(f"❌ No image from {camera.camera_index}")
                    continue
                print(f"✅ Got image from {camera.camera_index}: shape={image.shape}")
                executor.submit(save_image, image, camera.camera_index, frame_index, images_dir)

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break
            frame_index += 1

    print(f"Images have been saved to {images_dir}")
    print(f"Full path: {images_dir.resolve()}")

class OpenCVCamera:
    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.capture_width = config.width
        self.capture_height = config.height
        self.width = config.width
        self.height = config.height
        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock
        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.camera_index}) is already connected.")

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2
            cv2.setNumThreads(1)

        backend = (
            cv2.CAP_V4L2 if platform.system() == "Linux" else
            cv2.CAP_DSHOW if platform.system() == "Windows" else
            cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else
            cv2.CAP_ANY
        )

        camera_idx = self.camera_index
        if isinstance(camera_idx, int) or (isinstance(camera_idx, str) and camera_idx.startswith("/dev/video")):
            tmp_camera = cv2.VideoCapture(camera_idx, backend)
        else:
            tmp_camera = cv2.VideoCapture(camera_idx)

        is_camera_open = tmp_camera.isOpened()
        tmp_camera.release()
        del tmp_camera

        if not is_camera_open:
            if isinstance(self.camera_index, int):
                raise OSError(f"Can't access OpenCVCamera({self.camera_index}).")
            else:
                print(f"⚠️ Skipping validation for MJPEG URL camera: {self.camera_index}")

        if isinstance(camera_idx, int) or (isinstance(camera_idx, str) and camera_idx.startswith("/dev/video")):
            self.camera = cv2.VideoCapture(camera_idx, backend)
        else:
            self.camera = cv2.VideoCapture(camera_idx)

        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.capture_width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        if self.capture_height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        self.fps = round(self.camera.get(cv2.CAP_PROP_FPS))
        self.capture_width = round(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_height = round(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()
        ret, color_image = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode == "rgb":
            import cv2
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        self.color_image = color_image
        return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                self.color_image = self.read()
            except Exception as e:
                print(f"Error reading in thread: {e}")

    def async_read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop)
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while True:
            if self.color_image is not None:
                return self.color_image
            time.sleep(1 / self.fps)
            num_tries += 1
            if num_tries > self.fps * 2:
                raise TimeoutError("Timed out waiting for async_read() to start.")

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.release()
        self.camera = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-ids", type=parse_camera_id, nargs="*", default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--images-dir", type=Path, default="outputs/images_from_opencv_cameras")
    parser.add_argument("--record-time-s", type=float, default=4.0)
    args = parser.parse_args()

    if args.camera_ids is None:
        args.camera_ids = ["http://192.168.68.107:8080/video_feed"]

    save_images_from_cameras(**vars(args))
