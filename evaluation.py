import asyncio
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
import pyrealsense2 as rs
import cv2
from experiments.robot.assembly.run_assembly_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
import argparse

# Add UR5 controller package to path so we can reuse its RTDE helper classes
UR5_BUILDER_PARENT = Path(__file__).resolve().parent.parent  # /home/.../hanzhi
UR5_BUILDER_ROOT = UR5_BUILDER_PARENT / "UR5_dataset_builder"
if UR5_BUILDER_ROOT.is_dir():
    # Need the parent dir on sys.path to import package "UR5_dataset_builder"
    sys.path.append(str(UR5_BUILDER_PARENT))

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import importlib.util

# Try to load VacuumGripper while avoiding heavy optional deps (pyspacemouse, pyrealsense2) from ur5_controller/__init__.py
try:
    from UR5_dataset_builder.ur5_controller.vacuum_gripper import VacuumGripper  # type: ignore
    _vacuum_import_error = None
except ImportError as exc:  # pragma: no cover - optional dependency
    VacuumGripper = None
    _vacuum_import_error = exc
    # Directly load the module file to bypass ur5_controller/__init__.py side imports
    vacuum_path = UR5_BUILDER_ROOT / "ur5_controller" / "vacuum_gripper.py"
    if vacuum_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("vacuum_gripper", vacuum_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "VacuumGripper"):
                    VacuumGripper = mod.VacuumGripper
                    _vacuum_import_error = None
        except Exception as inner_exc:  # pragma: no cover
            _vacuum_import_error = inner_exc

# Lightweight copy of AsyncGripperController from UR5_dataset_builder (without SpaceMouse dependency)
class AsyncGripperController:
    def __init__(self, ip_address: str):
        if VacuumGripper is None:
            raise ImportError(
                f"VacuumGripper not available: {_vacuum_import_error if '_vacuum_import_error' in globals() else 'unknown error'}"
            )
        self.ip_address = ip_address
        self.loop = None
        self.gripper = None
        self._connected = False
        self._gripper_state = False  # False for open, True for closed

    def _ensure_event_loop(self):
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

    async def _init_gripper(self):
        if self.gripper is None:
            self.gripper = VacuumGripper(self.ip_address)
        if not self._connected:
            await self.gripper.connect()
            await self.gripper.activate()
            self._connected = True

    def get_gripper_state(self):
        return self._gripper_state

    def _run_async_task(self, coro):
        self._ensure_event_loop()
        if self.loop.is_running():
            asyncio.ensure_future(coro, loop=self.loop)
        else:
            self.loop.run_until_complete(coro)

    def control_gripper(self, close=True, force=100, speed=30):
        async def _control():
            await self._init_gripper()
            if close:
                self._gripper_state = 1.0
                await self.gripper.close_gripper(force=force, speed=speed)
            else:
                self._gripper_state = 0.0
                await self.gripper.open_gripper(force=force, speed=speed)

        self._run_async_task(_control())

    def control_gripper_position(self, value: float, force=100, speed=30):
        async def _control():
            await self._init_gripper()
            pos = float(value)
            target = int(np.clip(pos, 0.0, 1.0) * 255)
            await self.gripper._set_var(self.gripper.FOR, force)
            await self.gripper._set_var(self.gripper.SPE, speed)
            await self.gripper._set_var(self.gripper.POS, target)
            await self.gripper._set_var(self.gripper.GTO, 1)
            self._gripper_state = target / 255.0

        self._run_async_task(_control())

    def disconnect(self):
        async def _disconnect():
            if self.gripper and self._connected:
                await self.gripper.disconnect()
                self._connected = False
                self.gripper = None

        self._run_async_task(_disconnect())

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

CHECKPOINT = "VLA_Side_Side_Wrist_Cube" # included checkpoint based on finetuning script above
INSTRUCTION = "put the white cube into thered box" # text string
UR5_IP = os.environ.get("UR5_IP", "192.168.1.60")
# Default RealSense serials: side=D405, wrist=D435; override with env RS_SERIAL_SIDE / RS_SERIAL_WRIST
RS_SERIAL_SIDE = "218622277783"
RS_SERIAL_WRIST = "819612070593"
# Front D435if camera (defaults to provided serial, override via RS_SERIAL_FRONT or legacy RS_SERIAL_WRIST2 env)
RS_SERIAL_FRONT = "230322273810"
# Default UR5 "home" joint configuration (degrees converted to radians); override via --home_joints or env UR5_HOME_JOINTS
# Target initial pose (radians) before running the model
DEFAULT_HOME_JOINTS = [
    -1.07113515e-04,
    -1.57078392e00,
    1.57073641e00,
    -1.57088024e00,
    -1.57061321e00,
    1.57065701e00,
]

cfg = GenerateConfig(
    pretrained_checkpoint=CHECKPOINT,
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    num_images_in_input=3,
    unnorm_key="assembly_robot_data",
)

vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

def get_ur5_state(ip_address: str, gripper_state: float = 0.0) -> np.ndarray:
    """
    末端执行器状态：TCP位姿 (x, y, z, rx, ry, rz) + 夹爪。
    """
    rtde_r = RTDEReceiveInterface(ip_address)
    tcp_pose = rtde_r.getActualTCPPose()
    state = np.zeros(7, dtype=np.float32)
    state[:6] = np.asarray(tcp_pose[:6], dtype=np.float32)
    state[6] = float(gripper_state)
    return state


def get_ur5_state_with_receiver(rtde_r: RTDEReceiveInterface, gripper_state: float = 0.0) -> np.ndarray:
    tcp_pose = rtde_r.getActualTCPPose()
    state = np.zeros(7, dtype=np.float32)
    state[:6] = np.asarray(tcp_pose[:6], dtype=np.float32)
    state[6] = float(gripper_state)
    return state


class RealSenseStreams:
    """Keep RealSense side (D405), wrist (D435) and front (D435if) color streams open for repeated capture."""

    def __init__(self, serial_side: str, serial_wrist: str, serial_front: str):
        if not serial_side or not serial_wrist or not serial_front:
            raise ValueError("All RealSense serials (side, wrist, front) must be provided.")
        self.pipelines = {}
        self._start("side", serial_side)
        self._start("wrist", serial_wrist)
        self._start("front", serial_front)

    def _start(self, name: str, serial: str) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        self.pipelines[name] = pipeline

    def get_rgb(self) -> dict[str, np.ndarray]:
        frames = {}
        for name, pipeline in self.pipelines.items():
            frames[name] = pipeline.wait_for_frames()

        images = {}
        for name, frame_set in frames.items():
            color_frame = frame_set.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"Missing color frame from RealSense ({name}).")
            images[name] = np.asanyarray(color_frame.get_data())[:, :, ::-1]  # BGR->RGB

        return images

    def stop(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.stop()


def load_observation(rs_streams: RealSenseStreams, rtde_r: RTDEReceiveInterface, gripper: AsyncGripperController | None = None):
    # Pull current robot state directly from UR5; prefer live gripper state if available
    gripper_state = float(gripper.get_gripper_state())
    state = get_ur5_state_with_receiver(rtde_r, gripper_state=gripper_state)
    rgb_frames = rs_streams.get_rgb()
    obs = {
        "full_image": rgb_frames["front"],
        "wrist_image_side": rgb_frames["side"], # side image
        "wrist_image": rgb_frames["wrist"],
        "state": state,
        "task_description": INSTRUCTION,
    }
    return obs


def show_inputs(obs: dict) -> None:
    """Open a window to visualize model inputs: three images + EEF state array."""
    imgs = []
    for key in ["full_image", "wrist_image_side", "wrist_image"]:
        if key in obs:
            img = obs[key]
            # Convert RGB->BGR for OpenCV display
            imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if imgs:
        # Resize to the same height for side-by-side
        target_h = min(im.shape[0] for im in imgs)
        resized = [cv2.resize(im, (int(im.shape[1]*target_h/im.shape[0]), target_h)) for im in imgs]
        concat = cv2.hconcat(resized)
    else:
        concat = np.zeros((240, 320, 3), dtype=np.uint8)

    state = obs.get("state", np.zeros(7, dtype=np.float32))
    info_panel = np.zeros((120, concat.shape[1], 3), dtype=np.uint8)
    text = f"EEF pose+gripper: {np.array2string(state, precision=3)}"
    cv2.putText(info_panel, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    display = cv2.vconcat([concat, info_panel])
    cv2.imshow("Model Inputs (front | side | wrist)", display)
    cv2.waitKey(1)

def parse_home_joints(raw: str | None):
    """Parse comma-separated joints into a list[float] if valid."""
    if not raw:
        return None
    try:
        joints = [float(val) for val in raw.split(",")]
        if len(joints) != 6:
            raise ValueError("expected 6 joint values")
        return joints
    except Exception as exc:
        print(f"[WARN] Failed to parse home joints '{raw}': {exc}")
        return None

def move_ur5_home(rtde_c: RTDEControlInterface, joints: list[float]):
    """Move UR5 to a known starting joint configuration before inference."""
    print(f"[INFO] Moving UR5 to home position: {joints}")
    rtde_c.moveJ(
        joints,
        speed=float(os.environ.get("UR5_JOINT_SPEED", "0.3")),
        acceleration=float(os.environ.get("UR5_JOINT_ACC", "0.3")),
        asynchronous=False,
    )
    print("[INFO] UR5 reached home position.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_ur5", action="store_true", help="Send predicted actions to UR5 if set.")
    parser.add_argument("--max_loops", type=int, default=0, help="Number of cycles to run; 0 = infinite.")
    parser.add_argument("--loop_sleep", type=float, default=0.1, help="Sleep time (s) between cycles.")
    parser.add_argument("--use_relative", type=int, default=0, help="Treat actions as relative joint deltas (1) or absolute (0).")
    parser.add_argument(
        "--home_joints",
        type=str,
        default=None,
        help="Comma-separated UR5 joint angles (radians) to move to before running the model. "
             "Defaults to env UR5_HOME_JOINTS or a built-in pose.",
    )
    parser.add_argument(
        "--skip_home_move",
        action="store_true",
        help="Skip moving UR5 to the home pose before starting inference.",
    )
    parser.add_argument(
        "--show_inputs",
        action="store_true",
        help="Display a window with model input images and EEF state.",
    )
    args = parser.parse_args()

    rs_streams = RealSenseStreams(RS_SERIAL_SIDE, RS_SERIAL_WRIST, RS_SERIAL_FRONT)
    rtde_r = RTDEReceiveInterface(UR5_IP)
    rtde_c = None
    gripper = None
    
    rtde_c = RTDEControlInterface(UR5_IP)
    gripper = AsyncGripperController(UR5_IP)

    # Move UR5 to a known starting configuration before running the model
    if rtde_c is not None and not args.skip_home_move:
        home_joints = (
            parse_home_joints(args.home_joints)
            or parse_home_joints(os.environ.get("UR5_HOME_JOINTS"))
            or DEFAULT_HOME_JOINTS
        )
        move_ur5_home(rtde_c, home_joints)

    loop_idx = 0
    try:
        while True:
            if args.max_loops and loop_idx >= args.max_loops:
                break

            obs = load_observation(rs_streams, rtde_r, gripper)
            if args.show_inputs:
                show_inputs(obs)
            actions = get_vla_action(
                cfg,
                vla,
                processor,
                obs,
                obs["task_description"],
                action_head,
                proprio_projector,
            )
            print(f"[Loop {loop_idx}] Action chunk:")
            for a in actions:
                print(a)

            if rtde_c is not None and gripper is not None and len(actions) > 0:
                # 只用每个 action chunk 的最后一个动作驱动机器人
                action = actions[-1]
                current_joints = rtde_r.getActualQ()
                joint_cmd = np.asarray(action[:6], dtype=np.float64)
                if args.use_relative:
                    joint_cmd = np.asarray(current_joints[:6], dtype=np.float64) + joint_cmd
                rtde_c.moveJ(
                    joint_cmd.tolist(),
                    speed=float(os.environ.get("UR5_JOINT_SPEED", "0.3")),
                    acceleration=float(os.environ.get("UR5_JOINT_ACC", "0.3")),
                    asynchronous=True,
                )
                # 将模型输出的抓手值直接传给机械爪（映射至0..255位置指令）
                gripper_cmd = float(action[-1])
                gripper.control_gripper_position(gripper_cmd, force=100, speed=30)

            loop_idx += 1
            time.sleep(args.loop_sleep)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        if args.show_inputs:
            cv2.destroyAllWindows()
        rs_streams.stop()
        rtde_c.stopScript()
        gripper.disconnect()
