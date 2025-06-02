robot = ManipulatorRobot(
    KochRobotConfig(
        leader_arms={"main": leader_arm},
        follower_arms={"main": follower_arm},
        calibration_dir=".cache/calibration/koch",
        cameras={
            "phone": OpenCVCameraConfig(0, fps=30, width=640, height=480),
        },
    )
)
robot.connect()