import os
import argparse
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from my_detectors import HoopDetector, HoopFromJSONL, run_scoring_from_files
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    #FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
    HoopShadowDrawer,
    HoopBoxDrawer,
)
from configs import(
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    HOOP_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH
)

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_PATH,
                        help='Path to output video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DEFAULT_PATH,
                        help='Path to stub directory')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[Main] Input video: {args.input_video}")
    print(f"[Main] Output video will be saved to: {args.output_video}")
    print(f"[Main] Stub path: {args.stub_path}")

    # Read Video
    video_frames = read_video(args.input_video)
    print(f"[Main] Read {len(video_frames)} frames from input video")

    ## Initialize Tracker
    print("[Main] Initializing trackers...")
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    ## Initialize Keypoint Detector
    print("[Main] Initializing court keypoint detector...")
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Run Detectors
    print("[Main] Getting player tracks...")
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=os.path.join(args.stub_path, 'player_track_stubs.pkl')
                                      )
    print(f"[Main] Player tracks collected for {len(player_tracks)} frames")

    print("[Main] Loading hoop detections from JSONL (xywh center → xyxy)...")
    hoop_loader = HoopFromJSONL()
    hoop_detections = hoop_loader.get_detections(
        frames=video_frames,
        read_from_stub=False,
        stub_path=os.path.join(args.stub_path, 'hoop_dets_stubs.pkl'),
        jsonl_path="/home/ubuntu/scoring_detection/001/hoop/detections.jsonl",
        raw_frames_dir="/home/ubuntu/scoring_detection/001/raw_frames",  # helps align if frame names aren’t in `frames`
        keep_top_k=1
    )
    print(f"[Main] Hoop detections collected for {len(hoop_detections)} frames")

    print("[Main] Getting ball tracks...")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                 read_from_stub=False,
                                                 stub_path=os.path.join(args.stub_path, 'ball_track_stubs.pkl'),
                                                 anno_path='/home/ubuntu/basketball_analysis/ball_detections/001/detections.jsonl'
                                                 )
    print(f"[Main] Ball tracks collected for {len(ball_tracks)} frames")

    # Detect scoring events (uses hoop_detections + post-processed ball_tracks)
    print("[Main] Detecting scoring events...")

    run_scoring_from_files(
        frames_dir="/home/ubuntu/scoring_detection/001/raw_frames",
        hoop_annos_path="/home/ubuntu/scoring_detection/001/hoop/detections.jsonl",
        ball_annos_path="/home/ubuntu/scoring_detection/001/ball/detections.jsonl",
        output_dir=os.path.join(args.stub_path, "scoring_event_detection"),
        output_name="scoring_events.jsonl",
        draw_debug=False,
    )

    ## Run KeyPoint Extractor
    print("[Main] Getting court keypoints...")
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'court_key_points_stub.pkl')
                                                                    )
    print(f"[Main] Court keypoints collected for {len(court_keypoints_per_frame)} frames")

    # Remove Wrong Ball Detections
    print("[Main] Removing wrong ball detections...")
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate Ball Tracks
    print("[Main] Interpolating ball tracks...")
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Teams
    print("[Main] Assigning teams...")
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                    player_tracks,
                                                                    read_from_stub=True,
                                                                    stub_path=os.path.join(args.stub_path, 'player_assignment_stub.pkl')
                                                                    )
    print(f"[Main] Player assignments complete ({len(player_assignment)} frames)")

    # Ball Acquisition
    print("[Main] Detecting ball acquisition...")
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    # Detect Passes
    print("[Main] Detecting passes and interceptions...")
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)

    # Tactical View
    print("[Main] Initializing tactical view converter...")
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    print("[Main] Validating court keypoints...")
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    print("[Main] Transforming players to tactical view...")
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame,player_tracks)

    # Speed and Distance Calculator
    print("[Main] Calculating speed and distance...")
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(tactical_player_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)

    # Draw output
    print("[Main] Initializing drawers...")
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    #frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()
    hoop_box_drawer = HoopBoxDrawer()

    ## Draw object Tracks
    print("[Main] Drawing player tracks...")
    output_video_frames = player_tracks_drawer.draw(video_frames,
                                                    player_tracks,
                                                    player_assignment,
                                                    ball_aquisition)
    print("[Main] Drawing ball tracks...")
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    ## Draw KeyPoints
    print("[Main] Drawing court keypoints...")
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints_per_frame)

    ## Draw Frame Number
    print("[Main] Drawing frame numbers...")
    #output_video_frames = frame_number_drawer.draw(output_video_frames)

    # Draw Team Ball Control
    print("[Main] Drawing team ball control...")
    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_aquisition)

    # Draw Passes and Interceptions
    print("[Main] Drawing passes and interceptions...")
    output_video_frames = pass_and_interceptions_drawer.draw(output_video_frames,
                                                             passes,
                                                             interceptions)

    # Speed and Distance Drawer
    print("[Main] Drawing speed and distance...")
    output_video_frames = speed_and_distance_drawer.draw(output_video_frames,
                                                         player_tracks,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )

    ## Draw Tactical View
    print("[Main] Drawing tactical view...")
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                    tactical_view_converter.court_image_path,
                                                    tactical_view_converter.width,
                                                    tactical_view_converter.height,
                                                    tactical_view_converter.key_points,
                                                    tactical_player_positions,
                                                    player_assignment,
                                                    ball_aquisition,
                                                    )

    ## Draw Hoop BBox
    print("[Main] Drawing hoop boxes...")
    output_video_frames = hoop_box_drawer.draw(output_video_frames, hoop_detections)

    ## Draw Hoop Shadow (camera frame)
    print("[Main] Drawing hoop shadows...")
    hoop_shadow_drawer = HoopShadowDrawer(
        key_points=tactical_view_converter.key_points,
        tactical_width=tactical_view_converter.width,
        tactical_height=tactical_view_converter.height,
        color=(0, 0, 255)  # red-ish in BGR
    )
    output_video_frames = hoop_shadow_drawer.draw(output_video_frames, court_keypoints_per_frame)

    # Save video
    print(f"[Main] Saving {len(output_video_frames)} frames to {args.output_video}")
    if len(output_video_frames) > 0:
        print(f"[Main] First frame shape: {output_video_frames[0].shape}")
    save_video(output_video_frames, args.output_video)
    print("[Main] Done!")

if __name__ == '__main__':
    main()

