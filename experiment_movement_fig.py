# %%
import cv2
import numpy as np


def create_motion_trail(
    video_path,
    output_path,
    decay=0.9,
    interval=5,
    start_frame=0,
    final_frame=None,
):
    """
    Creates a composite image where the newest frame has high opacity
    and older frames fade out (onion skinning effect).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    composite = first_frame.astype(np.float32)
    frame_count = start_frame

    while True:
        if final_frame is not None and frame_count >= final_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % interval != 0:
            continue

        # THE FIX: Swap decay and (1 - decay)
        # Now, 'frame' gets the 'decay' weight (e.g., 0.9), making it strong.
        # The 'composite' (old frames) gets the (1 - decay) weight (e.g., 0.1), making them fade.
        composite = cv2.addWeighted(
            composite, 1 - decay, frame.astype(np.float32), decay, 0
        )

    final_image = np.clip(composite, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, final_image)
    print(f"Motion trail image saved to {output_path}")

    cap.release()


# --- Usage ---
fps = 30
decay = 0.55
n_snapshots = 3

final_time = 60
init_time = 0
interval = int(fps * (final_time - init_time) / n_snapshots)
mov_name = "XY"

create_motion_trail(
    f"./media/mov_{mov_name}.mp4",
    f"./media/trail_{mov_name}.png",
    decay=decay,
    interval=interval,
    start_frame=fps * init_time,
    final_frame=fps * final_time,
)

# Name is manually changed to "trail_{init_time}-{final_time}.png",
# where XZ: [0, 60], YZ: [60, 120], XY: [120, 180]

