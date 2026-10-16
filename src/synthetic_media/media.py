from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
from .components import Artifact, Context

LABEL = "SYNTHETIC MEDIA | RESEARCH"

def mark(frame):
    height, width = frame.shape[:2]
    scale = max(0.25, min(1.0, width / 700))
    thickness = max(1, round(scale * 2))
    size, baseline = cv2.getTextSize(LABEL, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (0, max(0, height - size[1] - baseline - 16)), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, LABEL, (5, height - baseline - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def mark_footer(frame):
    """Keep every content pixel; put disclosure in an added footer."""
    width=frame.shape[1]
    scale=max(.20,min(.8,width/700))
    thickness=max(1,round(scale*2))
    size,baseline=cv2.getTextSize(LABEL,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    if size[0]>width-10:
        scale*=max(1,width-10)/size[0]
        size,baseline=cv2.getTextSize(LABEL,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    footer_height=max(24,size[1]+baseline+14)
    output=cv2.copyMakeBorder(frame,0,footer_height,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(output,LABEL,(5,frame.shape[0]+size[1]+6),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
    return output

class FaceProcessor:
    name = "face_processing"

    def process(self, inputs: dict[str, Artifact], context: Context):
        source = inputs["reference"].path
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        still = cv2.imread(str(source))
        capture = None
        if still is None:
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError("Reference must be a decodable image or video")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                capture.release()
                raise ValueError("Reference video has invalid frame rate")
            stride = max(1, round(fps))
        else:
            stride = 1
        records = []
        count = 0
        try:
            while True:
                if still is not None:
                    frame = still
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                if count % stride == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
                    faces = []
                    for index, (x, y, w, h) in enumerate(boxes):
                        path = context.output_dir / f"face-{count:07d}-{index}.png"
                        if not cv2.imwrite(str(path), frame[y:y+h, x:x+w]):
                            raise RuntimeError("Failed to save face crop")
                        faces.append({"box": [int(x), int(y), int(w), int(h)], "crop": path.name})
                    records.append({"frame": count, "faces": faces})
                count += 1
                if still is not None:
                    break
        finally:
            if capture is not None:
                capture.release()
        return {"faces": Artifact(source, "face_analysis", {"frames": count, "samples": records,
                "detector": "opencv-haar-frontal", "identity_embedding": False})}

def render_labeled(source: Path, destination: Path):
    """Label an image or transcode a video, retaining source audio when present."""
    still = cv2.imread(str(source))
    if still is not None:
        output = destination.with_suffix(".png")
        if not cv2.imwrite(str(output), mark_footer(still)):
            raise RuntimeError("Failed to write labeled image")
        return output
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot decode input media")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise ValueError("Invalid video dimensions or frame rate")
    output = destination.with_suffix(".mp4")
    # Raw frames retain original dimensions and timing; audio is mapped from source.
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
            "-crf", "18", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-metadata", "comment=Synthetic media research export", str(output)]
    import tempfile
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=errors)
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    process.stdin.write(mark(frame).tobytes())
                process.stdin.close()
                code = process.wait()
                if code:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace")[-4000:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    finally:
        capture.release()
    return output
