def compute_keypoints() -> list[tuple[int, int]]:
    # NOTE: these keypoints are fake/random - they should be computed via a dedicated model
    WIDTH = 960  # NOTE: this will vary depending on challenge size
    HEIGHT = 540  # NOTE: this will vary depending on challenge size
    N_KEYPOINTS = 32  # NOTE: this will vary depending on challenge type (currently set to keypoints for football challenge)
    return [(randint(0, WIDTH), randint(0, HEIGHT)) for _ in range(N_KEYPOINTS)]


def bbox_model_predict_batch(
    model, batch_images: list[Image.Image], offset: int
) -> list[SVFrameResult]:
    batch_results = []
    if model is None:
        return [
            SVFrameResult(
                frame_id=offset + frame_number_in_batch,
                boxes=[],
                keypoints=compute_keypoints(),
            )
            for frame_number_in_batch in range(len(batch_images))
        ]
    detections = model(batch_images)
    for frame_number_in_batch, detection in enumerate(detections):
        boxes = []
        if hasattr(detection, "boxes") and detection.boxes is not None:
            for box in detection.boxes.data:
                x1, y1, x2, y2, conf, cls_id = box.tolist()
                boxes.append(
                    SVBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        cls_id=int(cls_id),
                        conf=float(conf),
                    )
                )
            keypoints = compute_keypoints()
            batch_results.append(
                SVFrameResult(
                    frame_id=offset + frame_number_in_batch,
                    boxes=boxes,
                    keypoints=keypoints,
                )
            )
    return batch_results


def get_video_frames_in_batches(
    video: VideoCapture, batch_size: int
) -> Generator[list[ndarray], None, None]:
    batch = []
    while True:
        ok, frame = video.read()
        if not ok:
            if batch:
                yield batch
            break

        batch.append(frame)

        if len(batch) >= batch_size:
            yield batch
            batch = []


def _predict(
    model: Any | None, data: SVPredictInput, model_name: str
) -> SVPredictOutput:
    try:
        if not model:
            return SVPredictOutput(
                success=False, error="Model not loaded", model=model_name
            )

        meta = data.meta or {}
        batch_size = meta.get("batch_size", 128)
        print(f"Batch Size = {batch_size}")
        mock_after_n_frames = meta.get("mock_after_n_frames", 750)
        print(f"Will only compute first {mock_after_n_frames} frames")

        print(f"Downloading challenge video at {data.url}")
        response = get(data.url)
        try:
            response.raise_for_status()
        except Exception as e:
            return SVPredictOutput(
                success=False, error=f"Problem downloading video: {e}", model=model_name
            )

        print(f"Response successful. Saving video to temporary file.")
        with NamedTemporaryFile(prefix="sv_video_", suffix=".mp4") as f:
            f.write(response.content)
            cap = VideoCapture(f.name)
            if not cap.isOpened():
                return SVPredictOutput(
                    success=False,
                    error=f"Problem accessing downloaded video {f.name}",
                    model=model_name,
                )

            n_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
            print(f"Processing video with {n_frames} frames in batches of {batch_size}")
            frame_results = []
            for batch_number, images in enumerate(
                get_video_frames_in_batches(video=cap, batch_size=batch_size)
            ):
                frame_number = batch_size * batch_number
                print(f"Predicting Batch: {batch_number+1}")
                if frame_number > mock_after_n_frames:
                    batch_frame_results = bbox_model_predict_batch(
                        model=None, batch_images=images, offset=frame_number
                    )
                else:
                    batch_frame_results = bbox_model_predict_batch(
                        model=model, batch_images=images, offset=frame_number
                    )
                frame_results.extend(batch_frame_results)

            cap.release()

        return SVPredictOutput(
            success=True, model=model_name, predictions={"frames": frame_results}
        )

    except Exception as e:
        print(f"Error in predict_scorevision: {str(e)}")
        print(format_exc())
        return SVPredictOutput(success=False, error=str(e), model=model_name)
