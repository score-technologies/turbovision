def model_predict_batch(
    model, batch_images: list[Image.Image], offset: int
) -> list[SVFrameResult]:
    batch_results = []
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
            # TODO:
            WIDTH = 960  # NOTE: this will vary depending on challenge size
            HEIGHT = 540  # NOTE: this will vary depending on challenge size
            N_KEYPOINTS = 32  # NOTE: this will vary depending on challenge type
            keypoints = [
                (randint(0, WIDTH), randint(0, HEIGHT)) for _ in range(N_KEYPOINTS)
            ]
            batch_results.append(
                SVFrameResult(
                    frame_id=offset + frame_number_in_batch,
                    boxes=boxes,
                    keypoints=keypoints,
                )
            )
    return batch_results


def _predict(
    model: Any | None, data: SVPredictInput, model_name: str
) -> SVPredictOutput:
    try:
        if not model:
            return SVPredictOutput(
                success=False, error="Model not loaded", model=model_name
            )

        if not data.frames:
            return SVPredictOutput(
                success=False, error="No frames provided", model=model_name
            )

        frame_results = []
        batch_size = 4
        n_frames = len(data.frames)
        for frame_number in range(0, n_frames, batch_size):
            images = []
            print(
                f"Predicting Batch of Frames: ({frame_number}-{frame_number+batch_size})/{n_frames}"
            )
            for frame in data.frames[frame_number : frame_number + batch_size]:
                try:
                    images.append(frame.image)
                except Exception as e:
                    return SVPredictOutput(
                        success=False,
                        error=f"Failed to decode frame {frame.frame_id}: {str(e)}",
                        model=model_name,
                    )

            frame_results.extend(
                model_predict_batch(
                    model=model, batch_images=images, offset=frame_number
                )
            )

        return SVPredictOutput(
            success=True, model=model_name, predictions={"frames": frame_results}
        )

    except Exception as e:
        print(f"Error in predict_scorevision: {str(e)}")
        print(format_exc())
        return SVPredictOutput(success=False, error=str(e), model=model_name)
