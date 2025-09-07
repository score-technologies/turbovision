def model_predict(model, images: list[Image.Image]) -> list[SVFrameResult]:
    frame_results = []
    detections = model(images)
    for i, detection in enumerate(detections):
        boxes = []
        if hasattr(detection, "boxes") and detection.boxes is not None:
            for box in detection.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                boxes.append(
                    SVBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        cls="player",
                        conf=float(conf),
                    )
                )
            frame_results.append(SVFrameResult(frame_id=i, boxes=boxes))
    return frame_results


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

        images = []
        for frame in data.frames:
            try:
                images.append(frame.image)
            except Exception as e:
                return SVPredictOutput(
                    success=False,
                    error=f"Failed to decode frame {frame.frame_id}: {str(e)}",
                    model=model_name,
                )

        frame_results = model_predict(model=model, images=images)

        return SVPredictOutput(
            success=True, model=model_name, predictions={"frames": frame_results}
        )

    except Exception as e:
        print(f"Error in predict_scorevision: {str(e)}")
        print(format_exc())
        return SVPredictOutput(success=False, error=str(e), model=model_name)
