from scorevision.utils.evaluate import post_vlm_ranking, get_element_scores


def test_post_vlm_ranking(
    dummy_manifest,
    dummy_pseudo_gt_annotations,
    fake_miner_predictions,
    fake_payload,
    fake_challenge,
    fake_frame_store,
) -> None:
    evaluation = post_vlm_ranking(
        payload=fake_payload,
        miner_run=fake_miner_predictions,
        challenge=fake_challenge,
        pseudo_gt_annotations=dummy_pseudo_gt_annotations,
        frame_store=fake_frame_store,
        manifest=dummy_manifest,
    )
    assert isinstance(evaluation.acc_breakdown, dict)
    assert isinstance(evaluation.details, dict)
    assert any(evaluation.acc_breakdown)
    assert any(evaluation.details)
    assert evaluation.latency_ms == 0.0
    assert evaluation.acc > 0.0


def test_get_element_scores(
    dummy_manifest,
    dummy_pseudo_gt_annotations,
    fake_frame_store,
    fake_miner_predictions,
) -> None:
    score, breakdown = get_element_scores(
        manifest=dummy_manifest,
        pseudo_gt_annotations=dummy_pseudo_gt_annotations,
        miner_run=fake_miner_predictions,
        frame_store=fake_frame_store,
    )
    assert True
    # TODO
