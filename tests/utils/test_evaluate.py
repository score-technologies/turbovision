from scorevision.utils.evaluate import post_vlm_ranking


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
    assert True  # TODO
