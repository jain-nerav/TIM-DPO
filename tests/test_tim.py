import torch

from config import TIMConfig
from tim import TokenImportanceMemory


def test_initial_state():
    config = TIMConfig()

    tim = TokenImportanceMemory(
        vocab_size=10,
        tim_config=config,
        device="cpu",
    )

    assert torch.all(tim.ema_table == 1.0)
    assert torch.all(tim.update_counts == 0)

    weights = tim.get_weights(torch.tensor([[1, 2, 3]]))

    assert torch.allclose(
        weights,
        torch.ones_like(weights),
    )

    print("PASS: initial state")


def test_repeated_token_averaging():
    config = TIMConfig(
        beta=0.0,
        normalize_batch_importance=False,
        clip_importance=False,
    )

    tim = TokenImportanceMemory(
        vocab_size=10,
        tim_config=config,
        device="cpu",
    )

    token_ids = torch.tensor([2, 2, 2, 4])
    importance = torch.tensor([1.0, 3.0, 5.0, 2.0])

    tim.update(
        token_ids=token_ids,
        importance=importance,
    )

    # Token 2:
    # (1 + 3 + 5) / 3 = 3
    assert torch.isclose(
        tim.ema_table[2],
        torch.tensor(3.0),
    )

    # Token 4:
    # 2 / 1 = 2
    assert torch.isclose(
        tim.ema_table[4],
        torch.tensor(2.0),
    )

    print("PASS: repeated-token averaging")


def test_ema_update():
    config = TIMConfig(
        beta=0.5,
        normalize_batch_importance=False,
        clip_importance=False,
    )

    tim = TokenImportanceMemory(
        vocab_size=10,
        tim_config=config,
        device="cpu",
    )

    tim.update(
        token_ids=torch.tensor([3]),
        importance=torch.tensor([5.0]),
    )

    # 0.5 * 1 + 0.5 * 5 = 3
    assert torch.isclose(
        tim.ema_table[3],
        torch.tensor(3.0),
    )

    print("PASS: EMA update")


def test_previous_batch_mode():
    config = TIMConfig(
        memory_mode="previous_batch",
        normalize_batch_importance=False,
        clip_importance=False,
    )

    tim = TokenImportanceMemory(
        vocab_size=10,
        tim_config=config,
        device="cpu",
    )

    tim.update(
        token_ids=torch.tensor([3]),
        importance=torch.tensor([7.0]),
    )

    assert torch.isclose(
        tim.ema_table[3],
        torch.tensor(7.0),
    )

    print("PASS: previous-batch mode")


def test_untouched_tokens_remain_one():
    config = TIMConfig(
        beta=0.0,
        normalize_batch_importance=False,
        clip_importance=False,
    )

    tim = TokenImportanceMemory(
        vocab_size=10,
        tim_config=config,
        device="cpu",
    )

    tim.update(
        token_ids=torch.tensor([3]),
        importance=torch.tensor([5.0]),
    )

    weights = tim.get_weights(
        torch.tensor([[1, 2, 3, 4]])
    )

    assert torch.isclose(
        tim.ema_table[3],
        torch.tensor(5.0),
    )

    # Only token 3 has been updated.
    assert torch.isclose(weights[0, 0], torch.tensor(1.0))
    assert torch.isclose(weights[0, 1], torch.tensor(1.0))
    assert torch.isclose(weights[0, 3], torch.tensor(1.0))

    print("PASS: untouched tokens remain at 1")


def test_clipping():
    config = TIMConfig(
        beta=0.0,
        normalize_batch_importance=False,
        clip_importance=True,
        clip_min=0.25,
        clip_max=4.0,
    )

    tim = TokenImportanceMemory(
        vocab_size=10,
        tim_config=config,
        device="cpu",
    )

    tim.update(
        token_ids=torch.tensor([2, 3]),
        importance=torch.tensor([10.0, 0.01]),
    )

    assert torch.isclose(
        tim.ema_table[2],
        torch.tensor(4.0),
    )

    assert torch.isclose(
        tim.ema_table[3],
        torch.tensor(0.25),
    )

    print("PASS: clipping")


def run_all_tests():
    test_initial_state()
    test_repeated_token_averaging()
    test_ema_update()
    test_previous_batch_mode()
    test_untouched_tokens_remain_one()
    test_clipping()

    print("\nALL TIM TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()