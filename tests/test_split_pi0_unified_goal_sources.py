from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "datasets" / "split_pi0_unified_goal_sources.py"
    spec = spec_from_file_location("split_pi0_unified_goal_sources", script)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_sources_is_stratified_disjoint_and_reproducible():
    module = load_module()
    episodes = [
        Path(f"data/cr3_real_drag_raw/{kind}/drag_episode_{index:02d}")
        for kind in ("red", "green", "yellow", "full")
        for index in range(10)
    ]

    train_a, validation_a = module.split_sources(episodes, validation_fraction=0.2, seed=7)
    train_b, validation_b = module.split_sources(episodes, validation_fraction=0.2, seed=7)

    assert train_a == train_b
    assert validation_a == validation_b
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(episodes)
    assert {path.parent.name: 0 for path in validation_a} == {"red": 0, "green": 0, "yellow": 0, "full": 0}
    assert {kind: sum(path.parent.name == kind for path in validation_a) for kind in ("red", "green", "yellow", "full")} == {
        "red": 2,
        "green": 2,
        "yellow": 2,
        "full": 2,
    }
