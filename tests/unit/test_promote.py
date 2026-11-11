from promote import promote, rollback


class FakeRegistry:
    def __init__(self, **aliases):
        self.aliases = dict(aliases)

    def get_alias(self, name, alias):
        return self.aliases.get(alias)

    def set_alias(self, name, alias, version):
        self.aliases[alias] = version


def test_passing_candidate_becomes_production_and_old_production_is_kept_as_previous():
    registry = FakeRegistry(production="3")
    assert promote(registry, "laya-router", "4", gate_passed=True) == "promoted"
    assert registry.aliases == {"production": "4", "previous": "3"}


def test_failing_candidate_is_rejected_and_nothing_changes():
    registry = FakeRegistry(production="3")
    assert promote(registry, "laya-router", "4", gate_passed=False) == "rejected"
    assert registry.aliases == {"production": "3"}


def test_first_ever_promotion_has_no_previous():
    registry = FakeRegistry()
    promote(registry, "laya-router", "1", gate_passed=True)
    assert registry.aliases == {"production": "1"}


def test_rollback_restores_the_previous_version():
    registry = FakeRegistry(production="4", previous="3")
    assert rollback(registry, "laya-router") == "rolled_back"
    assert registry.aliases["production"] == "3"


def test_rollback_without_previous_is_refused():
    assert rollback(FakeRegistry(production="1"), "laya-router") == "nothing_to_roll_back"
