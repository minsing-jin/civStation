from computer_use_test.agent.modules.context.context_manager import ContextManager


class TestContextManagerPolicyCache:
    def setup_method(self):
        ContextManager.reset_instance()
        self.ctx = ContextManager.get_instance()

    def teardown_method(self):
        ContextManager.reset_instance()

    def test_policy_tab_cache_persists_across_advance_turn(self):
        self.ctx.replace_policy_tab_cache(
            positions={
                "군사": {"screen_x": 200, "screen_y": 100, "confirmed": True},
                "경제": {"screen_x": 260, "screen_y": 100, "confirmed": True},
                "외교": {"screen_x": 320, "screen_y": 100, "confirmed": True},
                "와일드카드": {"screen_x": 380, "screen_y": 100, "confirmed": True},
                "암흑": {"screen_x": 440, "screen_y": 100, "confirmed": True},
            },
            confirmed_tabs=["군사", "경제", "외교", "와일드카드", "암흑"],
            provisional_tabs=[],
            capture_geometry={"region_w": 1600, "region_h": 900, "x_offset": 100, "y_offset": 50},
            calibration_complete=True,
        )

        self.ctx.advance_turn(primitive_used="policy_primitive", success=True)

        cache = self.ctx.get_policy_tab_cache()
        assert cache.is_full() is True
        assert cache.positions["경제"].screen_x == 260
        assert cache.capture_geometry is not None
        assert cache.capture_geometry.region_w == 1600

    def test_reset_instance_clears_policy_tab_cache(self):
        self.ctx.replace_policy_tab_cache(
            positions={"군사": {"screen_x": 200, "screen_y": 100, "confirmed": True}},
            confirmed_tabs=["군사"],
            provisional_tabs=[],
            capture_geometry={"region_w": 1600, "region_h": 900, "x_offset": 100, "y_offset": 50},
            calibration_complete=False,
        )

        ContextManager.reset_instance()
        fresh = ContextManager.get_instance()

        assert fresh.get_policy_tab_cache().positions == {}

    def test_policy_tab_cache_keeps_geometry_metadata(self):
        self.ctx.replace_policy_tab_cache(
            positions={"군사": {"screen_x": 6940, "screen_y": 2680, "confirmed": True}},
            confirmed_tabs=["군사"],
            provisional_tabs=[],
            capture_geometry={"region_w": 2560, "region_h": 1440, "x_offset": 300, "y_offset": 120},
            calibration_complete=False,
        )

        cache = self.ctx.get_policy_tab_cache()

        assert cache.capture_geometry is not None
        assert cache.capture_geometry.region_w == 2560
        assert cache.capture_geometry.x_offset == 300
