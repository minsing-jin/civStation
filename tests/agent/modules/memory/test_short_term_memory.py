"""Unit tests for the new choice-catalog short-term memory."""

from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory


class TestShortTermMemoryChoiceCatalog:
    def test_remembers_visible_choices_and_anchor(self):
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        memory.remember_choices(
            [
                {"label": "신성한 불꽃", "note": "문화 보너스"},
                {"label": "풍요의 의식", "disabled": True},
            ],
            end_of_list=False,
            scroll_anchor={"x": 420, "y": 500, "left": 250, "top": 150, "right": 650, "bottom": 900},
        )

        assert len(memory.choice_catalog.candidates) == 2
        assert memory.get_scroll_anchor() is not None
        assert memory.get_scroll_anchor().x == 420
        assert memory.choice_catalog.end_reached is False

    def test_rejects_non_normalized_scroll_anchor(self):
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", normalizing_range=500, enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        memory.remember_choices(
            [{"label": "신성한 불꽃"}],
            end_of_list=False,
            scroll_anchor={"x": 620, "y": 500, "left": 250, "top": 150, "right": 650, "bottom": 900},
        )

        assert memory.get_scroll_anchor() is None

    def test_marks_previous_choices_as_above_after_scrolling_down(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        memory.remember_choices([{"label": "기념비"}], end_of_list=False)
        memory.remember_choices([{"label": "개척자"}], end_of_list=True, scroll_direction="down")

        monument = next(v for v in memory.choice_catalog.candidates.values() if v.label == "기념비")
        settler = next(v for v in memory.choice_catalog.candidates.values() if v.label == "개척자")

        assert monument.position_hint == "above"
        assert settler.position_hint == "visible"
        assert memory.choice_catalog.end_reached is True

    def test_restore_last_checkpoint_restores_choice_catalog(self):
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        memory.remember_choices([{"label": "신성한 불꽃"}], end_of_list=True)
        memory.set_best_choice(label="신성한 불꽃", reason="문화 중심 전략")

        memory.begin_stage("choose_from_memory")
        memory.remember_choices([{"label": "사냥의 여신"}], end_of_list=False)

        restored = memory.restore_last_checkpoint()

        assert restored is True
        assert "신성한 불꽃" in {v.label for v in memory.choice_catalog.candidates.values()}

    def test_prompt_summary_keeps_all_choice_catalog_candidates(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        for idx in range(10):
            memory.remember_choices(
                [{"label": f"후보{idx}"}],
                end_of_list=idx == 9,
                scroll_direction="down",
            )

        summary = memory.to_prompt_string()

        assert "[choice_catalog] 확인한 후보 10개" in summary
        assert "- 후보0" in summary
        assert "- 후보9" in summary

    def test_choice_catalog_decision_prompt_includes_candidate_ids(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")
        memory.remember_choices(
            [
                {"id": "monument", "label": "기념비"},
                {"id": "library", "label": "도서관"},
            ],
            end_of_list=True,
        )

        summary = memory.choice_catalog_decision_prompt()

        assert "id=monument" in summary
        assert "id=library" in summary

    def test_checked_choice_cannot_be_set_as_best_option(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        memory.remember_choices(
            [
                {"label": "곡창", "selected": True},
                {"label": "기념비"},
            ],
            end_of_list=True,
        )
        memory.set_best_choice(label="곡창", reason="이미 지은 건물을 잘못 고름")

        assert memory.get_best_choice() is None

    def test_city_production_prompt_excludes_disabled_and_checked_choices(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")

        memory.remember_choices(
            [
                {"id": "campus", "label": "캠퍼스", "disabled": True},
                {"id": "granary", "label": "곡창", "selected": True},
                {"id": "monument", "label": "기념비"},
            ],
            end_of_list=True,
        )

        decision_prompt = memory.choice_catalog_decision_prompt()
        summary = memory.to_prompt_string()

        assert "[choice_catalog] 확인한 후보 1개" in decision_prompt
        assert "기념비" in decision_prompt
        assert "캠퍼스" not in decision_prompt
        assert "곡창" not in decision_prompt
        assert "[choice_catalog] 확인한 후보 1개" in summary
        assert "- 기념비" in summary
        assert "캠퍼스" not in summary
        assert "곡창" not in summary

    def test_policy_state_rejects_negative_absolute_coordinates(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", normalizing_range=500, enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "screen_x": -1, "screen_y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )

        assert memory.policy_state.tab_positions == {}
        assert "military_1" in memory.policy_state.slot_inventory

    def test_policy_slot_accepts_matching_category_or_wild(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", normalizing_range=500, enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 200, "y": 100},
                {"tab_name": "경제", "x": 260, "y": 100},
            ],
            eligible_tabs_queue=["경제", "군사"],
            slot_inventory=[
                {"slot_id": "military_1", "slot_type": "군사", "is_empty": True},
                {"slot_id": "economic_1", "slot_type": "경제", "is_empty": True},
                {"slot_id": "wild_1", "slot_type": "와일드카드", "is_empty": True, "is_wild": True},
            ],
            wild_slot_active=True,
        )

        military_slot = memory.policy_state.slot_inventory["military_1"]
        economic_slot = memory.policy_state.slot_inventory["economic_1"]
        wild_slot = memory.policy_state.slot_inventory["wild_1"]

        assert memory.policy_slot_accepts_source_tab("경제", military_slot) is False
        assert memory.policy_slot_accepts_source_tab("경제", economic_slot) is True
        assert memory.policy_slot_accepts_source_tab("경제", wild_slot) is True

    def test_policy_remaining_queue_excludes_current_tab(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 200, "y": 100},
                {"tab_name": "경제", "x": 260, "y": 100},
                {"tab_name": "외교", "x": 320, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제", "외교"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )

        assert memory.get_policy_current_tab_name() == "군사"
        assert memory.get_policy_next_tab_name() == "경제"
        assert memory.get_policy_next_tab_position() is not None
        assert memory.get_policy_next_tab_position().screen_x == 260
        assert memory.get_policy_remaining_queue() == ["경제", "외교"]

    def test_policy_bootstrap_restart_preserves_progress(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 200, "y": 100},
                {"tab_name": "경제", "x": 260, "y": 100},
                {"tab_name": "외교", "x": 320, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제", "외교"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()

        memory.clear_policy_bootstrap(preserve_entry_done=True, preserve_progress=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 201, "y": 101},
                {"tab_name": "경제", "x": 261, "y": 101},
                {"tab_name": "외교", "x": 321, "y": 101},
            ],
            eligible_tabs_queue=["군사", "경제", "외교"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )

        assert memory.get_policy_current_tab_name() == "경제"
        assert memory.policy_state.completed_tabs == ["군사"]

    def test_policy_prompt_summary_includes_last_popped_tab(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 200, "y": 100},
                {"tab_name": "경제", "x": 260, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )

        memory.mark_policy_tab_completed("군사")
        summary = memory.to_prompt_string()

        assert "last_popped=군사" in summary

    def test_policy_slot_selection_persists_slot_source_and_reason(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 200, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "wild_1", "slot_type": "와일드카드", "is_empty": True, "is_wild": True}],
            wild_slot_active=True,
        )

        memory.mark_policy_slot_selected(
            card_name="도시 계획",
            source_tab="경제",
            target_slot_id="wild_1",
            reasoning="과학 승리 기준으로 도시 성장 보너스가 더 중요",
        )

        slot = memory.policy_state.slot_inventory["wild_1"]
        assert slot.current_card_name == "도시 계획"
        assert slot.selected_from_tab == "경제"
        assert slot.selection_reason == "과학 승리 기준으로 도시 성장 보너스가 더 중요"

    def test_seed_policy_tab_cache_loads_confirmed_positions_when_geometry_matches(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.set_policy_capture_geometry(1600, 900, 100, 50)

        memory.seed_policy_tab_cache(
            {
                "positions": {
                    "군사": {"screen_x": 820, "screen_y": 310, "confirmed": True},
                    "경제": {"screen_x": 880, "screen_y": 310, "confirmed": True},
                },
                "confirmed_tabs": ["군사", "경제"],
                "provisional_tabs": [],
                "capture_geometry": {"region_w": 1600, "region_h": 900, "x_offset": 100, "y_offset": 50},
            }
        )

        assert memory.policy_state.cache_source == "session_cache"
        assert memory.policy_state.tab_positions["군사"].confirmed is True
        assert memory.policy_state.tab_positions["경제"].screen_x == 880

    def test_seed_policy_tab_cache_rejects_missing_geometry_metadata(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.set_policy_capture_geometry(1600, 900, 100, 50)

        memory.seed_policy_tab_cache(
            {
                "positions": {
                    "군사": {"screen_x": 820, "screen_y": 310, "confirmed": True},
                },
                "confirmed_tabs": ["군사"],
                "provisional_tabs": [],
            }
        )

        assert memory.policy_state.tab_positions == {}

    def test_seed_policy_tab_cache_rejects_mismatched_geometry(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.set_policy_capture_geometry(1600, 900, 100, 50)

        memory.seed_policy_tab_cache(
            {
                "positions": {
                    "군사": {"screen_x": 820, "screen_y": 310, "confirmed": True},
                },
                "confirmed_tabs": ["군사"],
                "provisional_tabs": [],
                "capture_geometry": {"region_w": 1600, "region_h": 900, "x_offset": 120, "y_offset": 50},
            }
        )

        assert memory.policy_state.tab_positions == {}

    def test_policy_tab_click_verification_only_required_for_provisional_or_calibration(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 200, "y": 100, "confirmed": True},
                {"tab_name": "경제", "x": 260, "y": 100, "confirmed": True},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("click_next_tab")

        assert memory.should_verify_policy_tab_click() is False

        memory.mark_policy_tab_provisional("경제")

        assert memory.should_verify_policy_tab_click() is True

        memory.policy_state.calibration_pending_tabs = ["경제"]
        memory.begin_stage("calibrate_tabs")

        assert memory.should_verify_policy_tab_click() is True

    def test_clear_policy_bootstrap_preserves_tab_positions_when_requested(self):
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 200, "y": 100, "confirmed": True},
                {"tab_name": "경제", "x": 260, "y": 100, "confirmed": False},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["경제"],
        )

        memory.clear_policy_bootstrap(preserve_entry_done=True, preserve_progress=False, preserve_tab_positions=True)

        assert set(memory.policy_state.tab_positions) == {"군사", "경제"}
        assert memory.policy_state.provisional_tabs == {"경제"}
