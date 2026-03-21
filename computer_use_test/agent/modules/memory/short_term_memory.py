"""
Short-Term Memory for multi-step primitive execution.

The core goal is to support working-memory style tasks:
- remember choices seen while scrolling
- remember where the final best choice is relative to the current viewport
- keep lightweight stage / failure state for process recovery

More capabilities can be added later, but the first formal capability is
the choice catalog used by observation-assisted primitives.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PROMPT_WINDOW = 6
_PROMPT_POLICY_TAB_ORDER = ["군사", "경제", "외교", "와일드카드", "암흑"]


def _slugify(text: str) -> str:
    base = re.sub(r"\s+", "_", text.strip().lower())
    base = re.sub(r"[^0-9a-zA-Z_가-힣]+", "", base)
    return base or "option"


@dataclass
class ActionTrace:
    """A recent action / reasoning trace for prompt summarization."""

    step: int
    stage: str
    action_summary: str
    reasoning: str
    result: str = ""


@dataclass
class ScrollAnchor:
    """Normalized popup/list hover point for safe scrolling."""

    x: int
    y: int
    left: int = 0
    top: int = 0
    right: int = 1000
    bottom: int = 1000

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass
class ChoiceCandidate:
    """One choice remembered from observation."""

    id: str
    label: str
    score: float = 0.0
    visible_now: bool = False
    position_hint: str = "unknown"  # visible | above | below | unknown
    metadata: dict[str, str | bool | int | float] = field(default_factory=dict)


@dataclass
class ChoiceCatalogState:
    """State for scrollable choice memory."""

    enabled: bool = False
    candidates: dict[str, ChoiceCandidate] = field(default_factory=dict)
    best_option_id: str = ""
    best_option_reason: str = ""
    scroll_anchor: ScrollAnchor | None = None
    end_reached: bool = False
    scan_end_reason: str = ""
    last_scroll_direction: str = "down"
    last_new_candidate_count: int = 0
    last_visible_option_ids: tuple[str, ...] = ()
    downward_scan_scrolls: int = 0
    downward_no_new_streak: int = 0


@dataclass
class CityPlacementState:
    """State for city-production placement follow-up."""

    has_target: bool = False
    target_x: int = 0
    target_y: int = 0
    target_button: str = "right"
    target_reason: str = ""
    target_origin: str = ""  # direct_tile | purchase_button
    target_tile_color: str = ""
    reclick_attempts: int = 0


@dataclass
class PolicyTabPosition:
    """Cached policy tab location."""

    tab_name: str
    screen_x: int
    screen_y: int
    confirmed: bool = False


@dataclass
class PolicyCaptureGeometry:
    """Game-window geometry for policy-screen capture."""

    region_w: int
    region_h: int
    x_offset: int
    y_offset: int

    def matches(self, other: PolicyCaptureGeometry | None) -> bool:
        return other is not None and (
            self.region_w == other.region_w
            and self.region_h == other.region_h
            and self.x_offset == other.x_offset
            and self.y_offset == other.y_offset
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "region_w": self.region_w,
            "region_h": self.region_h,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
        }


@dataclass
class PolicySlotState:
    """One policy slot on the left panel."""

    slot_id: str
    slot_type: str
    current_card_name: str = ""
    is_empty: bool = False
    active: bool = True
    is_wild: bool = False
    allowed_categories: list[str] = field(default_factory=list)
    selected_from_tab: str = ""
    selection_reason: str = ""


@dataclass
class PolicyState:
    """Process-memory for policy primitive."""

    enabled: bool = False
    mode: str = "structured"
    entry_done: bool = False
    bootstrap_complete: bool = False
    bootstrap_failures: int = 0
    bootstrap_summary: str = ""
    last_event: str = ""
    last_bundle_action_count: int = 0
    overview_mode: bool = False
    wild_slot_active: bool = False
    selected_tab_name: str = ""
    visible_tabs: list[str] = field(default_factory=list)
    tab_positions: dict[str, PolicyTabPosition] = field(default_factory=dict)
    capture_geometry: PolicyCaptureGeometry | None = None
    cache_geometry: PolicyCaptureGeometry | None = None
    provisional_tabs: set[str] = field(default_factory=set)
    calibration_pending_tabs: list[str] = field(default_factory=list)
    eligible_tabs_queue: list[str] = field(default_factory=list)
    current_tab_index: int = 0
    slot_inventory: dict[str, PolicySlotState] = field(default_factory=dict)
    completed_tabs: list[str] = field(default_factory=list)
    last_popped_tab: str = ""
    tab_failure_counts: dict[str, int] = field(default_factory=dict)
    generic_fallback_used: set[str] = field(default_factory=set)
    restart_current_tab_index: int = 0
    restart_completed_tabs: list[str] = field(default_factory=list)
    cache_source: str = ""
    last_similarity_result: str = ""
    last_tab_check_result: str = ""
    last_relocalize_result: str = ""
    distinct_failed_tabs: set[str] = field(default_factory=set)


@dataclass
class VotingState:
    """Process-memory for the world-congress voting primitive."""

    enabled: bool = False
    current_agenda_id: str = ""
    current_agenda_label: str = ""
    selected_resolution: str = ""
    selected_vote_direction: str = ""
    selected_target_label: str = ""
    completed_agenda_ids: list[str] = field(default_factory=list)


@dataclass
class GovernorState:
    """Process-memory for governor target tracking across subflows."""

    target_governor_id: str = ""
    target_governor_label: str = ""
    target_governor_note: str = ""


@dataclass
class FailureCheckpoint:
    """Stable restore point for observation-assisted recovery."""

    stage: str = ""
    branch: str = ""
    step_count: int = 0
    choice_catalog: ChoiceCatalogState = field(default_factory=ChoiceCatalogState)
    city_placement_state: CityPlacementState = field(default_factory=CityPlacementState)
    policy_state: PolicyState = field(default_factory=PolicyState)
    voting_state: VotingState = field(default_factory=VotingState)
    governor_state: GovernorState = field(default_factory=GovernorState)
    completed_substeps: list[str] = field(default_factory=list)
    action_log: list[ActionTrace] = field(default_factory=list)
    stage_failure_counts: dict[str, int] = field(default_factory=dict)
    stage_fallback_used: set[str] = field(default_factory=set)
    fallback_return_stage: str = ""
    fallback_return_key: str = ""
    last_semantic_verify: str = ""
    last_observation_summary: str = ""
    last_observation_anchor: str = ""
    last_planned_action: str = ""
    last_executed_action: str = ""
    task_hitl_directive: str = ""
    task_hitl_status: str = ""
    task_hitl_reason: str = ""


@dataclass
class ShortTermMemory:
    """Per-task working memory."""

    primitive_name: str = ""
    normalizing_range: int = 1000
    current_stage: str = ""
    branch: str = ""
    step_count: int = 0
    max_steps: int = 10
    failure_count: int = 0
    completed_substeps: list[str] = field(default_factory=list)
    action_log: list[ActionTrace] = field(default_factory=list)
    choice_catalog: ChoiceCatalogState = field(default_factory=ChoiceCatalogState)
    city_placement_state: CityPlacementState = field(default_factory=CityPlacementState)
    policy_state: PolicyState = field(default_factory=PolicyState)
    voting_state: VotingState = field(default_factory=VotingState)
    governor_state: GovernorState = field(default_factory=GovernorState)
    stage_failure_counts: dict[str, int] = field(default_factory=dict)
    stage_fallback_used: set[str] = field(default_factory=set)
    fallback_return_stage: str = ""
    fallback_return_key: str = ""
    last_semantic_verify: str = ""
    last_stable_checkpoint: FailureCheckpoint | None = None
    last_observation_summary: str = ""
    last_observation_anchor: str = ""
    last_planned_action: str = ""
    last_executed_action: str = ""
    task_hitl_directive: str = ""
    task_hitl_status: str = ""
    task_hitl_reason: str = ""

    def start_task(
        self,
        primitive_name: str,
        max_steps: int = 10,
        *,
        normalizing_range: int = 1000,
        enable_choice_catalog: bool = False,
        enable_policy_state: bool = False,
        enable_voting_state: bool = False,
    ) -> None:
        """Initialize memory for a new task."""
        self.reset()
        self.primitive_name = primitive_name
        self.max_steps = max_steps
        self.normalizing_range = normalizing_range
        self.choice_catalog.enabled = enable_choice_catalog
        self.policy_state.enabled = enable_policy_state
        self.voting_state.enabled = enable_voting_state
        logger.debug(
            "ShortTermMemory started: %s (max_steps=%s, choice_catalog=%s, policy_state=%s, voting_state=%s)",
            primitive_name,
            max_steps,
            enable_choice_catalog,
            enable_policy_state,
            enable_voting_state,
        )

    def reset(self) -> None:
        """Clear all state."""
        self.primitive_name = ""
        self.normalizing_range = 1000
        self.current_stage = ""
        self.branch = ""
        self.step_count = 0
        self.max_steps = 10
        self.failure_count = 0
        self.completed_substeps = []
        self.action_log = []
        self.choice_catalog = ChoiceCatalogState()
        self.city_placement_state = CityPlacementState()
        self.policy_state = PolicyState()
        self.voting_state = VotingState()
        self.governor_state = GovernorState()
        self.stage_failure_counts = {}
        self.stage_fallback_used = set()
        self.fallback_return_stage = ""
        self.fallback_return_key = ""
        self.last_semantic_verify = ""
        self.last_stable_checkpoint = None
        self.last_observation_summary = ""
        self.last_observation_anchor = ""
        self.last_planned_action = ""
        self.last_executed_action = ""
        self.task_hitl_directive = ""
        self.task_hitl_status = ""
        self.task_hitl_reason = ""

    def begin_stage(self, stage: str) -> None:
        """Update the current stage label."""
        self.current_stage = stage

    def set_task_hitl_directive(self, directive: str, *, reason: str = "") -> None:
        """Store one task-local HITL directive until the current task ends."""
        normalized = directive.strip()
        if not normalized:
            return
        self.task_hitl_directive = normalized
        self.task_hitl_status = "pending"
        self.task_hitl_reason = reason.strip()

    def get_task_hitl_directive(self) -> str:
        """Return the current task-local HITL directive, if any."""
        return self.task_hitl_directive.strip()

    def clear_task_hitl_directive(self) -> None:
        """Drop any task-local HITL directive."""
        self.task_hitl_directive = ""
        self.task_hitl_status = ""
        self.task_hitl_reason = ""

    def resolve_task_hitl_choice_candidate(self) -> tuple[ChoiceCandidate | None, str]:
        """Try to map the current task-local HITL directive to one selectable candidate."""
        directive = self.get_task_hitl_directive()
        if not directive:
            return None, ""

        normalized_directive = _slugify(directive)
        matches: list[ChoiceCandidate] = []
        for candidate in self._selectable_choice_candidates():
            label_key = _slugify(candidate.label)
            id_key = _slugify(candidate.id)
            keys = [key for key in (label_key, id_key) if len(key) >= 2]
            if any(key in normalized_directive for key in keys):
                matches.append(candidate)

        if len(matches) == 1:
            reason = f"task HITL matched candidate '{matches[0].label}'"
            self.task_hitl_status = "applied"
            self.task_hitl_reason = reason
            return matches[0], reason

        self.task_hitl_status = "ignored"
        if len(matches) > 1:
            labels = ", ".join(candidate.label for candidate in matches[:3])
            self.task_hitl_reason = f"task HITL ambiguous across candidates: {labels}"
        else:
            self.task_hitl_reason = "task HITL did not match any selectable candidate"
        return None, ""

    def set_branch(self, branch: str) -> None:
        """Persist the active process branch."""
        self.branch = branch

    def set_governor_target(self, *, option_id: str = "", label: str = "", note: str = "") -> None:
        """Persist the selected governor so later subflows can reuse it."""
        self.governor_state.target_governor_id = option_id.strip()
        self.governor_state.target_governor_label = label.strip()
        self.governor_state.target_governor_note = note.strip()

    def get_governor_target_label(self) -> str:
        """Return the saved governor label, if any."""
        return self.governor_state.target_governor_label.strip()

    def clear_governor_target(self) -> None:
        """Drop saved governor target state."""
        self.governor_state = GovernorState()

    def mark_substep(self, substep: str) -> None:
        """Record completion of a process substep once."""
        if substep not in self.completed_substeps:
            self.completed_substeps.append(substep)

    def add_observation(self, reasoning: str, action_summary: str, result: str = "", stage: str | None = None) -> None:
        """Record a completed action / decision trace."""
        self.step_count += 1
        self.action_log.append(
            ActionTrace(
                step=self.step_count,
                stage=stage or self.current_stage or "unknown",
                action_summary=action_summary,
                reasoning=reasoning,
                result=result,
            )
        )
        logger.debug("STM step %s (%s): %s", self.step_count, stage or self.current_stage, action_summary)

    def set_last_observation_debug(
        self,
        summary: str,
        *,
        scroll_anchor: dict | ScrollAnchor | None = None,
    ) -> None:
        """Persist the latest observation summary for prompts and Rich debug."""
        self.last_observation_summary = summary.strip()
        self.last_observation_anchor = self._format_scroll_anchor(scroll_anchor)

    def set_last_planned_action_debug(self, summary: str) -> None:
        """Persist the latest planned action summary for prompts and Rich debug."""
        self.last_planned_action = summary.strip()

    def set_last_executed_action_debug(self, summary: str) -> None:
        """Persist the latest executed action summary for prompts and Rich debug."""
        self.last_executed_action = summary.strip()

    def recent_actions_prompt(self) -> str:
        """Summarize recent task-local actions."""
        if not self.action_log:
            return "없음"

        traces = self.action_log[-_PROMPT_WINDOW:]
        lines = []
        for trace in traces:
            line = f"[{trace.step}:{trace.stage}] {trace.action_summary}"
            if trace.result:
                line += f" -> {trace.result}"
            lines.append(line)
        return "\n".join(lines)

    def remember_choices(
        self,
        visible_options: list[dict],
        *,
        end_of_list: bool,
        scroll_anchor: dict | ScrollAnchor | None = None,
        scroll_direction: str = "down",
    ) -> None:
        """Merge newly observed choices into the choice catalog."""
        if not self.choice_catalog.enabled:
            return

        self.choice_catalog.last_scroll_direction = scroll_direction

        if isinstance(scroll_anchor, dict):
            normalized_anchor = self._build_scroll_anchor(scroll_anchor)
            if normalized_anchor is not None:
                self.choice_catalog.scroll_anchor = normalized_anchor
        elif isinstance(scroll_anchor, ScrollAnchor):
            if self._is_normalized_rect(
                scroll_anchor.left,
                scroll_anchor.top,
                scroll_anchor.right,
                scroll_anchor.bottom,
            ) and self._is_normalized_coord(scroll_anchor.x, scroll_anchor.y):
                self.choice_catalog.scroll_anchor = scroll_anchor

        current_ids: set[str] = set()
        raw_visible_ids: set[str] = set()
        new_candidate_count = 0
        visible_option_ids: list[str] = []
        for raw in visible_options:
            label = str(raw.get("label", "")).strip()
            if not label:
                continue

            raw_option_id = str(raw.get("id", "")).strip() or _slugify(label)
            option_id = self._resolve_observed_choice_id(raw_option_id, label)
            raw_visible_ids.add(option_id)
            candidate = self.choice_catalog.candidates.get(option_id)
            was_selectable = self._candidate_is_selectable(candidate) if candidate is not None else False
            if candidate is None:
                candidate = ChoiceCandidate(id=option_id, label=label)
                self.choice_catalog.candidates[option_id] = candidate

            candidate.label = label

            raw_score = raw.get("score")
            try:
                if raw_score is not None:
                    candidate.score = max(candidate.score, float(raw_score))
            except (TypeError, ValueError):
                pass

            metadata = {
                "disabled": bool(raw.get("disabled", False)),
                "selected": bool(raw.get("selected", False)),
                "built": bool(raw.get("built", raw.get("selected", False))),
                "note": str(raw.get("note", "")).strip(),
            }
            metadata = {k: v for k, v in metadata.items() if v not in ("", None)}
            candidate.metadata.update(metadata)

            if self._raw_choice_is_selectable(raw):
                visible_option_ids.append(option_id)
                current_ids.add(option_id)
                candidate.visible_now = True
                candidate.position_hint = "visible"
                if not was_selectable:
                    new_candidate_count += 1
            else:
                candidate.visible_now = False
                candidate.position_hint = "unknown"

        for option_id, candidate in self.choice_catalog.candidates.items():
            if option_id in current_ids:
                continue
            if option_id in raw_visible_ids and not self._candidate_is_selectable(candidate):
                candidate.visible_now = False
                candidate.position_hint = "unknown"
                continue
            if candidate.visible_now:
                candidate.visible_now = False
                candidate.position_hint = "above" if scroll_direction == "down" else "below"

        self.choice_catalog.last_new_candidate_count = new_candidate_count
        self.choice_catalog.last_visible_option_ids = tuple(visible_option_ids)
        if not self.choice_catalog.end_reached:
            self.choice_catalog.scan_end_reason = ""

        if scroll_direction == "down" and self.choice_catalog.downward_scan_scrolls > 0:
            if new_candidate_count == 0:
                self.choice_catalog.downward_no_new_streak += 1
            else:
                self.choice_catalog.downward_no_new_streak = 0
        else:
            self.choice_catalog.downward_no_new_streak = 0

        if end_of_list:
            self.choice_catalog.end_reached = True
            self.choice_catalog.scan_end_reason = "observer_end_of_list"
            self.choice_catalog.downward_no_new_streak = 0
            for option_id, candidate in self.choice_catalog.candidates.items():
                if option_id not in current_ids and candidate.position_hint == "unknown":
                    candidate.position_hint = "above"
        elif (
            scroll_direction == "down"
            and self.choice_catalog.downward_scan_scrolls >= 3
            and self.choice_catalog.downward_no_new_streak >= 3
        ):
            self.choice_catalog.end_reached = True
            self.choice_catalog.scan_end_reason = "down_scroll_no_new_candidates"

        self.capture_checkpoint()

    def _resolve_observed_choice_id(self, option_id: str, label: str) -> str:
        """Collapse unstable observer ids back to one catalog key when safe."""
        if option_id in self.choice_catalog.candidates:
            return option_id
        if self.primitive_name != "city_production_primitive":
            return option_id

        normalized_label = _slugify(label)
        label_matches = [
            candidate.id
            for candidate in self.choice_catalog.candidates.values()
            if candidate.label == label or _slugify(candidate.label) == normalized_label
        ]
        if len(label_matches) == 1:
            return label_matches[0]
        return option_id

    def set_best_choice(
        self,
        *,
        option_id: str | None = None,
        label: str | None = None,
        reason: str = "",
    ) -> None:
        """Persist the current best option chosen from memory."""
        if option_id is None and label is not None:
            normalized = _slugify(label)
            for candidate in self.choice_catalog.candidates.values():
                if candidate.id == normalized or candidate.label == label:
                    option_id = candidate.id
                    break
        if option_id is None:
            return

        candidate = self.choice_catalog.candidates.get(option_id)
        if candidate is None or not self._candidate_is_selectable(candidate):
            return

        self.choice_catalog.best_option_id = option_id
        self.choice_catalog.best_option_reason = reason

    def get_best_choice(self) -> ChoiceCandidate | None:
        """Return the current best choice, if any."""
        option_id = self.choice_catalog.best_option_id
        if not option_id:
            return None
        candidate = self.choice_catalog.candidates.get(option_id)
        if candidate is None or not self._candidate_is_selectable(candidate):
            return None
        return candidate

    @staticmethod
    def _raw_choice_is_selectable(raw: dict) -> bool:
        """Whether a visible raw choice can be acted on right now."""
        selected = bool(raw.get("selected", False))
        return not bool(raw.get("disabled", False)) and not selected and not bool(raw.get("built", selected))

    @staticmethod
    def _candidate_is_selectable(candidate: ChoiceCandidate) -> bool:
        """Checked/disabled catalog entries are not valid final choices."""
        return (
            not bool(candidate.metadata.get("disabled"))
            and not bool(candidate.metadata.get("selected"))
            and not bool(candidate.metadata.get("built"))
        )

    def register_choice_scroll(self, *, direction: str) -> None:
        """Persist successful scan scroll direction for later observation heuristics."""
        self.choice_catalog.last_scroll_direction = direction
        if direction == "down":
            self.choice_catalog.downward_scan_scrolls += 1

    def choice_catalog_decision_max_tokens(self) -> int:
        """Return a token budget sized for whole-catalog decision prompts."""
        candidate_count = len(self._selectable_choice_candidates())
        return min(4096, max(1024, 512 + candidate_count * 96))

    @staticmethod
    def _format_choice_candidate_line(candidate: ChoiceCandidate, *, include_id: bool) -> str:
        flags = []
        if candidate.visible_now:
            flags.append("보임")
        elif candidate.position_hint != "unknown":
            flags.append(candidate.position_hint)
        if candidate.metadata.get("disabled"):
            flags.append("비활성")
        if candidate.metadata.get("selected"):
            flags.append("체크됨")
        if candidate.metadata.get("built"):
            flags.append("이미 지음")
        note = str(candidate.metadata.get("note", "")).strip()
        suffix = f" / {note}" if note else ""
        flag_text = f" ({', '.join(flags)})" if flags else ""
        if include_id:
            return f"- id={candidate.id} | label={candidate.label}{flag_text}{suffix}"
        return f"- {candidate.label}{flag_text}{suffix}"

    def choice_catalog_decision_prompt(self) -> str:
        """Return the full choice catalog with stable ids for memory-only decision calls."""
        if not self.choice_catalog.enabled:
            return "choice catalog disabled"

        selectable_candidates = self._selectable_choice_candidates()
        lines = [
            f"[choice_catalog] 확인한 후보 {len(selectable_candidates)}개 / 목록끝={self.choice_catalog.end_reached}"
        ]
        for candidate in selectable_candidates:
            lines.append(self._format_choice_candidate_line(candidate, include_id=True))
        return "\n".join(lines)

    def get_scroll_anchor(self) -> ScrollAnchor | None:
        """Return the saved hover anchor for scrollable popups."""
        return self.choice_catalog.scroll_anchor

    def reset_choice_catalog(self) -> None:
        """Clear choice-catalog contents while preserving enablement."""
        enabled = self.choice_catalog.enabled
        self.choice_catalog = ChoiceCatalogState(enabled=enabled)

    def remember_city_placement_target(
        self,
        *,
        x: int,
        y: int,
        button: str = "right",
        reason: str = "",
        origin: str = "",
        tile_color: str = "",
    ) -> None:
        """Persist the most recent placement follow-up target."""
        if not self._is_normalized_coord(x, y, normalizing_range=self.normalizing_range):
            return
        self.city_placement_state.target_x = x
        self.city_placement_state.target_y = y
        self.city_placement_state.target_button = button or "right"
        self.city_placement_state.target_reason = reason.strip()
        self.city_placement_state.target_origin = origin.strip()
        self.city_placement_state.target_tile_color = tile_color.strip()
        self.city_placement_state.reclick_attempts = 0
        self.city_placement_state.has_target = True

    def get_city_placement_target(self) -> tuple[int, int, str] | None:
        """Return the saved placement target, if present."""
        if not self.city_placement_state.has_target:
            return None
        if not self._is_normalized_coord(
            self.city_placement_state.target_x,
            self.city_placement_state.target_y,
            normalizing_range=self.normalizing_range,
        ):
            return None
        return (
            self.city_placement_state.target_x,
            self.city_placement_state.target_y,
            self.city_placement_state.target_button or "right",
        )

    def bump_city_placement_reclick_attempt(self) -> int:
        """Record one automatic re-click attempt for a purchased blue tile."""
        self.city_placement_state.reclick_attempts += 1
        return self.city_placement_state.reclick_attempts

    def clear_city_placement_target(self) -> None:
        """Reset placement follow-up state."""
        self.city_placement_state = CityPlacementState()

    # ------------------------------------------------------------------
    # Voting process-memory helpers
    # ------------------------------------------------------------------

    def init_voting_state(self) -> None:
        """Initialize or reset world-congress voting task state."""
        self.voting_state.enabled = True
        self.voting_state.current_agenda_id = ""
        self.voting_state.current_agenda_label = ""
        self.voting_state.selected_resolution = ""
        self.voting_state.selected_vote_direction = ""
        self.voting_state.selected_target_label = ""
        self.voting_state.completed_agenda_ids = []

    def set_current_voting_agenda(self, *, option_id: str, label: str | None = None) -> None:
        """Persist the agenda currently being processed."""
        candidate = self.choice_catalog.candidates.get(option_id)
        self.voting_state.enabled = True
        self.voting_state.current_agenda_id = option_id
        self.voting_state.current_agenda_label = label or (candidate.label if candidate is not None else option_id)
        self.voting_state.selected_resolution = ""
        self.voting_state.selected_vote_direction = ""
        self.voting_state.selected_target_label = ""

    def clear_current_voting_agenda(self) -> None:
        """Clear the active world-congress agenda while keeping completion history."""
        self.voting_state.current_agenda_id = ""
        self.voting_state.current_agenda_label = ""
        self.voting_state.selected_resolution = ""
        self.voting_state.selected_vote_direction = ""
        self.voting_state.selected_target_label = ""

    def mark_current_voting_resolution(self, selection: str) -> None:
        """Persist the chosen A/B resolution branch."""
        self.voting_state.selected_resolution = selection.strip()

    def mark_current_voting_direction(self, selection: str) -> None:
        """Persist the chosen vote direction."""
        self.voting_state.selected_vote_direction = selection.strip()

    def mark_current_voting_target(self, label: str) -> None:
        """Persist the chosen target label for the active agenda."""
        self.voting_state.selected_target_label = label.strip()

    def mark_current_voting_agenda_complete(self) -> None:
        """Mark the active agenda as complete and exclude it from future picks."""
        option_id = self.voting_state.current_agenda_id.strip()
        if not option_id:
            return
        if option_id not in self.voting_state.completed_agenda_ids:
            self.voting_state.completed_agenda_ids.append(option_id)
        candidate = self.choice_catalog.candidates.get(option_id)
        if candidate is not None:
            candidate.metadata["selected"] = True
            candidate.visible_now = False
            candidate.position_hint = "unknown"
        if self.choice_catalog.best_option_id == option_id:
            self.choice_catalog.best_option_id = ""
            self.choice_catalog.best_option_reason = ""
        self.clear_current_voting_agenda()

    def get_next_pending_voting_agenda(self) -> ChoiceCandidate | None:
        """Return the next uncompleted, selectable agenda in scan order."""
        if not self.voting_state.enabled:
            return None
        completed = set(self.voting_state.completed_agenda_ids)
        for candidate in self.choice_catalog.candidates.values():
            if candidate.id in completed:
                continue
            if self._candidate_is_selectable(candidate):
                return candidate
        return None

    def capture_checkpoint(self) -> None:
        """Store the latest stable restore point."""
        self.last_stable_checkpoint = FailureCheckpoint(
            stage=self.current_stage,
            branch=self.branch,
            step_count=self.step_count,
            choice_catalog=copy.deepcopy(self.choice_catalog),
            city_placement_state=copy.deepcopy(self.city_placement_state),
            policy_state=copy.deepcopy(self.policy_state),
            voting_state=copy.deepcopy(self.voting_state),
            governor_state=copy.deepcopy(self.governor_state),
            completed_substeps=list(self.completed_substeps),
            action_log=copy.deepcopy(self.action_log),
            stage_failure_counts=dict(self.stage_failure_counts),
            stage_fallback_used=set(self.stage_fallback_used),
            fallback_return_stage=self.fallback_return_stage,
            fallback_return_key=self.fallback_return_key,
            last_semantic_verify=self.last_semantic_verify,
            last_observation_summary=self.last_observation_summary,
            last_observation_anchor=self.last_observation_anchor,
            last_planned_action=self.last_planned_action,
            last_executed_action=self.last_executed_action,
            task_hitl_directive=self.task_hitl_directive,
            task_hitl_status=self.task_hitl_status,
            task_hitl_reason=self.task_hitl_reason,
        )

    def restore_last_checkpoint(self) -> bool:
        """Rollback to the latest stable observation checkpoint."""
        if self.last_stable_checkpoint is None:
            return False

        checkpoint = self.last_stable_checkpoint
        self.current_stage = checkpoint.stage
        self.branch = checkpoint.branch
        self.step_count = checkpoint.step_count
        self.choice_catalog = copy.deepcopy(checkpoint.choice_catalog)
        self.city_placement_state = copy.deepcopy(checkpoint.city_placement_state)
        self.policy_state = copy.deepcopy(checkpoint.policy_state)
        self.voting_state = copy.deepcopy(checkpoint.voting_state)
        self.governor_state = copy.deepcopy(checkpoint.governor_state)
        self.completed_substeps = list(checkpoint.completed_substeps)
        self.action_log = copy.deepcopy(checkpoint.action_log)
        self.stage_failure_counts = dict(checkpoint.stage_failure_counts)
        self.stage_fallback_used = set(checkpoint.stage_fallback_used)
        self.fallback_return_stage = checkpoint.fallback_return_stage
        self.fallback_return_key = checkpoint.fallback_return_key
        self.last_semantic_verify = checkpoint.last_semantic_verify
        self.last_observation_summary = checkpoint.last_observation_summary
        self.last_observation_anchor = checkpoint.last_observation_anchor
        self.last_planned_action = checkpoint.last_planned_action
        self.last_executed_action = checkpoint.last_executed_action
        self.task_hitl_directive = checkpoint.task_hitl_directive
        self.task_hitl_status = checkpoint.task_hitl_status
        self.task_hitl_reason = checkpoint.task_hitl_reason
        self.failure_count = 0
        logger.debug("STM restored checkpoint for %s at stage=%s", self.primitive_name, self.current_stage)
        return True

    def is_max_steps_reached(self) -> bool:
        """Check if the max action count has been reached."""
        return self.step_count >= self.max_steps

    # ------------------------------------------------------------------
    # Policy process-memory helpers
    # ------------------------------------------------------------------

    def init_policy_state(
        self,
        *,
        tab_positions: list[dict] | dict[str, dict],
        eligible_tabs_queue: list[str],
        slot_inventory: list[dict],
        wild_slot_active: bool,
        overview_mode: bool = False,
        visible_tabs: list[str] | None = None,
        selected_tab_name: str = "",
        provisional_tabs: list[str] | set[str] | None = None,
        calibration_pending_tabs: list[str] | None = None,
        cache_source: str = "",
        capture_geometry: PolicyCaptureGeometry | dict[str, int] | None = None,
    ) -> None:
        """Initialize policy tab cache, slot inventory, and execution queue."""
        resume_index = self.policy_state.restart_current_tab_index
        resume_completed = list(self.policy_state.restart_completed_tabs)
        self.policy_state.enabled = True
        self.policy_state.mode = "structured"
        self.policy_state.bootstrap_complete = True
        self.policy_state.bootstrap_failures = 0
        self.policy_state.bootstrap_summary = ""
        self.policy_state.last_event = ""
        self.policy_state.last_bundle_action_count = 0
        self.policy_state.overview_mode = overview_mode
        self.policy_state.wild_slot_active = wild_slot_active
        self.policy_state.selected_tab_name = selected_tab_name
        self.policy_state.visible_tabs = list(visible_tabs or [])
        self.policy_state.provisional_tabs = {str(tab) for tab in (provisional_tabs or [])}
        self.policy_state.calibration_pending_tabs = list(calibration_pending_tabs or [])
        self.policy_state.current_tab_index = 0
        self.policy_state.completed_tabs = []
        self.policy_state.last_popped_tab = ""
        self.policy_state.tab_failure_counts = {}
        self.policy_state.generic_fallback_used = set()
        self.policy_state.restart_current_tab_index = 0
        self.policy_state.restart_completed_tabs = []
        self.policy_state.cache_source = cache_source.strip()
        self.policy_state.last_similarity_result = ""
        self.policy_state.last_tab_check_result = ""
        self.policy_state.last_relocalize_result = ""
        self.policy_state.distinct_failed_tabs = set()
        parsed_geometry = self._parse_policy_capture_geometry(capture_geometry)
        if parsed_geometry is not None:
            self.policy_state.cache_geometry = parsed_geometry
        elif self.policy_state.capture_geometry is not None and tab_positions:
            self.policy_state.cache_geometry = copy.deepcopy(self.policy_state.capture_geometry)
        else:
            self.policy_state.cache_geometry = None

        tabs: dict[str, PolicyTabPosition] = {}
        if isinstance(tab_positions, dict):
            iterator = []
            for tab_name, payload in tab_positions.items():
                if isinstance(payload, dict):
                    iterator.append({"tab_name": tab_name, **payload})
        else:
            iterator = [item for item in tab_positions if isinstance(item, dict)]

        for item in iterator:
            tab_name = str(item.get("tab_name", item.get("name", ""))).strip()
            if not tab_name:
                continue
            screen_x = int(item.get("screen_x", item.get("x", 0)))
            screen_y = int(item.get("screen_y", item.get("y", 0)))
            if not self._is_absolute_coord(screen_x, screen_y):
                continue
            tabs[tab_name] = PolicyTabPosition(
                tab_name=tab_name,
                screen_x=screen_x,
                screen_y=screen_y,
                confirmed=bool(item.get("confirmed", False)) and tab_name not in self.policy_state.provisional_tabs,
            )
        self.policy_state.tab_positions = tabs

        slots: dict[str, PolicySlotState] = {}
        for item in slot_inventory:
            if not isinstance(item, dict):
                continue
            slot_id = str(item.get("slot_id", "")).strip()
            slot_type = str(item.get("slot_type", "")).strip()
            if not slot_id or not slot_type:
                continue
            is_wild = bool(item.get("is_wild", False)) or slot_type == "와일드카드"
            allowed = item.get("allowed_categories")
            if isinstance(allowed, list):
                allowed_categories = [str(v) for v in allowed]
            elif is_wild:
                allowed_categories = ["군사", "경제", "외교", "와일드카드", "암흑"]
            else:
                allowed_categories = [slot_type]
            slots[slot_id] = PolicySlotState(
                slot_id=slot_id,
                slot_type=slot_type,
                current_card_name=str(item.get("current_card_name", "")).strip(),
                is_empty=bool(item.get("is_empty", False)),
                active=bool(item.get("active", True)),
                is_wild=is_wild,
                allowed_categories=allowed_categories,
                selected_from_tab=str(item.get("selected_from_tab", "")).strip(),
                selection_reason=str(item.get("selection_reason", "")).strip(),
            )
        self.policy_state.slot_inventory = slots

        queue: list[str] = []
        for tab in eligible_tabs_queue:
            if tab not in queue:
                queue.append(tab)
        self.policy_state.eligible_tabs_queue = queue
        if queue:
            self.policy_state.current_tab_index = max(0, min(resume_index, len(queue)))
            self.policy_state.completed_tabs = [tab for tab in resume_completed if tab in queue]
        self.capture_checkpoint()

    def seed_policy_tab_cache(self, cache_payload) -> None:
        """Seed task-local policy tab positions from a session cache payload."""
        if cache_payload is None:
            return

        positions = getattr(cache_payload, "positions", None)
        if positions is None and isinstance(cache_payload, dict):
            positions = cache_payload.get("positions", {})
        if not isinstance(positions, dict):
            return
        if not positions:
            return

        confirmed_tabs = getattr(cache_payload, "confirmed_tabs", None)
        if confirmed_tabs is None and isinstance(cache_payload, dict):
            confirmed_tabs = cache_payload.get("confirmed_tabs", [])
        provisional_tabs = getattr(cache_payload, "provisional_tabs", None)
        if provisional_tabs is None and isinstance(cache_payload, dict):
            provisional_tabs = cache_payload.get("provisional_tabs", [])
        capture_geometry = getattr(cache_payload, "capture_geometry", None)
        if capture_geometry is None and isinstance(cache_payload, dict):
            capture_geometry = cache_payload.get("capture_geometry")
        cache_geometry = self._parse_policy_capture_geometry(capture_geometry)
        current_geometry = self.policy_state.capture_geometry
        if cache_geometry is None:
            logger.info("Discarded session policy tab cache without capture_geometry metadata")
            return
        if current_geometry is None or not cache_geometry.matches(current_geometry):
            logger.info(
                "Discarded session policy tab cache due to geometry mismatch | cache=%s current=%s",
                self._format_policy_capture_geometry(cache_geometry),
                self._format_policy_capture_geometry(current_geometry),
            )
            return

        self.policy_state.enabled = True
        self.policy_state.cache_source = "session_cache"
        self.policy_state.provisional_tabs = {str(tab) for tab in (provisional_tabs or [])}
        self.policy_state.tab_positions = {}
        self.policy_state.cache_geometry = copy.deepcopy(cache_geometry)

        confirmed_lookup = {str(tab) for tab in (confirmed_tabs or [])}
        for tab_name, raw_entry in positions.items():
            if hasattr(raw_entry, "screen_x") and hasattr(raw_entry, "screen_y"):
                screen_x = int(raw_entry.screen_x)
                screen_y = int(raw_entry.screen_y)
                confirmed = bool(getattr(raw_entry, "confirmed", False))
                provisional = bool(getattr(raw_entry, "provisional", False))
            elif isinstance(raw_entry, dict):
                try:
                    screen_x = int(raw_entry.get("screen_x", 0))
                    screen_y = int(raw_entry.get("screen_y", 0))
                except (TypeError, ValueError):
                    continue
                confirmed = bool(raw_entry.get("confirmed", False))
                provisional = bool(raw_entry.get("provisional", False))
            else:
                continue
            if not self._is_absolute_coord(screen_x, screen_y):
                continue
            normalized_tab = str(tab_name)
            if normalized_tab in confirmed_lookup:
                confirmed = True
            if provisional:
                self.policy_state.provisional_tabs.add(normalized_tab)
                confirmed = False
            self.policy_state.tab_positions[normalized_tab] = PolicyTabPosition(
                tab_name=normalized_tab,
                screen_x=screen_x,
                screen_y=screen_y,
                confirmed=confirmed,
            )

    def export_policy_tab_cache(self) -> dict[str, object]:
        """Export task-local policy tab data into a session-cache payload."""
        return {
            "positions": {
                tab_name: {
                    "tab_name": tab.tab_name,
                    "screen_x": tab.screen_x,
                    "screen_y": tab.screen_y,
                    "confirmed": tab.confirmed,
                    "provisional": tab_name in self.policy_state.provisional_tabs,
                }
                for tab_name, tab in self.policy_state.tab_positions.items()
            },
            "confirmed_tabs": sorted(
                tab_name for tab_name, tab in self.policy_state.tab_positions.items() if tab.confirmed
            ),
            "provisional_tabs": sorted(self.policy_state.provisional_tabs),
            "capture_geometry": self.policy_state.cache_geometry.to_dict()
            if self.policy_state.cache_geometry is not None
            else None,
            "calibration_complete": self.has_full_policy_tab_cache(),
        }

    def set_policy_capture_geometry(self, region_w: int, region_h: int, x_offset: int, y_offset: int) -> None:
        """Persist the current policy-screen capture geometry for conversions."""
        geometry = self._parse_policy_capture_geometry(
            {
                "region_w": region_w,
                "region_h": region_h,
                "x_offset": x_offset,
                "y_offset": y_offset,
            }
        )
        self.policy_state.capture_geometry = geometry

    def policy_cache_matches_current_geometry(self) -> bool:
        """Whether the cached absolute tab positions still match the live game window geometry."""
        current = self.policy_state.capture_geometry
        cached = self.policy_state.cache_geometry
        return current is not None and cached is not None and cached.matches(current)

    @staticmethod
    def _parse_policy_capture_geometry(payload) -> PolicyCaptureGeometry | None:
        """Parse capture geometry from dict-like payloads."""
        if payload is None:
            return None
        if isinstance(payload, dict):
            getter = payload.get
        else:

            def getter(key: str):
                return getattr(payload, key, None)

        try:
            region_w = int(getter("region_w"))
            region_h = int(getter("region_h"))
            x_offset = int(getter("x_offset"))
            y_offset = int(getter("y_offset"))
        except (AttributeError, TypeError, ValueError):
            return None
        if region_w <= 0 or region_h <= 0 or x_offset < 0 or y_offset < 0:
            return None
        return PolicyCaptureGeometry(
            region_w=region_w,
            region_h=region_h,
            x_offset=x_offset,
            y_offset=y_offset,
        )

    @staticmethod
    def _format_policy_capture_geometry(geometry: PolicyCaptureGeometry | None) -> str:
        """Return a compact geometry string for logs."""
        if geometry is None:
            return "<none>"
        return f"{geometry.region_w}x{geometry.region_h}+({geometry.x_offset},{geometry.y_offset})"

    def has_full_policy_tab_cache(self) -> bool:
        """Whether task-local policy cache already has 5 confirmed tabs."""
        return (
            set(self.policy_state.tab_positions) == set(_PROMPT_POLICY_TAB_ORDER)
            and all(self.is_policy_tab_confirmed(tab_name) for tab_name in _PROMPT_POLICY_TAB_ORDER)
            and not self.policy_state.provisional_tabs
        )

    def has_policy_provisional_tab(self, tab_name: str) -> bool:
        """Whether this tab still needs a successful post-click verification."""
        return tab_name in self.policy_state.provisional_tabs

    def mark_policy_tab_provisional(self, tab_name: str) -> None:
        """Mark a task-local tab as provisional until a later successful verification."""
        if tab_name not in self.policy_state.tab_positions:
            return
        self.policy_state.provisional_tabs.add(tab_name)
        self.policy_state.tab_positions[tab_name].confirmed = False

    def mark_policy_tab_confirmed(self, tab_name: str) -> None:
        """Mark a cached tab as confirmed."""
        tab = self.policy_state.tab_positions.get(tab_name)
        if tab is not None:
            tab.confirmed = True
            self.policy_state.provisional_tabs.discard(tab_name)

    def start_policy_calibration(self, tabs: list[str] | None = None) -> None:
        """Start a deterministic calibration pass for provisional policy tabs."""
        calibration_tabs = tabs or list(_PROMPT_POLICY_TAB_ORDER)
        self.policy_state.calibration_pending_tabs = [
            tab for tab in calibration_tabs if tab in self.policy_state.tab_positions
        ]

    def has_policy_calibration_pending(self) -> bool:
        """Whether policy bootstrap still needs tab calibration clicks."""
        return bool(self.policy_state.calibration_pending_tabs)

    def get_policy_calibration_target_name(self) -> str:
        """Return the next tab that must be calibrated."""
        if not self.policy_state.calibration_pending_tabs:
            return ""
        return self.policy_state.calibration_pending_tabs[0]

    def complete_policy_calibration_target(self, tab_name: str) -> None:
        """Mark one calibration target as confirmed and remove it from the pending list."""
        self.mark_policy_tab_confirmed(tab_name)
        self.policy_state.calibration_pending_tabs = [
            tab for tab in self.policy_state.calibration_pending_tabs if tab != tab_name
        ]

    def get_policy_click_target_name(self) -> str:
        """Return the tab targeted by the current click stage."""
        if self.current_stage == "calibrate_tabs":
            return self.get_policy_calibration_target_name()
        if self.current_stage == "click_next_tab":
            return self.get_policy_current_tab_name()
        if self.current_stage == "click_cached_tab":
            return self.get_policy_current_tab_name()
        return ""

    def should_verify_policy_tab_click(self) -> bool:
        """Whether the current policy tab click still needs explicit verification."""
        if self.current_stage not in {"calibrate_tabs", "click_cached_tab", "click_next_tab"}:
            return False
        tab_name = self.get_policy_click_target_name()
        if not tab_name:
            return False
        if self.current_stage == "calibrate_tabs":
            return True
        return not self.is_policy_tab_confirmed(tab_name)

    def record_policy_failed_tab(self, tab_name: str) -> None:
        """Track distinct tab failures inside one policy primitive run."""
        if tab_name:
            self.policy_state.distinct_failed_tabs.add(tab_name)

    def clear_policy_failed_tabs(self) -> None:
        """Clear accumulated distinct tab failures for the current primitive run."""
        self.policy_state.distinct_failed_tabs.clear()

    def has_policy_bootstrap(self) -> bool:
        """Whether policy tab bootstrap is available."""
        return self.policy_state.enabled and self.policy_state.bootstrap_complete

    def increment_policy_bootstrap_failure(self) -> int:
        """Increment and return bootstrap parse failure count."""
        self.policy_state.bootstrap_failures += 1
        return self.policy_state.bootstrap_failures

    def reset_policy_bootstrap_failure(self) -> None:
        """Reset bootstrap parse failure count."""
        self.policy_state.bootstrap_failures = 0

    def is_policy_entry_done(self) -> bool:
        """Whether the policy entry branch has reached the card-management screen."""
        return self.policy_state.enabled and self.policy_state.entry_done

    def mark_policy_entry_done(self) -> None:
        """Record that the policy entry branch completed."""
        self.policy_state.enabled = True
        self.policy_state.entry_done = True

    def set_policy_selected_tab(self, tab_name: str) -> None:
        """Persist the currently selected policy tab."""
        self.policy_state.selected_tab_name = tab_name

    def get_policy_selected_tab(self) -> str:
        """Return the currently selected policy tab if known."""
        return self.policy_state.selected_tab_name

    def get_policy_current_tab_name(self) -> str:
        """Return the current tab from the queue."""
        idx = self.policy_state.current_tab_index
        if 0 <= idx < len(self.policy_state.eligible_tabs_queue):
            return self.policy_state.eligible_tabs_queue[idx]
        return ""

    def get_policy_next_tab_name(self) -> str:
        """Return the next queued tab after the current one, if any."""
        idx = self.policy_state.current_tab_index + 1
        if 0 <= idx < len(self.policy_state.eligible_tabs_queue):
            return self.policy_state.eligible_tabs_queue[idx]
        return ""

    def get_policy_remaining_queue(self) -> list[str]:
        """Return tabs still pending after the current tab."""
        idx = self.policy_state.current_tab_index
        if idx < 0:
            idx = 0
        start = idx + 1 if self.get_policy_current_tab_name() else idx
        return list(self.policy_state.eligible_tabs_queue[start:])

    def get_policy_current_tab_position(self) -> PolicyTabPosition | None:
        """Return the cached position for the current policy tab."""
        current_tab = self.get_policy_current_tab_name()
        if not current_tab:
            return None
        return self.policy_state.tab_positions.get(current_tab)

    def get_policy_next_tab_position(self) -> PolicyTabPosition | None:
        """Return the cached position for the next queued policy tab."""
        next_tab = self.get_policy_next_tab_name()
        if not next_tab:
            return None
        return self.policy_state.tab_positions.get(next_tab)

    def update_policy_tab_position(
        self,
        tab_name: str,
        screen_x: int,
        screen_y: int,
        *,
        confirmed: bool = False,
    ) -> None:
        """Overwrite one cached tab position."""
        if not self._is_absolute_coord(screen_x, screen_y):
            return
        existing = self.policy_state.tab_positions.get(tab_name)
        next_confirmed = confirmed if existing is None else (existing.confirmed or confirmed)
        self.policy_state.tab_positions[tab_name] = PolicyTabPosition(
            tab_name=tab_name,
            screen_x=screen_x,
            screen_y=screen_y,
            confirmed=next_confirmed and tab_name not in self.policy_state.provisional_tabs,
        )
        if next_confirmed:
            self.policy_state.provisional_tabs.discard(tab_name)

    def is_policy_tab_confirmed(self, tab_name: str) -> bool:
        """Whether this tab location has already been validated by a successful click."""
        tab = self.policy_state.tab_positions.get(tab_name)
        return bool(tab and tab.confirmed)

    def increment_policy_tab_failure(self, tab_name: str) -> int:
        """Increment and return the click-failure count for one tab."""
        next_count = self.policy_state.tab_failure_counts.get(tab_name, 0) + 1
        self.policy_state.tab_failure_counts[tab_name] = next_count
        return next_count

    def reset_policy_tab_failure(self, tab_name: str) -> None:
        """Reset a tab's failure count."""
        self.policy_state.tab_failure_counts[tab_name] = 0

    def has_policy_generic_fallback_used(self, tab_name: str) -> bool:
        """Whether generic fallback has already been used for this tab."""
        return tab_name in self.policy_state.generic_fallback_used

    def mark_policy_generic_fallback_used(self, tab_name: str) -> None:
        """Record one generic fallback use for this tab."""
        self.policy_state.generic_fallback_used.add(tab_name)

    def set_policy_mode(self, mode: str) -> None:
        """Update the current policy execution mode."""
        self.policy_state.mode = mode

    def set_policy_cache_source(self, source: str) -> None:
        """Persist where the current task-local policy tab cache came from."""
        self.policy_state.cache_source = source.strip()

    def set_policy_bootstrap_summary(self, summary: str) -> None:
        """Persist a short bootstrap summary for debugging."""
        self.policy_state.bootstrap_summary = summary.strip()

    def set_policy_event(self, event: str) -> None:
        """Persist a short policy runtime event message for debugging."""
        self.policy_state.last_event = event.strip()

    def set_policy_last_similarity_result(self, result: str) -> None:
        """Persist the latest post-click similarity outcome for policy debugging."""
        self.policy_state.last_similarity_result = result.strip()

    def set_policy_last_tab_check_result(self, result: str) -> None:
        """Persist the latest explicit policy tab-switch verification result."""
        self.policy_state.last_tab_check_result = result.strip()

    def set_policy_last_relocalize_result(self, result: str) -> None:
        """Persist the latest relocalize outcome for policy debugging."""
        self.policy_state.last_relocalize_result = result.strip()

    def set_policy_bundle_action_count(self, count: int) -> None:
        """Persist how many actions are in the current tab bundle."""
        self.policy_state.last_bundle_action_count = max(0, count)

    @staticmethod
    def policy_slot_accepts_source_tab(source_tab: str, target_slot: PolicySlotState) -> bool:
        """Whether a card from one policy tab may legally target the slot."""
        normalized_tab = source_tab.strip()
        if not normalized_tab:
            return False
        if target_slot.is_wild or target_slot.slot_type == "와일드카드":
            return True
        if normalized_tab == "와일드카드":
            return target_slot.slot_type == "와일드카드"
        return target_slot.slot_type == normalized_tab

    def mark_policy_slot_selected(
        self,
        *,
        card_name: str,
        source_tab: str,
        target_slot_id: str,
        reasoning: str = "",
    ) -> None:
        """Update slot state after a successful policy drag."""
        slot = self.policy_state.slot_inventory.get(target_slot_id)
        if slot is not None:
            slot.current_card_name = card_name
            slot.is_empty = False
            slot.selected_from_tab = source_tab
            slot.selection_reason = reasoning

    def mark_policy_tab_completed(self, tab_name: str) -> None:
        """Mark a tab as fully processed."""
        if tab_name not in self.policy_state.completed_tabs:
            self.policy_state.completed_tabs.append(tab_name)
        self.policy_state.last_popped_tab = tab_name

    def advance_policy_tab(self) -> None:
        """Advance to the next queued policy tab."""
        self.policy_state.current_tab_index += 1

    def is_policy_complete(self) -> bool:
        """Whether all queued policy tabs have been processed."""
        return self.policy_state.current_tab_index >= len(self.policy_state.eligible_tabs_queue)

    def increment_stage_failure(self, stage_key: str) -> int:
        """Increment and return a generic per-stage failure counter."""
        next_count = self.stage_failure_counts.get(stage_key, 0) + 1
        self.stage_failure_counts[stage_key] = next_count
        return next_count

    def reset_stage_failure(self, stage_key: str) -> None:
        """Reset a generic per-stage failure counter."""
        self.stage_failure_counts[stage_key] = 0

    def has_stage_fallback_used(self, stage_key: str) -> bool:
        """Whether generic fallback has already been used for this stage key."""
        return stage_key in self.stage_fallback_used

    def mark_stage_fallback_used(self, stage_key: str) -> None:
        """Record that generic fallback was used for this stage key."""
        self.stage_fallback_used.add(stage_key)

    def clear_stage_fallback_used(self, stage_key: str) -> None:
        """Clear the generic-fallback-used marker for one stage key."""
        self.stage_fallback_used.discard(stage_key)

    def set_fallback_return_stage(self, stage: str, stage_key: str = "") -> None:
        """Record which stage should resume after generic fallback succeeds."""
        self.fallback_return_stage = stage
        self.fallback_return_key = stage_key

    def clear_fallback_return_stage(self) -> None:
        """Clear the stored fallback resume target."""
        self.fallback_return_stage = ""
        self.fallback_return_key = ""

    def consume_fallback_return_stage(self) -> tuple[str, str]:
        """Pop the stored generic-fallback return stage and its recovery key."""
        stage = self.fallback_return_stage
        stage_key = self.fallback_return_key
        self.fallback_return_stage = ""
        self.fallback_return_key = ""
        return stage, stage_key

    def clear_policy_bootstrap(
        self,
        *,
        preserve_entry_done: bool = True,
        preserve_progress: bool = False,
        preserve_tab_positions: bool = True,
    ) -> None:
        """Drop cached policy-screen bootstrap data while keeping the task alive."""
        entry_done = self.policy_state.entry_done if preserve_entry_done else False
        restart_index = self.policy_state.current_tab_index if preserve_progress else 0
        restart_completed = list(self.policy_state.completed_tabs) if preserve_progress else []
        restart_tabs = copy.deepcopy(self.policy_state.tab_positions) if preserve_tab_positions else {}
        restart_provisional = set(self.policy_state.provisional_tabs) if preserve_tab_positions else set()
        restart_cache_geometry = copy.deepcopy(self.policy_state.cache_geometry) if preserve_tab_positions else None
        self.policy_state.enabled = True
        self.policy_state.mode = "structured"
        self.policy_state.entry_done = entry_done
        self.policy_state.bootstrap_complete = False
        self.policy_state.bootstrap_failures = 0
        self.policy_state.bootstrap_summary = ""
        self.policy_state.last_event = ""
        self.policy_state.last_bundle_action_count = 0
        self.policy_state.overview_mode = False
        self.policy_state.wild_slot_active = False
        self.policy_state.selected_tab_name = ""
        self.policy_state.visible_tabs = []
        self.policy_state.tab_positions = restart_tabs
        self.policy_state.cache_geometry = restart_cache_geometry
        self.policy_state.provisional_tabs = restart_provisional
        self.policy_state.calibration_pending_tabs = []
        self.policy_state.eligible_tabs_queue = list(self.policy_state.eligible_tabs_queue) if preserve_progress else []
        self.policy_state.current_tab_index = restart_index
        self.policy_state.slot_inventory = {}
        self.policy_state.completed_tabs = restart_completed
        self.policy_state.last_popped_tab = restart_completed[-1] if restart_completed else ""
        self.policy_state.tab_failure_counts = {}
        self.policy_state.restart_current_tab_index = restart_index
        self.policy_state.restart_completed_tabs = restart_completed
        self.policy_state.cache_source = self.policy_state.cache_source if preserve_tab_positions else ""
        self.policy_state.last_similarity_result = ""
        self.policy_state.last_tab_check_result = ""
        self.policy_state.last_relocalize_result = ""
        self.policy_state.distinct_failed_tabs = set()

    def set_last_semantic_verify(self, status: str, reason: str = "") -> None:
        """Persist a short semantic verification summary for debugging and prompts."""
        self.last_semantic_verify = f"{status}: {reason}".strip(": ")

    def to_prompt_string(self) -> str:
        """Compact prompt summary for the live planner."""
        lines = []

        if self.current_stage:
            lines.append(f"현재 stage: {self.current_stage}")
        if self.branch:
            lines.append(f"현재 branch: {self.branch}")
        if self.governor_state.target_governor_label:
            note = self.governor_state.target_governor_note
            suffix = f" / {note}" if note else ""
            lines.append(f"[governor_target] {self.governor_state.target_governor_label}{suffix}")

        if self.choice_catalog.enabled:
            selectable_candidates = self._selectable_choice_candidates()
            candidate_count = len(selectable_candidates)
            best = self.get_best_choice()
            lines.append(f"[choice_catalog] 확인한 후보 {candidate_count}개 / 목록끝={self.choice_catalog.end_reached}")
            if self.choice_catalog.downward_scan_scrolls:
                lines.append(
                    f"scan_scrolls={self.choice_catalog.downward_scan_scrolls} / "
                    f"last_new={self.choice_catalog.last_new_candidate_count}"
                )
            if self.choice_catalog.scan_end_reason:
                lines.append(f"scan_end_reason={self.choice_catalog.scan_end_reason}")
            if self.last_observation_summary:
                lines.append(f"obs_summary={self.last_observation_summary[:180]}")
            if self.last_observation_anchor:
                lines.append(f"scroll_anchor={self.last_observation_anchor}")
            if self.last_planned_action:
                lines.append(f"planned_action={self.last_planned_action[:180]}")
            if self.last_executed_action:
                lines.append(f"executed_action={self.last_executed_action[:180]}")
            if self.task_hitl_directive:
                hitl_status = self.task_hitl_status or "pending"
                lines.append(f"[task_hitl] {hitl_status} / {self.task_hitl_directive[:160]}")
                if self.task_hitl_reason:
                    lines.append(f"task_hitl_reason={self.task_hitl_reason[:180]}")
            if best is not None:
                best_note = self.choice_catalog.best_option_reason or "전략상 최적"
                lines.append(f"최종 선택 후보: {best.label} ({best.position_hint}) - {best_note}")

            display_candidates = selectable_candidates
            for candidate in display_candidates:
                lines.append(self._format_choice_candidate_line(candidate, include_id=False))

        placement_target = self.get_city_placement_target()
        if placement_target is not None:
            x, y, button = placement_target
            lines.append(
                f"[city_placement] target=({x},{y}) button={button} "
                f"reclick_attempts={self.city_placement_state.reclick_attempts}"
            )
            if self.city_placement_state.target_origin:
                lines.append(f"target_origin={self.city_placement_state.target_origin}")
            if self.city_placement_state.target_tile_color:
                lines.append(f"target_tile_color={self.city_placement_state.target_tile_color}")
            if self.city_placement_state.target_reason:
                lines.append(f"target_reason={self.city_placement_state.target_reason}")

        if self.policy_state.enabled:
            current_tab = self.get_policy_current_tab_name() or "-"
            remaining_queue = self.get_policy_remaining_queue()
            lines.append(
                f"[policy] mode={self.policy_state.mode} tabs={len(self.policy_state.eligible_tabs_queue)} "
                f"current={current_tab} wild={self.policy_state.wild_slot_active}"
            )
            lines.append(f"overview_mode={self.policy_state.overview_mode}")
            lines.append(f"bootstrap_failures={self.policy_state.bootstrap_failures}")
            if self.policy_state.entry_done:
                lines.append("entry_done=true")
            if self.policy_state.bootstrap_summary:
                lines.append(f"bootstrap={self.policy_state.bootstrap_summary}")
            if self.policy_state.cache_source:
                lines.append(f"cache_source={self.policy_state.cache_source}")
            if self.policy_state.cache_geometry is not None:
                lines.append(f"cache_geometry={self._format_policy_capture_geometry(self.policy_state.cache_geometry)}")
            if self.policy_state.visible_tabs:
                lines.append(f"visible_tabs: {', '.join(self.policy_state.visible_tabs)}")
            if self.policy_state.selected_tab_name:
                lines.append(f"selected_tab={self.policy_state.selected_tab_name}")
            lines.append(f"current_tab={current_tab}")
            lines.append(f"remaining_queue: {', '.join(remaining_queue) if remaining_queue else '<empty>'}")
            if self.policy_state.completed_tabs:
                lines.append(f"completed_tabs: {', '.join(self.policy_state.completed_tabs)}")
            if self.policy_state.last_popped_tab:
                lines.append(f"last_popped={self.policy_state.last_popped_tab}")
            confirmed_tabs = [tab for tab in _PROMPT_POLICY_TAB_ORDER if self.is_policy_tab_confirmed(tab)]
            if confirmed_tabs:
                lines.append(f"confirmed_tabs: {', '.join(confirmed_tabs)}")
            if self.policy_state.provisional_tabs:
                provisional = [tab for tab in _PROMPT_POLICY_TAB_ORDER if tab in self.policy_state.provisional_tabs]
                lines.append(f"provisional_tabs: {', '.join(provisional)}")
            if self.policy_state.calibration_pending_tabs:
                lines.append(f"calibration_pending: {', '.join(self.policy_state.calibration_pending_tabs)}")
            if self.policy_state.tab_positions:
                cache_parts = []
                for tab_name in _PROMPT_POLICY_TAB_ORDER:
                    tab = self.policy_state.tab_positions.get(tab_name)
                    if tab is None:
                        continue
                    suffix = "*" if tab.confirmed else ("?" if tab_name in self.policy_state.provisional_tabs else "")
                    cache_parts.append(f"{tab_name}=({tab.screen_x},{tab.screen_y}){suffix}")
                if cache_parts:
                    lines.append(f"tab_cache: {', '.join(cache_parts)}")
            if self.policy_state.last_bundle_action_count:
                lines.append(f"bundle_actions={self.policy_state.last_bundle_action_count}")
            if self.policy_state.last_similarity_result:
                lines.append(f"similarity={self.policy_state.last_similarity_result}")
            if self.policy_state.last_tab_check_result:
                lines.append(f"tab_check={self.policy_state.last_tab_check_result}")
            if self.policy_state.last_relocalize_result:
                lines.append(f"relocalize={self.policy_state.last_relocalize_result}")
            if self.policy_state.distinct_failed_tabs:
                failed_tabs = [tab for tab in _PROMPT_POLICY_TAB_ORDER if tab in self.policy_state.distinct_failed_tabs]
                lines.append(f"fallback_state={','.join(failed_tabs)}")
            if self.policy_state.last_event:
                lines.append(f"event={self.policy_state.last_event}")

        if self.voting_state.enabled:
            lines.append(f"[voting] current={self.voting_state.current_agenda_label or '-'}")
            if self.voting_state.selected_resolution:
                lines.append(f"resolution={self.voting_state.selected_resolution}")
            if self.voting_state.selected_vote_direction:
                lines.append(f"vote_direction={self.voting_state.selected_vote_direction}")
            if self.voting_state.selected_target_label:
                lines.append(f"target={self.voting_state.selected_target_label}")
            if self.voting_state.completed_agenda_ids:
                lines.append(f"completed_agendas={', '.join(self.voting_state.completed_agenda_ids)}")

        if self.last_semantic_verify:
            lines.append(f"[semantic] {self.last_semantic_verify}")

        if self.action_log:
            lines.append("[최근 task action]")
            lines.append(self.recent_actions_prompt())

        return "\n".join(lines) if lines else "없음"

    def to_observer_prompt_string(self) -> str:
        """Compact prompt summary for observer-only passes without echoing candidate labels."""
        lines = []

        if self.current_stage:
            lines.append(f"현재 stage: {self.current_stage}")
        if self.branch:
            lines.append(f"현재 branch: {self.branch}")

        if self.choice_catalog.enabled:
            lines.append(f"[choice_scan] 목록끝={self.choice_catalog.end_reached}")
            if self.choice_catalog.downward_scan_scrolls:
                lines.append(
                    f"scan_scrolls={self.choice_catalog.downward_scan_scrolls} / "
                    f"last_new={self.choice_catalog.last_new_candidate_count}"
                )
            if self.choice_catalog.scan_end_reason:
                lines.append(f"scan_end_reason={self.choice_catalog.scan_end_reason}")
            if self.choice_catalog.scroll_anchor is not None:
                lines.append(f"scroll_anchor={self._format_scroll_anchor(self.choice_catalog.scroll_anchor)}")

        if self.last_semantic_verify:
            lines.append(f"[semantic] {self.last_semantic_verify}")

        return "\n".join(lines) if lines else "없음"

    @staticmethod
    def _is_normalized_coord(x: int, y: int, normalizing_range: int = 1000) -> bool:
        """Whether the coordinate pair fits the normalized live-action range."""
        return 0 <= x <= normalizing_range and 0 <= y <= normalizing_range

    @staticmethod
    def _is_absolute_coord(x: int, y: int) -> bool:
        """Whether the coordinate pair fits logical absolute screen coordinates."""
        return x >= 0 and y >= 0

    @staticmethod
    def _is_normalized_rect(
        left: int,
        top: int,
        right: int,
        bottom: int,
        normalizing_range: int = 1000,
    ) -> bool:
        """Whether a bounding box fits the normalized live-action range."""
        return 0 <= left <= right <= normalizing_range and 0 <= top <= bottom <= normalizing_range

    def _build_scroll_anchor(self, scroll_anchor: dict) -> ScrollAnchor | None:
        """Return a validated normalized scroll anchor from raw JSON."""
        x = int(scroll_anchor.get("x", 0))
        y = int(scroll_anchor.get("y", 0))
        left = int(scroll_anchor.get("left", 0))
        top = int(scroll_anchor.get("top", 0))
        right = int(scroll_anchor.get("right", self.normalizing_range))
        bottom = int(scroll_anchor.get("bottom", self.normalizing_range))

        if not self._is_normalized_coord(x, y, self.normalizing_range):
            return None
        if not self._is_normalized_rect(left, top, right, bottom, self.normalizing_range):
            return None
        if not (left <= x <= right and top <= y <= bottom):
            return None

        return ScrollAnchor(
            x=x,
            y=y,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )

    def _format_scroll_anchor(self, scroll_anchor: dict | ScrollAnchor | None) -> str:
        """Return a compact one-line debug summary for a scroll anchor."""
        anchor: ScrollAnchor | None = None
        if isinstance(scroll_anchor, dict):
            anchor = self._build_scroll_anchor(scroll_anchor)
        elif isinstance(scroll_anchor, ScrollAnchor):
            anchor = scroll_anchor
        if anchor is None:
            return ""
        return f"({anchor.x},{anchor.y}) [{anchor.left},{anchor.top}]→[{anchor.right},{anchor.bottom}]"

    def _selectable_choice_candidates(self) -> list[ChoiceCandidate]:
        """Return only candidates that can still be selected."""
        return [
            candidate
            for candidate in self.choice_catalog.candidates.values()
            if self._candidate_is_selectable(candidate)
        ]
