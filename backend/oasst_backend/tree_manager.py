import random
from datetime import datetime, timedelta
from enum import Enum
from http import HTTPStatus
from typing import Optional, Tuple
from uuid import UUID

import numpy as np
import pydantic
import sqlalchemy as sa
from loguru import logger
from oasst_backend.api.v1.utils import prepare_conversation, prepare_conversation_message_list
from oasst_backend.config import TreeManagerConfiguration, settings
from oasst_backend.models import (
    Message,
    MessageEmoji,
    MessageReaction,
    MessageTreeState,
    Task,
    TextLabels,
    User,
    UserStats,
    UserStatsTimeFrame,
    message_tree_state,
)
from oasst_backend.prompt_repository import PromptRepository
from oasst_backend.scheduled_tasks import hf_feature_extraction, toxicity
from oasst_backend.utils.database_utils import (
    CommitMode,
    async_managed_tx_method,
    managed_tx_function,
    managed_tx_method,
)
from oasst_backend.utils.ranking import ranked_pairs
from oasst_shared.exceptions.oasst_api_error import OasstError, OasstErrorCode
from oasst_shared.schemas import protocol as protocol_schema
from oasst_shared.utils import utcnow
from sqlalchemy.sql.functions import coalesce
from sqlmodel import Session, and_, func, not_, or_, text, update


class TaskType(Enum):
    NONE = -1
    RANKING = 0
    LABEL_REPLY = 1
    REPLY = 2
    LABEL_PROMPT = 3
    PROMPT = 4


class TaskRole(Enum):
    ANY = 0
    PROMPTER = 1
    ASSISTANT = 2


class TreeStateStats(pydantic.BaseModel):
    initial_prompt_review: int
    growing: int
    ranking: int
    ready_for_scoring: int
    scoring_failed: int
    ready_for_export: int
    aborted_low_grade: int
    halted_by_moderator: int
    backlog_ranking: int
    prompt_lottery_waiting: int


class ActiveTreeSizeRow(pydantic.BaseModel):
    message_tree_id: UUID
    goal_tree_size: int
    tree_size: int
    awaiting_review: Optional[int]

    @property
    def remaining_messages(self) -> int:
        return max(0, self.goal_tree_size - self.tree_size)

    class Config:
        orm_mode = True


class ExtendibleParentRow(pydantic.BaseModel):
    parent_id: UUID
    parent_role: str
    depth: int
    message_tree_id: UUID
    active_children_count: int

    class Config:
        orm_mode = True


class IncompleteRankingsRow(pydantic.BaseModel):
    parent_id: UUID
    role: str
    children_count: int
    child_min_ranking_count: int
    message_tree_id: UUID

    class Config:
        orm_mode = True


class TreeMessageCountStats(pydantic.BaseModel):
    message_tree_id: UUID
    state: str
    depth: int
    oldest: datetime
    youngest: datetime
    count: int
    goal_tree_size: int

    @property
    def completed(self) -> int:
        return self.count / self.goal_tree_size


class TreeManagerStats(pydantic.BaseModel):
    state_counts: dict[str, int]
    message_counts: list[TreeMessageCountStats]


def halt_prompts_of_disabled_users(db: Session):
    _sql_halt_prompts_of_disabled_users = """
-- remove prompts of disabled & deleted users from prompt lottery
WITH cte AS (
SELECT mts.message_tree_id
FROM message_tree_state mts
JOIN message m ON mts.message_tree_id = m.id
JOIN "user" u ON m.user_id = u.id
WHERE state = :prompt_lottery_waiting_state AND (NOT u.enabled OR u.deleted)
)
UPDATE message_tree_state mts2
SET active=false, state=:halted_by_moderator_state
FROM cte
WHERE mts2.message_tree_id = cte.message_tree_id;
"""

    r = db.execute(
        text(_sql_halt_prompts_of_disabled_users),
        {
            "prompt_lottery_waiting_state": message_tree_state.State.PROMPT_LOTTERY_WAITING,
            "halted_by_moderator_state": message_tree_state.State.HALTED_BY_MODERATOR,
        },
    )
    if r.rowcount > 0:
        logger.info(f"Halted {r.rowcount} prompts of disabled users.")


class TreeManager:
    def __init__(
        self,
        db: Session,
        prompt_repository: PromptRepository,
        cfg: Optional[TreeManagerConfiguration] = None,
    ):
        self.db = db
        self.cfg = cfg or settings.tree_manager
        self.pr = prompt_repository

    def _random_task_selection(
        self,
        num_ranking_tasks: int,
        num_replies_need_review: int,
        num_prompts_need_review: int,
        num_missing_prompts: int,
        num_missing_replies: int,
    ) -> TaskType:
        """
        Determines which task to hand out to human worker.
        The task type is drawn with relative weight (e.g. ranking has highest priority)
        depending on what is possible with the current message trees in the database.
        """

        logger.debug(
            f"TreeManager._random_task_selection({num_ranking_tasks=}, {num_replies_need_review=}, "
            f"{num_prompts_need_review=}, {num_missing_prompts=}, {num_missing_replies=})"
        )

        task_type = TaskType.NONE
        task_weights = [0] * 5

        if num_ranking_tasks > 0:
            task_weights[TaskType.RANKING.value] = 10

        if num_replies_need_review > 0:
            task_weights[TaskType.LABEL_REPLY.value] = 5

        if num_prompts_need_review > 0:
            task_weights[TaskType.LABEL_PROMPT.value] = 5

        if num_missing_replies > 0:
            task_weights[TaskType.REPLY.value] = 2

        if num_missing_prompts > 0:
            task_weights[TaskType.PROMPT.value] = 0.01

        task_weights = np.array(task_weights)
        weight_sum = task_weights.sum()
        if weight_sum > 1e-8:
            task_weights = task_weights / weight_sum
            task_type = TaskType(np.random.choice(a=len(task_weights), p=task_weights))

        logger.debug(f"Selected {task_type=}")
        return task_type

    def _determine_task_availability_internal(
        self,
        num_missing_prompts: int,
        extendible_parents: list[ExtendibleParentRow],
        prompts_need_review: list[Message],
        replies_need_review: list[Message],
        incomplete_rankings: list[IncompleteRankingsRow],
    ) -> dict[protocol_schema.TaskRequestType, int]:
        task_count_by_type: dict[protocol_schema.TaskRequestType, int] = {t: 0 for t in protocol_schema.TaskRequestType}

        task_count_by_type[protocol_schema.TaskRequestType.initial_prompt] = max(0, num_missing_prompts)

        task_count_by_type[protocol_schema.TaskRequestType.prompter_reply] = len(
            list(filter(lambda x: x.parent_role == "assistant", extendible_parents))
        )
        task_count_by_type[protocol_schema.TaskRequestType.assistant_reply] = len(
            list(filter(lambda x: x.parent_role == "prompter", extendible_parents))
        )

        task_count_by_type[protocol_schema.TaskRequestType.label_initial_prompt] = len(prompts_need_review)
        task_count_by_type[protocol_schema.TaskRequestType.label_assistant_reply] = len(
            list(filter(lambda m: m.role == "assistant", replies_need_review))
        )
        task_count_by_type[protocol_schema.TaskRequestType.label_prompter_reply] = len(
            list(filter(lambda m: m.role == "prompter", replies_need_review))
        )

        if self.cfg.rank_prompter_replies:
            task_count_by_type[protocol_schema.TaskRequestType.rank_prompter_replies] = len(
                list(filter(lambda r: r.role == "prompter", incomplete_rankings))
            )

        task_count_by_type[protocol_schema.TaskRequestType.rank_assistant_replies] = len(
            list(filter(lambda r: r.role == "assistant", incomplete_rankings))
        )

        task_count_by_type[protocol_schema.TaskRequestType.random] = sum(
            task_count_by_type[t] for t in protocol_schema.TaskRequestType if t in task_count_by_type
        )

        return task_count_by_type

    def _prompt_lottery(self, lang: str, max_activate: int = 1) -> int:
        # Under high load the DB runs into deadlocks when many trees are released
        # simultaneously (happens whens the max_active_trees setting is increased).
        # To reduce the chance of write conflicts during updates of rows in the
        # message_tree_state table we limit the number of trees that are activated
        # per _prompt_lottery() call to max_activate.
        activated = 0

        while True:
            stats = self.tree_counts_by_state_stats(lang=lang, only_active=True)
            prompt_lottery_waiting = self.query_prompt_lottery_waiting(lang=lang)
            remaining_lottery_entries = max(0, self.cfg.max_prompt_lottery_waiting - prompt_lottery_waiting)
            remaining_prompt_review = max(0, self.cfg.max_initial_prompt_review - stats.initial_prompt_review)
            num_missing_growing = max(0, self.cfg.max_active_trees - stats.growing)
            logger.info(f"_prompt_lottery {remaining_prompt_review=}, {num_missing_growing=}")

            if num_missing_growing == 0 or activated >= max_activate:
                return min(num_missing_growing + remaining_prompt_review, remaining_lottery_entries)

            @managed_tx_function(CommitMode.COMMIT)
            def activate_one(db: Session) -> int:
                # select among distinct users
                authors_qry = (
                    db.query(Message.user_id, func.coalesce(UserStats.reply_ranked_1, 0).label("reply_ranked_1"))
                    .select_from(MessageTreeState)
                    .join(Message, MessageTreeState.message_tree_id == Message.id)
                    .join(User, Message.user_id == User.id)
                    .outerjoin(
                        UserStats, and_(UserStats.user_id == User.id, UserStats.time_frame == UserStatsTimeFrame.month)
                    )
                    .filter(
                        MessageTreeState.state == message_tree_state.State.PROMPT_LOTTERY_WAITING,
                        Message.lang == lang,
                        not_(Message.deleted),
                        Message.review_result,
                        User.enabled,
                        not_(User.deleted),
                    )
                    .distinct(Message.user_id)
                )

                author_data = authors_qry.all()
                if len(author_data) == 0:
                    logger.info(
                        f"No prompts for prompt lottery available ({num_missing_growing=}, trees missing for {lang=})."
                    )
                    return False

                author_ids = [data["user_id"] for data in author_data]
                # add one to avoid any scenario where all weights are 0
                # this also means inactive users can still occasionally be selected
                weights = [data["reply_ranked_1"] + 1 for data in author_data]

                # first select an author
                prompt_author_id: UUID = random.choices(author_ids, weights=weights)[0]
                logger.info(f"Selected random prompt author {prompt_author_id} among {len(author_data)} candidates.")

                # select random prompt of author
                qry = (
                    db.query(MessageTreeState, Message)
                    .select_from(MessageTreeState)
                    .join(Message, MessageTreeState.message_tree_id == Message.id)
                    .filter(
                        MessageTreeState.state == message_tree_state.State.PROMPT_LOTTERY_WAITING,
                        Message.user_id == prompt_author_id,
                        Message.lang == lang,
                        not_(Message.deleted),
                        Message.review_result,
                    )
                    .limit(100)
                )

                prompt_candidates = qry.all()
                if len(prompt_candidates) == 0:
                    logger.warning("No prompt candidates of selected author found.")
                    return False

                winner_prompt = random.choice(prompt_candidates)
                message: Message = winner_prompt.Message
                logger.info(f"Prompt lottery winner: {message.id=}")

                mts: MessageTreeState = winner_prompt.MessageTreeState
                mts.state = message_tree_state.State.GROWING
                mts.active = True
                db.add(mts)

                if mts.won_prompt_lottery_date is None:
                    mts.won_prompt_lottery_date = utcnow()
                logger.info(f"Tree entered '{mts.state}' state ({mts.message_tree_id=})")

                return True

            if not activate_one():
                return min(num_missing_growing + remaining_prompt_review, remaining_lottery_entries)

            activated += 1

    def _auto_moderation(self, lang: str) -> None:
        if not self.cfg.auto_mod_enabled:
            return

        bad_messages = self.query_moderation_bad_messages(lang=lang)
        for m in bad_messages:
            num_red_flag = m.emojis.get(protocol_schema.EmojiCode.red_flag)

            if num_red_flag is not None and num_red_flag >= self.cfg.auto_mod_red_flags:
                if m.parent_id is None:
                    logger.warning(
                        f"[AUTO MOD] Halting tree {m.message_tree_id}, initial prompt got too many red flags ({m.emojis})."
                    )
                    self.enter_low_grade_state(m.message_tree_id)
                else:
                    logger.warning(f"[AUTO MOD] Deleting message {m.id=}, it received too many red flags ({m.emojis}).")
                    self.pr.mark_messages_deleted(m.id, recursive=True)

            num_skip_reply = m.emojis.get(protocol_schema.EmojiCode.skip_reply)
            if num_skip_reply is not None and num_skip_reply >= self.cfg.auto_mod_max_skip_reply:
                logger.warning(
                    f"[AUTO MOD] Halting tree {m.message_tree_id} due to high skip-reply count of message {m.id=} ({m.emojis})."
                )
                self.halt_tree(m.id, halt=True)

    def determine_task_availability(self, lang: str) -> dict[protocol_schema.TaskRequestType, int]:
        self.pr.ensure_user_is_enabled()

        if not lang:
            lang = "en"
            logger.warning("Task availability request without lang tag received, assuming lang='en'.")

        if lang in self.cfg.init_prompt_disabled_langs_list:
            num_missing_prompts = 0
        else:
            num_missing_prompts = self._prompt_lottery(lang=lang, max_activate=1)

        self._auto_moderation(lang=lang)
        extendible_parents, _ = self.query_extendible_parents(lang=lang)
        prompts_need_review = self.query_prompts_need_review(lang=lang)
        replies_need_review = self.query_replies_need_review(lang=lang)
        incomplete_rankings = self.query_incomplete_rankings(lang=lang)

        return self._determine_task_availability_internal(
            num_missing_prompts=num_missing_prompts,
            extendible_parents=extendible_parents,
            prompts_need_review=prompts_need_review,
            replies_need_review=replies_need_review,
            incomplete_rankings=incomplete_rankings,
        )

    @staticmethod
    def _get_label_descriptions(valid_labels: list[TextLabels]) -> list[protocol_schema.LabelDescription]:
        return [
            protocol_schema.LabelDescription(
                name=l.value, widget=l.widget.value, display_text=l.display_text, help_text=l.help_text
            )
            for l in valid_labels
        ]

    def next_task(
        self,
        desired_task_type: protocol_schema.TaskRequestType = protocol_schema.TaskRequestType.random,
        lang: str = "en",
    ) -> Tuple[protocol_schema.Task, Optional[UUID], Optional[UUID]]:
        logger.debug(f"TreeManager.next_task({desired_task_type=}, {lang=})")

        self.pr.ensure_user_is_enabled()

        if not lang:
            lang = "en"
            logger.warning("Task request without lang tag received, assuming 'en'.")

        self._auto_moderation(lang=lang)
        num_missing_prompts = self._prompt_lottery(lang=lang, max_activate=2)

        # check user's pending tasks
        recent_tasks_span = timedelta(seconds=self.cfg.recent_tasks_span_sec)
        users_pending_tasks = self.pr.task_repository.fetch_pending_tasks_of_user(
            self.pr.user_id,
            max_age=recent_tasks_span,
            limit=self.cfg.max_pending_tasks_per_user + 1,
        )
        num_pending_tasks = len(users_pending_tasks)
        if num_pending_tasks >= self.cfg.max_pending_tasks_per_user:
            logger.warning(
                f"Rejecting task request. User {self.pr.user_id} has {num_pending_tasks} pending tasks. "
                f"Oldest age: {utcnow()-users_pending_tasks[0].created_date}."
            )
            raise OasstError(
                "User has too many pending tasks.",
                OasstErrorCode.TASK_TOO_MANY_PENDING,
            )
        elif num_pending_tasks > 0:
            logger.debug(
                f"User {self.pr.user_id} has {num_pending_tasks} pending tasks. Oldest age: {utcnow()-users_pending_tasks[0].created_date}"
            )

        prompts_need_review = self.query_prompts_need_review(lang=lang)
        replies_need_review = self.query_replies_need_review(lang=lang)
        extendible_parents, active_tree_sizes = self.query_extendible_parents(lang=lang)

        incomplete_rankings = self.query_incomplete_rankings(lang=lang)
        if not self.cfg.rank_prompter_replies:
            incomplete_rankings = list(filter(lambda r: r.role == "assistant", incomplete_rankings))

        # determine type of task to generate
        num_missing_replies = sum(x.remaining_messages for x in active_tree_sizes)

        task_role = TaskRole.ANY
        if desired_task_type == protocol_schema.TaskRequestType.random:
            task_type = self._random_task_selection(
                num_ranking_tasks=len(incomplete_rankings),
                num_replies_need_review=len(replies_need_review),
                num_prompts_need_review=len(prompts_need_review),
                num_missing_prompts=num_missing_prompts,
                num_missing_replies=num_missing_replies,
            )

            if task_type == TaskType.NONE:
                logger.warning(f"No random tasks currently available, user: {self.pr.user_id}")
                raise OasstError(
                    f"No tasks of type '{protocol_schema.TaskRequestType.random.value}' are currently available.",
                    OasstErrorCode.TASK_REQUESTED_TYPE_NOT_AVAILABLE,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
        else:
            task_count_by_type = self._determine_task_availability_internal(
                num_missing_prompts=num_missing_prompts,
                extendible_parents=extendible_parents,
                prompts_need_review=prompts_need_review,
                replies_need_review=replies_need_review,
                incomplete_rankings=incomplete_rankings,
            )

            available_count = task_count_by_type.get(desired_task_type)
            if not available_count:
                logger.warning(f"No '{desired_task_type.value}' tasks currently available, user: {self.pr.user_id}")
                raise OasstError(
                    f"No tasks of type '{desired_task_type.value}' are currently available.",
                    OasstErrorCode.TASK_REQUESTED_TYPE_NOT_AVAILABLE,
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )

            task_type_role_map = {
                protocol_schema.TaskRequestType.initial_prompt: (TaskType.PROMPT, TaskRole.ANY),
                protocol_schema.TaskRequestType.prompter_reply: (TaskType.REPLY, TaskRole.PROMPTER),
                protocol_schema.TaskRequestType.assistant_reply: (TaskType.REPLY, TaskRole.ASSISTANT),
                protocol_schema.TaskRequestType.rank_prompter_replies: (TaskType.RANKING, TaskRole.PROMPTER),
                protocol_schema.TaskRequestType.rank_assistant_replies: (TaskType.RANKING, TaskRole.ASSISTANT),
                protocol_schema.TaskRequestType.label_initial_prompt: (TaskType.LABEL_PROMPT, TaskRole.ANY),
                protocol_schema.TaskRequestType.label_assistant_reply: (TaskType.LABEL_REPLY, TaskRole.ASSISTANT),
                protocol_schema.TaskRequestType.label_prompter_reply: (TaskType.LABEL_REPLY, TaskRole.PROMPTER),
            }

            task_type, task_role = task_type_role_map[desired_task_type]

        message_tree_id = None
        parent_message_id = None

        logger.debug(f"selected {task_type=}")
        match task_type:
            case TaskType.RANKING:
                if task_role == TaskRole.PROMPTER:
                    incomplete_rankings = list(filter(lambda m: m.role == "prompter", incomplete_rankings))
                elif task_role == TaskRole.ASSISTANT:
                    incomplete_rankings = list(filter(lambda m: m.role == "assistant", incomplete_rankings))

                if len(incomplete_rankings) > 0:
                    ranking_parent_id = random.choice(incomplete_rankings).parent_id

                    messages = self.pr.fetch_message_conversation(ranking_parent_id)
                    assert len(messages) > 0 and messages[-1].id == ranking_parent_id
                    ranking_parent = messages[-1]
                    assert not ranking_parent.deleted and ranking_parent.review_result
                    conversation = prepare_conversation(messages)
                    replies = self.pr.fetch_message_children(ranking_parent_id, review_result=True, deleted=False)

                    assert len(replies) > 1
                    random.shuffle(replies)  # hand out replies in random order
                    reply_messages = prepare_conversation_message_list(replies)
                    if any(not m.synthetic for m in reply_messages):
                        reveal_synthetic = False
                        for rm in reply_messages:
                            rm.synthetic = None
                    else:
                        reveal_synthetic = True

                    replies = [p.text for p in replies]
                    if messages[-1].role == "assistant":
                        logger.info("Generating a RankPrompterRepliesTask.")
                        task = protocol_schema.RankPrompterRepliesTask(
                            conversation=conversation,
                            replies=replies,
                            reply_messages=reply_messages,
                            ranking_parent_id=ranking_parent.id,
                            message_tree_id=ranking_parent.message_tree_id,
                            reveal_synthetic=reveal_synthetic,
                        )
                    else:
                        logger.info("Generating a RankAssistantRepliesTask.")
                        task = protocol_schema.RankAssistantRepliesTask(
                            conversation=conversation,
                            replies=replies,
                            reply_messages=reply_messages,
                            ranking_parent_id=ranking_parent.id,
                            message_tree_id=ranking_parent.message_tree_id,
                            reveal_synthetic=reveal_synthetic,
                        )

                    parent_message_id = ranking_parent_id
                    message_tree_id = messages[-1].message_tree_id

            case TaskType.LABEL_REPLY:
                if task_role == TaskRole.PROMPTER:
                    replies_need_review = list(filter(lambda m: m.role == "prompter", replies_need_review))
                elif task_role == TaskRole.ASSISTANT:
                    replies_need_review = list(filter(lambda m: m.role == "assistant", replies_need_review))

                if len(replies_need_review) > 0:
                    random_reply_message = random.choice(replies_need_review)
                    messages = self.pr.fetch_message_conversation(random_reply_message)

                    conversation = prepare_conversation(messages)
                    message = messages[-1]

                    self.cfg.p_full_labeling_review_reply_prompter: float = 0.1

                    label_mode = protocol_schema.LabelTaskMode.full
                    label_disposition = protocol_schema.LabelTaskDisposition.quality

                    if message.role == "assistant":
                        valid_labels = self.cfg.labels_assistant_reply
                        if (
                            desired_task_type == protocol_schema.TaskRequestType.random
                            and random.random() > self.cfg.p_full_labeling_review_reply_assistant
                        ):
                            label_mode = protocol_schema.LabelTaskMode.simple
                            label_disposition = protocol_schema.LabelTaskDisposition.spam
                            valid_labels = self.cfg.mandatory_labels_assistant_reply.copy()
                            if protocol_schema.TextLabel.lang_mismatch not in valid_labels:
                                valid_labels.append(protocol_schema.TextLabel.lang_mismatch)
                            if protocol_schema.TextLabel.quality not in valid_labels:
                                valid_labels.append(protocol_schema.TextLabel.quality)

                        logger.info(f"Generating a LabelAssistantReplyTask. ({label_mode=:s})")
                        task = protocol_schema.LabelAssistantReplyTask(
                            message_id=message.id,
                            conversation=conversation,
                            reply=message.text,
                            valid_labels=list(map(lambda x: x.value, valid_labels)),
                            mandatory_labels=list(map(lambda x: x.value, self.cfg.mandatory_labels_assistant_reply)),
                            mode=label_mode,
                            disposition=label_disposition,
                            labels=self._get_label_descriptions(valid_labels),
                        )
                    else:
                        valid_labels = self.cfg.labels_prompter_reply
                        if (
                            desired_task_type == protocol_schema.TaskRequestType.random
                            and random.random() > self.cfg.p_full_labeling_review_reply_prompter
                        ):
                            label_mode = protocol_schema.LabelTaskMode.simple
                            label_disposition = protocol_schema.LabelTaskDisposition.spam
                            valid_labels = self.cfg.mandatory_labels_prompter_reply.copy()
                            if protocol_schema.TextLabel.lang_mismatch not in valid_labels:
                                valid_labels.append(protocol_schema.TextLabel.lang_mismatch)
                            if protocol_schema.TextLabel.quality not in valid_labels:
                                valid_labels.append(protocol_schema.TextLabel.quality)

                        logger.info(f"Generating a LabelPrompterReplyTask. ({label_mode=:s})")
                        task = protocol_schema.LabelPrompterReplyTask(
                            message_id=message.id,
                            conversation=conversation,
                            reply=message.text,
                            valid_labels=list(map(lambda x: x.value, valid_labels)),
                            mandatory_labels=list(map(lambda x: x.value, self.cfg.mandatory_labels_prompter_reply)),
                            mode=label_mode,
                            disposition=label_disposition,
                            labels=self._get_label_descriptions(valid_labels),
                        )

                    parent_message_id = message.id
                    message_tree_id = message.message_tree_id

            case TaskType.REPLY:
                if task_role == TaskRole.PROMPTER:
                    extendible_parents = list(filter(lambda x: x.parent_role == "assistant", extendible_parents))
                elif task_role == TaskRole.ASSISTANT:
                    extendible_parents = list(filter(lambda x: x.parent_role == "prompter", extendible_parents))

                # select a tree with missing replies
                if len(extendible_parents) > 0:
                    random_parent: ExtendibleParentRow = None
                    if self.cfg.p_lonely_child_extension > 0 and self.cfg.lonely_children_count > 1:
                        # check if we have extendible prompter parents with a small number of replies
                        lonely_children_parents = [
                            p
                            for p in extendible_parents
                            if 0 < p.active_children_count < self.cfg.lonely_children_count
                            and p.parent_role == "prompter"
                        ]
                        if len(lonely_children_parents) > 0 and random.random() < self.cfg.p_lonely_child_extension:
                            random_parent = random.choice(lonely_children_parents)

                    if random_parent is None:
                        random_parent = random.choice(extendible_parents)

                    # fetch random conversation to extend
                    logger.debug(f"selected {random_parent=}")
                    messages = self.pr.fetch_message_conversation(random_parent.parent_id)
                    assert all(m.review_result for m in messages)  # ensure all messages have positive reviews
                    conversation = prepare_conversation(messages)

                    # generate reply task depending on last message
                    if messages[-1].role == "assistant":
                        logger.info("Generating a PrompterReplyTask.")
                        task = protocol_schema.PrompterReplyTask(conversation=conversation)
                    else:
                        logger.info("Generating a AssistantReplyTask.")
                        task = protocol_schema.AssistantReplyTask(conversation=conversation)

                    parent_message_id = messages[-1].id
                    message_tree_id = messages[-1].message_tree_id

            case TaskType.LABEL_PROMPT:
                assert len(prompts_need_review) > 0
                message = random.choice(prompts_need_review)
                message = self.pr.fetch_message(message.id)  # re-fetch message including emojis

                label_mode = protocol_schema.LabelTaskMode.full
                label_disposition = protocol_schema.LabelTaskDisposition.quality
                valid_labels = self.cfg.labels_initial_prompt

                if random.random() > self.cfg.p_full_labeling_review_prompt:
                    valid_labels = self.cfg.mandatory_labels_initial_prompt.copy()
                    label_mode = protocol_schema.LabelTaskMode.simple
                    label_disposition = protocol_schema.LabelTaskDisposition.spam
                    if protocol_schema.TextLabel.lang_mismatch not in valid_labels:
                        valid_labels.append(protocol_schema.TextLabel.lang_mismatch)

                logger.info(f"Generating a LabelInitialPromptTask ({label_mode=:s}).")
                task = protocol_schema.LabelInitialPromptTask(
                    message_id=message.id,
                    prompt=message.text,
                    conversation=prepare_conversation([message]),
                    valid_labels=list(map(lambda x: x.value, valid_labels)),
                    mandatory_labels=list(map(lambda x: x.value, self.cfg.mandatory_labels_initial_prompt)),
                    mode=label_mode,
                    disposition=label_disposition,
                    labels=self._get_label_descriptions(valid_labels),
                )

                parent_message_id = message.id
                message_tree_id = message.message_tree_id

            case TaskType.PROMPT:
                logger.info("Generating an InitialPromptTask.")
                task = protocol_schema.InitialPromptTask(hint=None)

            case _:
                task = None

        if task is None:
            raise OasstError(
                f"No task of type '{desired_task_type.value}' is currently available.",
                OasstErrorCode.TASK_REQUESTED_TYPE_NOT_AVAILABLE,
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

        logger.info(f"Generated task (type={task.type}, id={task.id})")
        logger.debug(f"Generated {task=}.")

        return task, message_tree_id, parent_message_id

    @async_managed_tx_method(CommitMode.FLUSH)
    async def handle_interaction(self, interaction: protocol_schema.AnyInteraction) -> protocol_schema.Task:
        pr = self.pr
        pr.ensure_user_is_enabled()
        match type(interaction):
            case protocol_schema.TextReplyToMessage:
                logger.info(
                    f"Frontend reports text reply to message_id={interaction.message_id} by user={interaction.user}."
                )
                logger.debug(f"with {interaction.text=}")
                # here we store the text reply in the database
                message = pr.store_text_reply(
                    text=interaction.text,
                    lang=interaction.lang,
                    frontend_message_id=interaction.message_id,
                    user_frontend_message_id=interaction.user_message_id,
                )

                if not message.parent_id:
                    logger.info(
                        f"TreeManager: Inserting new tree state for initial prompt {message.id=} [{message.lang}]"
                    )
                    self._insert_default_state(message.id, lang=message.lang)

                if not settings.DEBUG_SKIP_EMBEDDING_COMPUTATION:
                    try:
                        hf_feature_extraction.delay(interaction.text, message.id, pr.api_client.dict())
                        logger.debug("Extract Embedding")
                    except OasstError:
                        logger.error(
                            f"Could not fetch embbeddings for text reply to {interaction.message_id=} with {interaction.text=} by {interaction.user=}."
                        )
                if not settings.DEBUG_SKIP_TOXICITY_CALCULATION:
                    try:
                        toxicity.delay(interaction.text, message.id, pr.api_client.dict())
                        logger.debug("Sent Toxicity")
                    except OasstError:
                        logger.error(
                            f"Could not compute toxicity for text reply to {interaction.message_id=} with {interaction.text=} by {interaction.user=}."
                        )

            case protocol_schema.MessageRating:
                logger.info(
                    f"Frontend reports rating of message_id={interaction.message_id} by user={interaction.user}."
                )
                logger.debug(f"with {interaction.rating=}")

                pr.store_rating(interaction)

            case protocol_schema.MessageRanking:
                logger.info(
                    f"Frontend reports ranking of message_id={interaction.message_id} by user={interaction.user}."
                )
                logger.debug(f"with {interaction.ranking=}")

                _, task = pr.store_ranking(interaction)
                self.check_condition_for_scoring_state(task.message_tree_id)

            case protocol_schema.TextLabels:
                logger.info(
                    f"Frontend reports labels of message_id={interaction.message_id} by user={interaction.user}."
                )
                logger.debug(f"with {interaction.labels=}")

                _, task, msg = pr.store_text_labels(interaction)

                # if it was a response for a task, check if we have enough reviews to calc review_result
                if task and msg:
                    reviews = self.query_reviews_for_message(msg.id)
                    acceptance_score = self._calculate_acceptance(reviews)
                    logger.debug(
                        f"Message {msg.id=}, {acceptance_score=}, {len(reviews)=}, {msg.review_result=}, {msg.review_count=}"
                    )
                    if msg.parent_id is None:
                        if not msg.review_result and msg.review_count >= self.cfg.num_reviews_initial_prompt:
                            if acceptance_score > self.cfg.acceptance_threshold_initial_prompt:
                                msg.review_result = True
                                self.db.add(msg)
                                logger.info(
                                    f"Initial prompt message was accepted: {msg.id=}, {acceptance_score=}, {len(reviews)=}"
                                )
                            else:
                                if msg.review_result is None:
                                    msg.review_result = False
                                    self.db.add(msg)
                                self.enter_low_grade_state(msg.message_tree_id)
                        self.check_condition_for_prompt_lottery(msg.message_tree_id)
                    elif msg.review_count >= self.cfg.num_reviews_reply:
                        if not msg.review_result and acceptance_score > self.cfg.acceptance_threshold_reply:
                            msg.review_result = True
                            self.db.add(msg)
                            logger.info(
                                f"Reply message message accepted: {msg.id=}, {acceptance_score=}, {len(reviews)=}"
                            )
                        elif msg.review_result is None:  # do not overwrite existing review result
                            msg.review_result = False
                            self.db.add(msg)

                    self.check_condition_for_ranking_state(msg.message_tree_id)

            case _:
                raise OasstError("Invalid response type.", OasstErrorCode.TASK_INVALID_RESPONSE_TYPE)

        return protocol_schema.TaskDone()

    def _enter_state(self, mts: MessageTreeState, state: message_tree_state.State):
        assert mts

        is_terminal = state in message_tree_state.TERMINAL_STATES
        was_active = mts.active
        mts.active = not is_terminal
        mts.state = state.value
        self.db.add(mts)
        self.db.flush

        if is_terminal:
            logger.info(f"Tree entered terminal '{mts.state}' state ({mts.message_tree_id=})")
            root_msg = self.pr.fetch_message(message_id=mts.message_tree_id, fail_if_missing=False)
            if root_msg and was_active:
                if random.random() < self.cfg.p_activate_backlog_tree:
                    self.activate_backlog_tree(lang=root_msg.lang)

                if self.cfg.min_active_rankings_per_lang > 0:
                    incomplete_rankings = self.query_incomplete_rankings(lang=root_msg.lang, user_filter=False)
                    if len(incomplete_rankings) < self.cfg.min_active_rankings_per_lang:
                        self.activate_backlog_tree(lang=root_msg.lang)
        else:
            if mts.state == message_tree_state.State.GROWING and mts.won_prompt_lottery_date is None:
                mts.won_prompt_lottery_date = utcnow()
            logger.info(f"Tree entered '{mts.state}' state ({mts.message_tree_id=})")

    def enter_low_grade_state(self, message_tree_id: UUID) -> None:
        logger.debug(f"enter_low_grade_state({message_tree_id=})")
        mts = self.pr.fetch_tree_state(message_tree_id)
        self._enter_state(mts, message_tree_state.State.ABORTED_LOW_GRADE)

    def check_condition_for_prompt_lottery(self, message_tree_id: UUID) -> bool:
        logger.debug(f"check_condition_for_prompt_lottery({message_tree_id=})")

        mts = self.pr.fetch_tree_state(message_tree_id)
        if not mts.active or mts.state != message_tree_state.State.INITIAL_PROMPT_REVIEW:
            logger.debug(f"False {mts.active=}, {mts.state=}")
            return False

        # check if initial prompt was accepted
        initial_prompt = self.pr.fetch_message(message_tree_id)
        if not initial_prompt.review_result:
            logger.debug(f"False {initial_prompt.review_result=}")
            return False

        self._enter_state(mts, message_tree_state.State.PROMPT_LOTTERY_WAITING)
        return True

    def check_condition_for_ranking_state(self, message_tree_id: UUID) -> bool:
        logger.debug(f"check_condition_for_ranking_state({message_tree_id=})")

        mts = self.pr.fetch_tree_state(message_tree_id)
        if not mts.active or mts.state != message_tree_state.State.GROWING:
            logger.debug(f"False {mts.active=}, {mts.state=}")
            return False

        # check if desired tree size has been reached and all nodes have been reviewed
        tree_size = self.query_tree_size(message_tree_id)
        if tree_size.tree_size == 0:
            logger.warning(
                f"All messages of message tree {message_tree_id} were deleted (tree_size == 0), halting tree."
            )
            self._enter_state(mts, message_tree_state.State.HALTED_BY_MODERATOR)
            return False

        if tree_size.remaining_messages > 0 or tree_size.awaiting_review > 0:
            logger.debug(f"False {tree_size.remaining_messages=}, {tree_size.awaiting_review=}")
            return False

        self._enter_state(mts, message_tree_state.State.RANKING)
        return True

    def check_condition_for_scoring_state(self, message_tree_id: UUID) -> bool:
        logger.debug(f"check_condition_for_scoring_state({message_tree_id=})")

        mts = self.pr.fetch_tree_state(message_tree_id)
        if mts.state != message_tree_state.State.SCORING_FAILED:
            if not mts.active or mts.state not in (
                message_tree_state.State.RANKING,
                message_tree_state.State.READY_FOR_SCORING,
            ):
                logger.debug(f"False {mts.active=}, {mts.state=}")
                return False

        ranking_role_filter = None if self.cfg.rank_prompter_replies else "assistant"
        rankings_by_message = self.query_tree_ranking_results(message_tree_id, role_filter=ranking_role_filter)
        for parent_msg_id, ranking in rankings_by_message.items():
            if len(ranking) < self.cfg.num_required_rankings:
                logger.debug(f"False {parent_msg_id=} {len(ranking)=}")
                return False

        if (
            mts.state != message_tree_state.State.SCORING_FAILED
            and mts.state != message_tree_state.State.READY_FOR_SCORING
        ):
            self._enter_state(mts, message_tree_state.State.READY_FOR_SCORING)
        self.update_message_ranks(message_tree_id, rankings_by_message)
        return True

    def ranked_pairs_update(self, rankings: list[MessageReaction]) -> int:
        assert len(rankings) > 0

        num_updated = 0
        ordered_ids_list: list[list[UUID]] = [
            msg_reaction.payload.payload.ranked_message_ids for msg_reaction in rankings
        ]

        common_set: set[UUID] = set.intersection(*map(set, ordered_ids_list))
        if len(common_set) < 2:
            logger.warning("The intersection of ranking results ID sets has less than two elements. Skipping.")
            return

        # keep only elements in common set
        ordered_ids_list = [list(filter(lambda x: x in common_set, ids)) for ids in ordered_ids_list]
        assert all(len(x) == len(common_set) for x in ordered_ids_list)

        logger.debug(f"SORTED MESSAGE IDS {ordered_ids_list}")
        consensus = ranked_pairs(ordered_ids_list)
        assert len(consensus) == len(common_set)
        logger.debug(f"CONSENSUS: {consensus}\n\n")

        # fetch all siblings and index by id
        siblings = self.pr.fetch_message_siblings(consensus[0], review_result=None, deleted=None)
        siblings = {m.id: m for m in siblings}

        # set rank for each message that was part of the common set
        for rank, message_id in enumerate(consensus):
            message = siblings.get(message_id)
            if message:
                if message.rank != rank:
                    message.rank = rank
                    self.db.add(message)
                    num_updated += 1
            else:
                logger.warning(f"Message {message_id=} not found among siblings.")

        # clear rank of sibling messages not in consensus
        for message in siblings.values():
            if message.id not in consensus and message.rank is not None:
                message.rank = None
                self.db.add(message)
                num_updated += 1

        return num_updated

    def update_message_ranks(
        self, message_tree_id: UUID, rankings_by_message: dict[UUID, list[MessageReaction]]
    ) -> bool:
        mts = self.pr.fetch_tree_state(message_tree_id)
        # check state, allow retry if in SCORING_FAILED state
        if mts.state not in (message_tree_state.State.READY_FOR_SCORING, message_tree_state.State.SCORING_FAILED):
            logger.debug(f"False {mts.active=}, {mts.state=}")
            return False

        if mts.state == message_tree_state.State.SCORING_FAILED:
            mts.active = True
            mts.state = message_tree_state.State.READY_FOR_SCORING

        try:
            for rankings in rankings_by_message.values():
                if len(rankings) > 0:
                    self.ranked_pairs_update(rankings)

        except Exception:
            logger.exception(f"update_message_ranks({message_tree_id=}) failed")
            self._enter_state(mts, message_tree_state.State.SCORING_FAILED)
            return False

        self._enter_state(mts, message_tree_state.State.READY_FOR_EXPORT)

        return True

    def activate_backlog_tree(self, lang: str) -> MessageTreeState:
        while True:
            # find tree in backlog state
            backlog_tree: MessageTreeState = (
                self.db.query(MessageTreeState)
                .join(Message, MessageTreeState.message_tree_id == Message.id)  # root msg
                .filter(MessageTreeState.state == message_tree_state.State.BACKLOG_RANKING)
                .filter(Message.lang == lang)
                .limit(1)
                .one_or_none()
            )

            if not backlog_tree:
                return None

            if len(self.query_tree_ranking_results(message_tree_id=backlog_tree.message_tree_id)) == 0:
                logger.info(
                    f"Backlog tree {backlog_tree.message_tree_id} has no children to rank, aborting with 'aborted_low_grade' state."
                )
                self._enter_state(backlog_tree, message_tree_state.State.ABORTED_LOW_GRADE)
            else:
                logger.info(f"Activating backlog tree {backlog_tree.message_tree_id}")
                backlog_tree.active = True
                self._enter_state(backlog_tree, message_tree_state.State.RANKING)
                return backlog_tree

    def _calculate_acceptance(self, labels: list[TextLabels]):
        # calculate acceptance based on lang_mismatch & spam label
        lang_mismatch = np.mean([(l.labels.get(protocol_schema.TextLabel.lang_mismatch) or 0) for l in labels])
        spam = np.mean([l.labels[protocol_schema.TextLabel.spam] for l in labels])
        acceptance_score = 1 - (spam + lang_mismatch)
        logger.debug(f"{acceptance_score=} ({spam=}, {lang_mismatch=})")
        return acceptance_score

    def _query_need_review(
        self, state: message_tree_state.State, required_reviews: int, root: bool, lang: str
    ) -> list[Message]:
        need_review = (
            self.db.query(Message)
            .select_from(MessageTreeState)
            .join(Message, MessageTreeState.message_tree_id == Message.message_tree_id)
            .outerjoin(
                MessageEmoji,
                and_(
                    Message.id == MessageEmoji.message_id,
                    MessageEmoji.user_id == self.pr.user_id,
                    MessageEmoji.emoji == protocol_schema.EmojiCode.skip_labeling,
                ),
            )
            .filter(
                MessageTreeState.active,
                MessageTreeState.state == state,
                or_(Message.review_result.is_(None), not_(Message.review_result)),
                not_(Message.deleted),
                Message.review_count < required_reviews,
                Message.lang == lang,
                MessageEmoji.message_id.is_(None),
            )
        )

        if root:
            need_review = need_review.filter(Message.parent_id.is_(None))
        else:
            need_review = need_review.filter(Message.parent_id.is_not(None))

        if not settings.DEBUG_ALLOW_SELF_LABELING:
            need_review = need_review.filter(Message.user_id != self.pr.user_id)

        if settings.DEBUG_ALLOW_DUPLICATE_TASKS:
            qry = need_review
        else:
            user_id = self.pr.user_id
            need_review = need_review.cte(name="need_review")
            qry = (
                self.db.query(Message)
                .select_entity_from(need_review)
                .outerjoin(TextLabels, need_review.c.id == TextLabels.message_id)
                .group_by(need_review)
                .having(
                    func.count(TextLabels.id).filter(TextLabels.task_id.is_not(None), TextLabels.user_id == user_id)
                    == 0
                )
            )

        return qry.all()

    def query_prompts_need_review(self, lang: str) -> list[Message]:
        """
        Select initial prompt messages with less than required rankings in active message tree
        (active == True in message_tree_state)
        """
        return self._query_need_review(
            message_tree_state.State.INITIAL_PROMPT_REVIEW, self.cfg.num_reviews_initial_prompt, True, lang
        )

    def query_replies_need_review(self, lang: str) -> list[Message]:
        """
        Select child messages (parent_id IS NOT NULL) with less than required rankings
        in active message tree (active == True in message_tree_state)
        """
        return self._query_need_review(message_tree_state.State.GROWING, self.cfg.num_reviews_reply, False, lang)

    _sql_find_incomplete_rankings = """
-- find incomplete rankings
SELECT m.parent_id, m.role, COUNT(m.id) children_count, MIN(m.ranking_count) child_min_ranking_count,
    COUNT(m.id) FILTER (WHERE m.ranking_count >= :num_required_rankings) as completed_rankings,
    mts.message_tree_id
FROM message_tree_state mts
    INNER JOIN message m ON mts.message_tree_id = m.message_tree_id
    INNER JOIN message p ON m.parent_id = p.id
    LEFT JOIN message_emoji me on
        (m.parent_id = me.message_id
        AND :skip_user_id IS NOT NULL
        AND me.user_id = :skip_user_id
        AND me.emoji = :skip_ranking)
WHERE mts.active                        -- only consider active trees
    AND mts.state = :ranking_state      -- message tree must be in ranking state
    AND m.review_result                 -- must be reviewed
    AND p.lang = :lang                  -- parent lang matches
    AND NOT m.deleted                   -- not deleted
    AND m.parent_id IS NOT NULL         -- ignore initial prompts
    AND me.message_id IS NULL           -- no skip ranking emoji for user
GROUP BY m.parent_id, m.role, mts.message_tree_id
HAVING COUNT(m.id) > 1                                      -- more than one child
    AND MIN(m.ranking_count) < :num_required_rankings       -- not complete
    AND COUNT(m.id) FILTER (WHERE m.user_id = :rank_user_id) = 0 -- no self-ranking
"""

    _sql_find_incomplete_rankings_ex = f"""
-- incomplete rankings but exclude of current user
WITH incomplete_rankings AS ({_sql_find_incomplete_rankings})
SELECT ir.* FROM incomplete_rankings ir
    LEFT JOIN message_reaction mr ON ir.parent_id = mr.message_id AND mr.payload_type = 'RankingReactionPayload'
GROUP BY ir.parent_id, ir.role, ir.children_count, ir.child_min_ranking_count, ir.completed_rankings,
    ir.message_tree_id
HAVING COUNT(mr.message_id) FILTER (WHERE mr.user_id = :dupe_user_id) = 0
"""

    def query_incomplete_rankings(self, lang: str, user_filter: bool = True) -> list[IncompleteRankingsRow]:
        """Query parents which have children that need further rankings"""

        dupe_user_id = None
        skip_user_id = None
        rank_user_id = None
        if user_filter:
            if not settings.DEBUG_ALLOW_DUPLICATE_TASKS:
                dupe_user_id = self.pr.user_id
            if not settings.DEBUG_ALLOW_SELF_RANKING:
                rank_user_id = self.pr.user_id
            skip_user_id = self.pr.user_id
        r = self.db.execute(
            text(self._sql_find_incomplete_rankings_ex),
            {
                "num_required_rankings": self.cfg.num_required_rankings,
                "lang": lang,
                "dupe_user_id": dupe_user_id,
                "skip_user_id": skip_user_id,
                "rank_user_id": rank_user_id,
                "ranking_state": message_tree_state.State.RANKING,
                "skip_ranking": protocol_schema.EmojiCode.skip_ranking,
            },
        )
        return [IncompleteRankingsRow.from_orm(x) for x in r.all()]

    _sql_find_extendible_parents = """
-- find all extendible parent nodes
WITH recent_reply_tasks (parent_message_id) AS (
    -- recent incomplete tasks to exclude
    SELECT parent_message_id FROM task
    WHERE not done
        AND not skipped
        AND created_date > (CURRENT_TIMESTAMP - :recent_tasks_interval)
        AND (payload_type = 'AssistantReplyPayload' OR payload_type = 'PrompterReplyPayload')
)
SELECT m.id as parent_id, m.role as parent_role, m.depth, m.message_tree_id, COUNT(c.id) active_children_count
FROM message_tree_state mts
    INNER JOIN message m ON mts.message_tree_id = m.message_tree_id     -- all elements of message tree
    LEFT JOIN message_emoji me ON
        (m.id = me.message_id
        AND :skip_user_id IS NOT NULL
        AND me.user_id = :skip_user_id
        AND me.emoji = :skip_reply)
    LEFT JOIN recent_reply_tasks rrt ON m.id = rrt.parent_message_id    -- recent tasks
    LEFT JOIN message c ON m.id = c.parent_id  -- child nodes
WHERE mts.active                        -- only consider active trees
    AND mts.state = :growing_state      -- message tree must be growing
    AND NOT m.deleted                   -- ignore deleted messages as parents
    AND m.depth < mts.max_depth         -- ignore leaf nodes as parents
    AND m.review_result                 -- parent node must have positive review
    AND m.lang = :lang                  -- parent matches lang
    AND me.message_id IS NULL           -- no skip reply emoji for user
    AND rrt.parent_message_id IS NULL   -- no recent reply task found
    AND NOT coalesce(c.deleted, FALSE)  -- don't count deleted children
    AND (c.review_result OR coalesce(c.review_count, 0) < :num_reviews_reply) -- don't count children with negative review but count elements under review
GROUP BY m.id, m.role, m.depth, m.message_tree_id, mts.max_children_count
HAVING COUNT(c.id) < mts.max_children_count -- below maximum number of children
    AND (COUNT(c.id) < :num_prompter_replies OR m.role = 'prompter')   -- limit replies to assistant messages
    AND COUNT(c.id) FILTER (WHERE c.user_id = :user_id) = 0  -- without reply by user
"""

    def query_extendible_parents(self, lang: str) -> tuple[list[ExtendibleParentRow], list[ActiveTreeSizeRow]]:
        """Query parent messages that have not reached the maximum number of replies."""

        user_id = self.pr.user_id if not settings.DEBUG_ALLOW_DUPLICATE_TASKS else None
        r = self.db.execute(
            text(self._sql_find_extendible_parents),
            {
                "growing_state": message_tree_state.State.GROWING,
                "num_reviews_reply": self.cfg.num_reviews_reply,
                "num_prompter_replies": self.cfg.num_prompter_replies,
                "lang": lang,
                "user_id": user_id,
                "skip_user_id": self.pr.user_id,
                "skip_reply": protocol_schema.EmojiCode.skip_reply,
                "recent_tasks_interval": timedelta(seconds=self.cfg.recent_tasks_span_sec),
            },
        )

        potential_parents = [ExtendibleParentRow.from_orm(x) for x in r.all()]
        extendible_trees = self.query_extendible_trees(lang=lang)
        extendible_tree_ids = set(t.message_tree_id for t in extendible_trees)
        extendible_parents = list(p for p in potential_parents if p.message_tree_id in extendible_tree_ids)

        return extendible_parents, extendible_trees

    _sql_find_extendible_trees = f"""
-- find extendible trees
SELECT m.message_tree_id, mts.goal_tree_size, COUNT(m.id) AS tree_size
FROM (
        SELECT DISTINCT message_tree_id FROM ({_sql_find_extendible_parents}) extendible_parents
    ) trees INNER JOIN message_tree_state mts ON trees.message_tree_id = mts.message_tree_id
    INNER JOIN message m ON mts.message_tree_id = m.message_tree_id
WHERE NOT m.deleted
    AND (
        m.parent_id IS NOT NULL AND (m.review_result OR m.review_count < :num_reviews_reply) -- children
        OR m.parent_id IS NULL AND m.review_result -- prompts (root nodes) must have positive review
    )
GROUP BY m.message_tree_id, mts.goal_tree_size
HAVING COUNT(m.id) < mts.goal_tree_size
"""

    def query_extendible_trees(self, lang: str) -> list[ActiveTreeSizeRow]:
        """Query size of active message trees in growing state."""

        user_id = self.pr.user_id if not settings.DEBUG_ALLOW_DUPLICATE_TASKS else None
        r = self.db.execute(
            text(self._sql_find_extendible_trees),
            {
                "growing_state": message_tree_state.State.GROWING,
                "num_reviews_reply": self.cfg.num_reviews_reply,
                "num_prompter_replies": self.cfg.num_prompter_replies,
                "lang": lang,
                "user_id": user_id,
                "skip_user_id": self.pr.user_id,
                "skip_reply": protocol_schema.EmojiCode.skip_reply,
                "recent_tasks_interval": timedelta(seconds=self.cfg.recent_tasks_span_sec),
            },
        )
        return [ActiveTreeSizeRow.from_orm(x) for x in r.all()]

    def query_tree_size(self, message_tree_id: UUID) -> ActiveTreeSizeRow:
        """Returns the number of reviewed not deleted messages in the message tree."""

        required_reviews = settings.tree_manager.num_reviews_reply
        qry = (
            self.db.query(
                MessageTreeState.message_tree_id.label("message_tree_id"),
                MessageTreeState.goal_tree_size.label("goal_tree_size"),
                func.count(Message.id).filter(Message.review_result).label("tree_size"),
                func.count(Message.id)
                .filter(
                    or_(Message.review_result.is_(None), not_(Message.review_result)),
                    Message.review_count < required_reviews,
                )
                .label("awaiting_review"),
            )
            .select_from(MessageTreeState)
            .outerjoin(
                Message, and_(MessageTreeState.message_tree_id == Message.message_tree_id, not_(Message.deleted))
            )
            .filter(
                MessageTreeState.active,
                MessageTreeState.message_tree_id == message_tree_id,
            )
            .group_by(MessageTreeState.message_tree_id, MessageTreeState.goal_tree_size)
        )

        return ActiveTreeSizeRow.from_orm(qry.one())

    def query_misssing_tree_states(self) -> list[Tuple[UUID, str]]:
        """Find all initial prompt messages that have no associated message tree state"""
        qry_missing_tree_states = (
            self.db.query(Message.id, Message.lang)
            .outerjoin(MessageTreeState, Message.message_tree_id == MessageTreeState.message_tree_id)
            .filter(
                Message.parent_id.is_(None),
                Message.message_tree_id == Message.id,
                MessageTreeState.message_tree_id.is_(None),
            )
        )

        return [(m.id, m.lang) for m in qry_missing_tree_states.all()]

    _sql_find_tree_ranking_results = """
-- get all ranking results of completed tasks for all parents with >= 2 children
SELECT p.parent_id, mr.* FROM
(
    -- find parents with > 1 children
    SELECT m.parent_id, m.message_tree_id, COUNT(m.id) children_count
    FROM message_tree_state mts
       INNER JOIN message m ON mts.message_tree_id = m.message_tree_id
    WHERE m.review_result                  -- must be reviewed
       AND NOT m.deleted                   -- not deleted
       AND m.parent_id IS NOT NULL         -- ignore initial prompts
       AND (:role IS NULL OR m.role = :role) -- children with matching role
       AND mts.message_tree_id = :message_tree_id
    GROUP BY m.parent_id, m.message_tree_id
    HAVING COUNT(m.id) > 1
) as p
LEFT JOIN task t ON p.parent_id = t.parent_message_id AND t.done AND (t.payload_type = 'RankPrompterRepliesPayload' OR t.payload_type = 'RankAssistantRepliesPayload')
LEFT JOIN message_reaction mr ON mr.task_id = t.id AND mr.payload_type = 'RankingReactionPayload'
"""

    def query_tree_ranking_results(
        self,
        message_tree_id: UUID,
        role_filter: str = "assistant",
    ) -> dict[UUID, list[MessageReaction]]:
        """Finds all completed ranking results for a message_tree"""

        assert role_filter in (None, "assistant", "prompter")

        r = self.db.execute(
            text(self._sql_find_tree_ranking_results),
            {
                "message_tree_id": message_tree_id,
                "role": role_filter,
            },
        )

        rankings_by_message = {}
        for x in r.all():
            parent_id = x["parent_id"]
            if parent_id not in rankings_by_message:
                rankings_by_message[parent_id] = []
            if x["task_id"]:
                rankings_by_message[parent_id].append(MessageReaction.from_orm(x))
        return rankings_by_message

    @managed_tx_method(CommitMode.COMMIT)
    def ensure_tree_states(self) -> None:
        """Add message tree state rows for all root nodes (initial prompt messages)."""

        missing_tree_ids = self.query_misssing_tree_states()
        for id, lang in missing_tree_ids:
            tree_size = self.db.query(func.count(Message.id)).filter(Message.message_tree_id == id).scalar()
            state = message_tree_state.State.INITIAL_PROMPT_REVIEW
            if tree_size > 1:
                state = message_tree_state.State.GROWING
                logger.info(f"Inserting missing message tree state for message: {id} ({tree_size=}, {state=:s})")
            self._insert_default_state(id, lang=lang, state=state)

        halt_prompts_of_disabled_users(self.db)

        # check tree state transitions (maybe variables haves changes): prompt review -> growing -> ranking -> scoring
        prompt_review_trees: list[MessageTreeState] = (
            self.db.query(MessageTreeState)
            .filter(MessageTreeState.state == message_tree_state.State.INITIAL_PROMPT_REVIEW, MessageTreeState.active)
            .all()
        )
        if len(prompt_review_trees) > 0:
            logger.info(
                f"Checking state of {len(prompt_review_trees)} active message trees in 'initial_prompt_review' state."
            )
            for t in prompt_review_trees:
                self.check_condition_for_prompt_lottery(t.message_tree_id)

        growing_trees: list[MessageTreeState] = (
            self.db.query(MessageTreeState)
            .filter(MessageTreeState.state == message_tree_state.State.GROWING, MessageTreeState.active)
            .all()
        )
        if len(growing_trees) > 0:
            logger.info(f"Checking state of {len(growing_trees)} active message trees in 'growing' state.")
            for t in growing_trees:
                self.check_condition_for_ranking_state(t.message_tree_id)

        ranking_trees: list[MessageTreeState] = (
            self.db.query(MessageTreeState)
            .filter(
                or_(
                    MessageTreeState.state == message_tree_state.State.RANKING,
                    MessageTreeState.state == message_tree_state.State.READY_FOR_SCORING,
                ),
                MessageTreeState.active,
            )
            .all()
        )
        if len(ranking_trees) > 0:
            logger.info(f"Checking state of {len(ranking_trees)} active message trees in 'ranking' state.")
            for t in ranking_trees:
                self.check_condition_for_scoring_state(t.message_tree_id)

    def query_num_growing_trees(self, lang: str) -> int:
        """Count all active trees in growing state."""
        query = (
            self.db.query(func.count(MessageTreeState.message_tree_id))
            .join(Message, MessageTreeState.message_tree_id == Message.id)
            .filter(
                MessageTreeState.active,
                MessageTreeState.state == message_tree_state.State.GROWING,
                Message.lang == lang,
            )
        )
        return query.scalar()

    def query_prompt_lottery_waiting(self, lang: str) -> int:
        query = self.db.query(func.count(MessageTreeState.message_tree_id)).filter(
            MessageTreeState.state == message_tree_state.State.PROMPT_LOTTERY_WAITING, MessageTreeState.lang == lang
        )
        return query.scalar()

    def query_num_active_trees(
        self, lang: str, exclude_ranking: bool = True, exclude_prompt_review: bool = True
    ) -> int:
        """Count all active trees (optionally exclude those in ranking and initial prompt review states)."""
        query = (
            self.db.query(func.count(MessageTreeState.message_tree_id))
            .join(Message, MessageTreeState.message_tree_id == Message.id)
            .filter(MessageTreeState.active, Message.lang == lang)
        )
        if exclude_ranking:
            query = query.filter(MessageTreeState.state != message_tree_state.State.RANKING)
        if exclude_prompt_review:
            query = query.filter(MessageTreeState.state != message_tree_state.State.INITIAL_PROMPT_REVIEW)
        return query.scalar()

    def query_reviews_for_message(self, message_id: UUID) -> list[TextLabels]:
        qry = (
            self.db.query(TextLabels)
            .select_from(Task)
            .join(TextLabels, Task.id == TextLabels.id)
            .filter(Task.done, TextLabels.message_id == message_id)
        )
        return qry.all()

    def query_moderation_bad_messages(self, lang: str) -> list[Message]:
        qry = (
            self.db.query(Message)
            .select_from(MessageTreeState)
            .join(Message, MessageTreeState.message_tree_id == Message.message_tree_id)
            .filter(
                MessageTreeState.active,
                or_(
                    MessageTreeState.state == message_tree_state.State.INITIAL_PROMPT_REVIEW,
                    MessageTreeState.state == message_tree_state.State.GROWING,
                ),
                or_(
                    Message.parent_id.is_(None),
                    Message.review_result,
                    and_(Message.parent_id.is_not(None), Message.review_count < self.cfg.num_reviews_reply),
                ),
                not_(Message.deleted),
                or_(
                    coalesce(Message.emojis[protocol_schema.EmojiCode.red_flag].cast(sa.Integer), 0)
                    >= self.cfg.auto_mod_red_flags,
                    coalesce(Message.emojis[protocol_schema.EmojiCode.skip_reply].cast(sa.Integer), 0)
                    >= self.cfg.auto_mod_max_skip_reply,
                ),
            )
        )

        if lang is not None:
            qry = qry.filter(Message.lang == lang)

        return qry.all()

    @managed_tx_method(CommitMode.FLUSH)
    def _insert_tree_state(
        self,
        root_message_id: UUID,
        goal_tree_size: int,
        max_depth: int,
        max_children_count: int,
        active: bool,
        lang: str,
        state: message_tree_state.State = message_tree_state.State.INITIAL_PROMPT_REVIEW,
    ) -> MessageTreeState:
        model = MessageTreeState(
            message_tree_id=root_message_id,
            goal_tree_size=goal_tree_size,
            max_depth=max_depth,
            max_children_count=max_children_count,
            state=state.value,
            active=active,
            lang=lang,
        )

        self.db.add(model)
        return model

    @managed_tx_method(CommitMode.FLUSH)
    def _insert_default_state(
        self,
        root_message_id: UUID,
        lang: str,
        state: message_tree_state.State = message_tree_state.State.INITIAL_PROMPT_REVIEW,
        *,
        goal_tree_size: int = None,
    ) -> MessageTreeState:
        if goal_tree_size is None:
            if self.cfg.random_goal_tree_size and self.cfg.min_goal_tree_size < self.cfg.goal_tree_size:
                goal_tree_size = random.randint(self.cfg.min_goal_tree_size, self.cfg.goal_tree_size)
            else:
                goal_tree_size = self.cfg.goal_tree_size
        return self._insert_tree_state(
            root_message_id=root_message_id,
            goal_tree_size=goal_tree_size,
            max_depth=self.cfg.max_tree_depth,
            max_children_count=self.cfg.max_children_count,
            active=True,
            lang=lang,
            state=state,
        )

    def tree_counts_by_state(self, lang: str = None, only_active: bool = False) -> dict[str, int]:
        qry = self.db.query(MessageTreeState.state, func.count(MessageTreeState.message_tree_id).label("count"))

        if lang is not None:
            qry = (
                qry.select_from(MessageTreeState)
                .join(Message, MessageTreeState.message_tree_id == Message.id)
                .filter(Message.lang == lang)
            )
        if only_active:
            qry = qry.filter(MessageTreeState.active)

        qry = qry.group_by(MessageTreeState.state)
        return {x["state"]: x["count"] for x in qry}

    def tree_counts_by_state_stats(self, lang: str = None, only_active: bool = False) -> TreeStateStats:
        count_by_state = self.tree_counts_by_state(lang=lang, only_active=only_active)
        r = TreeStateStats(
            initial_prompt_review=count_by_state.get(message_tree_state.State.INITIAL_PROMPT_REVIEW) or 0,
            growing=count_by_state.get(message_tree_state.State.GROWING) or 0,
            ranking=count_by_state.get(message_tree_state.State.RANKING) or 0,
            ready_for_scoring=count_by_state.get(message_tree_state.State.READY_FOR_SCORING) or 0,
            ready_for_export=count_by_state.get(message_tree_state.State.READY_FOR_EXPORT) or 0,
            scoring_failed=count_by_state.get(message_tree_state.State.SCORING_FAILED) or 0,
            halted_by_moderator=count_by_state.get(message_tree_state.State.HALTED_BY_MODERATOR) or 0,
            backlog_ranking=count_by_state.get(message_tree_state.State.BACKLOG_RANKING) or 0,
            prompt_lottery_waiting=count_by_state.get(message_tree_state.State.PROMPT_LOTTERY_WAITING) or 0,
            aborted_low_grade=count_by_state.get(message_tree_state.State.ABORTED_LOW_GRADE) or 0,
        )
        return r

    def tree_message_count_stats(self, only_active: bool = True) -> list[TreeMessageCountStats]:
        qry = (
            self.db.query(
                MessageTreeState.message_tree_id,
                func.max(Message.depth).label("depth"),
                func.min(Message.created_date).label("oldest"),
                func.max(Message.created_date).label("youngest"),
                func.count(Message.id).label("count"),
                MessageTreeState.goal_tree_size,
                MessageTreeState.state,
            )
            .select_from(MessageTreeState)
            .join(Message, MessageTreeState.message_tree_id == Message.message_tree_id)
            .filter(not_(Message.deleted))
            .group_by(MessageTreeState.message_tree_id)
        )

        if only_active:
            qry = qry.filter(MessageTreeState.active)

        return [TreeMessageCountStats(**x) for x in qry]

    def stats(self) -> TreeManagerStats:
        return TreeManagerStats(
            state_counts=self.tree_counts_by_state(),
            message_counts=self.tree_message_count_stats(only_active=True),
        )

    def get_user_messages_by_tree(
        self,
        user_id: UUID,
        min_date: datetime = None,
        max_date: datetime = None,
    ) -> Tuple[dict[UUID, list[Message]], list[Message]]:
        """Returns a dict with replies by tree (excluding initial prompts) and list of initial prompts
        associated with user_id."""

        # query all messages of the user
        qry = self.db.query(Message).filter(Message.user_id == user_id)
        if min_date:
            qry = qry.filter(Message.created_date >= min_date)
        if max_date:
            qry = qry.filter(Message.created_date <= max_date)

        prompts: list[Message] = []
        replies_by_tree: dict[UUID, list[Message]] = {}

        # walk over result set and distinguish between initial prompts and replies
        for m in qry:
            m: Message

            if m.message_tree_id == m.id:
                prompts.append(m)
            else:
                message_list = replies_by_tree.get(m.message_tree_id)
                if message_list is None:
                    message_list = [m]
                    replies_by_tree[m.message_tree_id] = message_list
                else:
                    message_list.append(m)

        return replies_by_tree, prompts

    def _purge_message_internal(self, message_id: UUID) -> None:
        """This internal function deletes a single message. It does not take care of
        descendants, children_count in parent etc."""

        sql_purge_message = """
DELETE FROM journal j USING message m WHERE j.message_id = :message_id;
DELETE FROM message_embedding e WHERE e.message_id = :message_id;
DELETE FROM message_toxicity t WHERE t.message_id = :message_id;
DELETE FROM text_labels l WHERE l.message_id = :message_id;
-- delete all ranking results that contain message
DELETE FROM message_reaction r WHERE r.payload_type = 'RankingReactionPayload' AND r.task_id IN (
        SELECT t.id FROM message m
            JOIN task t ON m.parent_id = t.parent_message_id
        WHERE m.id = :message_id);
-- delete task which inserted message
DELETE FROM task t using message m WHERE t.id = m.task_id AND m.id = :message_id;
DELETE FROM task t WHERE t.parent_message_id = :message_id;
DELETE FROM message WHERE id = :message_id;
"""
        parent_id = self.pr.fetch_message(message_id=message_id).parent_id
        r = self.db.execute(text(sql_purge_message), {"message_id": message_id})
        logger.debug(f"purge_message({message_id=}): {r.rowcount} rows.")

        sql_update_ranking_counts = """
WITH r AS (
    -- find ranking results and count per child
    SELECT c.id,
        count(*) FILTER (
            WHERE mr.payload#>'{payload, ranked_message_ids}' ? CAST(c.id AS varchar)
        ) AS ranking_count
    FROM message c
    LEFT JOIN message_reaction mr ON mr.payload_type = 'RankingReactionPayload'
        AND mr.message_id = c.parent_id
    WHERE c.parent_id = :parent_id
    GROUP BY c.id
)
UPDATE message m SET ranking_count = r.ranking_count
FROM r WHERE m.id = r.id AND m.ranking_count != r.ranking_count;
"""

        if parent_id is not None:
            # update ranking counts of remaining children
            r = self.db.execute(text(sql_update_ranking_counts), {"parent_id": parent_id})
            logger.debug(f"ranking_count updated for {r.rowcount} rows.")

    def purge_message_tree(self, message_tree_id: UUID) -> None:
        sql_purge_message_tree = """
DELETE FROM journal j USING message m WHERE j.message_id = m.Id AND m.message_tree_id = :message_tree_id;
DELETE FROM message_embedding e USING message m WHERE e.message_id = m.Id AND m.message_tree_id = :message_tree_id;
DELETE FROM message_toxicity t USING message m WHERE t.message_id = m.Id AND m.message_tree_id = :message_tree_id;
DELETE FROM text_labels l USING message m WHERE l.message_id = m.Id AND m.message_tree_id = :message_tree_id;
DELETE FROM message_reaction r USING task t WHERE r.task_id = t.id AND t.message_tree_id = :message_tree_id;
DELETE FROM task t WHERE t.message_tree_id = :message_tree_id;
DELETE FROM message_tree_state WHERE message_tree_id = :message_tree_id;
DELETE FROM message WHERE message_tree_id = :message_tree_id;
"""
        r = self.db.execute(text(sql_purge_message_tree), {"message_tree_id": message_tree_id})
        logger.debug(f"purge_message_tree({message_tree_id=}) {r.rowcount} rows.")

    def _reactivate_tree(self, mts: MessageTreeState):
        if mts.state == message_tree_state.State.PROMPT_LOTTERY_WAITING:
            return

        tree_id = mts.message_tree_id
        if mts.won_prompt_lottery_date is not None:
            self._enter_state(mts, message_tree_state.State.GROWING)
            if self.check_condition_for_ranking_state(tree_id):
                self.check_condition_for_scoring_state(tree_id)
        else:
            self._enter_state(mts, message_tree_state.State.INITIAL_PROMPT_REVIEW)
            self.check_condition_for_prompt_lottery(tree_id)

    @managed_tx_method(CommitMode.FLUSH)
    def purge_user_messages(
        self,
        user_id: UUID,
        purge_initial_prompts: bool = True,
        min_date: datetime = None,
        max_date: datetime = None,
    ):
        # find all affected message trees
        replies_by_tree, prompts = self.get_user_messages_by_tree(user_id, min_date, max_date)
        total_messages = sum(len(x) for x in replies_by_tree.values())
        logger.debug(f"found: {len(replies_by_tree)} trees; {len(prompts)} prompts; {total_messages} messages;")

        # remove all trees based on initial prompts of the user
        if purge_initial_prompts:
            for p in prompts:
                self.purge_message_tree(p.message_tree_id)
                if p.message_tree_id in replies_by_tree:
                    del replies_by_tree[p.message_tree_id]

        # patch all affected message trees
        for tree_id, replies in replies_by_tree.items():
            bad_parent_ids = set(m.id for m in replies)
            logger.debug(f"patching tree {tree_id=}, {bad_parent_ids=}")

            tree_messages = self.pr.fetch_message_tree(tree_id, reviewed=False, include_deleted=True)
            logger.debug(f"{tree_id=}, {len(bad_parent_ids)=}, {len(tree_messages)=}")
            by_id = {m.id: m for m in tree_messages}

            def ancestor_ids(msg: Message) -> list[UUID]:
                t = []
                while msg.parent_id is not None:
                    msg = by_id[msg.parent_id]
                    t.append(msg.id)
                return t

            def is_descendant_of_deleted(m: Message) -> bool:
                if m.id in bad_parent_ids:
                    return True
                ancestors = ancestor_ids(m)
                if any(a in bad_parent_ids for a in ancestors):
                    return True
                return False

            # start with deepest messages first
            tree_messages.sort(key=lambda x: x.depth, reverse=True)
            for m in tree_messages:
                if is_descendant_of_deleted(m):
                    logger.debug(f"purging message: {m.id}")
                    self._purge_message_internal(m.id)

            # update children counts
            self.pr.update_children_counts(m.message_tree_id)

            # reactivate tree
            logger.info(f"reactivating message tree {tree_id}")
            mts = self.pr.fetch_tree_state(tree_id)
            mts.active = True
            self._reactivate_tree(mts)

    @managed_tx_method(CommitMode.FLUSH)
    def purge_user(self, user_id: UUID, ban: bool = True) -> None:
        self.purge_user_messages(user_id, purge_initial_prompts=True)

        # delete all remaining rows and ban user
        sql_purge_user = """
DELETE FROM journal WHERE user_id = :user_id;
DELETE FROM message_reaction WHERE user_id = :user_id;
DELETE FROM message_emoji WHERE user_id = :user_id;
DELETE FROM task WHERE user_id = :user_id;
DELETE FROM message WHERE user_id = :user_id;
DELETE FROM user_stats WHERE user_id = :user_id;
"""

        r = self.db.execute(text(sql_purge_user), {"user_id": user_id})
        logger.debug(f"purge_user({user_id=}): {r.rowcount} rows.")

        if ban:
            self.db.execute(update(User).filter(User.id == user_id).values(deleted=True, enabled=False))

    @managed_tx_method(CommitMode.COMMIT)
    def retry_scoring_failed_message_trees(self):
        query = self.db.query(MessageTreeState).filter(
            MessageTreeState.state == message_tree_state.State.SCORING_FAILED
        )
        for mts in query.all():
            mts: MessageTreeState
            try:
                if not self.check_condition_for_scoring_state(mts.message_tree_id):
                    mts.active = True
                    self._enter_state(mts, message_tree_state.State.RANKING)
            except Exception:
                logger.exception(f"retry_scoring_failed_message_trees failed for ({mts.message_tree_id=})")

    @managed_tx_method(CommitMode.FLUSH)
    def halt_tree(self, message_id: UUID, halt: bool = True) -> MessageTreeState:
        message = self.pr.fetch_message(message_id, fail_if_missing=True)
        mts = self.pr.fetch_tree_state(message.message_tree_id)

        if halt:
            self._enter_state(mts, message_tree_state.State.HALTED_BY_MODERATOR)
        else:
            self._reactivate_tree(mts)

        return mts


if __name__ == "__main__":
    from oasst_backend.api.deps import api_auth

    # from oasst_backend.api.deps import create_api_client
    from oasst_backend.database import engine
    from oasst_backend.prompt_repository import PromptRepository

    with Session(engine) as db:
        api_client = api_auth(settings.OFFICIAL_WEB_API_KEY, db=db)
        # api_client = create_api_client(session=db, description="test", frontend_type="bot")
        # dummy_user = protocol_schema.User(id="__dummy_user__", display_name="Dummy User", auth_method="local")
        dummy_user = protocol_schema.User(id="1234", display_name="bulb", auth_method="local")

        pr = PromptRepository(db=db, api_client=api_client, client_user=dummy_user)
        cfg = TreeManagerConfiguration()
        tm = TreeManager(db, pr, cfg)
        tm.ensure_tree_states()

        # tm.purge_user_messages(user_id=UUID("2ef9ad21-0dc5-442d-8750-6f7f1790723f"), purge_initial_prompts=False)
        # tm.purge_user(user_id=UUID("2ef9ad21-0dc5-442d-8750-6f7f1790723f"))
        # db.commit()

        # print("query_num_active_trees", tm.query_num_active_trees())
        # print("query_incomplete_rankings", tm.query_incomplete_rankings())
        # print("query_replies_need_review", tm.query_replies_need_review())
        # print("query_incomplete_reply_reviews", tm.query_replies_need_review())
        # xs = tm.query_prompts_need_review(lang="en")
        # print("xs", len(xs))
        # for x in xs:
        #    print(x.id, x.emojis)
        # print("query_incomplete_initial_prompt_reviews", tm.query_prompts_need_review(lang="en"))
        # print("query_extendible_trees", tm.query_extendible_trees())
        # print("query_extendible_parents", tm.query_extendible_parents())

        # print("next_task:", tm.next_task())

        # print(
        #     ".query_tree_ranking_results", tm.query_tree_ranking_results(UUID("21f9d585-d22c-44ab-a696-baa3d83b5f1b"))
        # )
