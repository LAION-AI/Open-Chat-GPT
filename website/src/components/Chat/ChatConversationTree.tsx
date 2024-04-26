import { Box } from "@chakra-ui/react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { put } from "src/lib/api";
import { API_ROUTES } from "src/lib/routes";
import { InferenceMessage } from "src/types/Chat";
import { buildTree, Tree } from "src/utils/buildTree";
import { StrictOmit } from "ts-essentials";
import { useIsomorphicLayoutEffect } from "usehooks-ts";

import { BaseMessageEmojiButton } from "../Messages/MessageEmojiButton";
import { MessageInlineEmojiRow } from "../Messages/MessageInlineEmojiRow";
import { ChatMessageEntry, ChatMessageEntryProps } from "./ChatMessageEntry";

type ChatConversationTreeProps = {
  messages: InferenceMessage[];
  retryingParentId: string | null;
  activeThreadTailMessageId: string | null;
} & StrictOmit<ChatMessageEntryProps, "message">;

export const LAST_ASSISTANT_MESSAGE_ID = "last_assistant_message";

export const ChatConversationTree = memo(function ChatConversationTree({
  messages,
  retryingParentId,
  activeThreadTailMessageId,
  ...props
}: ChatConversationTreeProps) {
  const tree = useMemo(() => buildTree(messages), [messages]);
  if (!tree) return null;

  return (
    <>
      <ChatMessageEntry message={tree} {...props}></ChatMessageEntry>
      {tree.children.length > 0 && (
        <TreeChildren
          trees={tree.children}
          {...props}
          retryingParentId={retryingParentId}
          activeThreadTailMessageId={activeThreadTailMessageId}
        />
      )}
    </>
  );
});

const TreeChildren = ({
  trees,
  retryingParentId,
  activeThreadTailMessageId,
  ...props
}: {
  trees: Tree<InferenceMessage>[];
  retryingParentId: string | null;
  activeThreadTailMessageId: string | null;
} & StrictOmit<ChatMessageEntryProps, "message" | "canRetry">) => {
  const sortedTrees = useMemo(() => trees.sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at)), [trees]);
  const [index, setIndex] = useState(0);
  const currentTree = sortedTrees[Math.min(index, trees.length - 1)];

  const length = trees.length;
  const hasSibling = length > 1;

  useIsomorphicLayoutEffect(() => {
    const activeIndex = trees.findIndex((tree) => {
      const searchInChildren = (children) => {
        return children.some((child) => {
          if (child.id === activeThreadTailMessageId) {
            return true;
          }
          return searchInChildren(child.children);
        });
      };
      return tree.children.length === 0 ? tree.id === activeThreadTailMessageId : searchInChildren(tree.children);
    });

    setIndex(activeIndex === -1 ? trees.length - 1 : activeIndex);
  }, [trees.length, activeThreadTailMessageId]);

  useEffect(() => {
    const updateActiveThread = async () => {
      await put(API_ROUTES.UPDATE_CHAT(), {
        arg: { chat_id: currentTree.chat_id, active_thread_tail_message_id: currentTree.id },
      });
    };

    if (currentTree.children.length === 0) {
      updateActiveThread().catch(console.error);
    }
  }),
    [currentTree];

  const handlePrevious = useCallback(() => {
    setIndex((i) => (i > 0 ? i - 1 : i));
  }, [setIndex]);
  const handleNext = useCallback(() => {
    setIndex((i) => (i < length - 1 ? i + 1 : i));
  }, [length, setIndex]);

  const isRegenerating = retryingParentId !== null && trees.findIndex((t) => t.parent_id === retryingParentId) !== -1;

  if (isRegenerating && trees[0].role === "assistant") {
    return null; // this node is being regenerated, so we skip render
  }

  const isLeaf = currentTree.children.length === 0;

  const node = (
    <ChatMessageEntry
      message={currentTree}
      {...props}
      canRetry={isLeaf}
      showEncourageMessage={props.showEncourageMessage && isLeaf}
      // TODO refactor away from this dirty hack
      id={isLeaf && currentTree.role === "assistant" ? LAST_ASSISTANT_MESSAGE_ID : undefined}
      data-id={currentTree.id}
      pagingSlot={
        hasSibling ? (
          <>
            <MessageInlineEmojiRow gap="0.5">
              <BaseMessageEmojiButton
                emoji={ChevronLeft}
                onClick={handlePrevious}
                isDisabled={index === 0}
              ></BaseMessageEmojiButton>
              <Box fontSize="xs">{`${index + 1}/${length}`}</Box>
              <BaseMessageEmojiButton
                emoji={ChevronRight}
                onClick={handleNext}
                isDisabled={index === length - 1}
              ></BaseMessageEmojiButton>
            </MessageInlineEmojiRow>
          </>
        ) : undefined
      }
    ></ChatMessageEntry>
  );

  if (isRegenerating) {
    return node;
  }

  return (
    <>
      {node}
      {currentTree.children.length > 0 && (
        <TreeChildren
          retryingParentId={retryingParentId}
          trees={currentTree.children}
          activeThreadTailMessageId={activeThreadTailMessageId}
          {...props}
        />
      )}
    </>
  );
};
