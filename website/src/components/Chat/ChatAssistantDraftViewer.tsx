import { Avatar, Box, Button, Collapse, Divider, Flex, Spinner, Tag, Text, useColorModeValue } from "@chakra-ui/react";
import { useEffect, useState } from "react";

import { WorkParametersDisplay } from "./WorkParameters";

export const ChatAssistantDraftViewer = ({ streamedDrafts = [], draftMessages = [] }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  const [expandedMessage, setExpandedMessage] = useState<Number>(0);

  useEffect(() => {
    const areAllMessagesEmpty = streamedDrafts.every((message) => message === "");
    setIsLoading(areAllMessagesEmpty);
  }, [streamedDrafts]);

  useEffect(() => {
    const areAllMessagesComplete =
      draftMessages &&
      draftMessages.every((message) =>
        ["complete", "aborted_by_worker", "cancelled", "timeout"].includes(message.state)
      );
    setIsComplete(areAllMessagesComplete);
  }, [draftMessages]);

  const handleToggleExpand = (index) => {
    setExpandedMessage(index === expandedMessage ? -1 : index);
  };

  const containerBackgroundColor = useColorModeValue("#DFE8F1", "#42536B");
  const responseHoverBackgroundColor = useColorModeValue("gray.300", "blackAlpha.300");

  return (
    <Flex
      key="chat-message-select"
      width="full"
      gap={0.5}
      flexDirection={{ base: "column", md: "row" }}
      alignItems="start"
      position="relative"
      padding={{ base: 3, md: 0 }}
      background={{ base: containerBackgroundColor, md: "transparent" }}
      borderRadius={{ base: "18px", md: 0 }}
      maxWidth={{ base: "3xl", "2xl": "4xl" }}
    >
      <Avatar
        size={{ base: "xs", md: "sm" }}
        marginRight={{ base: 0, md: 2 }}
        marginTop={{ base: 0, md: `6px` }}
        marginBottom={{ base: 1.5, md: 0 }}
        borderColor={"blackAlpha.200"}
        _dark={{ backgroundColor: "blackAlpha.300" }}
        src="/images/logos/logo.png"
      />
      <Flex
        background={containerBackgroundColor}
        borderRadius={{ base: 0, md: "18px" }}
        gap={1}
        width={isLoading ? "fit-content" : "full"}
        padding={{ base: 0, md: 4 }}
        flexDirection="column"
      >
        {isLoading ? (
          <Spinner color="blue.500" />
        ) : (
          <>
            {streamedDrafts.map((message, index) => (
              <>
                <Box
                  key={`message-${index}`}
                  borderRadius="md"
                  padding={2}
                  gap={2}
                  width={"full"}
                  rounded="md"
                  cursor="pointer"
                  _hover={{ backgroundColor: responseHoverBackgroundColor }}
                  transition="background-color 0.2 ease-in-out"
                >
                  <Tag size="md" variant="outline" maxW="max-content" marginBottom={1.5} borderRadius="full">
                    Draft {index + 1}
                  </Tag>
                  <Collapse startingHeight={100} in={expandedMessage === index}>
                    <Text
                      bgGradient={
                        expandedMessage === index
                          ? "linear(to-b, #000000, #000000)"
                          : "linear(to-b, #000000 50px, #00000000 100px)"
                      }
                      _dark={{
                        bgGradient:
                          expandedMessage === index
                            ? "linear(to-b, #FFFFFF, #FFFFFF)"
                            : "linear(to-b, #FFFFFF 50px, #00000000 100px)",
                      }}
                      backgroundClip={"text"}
                    >
                      {message}
                    </Text>
                  </Collapse>
                  <Button
                    size="sm"
                    width={"fit-content"}
                    marginTop="1rem"
                    marginLeft={"auto"}
                    background={"white"}
                    _dark={{
                      background: "gray.500",
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleToggleExpand(index);
                    }}
                  >
                    Show {expandedMessage === index ? "Less" : "More"}
                  </Button>
                </Box>
                {index !== streamedDrafts.length - 1 && (
                  <Divider
                    key={`divider-${index}`}
                    marginY="2.0"
                    borderColor="blackAlpha.300"
                    _dark={{
                      borderColor: "blackAlpha.700",
                    }}
                  />
                )}
              </>
            ))}
            {isComplete && draftMessages && draftMessages[0]?.work_parameters && (
              <WorkParametersDisplay parameters={draftMessages[0].work_parameters} />
            )}
          </>
        )}
      </Flex>
    </Flex>
  );
};
