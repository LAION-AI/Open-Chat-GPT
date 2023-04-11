import { Avatar, AvatarProps, Box, BoxProps, Flex, useColorModeValue } from "@chakra-ui/react";
import { forwardRef, lazy, Suspense, useMemo } from "react";
import { StrictOmit } from "ts-essentials";

const RenderedMarkdown = lazy(() => import("./RenderedMarkdown"));

export type BaseMessageEntryProps = StrictOmit<BoxProps, "bg"> & {
  content: string;
  avatarProps: Pick<AvatarProps, "name" | "src">;
  bg?: string;
};

export const BaseMessageEntry = forwardRef<HTMLDivElement, BaseMessageEntryProps>(function BaseMessageEntry(
  { content, avatarProps, children, ...props },
  ref
) {
  const bg = useColorModeValue("#DFE8F1", "#42536B");
  const actualBg = props.bg ?? bg;
  const avatar = useMemo(
    () => (
      <Avatar
        borderColor="blackAlpha.200"
        _dark={{
          borderColor: "whiteAlpha.200",
        }}
        size={{ base: "xs", md: "sm" }}
        mr={{ base: 0, md: 2 }}
        mt={{ base: 0, md: `6px` }}
        mb={{ base: 1.5, md: 0 }}
        {...avatarProps}
      />
    ),
    [avatarProps]
  );
  return (
    <Flex
      ref={ref}
      gap={0.5}
      flexDirection={{ base: "column", md: "row" }}
      alignItems="start"
      maxWidth="full"
      position="relative"
      p={{ base: 3, md: 0 }}
      borderRadius={{ base: "18px", md: 0 }}
      {...props}
      bg={{ base: actualBg, md: "transparent" }}
    >
      {avatar}
      <Box
        width={["full", "full", "full", "fit-content"]}
        maxWidth={["full", "full", "full", "2xl"]}
        p={{ base: 0, md: 4 }}
        borderRadius={{base: 0, md: "18px"}}
        bg={bg}
        overflowX="auto"
        {...props}
      >
        <Suspense fallback={content}>
          <RenderedMarkdown markdown={content}></RenderedMarkdown>
        </Suspense>
        {children}
      </Box>
    </Flex>
  );
});
