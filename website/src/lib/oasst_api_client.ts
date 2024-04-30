import type { EmojiOp, FetchMessagesCursorResponse, Message, MessageRevision } from "src/types/Conversation";
import { LeaderboardReply, LeaderboardTimeFrame } from "src/types/Leaderboard";
import { Stats } from "src/types/Stat";
import type { AvailableTasks } from "src/types/Task";
import { FetchTrollBoardResponse, TrollboardTimeFrame } from "src/types/Trollboard";
import type { BackendUser, BackendUserCore, FetchUsersParams, FetchUsersResponse } from "src/types/Users";

export class OasstError {
  message: string;
  errorCode: number;
  httpStatusCode: number;
  path: string;
  method: string;

  constructor({
    errorCode,
    httpStatusCode,
    message,
    path,
    method,
  }: {
    message: string;
    errorCode: number;
    httpStatusCode: number;
    path: string;
    method: string;
  }) {
    this.message = message;
    this.errorCode = errorCode;
    this.httpStatusCode = httpStatusCode;
    this.path = path;
    this.method = method;
  }

  toString() {
    return JSON.stringify(this);
  }
}

export class OasstApiClient {
  oasstApiUrl: string;
  oasstApiKey: string;
  userHeaders: Record<string, string> = {};

  constructor(oasstApiUrl: string, oasstApiKey: string, user?: BackendUserCore) {
    this.oasstApiUrl = oasstApiUrl;
    this.oasstApiKey = oasstApiKey;
    if (user) {
      this.userHeaders = {
        "X-OASST-USER": `${user.auth_method}:${user.id}`,
      };
    }
  }

  private async request<T>(method: "GET" | "POST" | "PUT" | "DELETE", path: string, init?: RequestInit): Promise<T> {
    const resp = await fetch(`${this.oasstApiUrl}${path}`, {
      method,
      ...init,
      headers: {
        ...init?.headers,
        ...this.userHeaders,
        "X-API-Key": this.oasstApiKey,
        "Content-Type": "application/json",
      },
    });

    if (resp.status === 204) {
      return null as T;
    }

    if (resp.status >= 300) {
      const errorText = await resp.text();
      let error;
      try {
        error = JSON.parse(errorText);
      } catch (e) {
        throw new OasstError({
          message: errorText,
          errorCode: 0,
          httpStatusCode: resp.status,
          path,
          method,
        });
      }
      throw new OasstError({
        message: error.message ?? error,
        errorCode: error.error_code,
        httpStatusCode: resp.status,
        path,
        method,
      });
    }

    return resp.json();
  }

  private async post<T>(path: string, body: unknown) {
    return this.request<T>("POST", path, {
      body: JSON.stringify(body),
    });
  }

  private async put<T>(path: string) {
    return this.request<T>("PUT", path);
  }

  private async get<T>(path: string, query?: Record<string, string | number | boolean | undefined>) {
    if (!query) {
      return this.request<T>("GET", path);
    }

    const filteredQuery = Object.fromEntries(
      Object.entries(query).filter(([, value]) => value !== undefined)
    ) as Record<string, string>;

    const params = new URLSearchParams(filteredQuery).toString();

    return this.request<T>("GET", `${path}?${params}`);
  }

  private async delete<T>(path: string) {
    return this.request<T>("DELETE", path);
  }

  // TODO return a strongly typed Task?
  // This method is used to store a task in RegisteredTask.task.
  // This is a raw Json type, so we can't use it to strongly type the task.
  async fetchTask(taskType: string, user: BackendUserCore, lang: string): Promise<any> {
    return this.post("/api/v1/tasks/", {
      type: taskType,
      user,
      lang,
    });
  }

  async ackTask(taskId: string, messageId: string): Promise<null> {
    return this.post(`/api/v1/tasks/${taskId}/ack`, { message_id: messageId });
  }

  async nackTask(taskId: string): Promise<null> {
    return this.post(`/api/v1/tasks/${taskId}/nack`, {});
  }

  // TODO return a strongly typed Task?
  // This method is used to record interaction with task while fetching next task.
  // This is a raw Json type, so we can't use it to strongly type the task.
  async interactTask(
    updateType: string,
    taskId: string,
    messageId: string,
    userMessageId: string,
    content: object,
    user: BackendUserCore,
    lang: string
  ): Promise<any> {
    return this.post("/api/v1/tasks/interaction", {
      type: updateType,
      user,
      task_id: taskId,
      message_id: messageId,
      user_message_id: userMessageId,
      lang,
      ...content,
    });
  }

  fetch_full_settings() {
    return this.get<Record<string, any>>("/api/v1/admin/backend_settings/full");
  }

  fetch_public_settings() {
    return this.get<Record<string, any>>("/api/v1/admin/backend_settings/public");
  }

  /**
   * Returns the tasks availability information for given `user`.
   */
  async fetch_tasks_availability(user: object): Promise<AvailableTasks | null> {
    return this.post<AvailableTasks>("/api/v1/tasks/availability", user);
  }

  /**
   * Returns the `Message`s associated with `user_id` in the backend.
   */
  async fetch_message(message_id: string, user: BackendUserCore): Promise<Message> {
    return this.get<Message>(`/api/v1/messages/${message_id}?username=${user.id}&auth_method=${user.auth_method}`);
  }

  async fetch_message_tree(message_id: string, options?: { include_spam?: boolean; include_deleted?: boolean }) {
    return this.get<{
      id: string;
      messages: Message[];
    }>(`/api/v1/messages/${message_id}/tree`, options);
  }

  /**
   *  Returns the Message's tree state
   */
  async fetch_message_tree_state(message_id: string) {
    return this.get<{
      message_tree_id: string;
      state: string;
      active: boolean;
      goal_tree_size: number;
      max_children_count: number;
      max_depth: number;
      origin: string;
    }>(`/api/v1/messages/${message_id}/tree/state`);
  }

  /**
   * Returns a list of revisions associated with `message_id`.
   */
  async fetch_message_revision_history(message_id: string): Promise<MessageRevision[]> {
    return this.get<MessageRevision[]>(`/api/v1/messages/${message_id}/history`);
  }

  /**
   * Delete a message by its id
   */
  async delete_message(message_id: string): Promise<void> {
    return this.delete<void>(`/api/v1/messages/${message_id}`);
  }

  /**
   * Undelete a message by its id
   */
  async undelete_message(message_id: string): Promise<void> {
    return this.put<void>(`/api/v1/messages/${message_id}/undelete`);
  }

  /**
   * Modify a message's content and save its previous content as a revision
   */
  async edit_message(message_id: string, user: BackendUserCore, new_content: string) {
    return this.post<void>(`/api/v1/messages/${message_id}/edit`, {
      message_id,
      user,
      new_content,
    });
  }

  /**
   * Set whether a tree is halted
   */
  async set_tree_halted(message_id: string, halt: boolean): Promise<void> {
    return this.put<void>(`/api/v1/messages/${message_id}/tree/state?halt=${halt}`);
  }

  /**
   * Send a report about a message
   */
  async send_report(message_id: string, user: BackendUserCore, text: string) {
    return this.post("/api/v1/text_labels/", {
      type: "text_labels",
      message_id,
      labels: [], // Not yet implemented
      text,
      is_report: true,
      user,
    });
  }

  /**
   * Returns cached dataset stats from the backend.
   */
  async fetch_cached_stats(): Promise<Stats> {
    return this.get("/api/v1/stats/cached");
  }

  /**
   * Returns the message stats from the backend.
   */
  async fetch_stats(): Promise<any> {
    return this.get("/api/v1/stats/");
  }

  /**
   * Returns the tree manager stats from the backend.
   */
  async fetch_tree_manager(): Promise<any> {
    return this.get("/api/v1/stats/tree_manager");
  }

  /**
   * Returns the `BackendUser` associated with `user_id`
   */
  async fetch_user(user_id: string): Promise<BackendUser | null> {
    return this.get(`/api/v1/users/${user_id}`);
  }

  /**
   * Returns the set of `BackendUser`s stored by the backend.
   */
  async fetch_users({
    direction,
    limit,
    cursor,
    searchDisplayName,
    sortKey = "display_name",
  }: FetchUsersParams): Promise<FetchUsersResponse | null> {
    return this.get<FetchUsersResponse>(`/api/v1/users/cursor`, {
      search_text: searchDisplayName,
      sort_key: sortKey,
      max_count: limit,
      after: direction === "forward" ? cursor : undefined,
      before: direction === "back" ? cursor : undefined,
    });
  }

  /**
   * Returns the `Message`s associated with `user_id` in the backend.
   */
  async fetch_user_messages(user_id: string): Promise<Message[] | null> {
    return this.get<Message[]>(`/api/v1/users/${user_id}/messages`);
  }

  async fetch_user_messages_cursor(
    user_id: string,
    {
      direction,
      cursor,
      ...rest
    }: { include_deleted?: boolean; max_count?: number; cursor?: string; direction: "forward" | "back"; desc?: boolean }
  ) {
    return this.get<FetchMessagesCursorResponse>(`/api/v1/users/${user_id}/messages/cursor`, {
      ...rest,
      after: direction === "forward" ? cursor : undefined,
      before: direction === "back" ? cursor : undefined,
    });
  }

  /**
   * Updates the backend's knowledge about the `user_id`.
   */
  async set_user_status(
    user_id: string,
    is_enabled: boolean,
    notes: string,
    show_on_leaderboard: boolean,
    display_name: string
  ): Promise<void> {
    await this.put(
      `/api/v1/users/${user_id}?enabled=${is_enabled}&notes=${notes}&show_on_leaderboard=${show_on_leaderboard}&display_name=${display_name}`
    );
  }

  /**
   * Returns the valid labels for messages.
   */
  async fetch_valid_text(messageId?: string): Promise<any> {
    return this.get("/api/v1/text_labels/valid_labels", { message_id: messageId });
  }

  /**
   * Returns the current leaderboard ranking.
   */
  async fetch_leaderboard(
    time_frame: LeaderboardTimeFrame,
    { limit = 20 }: { limit?: number }
  ): Promise<LeaderboardReply | null> {
    return this.get<LeaderboardReply>(`/api/v1/leaderboards/${time_frame}`, { max_count: limit });
  }

  /**
   * Returns the counts of all tasks (some might be zero)
   */
  async fetch_available_tasks(user: BackendUserCore, lang: string): Promise<AvailableTasks | null> {
    return this.post<AvailableTasks>(`/api/v1/tasks/availability?lang=${lang}`, user);
  }

  /**
   * Add/remove an emoji on a message for a user
   */
  async set_user_message_emoji(message_id: string, user: BackendUserCore, emoji: string, op: EmojiOp): Promise<void> {
    await this.post(`/api/v1/messages/${message_id}/emoji`, {
      user,
      emoji,
      op,
    });
  }

  fetch_my_messages(user: BackendUserCore) {
    const params = new URLSearchParams({
      username: user.id,
      auth_method: user.auth_method,
    });
    return this.get<Message[]>(`/api/v1/messages/?${params}`);
  }

  fetch_my_messages_cursor(
    user: BackendUserCore,
    {
      direction,
      cursor,
      ...rest
    }: { include_deleted?: boolean; max_count?: number; cursor?: string; direction: "forward" | "back"; desc?: boolean }
  ) {
    return this.get<FetchMessagesCursorResponse>(`/api/v1/messages/cursor`, {
      ...rest,
      username: user.id,
      auth_method: user.auth_method,
      after: direction === "forward" ? cursor : undefined,
      before: direction === "back" ? cursor : undefined,
    });
  }

  fetch_recent_messages(lang: string) {
    return this.get<Message[]>(`/api/v1/messages/`, { lang });
  }

  fetch_message_children(messageId: string) {
    return this.get<Message[]>(`/api/v1/messages/${messageId}/children`);
  }

  fetch_conversation(messageId: string) {
    return this.get(`/api/v1/messages/${messageId}/conversation`);
  }

  async set_tos_acceptance(user: BackendUserCore) {
    // NOTE: we do a post here to force create the user if it does not exist
    const backendUser = await this.upsert_frontend_user(user);
    await this.put<void>(`/api/v1/users/${backendUser.user_id}?tos_acceptance=true`);
  }

  async fetch_user_stats(user: BackendUserCore) {
    const backendUser = await this.get<BackendUser>(`/api/v1/frontend_users/${user.auth_method}/${user.id}`);
    return this.get(`/api/v1/users/${backendUser.user_id}/stats`);
  }

  fetch_user_stats_window(user_id: string, time_frame: LeaderboardTimeFrame, window_size?: number) {
    return this.get<LeaderboardReply>(`/api/v1/users/${user_id}/stats/${time_frame}/window`, {
      window_size,
    });
  }

  fetch_frontend_user(user: BackendUserCore) {
    return this.get<BackendUser>(`/api/v1/frontend_users/${user.auth_method}/${user.id}`);
  }

  // TODO: add update-able fields eg: enabled, notes, show_on_leaderboard, etc..
  upsert_frontend_user(user: BackendUserCore) {
    // the backend does a upsert operation with this call
    return this.post<BackendUser>(`/api/v1/frontend_users/`, user);
  }

  fetch_trollboard(time_frame: TrollboardTimeFrame, { limit, enabled }: { limit?: number; enabled?: boolean }) {
    return this.get<FetchTrollBoardResponse>(`/api/v1/trollboards/${time_frame}`, {
      max_count: limit,
      enabled: enabled,
    });
  }

  fetch_messages_cursor({
    direction,
    cursor,
    ...rest
  }: {
    direction: "back" | "forward";
    cursor?: string;
    user_id?: string;
    auth_method?: string;
    username?: string;
    api_client_id?: string;
    only_roots?: boolean;
    include_deleted?: boolean;
    max_count?: number;
    desc?: boolean;
    lang?: string;
    include_user?: boolean;
    search_query?: string;
  }) {
    return this.get<FetchMessagesCursorResponse>("/api/v1/messages/cursor", {
      ...rest,
      after: direction === "forward" ? cursor : undefined,
      before: direction === "back" ? cursor : undefined,
    });
  }

  async delete_account(user: BackendUserCore) {
    const backendUser = await this.fetch_frontend_user(user);
    return this.delete<void>(`/api/v1/users/${backendUser.user_id}`);
  }
}
