import logging
from typing_extensions import override
from typing import Any, AsyncGenerator, Optional, cast

import openai

from .openai import OpenAIModel
from ..types.content import Messages
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec

logger = logging.getLogger(__name__)

class AzureOpenAIModel(OpenAIModel):
  @override
  def format_request(
      self,
      messages: Messages,
      tool_specs: Optional[list[ToolSpec]] = None,
      system_prompt: Optional[str] = None,
      tool_choice: ToolChoice | None = None,
  ) -> dict[str, Any]:
      """Format an OpenAI compatible chat streaming request.

      Args:
          messages: List of message objects to be processed by the model.
          tool_specs: List of tool specifications to make available to the model.
          system_prompt: System prompt to provide context to the model.
          tool_choice: Selection strategy for tool invocation.

      Returns:
          An OpenAI compatible chat streaming request.

      Raises:
          TypeError: If a message contains a content block type that cannot be converted to an OpenAI-compatible
              format.
      """
      request = {
          "messages": self.format_request_messages(messages, system_prompt),
          "model": self.config["model_id"],
          "stream": True,
          "stream_options": {"include_usage": True},
          **(self._format_request_tool_choice(tool_choice)),
          **cast(dict[str, Any], self.config.get("params", {})),
      }

      if tool_specs is not None:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "parameters": tool_spec["inputSchema"]["json"],
                },
            }
            for tool_spec in tool_specs
        ]

      return request

  @override
  async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the OpenAI model from Azure.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by OpenAI (rate limits).
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("formatted request=<%s>", request)

        logger.debug("invoking model")

        async with openai.AsyncAzureOpenAI(**self.client_args) as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                # Check if this is a context length exceeded error
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    logger.warning("OpenAI threw context window overflow error")
                    raise ContextWindowOverflowException(str(e)) from e
                # Re-raise other BadRequestError exceptions
                raise
            except openai.RateLimitError as e:
                # All rate limit errors should be treated as throttling, not context overflow
                # Rate limits (including TPM) require waiting/retrying, not context reduction
                logger.warning("OpenAI threw rate limit error")
                raise ModelThrottledException(str(e)) from e

            logger.debug("got response from model")
            yield self.format_chunk({"chunk_type": "message_start"})
            yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

            tool_calls: dict[int, list[Any]] = {}

            async for event in response:
                # Defensive: skip events with empty or missing choices
                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]

                if choice.delta.content:
                    yield self.format_chunk(
                        {"chunk_type": "content_delta", "data_type": "text", "data": choice.delta.content}
                    )

                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "reasoning_content",
                            "data": choice.delta.reasoning_content,
                        }
                    )

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    break

            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

            for tool_deltas in tool_calls.values():
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

                for tool_delta in tool_deltas:
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

            # Skip remaining events as we don't have use for anything except the final usage payload
            async for event in response:
                pass

            if event.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})

        logger.debug("finished streaming response from model")