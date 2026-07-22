"""
Argo API LLM Client - Clean adapter for chat completion.

This module provides a LangChain-compatible chat model that uses the Argo API
for language model inference. Supports streaming responses and proper error handling.
"""
import httpx
import logging
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

logger = logging.getLogger(__name__)


class ArgoLLMClient(BaseChatModel):
    """
    Argo API Chat Model wrapper for LangChain.
    
    Provides clean integration with Argo API for chat completions,
    with support for streaming responses and robust error handling.
    
    Attributes:
        base_url: Base URL for Argo API
        model: Model name (e.g., 'gpt-4o', 'llama3')
        username: Argo API username
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
    
    Example:
        ```python
        from src.integrations.argo import ArgoLLMClient
        
        llm = ArgoLLMClient(
            model="gpt-4o",
            username="your-username",
            temperature=0.1
        )
        
        response = llm.invoke("What is the status of the facility?")
        ```
    """
    
    base_url: str = "https://your-argo-gateway.example.com/api/v1/resource" # https://your-argo-gateway.example.com/api/v1/resource
    model: str = "gpt-4o"
    username: str = "your-username"
    temperature: float = 0.1
    max_tokens: int = 12_000
    timeout: float = 240.0
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type"""
        return "argo_chat"
    
    def _get_chat_url(self) -> str:
        """Get the chat URL - using streamchat endpoint"""
        return f"{self.base_url}/streamchat/" # f"{self.base_url}/streamchat/"
    
    def _convert_messages_to_argo_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to Argo API format.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            List of message dictionaries in Argo format
        """
        argo_messages = []
        
        for msg_index, message in enumerate(messages):
            if isinstance(message, SystemMessage):
                argo_messages.append({"role": "system", "content": message.content})
                logger.debug(f"System message {msg_index}: {message.content[:100]}...")
            elif isinstance(message, HumanMessage):
                argo_messages.append({"role": "user", "content": message.content})
                logger.debug(f"User message {msg_index}: {message.content[:100]}...")
            elif isinstance(message, AIMessage):
                argo_messages.append({"role": "assistant", "content": message.content})
                logger.debug(f"Assistant message {msg_index}: {message.content[:100]}...")
        
        return argo_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat response using Argo API with streaming.
        
        Args:
            messages: List of chat messages
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional generation parameters
            
        Returns:
            ChatResult with generated response
            
        Raises:
            Exception: If API call fails
        """
        
        chat_url = self._get_chat_url()
        
        logger.info("🚀 ARGO API CALL INITIATED")
        logger.info(f"📍 URL: {chat_url}")
        logger.info(f"👤 Username: {self.username}")
        logger.info(f"🤖 Model: {self.model}")
        logger.info(f"📝 Number of messages: {len(messages)}")
        
        # Convert messages to Argo format
        argo_messages = self._convert_messages_to_argo_format(messages)
        
        # Prepare Argo API request
        data = {
            "user": self.username,
            "model": self.model,
            "messages": argo_messages,
            "stop": stop or [],
            "temperature": self.temperature
        }
        
        # Headers for streaming
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",
            "Cache-Control": "no-cache",
            "Accept-Encoding": "identity",
        }
        
        try:
            full_response = ""
            
            with httpx.stream(
                "POST", chat_url, json=data, headers=headers, timeout=self.timeout
            ) as response:
                if response.status_code == 200:
                    for chunk in response.iter_text():
                        if chunk:
                            full_response += chunk
                else:
                    error_msg = f"Argo API error: {response.status_code}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            
            logger.info(f"✅ Response received: {len(full_response)} characters")
            
            message = AIMessage(content=full_response.strip())
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except httpx.TimeoutException as e:
            logger.error(f"Argo API timeout after {self.timeout}s: {e}")
            raise Exception(f"Argo API timeout: {e}")
        except Exception as e:
            logger.error(f"Argo API error: {e}")
            raise e
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async generate - delegates to sync for now.
        
        TODO: Implement true async with httpx.AsyncClient
        """
        return self._generate(messages, stop, None, **kwargs)
