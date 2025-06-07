import os
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator
import openai
import anthropic
import google.generativeai as genai
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content: str
    usage: Dict[str, Any] = {}
    model: str

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream_generate(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(self, messages: list, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=response.usage.model_dump() if response.usage else {},
            model=self.model
        )
    
    async def stream_generate(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(self, messages: list, **kwargs) -> LLMResponse:
        # Convert OpenAI format to Anthropic format
        system_msg = None
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        response = await self.client.messages.create(
            model=self.model,
            messages=user_messages,
            system=system_msg,
            max_tokens=kwargs.get('max_tokens', 1000),
            **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
        )
        
        return LLMResponse(
            content=response.content[0].text,
            usage={"input_tokens": response.usage.input_tokens, 
                   "output_tokens": response.usage.output_tokens},
            model=self.model
        )
    
    async def stream_generate(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        system_msg = None
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        async with self.client.messages.stream(
            model=self.model,
            messages=user_messages,
            system=system_msg,
            max_tokens=kwargs.get('max_tokens', 1000),
            **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
        ) as stream:
            async for text in stream.text_stream:
                yield text

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    async def generate(self, messages: list, **kwargs) -> LLMResponse:
        # Convert to Gemini format
        prompt = self._convert_messages_to_prompt(messages)
        response = await self.model.generate_content_async(prompt)
        
        return LLMResponse(
            content=response.text,
            usage={"total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0},
            model=self.model_name
        )
    
    async def stream_generate(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        prompt = self._convert_messages_to_prompt(messages)
        response = await self.model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def _convert_messages_to_prompt(self, messages: list) -> str:
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        return prompt.strip()

class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_name: str, api_key: str, model: str) -> BaseLLMProvider:
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return providers[provider_name](api_key, model)