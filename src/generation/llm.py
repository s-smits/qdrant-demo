"""
LLM abstraction layer for answer generation.

Supports multiple providers with a unified interface.
"""

from typing import Optional, Union
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def invoke(self, prompt: str):
        """Invoke with LangChain-compatible interface."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM wrapper."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            model_name: Model to use
            api_key: OpenAI API key (from env if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        from langchain_openai import ChatOpenAI
        import os
        
        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        """Generate a response."""
        response = self.model.invoke(prompt)
        return response.content
    
    def invoke(self, prompt: str):
        """LangChain-compatible invoke."""
        return self.model.invoke(prompt)


class LocalLLM(BaseLLM):
    """
    Local LLM using transformers.
    
    Uses lightweight models like Qwen3-0.6B for testing.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        max_new_tokens: int = 512,
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: HuggingFace model name
            device: Device for inference
            max_new_tokens: Maximum tokens to generate
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else None,
        )
        
        if device == "mps":
            self.model = self.model.to(device)
    
    def generate(self, prompt: str) -> str:
        """Generate a response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def invoke(self, prompt: str):
        """LangChain-compatible invoke."""
        from dataclasses import dataclass
        
        @dataclass
        class Response:
            content: str
        
        return Response(content=self.generate(prompt))


class RAGGenerator:
    """
    RAG answer generator.
    
    Combines retrieved context with LLM to generate answers.
    """
    
    DEFAULT_PROMPT = """Based on the following context, please answer the question. 
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize RAG generator.
        
        Args:
            llm: LLM for generation
            prompt_template: Custom prompt template with {context} and {question}
        """
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
    
    def generate(
        self,
        question: str,
        context_docs: list[dict],
        max_context_length: int = 4000,
    ) -> str:
        """
        Generate an answer using retrieved context.
        
        Args:
            question: User question
            context_docs: Retrieved documents with 'text' key
            max_context_length: Maximum context characters
            
        Returns:
            Generated answer
        """
        # Build context from documents
        context_parts = []
        current_length = 0
        
        for doc in context_docs:
            text = doc.get("text", "")
            if current_length + len(text) > max_context_length:
                break
            context_parts.append(text)
            current_length += len(text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )
        
        # Generate response
        return self.llm.generate(prompt)
    
    def generate_with_sources(
        self,
        question: str,
        context_docs: list[dict],
    ) -> dict:
        """
        Generate answer with source attribution.
        
        Args:
            question: User question
            context_docs: Retrieved documents
            
        Returns:
            Dict with 'answer' and 'sources'
        """
        answer = self.generate(question, context_docs)
        
        sources = [
            {
                "text": doc.get("text", "")[:200] + "...",
                "source": doc.get("metadata", {}).get("source", "unknown"),
            }
            for doc in context_docs[:3]
        ]
        
        return {
            "answer": answer,
            "sources": sources,
        }


def get_llm(
    provider: str = "openai",
    **kwargs,
) -> BaseLLM:
    """
    Factory function to get LLM instance.
    
    Args:
        provider: "openai" or "local"
        **kwargs: Provider-specific arguments
        
    Returns:
        LLM instance
    """
    if provider == "openai":
        return OpenAILLM(**kwargs)
    elif provider == "local":
        return LocalLLM(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
