"""
Claude Multimodal License Plate Judge
"""
from typing import Dict, Any
import anthropic
import time
import asyncio
import base64
import json

PROMPT = '''\
You are an MLS compliance AI inspector.

Task:
Carefully analyze the image and determine whether any visible vehicle license plate is present.

Important rules:
- License plates include any alphanumeric plate attached to cars, trucks, bikes, trailers, or parked vehicles.
- Even partially visible plates count.
- Reflections count.
- Blurry but readable plates count.
- Decorative signs that are NOT vehicle plates should NOT count.
- If uncertain, return UNCERTAIN with confidence below 50.

Return STRICT JSON only:
{
  "license_plate_detected": true/false,
  "confidence": 0-100,
  "reasoning": "brief technical reasoning"
}
'''

class BaseLLMJudge:
    def analyze(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def analyze_async(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError

class ClaudeJudge(BaseLLMJudge):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        if image_path.lower().endswith(".png"):
            media_type = "image/png"
        elif image_path.lower().endswith(".webp"):
            media_type = "image/webp"
        elif image_path.lower().endswith(".gif"):
            media_type = "image/gif"
        else:
            media_type = "image/jpeg"
        image_data = base64.standard_b64encode(img_bytes).decode("utf-8")
        return image_data, media_type

    def _build_messages(self, image_data: str, media_type: str):
        return [
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "text", "text": "Analyze this image for license plates."},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}}
            ]}
        ]

    def analyze(self, image_path: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                image_data, media_type = self._encode_image(image_path)
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=self._build_messages(image_data, media_type)
                )
                content = response.content[0].text
                return self._parse_response(content)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return {"license_plate_detected": None, "confidence": 0, "reasoning": f"API error: {str(e)[:100]}"}

    async def analyze_async(self, image_path: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                image_data, media_type = self._encode_image(image_path)
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=self._build_messages(image_data, media_type)
                )
                content = response.content[0].text
                return self._parse_response(content)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return {"license_plate_detected": None, "confidence": 0, "reasoning": f"API error: {str(e)[:100]}"}

    @staticmethod
    def _parse_response(content: str) -> Dict[str, Any]:
        import re
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", content.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\n?```\s*$", "", stripped.strip())
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        # Try to extract the first JSON object from the text
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"license_plate_detected": None, "confidence": 0, "reasoning": f"Invalid JSON from LLM: {content[:100]}"}
