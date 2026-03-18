"""
Gemini 2.0 Flash Multimodal License Plate Judge
"""
from typing import Dict, Any
import time
import asyncio
import json
import google.generativeai as genai
from PIL import Image

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

class GeminiJudge(BaseLLMJudge):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        genai.configure(api_key=api_key)

    def analyze(self, image_path: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                img = Image.open(image_path)
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    [PROMPT, "Analyze this image for license plates.", img],
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                return self._parse_response(response.text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return {"license_plate_detected": None, "confidence": 0, "reasoning": f"API error: {str(e)[:100]}"}

    async def analyze_async(self, image_path: str) -> Dict[str, Any]:
        """Async wrapper — Gemini SDK lacks native async, so we delegate to a thread."""
        return await asyncio.to_thread(self.analyze, image_path)

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
