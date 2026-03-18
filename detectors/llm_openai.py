"""
OpenAI GPT-4o Multimodal License Plate Judge
"""
from typing import Dict, Any
import openai
import time
import asyncio
import base64
import json

PROMPT = '''\
You are a computer vision inspection expert.

Your task is to analyze the provided image and determine whether a VEHICLE LICENSE PLATE is visible anywhere in the image.

A license plate is defined as a rectangular plate attached to a motor vehicle that displays the vehicle’s registration number issued by a government authority.

Vehicles may include:
- Cars
- Motorcycles
- Trucks
- Vans
- Buses
- Auto-rickshaws
- Scooters
- Commercial vehicles
- Trailers

The license plate may appear:
- On the front or rear of the vehicle
- On motorcycles or scooters (usually smaller plates)
- On parked or moving vehicles
- On partially visible vehicles

Carefully inspect the entire image and follow these rules.

------------------------------------------------

LICENSE PLATE SHOULD BE MARKED AS VISIBLE IF:

1. A rectangular license plate plate object is clearly visible on a vehicle.
2. Any portion of a license plate is visible even if the entire plate is not visible.
3. The plate is visible but:
   - partially occluded
   - angled
   - dirty
   - blurred
   - far away
4. Only part of the plate characters or frame are visible but the object can still be reasonably identified as a license plate.
5. Multiple vehicles exist and at least one vehicle has a visible plate.

------------------------------------------------

LICENSE PLATE SHOULD BE MARKED AS NOT VISIBLE IF:

1. No vehicles are present.
2. Vehicles are present but no plate can be seen.
3. The vehicle is visible only from angles where plates normally cannot be seen.
4. The plate area is completely blocked or outside the frame.
5. The image resolution is too low to detect the plate object.
6. Only reflections or vague shapes exist that cannot be confidently identified as a license plate.

------------------------------------------------

SPECIAL CASES

PARTIAL PLATES:
If only a small part of the plate is visible but the plate structure is recognizable, mark as visible.

BLURRED PLATES:
If the plate object is visible but characters cannot be read due to blur, mark as visible.

COVERED / OBSCURED:
If a plate exists but is completely covered with cloth, mud, tape, or objects such that the plate itself cannot be identified, mark as not visible.

REFLECTIONS:
If a license plate appears in mirrors or reflections and is clearly recognizable, mark as visible.

POSTERS OR SIGNS:
Ignore license plates printed on posters, advertisements, or images of vehicles.

TOY VEHICLES:
Ignore plates on toy cars or scale models.

------------------------------------------------

OUTPUT FORMAT

Return your answer strictly in the following JSON format.

{
  "license_plate_visible": true or false,
  "confidence": 0.0-1.0,
  "detected_vehicle_count": integer,
  "vehicles_with_visible_plate": integer,
  "reasoning": "Brief explanation describing how the decision was made"
}

------------------------------------------------

EXAMPLES

Example 1:
Image: Car parked facing camera with visible front plate

{
  "license_plate_visible": true,
  "confidence": 0.98,
  "detected_vehicle_count": 1,
  "vehicles_with_visible_plate": 1,
  "reasoning": "A car is visible and the front rectangular license plate is clearly visible."
}

Example 2:
Image: Side view of car with no plate visible

{
  "license_plate_visible": false,
  "confidence": 0.92,
  "detected_vehicle_count": 1,
  "vehicles_with_visible_plate": 0,
  "reasoning": "Vehicle visible but license plate area is not visible from this viewing angle."
}

Example 3:
Image: Motorcycle rear with partially blurred plate

{
  "license_plate_visible": true,
  "confidence": 0.85,
  "detected_vehicle_count": 1,
  "vehicles_with_visible_plate": 1,
  "reasoning": "Motorcycle rear is visible and license plate object is present though blurred."
}
'''

class BaseLLMJudge:
    def analyze(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def analyze_async(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIGPT4oJudge(BaseLLMJudge):
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        openai.api_key = api_key
        self.async_client = openai.AsyncOpenAI(api_key=api_key)

    def _build_messages(self, image_data: str):
        return [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this image for license plates."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]}
        ]

    def _build_messages_from_url(self, url: str):
        return [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this image for license plates."},
                {"type": "image_url", "image_url": {"url": url}}
            ]}
        ]

    def analyze_from_url(self, url: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages_from_url(url),
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = response.choices[0].message.content
                return self._parse_response(content)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return {"license_plate_detected": None, "confidence": 0, "reasoning": f"API error: {str(e)[:100]}"}

    async def analyze_from_url_async(self, url: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages_from_url(url),
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = response.choices[0].message.content
                return self._parse_response(content)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return {"license_plate_detected": None, "confidence": 0, "reasoning": f"API error: {str(e)[:100]}"}

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.standard_b64encode(img_file.read()).decode("utf-8")

    def analyze(self, image_path: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                image_data = self._encode_image(image_path)
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages(image_data),
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = response.choices[0].message.content
                return self._parse_response(content)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return {"license_plate_detected": None, "confidence": 0, "reasoning": f"API error: {str(e)[:100]}"}

    async def analyze_async(self, image_path: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                image_data = self._encode_image(image_path)
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages(image_data),
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = response.choices[0].message.content
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
