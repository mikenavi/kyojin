#!/usr/bin/env python3
"""Explicit GPU/API qualification against an idle Kyojin server; never starts it.

Run sequentially on a parent (--baseline) and candidate. The candidate must have
attachment spill enabled. Use --output for receipts; API credentials, if needed,
come only from KYOJIN_TEST_API_KEY. Requires Pillow and an actual vision model.
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageFont


def image_url(index, code=None):
    if code:
        image = Image.new("RGB", (800, 170), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
        except OSError:
            font = ImageFont.load_default(size=50)
        draw.text((35, 45), "TEST CODE: " + code, fill="black", font=font)
    else:
        image = Image.new("RGB", (64, 64), (index % 256, (index * 37) % 256, (index * 71) % 256))
        ImageDraw.Draw(image).rectangle((12, 12, 52, 52), fill="white")
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    headers = {"Content-Type": "application/json"}
    if os.environ.get("KYOJIN_TEST_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["KYOJIN_TEST_API_KEY"]
    def get(path):
        return json.load(urllib.request.urlopen(urllib.request.Request(args.url + path, headers=headers), timeout=10))
    slots = get("/slots")
    if any(s.get("is_processing") for s in slots):
        raise SystemExit("Server is busy; no test request sent")
    health = get("/health")
    assert health.get("vision"), "Vision must be loaded"
    (args.output / "health.json").write_text(json.dumps(health, indent=2) + "\n")
    pictures = [image_url(0, "7319")] + [image_url(i) for i in range(1, 129)]
    prefix = "Attachment cache qualification. " * 256
    results = []
    def request(name, images, salt, code="7319", cache=True):
        content = [{"type": "text", "text": prefix}]
        content += [{"type": "image_url", "image_url": {"url": image}} for image in images]
        content.append({"type": "text", "text": "Read the four-digit TEST CODE in the FIRST image. Ignore all other images. Reply with only the four digits."})
        body = dict(model=args.model, messages=[dict(role="user", content=content)], max_tokens=32,
                    temperature=0, seed=42, chat_template_kwargs={"enable_thinking": False},
                    cache_salt=salt, cache_prompt=cache)
        (args.output / (name + "-request.json")).write_text(json.dumps(body) + "\n")
        start = time.monotonic()
        req = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        elapsed = time.monotonic() - start
        (args.output / (name + "-response.json")).write_text(json.dumps(result, indent=2) + "\n")
        choice = result["choices"][0]
        answer = choice["message"]["content"].strip()
        usage = result["usage"]
        row = dict(name=name, images=len(images), seconds=round(elapsed, 3), answer=answer,
                   finish_reason=choice["finish_reason"], prompt_tokens=usage["prompt_tokens"],
                   cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                   completion_tokens=usage["completion_tokens"], timings=result.get("timings"))
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        assert answer == code, (name, answer, code)
        assert choice["finish_reason"] == "stop", (name, choice["finish_reason"])
        return row
    salt = "attachment-129-" + uuid.uuid4().hex
    cold = request("129-first", pictures, salt)
    warm = request("129-repeat", pictures, salt)
    if args.baseline:
        assert warm["prompt_tokens"] - warm["cached_tokens"] > 1024, "Parent did not reproduce image eviction"
        return
    assert warm["cached_tokens"] >= warm["prompt_tokens"] - 512, (cold, warm)
    forced = request("129-forced-cold", pictures, salt, cache=False)
    assert forced["cached_tokens"] == 0, forced
    assert (warm["answer"], warm["completion_tokens"]) == (forced["answer"], forced["completion_tokens"])
    # Changed content must never inherit the old image's cached answer.
    request("129-changed-first", [image_url(0, "8642")] + pictures[1:], salt, code="8642")
    # Equal bytes at two positions need distinct, stable occurrence identities.
    duplicate = pictures[:17] + [pictures[0]] + pictures[17:33]
    salt = "attachment-duplicates-" + uuid.uuid4().hex
    request("duplicates-first", duplicate, salt)
    warm = request("duplicates-repeat", duplicate, salt)
    assert warm["cached_tokens"] >= warm["prompt_tokens"] - 512, warm
    request("duplicates-forced-cold", duplicate, salt, cache=False)
    print("PASS: GPU vision, 129 attachments, warm/cold answer parity, content invalidation and repeated occurrences", flush=True)


if __name__ == "__main__":
    main()
