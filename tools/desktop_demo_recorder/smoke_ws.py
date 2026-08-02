#!/usr/bin/env python3
"""Non-destructive WebSocket smoke test for the teleop control path."""

import asyncio
import json

import aiohttp


async def receive_until(ws, predicate, limit=80):
    for _ in range(limit):
        message = await asyncio.wait_for(ws.receive(), timeout=3)
        if message.type != aiohttp.WSMsgType.TEXT:
            raise AssertionError(f"Unexpected WebSocket message: {message.type}")
        payload = json.loads(message.data)
        if predicate(payload):
            return payload
    raise AssertionError("Expected WebSocket state was not observed")


async def main():
    async with aiohttp.ClientSession() as session:
      async with session.ws_connect("ws://127.0.0.1:8765/ws", max_msg_size=4 * 1024 * 1024) as ws:
        first = await receive_until(ws, lambda p: p.get("type") == "snapshot")
        assert set(first["images"]) == {
            "head",
            "external",
            "left_wrist",
            "right_wrist",
        }
        assert len(first["controls"]["arms"]) == 10

        await ws.send_json({"command": "record", "seed": 424242})
        recording = await receive_until(
            ws,
            lambda p: p.get("type") == "snapshot"
            and p["state"]["recording"]
            and p["state"]["episode_step"] >= 2,
        )

        controls = recording["controls"]
        await ws.send_json(
                {
                    "command": "control",
                    "keys": ["KeyW", "KeyT", "Shift+KeyT"],
                    "arms": controls["arms"],
                "grippers": controls["grippers"],
            }
        )
        moved = await receive_until(
            ws,
            lambda p: p.get("type") == "snapshot"
            and p["state"]["episode_step"] >= recording["state"]["episode_step"] + 5,
        )
        left_arm_delta = moved["controls"]["arms"][0] - controls["arms"][0]
        right_arm_delta = moved["controls"]["arms"][5] - controls["arms"][5]
        assert left_arm_delta >= 0.04
        assert right_arm_delta >= 0.04
        await ws.send_json(
            {
                "command": "control",
                "keys": [],
                "grippers": [1, 1],
            }
        )
        closed = await receive_until(
            ws,
            lambda p: p.get("type") == "snapshot"
            and p["controls"]["grippers"] == [1.0, 1.0],
        )
        await ws.send_json(
            {
                "command": "control",
                "keys": [],
                "grippers": [0, 0],
            }
        )
        opened = await receive_until(
            ws,
            lambda p: p.get("type") == "snapshot"
            and p["controls"]["grippers"] == [0.0, 0.0],
        )
        await ws.send_json({"command": "control", "keys": []})
        await ws.send_json({"command": "pause"})
        paused = await receive_until(
            ws, lambda p: p.get("type") == "snapshot" and p["state"]["paused"]
        )

        await ws.send_json({"command": "save"})
        rejected = await receive_until(ws, lambda p: p.get("type") == "result")
        assert rejected["ok"] is False and "not succeeded" in rejected["message"]

        await ws.send_json({"command": "discard"})
        discarded = await receive_until(
            ws,
            lambda p: p.get("type") == "snapshot"
            and not p["state"]["recording"]
            and p["state"]["paused"],
        )
        print(
            json.dumps(
                {
                    "initial_images": {k: len(v) for k, v in first["images"].items()},
                    "recording_step": recording["state"]["episode_step"],
                    "moved_step": moved["state"]["episode_step"],
                    "left_arm_delta": round(left_arm_delta, 4),
                    "right_arm_delta": round(right_arm_delta, 4),
                    "grippers_closed": closed["controls"]["grippers"],
                    "grippers_opened": opened["controls"]["grippers"],
                    "paused_step": paused["state"]["episode_step"],
                    "save_rejected": rejected["message"],
                    "final_message": discarded["state"]["message"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
