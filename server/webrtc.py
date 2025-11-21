import asyncio
import numpy as np
from aiohttp import web
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
import av
import time
import json
import cv2

pcs = set()


class ImageVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.counter = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = av.VideoFrame.from_ndarray(self.image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        await asyncio.sleep(1 / 30)  # 30 FPS
        return frame


class ReceiverTrack(VideoStreamTrack):
    def __init__(self, track, frame_queue):
        super().__init__()
        self.track = track
        self.frame_queue = frame_queue

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        self.frame_queue.put((img, time.time()))
        cv2.imshow("WebRTC Raw", img)
        cv2.waitKey(1)
        return frame


class SenderTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, output_queue):
        super().__init__()
        self.output_queue = output_queue

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        try:
            image: np.ndarray = self.output_queue.get()[0]
        except Exception as e:
            print(e)

        # image = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        await asyncio.sleep(1 / 30)  # 30 FPS
        return frame


async def offer_debug(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    pc.addTrack(ImageVideoTrack())
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def offer(request):

    print("Add SenderTrack foooooooooooooooooo")
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()

    print("Add SenderTrack")
    # frame_queue = request.app["frame_queue"]
    output_queue = request.app["output_queue"]
    # pc.addTrack(SenderTrack(output_queue))

    @pc.on("track")
    def on_track(track):
        print("WebRTC track received:", track.kind)
        if track.kind == "video":
            # pc.addTrack(ReceiverTrack(track, frame_queue))
            pc.addTrack(SenderTrack(output_queue))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )


async def on_shutdown(app):
    # Cleanup
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def index(request):
    return web.Response(content_type="text/html", text=open("scripts/static/index.html").read())


def run_webrtc_server(frame_queue, output_queue):
    print("Starting WebRTC server...")
    app = web.Application()
    app["output_queue"] = output_queue
    app["frame_queue"] = frame_queue
    app.on_shutdown.append(on_shutdown)
    app.add_routes([
        web.get("/", index),
        web.post("/offer", offer),
    ])
    web.run_app(app, port=8080)




