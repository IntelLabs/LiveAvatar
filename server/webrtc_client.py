import asyncio
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import aiohttp


async def run(pc, player, server_url):
    video_track = player.video
    if video_track:
        sender = pc.addTrack(video_track)

        # Boost bitrate for better quality
        @video_track.on("ready")
        async def boost_bitrate():
            params = sender.getParameters()
            if params.encodings:
                for enc in params.encodings:
                    enc.maxBitrate = 2_000_000  # 2 Mbps
                await sender.setParameters(params)
                print("Bitrate set to 2 Mbps")

    # Create offer and send to server
    await pc.setLocalDescription(await pc.createOffer())

    async with aiohttp.ClientSession() as session:
        async with session.post(server_url, json={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }) as resp:
            data = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=data["sdp"],
        type=data["type"]
    ))
    print("Connected to server and streaming started.")


if __name__ == "__main__":
    # Setup webcam capture (Linux/Mac)
    player = MediaPlayer("/dev/video0", format="v4l2", options={
        "video_size": "1280x720",
        "framerate": "30"
    })

    pc = RTCPeerConnection()
    server_url = "http://localhost:8080/offer"

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(pc, player, server_url))

    print("Press Ctrl+C to stop.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing connection...")
        loop.run_until_complete(pc.close())
        if player:
            loop.run_until_complete(player.stop())
