#include "pch.h"
#include "framework.h"
#include "RobotoYaoS.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <d3d11.h>
#include <turbojpeg.h>
#include <wrl/client.h>
#include <zmq.h>

namespace
{
using Clock = std::chrono::steady_clock;

struct DecodedStereoFrame
{
    int width = 0;
    int height = 0;
    uint64_t frameId = 0;
    std::vector<uint8_t> leftBgra;
    std::vector<uint8_t> rightBgra;
};

struct ReceiverState
{
    std::atomic<bool> running{ false };
    std::thread worker;

    std::mutex frameMutex;
    std::shared_ptr<DecodedStereoFrame> latestFrame;

    std::mutex textureMutex;
    void* leftTexture = nullptr;
    void* rightTexture = nullptr;
    int textureWidth = 0;
    int textureHeight = 0;
    uint64_t lastUploadedFrameId = 0;

    std::atomic<uint64_t> receivedFrames{ 0 };
    std::atomic<uint64_t> decodedFrames{ 0 };
    std::atomic<uint64_t> uploadedFrames{ 0 };
    std::atomic<uint64_t> failedFrames{ 0 };
    std::atomic<uint64_t> lastFrameId{ 0 };
    std::atomic<int> streamWidth{ 0 };
    std::atomic<int> streamHeight{ 0 };

    std::mutex statsMutex;
    Clock::time_point startTime = Clock::now();
    double lastDecodeMs = 0.0;
    double lastUploadMs = 0.0;
    char lastError[256] = {};
};

ReceiverState g_state;

void SetError(const std::string& message)
{
    std::lock_guard<std::mutex> lock(g_state.statsMutex);
    const size_t count = (std::min)(message.size(), sizeof(g_state.lastError) - 1);
    std::memcpy(g_state.lastError, message.data(), count);
    g_state.lastError[count] = '\0';
}

void ClearError()
{
    std::lock_guard<std::mutex> lock(g_state.statsMutex);
    g_state.lastError[0] = '\0';
}

uint64_t ExtractFrameId(const std::vector<uint8_t>& header)
{
    const std::string text(header.begin(), header.end());
    const std::string key = "\"frame_id\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos)
        return 0;

    pos = text.find(':', pos + key.size());
    if (pos == std::string::npos)
        return 0;

    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
        ++pos;

    uint64_t value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
    {
        value = value * 10 + static_cast<uint64_t>(text[pos] - '0');
        ++pos;
    }
    return value;
}

std::string ExtractEncoding(const std::vector<uint8_t>& header)
{
    const std::string text(header.begin(), header.end());
    const std::string key = "\"encoding\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos)
        return "jpg";

    pos = text.find(':', pos + key.size());
    if (pos == std::string::npos)
        return "jpg";

    pos = text.find('"', pos);
    if (pos == std::string::npos)
        return "jpg";

    const size_t begin = pos + 1;
    const size_t end = text.find('"', begin);
    if (end == std::string::npos || end <= begin)
        return "jpg";

    return text.substr(begin, end - begin);
}

bool ReceiveMultipart(void* socket, std::vector<std::vector<uint8_t>>& parts)
{
    parts.clear();

    while (true)
    {
        zmq_msg_t message;
        zmq_msg_init(&message);
        const int bytes = zmq_msg_recv(&message, socket, 0);
        if (bytes < 0)
        {
            zmq_msg_close(&message);
            return false;
        }

        const size_t size = zmq_msg_size(&message);
        const uint8_t* data = static_cast<const uint8_t*>(zmq_msg_data(&message));
        parts.emplace_back(data, data + size);

        int more = 0;
        size_t moreSize = sizeof(more);
        zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &moreSize);
        zmq_msg_close(&message);

        if (!more)
            return true;
    }
}

bool DecodeJpegToBgra(
    const std::vector<uint8_t>& jpeg,
    int expectedWidth,
    int expectedHeight,
    std::vector<uint8_t>& bgra,
    std::string& error)
{
    if (jpeg.empty())
    {
        error = "Empty JPEG payload.";
        return false;
    }

    tjhandle handle = tjInitDecompress();
    if (handle == nullptr)
    {
        error = "tjInitDecompress failed.";
        return false;
    }

    int width = 0;
    int height = 0;
    int subsamp = 0;
    int colorspace = 0;
    int rc = tjDecompressHeader3(
        handle,
        jpeg.data(),
        static_cast<unsigned long>(jpeg.size()),
        &width,
        &height,
        &subsamp,
        &colorspace);
    if (rc != 0)
    {
        error = tjGetErrorStr2(handle);
        tjDestroy(handle);
        return false;
    }

    if (expectedWidth > 0 && expectedHeight > 0 && (width != expectedWidth || height != expectedHeight))
    {
        error = "JPEG dimensions do not match the configured stream dimensions.";
        tjDestroy(handle);
        return false;
    }

    bgra.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    rc = tjDecompress2(
        handle,
        jpeg.data(),
        static_cast<unsigned long>(jpeg.size()),
        bgra.data(),
        width,
        width * 4,
        height,
        TJPF_BGRA,
        TJFLAG_FASTDCT);

    if (rc != 0)
    {
        error = tjGetErrorStr2(handle);
        tjDestroy(handle);
        return false;
    }

    tjDestroy(handle);
    return true;
}

bool ConvertRgb8ToBgra(
    const std::vector<uint8_t>& rgb,
    int width,
    int height,
    std::vector<uint8_t>& bgra,
    std::string& error)
{
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t expectedBytes = pixelCount * 3u;
    if (rgb.size() != expectedBytes)
    {
        error = "RAW RGB8 payload size does not match configured stream dimensions.";
        return false;
    }

    bgra.resize(pixelCount * 4u);
    for (size_t i = 0, j = 0; i < pixelCount; ++i, j += 3)
    {
        const size_t k = i * 4u;
        bgra[k + 0] = rgb[j + 2];
        bgra[k + 1] = rgb[j + 1];
        bgra[k + 2] = rgb[j + 0];
        bgra[k + 3] = 255;
    }
    return true;
}

void ReceiverLoop(std::string endpoint, std::string topic, int width, int height)
{
    void* context = zmq_ctx_new();
    if (context == nullptr)
    {
        SetError("zmq_ctx_new failed.");
        g_state.running.store(false);
        return;
    }

    void* socket = zmq_socket(context, ZMQ_SUB);
    if (socket == nullptr)
    {
        SetError("zmq_socket(ZMQ_SUB) failed.");
        zmq_ctx_term(context);
        g_state.running.store(false);
        return;
    }

    const int lingerMs = 0;
    const int timeoutMs = 100;
    const int highWaterMark = 2;
    zmq_setsockopt(socket, ZMQ_LINGER, &lingerMs, sizeof(lingerMs));
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeoutMs, sizeof(timeoutMs));
    zmq_setsockopt(socket, ZMQ_RCVHWM, &highWaterMark, sizeof(highWaterMark));
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, topic.data(), static_cast<int>(topic.size()));

    if (zmq_connect(socket, endpoint.c_str()) != 0)
    {
        SetError(std::string("zmq_connect failed: ") + zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_term(context);
        g_state.running.store(false);
        return;
    }

    std::vector<std::vector<uint8_t>> parts;
    while (g_state.running.load())
    {
        if (!ReceiveMultipart(socket, parts))
        {
            const int err = zmq_errno();
            if (err == EAGAIN || err == ETERM)
                continue;
            SetError(std::string("ZMQ receive failed: ") + zmq_strerror(err));
            g_state.failedFrames.fetch_add(1);
            continue;
        }

        g_state.receivedFrames.fetch_add(1);
        if (parts.size() < 4)
        {
            SetError("Invalid stereo packet: expected [topic, header, left, right].");
            g_state.failedFrames.fetch_add(1);
            continue;
        }

        const uint64_t frameId = ExtractFrameId(parts[1]);
        const std::string encoding = ExtractEncoding(parts[1]);
        const auto decodeStart = Clock::now();

        auto frame = std::make_shared<DecodedStereoFrame>();
        frame->width = width;
        frame->height = height;
        frame->frameId = frameId;

        std::string error;
        const bool isRaw = encoding == "raw";
        const bool leftOk = isRaw
            ? ConvertRgb8ToBgra(parts[2], width, height, frame->leftBgra, error)
            : DecodeJpegToBgra(parts[2], width, height, frame->leftBgra, error);
        if (!leftOk)
        {
            SetError(std::string("Left image decode failed: ") + error);
            g_state.failedFrames.fetch_add(1);
            continue;
        }

        const bool rightOk = isRaw
            ? ConvertRgb8ToBgra(parts[3], width, height, frame->rightBgra, error)
            : DecodeJpegToBgra(parts[3], width, height, frame->rightBgra, error);
        if (!rightOk)
        {
            SetError(std::string("Right image decode failed: ") + error);
            g_state.failedFrames.fetch_add(1);
            continue;
        }

        const auto decodeEnd = Clock::now();
        {
            std::lock_guard<std::mutex> lock(g_state.statsMutex);
            g_state.lastDecodeMs = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
        }

        {
            std::lock_guard<std::mutex> lock(g_state.frameMutex);
            g_state.latestFrame = frame;
        }

        g_state.lastFrameId.store(frameId);
        g_state.decodedFrames.fetch_add(1);
        ClearError();
    }

    zmq_close(socket);
    zmq_ctx_term(context);
}

void __stdcall OnRenderEvent(int eventId)
{
    (void)eventId;

    std::shared_ptr<DecodedStereoFrame> frame;
    {
        std::lock_guard<std::mutex> lock(g_state.frameMutex);
        frame = g_state.latestFrame;
    }

    if (!frame || frame->frameId == g_state.lastUploadedFrameId)
        return;

    void* leftPtr = nullptr;
    void* rightPtr = nullptr;
    int textureWidth = 0;
    int textureHeight = 0;
    {
        std::lock_guard<std::mutex> lock(g_state.textureMutex);
        leftPtr = g_state.leftTexture;
        rightPtr = g_state.rightTexture;
        textureWidth = g_state.textureWidth;
        textureHeight = g_state.textureHeight;
    }

    if (leftPtr == nullptr || rightPtr == nullptr)
        return;
    if (textureWidth != frame->width || textureHeight != frame->height)
    {
        SetError("Unity texture dimensions do not match decoded frame dimensions.");
        return;
    }

    auto* leftTexture = static_cast<ID3D11Texture2D*>(leftPtr);
    auto* rightTexture = static_cast<ID3D11Texture2D*>(rightPtr);
    if (leftTexture == nullptr || rightTexture == nullptr)
        return;

    const auto uploadStart = Clock::now();
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    leftTexture->GetDevice(device.GetAddressOf());
    if (!device)
    {
        SetError("Could not get D3D11 device from Unity texture.");
        return;
    }

    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    device->GetImmediateContext(context.GetAddressOf());
    if (!context)
    {
        SetError("Could not get D3D11 immediate context.");
        return;
    }

    context->UpdateSubresource(leftTexture, 0, nullptr, frame->leftBgra.data(), frame->width * 4, 0);
    context->UpdateSubresource(rightTexture, 0, nullptr, frame->rightBgra.data(), frame->width * 4, 0);

    const auto uploadEnd = Clock::now();
    {
        std::lock_guard<std::mutex> lock(g_state.statsMutex);
        g_state.lastUploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadStart).count();
    }

    g_state.lastUploadedFrameId = frame->frameId;
    g_state.uploadedFrames.fetch_add(1);
}
}

extern "C"
{
ROBOTOYAOS_API int ROBOTYAO_CALL RY_StartReceiver(const char* endpoint, const char* topic, int width, int height)
{
    if (endpoint == nullptr || topic == nullptr || width <= 0 || height <= 0)
    {
        SetError("Invalid RY_StartReceiver arguments.");
        return 0;
    }

    if (g_state.running.load())
        return 1;

    g_state.streamWidth.store(width);
    g_state.streamHeight.store(height);
    g_state.receivedFrames.store(0);
    g_state.decodedFrames.store(0);
    g_state.uploadedFrames.store(0);
    g_state.failedFrames.store(0);
    g_state.lastFrameId.store(0);
    g_state.lastUploadedFrameId = 0;
    g_state.startTime = Clock::now();
    ClearError();

    {
        std::lock_guard<std::mutex> lock(g_state.frameMutex);
        g_state.latestFrame.reset();
    }

    g_state.running.store(true);
    try
    {
        g_state.worker = std::thread(ReceiverLoop, std::string(endpoint), std::string(topic), width, height);
    }
    catch (const std::exception& ex)
    {
        g_state.running.store(false);
        SetError(std::string("Failed to start receiver thread: ") + ex.what());
        return 0;
    }

    return 1;
}

ROBOTOYAOS_API void ROBOTYAO_CALL RY_StopReceiver()
{
    g_state.running.store(false);
    if (g_state.worker.joinable())
        g_state.worker.join();

    std::lock_guard<std::mutex> lock(g_state.frameMutex);
    g_state.latestFrame.reset();
}

ROBOTOYAOS_API void ROBOTYAO_CALL RY_SetStereoTextures(void* leftTexture, void* rightTexture, int width, int height)
{
    std::lock_guard<std::mutex> lock(g_state.textureMutex);
    g_state.leftTexture = leftTexture;
    g_state.rightTexture = rightTexture;
    g_state.textureWidth = width;
    g_state.textureHeight = height;
    g_state.lastUploadedFrameId = 0;
}

ROBOTOYAOS_API void ROBOTYAO_CALL RY_GetStats(RobotYaoReceiverStats* outStats)
{
    if (outStats == nullptr)
        return;

    std::memset(outStats, 0, sizeof(RobotYaoReceiverStats));
    outStats->isRunning = g_state.running.load() ? 1 : 0;
    outStats->width = g_state.streamWidth.load();
    outStats->height = g_state.streamHeight.load();
    outStats->receivedFrames = g_state.receivedFrames.load();
    outStats->decodedFrames = g_state.decodedFrames.load();
    outStats->uploadedFrames = g_state.uploadedFrames.load();
    outStats->failedFrames = g_state.failedFrames.load();
    outStats->lastFrameId = g_state.lastFrameId.load();

    {
        std::lock_guard<std::mutex> lock(g_state.frameMutex);
        outStats->hasFrame = g_state.latestFrame ? 1 : 0;
    }

    const double elapsed = std::chrono::duration<double>(Clock::now() - g_state.startTime).count();
    outStats->receiveFps = elapsed > 0.0 ? static_cast<double>(outStats->decodedFrames) / elapsed : 0.0;

    {
        std::lock_guard<std::mutex> lock(g_state.statsMutex);
        outStats->lastDecodeMs = g_state.lastDecodeMs;
        outStats->lastUploadMs = g_state.lastUploadMs;
        std::memcpy(outStats->lastError, g_state.lastError, sizeof(outStats->lastError));
    }
}

ROBOTOYAOS_API RobotYaoUnityRenderingEvent ROBOTYAO_CALL RY_GetRenderEventFunc()
{
    return OnRenderEvent;
}
}
