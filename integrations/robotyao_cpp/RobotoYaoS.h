#pragma once

#ifdef ROBOTOYAOS_EXPORTS
#define ROBOTOYAOS_API __declspec(dllexport)
#else
#define ROBOTOYAOS_API __declspec(dllimport)
#endif

#include <stdint.h>

#define ROBOTYAO_CALL __cdecl

extern "C" {

struct RobotYaoReceiverStats
{
    int isRunning;
    int hasFrame;
    int width;
    int height;
    uint64_t receivedFrames;
    uint64_t decodedFrames;
    uint64_t uploadedFrames;
    uint64_t failedFrames;
    uint64_t lastFrameId;
    double receiveFps;
    double lastDecodeMs;
    double lastUploadMs;
    char lastError[256];
};

typedef void(__stdcall* RobotYaoUnityRenderingEvent)(int eventId);

ROBOTOYAOS_API int ROBOTYAO_CALL RY_StartReceiver(const char* endpoint, const char* topic, int width, int height);
ROBOTOYAOS_API void ROBOTYAO_CALL RY_StopReceiver();
ROBOTOYAOS_API void ROBOTYAO_CALL RY_SetStereoTextures(void* leftTexture, void* rightTexture, int width, int height);
ROBOTOYAOS_API void ROBOTYAO_CALL RY_GetStats(RobotYaoReceiverStats* outStats);
ROBOTOYAOS_API RobotYaoUnityRenderingEvent ROBOTYAO_CALL RY_GetRenderEventFunc();

}
