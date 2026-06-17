using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace RobotYao
{
    public sealed class RobotYaoStereoFisheyeApp : MonoBehaviour
    {
        private const string PluginName = "RobotoYaoS";

        [Header("Stream")]
        [SerializeField] private string endpoint = "tcp://127.0.0.1:5556";
        [SerializeField] private string topic = "robotyao.stereo.fisheye.v1";
        [SerializeField] private int width = 1280;
        [SerializeField] private int height = 1280;

        [Header("Fisheye")]
        [SerializeField] private float fisheyeFovDegrees = 180.0f;
        [SerializeField] private bool flipY = false;
        [SerializeField] private bool swapRedBlue = false;

        [Header("Preview")]
        [SerializeField] private float domeRadius = 5.0f;
        [SerializeField] private int longitudeSegments = 96;
        [SerializeField] private int latitudeSegments = 32;
        [SerializeField] private float lookSpeed = 0.18f;
        [SerializeField] private bool showStats = true;

        private RenderTexture leftTexture;
        private RenderTexture rightTexture;
        private IntPtr renderEventFunc;
        private bool nativeStarted;
        private Material leftMaterial;
        private Material rightMaterial;
        private Transform leftCameraTransform;
        private Transform rightCameraTransform;
        private float yaw;
        private float pitch;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void CreateRuntimeInstance()
        {
            if (FindAnyObjectByType<RobotYaoStereoFisheyeApp>() != null)
                return;

            var go = new GameObject("RobotYao Stereo Fisheye App");
            DontDestroyOnLoad(go);
            go.AddComponent<RobotYaoStereoFisheyeApp>();
        }

        private void Awake()
        {
            Application.runInBackground = true;
            CreateTextures();
            CreatePreviewRig();
            StartNativeReceiver();
        }

        private void Update()
        {
            UpdateLook();

            if (nativeStarted && renderEventFunc != IntPtr.Zero)
                GL.IssuePluginEvent(renderEventFunc, 0);
        }

        private void OnDestroy()
        {
            if (nativeStarted)
            {
                RY_StopReceiver();
                nativeStarted = false;
            }

            if (leftTexture != null)
                leftTexture.Release();
            if (rightTexture != null)
                rightTexture.Release();
        }

        private void OnGUI()
        {
            if (!showStats)
                return;

            RobotYaoReceiverStats stats = default;
            if (nativeStarted)
                RY_GetStats(ref stats);

            const int widthPx = 520;
            GUILayout.BeginArea(new Rect(12, 12, widthPx, 160), GUI.skin.box);
            GUILayout.Label("RobotYao Stereo Fisheye");
            GUILayout.Label($"Endpoint: {endpoint}");
            GUILayout.Label($"Running: {stats.isRunning != 0}, Frame: {stats.lastFrameId}, Decoded: {stats.decodedFrames}, Uploaded: {stats.uploadedFrames}, Failed: {stats.failedFrames}");
            GUILayout.Label($"FPS: {stats.receiveFps:F1}, Decode: {stats.lastDecodeMs:F2} ms, Upload: {stats.lastUploadMs:F2} ms");
            if (!string.IsNullOrEmpty(stats.lastError))
                GUILayout.Label($"Error: {stats.lastError}");
            GUILayout.EndArea();
        }

        private void CreateTextures()
        {
            leftTexture = CreateEyeTexture("RobotYao Left Eye RT");
            rightTexture = CreateEyeTexture("RobotYao Right Eye RT");
        }

        private RenderTexture CreateEyeTexture(string textureName)
        {
            var texture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                name = textureName,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
                useMipMap = false,
                autoGenerateMips = false,
                antiAliasing = 1
            };
            texture.Create();
            return texture;
        }

        private void StartNativeReceiver()
        {
            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D11)
            {
                Debug.LogError(
                    "RobotYao native texture upload currently requires Direct3D11. " +
                    $"Current graphics API is {SystemInfo.graphicsDeviceType}. " +
                    "Set Project Settings > Player > Other Settings > Graphics APIs for Windows to Direct3D11."
                );
                return;
            }

            try
            {
                RY_SetStereoTextures(leftTexture.GetNativeTexturePtr(), rightTexture.GetNativeTexturePtr(), width, height);
                int ok = RY_StartReceiver(endpoint, topic, width, height);
                renderEventFunc = RY_GetRenderEventFunc();
                nativeStarted = ok != 0 && renderEventFunc != IntPtr.Zero;
                if (!nativeStarted)
                    Debug.LogError("RobotYao native receiver failed to start.");
            }
            catch (DllNotFoundException ex)
            {
                Debug.LogError($"RobotYao native plugin was not found: {ex.Message}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"RobotYao native receiver start failed: {ex}");
            }
        }

        private void CreatePreviewRig()
        {
            Shader shader = Shader.Find("RobotYao/Fisheye180Inside");
            if (shader == null)
            {
                Debug.LogError("Missing shader RobotYao/Fisheye180Inside.");
                return;
            }

            leftMaterial = CreateEyeMaterial(shader, leftTexture);
            rightMaterial = CreateEyeMaterial(shader, rightTexture);
            Mesh domeMesh = CreateHemisphereMesh(domeRadius, longitudeSegments, latitudeSegments);

            Transform leftRig = CreateEyeRig("Left Preview Rig", new Vector3(-20.0f, 0.0f, 0.0f), leftMaterial, domeMesh, new Rect(0.0f, 0.0f, 0.5f, 1.0f));
            Transform rightRig = CreateEyeRig("Right Preview Rig", new Vector3(20.0f, 0.0f, 0.0f), rightMaterial, domeMesh, new Rect(0.5f, 0.0f, 0.5f, 1.0f));

            leftCameraTransform = leftRig.Find("Camera");
            rightCameraTransform = rightRig.Find("Camera");
        }

        private Material CreateEyeMaterial(Shader shader, Texture texture)
        {
            var material = new Material(shader)
            {
                name = texture.name + " Material"
            };
            material.SetTexture("_MainTex", texture);
            material.SetFloat("_FovDeg", fisheyeFovDegrees);
            material.SetFloat("_FlipY", flipY ? 1.0f : 0.0f);
            material.SetFloat("_SwapRedBlue", swapRedBlue ? 1.0f : 0.0f);
            material.SetFloat("_Exposure", 1.0f);
            return material;
        }

        private Transform CreateEyeRig(string rigName, Vector3 position, Material material, Mesh domeMesh, Rect cameraRect)
        {
            var rig = new GameObject(rigName);
            rig.transform.SetParent(transform, false);
            rig.transform.position = position;

            var dome = new GameObject("Fisheye180 Dome");
            dome.transform.SetParent(rig.transform, false);
            var meshFilter = dome.AddComponent<MeshFilter>();
            meshFilter.sharedMesh = domeMesh;
            var meshRenderer = dome.AddComponent<MeshRenderer>();
            meshRenderer.sharedMaterial = material;

            var cameraObject = new GameObject("Camera");
            cameraObject.transform.SetParent(rig.transform, false);
            var camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;
            camera.nearClipPlane = 0.01f;
            camera.farClipPlane = domeRadius + 1.0f;
            camera.fieldOfView = 90.0f;
            camera.rect = cameraRect;

            return rig.transform;
        }

        private static Mesh CreateHemisphereMesh(float radius, int lonSegments, int latSegments)
        {
            lonSegments = Mathf.Max(8, lonSegments);
            latSegments = Mathf.Max(4, latSegments);

            int vertexCount = (latSegments + 1) * (lonSegments + 1);
            var vertices = new Vector3[vertexCount];
            var normals = new Vector3[vertexCount];
            var uvs = new Vector2[vertexCount];

            int index = 0;
            for (int lat = 0; lat <= latSegments; lat++)
            {
                float theta = (Mathf.PI * 0.5f) * lat / latSegments;
                float sinTheta = Mathf.Sin(theta);
                float cosTheta = Mathf.Cos(theta);

                for (int lon = 0; lon <= lonSegments; lon++)
                {
                    float phi = Mathf.PI * 2.0f * lon / lonSegments;
                    var direction = new Vector3(Mathf.Cos(phi) * sinTheta, Mathf.Sin(phi) * sinTheta, cosTheta);
                    vertices[index] = direction * radius;
                    normals[index] = direction;
                    uvs[index] = new Vector2((float)lon / lonSegments, (float)lat / latSegments);
                    index++;
                }
            }

            var triangles = new int[latSegments * lonSegments * 6];
            int tri = 0;
            for (int lat = 0; lat < latSegments; lat++)
            {
                for (int lon = 0; lon < lonSegments; lon++)
                {
                    int a = lat * (lonSegments + 1) + lon;
                    int b = a + lonSegments + 1;
                    int c = b + 1;
                    int d = a + 1;

                    triangles[tri++] = a;
                    triangles[tri++] = c;
                    triangles[tri++] = b;
                    triangles[tri++] = a;
                    triangles[tri++] = d;
                    triangles[tri++] = c;
                }
            }

            var mesh = new Mesh
            {
                name = "RobotYao 180 Hemisphere",
                vertices = vertices,
                normals = normals,
                uv = uvs,
                triangles = triangles
            };
            mesh.RecalculateBounds();
            return mesh;
        }

        private void UpdateLook()
        {
#if ENABLE_INPUT_SYSTEM
            Mouse mouse = Mouse.current;
            if (mouse != null && mouse.leftButton.isPressed)
            {
                Vector2 delta = mouse.delta.ReadValue();
                yaw += delta.x * lookSpeed;
                pitch -= delta.y * lookSpeed;
                pitch = Mathf.Clamp(pitch, -80.0f, 80.0f);
            }
#else
            if (Input.GetMouseButton(0))
            {
                yaw += Input.GetAxis("Mouse X") * lookSpeed * 100.0f;
                pitch -= Input.GetAxis("Mouse Y") * lookSpeed * 100.0f;
                pitch = Mathf.Clamp(pitch, -80.0f, 80.0f);
            }
#endif

            Quaternion rotation = Quaternion.Euler(pitch, yaw, 0.0f);
            if (leftCameraTransform != null)
                leftCameraTransform.localRotation = rotation;
            if (rightCameraTransform != null)
                rightCameraTransform.localRotation = rotation;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        private struct RobotYaoReceiverStats
        {
            public int isRunning;
            public int hasFrame;
            public int width;
            public int height;
            public ulong receivedFrames;
            public ulong decodedFrames;
            public ulong uploadedFrames;
            public ulong failedFrames;
            public ulong lastFrameId;
            public double receiveFps;
            public double lastDecodeMs;
            public double lastUploadMs;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string lastError;
        }

        [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int RY_StartReceiver(string endpoint, string topic, int width, int height);

        [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void RY_StopReceiver();

        [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void RY_SetStereoTextures(IntPtr leftTexture, IntPtr rightTexture, int width, int height);

        [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void RY_GetStats(ref RobotYaoReceiverStats stats);

        [DllImport(PluginName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr RY_GetRenderEventFunc();
    }
}
