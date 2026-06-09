Shader "RobotYao/Fisheye180Inside"
{
    Properties
    {
        _MainTex ("Fisheye Texture", 2D) = "black" {}
        _FovDeg ("FOV Degrees", Float) = 180
        _FlipY ("Flip Y", Float) = 1
        _SwapRedBlue ("Swap Red Blue", Float) = 0
        _Exposure ("Exposure", Float) = 1
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Cull Front
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float _FovDeg;
            float _FlipY;
            float _SwapRedBlue;
            float _Exposure;

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 localDir : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.localDir = normalize(v.vertex.xyz);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float3 dir = normalize(i.localDir);
                float maxTheta = max(radians(_FovDeg * 0.5), 0.0001);
                float theta = acos(saturate(dir.z));
                float2 plane = dir.xy;
                float planeLen = length(plane);
                float2 radialDir = planeLen > 0.00001 ? plane / planeLen : float2(0.0, 0.0);
                float radius = saturate(theta / maxTheta) * 0.5;
                float2 uv = float2(0.5, 0.5) + radialDir * radius;

                if (_FlipY > 0.5)
                    uv.y = 1.0 - uv.y;

                fixed4 color = tex2D(_MainTex, uv);
                if (_SwapRedBlue > 0.5)
                    color = color.bgra;

                color.rgb *= _Exposure;
                color.a = 1.0;
                return color;
            }
            ENDCG
        }
    }
}
