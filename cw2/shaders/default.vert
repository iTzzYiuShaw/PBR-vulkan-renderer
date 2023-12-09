#version 450

layout( location = 0 ) in vec3 iPosition;
layout( location = 1 ) in vec2 iTexCoord;
layout( location = 2 ) in vec3 iNormal;
layout( location = 3 ) in vec4 iTangent;
layout( location = 4) in uint packedTBN;

layout( set = 0, binding = 0 ) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;
	vec3 cameraPos;
} uScene;



vec4 unpackQuat(uint packed) {
    // Unpack values
    uint ix = packed & 0xFFu;
    uint iy = (packed >> 8) & 0xFFu;
    uint iz = (packed >> 16) & 0xFFu;
    uint iw = (packed >> 24) & 0xFFu;

    // Convert to float in range -1 to 1
    float x = (float(ix) / 255.0) * 2.0 - 1.0;
    float y = (float(iy) / 255.0) * 2.0 - 1.0;
    float z = (float(iz) / 255.0) * 2.0 - 1.0;
    float w = (float(iw) / 255.0) * 2.0 - 1.0;

    // Return quaternion
    return vec4(x, y, z, w);
}


mat3 quaternionToMat3(vec4 q) {
    float qx2 = q.x * q.x;
    float qy2 = q.y * q.y;
    float qz2 = q.z * q.z;

    mat3 m;
    m[0][0] = 1.0 - 2.0 * (qy2 + qz2);
    m[0][1] = 2.0 * (q.x * q.y + q.z * q.w);
    m[0][2] = 2.0 * (q.x * q.z - q.y * q.w);

    m[1][0] = 2.0 * (q.x * q.y - q.z * q.w);
    m[1][1] = 1.0 - 2.0 * (qx2 + qz2);
    m[1][2] = 2.0 * (q.y * q.z + q.x * q.w);

    m[2][0] = 2.0 * (q.x * q.z + q.y * q.w);
    m[2][1] = 2.0 * (q.y * q.z - q.x * q.w);
    m[2][2] = 1.0 - 2.0 * (qx2 + qy2);

    return m;
}

layout( location = 0 ) out vec2 v2fTexCoord;
layout( location = 1) out vec3 v2fNormal;
layout( location = 2) out vec3 v2fFragCoord;	
layout( location = 3) out vec3 v2fCameraPos;
layout( location = 4) out vec4 v2fTangent;
layout( location = 5) out mat3 v2fUnPackedTBN;

void main()
{

	v2fTexCoord = iTexCoord;

	v2fNormal = iNormal;

	v2fFragCoord = iPosition;

	v2fCameraPos = uScene.cameraPos;

    vec4 quatTBN = unpackQuat(packedTBN);

    v2fUnPackedTBN = quaternionToMat3(quatTBN);

	v2fTangent = iTangent;

	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 
}
