#version 450

struct LightSource
{
	vec4 position;
	vec4 color;
	float intensity;
};

layout(push_constant) uniform VertexPushConstants {
    int isAlpha;
	int isNormalMap;
} vertexPushConst;

layout( location = 0 ) in vec2 v2fTexCoord;
layout( location = 1) in vec3 v2fNormal;
layout( location = 2) in vec3 v2fFragCoord;
layout( location = 3) in vec3 v2fCameraPos;
layout( location = 4) in vec4 v2fTangent;
layout( location = 5) flat in mat3 v2fUnPackedTBN;

layout( location = 0 ) out vec4 oColor; 

layout( set = 1, binding = 0 ) uniform sampler2D uTexColor;
layout( set = 1, binding = 1 ) uniform sampler2D uRoughtness;
layout( set = 1, binding = 2 ) uniform sampler2D uMetalness;
layout( set = 1, binding = 3 ) uniform sampler2D uAlphaTexture;
layout( set = 1, binding = 4 ) uniform sampler2D uNormalMap;

layout(set = 2, binding = 0) uniform LightData {
    LightSource light;
} lightData;


void main()
{
	
	vec3 lightPos = vec3((lightData.light.position).xyz);
	vec3 cameraPos = v2fCameraPos;
	vec3 fragPos = v2fFragCoord;
	mat3 quatTBN_mat3 = v2fUnPackedTBN;


	vec3 baseColor = vec3(1.0);
	float alpha = 1.0;
	//Texture
	if(vertexPushConst.isAlpha == 1)
	{
		baseColor = texture(uAlphaTexture,v2fTexCoord).rgb;
		alpha = texture(uAlphaTexture,v2fTexCoord).a;

	}else
	{
		baseColor = texture(uTexColor,v2fTexCoord).rgb * 0.8;
	}
	
	float roughness = texture(uRoughtness,v2fTexCoord).r; //Shininess
	float shininess = 2.0 / (pow(roughness,4) + 0.001) - 2;
	float metalness = texture(uMetalness,v2fTexCoord).r;
	

	//Direction settings
	vec3 T = normalize(vec3(v2fTangent.xyz));
	vec3 N = normalize(v2fNormal);
	//vec3 N = quatTBN_mat3[2];

	T = normalize(T - dot(T, N) * N);
	vec3 B = sign(v2fTangent.w) * cross(N,T);

	vec3 V = normalize(cameraPos - fragPos);
    vec3 L = normalize(lightPos - fragPos);
	vec3 H = normalize(L + V);

	
	mat3 TBN = transpose(mat3(T,B,N));
	//mat3 TBN = transpose(quatTBN_mat3);

	if(vertexPushConst.isNormalMap == 1)
	{
		vec3 normal = texture(uNormalMap,v2fTexCoord).rgb;
		N = normalize ((2.0 * normal - 1.0));

		V = normalize(TBN * V);
		L = normalize(TBN * L);
		H = normalize(L + V);
	}

	float pi = 3.1415926;
	float NdotL = max(dot(N,L), 0.0);
	float NdotH = max(dot(N,H),0.0);
	float NdotV = max(dot(N,V),0.0);
	float VdotH = dot(V,H);


	//Specular
	vec3 F0 = (1.0 - metalness) * vec3(0.04,0.04,0.04) + metalness*baseColor;
	vec3 Fv = F0 + (1.0 - F0) * pow( (1.0 - dot(H,V)) ,5);

	//Diffuse
	vec3 pDiffuse = max(baseColor/pi * (vec3(1.0) - Fv) * (1.0 - metalness),0);

	//Distribution function D
	float Dh = ((shininess + 2.0) / (2.0 * pi)) * pow(NdotH,shininess);

	//Cook-Torrance model
	float G1 = 2.0 * ( (NdotH * NdotV) / VdotH);
	float G2 = 2.0 * ( (NdotH * NdotL) / VdotH);
	float G = min(1.0, min(G1,G2));

	//Ambient
	vec3 pAmbient = (lightData.light.color).rgb * baseColor * 0.0001;

	//Specular
	vec3 specular = ( (Dh * Fv * G) / (4.0 * NdotV * NdotL) );
	
	//BRDF
	vec3 BRDF = max((pDiffuse + specular),0);
	BRDF = max(BRDF * lightData.light.color.rgb * NdotL,0);


	vec3 pColor = (pAmbient + BRDF) * alpha;
	oColor = vec4(pColor ,alpha);
}
