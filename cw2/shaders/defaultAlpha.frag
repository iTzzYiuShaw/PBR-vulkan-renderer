#version 450

struct LightSource
{
	vec4 position;
	vec4 color;
	float intensity;
};


layout( location = 0 ) in vec2 v2fTexCoord;
layout( location = 1) in vec3 v2fNormal;
layout( location = 2) in vec3 v2fFragCoord;
layout( location = 3) in vec3 v2fCameraPos;

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
	
	//Normal test
	//oColor = vec4( v2fNormal, 1.f ); 
	//FragCoord test
	//oColor = vec4( v2fFragCoord, 1.f ); 
	//CameraPos test
	//oColor = vec4( v2fCameraPos, 1.f );

	vec3 lightPos = vec3((lightData.light.position).xyz);


	//Texture
	vec3 baseColor = texture(uTexColor,v2fTexCoord).rgb;
	float roughness = texture(uRoughtness,v2fTexCoord).r; //Shininess
	float shininess = 2.0 / (pow(roughness,4) + 0.001) - 2;

	float metalness = texture(uMetalness,v2fTexCoord).r;

	//Direction settings
	vec3 N = normalize(v2fNormal);
    vec3 V = normalize(v2fCameraPos - v2fFragCoord);
    vec3 L = normalize(lightPos - v2fFragCoord);
	vec3 H = normalize(L + V);
	float pi = 3.1415926;

    vec3 R = reflect(-L, N);
    float NdotL = max(dot(N, L), 0.0);
    float RdotV = max(dot(R, V), 0.0);

	float NdotH = max(dot(N,H),0.0);
	float NdotV = max(dot(N,V),0.0);
	float VdotH = dot(V,H);


	//Specular
	vec3 F0 = (1.0 - metalness) * vec3(0.04,0.04,0.04) + metalness*baseColor;
	vec3 Fv = F0 + (1.0 - F0) * pow( (1.0 - dot(H,V)) ,5);

	//Diffuse
	vec3 pDiffuse = baseColor/pi * (vec3(1.0) - Fv) * (1.0 - metalness);

	//Distribution function D
	float Dh = ((shininess + 2.0) / (2.0 * pi)) * pow(NdotH,shininess);

	//Cook-Torrance model
	float G1 = 2.0 * ( (NdotH * NdotV) / VdotH);
	float G2 = 2.0 * ( (NdotH * NdotL) / VdotH);
	float G = min(1.0, min(G1,G2));

	//Ambient
	vec3 pAmbient = (lightData.light.color).rgb * baseColor * 0.02;

	//Specular
	vec3 specular = ( (Dh * Fv * G) / (4.0 * NdotV * NdotL) );
	
	//BRDF
	vec3 BRDF = max((pDiffuse + specular),0);


	vec3 pColor = pAmbient + BRDF * lightData.light.color.rgb * NdotL;


	// Calculate ambient light
    //vec3 ambient = 0.1 * baseColor; 
	// Calculate diffuse light
    // vec3 diffuse = baseColor * lightData.light.color * (1.0 - metalness) * NdotL;
	// Calculate specular light
	//vec3 specular = vec3(0.04) * (1.0 - metalness) + baseColor * metalness;
    //specular *= lightData.light.color * pow(RdotV, 2.0 / (roughness + 0.0001));
	//vec3 color = ambient + diffuse + specular;

	//light testing
	//oColor = vec4(L,1.0);

	//normal testing
	//oColor = vec4(N,1.0);

	//View testing
	//oColor = vec4(V,1.0);

	//H testing
	//oColor = vec4(roughness,roughness,roughness,1.0);


	//Dh testing
	//oColor = vec4(Dh,Dh,Dh, 1.0);

	
	//G testing
	//oColor = vec4(G,G,G, 1.0);

	//F testing
	//oColor = vec4(Fv.rgb, 1.0);

	//diffuse testing
	//oColor = vec4(pDiffuse,1.0);
	
	//specular testing
	//oColor = vec4(specular,1.0);

	//BRDF testing
	//oColor = vec4(BRDF.rgb,1.0);

	oColor = vec4(pColor.rgb, 1.0);
}
