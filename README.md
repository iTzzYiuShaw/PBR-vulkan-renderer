# PBR-vulkan-renderer
This renderer is Vulkan-based and has implemented features such as **model loading**, **PBR**, **Alpha masking**, **Normal mapping**, and **mesh data implementation**.

## Physical Based Rendering

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/d9622322-32f0-45bb-93c6-c9aa7c015fc8)

The NDF describes the distribution of microfacets on the surface of a material. 

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/5298f316-ac12-4c5b-b3fd-e81e301c02cb)

The masking term, also known as the geometry term, models the occlusion of light by micro facets, it affects the overall specular reflection. 

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/81fd113b-44f2-4435-af2c-cec44bb7dbef)

The Fresnel term accounts for the angle-dependent reflection of light from the surface of a material. 

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/cfe0226e-4d75-4283-8078-9b6c1188028c)


Specular

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/37419174-e5da-4a48-b1cd-69cfe58c1753)

Diffuse

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/ad58d1b2-6506-43ef-a65a-534fa76bafb6)

## Alpha masking

1.	Before implementing alpha masking pipeline
   
![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/5f4be916-7d1e-486f-8760-a5674ed461b7)

2.	After implementing alpha masking pipeline
![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/d1497d11-022b-4184-acc0-1c8586d84940)

## Normal mapping
1.	Before implementing normal mapping

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/7596ed72-5947-4514-9a89-4eb4b84e52e1)
![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/451c9907-3b5b-431d-acb8-26dea0e7f4e9)


2.	After implementing

![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/71253199-8c0d-490c-be1f-c16079f083c1)
![image](https://github.com/iTzzYiuShaw/PBR-vulkan-renderer/assets/110170509/74588082-7928-4e66-a4bd-333f5c6e504f)


## Mesh data optimizations

We encode the TBN matrix into a uint32 value.
First, since TBN matrix uses Tangent, Bitangent, and Normal as its columns and they are orthogonal vectors in 3D space, the TBN matrix is orthogonal. 
An orthogonal matrix can be used to represent a rotation. Therefore, the TBN matrix can be converted into a quaternion, which can also represent a rotation. 
After the conversion, the TBN now can be represented by 4 float values, and we normalize these float values into a range of [0,1] and then map the normalized values into a range of [0,255]. 
The reason why we would like to apply the mapping is that a uint32 value uses 32 bits and it can be seen as 4 uint8 values. Therefore, we can pack these 4 uint8 values into a single 32-bit integer by bit-shifting operation.
As a result, we can save (9 * 32) – (4 * 8) = 256 bits if we choose uint32 encoding. 


```
glm::mat3 TBN = glm::mat3(tan, bitangent, normal);

glm::quat tbnQuat = glm::quat(TBN);

// Get sign of w
uint32_t wSign = (tbnQuat.w >= 0) ? 0 : 1;

tbnQuat = glm::normalize(tbnQuat);

float x = tbnQuat.x;
float y = tbnQuat.y;
float z = tbnQuat.z;
float w = tbnQuat.w;

// Scale and bias each component to range [0, 1]
float fx = (x + 1.0f) / 2.0f;
float fy = (y + 1.0f) / 2.0f;
float fz = (z + 1.0f) / 2.0f;
float fw = (w + 1.0f) / 2.0f;


// Convert to integer in range 0 to 1023
glm::u8vec4 quatized = glm::u8vec4(fx * 255, fy * 255, fz * 255, fw * 255);

// Pack into 32-bit integer
uint32_t packed = 0;
packed |= quatized.x;
packed |= quatized.y << 8;
packed |= quatized.z << 16;
packed |= quatized.w << 24;
```





