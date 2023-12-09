#include "glm/fwd.hpp"


#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <volk/volk.h>
#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"
#include "MeshLoader.hpp"

#include <chrono>
#include<vulkan/vulkan.h>


namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "D:/Working/MSc Game Engineering/Vulkan/A2/cw2/assets/cw2/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";

		constexpr char const* kVertDensityShaderPath = SHADERDIR_ "defaultDensity.vert.spv";
		constexpr char const* kGeomDensityShaderPath = SHADERDIR_ "defaultDensity.geom.spv";
		constexpr char const* kFragDensityShaderPath = SHADERDIR_ "defaultDensity.frag.spv";
#		undef SHADERDIR_

#		define MODELDIR_ "assets/cw1/"
		constexpr char const* kObjectPath = MODELDIR_ "sponza_with_ship.obj";
		constexpr char const* kMateriaPath = MODELDIR_ "sponza_with_ship.mtl";
#		undef MODELDIR_

#		define TEXTUREDIR_ "assets/cw1/"
		constexpr char const* kFloorTexture = TEXTUREDIR_ "asphalt.png";
#		undef TEXTUREDIR_


		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;


		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;
		constexpr auto kCameraFov = 60.0_degf;


		constexpr float kCameraBaseSpeed = 1.7f; // units/second 
		constexpr float kCameraFastMult = 5.f; // speed multiplier 
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier 
		constexpr float kCameraMouseSensitivity = 0.001f; // radians per pixel 
	}

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};
		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;
		bool wasMousing = false;
		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};


	namespace glsl
	{


		struct SceneUniform
		{
			// Note: need to be careful about the packing/alignment here! 
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec3 cameraPos;
		};

		struct ColorUniform
		{
			glm::vec3 color;
		};	

		struct LightSource
		{
			glm::vec4 position;
			glm::vec4 color;
			float intensity;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");

	}

	// Local types/structures:

	// Local functions:
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	lut::RenderPass create_render_pass(lut::VulkanWindow const&);

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout const&, VkDescriptorSetLayout aObjectLayout, VkDescriptorSetLayout const& aLightSource
	);

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_mipmap_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_lightSource_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&);

	lut::Pipeline create_piepline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	VkBuffer create_color_uniform_buffer(std::vector<glsl::ColorUniform>const& colorUniform, lut::VulkanWindow const& window);

	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);


	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const& userState
	);

	void update_user_state(UserState&, float aElapsedTime);
	void record_commands(
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline pipeTexture,
		VkPipeline pipeAlpha,
		VkExtent2D const&,
		std::vector<IndexedMesh>* indexedMesh,
		VkBuffer aSceneUBO,
		glsl::SceneUniform
		const& aSceneUniform,
		VkBuffer aLightUBO,
		glsl::LightSource const& aLightUniform,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet lightDescriptors,
		std::vector<VkDescriptorSet*>* objectsDescriptors,
		std::vector<VkDescriptorSet*>* diffuseDescriptors
	);
	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);

	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);
}
int main() try
{
	//TODO-implement me.

	// Create our Vulkan Window
	lut::VulkanWindow window = lut::make_vulkan_window();

	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);


	//scene Uniform descriptor
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	//TODO- (Section 4) create object descriptor set layout
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);

	lut::DescriptorSetLayout lightLayout = create_lightSource_descriptor_layout(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass(window);

	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneLayout.handle, objectLayout.handle, lightLayout.handle);

	//Pipe line
	lut::Pipeline pipe = create_piepline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline alphaPipe = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	//Depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);


	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);


	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	//create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

		
	//Load model and meshes----------------------------------------------------------------------
	BakedModel bakedModel = load_baked_model("assets/cw2/sponza-pbr_tan_packed.comp5822mesh");
	std::vector<IndexedMesh>* indexedMesh = new std::vector<IndexedMesh>;
	for (int i = 0; i < bakedModel.meshes.size(); i++)
	{
		auto mesh = bakedModel.meshes[i];
		IndexedMesh temp = create_indexed_mesh(window, allocator, bakedModel, i);
		indexedMesh->emplace_back(std::move(temp));
	}
	//Load model and meshes----------------------------------------------------------------------

	//Samling textures----------------------------------------------------------------------
	lut::Sampler defalutSampler = lut::create_default_sampler(window);
	//lut::Sampler defalutSampler = lut::create_anisotrpic_sampler(window);
	lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

	std::vector<VkDescriptorSet*>* textureDescriptorsSet = new std::vector<VkDescriptorSet*>;
	std::vector<VkDescriptorSet*>* alphaDescriptorsSet = new std::vector<VkDescriptorSet*>;

	std::vector<lut::Image> imageSet;
	std::vector<lut::ImageView>imageViewSet;

	std::vector<lut::Image> alphaImageSet;
	std::vector<lut::ImageView>alphaImageViewSet;

	std::vector<lut::Image> normalMapImageSet;
	std::vector<lut::ImageView>normalMapImageViewSet;

	int alphaIndex = 0;
	int normalMapIndex = 0;
	for (int i = 0; i < indexedMesh->size(); i++)//changed
	{

		std::uint32_t materialId = ((*indexedMesh)[i].materialId);
		bool isAlpha = (*indexedMesh)[i].isAlphaMask;
		bool isNormalMap = (*indexedMesh)[i].isNormalMap;


		//Sampling base color
		std::uint32_t baseColorId = bakedModel.materials[materialId].baseColorTextureId;
		const char* baseColorPath = bakedModel.textures[baseColorId].path.c_str();

		imageSet.push_back(std::move((lut::load_image_texture2d(baseColorPath, window, loadCmdPool.handle, allocator))));
		imageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, imageSet[3*i].image, VK_FORMAT_R8G8B8A8_UNORM))));

		//Sampling roughness
		std::uint32_t roughnessId = bakedModel.materials[materialId].roughnessTextureId;
		const char* roughnessPath = bakedModel.textures[roughnessId].path.c_str();

		imageSet.push_back(std::move((lut::load_image_texture2d(roughnessPath, window, loadCmdPool.handle, allocator))));
		imageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, imageSet[3 * i +1].image, VK_FORMAT_R8G8B8A8_UNORM))));

		//Sampling metalness
		std::uint32_t metalnessId = bakedModel.materials[materialId].metalnessTextureId;
		const char* metalnessPath = bakedModel.textures[metalnessId].path.c_str();

		imageSet.push_back(std::move((lut::load_image_texture2d(metalnessPath, window, loadCmdPool.handle, allocator))));
		imageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, imageSet[3 * i + 2].image, VK_FORMAT_R8G8B8A8_UNORM))));

		//Sampling alphaTexture
		if (isAlpha)
		{
			std::uint32_t alphaMaskId = bakedModel.materials[materialId].alphaMaskTextureId;
			const char* alphaMaskPath = bakedModel.textures[alphaMaskId].path.c_str();

			alphaImageSet.push_back(std::move((lut::load_image_texture2d(alphaMaskPath, window, loadCmdPool.handle, allocator))));
			
			std::uint32_t index = alphaImageSet.size() - 1;
			alphaImageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, alphaImageSet[index].image, VK_FORMAT_R8G8B8A8_UNORM))));
		}

		//Sampling normalMap
		if (isNormalMap)
		{
			std::uint32_t normalMapId = bakedModel.materials[materialId].normalMapTextureId;
			const char* normalMapPath = bakedModel.textures[normalMapId].path.c_str();

			normalMapImageSet.push_back(std::move((lut::load_image_texture2d(normalMapPath, window, loadCmdPool.handle, allocator))));

			std::uint32_t index = normalMapImageSet.size() - 1;
			normalMapImageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, normalMapImageSet[index].image, VK_FORMAT_R8G8B8A8_UNORM))));
		}

		
		VkDescriptorSet* textureDescriptors = new VkDescriptorSet;
		*textureDescriptors = lut::alloc_desc_set(window, dpool.handle,
			objectLayout.handle);

		{
			VkWriteDescriptorSet desc[5]{};
			VkDescriptorImageInfo textureInfo[5]{};

			//Base color
			textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[0].imageView = imageViewSet[3 * i].handle;
			textureInfo[0].sampler = defalutSampler.handle;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = *textureDescriptors;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo[0];

			//Roughness
			textureInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[1].imageView = imageViewSet[3 * i +1].handle;
			textureInfo[1].sampler = defalutSampler.handle;

			desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[1].dstSet = *textureDescriptors;
			desc[1].dstBinding = 1;
			desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[1].descriptorCount = 1;
			desc[1].pImageInfo = &textureInfo[1];

			//Metalness
			textureInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[2].imageView = imageViewSet[3 * i + 2].handle;
			textureInfo[2].sampler = defalutSampler.handle;

			desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[2].dstSet = *textureDescriptors;
			desc[2].dstBinding = 2;
			desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[2].descriptorCount = 1;
			desc[2].pImageInfo = &textureInfo[2];


			//alphaMask
			textureInfo[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			if (isAlpha) //If this is not a mesh with alpha texture, then bind the baseColor on binding point
				textureInfo[3].imageView = alphaImageViewSet[alphaIndex++].handle;
			else 
				textureInfo[3].imageView = imageViewSet[3 * i].handle;

			textureInfo[3].sampler = defalutSampler.handle;

			desc[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[3].dstSet = *textureDescriptors;
			desc[3].dstBinding = 3;
			desc[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[3].descriptorCount = 1;
			desc[3].pImageInfo = &textureInfo[3];


			//normalMap
			textureInfo[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			if (isNormalMap) //If this is not a mesh with normalMap texture, then bind the baseColor on binding point
				textureInfo[4].imageView = normalMapImageViewSet[normalMapIndex++].handle;
			else
				textureInfo[4].imageView = imageViewSet[3 * i].handle;

			textureInfo[4].sampler = defalutSampler.handle;

			desc[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[4].dstSet = *textureDescriptors;
			desc[4].dstBinding = 4;
			desc[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[4].descriptorCount = 1;
			desc[4].pImageInfo = &textureInfo[4];

			vkUpdateDescriptorSets(window.device, 5, desc, 0, nullptr);
		}
		textureDescriptorsSet->push_back(textureDescriptors);
	}
	//Samling textures----------------------------------------------------------------------


	//Scene uniform----------------------------------------------------------------------
	lut::Buffer sceneUBO = lut::create_buffer(allocator, sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY
	);
	// allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}
	//Scene uniform----------------------------------------------------------------------


	//Light uniform----------------------------------------------------------------------

	lut::Buffer lightUBO = lut::create_buffer(allocator, sizeof(glsl::LightSource),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	VkDescriptorSet lightDescriptors = lut::alloc_desc_set(window, dpool.handle, lightLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorBufferInfo lightUboInfo{};
		lightUboInfo.buffer = lightUBO.buffer;
		lightUboInfo.range = VK_WHOLE_SIZE;
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = lightDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &lightUboInfo;
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//Light uniform----------------------------------------------------------------------



	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	glsl::SceneUniform sceneUniforms{};
	glsl::ColorUniform colorUniforms{};
	glsl::LightSource lightSourceUniforms{glm::vec4(0.0f, 2.0f, 0.0f,0.0f), glm::vec4(1.0f, 1.0f, 1.0f,0.0f), 1.0f };;

	// Application main loop
	bool recreateSwapchain = false;
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//TODO: re-create swapchain and associated resources!
			vkDeviceWaitIdle(window.device);

			// Recreate them 
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);


			if (changes.changedSize)
			{
				pipe = create_piepline(window, renderPass.handle, pipeLayout.handle);
				alphaPipe = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
				//pipe = create_density_pipeline(window, renderPass.handle, pipeLayout.handle);
			}

			if (changes.changedSize)
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);
			recreateSwapchain = false;
			continue;
		}

		//TODO: acquire swapchain image.
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE, &imageIndex);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			// This occurs e.g., when the window has been resized. In this case 
			// we need to recreate the swap chain to match the new dimensions. 
			// Any resources that directly depend on the swap chain need to be 
			// recreated as well. While rare, re-creating the swap chain may 
			// give us a different image format, which we should handle. 
			// 
			// In both cases, we set the flag that the swap chain has to be 
			// re-created and jump to the top of the loop. Technically, with 
			// the VK SUBOPTIMAL KHR return code, we could continue rendering 
			// with the current swap chain (unlike VK ERROR OUT OF DATE KHR, 
			// which does require us to recreate the swap chain). 
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str()
			);
		}

		//TODO: wait for command buffer to be available

		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE,
			std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//TODO: record and submit commands

		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			alphaPipe.handle,
			window.swapchainExtent,
			indexedMesh,
			sceneUBO.buffer,
			sceneUniforms,
			lightUBO.buffer,
			lightSourceUniforms,
			pipeLayout.handle,
			sceneDescriptors,
			lightDescriptors,
			textureDescriptorsSet,
			alphaDescriptorsSet
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		present_results(
			window.presentQueue,
			window.swapchain,
			imageIndex,
			renderFinished.handle,
			recreateSwapchain);


		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	delete indexedMesh;
	delete textureDescriptorsSet;
	delete alphaDescriptorsSet;
	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}



//Key response event
namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		default:
			;
		}
	}

	//The mouse button callback is fairly similar. It checks for presses of the right mouse button. When pressed,
//it toggles the EInputState::mousing state to indicate that mouse - look is now active.
	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;

			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		}
	}

	/*The final callback reports the position of the mouse cursor whenever the mouse is moved. Our implementation
	just updates the relevant variables in the UserState structure:
	*/
	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}
}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& userState)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis 

		//aSceneUniforms.camera = glm::translate(glm::vec3(0.f, -0.3f, -1.f));
		aSceneUniforms.camera = glm::inverse(userState.camera2world);

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;

		aSceneUniforms.cameraPos = glm::vec3(userState.camera2world[3]);

	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			// Only update the rotation on the second frame of mouse 7
			// navigation. This ensures that the previousX and Y variables are 8
			// initialized to sensible values. 9
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}
				
			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f)
			;

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));

	}
}



//Vulkan logic
namespace
{

	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //changed! 


		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;




		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; // this refers to attachments[1] 7
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		//Subpasses, subrendering procedures
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;



		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0; //changed! 
		passInfo.pDependencies = nullptr; //changed! 
		VkRenderPass rpass = VK_NULL_HANDLE;

		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);

	}


	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout aObjectLayout, 
		VkDescriptorSetLayout const& aLightSource
	)
	{
		//As mentioned, our shaders currently do not have any uniform inputs. 
		//While we still need to create a pipeline layout object(VkPipelineLayout),
		// -----------------------------------------------------------------------------
		//The lack of uniform inputs also means that we do not have to deal with descriptors, 
		//descriptor sets and descriptor set layouts just yet.They will appear in later exercises.

		VkDescriptorSetLayout layouts[] = { // Order must match the set = N in the shaders 
			aSceneLayout,// set 0
			aObjectLayout,
			//aMipmapLayout
			aLightSource
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;  // Make the push constant visible in fragment shader
		pushConstantRange.offset = 0;  // Start at the beginning of the push constant block
		pushConstantRange.size = sizeof(int) + sizeof(int);  // Size of the push constant block

		//create a pipeline layout object(VkPipelineLayout),
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}


	lut::Pipeline create_piepline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules 
		// For this example, we only use the vertex and fragment shaders.
		// Other shader stages (geometry, tessellation) aren’t used here, and as such we omit them.
		// Load the 
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);


		//There are 2 stages: VertexShader -> Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkVertexInputBindingDescription vertexInputs[5]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(float) * 4;
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[4].binding = 4;
		vertexInputs[4].stride = sizeof(std::uint32_t);
		vertexInputs[4].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//VkVertexInputBindingDescription vertexInputs[3]{};
		//vertexInputs[0].binding = 0;
		//vertexInputs[0].stride = sizeof(float) * 3;
		//vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//vertexInputs[1].binding = 1;
		//vertexInputs[1].stride = sizeof(float) * 2;
		//vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//vertexInputs[2].binding = 2;
		//vertexInputs[2].stride = sizeof(float) * 3;
		//vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;


		/**The vertex shader expects two inputs, the position and the color.
		Consequently, these are described with two VkVertexInputAttributeDescription instances*/
		VkVertexInputAttributeDescription vertexAttributes[5]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;

		vertexAttributes[3].binding = 3; // must match binding above 
		vertexAttributes[3].location = 3; // must match shader 
		vertexAttributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[3].offset = 0;

		vertexAttributes[4].binding = 4; // must match binding above 
		vertexAttributes[4].location = 4; // must match shader 
		vertexAttributes[4].format = VK_FORMAT_R32_UINT;
		vertexAttributes[4].offset = 0;

		//VkVertexInputAttributeDescription vertexAttributes[3]{};
		//vertexAttributes[0].binding = 0; // must match binding above 
		//vertexAttributes[0].location = 0; // must match shader 
		//vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		//vertexAttributes[0].offset = 0;

		//vertexAttributes[1].binding = 1; // must match binding above 
		//vertexAttributes[1].location = 1; // must match shader 
		//vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		//vertexAttributes[1].offset = 0;

		//vertexAttributes[2].binding = 2; // must match binding above 
		//vertexAttributes[2].location = 2; // must match shader 
		//vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		//vertexAttributes[2].offset = 0;


		//Vertex Input state
		//we specify what buffers vertices are sourced from, and what vertex attributes in our shaders these correspond to
		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 5; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 5; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		//VkPipelineVertexInputStateCreateInfo inputInfo{};
		//inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		//inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		//inputInfo.pVertexBindingDescriptions = vertexInputs;
		//inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		//inputInfo.pVertexAttributeDescriptions = vertexAttributes;



		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Tessellation State
		//For this exercise, we can leave it as nullptr
		//--To be implemented
		//VkPipelineTessellationDomainOriginStateCreateInfo tessellationInfo{};

		//Viewport State create info:
		//1: Initialize viewPort;
		//2: Initialize scissor;
		//3： createInfo
		VkViewport viewPort{};
		viewPort.x = 0.f;
		viewPort.y = 0.f;
		viewPort.width = float(aWindow.swapchainExtent.width);
		viewPort.height = float(aWindow.swapchainExtent.height);
		viewPort.minDepth = 0.f;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewPort;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Rasterization State
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		//Multisample State
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Depth/Stencil State levae it as nullptr
		//To be implemented:----
		//VkPipelineDepthStencilStateCreateInfo depthStencilInfo{ };


		//Color Blend State
		// Define blend state 1
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we don’t do any 
		// blending, so we can ignore most of the members. 

		//1: Initialize blendStates
		//2: Create colorblendStateInfo
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Dynamic States
		//Exercise 2 does not use any dynamic state, so the pDynamicState member is left at nullptr.
		//To be implemented:
		//VkPipelineDynamicStateCreateInfo dynamicInfo{};


		//Create the pipeLine
		//Some states that we won't use will be set to nullptr
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}


	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules 
		// For this example, we only use the vertex and fragment shaders.
		// Other shader stages (geometry, tessellation) aren’t used here, and as such we omit them.
		// Load the 
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);


		//There are 2 stages: VertexShader -> Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_FALSE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkVertexInputBindingDescription vertexInputs[5]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[3].binding = 3;
		vertexInputs[3].stride = sizeof(float) * 4;
		vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[4].binding = 4;
		vertexInputs[4].stride = sizeof(std::uint32_t);
		vertexInputs[4].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//VkVertexInputBindingDescription vertexInputs[3]{};
		//vertexInputs[0].binding = 0;
		//vertexInputs[0].stride = sizeof(float) * 3;
		//vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//vertexInputs[1].binding = 1;
		//vertexInputs[1].stride = sizeof(float) * 2;
		//vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//vertexInputs[2].binding = 2;
		//vertexInputs[2].stride = sizeof(float) * 3;
		//vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;


		/**The vertex shader expects two inputs, the position and the color.
		Consequently, these are described with two VkVertexInputAttributeDescription instances*/
		VkVertexInputAttributeDescription vertexAttributes[5]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;

		vertexAttributes[3].binding = 3; // must match binding above 
		vertexAttributes[3].location = 3; // must match shader 
		vertexAttributes[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[3].offset = 0;

		vertexAttributes[4].binding = 4; // must match binding above 
		vertexAttributes[4].location = 4; // must match shader 
		vertexAttributes[4].format = VK_FORMAT_R32_UINT;
		vertexAttributes[4].offset = 0;

		//VkVertexInputAttributeDescription vertexAttributes[3]{};
		//vertexAttributes[0].binding = 0; // must match binding above 
		//vertexAttributes[0].location = 0; // must match shader 
		//vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		//vertexAttributes[0].offset = 0;

		//vertexAttributes[1].binding = 1; // must match binding above 
		//vertexAttributes[1].location = 1; // must match shader 
		//vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		//vertexAttributes[1].offset = 0;

		//vertexAttributes[2].binding = 2; // must match binding above 
		//vertexAttributes[2].location = 2; // must match shader 
		//vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		//vertexAttributes[2].offset = 0;


		//Vertex Input state
		//we specify what buffers vertices are sourced from, and what vertex attributes in our shaders these correspond to
		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 5; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 5; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		//VkPipelineVertexInputStateCreateInfo inputInfo{};
		//inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		//inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		//inputInfo.pVertexBindingDescriptions = vertexInputs;
		//inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		//inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Tessellation State
		//For this exercise, we can leave it as nullptr
		//--To be implemented
		//VkPipelineTessellationDomainOriginStateCreateInfo tessellationInfo{};

		//Viewport State create info:
		//1: Initialize viewPort;
		//2: Initialize scissor;
		//3： createInfo
		VkViewport viewPort{};
		viewPort.x = 0.f;
		viewPort.y = 0.f;
		viewPort.width = float(aWindow.swapchainExtent.width);
		viewPort.height = float(aWindow.swapchainExtent.height);
		viewPort.minDepth = 0.f;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewPort;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Rasterization State
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		//Multisample State
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Depth/Stencil State levae it as nullptr
		//To be implemented:----
		//VkPipelineDepthStencilStateCreateInfo depthStencilInfo{ };


		//Color Blend State
		// Define blend state 1
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we don’t do any 
		// blending, so we can ignore most of the members. 

		//1: Initialize blendStates
		//2: Create colorblendStateInfo
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE; // New! Used to be VK FALSE. 
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD; // New! 
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // New! 
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // New! 
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		
		blendStates[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendStates[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendStates[0].alphaBlendOp = VK_BLEND_OP_ADD;


		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Dynamic States
		//Exercise 2 does not use any dynamic state, so the pDynamicState member is left at nullptr.
		//To be implemented:
		//VkPipelineDynamicStateCreateInfo dynamicInfo{};


		//Create the pipeLine
		//Some states that we won't use will be set to nullptr
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}



	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = {
			aWindow.swapViews[i],
			aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer 
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{

				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}


		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0,1,
			0,1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str()
			);
		}

		return { std::move(depthImage),lut::ImageView(aWindow.device,view) };
	}

	void record_commands(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, 
		VkPipeline aGraphicsPipe,
		VkPipeline aAlphaPipe,
		VkExtent2D const& aImageExtent,
		std::vector<IndexedMesh>* indexedMesh,
		VkBuffer aSceneUBO,
		glsl::SceneUniform
		const& aSceneUniform,
		VkBuffer aLightUBO,
		glsl::LightSource const& aLightUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet lightDescriptors,
		std::vector<VkDescriptorSet*>* objectsDescriptors,
		std::vector<VkDescriptorSet*>* diffuseDescriptors
		//VkBuffer aSpritePosBuffer,
		//VkBuffer aSpriteTexBuffer,
		//std::uint32_t aSpriteVertexCount,
		//VkDescriptorSet aSpriteObjDescriptors,
		//
	)
	{
		// Begin recording commands 
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n""vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}


		// Upload scene uniforms
		lut::buffer_barrier(aCmdBuff, aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		// Upload light uniforms
		lut::buffer_barrier(aCmdBuff, aLightUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, aLightUBO, 0, sizeof(glsl::LightSource), &aLightUniform);

		lut::buffer_barrier(aCmdBuff,
			aLightUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);


		// Begin render pass 
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background. 
		clearValues[0].color.float32[1] = 0.1f; // If we were debugging, this would potentially 
		clearValues[0].color.float32[2] = 0.1f; // help us see whether the render pass took 
		clearValues[0].color.float32[3] = 1.f; // place, even if nothing else was drawn. 

		clearValues[1].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Begin drawing indexed mesh with texture mesh pipeline 
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 2, 1, &lightDescriptors, 0, nullptr);


		//Bind vertex input for indexed mesh
		//Draw indexMesh that has no alphaMask, ensuring the "background items" are drew first
		for (int i = 0; i < indexedMesh->size(); i++)
		{
			if (!(*indexedMesh)[i].isAlphaMask)
			{
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, (*objectsDescriptors)[i], 0, nullptr);

				VkBuffer buffers[5] = { (*indexedMesh)[i].pos.buffer,(*indexedMesh)[i].texcoords.buffer,(*indexedMesh)[i].normals.buffer,(*indexedMesh)[i].tangent.buffer,(*indexedMesh)[i].packedTBN.buffer};
				VkDeviceSize offsets[5]{};
				vkCmdBindVertexBuffers(aCmdBuff, 0,5, buffers, offsets);

				vkCmdBindIndexBuffer(aCmdBuff, (*indexedMesh)[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				int isAlpha = 0;
				int isNormalMap = 0;
				if ((*indexedMesh)[i].isNormalMap)
				{
					isNormalMap = 1;
				}
				vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &isAlpha);
				vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(int), sizeof(int), &isNormalMap);
				vkCmdDrawIndexed(aCmdBuff, (*indexedMesh)[i].indexSize, 1, 0, 0, 0);
			}

		}

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aAlphaPipe);

		for (int i = 0; i < indexedMesh->size(); i++)
		{
			if ((*indexedMesh)[i].isAlphaMask)
			{
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, (*objectsDescriptors)[i], 0, nullptr);

				VkBuffer buffers[5] = { (*indexedMesh)[i].pos.buffer,(*indexedMesh)[i].texcoords.buffer,(*indexedMesh)[i].normals.buffer,(*indexedMesh)[i].tangent.buffer,(*indexedMesh)[i].packedTBN.buffer };
				VkDeviceSize offsets[5]{};
				vkCmdBindVertexBuffers(aCmdBuff, 0, 5, buffers, offsets);

				vkCmdBindIndexBuffer(aCmdBuff, (*indexedMesh)[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				int isAlpha = 1;
				int isNormalMap = 0;
				if ((*indexedMesh)[i].isNormalMap)
				{
					isNormalMap = 1;
				}
				vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &isAlpha);
				vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(int), sizeof(int), &isNormalMap);
				vkCmdDrawIndexed(aCmdBuff, (*indexedMesh)[i].indexSize, 1, 0, 0, 0);
			}
		}

		// End the render pass 
		vkCmdEndRenderPass(aCmdBuff);


		// End command recording 
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

	}

	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{

		//We must wait for the imageAvailable semaphore to become signalled, indicating that the swapchain image is ready,
		//before we draw to the image
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;

		submitInfo.pCommandBuffers = &aCmdBuff;
		submitInfo.waitSemaphoreCount = 1;

		//Wait for imageAvailable semaphore to become signalled
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		//Indicates which stage of the pipeline should wait
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		//we want to signal the renderFinished semaphore when commands have finished, to indicate that 
		//the rendered image is ready for presentation
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;
		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
			);
		}
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding 
		// binding = N declaration in the shader(s)! 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);


	}


	lut::DescriptorSetLayout create_lightSource_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);

	}

	lut::DescriptorSetLayout create_mipmap_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding 
		// binding = N declaration in the shader(s)! 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[5]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[1].binding = 1; // this must match the shaders 
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[2].binding = 2; // this must match the shaders 
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[3].binding = 3; // this must match the shaders 
		bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[3].descriptorCount = 1;
		bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[4].binding = 4; // this must match the shaders 
		bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[4].descriptorCount = 1;
		bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	VkBuffer create_color_uniform_buffer(std::vector<glsl::ColorUniform>const& colorUniform, lut::VulkanWindow const& window)
	{
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = colorUniform.size() * sizeof(glsl::ColorUniform);
		bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer colorBuffer;
		VkDeviceMemory colorBufferMemory;
		vkCreateBuffer(window.device, &bufferInfo, nullptr, &colorBuffer);

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(window.device, colorBuffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;

		vkAllocateMemory(window.device, &allocInfo, nullptr, &colorBufferMemory);
		vkBindBufferMemory(window.device, colorBuffer, colorBufferMemory, 0);

		void* data;
		vkMapMemory(window.device, colorBufferMemory, 0, bufferInfo.size, 0, &data);
		memcpy(data, colorUniform.data(), bufferInfo.size);
		vkUnmapMemory(window.device, colorBufferMemory);

		return std::move(colorBuffer);
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;

		//we pass the renderFinished semaphore to pWaitSemaphores, to indicate that presentation should only occur once the semaphore is signalled
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;
		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n" "vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}
}



//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
