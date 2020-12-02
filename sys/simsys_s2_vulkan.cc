/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */


#include "simsys.h"
#include "simsys_s2_vulkan.h"

#include "../simversion.h"
#include "../simdebug.h"
#include "../macros.h"

#include <sys/time.h>
#include <csignal>
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <sstream>
#include <array>


#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct swap_chain_support_details_t {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

static QueueFamilyIndices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface);
static bool check_device_extension_support(VkPhysicalDevice device);
static swap_chain_support_details_t query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface);
static bool is_device_suitable(VkPhysicalDevice device, VkSurfaceKHR surface);
static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, sim_window_t *window);
static VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
static std::vector<char> readShader(const std::string& filename);
static VkShaderModule create_shader_module(VkDevice device, const std::vector<char>& code);

static const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> device_extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static bool check_validation_layer_support() {
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

	DBG_MESSAGE("vulkan validation", pCallbackData->pMessage);

	return VK_FALSE;
}

static std::vector<const char*> get_required_extensions() {
	uint32_t extensionCount = 0;
	std::vector<const char*> extensions;

	if (!SDL_Vulkan_GetInstanceExtensions(nullptr, &extensionCount, nullptr)) {
		throw std::runtime_error("error getting SDL default extension list");
	}

	extensions.resize(extensionCount);
	SDL_Vulkan_GetInstanceExtensions(nullptr, &extensionCount,extensions.data());

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

static void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
}

static VkResult create_debug_utils_messenger_ext(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

static void destroy_debug_utils_messenger_ext(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}


void sim_renderer_t::run() {
	// (more or less) follow the vulkan-tutorial
	init_vulkan();
}

void sim_renderer_t::cleanup_swap_chain() {
	for (auto framebuffer : swap_chain_framebuffers) {
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}

	vkFreeCommandBuffers(device, command_pool, static_cast<uint32_t>(command_buffers.size()), command_buffers.data());

	vkDestroyPipeline(device, graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyRenderPass(device, renderPass, nullptr);

	for (auto image_view : swap_chain_image_views) {
		vkDestroyImageView(device, image_view, nullptr);
	}

	vkDestroySwapchainKHR(device, swap_chain, nullptr);
}

void sim_renderer_t::cleanup() {
	cleanup_swap_chain();

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
		vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
		vkDestroyFence(device, in_flight_fences[i], nullptr);
	}

	vkDestroyCommandPool(device, command_pool, nullptr);

	vkDestroyDevice(device, nullptr);

	if (enableValidationLayers) {
		destroy_debug_utils_messenger_ext(instance, debugMessenger, nullptr);
	}

	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
}

void sim_renderer_t::init_vulkan() {
	create_vulkan_instance();
	setup_debug_messenger();
	create_surface();
	pick_physical_device();
	create_logical_device();
	create_swap_chain();
	create_image_views();
	create_render_pass();

	create_graphics_pipeline();



	create_framebuffers();
	create_command_pool();
	create_command_buffers();
	create_sync_objects();
}

void sim_renderer_t::create_vulkan_instance() {
	if (enableValidationLayers && !check_validation_layer_support()) {
		throw std::runtime_error("validation layers requested, but not available!");
	}

	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = SIM_TITLE;
	appInfo.applicationVersion = VK_MAKE_VERSION(SIM_VERSION_MAJOR, SIM_VERSION_MINOR, SIM_VERSION_PATCH);
	appInfo.pEngineName = "Simutrans";
	appInfo.engineVersion = VK_MAKE_VERSION(SIM_VERSION_MAJOR, SIM_VERSION_MINOR, SIM_VERSION_PATCH);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	auto extensions = get_required_extensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		populate_debug_messenger_create_info(debugCreateInfo);
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
	} else {
		createInfo.enabledLayerCount = 0;
		createInfo.pNext = nullptr;
	}

	if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("failed to create instance");
	}
}

void sim_renderer_t::setup_debug_messenger() {
	if (!enableValidationLayers) return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo;
	populate_debug_messenger_create_info(createInfo);

	if (create_debug_utils_messenger_ext(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger!");
	}
}


void sim_renderer_t::create_surface() {
	window->create_vulkan_surface(instance, &surface);
}


// TODO: pick device based on something (e.g. RAM or user selected) instead of taking the first one
void sim_renderer_t::pick_physical_device() {
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	for (const auto& device : devices) {
		if (is_device_suitable(device, surface)) {
			physicalDevice = device;
			// msaa_samples = get_max_usable_sample_count();
			break;
		}
	}

	if (physicalDevice == VK_NULL_HANDLE) {
		throw std::runtime_error("failed to find a suitable GPU");
	}
}

void sim_renderer_t::create_logical_device() {
	QueueFamilyIndices indices = find_queue_families(physicalDevice, surface);

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();

	createInfo.pEnabledFeatures = &deviceFeatures;

	createInfo.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
	createInfo.ppEnabledExtensionNames = device_extensions.data();

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	} else {
		createInfo.enabledLayerCount = 0;
	}

	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device");
	}

	vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphics_queue);
	vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &present_queue);
}

void sim_renderer_t::recreate_swap_chain() {
	vkDeviceWaitIdle(device);

	cleanup_swap_chain();

	create_swap_chain();
	create_image_views();
	create_render_pass();
	create_graphics_pipeline();
	create_framebuffers();
	create_command_buffers();
}

void sim_renderer_t::create_swap_chain() {
	swap_chain_support_details_t swapChainSupport = query_swap_chain_support(physicalDevice, surface);

	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;

	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = find_queue_families(physicalDevice, surface);
	uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

	if (indices.graphicsFamily != indices.presentFamily) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	} else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0;
		createInfo.pQueueFamilyIndices = nullptr;
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swap_chain) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain");
	}

	vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, nullptr);
	swap_chain_images.resize(imageCount);
	vkGetSwapchainImagesKHR(device, swap_chain, &imageCount, swap_chain_images.data());

	swap_chain_image_format = surfaceFormat.format;
	swap_chain_extent = extent;
}

void sim_renderer_t::create_image_views() {
	swap_chain_image_views.resize(swap_chain_images.size());

	for (uint32_t i = 0; i < swap_chain_images.size(); i++) {
		swap_chain_image_views[i] = createImageView(device, swap_chain_images[i], swap_chain_image_format, VK_IMAGE_ASPECT_COLOR_BIT, 1 /* mip_levels */);
	}
}

void sim_renderer_t::create_render_pass() {
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = swap_chain_image_format;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // msaaSamples;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// VkAttachmentDescription depthAttachment{};
	// depthAttachment.format = findDepthFormat();
	// depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // msaaSamples;
	// depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	// depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	// depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	// depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	// depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// VkAttachmentDescription colorAttachmentResolve{};
	// colorAttachmentResolve.format = swap_chain_image_format;
	// colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
	// colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	// colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	// colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	// colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	// colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	// colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	// VkAttachmentReference depthAttachmentRef{};
	// depthAttachmentRef.attachment = 1;
	// depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	// VkAttachmentReference colorAttachmentResolveRef{};
	// colorAttachmentResolveRef.attachment = 2;
	// colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	// subpass.pDepthStencilAttachment = &depthAttachmentRef;
	// subpass.pResolveAttachments = &colorAttachmentResolveRef;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	std::array<VkAttachmentDescription, 1> attachments = {colorAttachment, /* depthAttachment, colorAttachmentResolve */};
	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass");
	}
}

void sim_renderer_t::create_graphics_pipeline() {
	auto vertShaderCode = readShader("vert.spv");
	auto fragShaderCode = readShader("frag.spv");

	VkShaderModule vertShaderModule = create_shader_module(device, vertShaderCode);
	VkShaderModule fragShaderModule = create_shader_module(device, fragShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	// auto bindingDescription = Vertex::getBindingDescription();
	// auto attributeDescriptions = Vertex::getAttributeDescriptions();

	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;

	// vertexInputInfo.vertexBindingDescriptionCount = 1;
	// vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	// vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	// vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float) swap_chain_extent.width;
	viewport.height = (float) swap_chain_extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = swap_chain_extent;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0;
	rasterizer.depthBiasClamp = 0.0;
	rasterizer.depthBiasSlopeFactor = 0.0;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	// multisampling.rasterizationSamples = msaaSamples;
	multisampling.minSampleShading = 1.0;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.minDepthBounds = 0.0;
	depthStencil.maxDepthBounds = 1.0;
	depthStencil.stencilTestEnable = VK_FALSE;
	depthStencil.front = {};
	depthStencil.back = {};


	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 0;  // 1;
	pipelineLayoutInfo.pSetLayouts = nullptr; // &descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;


	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline_layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout");
	}

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = pipeline_layout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
	pipelineInfo.basePipelineIndex = -1;

	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline");
	}

	vkDestroyShaderModule(device, fragShaderModule, nullptr);
	vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void sim_renderer_t::create_framebuffers() {
	swap_chain_framebuffers.resize(swap_chain_image_views.size());

	for (size_t i = 0; i < swap_chain_image_views.size(); i++) {
		std::array<VkImageView, 1> attachments = {
				// colorImageView,
				// depthImageView,
				swap_chain_image_views[i]
		};

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width =  swap_chain_extent.width;
		framebufferInfo.height = swap_chain_extent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swap_chain_framebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer");
		}
	}
}

void sim_renderer_t::create_command_pool() {
	QueueFamilyIndices queueFamilyIndices = find_queue_families(physicalDevice, surface);

	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
	poolInfo.flags = 0;

	if (vkCreateCommandPool(device, &poolInfo, nullptr, &command_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics command pool");
	}
}


void sim_renderer_t::create_command_buffers() {
	command_buffers.resize(swap_chain_framebuffers.size());

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = command_pool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t)command_buffers.size();

	if (vkAllocateCommandBuffers(device, &allocInfo, command_buffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers");
	}

	for (size_t i = 0; i < command_buffers.size(); i++) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(command_buffers[i], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swap_chain_framebuffers[i];
		renderPassInfo.renderArea.offset = {0, 0};
		renderPassInfo.renderArea.extent = swap_chain_extent;

		VkClearValue clear_color = {0.0f, 0.0f, 0.0f, 1.0f};
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clear_color;

		vkCmdBeginRenderPass(command_buffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			vkCmdDraw(command_buffers[i], 3, 1, 0, 0);

		vkCmdEndRenderPass(command_buffers[i]);

		if (vkEndCommandBuffer(command_buffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer");
		}
	}
}

void sim_renderer_t::create_sync_objects() {
	image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
	render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
	in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
	images_in_flight.resize(swap_chain_images.size(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphore_info{};
	semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_info{};
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (
				vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_semaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame");
		}
	}
}

void sim_renderer_t::draw_frame() {
	vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

	uint32_t image_index;
	VkResult result = vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX, image_available_semaphores[current_frame], VK_NULL_HANDLE, &image_index);
	if (images_in_flight[image_index] != VK_NULL_HANDLE) {
		vkWaitForFences(device, 1, &images_in_flight[image_index],  VK_TRUE, UINT64_MAX);
	}

	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		recreate_swap_chain();
		return;
	} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("failed to acquire swap chain image");
	}

	images_in_flight[image_index] = in_flight_fences[current_frame];

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { image_available_semaphores[current_frame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &command_buffers[image_index];

	VkSemaphore signal_semaphores[] = { render_finished_semaphores[current_frame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signal_semaphores;

	vkResetFences(device, 1, &in_flight_fences[current_frame]);
	if (vkQueueSubmit(graphics_queue, 1, &submitInfo, in_flight_fences[current_frame]) != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signal_semaphores;

	VkSwapchainKHR swap_chains[] = { swap_chain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swap_chains;
	presentInfo.pImageIndices = &image_index;

	result = vkQueuePresentKHR(present_queue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
		framebuffer_resized = false;
		recreate_swap_chain();
	} else if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to present swap chain image");
	}

	current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// Helpers implementations

static bool is_device_suitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
	QueueFamilyIndices indices = find_queue_families(device, surface);

	bool extensions_supported = check_device_extension_support(device);

	bool swapChainAdequate = false;
	if (extensions_supported) {
		swap_chain_support_details_t swap_chain_support = query_swap_chain_support(device, surface);
		swapChainAdequate = !swap_chain_support.formats.empty() && !swap_chain_support.presentModes.empty();
	}

	VkPhysicalDeviceFeatures supported_features;
	vkGetPhysicalDeviceFeatures(device, &supported_features);

	return indices.isComplete() && extensions_supported && swapChainAdequate && supported_features.samplerAnisotropy;
}


static QueueFamilyIndices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) {
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.graphicsFamily = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (presentSupport) {
			indices.presentFamily = i;
		}

		if (indices.isComplete()) {
			break;
		}

		i++;
	}

	return indices;
}

static bool check_device_extension_support(VkPhysicalDevice device) {
	uint32_t extension_count;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

	std::vector<VkExtensionProperties> available_extensions(extension_count);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

	std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

	for (const auto& extension : available_extensions) {
		required_extensions.erase(extension.extensionName);
	}

	return required_extensions.empty();
}

static swap_chain_support_details_t query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface) {
	swap_chain_support_details_t details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return availableFormat;
		}
	}

	return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			return availablePresentMode;
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, sim_window_t *window) {
	if (capabilities.currentExtent.width != UINT32_MAX) {
		return capabilities.currentExtent;
	} else {
		// TODO: dpi scaling
		resolution res = window->get_drawable_size();

		VkExtent2D actualExtent = {
				static_cast<uint32_t>(res.w),
				static_cast<uint32_t>(res.h)
		};

		actualExtent.width = sim::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = sim::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

static VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.subresourceRange.aspectMask = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture image view");
	}

	return imageView;
}

// TODO: Compile shaders on instantiation
// maybe allow pak creators to specify their own shaders
static std::vector<char> readShader(const std::string& filename) {
	// in simutrans dir
	std::ifstream file("./shaders/" + filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t) file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

static VkShaderModule create_shader_module(VkDevice device, const std::vector<char>& code) {
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module");
	}

	return shaderModule;
}


// old simsys_s2.cc


#ifdef _WIN32
#include <windows.h>
#endif

#include <stdio.h>
#include <string.h>

#ifdef __CYGWIN__
extern int __argc;
extern char **__argv;
#endif

#include "../macros.h"
#include "simsys_w32_png.h"
#include "simsys.h"
#include "../simversion.h"
#include "../simevent.h"
#include "../display/simgraph.h"
#include "../simdebug.h"
#include "../dataobj/environment.h"
#include "../gui/simwin.h"
#include "../gui/components/gui_component.h"
#include "../gui/components/gui_textinput.h"

// Maybe Linux is not fine too, had critical bugs...
#if !defined(__linux__)
#define USE_SDL_TEXTEDITING
#else
#endif

static Uint8 hourglass_cursor[] = {
		0x3F, 0xFE, //   *************
		0x30, 0x06, //   **         **
		0x3F, 0xFE, //   *************
		0x10, 0x04, //    *         *
		0x10, 0x04, //    *         *
		0x12, 0xA4, //    *  * * *  *
		0x11, 0x44, //    *  * * *  *
		0x18, 0x8C, //    **   *   **
		0x0C, 0x18, //     **     **
		0x06, 0xB0, //      ** * **
		0x03, 0x60, //       ** **
		0x03, 0x60, //       **H**
		0x06, 0x30, //      ** * **
		0x0C, 0x98, //     **     **
		0x18, 0x0C, //    **   *   **
		0x10, 0x84, //    *    *    *
		0x11, 0x44, //    *   * *   *
		0x12, 0xA4, //    *  * * *  *
		0x15, 0x54, //    * * * * * *
		0x3F, 0xFE, //   *************
		0x30, 0x06, //   **         **
		0x3F, 0xFE  //   *************
};

static Uint8 hourglass_cursor_mask[] = {
		0x3F, 0xFE, //   *************
		0x3F, 0xFE, //   *************
		0x3F, 0xFE, //   *************
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x0F, 0xF8, //     *********
		0x07, 0xF0, //      *******
		0x03, 0xE0, //       *****
		0x03, 0xE0, //       **H**
		0x07, 0xF0, //      *******
		0x0F, 0xF8, //     *********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x1F, 0xFC, //    ***********
		0x3F, 0xFE, //   *************
		0x3F, 0xFE, //   *************
		0x3F, 0xFE  //   *************
};

static Uint8 blank_cursor[] = {
		0x0,
		0x0,
};

static SDL_Window *window;
static SDL_Renderer *renderer;
static SDL_Texture *screen_tx;
static SDL_Surface *screen;

static int sync_blit = 0;
static int use_dirty_tiles = 1;

static SDL_Cursor *arrow;
static SDL_Cursor *hourglass;
static SDL_Cursor *blank;


// Number of fractional bits for screen scaling
#define SCALE_SHIFT_X 5
#define SCALE_SHIFT_Y 5

#define SCALE_NEUTRAL_X (1 << SCALE_SHIFT_X)
#define SCALE_NEUTRAL_Y (1 << SCALE_SHIFT_Y)

// Multiplier when converting from texture to screen coords, fixed point format
// Example: If x_scale==2*SCALE_NEUTRAL_X && y_scale==2*SCALE_NEUTRAL_Y,
// then things on screen are 2*2 = 4 times as big by area
sint32 x_scale = SCALE_NEUTRAL_X;
sint32 y_scale = SCALE_NEUTRAL_Y;

// When using -autodpi, attempt to scale things on screen to this DPI value
#define TARGET_DPI (96)


// screen -> texture coords
#define SCREEN_TO_TEX_X(x) (((x) * SCALE_NEUTRAL_X) / x_scale)
#define SCREEN_TO_TEX_Y(y) (((y) * SCALE_NEUTRAL_Y) / y_scale)

// texture -> screen coords
#define TEX_TO_SCREEN_X(x) (((x) * x_scale) / SCALE_NEUTRAL_X)
#define TEX_TO_SCREEN_Y(y) (((y) * y_scale) / SCALE_NEUTRAL_Y)


// no autoscaling yet
bool dr_auto_scale(bool on_off )
{
#if SDL_VERSION_ATLEAST(2,0,4)
	if(  on_off  ) {
		float hdpi, vdpi;
		SDL_Init( SDL_INIT_VIDEO );
		if(  SDL_GetDisplayDPI( 0, NULL, &hdpi, &vdpi )==0  ) {
			x_scale = ((sint64)hdpi * SCALE_NEUTRAL_X + 1) / TARGET_DPI;
			y_scale = ((sint64)vdpi * SCALE_NEUTRAL_Y + 1) / TARGET_DPI;
			return true;
		}
		return false;
	}
	else
#else
#pragma message "SDL version must be at least 2.0.4 to support autoscaling."
#endif
	{
		// 1.5 scale up by default
		x_scale = (3*SCALE_NEUTRAL_X)/2;
		y_scale = (3*SCALE_NEUTRAL_Y)/2;
		(void)on_off;
		return false;
	}
}

/*
 * Hier sind die Basisfunktionen zur Initialisierung der
 * Schnittstelle untergebracht
 * -> init,open,close
 */
bool dr_os_init(const int* parameter)
{
	if(  SDL_Init( SDL_INIT_VIDEO ) != 0  ) {
		dbg->error("dr_os_init(SDL2)", "Could not initialize SDL: %s", SDL_GetError() );
		return false;
	}

	dbg->message("dr_os_init(SDL2)", "SDL Driver: %s", SDL_GetCurrentVideoDriver() );

	// disable event types not interested in
#ifndef USE_SDL_TEXTEDITING
	SDL_EventState( SDL_TEXTEDITING, SDL_DISABLE );
#endif
	SDL_EventState( SDL_FINGERDOWN, SDL_DISABLE );
	SDL_EventState( SDL_FINGERUP, SDL_DISABLE );
	SDL_EventState( SDL_FINGERMOTION, SDL_DISABLE );
	SDL_EventState( SDL_DOLLARGESTURE, SDL_DISABLE );
	SDL_EventState( SDL_DOLLARRECORD, SDL_DISABLE );
	SDL_EventState( SDL_MULTIGESTURE, SDL_DISABLE );
	SDL_EventState( SDL_CLIPBOARDUPDATE, SDL_DISABLE );
	SDL_EventState( SDL_DROPFILE, SDL_DISABLE );

	sync_blit = parameter[0];  // hijack SDL1 -async flag for SDL2 vsync
	use_dirty_tiles = !parameter[1]; // hijack SDL1 -use_hw flag to turn off dirty tile updates (force fullscreen updates)

	// prepare for next event
	sys_event.type = SIM_NOEVENT;
	sys_event.code = 0;

	SDL_StartTextInput();

	atexit( SDL_Quit ); // clean up on exit
	return true;
}


resolution dr_query_screen_resolution()
{
	resolution res;
	SDL_DisplayMode mode;
	SDL_GetCurrentDisplayMode( 0, &mode );
	DBG_MESSAGE("dr_query_screen_resolution(SDL2)", "screen resolution width=%d, height=%d", mode.w, mode.h );
	res.w = mode.w;
	res.h = mode.h;
	return res;
}


bool internal_create_surfaces(int tex_width, int tex_height)
{
	// The pixel format needs to match the graphics code within simgraph16.cc.
	// Note that alpha is handled by simgraph16, not by SDL.
	const Uint32 pixel_format = SDL_PIXELFORMAT_RGB565;

#ifdef MSG_LEVEL
	// List all render drivers and their supported pixel formats.
	const int num_rend = SDL_GetNumRenderDrivers();
	std::string formatStrBuilder;
	for(  int i = 0;  i < num_rend;  i++  ) {
		SDL_RendererInfo ri;
		SDL_GetRenderDriverInfo( i, &ri );
		formatStrBuilder.clear();
		for(  Uint32 j = 0;  j < ri.num_texture_formats;  j++  ) {
			formatStrBuilder += ", ";
			formatStrBuilder += SDL_GetPixelFormatName(ri.texture_formats[j]);
		}
		DBG_DEBUG( "internal_create_surfaces(SDL2)", "Renderer: %s, Max_w: %d, Max_h: %d, Flags: %d, Formats: %d%s",
				   ri.name, ri.max_texture_width, ri.max_texture_height, ri.flags, ri.num_texture_formats, formatStrBuilder.c_str() );
	}
#endif

	Uint32 flags = SDL_RENDERER_ACCELERATED;
	if(  sync_blit  ) {
		flags |= SDL_RENDERER_PRESENTVSYNC;
	}
	renderer = SDL_CreateRenderer( window, -1, flags );
	if(  renderer == NULL  ) {
		dbg->warning( "internal_create_surfaces(SDL2)", "Couldn't create accelerated renderer: %s", SDL_GetError() );

		flags &= ~SDL_RENDERER_ACCELERATED;
		flags |= SDL_RENDERER_SOFTWARE;
		renderer = SDL_CreateRenderer( window, -1, flags );
		if(  renderer == NULL  ) {
			dbg->error( "internal_create_surfaces(SDL2)", "No suitable SDL2 renderer found!" );
			return false;
		}
		dbg->warning( "internal_create_surfaces(SDL2)", "Using fallback software renderer instead of accelerated: Performance may be low!");
	}

	SDL_RendererInfo ri;
	SDL_GetRendererInfo( renderer, &ri );
	DBG_DEBUG( "internal_create_surfaces(SDL2)", "Using: Renderer: %s, Max_w: %d, Max_h: %d, Flags: %d, Formats: %d, %s",
			   ri.name, ri.max_texture_width, ri.max_texture_height, ri.flags, ri.num_texture_formats, SDL_GetPixelFormatName(pixel_format) );

	screen_tx = SDL_CreateTexture( renderer, pixel_format, SDL_TEXTUREACCESS_STREAMING, tex_width, tex_height );
	if(  screen_tx == NULL  ) {
		dbg->error( "internal_create_surfaces(SDL2)", "Couldn't create texture: %s", SDL_GetError() );
		return false;
	}

	// Color component bitmasks for the RGB565 pixel format used by simgraph16.cc
	int bpp;
	Uint32 rmask, gmask, bmask, amask;
	if(  !SDL_PixelFormatEnumToMasks( pixel_format, &bpp, &rmask, &gmask, &bmask, &amask )  ) {
		dbg->error( "internal_create_surfaces(SDL2)", "Pixel format error. Couldn't generate masks: %s", SDL_GetError() );
		return false;
	} else if(  bpp != COLOUR_DEPTH  ||  amask != 0  ) {
		dbg->error( "internal_create_surfaces(SDL2)", "Pixel format error. Bpp got %d, needed %d. Amask got %d, needed 0.", bpp, COLOUR_DEPTH, amask );
		return false;
	}

	screen = SDL_CreateRGBSurface( 0, tex_width, tex_height, bpp, rmask, gmask, bmask, amask );
	if(  screen == NULL  ) {
		dbg->error( "internal_create_surfaces(SDL2)", "Couldn't get the window surface: %s", SDL_GetError() );
		return false;
	}

	return true;
}


// open the window
int dr_os_open(int screen_width, int screen_height, bool const fullscreen)
{
	// scale up
	const int tex_w = SCREEN_TO_TEX_X(screen_width);
	const int tex_h = SCREEN_TO_TEX_Y(screen_height);

	// some cards need those alignments
	// especially 64bit want a border of 8bytes
	const int tex_pitch = max((tex_w + 15) & 0x7FF0, 16);

	Uint32 flags = fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP: SDL_WINDOW_RESIZABLE;
	flags |= SDL_WINDOW_ALLOW_HIGHDPI; // apparently needed for Apple retina displays

	window = SDL_CreateWindow( SIM_TITLE, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screen_width, screen_height, flags );
	if(  window == NULL  ) {
		dbg->error("dr_os_open(SDL2)", "Could not open the window: %s", SDL_GetError() );
		return 0;
	}

	// Non-integer scaling -> enable bilinear filtering (must be done before texture creation)
	if ((x_scale & (SCALE_NEUTRAL_X - 1)) != 0 || (y_scale & (SCALE_NEUTRAL_Y - 1)) != 0) {
		SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1"); // 0=none, 1=bilinear, 2=anisotropic (DirectX only)
	}

	if(  !internal_create_surfaces( tex_pitch, tex_h )  ) {
		return 0;
	}
	DBG_MESSAGE("dr_os_open(SDL2)", "SDL realized screen size width=%d, height=%d (internal w=%d, h=%d)", screen_width, screen_height, screen->w, screen->h );

	SDL_ShowCursor(0);
	arrow = SDL_GetCursor();
	hourglass = SDL_CreateCursor( hourglass_cursor, hourglass_cursor_mask, 16, 22, 8, 11 );
	blank = SDL_CreateCursor( blank_cursor, blank_cursor, 8, 2, 0, 0 );
	SDL_ShowCursor(1);

	if(  !env_t::hide_keyboard  ) {
		// enable keyboard input at all times unless requested otherwise
		SDL_StartTextInput();
	}

	assert(tex_pitch <= screen->pitch / (int)sizeof(PIXVAL));
	assert(tex_h <= screen->h);
	assert(tex_w <= tex_pitch);

	display_set_actual_width( tex_w );
	display_set_height( tex_h );
	return tex_pitch;
}


// shut down SDL
void dr_os_close()
{
	SDL_FreeCursor( blank );
	SDL_FreeCursor( hourglass );
	SDL_DestroyRenderer( renderer );
	SDL_DestroyWindow( window );
	SDL_StopTextInput();
}


// resizes screen
int dr_textur_resize(unsigned short** const textur, int tex_w, int const tex_h)
{
	// enforce multiple of 16 pixels, or there are likely mismatches
	const int tex_pitch = max((tex_w + 15) & 0x7FF0, 16);

	SDL_UnlockTexture( screen_tx );
	if(  tex_pitch != screen->w  ||  tex_h != screen->h  ) {
		// Recreate the SDL surfaces at the new resolution.
		// First free surface and then renderer.
		SDL_FreeSurface( screen );
		screen = NULL;
		// This destroys texture as well.
		SDL_DestroyRenderer( renderer );
		renderer = NULL;
		screen_tx = NULL;

		internal_create_surfaces( tex_pitch, tex_h );
		if(  screen  ) {
			DBG_MESSAGE("dr_textur_resize(SDL2)", "SDL realized screen size width=%d, height=%d (internal w=%d, h=%d)", tex_w, tex_h, screen->w, screen->h );
		}
		else {
			dbg->error("dr_textur_resize(SDL2)", "screen is NULL. Good luck!");
		}
		fflush( NULL );
	}

	*textur = dr_textur_init();

	assert(tex_pitch <= screen->pitch / (int)sizeof(PIXVAL));
	assert(tex_h <= screen->h);
	assert(tex_w <= tex_pitch);

	display_set_actual_width( tex_w );
	return tex_pitch;
}


unsigned short *dr_textur_init()
{
	// SDL_LockTexture modifies pixels, so copy it first
	void *pixels = screen->pixels;
	int pitch = screen->pitch;

	SDL_LockTexture( screen_tx, NULL, &pixels, &pitch );
	return (unsigned short*)screen->pixels;
}


/**
 * Transform a 24 bit RGB color into the system format.
 * @return converted color value
 */
unsigned int get_system_color(unsigned int r, unsigned int g, unsigned int b)
{
	SDL_PixelFormat *fmt = SDL_AllocFormat( SDL_PIXELFORMAT_RGB565 );
	unsigned int ret = SDL_MapRGB( fmt, (Uint8)r, (Uint8)g, (Uint8)b );
	SDL_FreeFormat( fmt );
	return ret;
}


void dr_prepare_flush()
{
	return;
}


void dr_flush()
{
	display_flush_buffer();
	if(  !use_dirty_tiles  ) {
		SDL_UpdateTexture( screen_tx, NULL, screen->pixels, screen->pitch );
	}

	SDL_Rect rSrc  = { 0, 0, display_get_width(), display_get_height()  };
	SDL_RenderCopy( renderer, screen_tx, &rSrc, NULL );

	SDL_RenderPresent( renderer );
}


void dr_textur(int xp, int yp, int w, int h)
{
	if(  use_dirty_tiles  ) {
		SDL_Rect r;
		r.x = xp;
		r.y = yp;
		r.w = xp + w > screen->w ? screen->w - xp : w;
		r.h = yp + h > screen->h ? screen->h - yp : h;
		SDL_UpdateTexture( screen_tx, &r, (uint8 *)screen->pixels + yp * screen->pitch + xp * sizeof(PIXVAL), screen->pitch );
	}
}


// move cursor to the specified location
void move_pointer(int x, int y)
{
	SDL_WarpMouseInWindow( window, TEX_TO_SCREEN_X(x), TEX_TO_SCREEN_Y(y) );
}


// set the mouse cursor (pointer/load)
void set_pointer(int loading)
{
	SDL_SetCursor( loading ? hourglass : arrow );
}


/**
 * Some wrappers can save screenshots.
 * @return 1 on success, 0 if not implemented for a particular wrapper and -1
 *         in case of error.
 */
int dr_screenshot(const char *filename, int x, int y, int w, int h)
{
#ifdef WIN32
	if(  dr_screenshot_png( filename, w, h, screen->w, ((unsigned short *)(screen->pixels)) + x + y * screen->w, screen->format->BitsPerPixel )  ) {
		return 1;
	}
#endif
	(void)x; (void)y; (void)w; (void)h;
	return SDL_SaveBMP( screen, filename ) == 0 ? 1 : -1;
}


/*
 * Hier sind die Funktionen zur Messageverarbeitung
 */


static inline unsigned int ModifierKeys()
{
	SDL_Keymod mod = SDL_GetModState();

	return
			(mod & KMOD_SHIFT ? 1 : 0)
			| (mod & KMOD_CTRL ? 2 : 0)
#ifdef __APPLE__
		// Treat the Command key as a control key.
		| (mod & KMOD_GUI ? 2 : 0)
#endif
			;
}


static int conv_mouse_buttons(Uint8 const state)
{
	return
			(state & SDL_BUTTON_LMASK ? MOUSE_LEFTBUTTON  : 0) |
			(state & SDL_BUTTON_MMASK ? MOUSE_MIDBUTTON   : 0) |
			(state & SDL_BUTTON_RMASK ? MOUSE_RIGHTBUTTON : 0);
}


static void internal_GetEvents(bool const wait)
{
	// Apparently Cocoa SDL posts key events that meant to be used by IM...
	// Ignoring SDL_KEYDOWN during preedit seems to work fine.
	static bool composition_is_underway = false;
	static bool ignore_previous_number = false;

	SDL_Event event;
	event.type = 1;
	if(  wait  ) {
		int n;
		do {
			SDL_WaitEvent( &event );
			n = SDL_PollEvent( NULL );
		} while(  n != 0  &&  event.type == SDL_MOUSEMOTION  );
	}
	else {
		int n;
		bool got_one = false;
		do {
			n = SDL_PollEvent( &event );
			if(  n != 0  ) {
				got_one = true;
				if(  event.type == SDL_MOUSEMOTION  ) {
					sys_event.mx   = SCREEN_TO_TEX_X(event.motion.x);
					sys_event.my   = SCREEN_TO_TEX_Y(event.motion.y);
					sys_event.type = SIM_MOUSE_MOVE;
					sys_event.code = SIM_MOUSE_MOVED;
					sys_event.mb   = conv_mouse_buttons( event.motion.state );
				}
			}
		} while(  n != 0  &&  event.type == SDL_MOUSEMOTION  );
		if(  !got_one  ) {
			return;
		}
	}

	static char textinput[SDL_TEXTINPUTEVENT_TEXT_SIZE];
	switch(  event.type  ) {
		case SDL_WINDOWEVENT: {
			if(  event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED  ) {
				sys_event.new_window_size.w = SCREEN_TO_TEX_X(event.window.data1);
				sys_event.new_window_size.h = SCREEN_TO_TEX_Y(event.window.data2);
				sys_event.type = SIM_SYSTEM;
				sys_event.code = SYSTEM_RESIZE;
			}
			// Ignore other window events.
			break;
		}
		case SDL_MOUSEBUTTONDOWN: {
			sys_event.type    = SIM_MOUSE_BUTTONS;
			switch(  event.button.button  ) {
				case SDL_BUTTON_LEFT:   sys_event.code = SIM_MOUSE_LEFTBUTTON;  break;
				case SDL_BUTTON_MIDDLE: sys_event.code = SIM_MOUSE_MIDBUTTON;   break;
				case SDL_BUTTON_RIGHT:  sys_event.code = SIM_MOUSE_RIGHTBUTTON; break;
				case SDL_BUTTON_X1:     sys_event.code = SIM_MOUSE_WHEELUP;     break;
				case SDL_BUTTON_X2:     sys_event.code = SIM_MOUSE_WHEELDOWN;   break;
			}
			sys_event.mx      = SCREEN_TO_TEX_X(event.button.x);
			sys_event.my      = SCREEN_TO_TEX_Y(event.button.y);
			sys_event.mb      = conv_mouse_buttons( SDL_GetMouseState(0, 0) );
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_MOUSEBUTTONUP: {
			sys_event.type    = SIM_MOUSE_BUTTONS;
			switch(  event.button.button  ) {
				case SDL_BUTTON_LEFT:   sys_event.code = SIM_MOUSE_LEFTUP;  break;
				case SDL_BUTTON_MIDDLE: sys_event.code = SIM_MOUSE_MIDUP;   break;
				case SDL_BUTTON_RIGHT:  sys_event.code = SIM_MOUSE_RIGHTUP; break;
			}
			sys_event.mx      = SCREEN_TO_TEX_X(event.button.x);
			sys_event.my      = SCREEN_TO_TEX_Y(event.button.y);
			sys_event.mb      = conv_mouse_buttons( SDL_GetMouseState(0, 0) );
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_MOUSEWHEEL: {
			sys_event.type    = SIM_MOUSE_BUTTONS;
			sys_event.code    = event.wheel.y > 0 ? SIM_MOUSE_WHEELUP : SIM_MOUSE_WHEELDOWN;
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_KEYDOWN: {
			// Hack: when 2 byte character composition is under way, we have to leave the key processing with the IME
			// BUT: if not, we have to do it ourselves, or the cursor or return will not be recognised
			if(  composition_is_underway  ) {
				if(  gui_component_t *c = win_get_focus()  ) {
					if(  gui_textinput_t *tinp = dynamic_cast<gui_textinput_t *>(c)  ) {
						if(  tinp->get_composition()[0]  ) {
							// pending string, handled by IME
							break;
						}
					}
				}
			}

			unsigned long code;
#ifdef _WIN32
			// SDL doesn't set numlock state correctly on startup. Revert to win32 function as workaround.
			const bool key_numlock = ((GetKeyState( VK_NUMLOCK ) & 1) != 0);
#else
			const bool key_numlock = (SDL_GetModState() & KMOD_NUM);
#endif
			const bool numlock = key_numlock  &&  !env_t::numpad_always_moves_map;
			sys_event.key_mod = ModifierKeys();
			SDL_Keycode sym = event.key.keysym.sym;
			bool np = false; // to indicate we converted a numpad key

			switch(  sym  ) {
				case SDLK_BACKSPACE:  code = SIM_KEY_BACKSPACE;             break;
				case SDLK_TAB:        code = SIM_KEY_TAB;                   break;
				case SDLK_RETURN:     code = SIM_KEY_ENTER;                 break;
				case SDLK_ESCAPE:     code = SIM_KEY_ESCAPE;                break;
				case SDLK_DELETE:     code = SIM_KEY_DELETE;                break;
				case SDLK_DOWN:       code = SIM_KEY_DOWN;                  break;
				case SDLK_END:        code = SIM_KEY_END;                   break;
				case SDLK_HOME:       code = SIM_KEY_HOME;                  break;
				case SDLK_F1:         code = SIM_KEY_F1;                    break;
				case SDLK_F2:         code = SIM_KEY_F2;                    break;
				case SDLK_F3:         code = SIM_KEY_F3;                    break;
				case SDLK_F4:         code = SIM_KEY_F4;                    break;
				case SDLK_F5:         code = SIM_KEY_F5;                    break;
				case SDLK_F6:         code = SIM_KEY_F6;                    break;
				case SDLK_F7:         code = SIM_KEY_F7;                    break;
				case SDLK_F8:         code = SIM_KEY_F8;                    break;
				case SDLK_F9:         code = SIM_KEY_F9;                    break;
				case SDLK_F10:        code = SIM_KEY_F10;                   break;
				case SDLK_F11:        code = SIM_KEY_F11;                   break;
				case SDLK_F12:        code = SIM_KEY_F12;                   break;
				case SDLK_F13:        code = SIM_KEY_F13;                   break;
				case SDLK_F14:        code = SIM_KEY_F14;                   break;
				case SDLK_F15:        code = SIM_KEY_F15;                   break;
				case SDLK_KP_0:       np = true; code = numlock ? '0' : SIM_KEY_NUMPAD_BASE + 0; break;
				case SDLK_KP_1:       np = true; code = numlock ? '1' : SIM_KEY_NUMPAD_BASE+1; break;
				case SDLK_KP_2:       np = true; code = numlock ? '2' : SIM_KEY_NUMPAD_BASE+2; break;
				case SDLK_KP_3:       np = true; code = numlock ? '3' : SIM_KEY_NUMPAD_BASE+3; break;
				case SDLK_KP_4:       np = true; code = numlock ? '4' : SIM_KEY_NUMPAD_BASE+4; break;
				case SDLK_KP_5:       np = true; code = numlock ? '5' : SIM_KEY_NUMPAD_BASE+5; break;
				case SDLK_KP_6:       np = true; code = numlock ? '6' : SIM_KEY_NUMPAD_BASE+6; break;
				case SDLK_KP_7:       np = true; code = numlock ? '7' : SIM_KEY_NUMPAD_BASE+7; break;
				case SDLK_KP_8:       np = true; code = numlock ? '8' : SIM_KEY_NUMPAD_BASE+8; break;
				case SDLK_KP_9:       np = true; code = numlock ? '9' : SIM_KEY_NUMPAD_BASE+9; break;
				case SDLK_KP_ENTER:   code = SIM_KEY_ENTER;                 break;
				case SDLK_LEFT:       code = SIM_KEY_LEFT;                  break;
				case SDLK_PAGEDOWN:   code = '<';                           break;
				case SDLK_PAGEUP:     code = '>';                           break;
				case SDLK_RIGHT:      code = SIM_KEY_RIGHT;                 break;
				case SDLK_UP:         code = SIM_KEY_UP;                    break;
				case SDLK_PAUSE:      code = SIM_KEY_PAUSE;                 break;
				default: {
					// Handle CTRL-keys. SDL_TEXTINPUT event handles regular input
					if(  (sys_event.key_mod & 2)  &&  SDLK_a <= sym  &&  sym <= SDLK_z  ) {
						code = event.key.keysym.sym & 31;
					}
					else {
						code = 0;
					}
					break;
				}
			}
			ignore_previous_number = (np  &&   key_numlock  &&  env_t::numpad_always_moves_map);
			sys_event.type    = SIM_KEYBOARD;
			sys_event.code    = code;
			break;
		}

		case SDL_TEXTINPUT: {
			size_t in_pos = 0;
			utf32 uc = utf8_decoder_t::decode((utf8 const*)event.text.text, in_pos);
			if(  event.text.text[in_pos]==0  ) {
				// single character
				if( ignore_previous_number ) {
					ignore_previous_number = false;
					break;
				}
				sys_event.type    = SIM_KEYBOARD;
				sys_event.code    = (unsigned long)uc;
			}
			else {
				// string
				strcpy( textinput, event.text.text );
				sys_event.type    = SIM_STRING;
				sys_event.ptr     = (void*)textinput;
			}
			sys_event.key_mod = ModifierKeys();
			composition_is_underway = false;
			break;
		}
#ifdef USE_SDL_TEXTEDITING
			case SDL_TEXTEDITING: {
			//printf( "SDL_TEXTEDITING {timestamp=%d, \"%s\", start=%d, length=%d}\n", event.edit.timestamp, event.edit.text, event.edit.start, event.edit.length );
			strcpy( textinput, event.edit.text );
			if(  !textinput[0]  ) {
				composition_is_underway = false;
			}
			int i = 0;
			int start = 0;
			for(  ; i<event.edit.start; ++i  ) {
				start = utf8_get_next_char( (utf8 *)event.edit.text, start );
			}
			int end = start;
			for(  ; i<event.edit.start+event.edit.length; ++i  ) {
				end = utf8_get_next_char( (utf8*)event.edit.text, end );
			}

			if(  gui_component_t *c = win_get_focus()  ) {
				if(  gui_textinput_t *tinp = dynamic_cast<gui_textinput_t *>(c)  ) {
					tinp->set_composition_status( textinput, start, end-start );
				}
			}
			composition_is_underway = true;
			break;
		}
#endif
		case SDL_MOUSEMOTION: {
			sys_event.type    = SIM_MOUSE_MOVE;
			sys_event.code    = SIM_MOUSE_MOVED;
			sys_event.mx      = SCREEN_TO_TEX_X(event.motion.x);
			sys_event.my      = SCREEN_TO_TEX_Y(event.motion.y);
			sys_event.mb      = conv_mouse_buttons( event.motion.state );
			sys_event.key_mod = ModifierKeys();
			break;
		}
		case SDL_KEYUP: {
			sys_event.type = SIM_KEYBOARD;
			sys_event.code = 0;
			break;
		}
		case SDL_QUIT: {
			sys_event.type = SIM_SYSTEM;
			sys_event.code = SYSTEM_QUIT;
			break;
		}
		default: {
			sys_event.type = SIM_IGNORE_EVENT;
			sys_event.code = 0;
			break;
		}
	}
}


void GetEvents()
{
	internal_GetEvents( true );
}


void GetEventsNoWait()
{
	sys_event.type = SIM_NOEVENT;
	sys_event.code = 0;

	internal_GetEvents( false );
}


void show_pointer(int yesno)
{
	SDL_SetCursor( (yesno != 0) ? arrow : blank );
}


void ex_ord_update_mx_my()
{
	SDL_PumpEvents();
}


uint32 dr_time()
{
	return SDL_GetTicks();
}


void dr_sleep(uint32 usec)
{
	SDL_Delay( usec );
}


void dr_start_textinput()
{
	if(  env_t::hide_keyboard  ) {
		SDL_StartTextInput();
	}
}


void dr_stop_textinput()
{
	if(  env_t::hide_keyboard  ) {
		SDL_StopTextInput();
	}
}

void dr_notify_input_pos(int x, int y)
{
	SDL_Rect rect = { TEX_TO_SCREEN_X(x), TEX_TO_SCREEN_Y(y + LINESPACE), 1, 1};
	SDL_SetTextInputRect( &rect );
}

#ifdef _MSC_VER
// Needed for MS Visual C++ with /SUBSYSTEM:CONSOLE to work , if /SUBSYSTEM:WINDOWS this function is compiled but unreachable
#undef main
int main()
{
	return WinMain(NULL,NULL,NULL,NULL);
}
#endif


#ifdef _WIN32
int CALLBACK WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
#else
int main(int argc, char **argv)
#endif
{
#ifdef _WIN32
	int    const argc = __argc;
	char** const argv = __argv;
#endif
	return sysmain(argc, argv);
}
