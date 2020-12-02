/*
 * This file is part of the Simutrans-Extended project under the Artistic License.
 * (see LICENSE.txt)
 */


#include "simsys.h"
#include "simsys_s2_vulkan.h"

#include "../simversion.h"
#include "../simdebug.h"

#include <sys/time.h>
#include <csignal>
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <set>


#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

int display_error_message(const char *title, const char *message) {
	return SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, title, message, nullptr);
}

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
static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, SDL_Window *window);

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


sim_window_t::sim_window_t(scr_size window_size, bool _fullscreen)
		: width(window_size.w), height(window_size.h), fullscreen(_fullscreen) {

	if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
		display_error_message("SDL_Init", SDL_GetError());
		throw std::runtime_error(SDL_GetError());
	}
};

void sim_window_t::run() {
	// (more or less) follow the vulkan-tutorial
	init_window();
	init_vulkan();
}

void sim_window_t::init_window() {
	Uint32 flags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN;

	flags |= fullscreen ? SDL_WINDOW_FULLSCREEN : SDL_WINDOW_RESIZABLE;

	window = SDL_CreateWindow(
			SIM_TITLE,
			SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
			width, height,
			flags );

	if(  window == NULL  ) {
		display_error_message("Could not open the window", SDL_GetError());
		dbg->error("sim_window_t::show SDL2_vulkan", "Could not open the window: %s", SDL_GetError() );
	}
}

void sim_window_t::cleanup() {

	vkDestroyDevice(device, nullptr);

	if (enableValidationLayers) {
		destroy_debug_utils_messenger_ext(instance, debugMessenger, nullptr);
	}

	vkDestroyInstance(instance, nullptr);
}

void sim_window_t::init_vulkan() {
	create_vulkan_instance();
	setup_debug_messenger();
	create_surface();
	pick_physical_device();
	create_logical_device();
	create_swap_chain();
}

void sim_window_t::create_vulkan_instance() {
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

void sim_window_t::setup_debug_messenger() {
	if (!enableValidationLayers) return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo;
	populate_debug_messenger_create_info(createInfo);

	if (create_debug_utils_messenger_ext(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger!");
	}
}


void sim_window_t::create_surface() {
	if (!SDL_Vulkan_CreateSurface(window, instance, &surface)) {
		throw std::runtime_error("failed to create window surface");
	}
}


// TODO: pick device based on something (e.g. RAM or user selected) instead of taking the first one
void sim_window_t::pick_physical_device() {
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

void sim_window_t::create_logical_device() {
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

void sim_window_t::create_swap_chain() {
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

static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, SDL_Window *window) {
	if (capabilities.currentExtent.width != UINT32_MAX) {
		return capabilities.currentExtent;
	} else {
		int width, height;
		SDL_Vulkan_GetDrawableSize(window, &width, &height);

		VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
		};

		actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}

////////////////////////////////////////////


bool dr_auto_scale(bool)
{
	return false;
}

bool dr_os_init(const int*)
{
	return true;
}

resolution dr_query_screen_resolution()
{
	resolution const res = { 0, 0 };
	return res;
}



void dr_os_close()
{
}

// resizes screen
int dr_textur_resize(unsigned short** const textur, int, int)
{
	*textur = NULL;
	return 1;
}


unsigned short *dr_textur_init()
{
	return NULL;
}

unsigned int get_system_color(unsigned int, unsigned int, unsigned int)
{
	return 1;
}

void dr_prepare_flush()
{
}

void dr_flush()
{
}

void dr_textur(int, int, int, int)
{
}

void move_pointer(int, int)
{
}

void set_pointer(int)
{
}

int dr_screenshot(const char *,int,int,int,int)
{
	return -1;
}

void GetEvents()
{
}


void GetEventsNoWait()
{
}

void show_pointer(int)
{
}


void ex_ord_update_mx_my()
{
}

#ifndef _MSC_VER
static timeval first;
#endif

uint32 dr_time()
{
#ifndef _MSC_VER
	timeval second;
	gettimeofday(&second,NULL);
	if (first.tv_usec > second.tv_usec) {
		// since those are often unsigned
		second.tv_usec += 1000000;
		second.tv_sec--;
	}

	return (second.tv_sec - first.tv_sec)*1000ul + (second.tv_usec - first.tv_usec)/1000ul;
#else
	return timeGetTime();
#endif
}

void dr_sleep(uint32 msec)
{
/*
	// this would be 100% POSIX but is usually not very accurate ...
	if(  msec>0  ) {
		struct timeval tv;
		tv.sec = 0;
		tv.usec = msec*1000;
		select(0, 0, 0, 0, &tv);
	}
*/
#ifdef _WIN32
	Sleep( msec );
#else
	usleep( msec * 1000u );
#endif
}

void dr_start_textinput()
{
}

void dr_stop_textinput()
{
}

void dr_notify_input_pos(int, int)
{
}

static void posix_sigterm(int)
{
}

int main(int argc, char **argv) {
	signal( SIGTERM, posix_sigterm );
#ifndef _MSC_VER
	gettimeofday(&first,NULL);
#endif
	return sysmain(argc, argv);
}