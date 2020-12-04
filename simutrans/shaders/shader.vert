#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPos;


void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    // gl_Position = vec4(inPosition, 1.0);
    // gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
    // fragColor = inColor * dot(lightPos, inNormal);
    fragNormal = inNormal;
    fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
}
