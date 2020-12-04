#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;


layout(location = 0) out vec4 outColor;

vec3 lightPos = vec3(0.55, -0.3, 0.75);

void main() {
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = lightPos; // normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    // outColor = vec4(fragColor, 1.0) * dot(vec3(1.0), fragNormal);
    outColor = vec4(fragColor * diff, 1.0);
}
