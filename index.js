const glsl = x => x
const frag = vert = glsl

var vertexShaderSource = vert`#version 300 es
#pragma vscode_glsllint_stage : vert
in vec4 a_position;
void main() {
  gl_Position = a_position;
}
`;

var fragmentShaderSource = frag`#version 300 es
#pragma vscode_glsllint_stage : frag
precision highp float;
uniform float t;
uniform mat4 cameraMatrix;
out vec4 outColor;

mat4 translate(in vec3 o) {
  return mat4(
    vec4(1, 0, 0, 0),
    vec4(0, 1, 0, 0),
    vec4(0, 0, 1, 0),
    vec4(o,       1)
  );
}

float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p.xyz) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y,q.z)), 0.0);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

const float SUPPORT_DEPTH = 0.03;

float supportHoles(in vec3 p) {
  mat4 l = translate(vec3(-0.25, 0, 0));
  mat4 r = translate(vec3( 0.25, 0, 0));
  return min(
    sdRoundBox(
      (inverse(l) * vec4(p, 1)).xyz,
      vec3(0.10, 0.04, SUPPORT_DEPTH),
      0.02
    ),
    sdRoundBox(
      (inverse(r) * vec4(p, 1.0)).xyz,
      vec3(0.10, 0.04, SUPPORT_DEPTH),
      0.02
    )
  );
}

float supportMain(in vec3 p) {
  return sdBox(
    p,
    vec3(0.25, 0.3, SUPPORT_DEPTH)
  );
}

float chipSupport(in vec3 p) {
  return max(
     supportMain(p),
    -supportHoles(p)
  );
}

float chipPlate(in vec3 p) {
  const float oz = -(SUPPORT_DEPTH/2.0 + SUPPORT_DEPTH/4.0);
  mat4 f = translate(vec3(0, 0, oz));
  return sdBox(
    (inverse(f) * vec4(p, 1.0)).xyz,
    vec3(0.125, 0.3, SUPPORT_DEPTH/2.0)
  );
}

float wirePlates(in vec3 p) {
  const float oz = -SUPPORT_DEPTH;
  mat4 t = translate(vec3(0,  0.19, oz));
  mat4 b = translate(vec3(0, -0.19, oz));
  return min(
    sdBox(
      (inverse(t) * vec4(p, 1.0)).xyz,
      vec3(0.25, 0.11, SUPPORT_DEPTH)
    ),
    sdBox(
      (inverse(b) * vec4(p, 1.0)).xyz,
      vec3(0.25, 0.11, SUPPORT_DEPTH)
    )
  );
}

float sdWorld(in vec3 p) {
  return min(
    chipSupport(p),
    min(
      chipPlate(p),
      wirePlates(p)
    )
  );
}

vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = sdWorld(p.xyz + small_step.xyy) - sdWorld(p.xyz - small_step.xyy);
    float gradient_y = sdWorld(p.xyz + small_step.yxy) - sdWorld(p.xyz - small_step.yxy);
    float gradient_z = sdWorld(p.xyz + small_step.yyx) - sdWorld(p.xyz - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

vec4 ray_march(in vec3 ro, in vec3 rd) {
  float total_distance_traveled = 0.0;
  const int NUMBER_OF_STEPS = 256;
  const float MINIMUM_HIT_DISTANCE = 0.0001;
  const float MAXIMUM_TRACE_DISTANCE = 500.0;

  for (int i = 0; i < NUMBER_OF_STEPS; ++i) {
    vec3 current_position = ro + total_distance_traveled * rd;
    float distance_to_closest = sdWorld(current_position);

    // hit
    if (distance_to_closest < MINIMUM_HIT_DISTANCE) {
      vec3 normal = calculate_normal(current_position);
      vec3 light_position = vec3(2.0, -5.0, 3.0);
      vec3 direction_to_light = normalize(current_position - light_position);

      float diffuse_intensity = max(0.0, dot(normal, direction_to_light));

      return vec4(vec3(1.0, 0.0, 0.0) * diffuse_intensity, 1);
    }

    // miss
    if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE) {
      break;
    }

    total_distance_traveled += distance_to_closest;
  }

  return vec4(0);
}

void main() {
  vec2 uv = gl_FragCoord.xy/vec2(472, 472) * 2.0 - 1.0;

  vec4 expected = vec4(uv, -2, 1);
  vec4 camera_position = cameraMatrix * vec4(0, 0, 0, 1);
  vec4 ro = camera_position;
  vec4 positionInNearPlane = cameraMatrix * vec4(uv, 10, 1);
  vec4 rd = normalize(positionInNearPlane - camera_position) / 1.0;

  vec4 shaded_color = ray_march(ro.xyz, rd.xyz);

  outColor = shaded_color;
}
`;

main ()

function main () {
  // Create canvas, set resolution and get WebGL context
  const canvas = document.createElement('canvas')
  canvas.width = 840
  canvas.height = 472
  document.body.appendChild(canvas)
  const gl = canvas.getContext('webgl2')
  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)

  // Compile and link vertex and fragment shaders into a program
  const vs = compileShader(gl, vertexShaderSource, gl.VERTEX_SHADER)
  const fs = compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER)
  const program = createProgram(gl, vs, fs)

  // Pass position data to a WebGL buffer
  const positionBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  const positions = [
    -1, -1, 0,
    -1,  1, 0,
     1, -1, 0,

    -1,  1, 0,
     1,  1, 0,
     1, -1, 0
  ]
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

  // Create the collection of data and make the buffer be part of it
  const vao = gl.createVertexArray()
  const positionAttrLocation = gl.getAttribLocation(program, 'a_position')
  gl.bindVertexArray(vao)
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  gl.enableVertexAttribArray(positionAttrLocation)
  // Configure the way the data is pulled from the buffer
  gl.vertexAttribPointer(positionAttrLocation, 3, gl.FLOAT, false, 0, 0)

  // Clear the scene
  gl.clearColor(0, 0, 0, 0)
  gl.clear(gl.COLOR_CLEAR_VALUE)

  gl.useProgram(program)
  gl.bindVertexArray(vao)

  // Get uniform location
  const tLocation = gl.getUniformLocation(program, 't')
  const cameraMatrixLocation = gl.getUniformLocation(program, 'cameraMatrix')

  // Draw the data
  requestAnimationFrame(function _frame(t) {
    const cameraMatrix = m4.yRotation(t/1000)//m4.identity()
    m4.translate(cameraMatrix, 0, 0, -4, cameraMatrix)
    console.log(cameraMatrix)
    gl.uniform1f(tLocation, t)
    gl.uniformMatrix4fv(cameraMatrixLocation, false, cameraMatrix)
    gl.drawArrays(gl.TRIANGLES, 0, 6)
    requestAnimationFrame(_frame)
  })

}

/**
 * Creates and compiles a shader.
 *
 * @param {!WebGLRenderingContext} gl The WebGL Context.
 * @param {string} shaderSource The GLSL source code for the shader.
 * @param {number} shaderType The type of shader, VERTEX_SHADER or
 *     FRAGMENT_SHADER.
 * @return {!WebGLShader} The shader.
 */
function compileShader(gl, shaderSource, shaderType) {
  const shader = gl.createShader(shaderType)
  gl.shaderSource(shader, shaderSource)
  gl.compileShader(shader)
  const success = gl.getShaderParameter(shader, gl.COMPILE_STATUS)
  if (!success) {
    gl.deleteShader(shader)
    throw 'could not compile shader:' + gl.getShaderInfoLog(shader)
  }

  return shader
}

/**
* Creates a program from 2 shaders.
*
* @param {!WebGLRenderingContext) gl The WebGL context.
* @param {!WebGLShader} vertexShader A vertex shader.
* @param {!WebGLShader} fragmentShader A fragment shader.
* @return {!WebGLProgram} A program.
*/
function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram()
  gl.attachShader(program, vertexShader)
  gl.attachShader(program, fragmentShader)
  gl.linkProgram(program);
  const success = gl.getProgramParameter(program, gl.LINK_STATUS)
  if (!success) {
    gl.deleteProgram(program)
    throw ('program filed to link:' + gl.getProgramInfoLog (program))
  }

  return program
}