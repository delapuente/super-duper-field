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
uniform vec2 resolution;
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

const float SUPPORT_p = 0.03;

float supportHoles(in vec3 p) {
  mat4 l = translate(vec3(-0.25, 0, 0));
  mat4 r = translate(vec3( 0.25, 0, 0));
  return min(
    sdRoundBox(
      (inverse(l) * vec4(p, 1)).xyz,
      vec3(0.10, 0.04, SUPPORT_p),
      0.02
    ),
    sdRoundBox(
      (inverse(r) * vec4(p, 1.0)).xyz,
      vec3(0.10, 0.04, SUPPORT_p),
      0.02
    )
  );
}

float supportMain(in vec3 p) {
  return sdBox(
    p,
    vec3(0.25, 0.3, SUPPORT_p)
  );
}

float chipSupport(in vec3 p) {
  return max(
     supportMain(p),
    -supportHoles(p)
  );
}

float chipPlate(in vec3 p) {
  const float oz = -(SUPPORT_p/2.0 + SUPPORT_p/4.0);
  mat4 f = translate(vec3(0, 0, oz));
  return sdBox(
    (inverse(f) * vec4(p, 1.0)).xyz,
    vec3(0.125, 0.3, SUPPORT_p/2.0)
  );
}

float wirePlates(in vec3 p) {
  const float oz = -SUPPORT_p;
  mat4 t = translate(vec3(0,  0.19, oz));
  mat4 b = translate(vec3(0, -0.19, oz));
  return min(
    sdBox(
      (inverse(t) * vec4(p, 1.0)).xyz,
      vec3(0.25, 0.11, SUPPORT_p)
    ),
    sdBox(
      (inverse(b) * vec4(p, 1.0)).xyz,
      vec3(0.25, 0.11, SUPPORT_p)
    )
  );
}

float map(in vec3 p) {
  return min(
    chipSupport(p),
    min(
      chipPlate(p),
      wirePlates(p)
    )
  );
}

vec3 normal(in vec3 p) {
  const vec3 epsilon = vec3(0.001, 0.0, 0.0);
  float gradient_x = map(p.xyz + epsilon.xyy) - map(p.xyz - epsilon.xyy);
  float gradient_y = map(p.xyz + epsilon.yxy) - map(p.xyz - epsilon.yxy);
  float gradient_z = map(p.xyz + epsilon.yyx) - map(p.xyz - epsilon.yyx);
  vec3 gradient = vec3(gradient_x, gradient_y, gradient_z);
  return normalize(gradient);
}

const int MAX_STEPS = 256;
const float HIT_THRESHOLD = 0.0001;
const float MAX_DISTANCE = 500.0;

vec4 light(in vec3 p, in vec3 color) {
  vec3 n = normal(p);
  vec3 lightPos = vec3(2.0, -5.0, 3.0);
  vec3 lightRay = normalize(p - lightPos);
  float diffuse = max(0.0, dot(n, lightRay));
  return vec4(color * diffuse, 1);
}

vec4 intersect(in vec3 ro, in vec3 rd) {
  float traveled = 0.0;
  for (int i = 0; i < MAX_STEPS; ++i) {
    vec3 p = ro + rd * traveled;
    float distance = map(p);
    if (distance < HIT_THRESHOLD) {
      return light(p, vec3(1, 0, 1));
    }
    if (traveled > MAX_DISTANCE) {
      break;
    }
    traveled += distance;
  }
  return vec4(0);
}

void main() {
  vec2 uv = gl_FragCoord.xy/resolution.y * 2.0 - 1.0;
  vec4 ro = cameraMatrix * vec4(0, 0, 0, 1);
  vec4 np = cameraMatrix * vec4(uv, 10, 1); // pos in near plane
  vec4 rd = normalize(np - ro);
  vec4 shaded_color = intersect(ro.xyz, rd.xyz);
  outColor = shaded_color;
}
`;

main ()

function main () {
  // Create canvas, set resolution and get WebGL context
  const resolution = [840, 472]
  const canvas = document.createElement('canvas')
  canvas.width = resolution[0]
  canvas.height = resolution[1]
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
  const resolutionLocation = gl.getUniformLocation(program, 'resolution')

  // Draw the data
  requestAnimationFrame(function _frame(t) {
    const cameraMatrix = m4.yRotation(Math.sin(t/2000))//m4.identity()
    m4.translate(cameraMatrix, 0, 0, -4, cameraMatrix)
    console.log(cameraMatrix)
    gl.uniform1f(tLocation, t)
    gl.uniformMatrix4fv(cameraMatrixLocation, false, cameraMatrix)
    gl.uniform2fv(resolutionLocation, resolution)
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