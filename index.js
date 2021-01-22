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

float det(vec3 a, vec3 b, in vec3 v) {
  return dot(v,cross(a,b));
}

vec4 sdBezier(vec3 p, vec3 b0, vec3 b1, vec3 b2) {
  b0 -= p;
  b1 -= p;
  b2 -= p;

  vec3  d21 = b2-b1;
  vec3  d10 = b1-b0;
  vec3  d20 = (b2-b0)*0.5;

  vec3  n = normalize(cross(d10,d21));

  float a = det(b0,b2,n);
  float b = det(b1,b0,n);
  float d = det(b2,b1,n);
  vec3  g = b*d21 + d*d10 + a*d20;
  float f = a*a*0.25-b*d;

  vec3  z = cross(b0,n) + f*g/dot(g,g);
  float t = clamp( dot(z,d10-d20)/(a+b+d), 0.0 ,1.0);
  vec3 q = mix(mix(b0,b1,t), mix(b1,b2,t),t);

  float k = dot(q,n);
  return vec4(length(q),t,-k,-sign(f)*length(q-n*k));
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

float sdHexPrism(vec3 p, vec2 h) {
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
  vec2 d = vec2(
       length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

vec2 opU(vec2 d) {
  return d;
}

vec2 opU(vec2 d1, vec2 d2) {
  if (d1.x <= d2.x) return d1;
  return d2;
}

float opU(float d1, float d2) {
  return min(d1, d2);
}

float opU(float d) {
  return d;
}

const float SUPPORT_DEPTH = 0.03;

const float MAT_COPPER = 1.0;
const float MAT_METAL = 2.0;
const float MAT_GOLD = 3.0;

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

vec2 chipSupport(in vec3 p) {
  return vec2(max(
     supportMain(p),
    -supportHoles(p)
  ), MAT_COPPER);
}

vec2 chipPlate(in vec3 p) {
  const float oz = -(SUPPORT_DEPTH/2.0 + SUPPORT_DEPTH/4.0);
  mat4 f = translate(vec3(0, 0, oz));
  return vec2(
    sdBox(
      (inverse(f) * vec4(p, 1.0)).xyz,
      vec3(0.125, 0.3, SUPPORT_DEPTH/2.0)
    ),
    MAT_COPPER
  );
}

vec2 wirePlates(in vec3 p) {
  const float oz = -SUPPORT_DEPTH;
  mat4 t = translate(vec3(0,  0.19, oz));
  mat4 b = translate(vec3(0, -0.19, oz));
  return vec2(min(
    sdBox(
      (inverse(t) * vec4(p, 1.0)).xyz,
      vec3(0.25, 0.11, SUPPORT_DEPTH)
    ),
    sdBox(
      (inverse(b) * vec4(p, 1.0)).xyz,
      vec3(0.25, 0.11, SUPPORT_DEPTH)
    )
  ), MAT_METAL);
}

float honeyCombRows(vec3 p) {
  vec3 c = vec3(0.08, -0.042, 0);
  vec3 r = vec3(2, 3, 2);
  vec3 q = p-c*clamp(round(p/c),vec3(0),r);
  return sdHexPrism(q, vec2(0.02, SUPPORT_DEPTH/2.0));
}

float honeyComb(vec3 p) {
  mat4 even = translate(vec3(0.04, -0.02, 0.0));
  return min(
    honeyCombRows(p),
    honeyCombRows((inverse(even) * vec4(p, 1.0)).xyz)
  );
}

vec2 wireConnectors(vec3 p) {
  const float oz = -2.5*SUPPORT_DEPTH;
  mat4 t = translate(vec3(-0.1,  0.26, oz));
  mat4 b = translate(vec3(-0.1, -0.11, oz));
  return vec2(min(
    honeyComb((inverse(t) * vec4(p, 1.0)).xyz),
    honeyComb((inverse(b) * vec4(p, 1.0)).xyz)
  ), MAT_GOLD);
}

float wireBunchRows(vec3 p, float rowIndex, bool facingUp) {
  vec3 c = vec3(0.08, 0, 0);
  vec3 r = vec3(2, 1, 1);
  vec3 q = p-c*clamp(round(p/c),vec3(0),r);
  rowIndex = facingUp ? rowIndex : 3.0 - rowIndex;
  return sdBezier(
    q,
    vec3(0, 0, 0),
    vec3(0, 0, -0.10*(rowIndex+1.0)),
    vec3(0, facingUp ? 1 : -1, 0)
  ).x - 0.01;
}

float wireBunch(vec3 p, bool facingUp) {
  mat4 even = translate(vec3(0.04, -0.02, 0.0));
  return min(
    opU(
      wireBunchRows(p, 0.0, facingUp),
      wireBunchRows((inverse(even) * vec4(p, 1.0)).xyz, 0.0, facingUp)
    ),
    opU(
      opU(
        wireBunchRows(p + vec3(0, 0.04, 0), 1.0, facingUp),
        wireBunchRows((inverse(even) * (vec4(p, 1.0) + vec4(0, 0.04, 0, 0))).xyz, 1.0, facingUp)
      ),
      opU(
        opU(
          wireBunchRows(p + vec3(0, 0.08, 0), 2.0, facingUp),
          wireBunchRows((inverse(even) * (vec4(p, 1.0) + vec4(0, 0.08, 0, 0))).xyz, 2.0, facingUp)
        ),
        opU(
          wireBunchRows(p + vec3(0, 0.12, 0), 3.0, facingUp),
          wireBunchRows((inverse(even) * (vec4(p, 1.0) + vec4(0, 0.12, 0, 0))).xyz, 3.0, facingUp)
        )
      )
    )
  );
}

vec2 wires(vec3 p) {
  const float oz = -2.5*SUPPORT_DEPTH;
  mat4 t = translate(vec3(-0.1,  0.26, oz));
  mat4 b = translate(vec3(-0.1, -0.11, oz));
  return vec2(
  opU(
    wireBunch((inverse(t) * vec4(p, 1.0)).xyz, true),
    wireBunch((inverse(b) * vec4(p, 1.0)).xyz, false)
  )
  , MAT_METAL);
}

vec2 map(in vec3 p) {
  return opU(
    chipSupport(p),
    opU(
      opU(
        wirePlates(p),
        opU(
          wires(p),
          wireConnectors(p)
        )
      ),
      chipPlate(p)
    )
  );
}

vec3 normal(in vec3 p) {
  const vec3 epsilon = vec3(0.001, 0.0, 0.0);
  float gradient_x = map(p.xyz + epsilon.xyy).x - map(p.xyz - epsilon.xyy).x;
  float gradient_y = map(p.xyz + epsilon.yxy).x - map(p.xyz - epsilon.yxy).x;
  float gradient_z = map(p.xyz + epsilon.yyx).x - map(p.xyz - epsilon.yyx).x;
  vec3 gradient = vec3(gradient_x, gradient_y, gradient_z);
  return normalize(gradient);
}

const int MAX_STEPS = 256;
const float HIT_THRESHOLD = 0.0001;
const float MAX_DISTANCE = 500.0;

float apprSoftShadow(vec3 ro, vec3 rd, float mint, float tmax, float w) {
  float shadow = 1.0;
  for (float t=mint; t<tmax; ) {
    float h = map(ro + t*rd).x;
    if (h < HIT_THRESHOLD) return 0.0;
    shadow = min(shadow, w*h/t);
    t += h;
  }
  return shadow;
}

float ambientOcclusion(in vec3 p, in vec3 n) {
	float occlusion = 0.0;
  float sca = 1.0;
  for(int i=0; i<5; i++) {
    float h = 0.001 + 0.15*float(i)/4.0;
    float d = map(p + h*n).x;
    occlusion += (h-d)*sca;
    sca *= 0.95;
  }
  return clamp(1.0 - 1.5*occlusion, 0.0, 1.0);
}

vec4 light(vec3 p, vec3 rd, float matId) {
  vec3 color = vec3(1, 0, 1);
  if (matId == MAT_COPPER) { color = vec3(0.7, 0.2, 0); }
  else if (matId == MAT_METAL) { color = vec3(0.5, 0.5, 0.5); }
  else if (matId == MAT_GOLD) { color = vec3(0.8, 0.7, 0); }

  vec3 n = normal(p);
  vec3 lightPos = normalize(vec3(1.0, 1.5, -2.0));
  vec3 lightRay = normalize(lightPos - rd);

  float shadow = apprSoftShadow(p, lightPos, 0.01, 3.0, 32.0);
  float diffuse = clamp(dot(n, lightPos), 0.0, 1.0) * shadow;
  float specular = pow(clamp(dot(n, lightRay), 0.0, 1.0), 32.0);
  float occlusion = ambientOcclusion(p, n);
  float ambient = 0.5 + 0.5*n.y;

  vec3 res = color * diffuse +
             specular * diffuse * 7.0 * vec3(0.90,0.80,1.0) +
             color * ambient * occlusion * vec3(0.05,0.1,0.15);

  return vec4(pow(res, vec3(0.4545)), 1);
}

vec4 intersect(in vec3 ro, in vec3 rd) {
  float traveled = 0.0;
  for (int i = 0; i < MAX_STEPS; ++i) {
    vec3 p = ro + rd * traveled;
    vec2 result = map(p);
    float distance = result.x;
    float matId = result.y;
    if (distance < HIT_THRESHOLD) {
      return light(p, rd, matId);
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
  vec4 np = cameraMatrix * vec4(uv, 7, 1); // pos in near plane
  vec4 rd = normalize(np - ro);
  vec4 shaded_color = intersect(ro.xyz, rd.xyz);
  outColor = shaded_color;
}
`;

main ()

function main () {
  // Create canvas, set resolution and get WebGL context
  const resolution = [1600, 800]
  const scaleFactor = 1
  const canvas = document.createElement('canvas')
  canvas.style.width = `${scaleFactor * resolution[0]}px`
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