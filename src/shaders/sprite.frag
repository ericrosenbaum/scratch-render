precision mediump float;

#ifdef DRAW_MODE_silhouette
uniform vec4 u_silhouetteColor;
#else // DRAW_MODE_silhouette
# ifdef ENABLE_color
uniform float u_color;
# endif // ENABLE_color
# ifdef ENABLE_brightness
uniform float u_brightness;
# endif // ENABLE_brightness
#endif // DRAW_MODE_silhouette

#ifdef DRAW_MODE_colorMask
uniform vec3 u_colorMask;
uniform float u_colorMaskTolerance;
#endif // DRAW_MODE_colorMask

#ifdef ENABLE_fisheye
uniform float u_fisheye;
#endif // ENABLE_fisheye
#ifdef ENABLE_whirl
uniform float u_whirl;
#endif // ENABLE_whirl
#ifdef ENABLE_pixelate
uniform float u_pixelate;
uniform vec2 u_skinSize;
#endif // ENABLE_pixelate
#ifdef ENABLE_mosaic
uniform float u_mosaic;
#endif // ENABLE_mosaic
#ifdef ENABLE_ghost
uniform float u_ghost;
#endif // ENABLE_ghost

#ifdef DRAW_MODE_line
uniform vec4 u_lineColor;
uniform float u_lineThickness;
uniform vec4 u_penPoints;
#endif // DRAW_MODE_line

uniform sampler2D u_skin;

varying vec2 v_texCoord;

// Add this to divisors to prevent division by 0, which results in NaNs propagating through calculations.
// Smaller values can cause problems on some mobile devices.
const float epsilon = 1e-3;

#if !defined(DRAW_MODE_silhouette) && (defined(ENABLE_color))

// LAB COLOR STUFF
const float eps = 216./24389.;
const float kap = 24389./27.;
const vec3  d65_2deg = vec3(0.95047,1.00000,1.08883);
const mat3 rgb2xyz_mat = 
    mat3(0.4124564,0.3575761,0.1804375,
         0.2126729,0.7151522,0.0721750,
         0.0193339,0.1191920,0.9503041);
const mat3 xyz2rgb_mat =
	mat3( 3.2404542, -1.5371385, -0.4985314,
		 -0.9692660,  1.8760108,  0.0415560,
	      0.0556434, -0.2040259,  1.0572252);

float compand(float f){
    return pow(f,2.2);//>0.04045? pow(((f+0.055f)/1.055f),2.4f): f/12.92f;
}
float invcompand(float f){
    return pow(f,1./2.2);
}

vec3 rgb2xyz(vec3 rgb){
	rgb.r = compand(rgb.r);
	rgb.g = compand(rgb.g);
	rgb.b = compand(rgb.b);
    return rgb*rgb2xyz_mat;
}
vec3 xyz2rgb(vec3 xyz){
    xyz *= xyz2rgb_mat;
	xyz.x = invcompand(xyz.x);
	xyz.y = invcompand(xyz.y);
	xyz.z = invcompand(xyz.z);
    return xyz;
}

vec3 xyz2lab(vec3 xyz){
    vec3 f;
    f.x = xyz.x>eps? pow(xyz.x,1./3.) : (kap*xyz.x+16.)/116.;
    f.y = xyz.y>eps? pow(xyz.y,1./3.) : (kap*xyz.y+16.)/116.;
    f.z = xyz.z>eps? pow(xyz.z,1./3.) : (kap*xyz.z+16.)/116.;
    return vec3(116.* f.y-16.,
                500.*(f.x-f.y),
                200.*(f.y-f.z))/vec3(100)/d65_2deg;
}

vec3 lab2xyz(vec3 lab){
    lab *= 100.;
    lab *= d65_2deg;
    float fy = (lab.x+16.)/116.;
    float fx = lab.y/500. + fy;
    float fz = fy - (lab.z/200.);
    float fx3 = pow(fx,3.);
    float fz3 = pow(fz,3.);
    return vec3(
    	fx3>eps? fx3: (116.*fx-16.)/kap,
        lab.x > (kap*eps)? pow((lab.x+16.)/116.,3.): lab.x/kap,
        fz3>eps? fz3: (116.*fz-16.)/kap);
}


// Branchless color conversions based on code from:
// http://www.chilliant.com/rgb2hsv.html by Ian Taylor
// Based in part on work by Sam Hocevar and Emil Persson
// See also: https://en.wikipedia.org/wiki/HSL_and_HSV#Formal_derivation


// Convert an RGB color to Hue, Saturation, and Value.
// All components of input and output are expected to be in the [0,1] range.
vec3 convertRGB2HSV(vec3 rgb)
{
	// Hue calculation has 3 cases, depending on which RGB component is largest, and one of those cases involves a "mod"
	// operation. In order to avoid that "mod" we split the M==R case in two: one for G<B and one for B>G. The B>G case
	// will be calculated in the negative and fed through abs() in the hue calculation at the end.
	// See also: https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
	const vec4 hueOffsets = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);

	// temp1.xy = sort B & G (largest first)
	// temp1.z = the hue offset we'll use if it turns out that R is the largest component (M==R)
	// temp1.w = the hue offset we'll use if it turns out that R is not the largest component (M==G or M==B)
	vec4 temp1 = rgb.b > rgb.g ? vec4(rgb.bg, hueOffsets.wz) : vec4(rgb.gb, hueOffsets.xy);

	// temp2.x = the largest component of RGB ("M" / "Max")
	// temp2.yw = the smaller components of RGB, ordered for the hue calculation (not necessarily sorted by magnitude!)
	// temp2.z = the hue offset we'll use in the hue calculation
	vec4 temp2 = rgb.r > temp1.x ? vec4(rgb.r, temp1.yzx) : vec4(temp1.xyw, rgb.r);

	// m = the smallest component of RGB ("min")
	float m = min(temp2.y, temp2.w);

	// Chroma = M - m
	float C = temp2.x - m;

	// Value = M
	float V = temp2.x;

	return vec3(
		abs(temp2.z + (temp2.w - temp2.y) / (6.0 * C + epsilon)), // Hue
		C / (temp2.x + epsilon), // Saturation
		V); // Value
}

vec3 convertHue2RGB(float hue)
{
	float r = abs(hue * 6.0 - 3.0) - 1.0;
	float g = 2.0 - abs(hue * 6.0 - 2.0);
	float b = 2.0 - abs(hue * 6.0 - 4.0);
	return clamp(vec3(r, g, b), 0.0, 1.0);
}

vec3 convertHSV2RGB(vec3 hsv)
{
	vec3 rgb = convertHue2RGB(hsv.x);
	float c = hsv.z * hsv.y;
	return rgb * c + hsv.z - c;
}

float rgbToGray(vec3 rgb) {
	const vec3 W = vec3(0.2125, 0.7154, 0.0721);
    return dot(rgb, W);
}

#endif // !defined(DRAW_MODE_silhouette) && (defined(ENABLE_color))

const vec2 kCenter = vec2(0.5, 0.5);

void main()
{
	#ifndef DRAW_MODE_line
	vec2 texcoord0 = v_texCoord;

	#ifdef ENABLE_mosaic
	texcoord0 = fract(u_mosaic * texcoord0);
	#endif // ENABLE_mosaic

	#ifdef ENABLE_pixelate
	{
		// TODO: clean up "pixel" edges
		vec2 pixelTexelSize = u_skinSize / u_pixelate;
		texcoord0 = (floor(texcoord0 * pixelTexelSize) + kCenter) / pixelTexelSize;
	}
	#endif // ENABLE_pixelate

	#ifdef ENABLE_whirl
	{
		const float kRadius = 0.5;
		vec2 offset = texcoord0 - kCenter;
		float offsetMagnitude = length(offset);
		float whirlFactor = max(1.0 - (offsetMagnitude / kRadius), 0.0);
		float whirlActual = u_whirl * whirlFactor * whirlFactor;
		float sinWhirl = sin(whirlActual);
		float cosWhirl = cos(whirlActual);
		mat2 rotationMatrix = mat2(
			cosWhirl, -sinWhirl,
			sinWhirl, cosWhirl
		);

		texcoord0 = rotationMatrix * offset + kCenter;
	}
	#endif // ENABLE_whirl

	#ifdef ENABLE_fisheye
	{
		vec2 vec = (texcoord0 - kCenter) / kCenter;
		float vecLength = length(vec);
		float r = pow(min(vecLength, 1.0), u_fisheye) * max(1.0, vecLength);
		vec2 unit = vec / vecLength;

		texcoord0 = kCenter + r * unit * kCenter;
	}
	#endif // ENABLE_fisheye

	gl_FragColor = texture2D(u_skin, texcoord0);

	#if defined(ENABLE_color) || defined(ENABLE_brightness)
	// Divide premultiplied alpha values for proper color processing
	// Add epsilon to avoid dividing by 0 for fully transparent pixels
	gl_FragColor.rgb = clamp(gl_FragColor.rgb / (gl_FragColor.a + epsilon), 0.0, 1.0);

	#ifdef ENABLE_color
	{
		// ORIGINAL COLOR EFFECT
		// vec3 hsv = convertRGB2HSV(gl_FragColor.xyz);

		// // this code forces grayscale values to be slightly saturated
		// // so that some slight change of hue will be visible
		// const float minLightness = 0.11 / 2.0;
		// const float minSaturation = 0.09;
		// if (hsv.z < minLightness) hsv = vec3(0.0, 1.0, minLightness);
		// else if (hsv.y < minSaturation) hsv = vec3(0.0, minSaturation, hsv.z);

		// hsv.x = mod(hsv.x + u_color, 1.0);
		// if (hsv.x < 0.0) hsv.x += 1.0;

		// gl_FragColor.rgb = convertHSV2RGB(hsv);
		
		// SEGMENTATION BY HSV
		/*
		float gray = rgbToGray(gl_FragColor.xyz);
		
		vec3 hsv = convertRGB2HSV(gl_FragColor.xyz);

		gl_FragColor.rgb = vec3(gray);    
		
		float hue = hsv.x;
		float saturation = hsv.y;
		float value = hsv.z;

		float steps = 8.;
		float steppedHue = floor(hue * steps) / steps;
		gl_FragColor.rgb = convertHue2RGB(steppedHue);

		if (saturation < 0.6) {
			gl_FragColor.rgb = vec3(gray);
		}
		if (value < 0.4) {
			gl_FragColor.rgb = vec3(gray);
		}
		*/

		// LAB COLOR SEGMENTATION
		float gray = rgbToGray(gl_FragColor.xyz);

		vec3 lab = xyz2lab(rgb2xyz(gl_FragColor.xyz));

		// vec3 selectedLab = xyz2lab(rgb2xyz(vec3(1.,0.,0.)));
		vec3 selectedLab = vec3(.5, u_color, 0.);

		if (distance(lab, selectedLab) < 0.5) {
    		gl_FragColor.rgb = vec3(1.,0.,0.);
		} else {
			gl_FragColor.rgb = vec3(gray);
		}

	}
	#endif // ENABLE_color

	#ifdef ENABLE_brightness
	gl_FragColor.rgb = clamp(gl_FragColor.rgb + vec3(u_brightness), vec3(0), vec3(1));
	#endif // ENABLE_brightness

	// Re-multiply color values
	gl_FragColor.rgb *= gl_FragColor.a + epsilon;

	#endif // defined(ENABLE_color) || defined(ENABLE_brightness)

	#ifdef ENABLE_ghost
	gl_FragColor *= u_ghost;
	#endif // ENABLE_ghost

	#ifdef DRAW_MODE_silhouette
	// Discard fully transparent pixels for stencil test
	if (gl_FragColor.a == 0.0) {
		discard;
	}
	// switch to u_silhouetteColor only AFTER the alpha test
	gl_FragColor = u_silhouetteColor;
	#else // DRAW_MODE_silhouette

	#ifdef DRAW_MODE_colorMask
	vec3 maskDistance = abs(gl_FragColor.rgb - u_colorMask);
	vec3 colorMaskTolerance = vec3(u_colorMaskTolerance, u_colorMaskTolerance, u_colorMaskTolerance);
	if (any(greaterThan(maskDistance, colorMaskTolerance)))
	{
		discard;
	}
	#endif // DRAW_MODE_colorMask
	#endif // DRAW_MODE_silhouette

	#ifdef DRAW_MODE_straightAlpha
	// Un-premultiply alpha.
	gl_FragColor.rgb /= gl_FragColor.a + epsilon;
	#endif

	#else // DRAW_MODE_line
	// Maaaaagic antialiased-line-with-round-caps shader.
	// Adapted from Inigo Quilez' 2D distance function cheat sheet
	// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm

	// On (some?) devices with 16-bit float precision, sufficiently long lines will overflow the float's range.
	// Avoid this by scaling these problematic values down to fit within (-1, 1) then scaling them back up later.
	// TODO: Is this a problem on all drivers with 16-bit mediump floats, or just Mali?
	vec2 pointDiff = abs(u_penPoints.zw - u_penPoints.xy);
	float FLOAT_SCALING_INVERSE = max(1.0, max(pointDiff.x, pointDiff.y));
	float FLOAT_SCALING = 1.0 / FLOAT_SCALING_INVERSE;

	// The xy component of u_penPoints is the first point; the zw is the second point.
	// This is done to minimize the number of gl.uniform calls, which can add up.
	// vec2 pa = v_texCoord - u_penPoints.xy, ba = u_penPoints.zw - u_penPoints.xy;
	vec2 pa = (v_texCoord - u_penPoints.xy) * FLOAT_SCALING, ba = (u_penPoints.zw - u_penPoints.xy) * FLOAT_SCALING;

	// Avoid division by zero
	float baDot = dot(ba, ba);
	// the dot product of a vector and itself is always positive
	baDot = max(baDot, epsilon);

	// Magnitude of vector projection of this fragment onto the line (both relative to the line's start point).
	// This results in a "linear gradient" which goes from 0.0 at the start point to 1.0 at the end point.
	float projMagnitude = clamp(dot(pa, ba) / baDot, 0.0, 1.0);

	float lineDistance = length(pa - (ba * projMagnitude)) * FLOAT_SCALING_INVERSE;

	// The distance to the line allows us to create lines of any thickness.
	// Instead of checking whether this fragment's distance < the line thickness,
	// utilize the distance field to get some antialiasing. Fragments far away from the line are 0,
	// fragments close to the line are 1, and fragments that are within a 1-pixel border of the line are in between.
	float cappedLine = clamp((u_lineThickness + 1.0) * 0.5 - lineDistance, 0.0, 1.0);

	gl_FragColor = u_lineColor * cappedLine;
	#endif // DRAW_MODE_line
}
