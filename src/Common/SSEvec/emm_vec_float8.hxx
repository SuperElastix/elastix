/* This file provides the implementation of the  vec< float , 8>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */

// load vector:
template<>  vec<float, 8>::vec(const float *v) {
      xmm[0] = _mm_loadu_ps(v );
      xmm[1] = _mm_loadu_ps(v + 4);
}
template<>  vec<float, 8>::vec(const float v) {
      xmm[0] = _mm_load_ss(&v );
    xmm[0] = _mm_shuffle_ps(xmm[0],xmm[0],_MM_SHUFFLE(0, 0, 0, 0));
    xmm[1] = xmm[0];
}

//create as zero vector:
template <> vec<float, 8> vec<float, 8>::zero () {
  return vec<float,8>(_mm_setzero_ps(), _mm_setzero_ps());
}

// Store functions:
template <> void vec<float, 8>::store(float *v) {
  _mm_storeu_ps( (v    ), xmm[0]);
  _mm_storeu_ps( (v + 4), xmm[1]);
}

// Type conversion constructors; convert this file's type to different types :


// Operators, Specialized versions (single precision, length 8):
template <> vec<float, 8> vec<float, 8>::operator* (const vec<float,8> &v) const  {
  return vec<float,8>(_mm_mul_ps(xmm[0], v.xmm[0]), _mm_mul_ps(xmm[1], v.xmm[1]));
}
template <> vec<float, 8> vec<float, 8>::operator+ (const vec<float,8> &v) const  {
  return vec<float,8>(_mm_add_ps(xmm[0], v.xmm[0]), _mm_add_ps(xmm[1], v.xmm[1]));
}
template <> vec<float, 8> vec<float, 8>::operator- (const vec<float,8> &v) const  {
  return vec<float,8>(_mm_sub_ps(xmm[0], v.xmm[0]), _mm_sub_ps(xmm[1], v.xmm[1]));
}
template <> vec<float, 8> vec<float, 8>::operator/ (const vec<float,8> &v) const  {
  return vec<float,8>(_mm_div_ps(xmm[0], v.xmm[0]), _mm_div_ps(xmm[1], v.xmm[1]));
}
template <> inline void vec<float, 8>::operator*= (const vec<float, 8> &v) {
  xmm[0] = _mm_mul_ps(xmm[0], v.xmm[0]);
  xmm[1] = _mm_mul_ps(xmm[1], v.xmm[1]);
}
template <> inline void vec<float, 8>::operator+= (const vec<float, 8> &v) {
  xmm[0] = _mm_add_ps(xmm[0], v.xmm[0]);
  xmm[1] = _mm_add_ps(xmm[1], v.xmm[1]);
}
template <> inline void vec<float, 8>::operator-= (const vec<float, 8> &v) {
  xmm[0] = _mm_sub_ps(xmm[0], v.xmm[0]);
  xmm[1] = _mm_sub_ps(xmm[1], v.xmm[1]);
}
template <> inline void vec<float, 8>::operator/= (const vec<float, 8> &v) {
  xmm[0] = _mm_div_ps(xmm[0], v.xmm[0]);
  xmm[1] = _mm_div_ps(xmm[1], v.xmm[1]);
}

//  Operators, scalar versions of operators (float, length 8):
template <> vec<float, 8> vec<float, 8>::operator* (const float v) const  {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,8>(_mm_mul_ps(xmm[0], tmp), _mm_mul_ps(xmm[1], tmp));
}
template <> vec<float, 8> vec<float, 8>::operator+ (const float &v) const {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,8>(_mm_add_ps(xmm[0], tmp), _mm_add_ps(xmm[1], tmp));
}
template <> vec<float, 8> vec<float, 8>::operator- (const float &v) const {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,8>(_mm_sub_ps(xmm[0], tmp), _mm_sub_ps(xmm[1], tmp));
}
template <> vec<float, 8> vec<float, 8>::operator/ (const float &v) const {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,8>(_mm_div_ps(xmm[0], tmp), _mm_div_ps(xmm[1], tmp));
}
template <> inline void vec<float, 8>::operator*= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_mul_ps(xmm[0], tmp);
  xmm[1] = _mm_mul_ps(xmm[1], tmp);
}
template <> inline void vec<float, 8>::operator+= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_add_ps(xmm[0], tmp);
  xmm[1] = _mm_add_ps(xmm[1], tmp);
}
template <> inline void vec<float, 8>::operator-= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_sub_ps(xmm[0], tmp);
  xmm[1] = _mm_sub_ps(xmm[1], tmp);
}
template <> inline void vec<float, 8>::operator/= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_div_ps(xmm[0], tmp);
  xmm[1] = _mm_div_ps(xmm[1], tmp);
}


// other functions (min/max ..) (float, length 8)
inline vec<float, 8> max(const vec<float, 8> &a, const vec<float, 8> &b){
  return vec<float,8>(_mm_max_ps(a.xmm[0], b.xmm[0]), _mm_max_ps(a.xmm[1], b.xmm[1]));
}
inline vec<float, 8> min(const vec<float, 8> &a, const vec<float, 8> &b){
  return vec<float,8>(_mm_min_ps(a.xmm[0], b.xmm[0]), _mm_min_ps(a.xmm[1], b.xmm[1]));
}
#ifdef INCLUDE_SSE3
inline float accum(const vec<float, 8> &v){
  __m128 tmp = _mm_add_ps(v.xmm[0], v.xmm[1]);
  tmp = _mm_hadd_ps(tmp,tmp);
  tmp = _mm_hadd_ps(tmp,tmp);
  float tmpd;
  _mm_store_ss(&tmpd,tmp);
  return tmpd;
}
#endif
