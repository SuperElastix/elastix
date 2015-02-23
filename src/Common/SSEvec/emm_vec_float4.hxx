/* This file provides the implementation of the  vec< float , 4>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */


// load vector:
template<>  vec<float, 4>::vec(const float *v) {
      xmm[0] = _mm_loadu_ps(v );
}
template<> template<typename int_t>  vec<float, 4>::vec(const float *v, int_t stride) {
      xmm[0] = _mm_set_ps( v[0], v[stride], v[2*stride], v[3*stride]);
}
template<>  vec<float, 4>::vec(const float v) {
      xmm[0] = _mm_load_ss(&v );
    xmm[0] = _mm_shuffle_ps(xmm[0],xmm[0],_MM_SHUFFLE(0, 0, 0, 0));
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
}

//create as zero vector:
template <> vec<float, 4> vec<float, 4>::zero () {
  return vec<float,4>(_mm_setzero_ps());
}

// Store functions:
template <> void vec<float, 4>::store(float *v) {
  _mm_storeu_ps( (v    ), xmm[0]);
}
template <> void vec<float, 4>::storea(float *v) {
  _mm_store_ps( (v    ), xmm[0]);
}
template<> template<typename int_t>  void vec<float, 4>::store(float *v, int_t stride) {
  _mm_store_ss( (v           ), xmm[0]);
    xmm[0] = _mm_shuffle_ps(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1));
  _mm_store_ss( (v +   stride), xmm[0]);
    xmm[0] = _mm_shuffle_ps(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1));
  _mm_store_ss( (v + 2*stride), xmm[0]);
    xmm[0] = _mm_shuffle_ps(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1));
  _mm_store_ss( (v + 3*stride), xmm[0]);
    xmm[0] = _mm_shuffle_ps(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1)); // leave in original state
}

// Type conversion constructors; convert this file's type to different types :


// Operators, Specialized versions (single precision, length 4):
template <> vec<float, 4> vec<float, 4>::operator* (const vec<float,4> &v) const  {
  return vec<float,4>(_mm_mul_ps(xmm[0], v.xmm[0]));
}
template <> vec<float, 4> vec<float, 4>::operator+ (const vec<float,4> &v) const  {
  return vec<float,4>(_mm_add_ps(xmm[0], v.xmm[0]));
}
template <> vec<float, 4> vec<float, 4>::operator- (const vec<float,4> &v) const  {
  return vec<float,4>(_mm_sub_ps(xmm[0], v.xmm[0]));
}
template <> vec<float, 4> vec<float, 4>::operator/ (const vec<float,4> &v) const  {
  return vec<float,4>(_mm_div_ps(xmm[0], v.xmm[0]));
}
template <> inline void vec<float, 4>::operator*= (const vec<float, 4> &v) {
  xmm[0] = _mm_mul_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<float, 4>::operator+= (const vec<float, 4> &v) {
  xmm[0] = _mm_add_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<float, 4>::operator-= (const vec<float, 4> &v) {
  xmm[0] = _mm_sub_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<float, 4>::operator/= (const vec<float, 4> &v) {
  xmm[0] = _mm_div_ps(xmm[0], v.xmm[0]);
}

// Operators,  scalar versions (float, length 4):
template <> vec<float, 4> vec<float, 4>::operator* (const float v) const  {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,4>(_mm_mul_ps(xmm[0], tmp));
}
template <> vec<float, 4> vec<float, 4>::operator+ (const float &v) const {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,4>(_mm_add_ps(xmm[0], tmp));
}
template <> vec<float, 4> vec<float, 4>::operator- (const float &v) const {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,4>(_mm_sub_ps(xmm[0], tmp));
}
template <> vec<float, 4> vec<float, 4>::operator/ (const float &v) const {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  return vec<float,4>(_mm_div_ps(xmm[0], tmp));
}
template <> inline void vec<float, 4>::operator*= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_mul_ps(xmm[0], tmp);
}
template <> inline void vec<float, 4>::operator+= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_add_ps(xmm[0], tmp);
}
template <> inline void vec<float, 4>::operator-= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_sub_ps(xmm[0], tmp);
}
template <> inline void vec<float, 4>::operator/= (const float &v) {
  __m128 tmp = _mm_load_ss(&v);
  tmp = _mm_shuffle_ps(tmp,tmp,_MM_SHUFFLE(0, 0, 0, 0));
  xmm[0] = _mm_div_ps(xmm[0], tmp);
}

// repeat element
template <> vec<float, 4> vec<float, 4>::rep (int idx) const  {
  __m128 tmp = ((idx & 2) ? _mm_unpackhi_ps(xmm[0], xmm[0]) : _mm_unpacklo_ps(xmm[0], xmm[0]));
  tmp = ( (idx &1 ) ? _mm_unpackhi_ps(tmp, tmp) : _mm_unpacklo_ps(tmp, tmp));
  return vec<float,4>(tmp);
}


// other functions (min/max ..) (float, length 4)
inline vec<float, 4> max(const vec<float, 4> &a, const vec<float, 4> &b){
  return vec<float,4>(_mm_max_ps(a.xmm[0], b.xmm[0]));
}
inline vec<float, 4> min(const vec<float, 4> &a, const vec<float, 4> &b){
  return vec<float,4>(_mm_min_ps(a.xmm[0], b.xmm[0]));
}
#ifdef INCLUDE_SSE3
inline float accum(const vec<float, 4> &v){
  __m128 tmp = _mm_hadd_ps(v.xmm[0], v.xmm[0]);
  tmp = _mm_hadd_ps(tmp,tmp);
  float tmpd;
  _mm_store_ss(&tmpd,tmp);
  return tmpd;
}
#endif
inline vec<float, 4> abs(const vec<float, 4> &a){
    __m128i tmp = _mm_set1_epi32( (0x7FFFFFFF) );
  return vec<float,4>(_mm_and_ps(a.xmm[0], _mm_castsi128_ps(tmp)));
}

#ifdef INCLUDE_SSE4
inline vec<float, 4> round(const vec<float, 4> &v){
  return vec<float,4>(_mm_round_ps(v.xmm[0], _MM_FROUND_TO_NEAREST_INT ));
}
#endif
