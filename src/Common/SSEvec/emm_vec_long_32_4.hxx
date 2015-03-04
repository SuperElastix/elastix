/* This file provides the implementation of the  vec< long , 4>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */


// load vector:
template<>  vec<long, 4>::vec(const long * v) {
      xmm[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( v ) );
}
template<> template<typename int_t>  vec<long, 4>::vec(const long *v, int_t stride) {
      xmm[0] = _mm_set_epi32( v[0], v[stride], v[2*stride], v[3*stride]);
}
/*template<>  vec<long, 4>::vec(const long v) {
      xmm[0] = _mm_load_ss(&v );
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 0, 0, 0));
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
}*/

//create as zero vector:
template <> vec<long, 4> vec<long, 4>::zero () {
  return vec<long,4>(_mm_setzero_si128());
}

// Store functions:
template <> void vec<long, 4>::store(long *v) {
  _mm_storeu_si128( reinterpret_cast<__m128i *>(v    ), xmm[0]);
}
template <> void vec<long, 4>::storea(long *v) {
  _mm_store_si128( reinterpret_cast<__m128i *>(v    ), xmm[0]);
}
/*template<> template<typename int_t>  void vec<long, 4>::store(long *v, int_t stride) {
  _mm_store_ss( (v           ), xmm[0]);
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1));
  _mm_store_ss( (v +   stride), xmm[0]);
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1));
  _mm_store_ss( (v + 2*stride), xmm[0]);
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1));
  _mm_store_ss( (v + 3*stride), xmm[0]);
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 3, 2, 1)); // leave in original state
}*/

// Type conversion constructors; convert this file's type to different types :


// Operators, Specialized versions (single precision, length 4):
/*template <> vec<long, 4> vec<long, 4>::operator* (const vec<long,4> &v) const  {
  return vec<long,4>(_mm_mul_ps(xmm[0], v.xmm[0]));
}
template <> vec<long, 4> vec<long, 4>::operator+ (const vec<long,4> &v) const  {
  return vec<long,4>(_mm_add_epi32(xmm[0], v.xmm[0]));
}
template <> vec<long, 4> vec<long, 4>::operator- (const vec<long,4> &v) const  {
  return vec<long,4>(_mm_sub_ps(xmm[0], v.xmm[0]));
}
template <> vec<long, 4> vec<long, 4>::operator/ (const vec<long,4> &v) const  {
  return vec<long,4>(_mm_div_ps(xmm[0], v.xmm[0]));
}
template <> inline void vec<long, 4>::operator*= (const vec<long, 4> &v) {
  xmm[0] = _mm_mul_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<long, 4>::operator+= (const vec<long, 4> &v) {
  xmm[0] = _mm_add_epi32(xmm[0], v.xmm[0]);
}
template <> inline void vec<long, 4>::operator-= (const vec<long, 4> &v) {
  xmm[0] = _mm_sub_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<long, 4>::operator/= (const vec<long, 4> &v) {
  xmm[0] = _mm_div_ps(xmm[0], v.xmm[0]);
}

// Operators,  scalar versions (long, length 4):
template <> vec<long, 4> vec<long, 4>::operator* (const long v) const  {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<long,4>(_mm_mul_ps(xmm[0], tmp));
}*/
template <> vec<long, 4> vec<long, 4>::operator+ (const long &v) const {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<long,4>(_mm_add_epi32(xmm[0], tmp));
}
template <> vec<long, 4> vec<long, 4>::operator- (const long &v) const {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<long,4>(_mm_sub_epi32(xmm[0], tmp));
}
/*template <> vec<long, 4> vec<long, 4>::operator/ (const long &v) const {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<long,4>(_mm_div_ps(xmm[0], tmp));
}*/
template <> inline void vec<long, 4>::operator*= (const long &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_mullo_epi32(xmm[0], tmp);
}
template <> inline void vec<long, 4>::operator+= (const long &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_add_epi32(xmm[0], tmp);
}
template <> inline void vec<long, 4>::operator-= (const long &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_sub_epi32(xmm[0], tmp);
}
/*template <> inline void vec<long, 4>::operator/= (const long &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_div_ps(xmm[0], tmp);
}*/

/*// repeat element
template <> vec<long, 4> vec<long, 4>::rep (int idx) const  {
  __m128i tmp = ((idx & 2) ? _mm_unpackhi_ps(xmm[0], xmm[0]) : _mm_unpacklo_ps(xmm[0], xmm[0]));
  tmp = ( (idx &1 ) ? _mm_unpackhi_ps(tmp, tmp) : _mm_unpacklo_ps(tmp, tmp));
  return vec<long,4>(tmp);
}*/


/*// other functions (min/max ..) (long, length 4)
inline vec<long, 4> max(const vec<long, 4> &a, const vec<long, 4> &b){
  return vec<long,4>(_mm_max_ps(a.xmm[0], b.xmm[0]));
}
inline vec<long, 4> min(const vec<long, 4> &a, const vec<long, 4> &b){
  return vec<long,4>(_mm_min_ps(a.xmm[0], b.xmm[0]));
}
#ifdef INCLUDE_SSE3
inline long accum(const vec<long, 4> &v){
  __m128i tmp = _mm_hadd_ps(v.xmm[0], v.xmm[0]);
  tmp = _mm_hadd_ps(tmp,tmp);
  long tmpd;
  _mm_store_ss(&tmpd,tmp);
  return tmpd;
}
#endif
inline vec<long, 4> abs(const vec<long, 4> &a){
    __m128i tmp = _mm_set1_epi32( (0x7FFFFFFF) );
  return vec<long,4>(_mm_and_ps(a.xmm[0], _mm_castsi128_ps(tmp)));
}

#ifdef INCLUDE_SSE4
inline vec<long, 4> round(const vec<long, 4> &v){
  return vec<long,4>(_mm_round_ps(v.xmm[0], _MM_FROUND_TO_NEAREST_INT ));
}
#endif*/
