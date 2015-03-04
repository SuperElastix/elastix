/* This file provides the implementation of the  vec< ulong , 3>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 3-3-2015
 */


// load vector:
template<>  vec<ulong, 3>::vec(const ulong * v) {
      xmm[0] = _mm_set_epi32( 0,  v[2], v[1], v[0]);
}
template<> template<typename int_t>  vec<ulong, 3>::vec(const ulong *v, int_t stride) {
      xmm[0] = _mm_set_epi32( 0,  v[2*stride], v[stride], v[0]);
}
/*template<>  vec<ulong, 3>::vec(const ulong v) {
      xmm[0] = _mm_load_ss(&v );
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 0, 0, 0));
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
}*/

//create as zero vector:
template <> vec<ulong, 3> vec<ulong, 3>::zero () {
  return vec<ulong,3>(_mm_setzero_si128());
}

// Store functions:
template <> void vec<ulong, 3>::store(ulong *v) {
  v[0] = _mm_extract_epi32( xmm[0] , 0);
  v[1] = _mm_extract_epi32( xmm[0] , 1);
  v[2] = _mm_extract_epi32( xmm[0] , 2);
}
template <> void vec<ulong, 3>::storea(ulong *v) {
  v[0] = _mm_extract_epi32( xmm[0] , 0);
  v[1] = _mm_extract_epi32( xmm[0] , 1);
  v[2] = _mm_extract_epi32( xmm[0] , 2);
}
/*template<> template<typename int_t>  void vec<ulong, 3>::store(ulong *v, int_t stride) {
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
template<> template<> inline vec<ulong, 3>::vec(const vec<long, 3> &v) {
  xmm[0] = v.xmm[0];
}

// Operators, Specialized versions (single precision, length 3):
/*template <> vec<ulong, 3> vec<ulong, 3>::operator* (const vec<ulong,3> &v) const  {
  return vec<ulong,3>(_mm_mul_ps(xmm[0], v.xmm[0]));
}
template <> vec<ulong, 3> vec<ulong, 3>::operator+ (const vec<ulong,3> &v) const  {
  return vec<ulong,3>(_mm_add_epi32(xmm[0], v.xmm[0]));
}
template <> vec<ulong, 3> vec<ulong, 3>::operator- (const vec<ulong,3> &v) const  {
  return vec<ulong,3>(_mm_sub_ps(xmm[0], v.xmm[0]));
}
template <> vec<ulong, 3> vec<ulong, 3>::operator/ (const vec<ulong,3> &v) const  {
  return vec<ulong,3>(_mm_div_ps(xmm[0], v.xmm[0]));
}
template <> inline void vec<ulong, 3>::operator*= (const vec<ulong, 3> &v) {
  xmm[0] = _mm_mul_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<ulong, 3>::operator+= (const vec<ulong, 3> &v) {
  xmm[0] = _mm_add_epi32(xmm[0], v.xmm[0]);
}
template <> inline void vec<ulong, 3>::operator-= (const vec<ulong, 3> &v) {
  xmm[0] = _mm_sub_ps(xmm[0], v.xmm[0]);
}
template <> inline void vec<ulong, 3>::operator/= (const vec<ulong, 3> &v) {
  xmm[0] = _mm_div_ps(xmm[0], v.xmm[0]);
}

// Operators,  scalar versions (ulong, length 3):
template <> vec<ulong, 3> vec<ulong, 3>::operator* (const ulong v) const  {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  return vec<ulong,3>(_mm_mul_ps(xmm[0], tmp));
}*/
template <> vec<ulong, 3> vec<ulong, 3>::operator+ (const ulong &v) const {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  return vec<ulong,3>(_mm_add_epi32(xmm[0], tmp));
}
template <> vec<ulong, 3> vec<ulong, 3>::operator- (const ulong &v) const {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  return vec<ulong,3>(_mm_sub_epi32(xmm[0], tmp));
}
/*template <> vec<ulong, 3> vec<ulong, 3>::operator/ (const ulong &v) const {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  return vec<ulong,3>(_mm_div_ps(xmm[0], tmp));
}*/
template <> inline void vec<ulong, 3>::operator*= (const ulong &v) {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  xmm[0] = _mm_mullo_epi32(xmm[0], tmp);
}
template <> inline void vec<ulong, 3>::operator+= (const ulong &v) {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  xmm[0] = _mm_add_epi32(xmm[0], tmp);
}
template <> inline void vec<ulong, 3>::operator-= (const ulong &v) {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  xmm[0] = _mm_sub_epi32(xmm[0], tmp);
}
/*template <> inline void vec<ulong, 3>::operator/= (const ulong &v) {
  __m128i tmp = _mm_set_epi32( 0, v, v, v);
  xmm[0] = _mm_div_ps(xmm[0], tmp);
}*/

/*// repeat element
template <> vec<ulong, 3> vec<ulong, 3>::rep (int idx) const  {
  __m128i tmp = ((idx & 2) ? _mm_unpackhi_ps(xmm[0], xmm[0]) : _mm_unpacklo_ps(xmm[0], xmm[0]));
  tmp = ( (idx &1 ) ? _mm_unpackhi_ps(tmp, tmp) : _mm_unpacklo_ps(tmp, tmp));
  return vec<ulong,3>(tmp);
}*/


/*// other functions (min/max ..) (ulong, length 3)
inline vec<ulong, 3> max(const vec<ulong, 3> &a, const vec<ulong, 3> &b){
  return vec<ulong,3>(_mm_max_ps(a.xmm[0], b.xmm[0]));
}
inline vec<ulong, 3> min(const vec<ulong, 3> &a, const vec<ulong, 3> &b){
  return vec<ulong,3>(_mm_min_ps(a.xmm[0], b.xmm[0]));
}
#ifdef INCLUDE_SSE3
inline ulong accum(const vec<ulong, 3> &v){
  __m128i tmp = _mm_hadd_ps(v.xmm[0], v.xmm[0]);
  tmp = _mm_hadd_ps(tmp,tmp);
  ulong tmpd;
  _mm_store_ss(&tmpd,tmp);
  return tmpd;
}
#endif
inline vec<ulong, 3> abs(const vec<ulong, 3> &a){
    __m128i tmp = _mm_set1_epi32( (0x7FFFFFFF) );
  return vec<ulong,3>(_mm_and_ps(a.xmm[0], _mm_castsi128_ps(tmp)));
}

#ifdef INCLUDE_SSE4
inline vec<ulong, 3> round(const vec<ulong, 3> &v){
  return vec<ulong,3>(_mm_round_ps(v.xmm[0], _MM_FROUND_TO_NEAREST_INT ));
}
#endif*/
