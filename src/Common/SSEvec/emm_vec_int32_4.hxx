/* This file provides the implementation of the  vec< int32 , 4>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 19-4-2012
 */


// load vector:
template<>  vec<int32_t, 4>::vec(const int32_t * v) {
      xmm[0] = _mm_loadu_si128( reinterpret_cast<const __m128i *>( v ) );
}
template<> template<typename int_t>  vec<int32_t, 4>::vec(const int32_t *v, int_t stride) {
      xmm[0] = _mm_set_epi32( v[3*stride], v[2*stride], v[stride], v[0]);
}
template<>  vec<int32_t, 4>::vec(const int32_t v) {
    xmm[0] = _mm_set_epi32( v , v, v, v);
}
template<> vec<int32_t, 4>  vec<int32_t, 4>::loada( const int32_t * v ) {
  //return vec( v ); // can't do aligned more efficient. 
  return vec<int32_t,4>( _mm_load_si128( reinterpret_cast<const __m128i *>( v ) ) );
}

//create as zero vector:
template <> vec<int32_t, 4> vec<int32_t, 4>::zero () {
  return vec<int32_t,4>(_mm_setzero_si128());
}

// Store functions:
template <> void vec<int32_t, 4>::store(int32_t *v) {
  _mm_storeu_si128( reinterpret_cast<__m128i *>(v    ), xmm[0]);
}
template <> void vec<int32_t, 4>::storea(int32_t *v) {
  _mm_store_si128( reinterpret_cast<__m128i *>(v    ), xmm[0]);
}
template<> template<typename int_t>  void vec<int32_t, 4>::store(int32_t *v, int_t stride) {
#if defined(INCLUDE_SSE4) && !defined(BUGGY_MM_EXTRACT_EPI32)
  v[       0] = _mm_extract_epi32( xmm[0] , 0);
  v[  stride] = _mm_extract_epi32( xmm[0] , 1); 
  v[2*stride] = _mm_extract_epi32( xmm[0] , 2); 
  v[3*stride] = _mm_extract_epi32( xmm[0] , 3); 
#else
	// In Microsoft Visual C++ 2008 SP1 (_MSC_VER==1500 ) 
	// there seems to be a bug in optimizing _mm_extract_epi32; 
	// When using the commented lines below it sometimes uses the value extracted as pointer
	// and tries to store the pointer v, which obviously causes segfaults. 
	// Hence we take the detour of saving first the entire register to memory and subsequently copying 
	// the part we want to store. Should be optimized further
	int32_t tmp[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp[0] ), xmm[0] );
	*v = tmp[0];
  *(v+stride) = tmp[1];
  *(v+2*stride) = tmp[2];
  *(v+3*stride) = tmp[3];
#endif
}

// Type conversion constructors; convert this file's type to different types :


// Operators, Specialized versions (single precision, length 4):
template <> vec<int32_t, 4> vec<int32_t, 4>::operator* (const vec<int32_t,4> &v) const  {
#ifdef INCLUDE_SSE4
  return vec<int32_t,4>(_mm_mullo_epi32(xmm[0], v.xmm[0]));
#else
  int32_t tmp1[4];
  int32_t tmp2[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ), xmm[0] );
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp2[0] ), v.xmm[0] );
  return vec<int32_t,4>( _mm_set_epi32( tmp1[3] * tmp2[3],  tmp1[2] * tmp2[2], tmp1[1] * tmp2[1], tmp1[0] * tmp2[0]) );
#endif
}
template <> vec<int32_t, 4> vec<int32_t, 4>::operator+ (const vec<int32_t,4> &v) const  {
  return vec<int32_t,4>(_mm_add_epi32(xmm[0], v.xmm[0]));
}
template <> vec<int32_t, 4> vec<int32_t, 4>::operator- (const vec<int32_t,4> &v) const  {
  return vec<int32_t,4>(_mm_sub_epi32(xmm[0], v.xmm[0]));
}
/*template <> vec<int32_t, 4> vec<int32_t, 4>::operator/ (const vec<int32_t,4> &v) const  {
  return vec<int32_t,4>(_mm_div_ps(xmm[0], v.xmm[0]));
}*/
template <> inline void vec<int32_t, 4>::operator*= (const vec<int32_t, 4> &v) {
#ifdef INCLUDE_SSE4
  xmm[0] = _mm_mullo_epi32(xmm[0], v.xmm[0]);
#else
  int32_t tmp1[4];
  int32_t tmp2[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ), xmm[0] );
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp2[0] ), v.xmm[0] );
  xmm[0] =( _mm_set_epi32( tmp1[3] * tmp2[3],  tmp1[2] * tmp2[2], tmp1[1] * tmp2[1], tmp1[0] * tmp2[0]) );
#endif
}
template <> inline void vec<int32_t, 4>::operator+= (const vec<int32_t, 4> &v) {
  xmm[0] = _mm_add_epi32(xmm[0], v.xmm[0]);
}
template <> inline void vec<int32_t, 4>::operator-= (const vec<int32_t, 4> &v) {
  xmm[0] = _mm_sub_epi32(xmm[0], v.xmm[0]);
}
/*template <> inline void vec<int32_t, 4>::operator/= (const vec<int32_t, 4> &v) {
  xmm[0] = _mm_div_ps(xmm[0], v.xmm[0]);
}*/

// Operators,  scalar versions (int32_t, length 4):
template <> vec<int32_t, 4> vec<int32_t, 4>::operator* (const int32_t v) const  {
#ifdef INCLUDE_SSE4
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<int32_t,4>(_mm_mullo_epi32(xmm[0], tmp));
#else
  int32_t tmp1[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ),  xmm[0] );
  return vec<int32_t,4>( _mm_set_epi32( tmp1[3] * v,  tmp1[2] * v, tmp1[1] * v, tmp1[0] * v) );
#endif
}
template <> vec<int32_t, 4> vec<int32_t, 4>::operator+ (const int32_t &v) const {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<int32_t,4>(_mm_add_epi32(xmm[0], tmp));
}
template <> vec<int32_t, 4> vec<int32_t, 4>::operator- (const int32_t &v) const {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<int32_t,4>(_mm_sub_epi32(xmm[0], tmp));
}
/*template <> vec<int32_t, 4> vec<int32_t, 4>::operator/ (const int32_t &v) const {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  return vec<int32_t,4>(_mm_div_ps(xmm[0], tmp));
}*/
template <> inline void vec<int32_t, 4>::operator*= (const int32_t &v) {
#ifdef INCLUDE_SSE4
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_mullo_epi32(xmm[0], tmp);
#else
  int32_t tmp1[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ),  xmm[0] );
  xmm[0] = _mm_set_epi32( tmp1[3] * v,  tmp1[2] * v, tmp1[1] * v, tmp1[0] * v) ;
#endif
}
template <> inline void vec<int32_t, 4>::operator+= (const int32_t &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_add_epi32(xmm[0], tmp);
}
template <> inline void vec<int32_t, 4>::operator-= (const int32_t &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_sub_epi32(xmm[0], tmp);
}
/*template <> inline void vec<int32_t, 4>::operator/= (const int32_t &v) {
  __m128i tmp = _mm_set_epi32( v, v, v, v);
  xmm[0] = _mm_div_ps(xmm[0], tmp);
}*/
template <> inline vec<int32_t, 4> vec<int32_t, 4>::operator= (const int32_t &v) {
  xmm[0] = _mm_set_epi32( v, v, v, v);
  return vec<int32_t, 4>( xmm[0] );
}

/*// repeat element
template <> vec<int32_t, 4> vec<int32_t, 4>::rep (int idx) const  {
  __m128i tmp = ((idx & 2) ? _mm_unpackhi_ps(xmm[0], xmm[0]) : _mm_unpacklo_ps(xmm[0], xmm[0]));
  tmp = ( (idx &1 ) ? _mm_unpackhi_ps(tmp, tmp) : _mm_unpacklo_ps(tmp, tmp));
  return vec<int32_t,4>(tmp);
}*/


#ifdef INCLUDE_SSE4
// other functions (min/max ..) (int32, length 4)
inline vec<int32_t, 4> max(const vec<int32_t, 4> &a, const vec<int32_t, 4> &b){
  return vec<int32_t,4>(_mm_max_epi32(a.xmmi[0], b.xmmi[0]));
}
inline vec<int32_t, 4> min(const vec<int32_t, 4> &a, const vec<int32_t, 4> &b){
  return vec<int32_t,4>(_mm_min_epi32(a.xmmi[0], b.xmmi[0]));
}
#endif
