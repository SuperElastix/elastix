/* This file provides the implementation of the  vec< int32_t , 2>  type
 * Only for including in emm_vec.hxx.
 * Don't include anywhere else.
 *
 * Created by Dirk Poot, Erasmus MC,
 * Last modified 3-2-2015
 */


// load vector:
template<>  vec<int32_t, 2>::vec(const int32_t * v) {
      //xmm[0] = _mm_loadl_epi64( reinterpret_cast<const __m128i *>( v ) );
	xmm[0] = _mm_set_epi32( 0,  0, v[1], v[0]);
}
template<> template<typename int_t>  vec<int32_t, 2>::vec(const int32_t *v, int_t stride) {
      xmm[0] = _mm_set_epi32( 0, 0, v[stride], v[0]);
}
/*template<>  vec<int32_t, 2>::vec(const int32_t v) {
      xmm[0] = _mm_load_ss(&v );
    xmm[0] = _mm_shuffle_epi32(xmm[0],xmm[0],_MM_SHUFFLE(0, 0, 0, 0));
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
    //xmm[0] = _mm_unpacklo_ps(xmm[0],xmm[0]);
}*/

//create as zero vector:
template <> vec<int32_t, 2> vec<int32_t, 2>::zero () {
  return vec<int32_t,2>(_mm_setzero_si128());
}

// Store functions:
template <> void vec<int32_t, 2>::store(int32_t *v) {
#if defined(INCLUDE_SSE4) && !defined(BUGGY_MM_EXTRACT_EPI32)
  v[0] = _mm_extract_epi32( xmm[0] , 0);
  v[1] = _mm_extract_epi32( xmm[0] , 1); 
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
  *(v+1) = tmp[1];
#endif
}
template <> void vec<int32_t, 2>::storea(int32_t *v) {
  vec<int32_t, 2>::store(v);
/*#if defined(INCLUDE_SSE4) && !defined(BUGGY_MM_EXTRACT_EPI32)
  v[0] = _mm_extract_epi32( xmm[0] , 0);
  v[1] = _mm_extract_epi32( xmm[0] , 1);
#else
  int32_t tmp[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp[0] ), xmm[0] );
	*v = tmp[0];
  *(v+1) = tmp[1];
#endif*/
}
template<> template<typename int_t>  void vec<int32_t, 2>::store(int32_t *v, int_t stride) {
#if defined(INCLUDE_SSE4) && !defined(BUGGY_MM_EXTRACT_EPI32)
  v[0] = _mm_extract_epi32( xmm[0] , 0);
  v[stride] = _mm_extract_epi32( xmm[0] , 1); 
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
#endif
}

// Operators, Specialized versions (single precision, length 2):
template <> vec<int32_t, 2> vec<int32_t, 2>::operator* (const vec<int32_t,2> &v) const  {
#ifdef INCLUDE_SSE4
  return vec<int32_t,2>( _mm_mullo_epi32( xmm[0], v.xmm[0]) );
#else
  int32_t tmp1[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ), _mm_unpacklo_epi32( xmm[0] , v.xmm[0]));
  return vec<int32_t,2>( _mm_set_epi32( 0,  0, tmp1[2] * tmp1[3], tmp1[0] * tmp1[1]) );
#endif
}
template <> vec<int32_t, 2> vec<int32_t, 2>::operator+ (const vec<int32_t,2> &v) const  {
  return vec<int32_t,2>(_mm_add_epi32(xmm[0], v.xmm[0]));
}
template <> vec<int32_t, 2> vec<int32_t, 2>::operator- (const vec<int32_t,2> &v) const  {
  return vec<int32_t,2>(_mm_sub_epi32(xmm[0], v.xmm[0]));
}
/*template <> vec<int32_t, 2> vec<int32_t, 2>::operator/ (const vec<int32_t,2> &v) const  {
  return vec<int32_t,2>(_mm_div_ps(xmm[0], v.xmm[0]));
}*/
template <> inline void vec<int32_t, 2>::operator*= (const vec<int32_t, 2> &v) {
#ifdef INCLUDE_SSE4
  xmm[0] = _mm_mullo_epi32(xmm[0], v.xmm[0]);
#else
  int32_t tmp1[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ), _mm_unpacklo_epi32( xmm[0] , v.xmm[0]));
  xmm[0] = _mm_set_epi32( 0,  0, tmp1[2] * tmp1[3], tmp1[0] * tmp1[1]) ;
#endif
}
template <> inline void vec<int32_t, 2>::operator+= (const vec<int32_t, 2> &v) {
  xmm[0] = _mm_add_epi32(xmm[0], v.xmm[0]);
}
template <> inline void vec<int32_t, 2>::operator-= (const vec<int32_t, 2> &v) {
  xmm[0] = _mm_sub_epi32(xmm[0], v.xmm[0]);
}
/*template <> inline void vec<int32_t, 2>::operator/= (const vec<int32_t, 2> &v) {
  xmm[0] = _mm_div_ps(xmm[0], v.xmm[0]);
}*/

// Operators,  scalar versions (int32_t, length 2):
template <> vec<int32_t, 2> vec<int32_t, 2>::operator* (const int32_t v) const  {
#ifdef INCLUDE_SSE4
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  return vec<int32_t,2>(_mm_mullo_epi32(xmm[0], tmp));
#else
  int32_t tmp1[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ),  xmm[0] );
  return vec<int32_t,2>( _mm_set_epi32( 0,  0, tmp1[1] * v, tmp1[0] * v) );
#endif
}
template <> vec<int32_t, 2> vec<int32_t, 2>::operator+ (const int32_t &v) const {
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  return vec<int32_t,2>(_mm_add_epi32(xmm[0], tmp));
}
template <> vec<int32_t, 2> vec<int32_t, 2>::operator- (const int32_t &v) const {
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  return vec<int32_t,2>(_mm_sub_epi32(xmm[0], tmp));
}
/*template <> vec<int32_t, 2> vec<int32_t, 2>::operator/ (const int32_t &v) const {
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  return vec<int32_t,2>(_mm_div_ps(xmm[0], tmp));
}*/
template <> inline void vec<int32_t, 2>::operator*= (const int32_t &v) {
#ifdef INCLUDE_SSE4
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  xmm[0] = _mm_mullo_epi32(xmm[0], tmp);
#else
  int32_t tmp1[4];
  _mm_storeu_si128( reinterpret_cast<__m128i *>( &tmp1[0] ),  xmm[0] );
  xmm[0] = _mm_set_epi32( 0,  0, tmp1[1] * v, tmp1[0] * v) ;
#endif
}
template <> inline void vec<int32_t, 2>::operator+= (const int32_t &v) {
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  xmm[0] = _mm_add_epi32(xmm[0], tmp);
}
template <> inline void vec<int32_t, 2>::operator-= (const int32_t &v) {
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  xmm[0] = _mm_sub_epi32(xmm[0], tmp);
}
/*template <> inline void vec<int32_t, 2>::operator/= (const int32_t &v) {
  __m128i tmp = _mm_set_epi32( 0, 0, v, v);
  xmm[0] = _mm_div_ps(xmm[0], tmp);
}*/
template <> inline vec<int32_t, 2> vec<int32_t, 2>::operator= (const int32_t &v) {
	int32_t vv = v;
  xmm[0] = _mm_set_epi32( 0, 0, vv, vv);
  return vec<int32_t, 2>( xmm[0] );
}

/*// repeat element
template <> vec<int32_t, 2> vec<int32_t, 2>::rep (int idx) const  {
  __m128i tmp = ((idx & 2) ? _mm_unpackhi_ps(xmm[0], xmm[0]) : _mm_unpacklo_ps(xmm[0], xmm[0]));
  tmp = ( (idx &1 ) ? _mm_unpackhi_ps(tmp, tmp) : _mm_unpacklo_ps(tmp, tmp));
  return vec<int32_t,2>(tmp);
}*/


/*// other functions (min/max ..) (int32_t, length 2)
inline vec<int32_t, 2> max(const vec<int32_t, 2> &a, const vec<int32_t, 2> &b){
  return vec<int32_t,2>(_mm_max_ps(a.xmm[0], b.xmm[0]));
}
inline vec<int32_t, 2> min(const vec<int32_t, 2> &a, const vec<int32_t, 2> &b){
  return vec<int32_t,2>(_mm_min_ps(a.xmm[0], b.xmm[0]));
}
#ifdef INCLUDE_SSE3
inline int32_t accum(const vec<int32_t, 2> &v){
  __m128i tmp = _mm_hadd_ps(v.xmm[0], v.xmm[0]);
  tmp = _mm_hadd_ps(tmp,tmp);
  int32_t tmpd;
  _mm_store_ss(&tmpd,tmp);
  return tmpd;
}
#endif
inline vec<int32_t, 2> abs(const vec<int32_t, 2> &a){
    __m128i tmp = _mm_set1_epi32( (0x7FFFFFFF) );
  return vec<int32_t,2>(_mm_and_ps(a.xmm[0], _mm_castsi128_ps(tmp)));
}

#ifdef INCLUDE_SSE4
inline vec<int32_t, 2> round(const vec<int32_t, 2> &v){
  return vec<int32_t,2>(_mm_round_ps(v.xmm[0], _MM_FROUND_TO_NEAREST_INT ));
}
#endif*/
